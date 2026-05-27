"""Build train-split-only task evidence memory for Copernicus-Bench."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import random
import sys

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from datasets.data_module import BenchmarkDataModule  # noqa: E402
from factory import create_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a task evidence memory from the training split only."
    )
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "src" / "configs" / "config.yaml"),
        help="Optional base Copernicus-Bench config yaml.",
    )
    parser.add_argument(
        "--model-config",
        default=str(REPO_ROOT / "src" / "configs" / "model" / "copernicusfm_cls.yaml"),
        help="Model config yaml. Classification is supported in this first version.",
    )
    parser.add_argument(
        "--dataset-config",
        required=True,
        help="Dataset config yaml used by Copernicus-Bench.",
    )
    parser.add_argument("--output", required=True, help="Path to save memory .pt file.")
    parser.add_argument(
        "--task-type",
        default="classification",
        choices=("classification", "multilabel", "segmentation", "regression", "change"),
        help="Task evidence structure to build. Classification is implemented first.",
    )
    parser.add_argument(
        "--feature-type",
        default="pooled",
        choices=("pooled", "feature_map_pool", "change_diff"),
        help="Feature policy to save. Only pooled classification features are active now.",
    )
    parser.add_argument(
        "--memory-size",
        type=int,
        default=2048,
        help="Maximum number of task evidence atoms to keep. Use <=0 to keep all.",
    )
    parser.add_argument(
        "--atoms-per-class",
        type=int,
        default=128,
        help="Maximum number of evidence atoms sampled for each class.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=-1,
        help="Limit train batches for dry runs. Use -1 for the full train split.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for feature extraction.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_path(path: str | os.PathLike) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def load_yaml(path: str | os.PathLike):
    cfg = OmegaConf.load(resolve_path(path))
    container = OmegaConf.to_container(cfg, resolve=False)
    if isinstance(container, dict):
        container.pop("defaults", None)
    return OmegaConf.create(container)


def load_bench_config(args: argparse.Namespace):
    base_cfg = load_yaml(args.config) if args.config else OmegaConf.create({})
    base_model = load_yaml(REPO_ROOT / "src" / "configs" / "model" / "base_model.yaml")
    model_cfg = OmegaConf.merge(base_model, load_yaml(args.model_config))
    base_dataset = load_yaml(
        REPO_ROOT / "src" / "configs" / "dataset" / "base_dataset.yaml"
    )
    dataset_cfg = OmegaConf.merge(base_dataset, load_yaml(args.dataset_config))

    cfg = OmegaConf.merge(
        base_cfg,
        OmegaConf.create({"model": model_cfg, "dataset": dataset_cfg}),
    )
    cfg.task = dataset_cfg.task or model_cfg.task
    cfg.model.use_evidence_memory = False
    cfg.model.use_transfer_interface = False
    cfg.model.memory_path = None
    cfg.model.task_evidence_memory_path = None
    cfg.batch_size = args.batch_size or cfg.get("batch_size", 64)
    cfg.num_workers = args.num_workers if args.num_workers is not None else cfg.get(
        "num_workers", 8
    )
    cfg.pin_mem = cfg.get("pin_mem", True)
    return cfg


def extract_classification_features(
    model, batch, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(batch) == 3:
        images, labels, metas = batch
    else:
        images, labels = batch
        metas = torch.full((images.shape[0], 4), float("nan"))

    images = images.to(device, non_blocking=True)
    metas = metas.to(device, non_blocking=True)
    outputs = model(images, metas)
    if not isinstance(outputs, (tuple, list)) or len(outputs) < 2:
        raise RuntimeError(
            "Expected CopernicusFM classification forward to return (logits, feats)."
        )
    features = outputs[1]
    if features.ndim != 2:
        raise RuntimeError(
            "Expected pooled classification features with shape [B, D], "
            f"got {tuple(features.shape)}."
        )
    return features.detach().cpu(), labels.detach().cpu()


def add_to_class_buckets(
    buckets: dict[int, list[torch.Tensor]],
    features: torch.Tensor,
    labels: torch.Tensor,
) -> int:
    count = 0
    if labels.ndim == 1:
        for feature, label in zip(features, labels):
            class_id = int(label.item())
            buckets.setdefault(class_id, []).append(feature)
            count += 1
        return count

    if labels.ndim == 2:
        for feature, multi_label in zip(features, labels):
            positive_classes = (multi_label > 0).nonzero(as_tuple=False).reshape(-1)
            for class_id in positive_classes:
                buckets.setdefault(int(class_id.item()), []).append(feature)
                count += 1
        return count

    raise ValueError(
        "Classification labels must have shape [B] or multilabel shape [B, C], "
        f"got {tuple(labels.shape)}."
    )


def finalize_class_balanced_atoms(
    buckets: dict[int, list[torch.Tensor]],
    atoms_per_class: int,
    memory_size: int,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    if not buckets:
        raise RuntimeError("No labelled train features were extracted.")
    if atoms_per_class <= 0:
        raise ValueError("atoms_per_class must be positive.")

    classes = sorted(buckets)
    per_class_cap = atoms_per_class
    if memory_size > 0:
        per_class_cap = min(per_class_cap, max(1, memory_size // max(len(classes), 1)))

    sampled_features = []
    sampled_labels = []
    class_counts = {}
    for class_id in classes:
        class_features = buckets[class_id]
        if len(class_features) > per_class_cap:
            class_features = rng.sample(class_features, per_class_cap)
        class_counts[class_id] = len(class_features)
        sampled_features.extend(class_features)
        sampled_labels.extend([class_id] * len(class_features))

    if memory_size > 0 and len(sampled_features) > memory_size:
        keep_indices = rng.sample(range(len(sampled_features)), memory_size)
        sampled_features = [sampled_features[i] for i in keep_indices]
        sampled_labels = [sampled_labels[i] for i in keep_indices]

    if not sampled_features:
        raise RuntimeError("Class-balanced sampling produced an empty memory.")

    features = torch.stack(sampled_features, dim=0).float()
    labels = torch.tensor(sampled_labels, dtype=torch.long)
    metadata = {
        "class_counts": class_counts,
        "available_classes": classes,
        "atoms_per_class": atoms_per_class,
    }
    return features, labels, metadata


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    os.environ.setdefault("MODEL_WEIGHTS_DIR", str(REPO_ROOT / "fm_weights"))

    cfg = load_bench_config(args)
    if cfg.model.model_type != "copernicusfm" or cfg.task != "classification":
        raise NotImplementedError(
            "This first memory builder supports CopernicusFM classification only. "
            "Segmentation, regression, and change detection hooks are left for extension."
        )
    if args.task_type not in {"classification", "multilabel"}:
        raise NotImplementedError(
            "Only classification task evidence atoms are implemented. "
            "TODO: add segmentation patch evidence, regression value-bin evidence, "
            "and change transition evidence."
        )
    if args.feature_type != "pooled":
        raise NotImplementedError(
            "Only pooled classification features are supported in this first version."
        )

    device = torch.device(args.device)
    data_module = BenchmarkDataModule(
        dataset_config=cfg.dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_mem,
    )
    data_module.setup("fit")
    train_loader = data_module.train_dataloader()

    model = create_model(cfg, cfg.model, cfg.dataset)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    class_buckets: dict[int, list[torch.Tensor]] = {}
    seen_samples = 0
    seen_atoms = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(train_loader):
            if args.max_batches >= 0 and batch_idx >= args.max_batches:
                break
            features, labels = extract_classification_features(model, batch, device)
            seen_samples += int(features.shape[0])
            seen_atoms += add_to_class_buckets(class_buckets, features, labels)

    features, labels, sampling_metadata = finalize_class_balanced_atoms(
        class_buckets,
        atoms_per_class=args.atoms_per_class,
        memory_size=args.memory_size,
        rng=rng,
    )
    features = F.normalize(features, p=2, dim=-1, eps=1e-12)

    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    task_type = (
        "multilabel"
        if args.task_type == "multilabel" or cfg.dataset.get("multilabel", False)
        else "classification"
    )
    memory = {
        "features": features,
        "labels": labels,
        "atom_types": ["class_evidence"] * int(features.shape[0]),
        "task_type": task_type,
        "dataset_name": cfg.dataset.dataset_name,
        "model_name": "copernicusfm",
        "feature_type": args.feature_type,
        "created_from": "train_split_only",
        "num_atoms": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "metadata": {
            "split": "train",
            "note": "Task evidence memory for universal-to-task transfer interface.",
            "model_size": cfg.model.model_size,
            "memory_size": int(args.memory_size),
            "num_atoms": int(features.shape[0]),
            "num_train_samples_seen": int(seen_samples),
            "num_train_label_atoms_seen": int(seen_atoms),
            "max_batches": int(args.max_batches),
            "normalization": "l2",
            "model_config": str(resolve_path(args.model_config)),
            "dataset_config": str(resolve_path(args.dataset_config)),
            **sampling_metadata,
        },
    }
    torch.save(memory, output_path)
    print(
        "Saved task evidence memory to "
        f"{output_path} with features shape {tuple(features.shape)}. "
        "The memory was built from train_split_only."
    )


if __name__ == "__main__":
    main()
