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
        "--feature-type",
        default="pooled",
        choices=("pooled", "feature_map_pool", "change_diff"),
        help="Feature policy to save. Only pooled classification features are active now.",
    )
    parser.add_argument(
        "--memory-size",
        type=int,
        default=1024,
        help="Maximum number of train features to keep. Use <=0 to keep all.",
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
    cfg.model.memory_path = None
    cfg.batch_size = args.batch_size or cfg.get("batch_size", 64)
    cfg.num_workers = args.num_workers if args.num_workers is not None else cfg.get(
        "num_workers", 8
    )
    cfg.pin_mem = cfg.get("pin_mem", True)
    return cfg


def update_reservoir(
    reservoir: list[torch.Tensor],
    features: torch.Tensor,
    memory_size: int,
    seen: int,
    rng: random.Random,
) -> int:
    features = features.detach().cpu()
    if memory_size <= 0:
        reservoir.append(features)
        return seen + features.shape[0]

    for feature in features:
        seen += 1
        if len(reservoir) < memory_size:
            reservoir.append(feature)
        else:
            index = rng.randint(0, seen - 1)
            if index < memory_size:
                reservoir[index] = feature
    return seen


def finalize_reservoir(reservoir: list[torch.Tensor], memory_size: int) -> torch.Tensor:
    if not reservoir:
        raise RuntimeError("No features were extracted from the train dataloader.")
    if memory_size <= 0:
        return torch.cat(reservoir, dim=0)
    return torch.stack(reservoir, dim=0)


def extract_classification_features(model, batch, device: torch.device) -> torch.Tensor:
    if len(batch) == 3:
        images, _, metas = batch
    else:
        images, _ = batch
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
    return features


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

    reservoir: list[torch.Tensor] = []
    seen = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(train_loader):
            if args.max_batches >= 0 and batch_idx >= args.max_batches:
                break
            features = extract_classification_features(model, batch, device)
            seen = update_reservoir(
                reservoir, features, args.memory_size, seen=seen, rng=rng
            )

    features = finalize_reservoir(reservoir, args.memory_size).float()
    features = F.normalize(features, p=2, dim=-1, eps=1e-12)

    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    memory = {
        "features": features,
        "feature_type": args.feature_type,
        "dataset_name": cfg.dataset.dataset_name,
        "task": cfg.task,
        "model_name": cfg.model.model_type,
        "created_from": "train_split_only",
        "feature_dim": int(features.shape[1]),
        "normalization": "l2",
        "metadata": {
            "model_size": cfg.model.model_size,
            "memory_size": int(features.shape[0]),
            "num_train_features_seen": int(seen),
            "max_batches": int(args.max_batches),
            "model_config": str(resolve_path(args.model_config)),
            "dataset_config": str(resolve_path(args.dataset_config)),
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
