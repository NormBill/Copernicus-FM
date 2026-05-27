"""Export per-sample task evidence validity diagnostics."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys

import torch
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from datasets.data_module import BenchmarkDataModule  # noqa: E402
from factory import create_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export per-sample task evidence validity diagnostics."
    )
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "src" / "configs" / "config.yaml"),
        help="Optional base Copernicus-Bench config yaml.",
    )
    parser.add_argument(
        "--model-config",
        default=str(REPO_ROOT / "src" / "configs" / "model" / "copernicusfm_cls.yaml"),
        help="Model config yaml.",
    )
    parser.add_argument(
        "--dataset-config",
        required=True,
        help="Dataset config yaml used by Copernicus-Bench.",
    )
    parser.add_argument("--memory-path", required=True, help="Task evidence memory .pt.")
    parser.add_argument("--checkpoint", required=True, help="Lightning checkpoint path.")
    parser.add_argument("--output", required=True, help="CSV file to write.")
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="test",
        help="Dataset split to export.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for evaluation.",
    )
    parser.add_argument("--validity-alpha", type=float, default=1.0)
    parser.add_argument("--validity-beta", type=float, default=0.01)
    parser.add_argument("--validity-bias", type=float, default=1.0)
    parser.add_argument("--validity-temperature", type=float, default=1.0)
    parser.add_argument(
        "--calibration-mode",
        default="score_only",
        choices=("score_only", "logit_scale", "feature_scale", "none"),
    )
    parser.add_argument(
        "--top-coverages",
        default="1.0,0.9,0.8,0.7,0.5",
        help="Comma-separated coverage values for printed selective accuracy.",
    )
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
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    cfg.pin_mem = cfg.get("pin_mem", True)
    cfg.model.use_evidence_memory = True
    cfg.model.memory_path = str(resolve_path(args.memory_path))
    cfg.model.calibration_mode = args.calibration_mode
    cfg.model.validity_alpha = args.validity_alpha
    cfg.model.validity_beta = args.validity_beta
    cfg.model.validity_bias = args.validity_bias
    cfg.model.validity_temperature = args.validity_temperature
    return cfg


def get_loader(data_module: BenchmarkDataModule, split: str):
    if split == "train":
        return data_module.train_dataloader()
    if split == "val":
        return data_module.val_dataloader()
    return data_module.test_dataloader()


def load_checkpoint(model, checkpoint_path: str, device: torch.device) -> None:
    checkpoint = torch.load(resolve_path(checkpoint_path), map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint with missing={msg.missing_keys}, unexpected={msg.unexpected_keys}")


def write_rows(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "index",
        "label",
        "pred",
        "confidence",
        "correct",
        "validity",
        "ot_cost",
        "transport_entropy",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_selective_accuracy(rows: list[dict], coverages: list[float]) -> None:
    if not rows:
        return
    sorted_rows = sorted(rows, key=lambda row: row["validity"], reverse=True)
    full_acc = sum(row["correct"] for row in rows) / len(rows)
    mean_validity = sum(row["validity"] for row in rows) / len(rows)
    print(f"Full accuracy: {full_acc:.6f}")
    print(f"Mean validity: {mean_validity:.6f}")
    for coverage in coverages:
        keep = max(1, int(round(len(sorted_rows) * coverage)))
        selected = sorted_rows[:keep]
        acc = sum(row["correct"] for row in selected) / keep
        print(f"Top {coverage:.2f} validity coverage: n={keep}, acc={acc:.6f}")


def main() -> None:
    args = parse_args()
    cfg = load_bench_config(args)
    if cfg.model.model_type != "copernicusfm" or cfg.task != "classification":
        raise NotImplementedError("Only CopernicusFM classification export is supported.")
    if cfg.dataset.multilabel:
        raise NotImplementedError(
            "This first export script supports single-label classification. "
            "Use EuroSAT or LCZ first; add multilabel export later for BigEarthNet/LC100."
        )

    os.environ.setdefault("MODEL_WEIGHTS_DIR", str(REPO_ROOT / "fm_weights"))
    device = torch.device(args.device)

    data_module = BenchmarkDataModule(
        dataset_config=cfg.dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_mem,
    )
    data_module.setup(args.split)
    loader = get_loader(data_module, args.split)

    model = create_model(cfg, cfg.model, cfg.dataset)
    load_checkpoint(model, args.checkpoint, device)
    model.to(device)
    model.eval()

    rows = []
    sample_index = 0
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                images, targets, metas = batch
            else:
                images, targets = batch
                metas = torch.full((images.shape[0], 4), float("nan"))

            images = images.to(device, non_blocking=True)
            metas = metas.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).long()
            outputs = model(images, metas)
            logits = outputs[0]
            evidence = outputs[-1]
            if not isinstance(evidence, dict):
                raise RuntimeError("Model did not return task evidence diagnostics.")

            probs = torch.softmax(logits, dim=1)
            confidence, preds = probs.max(dim=1)
            correct = preds.eq(targets)

            batch_size = images.shape[0]
            for i in range(batch_size):
                rows.append(
                    {
                        "index": sample_index,
                        "label": int(targets[i].item()),
                        "pred": int(preds[i].item()),
                        "confidence": float(confidence[i].item()),
                        "correct": int(correct[i].item()),
                        "validity": float(evidence["validity"][i].item()),
                        "ot_cost": float(evidence["ot_cost"][i].item()),
                        "transport_entropy": float(
                            evidence["transport_entropy"][i].item()
                        ),
                    }
                )
                sample_index += 1

    output_path = resolve_path(args.output)
    write_rows(rows, output_path)
    print(f"Saved {len(rows)} rows to {output_path}")
    coverages = [float(item) for item in args.top_coverages.split(",") if item]
    print_selective_accuracy(rows, coverages)


if __name__ == "__main__":
    main()
