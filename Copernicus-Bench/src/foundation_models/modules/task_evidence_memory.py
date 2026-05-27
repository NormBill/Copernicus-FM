"""Task-structured evidence memory for universal-to-task transfer.

EO foundation models learn universal embeddings, but downstream tasks still use
them through task-specific predictors trained with direct task losses. This
leaves the application of universal representations largely implicit. We
introduce a universal-to-task transfer interface that assigns encoder features
to task evidence structures before prediction.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn.functional as F


class TaskEvidenceMemory(torch.nn.Module):
    """Load task evidence atoms built from frozen train-split features."""

    REQUIRED_KEYS = {
        "features",
        "task_type",
        "dataset_name",
        "model_name",
        "feature_type",
        "created_from",
        "num_atoms",
        "feature_dim",
        "metadata",
    }

    def __init__(
        self,
        memory_path: str,
        normalize: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if not memory_path:
            raise ValueError("TaskEvidenceMemory requires a non-empty memory_path.")

        memory_path = os.path.expanduser(memory_path)
        if not os.path.exists(memory_path):
            raise FileNotFoundError(
                f"Task evidence memory file does not exist: {memory_path}"
            )

        try:
            memory = torch.load(memory_path, map_location=device or "cpu")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load task evidence memory from {memory_path}: {exc}"
            ) from exc

        if not isinstance(memory, dict):
            raise ValueError("Task evidence memory must be a dictionary.")

        missing = sorted(self.REQUIRED_KEYS.difference(memory.keys()))
        if missing:
            raise ValueError(
                "Invalid task evidence memory: missing required keys "
                f"{missing}."
            )

        if memory.get("created_from") != "train_split_only":
            raise ValueError(
                "Invalid task evidence memory: expected "
                "'created_from' == 'train_split_only', got "
                f"{memory.get('created_from')!r}."
            )

        features = memory["features"]
        if not torch.is_tensor(features):
            raise TypeError("Invalid task evidence memory: 'features' must be a tensor.")
        if not torch.is_floating_point(features):
            raise TypeError(
                "Invalid task evidence memory: 'features' must be a floating tensor."
            )
        if features.ndim != 2:
            raise ValueError(
                "Invalid task evidence memory: 'features' must have shape [M, D], "
                f"got {tuple(features.shape)}."
            )
        if features.shape[0] == 0 or features.shape[1] == 0:
            raise ValueError("Invalid task evidence memory: 'features' is empty.")
        if not torch.isfinite(features).all():
            raise ValueError("Invalid task evidence memory: 'features' contains NaN/Inf.")

        num_atoms = int(memory["num_atoms"])
        feature_dim = int(memory["feature_dim"])
        if num_atoms != int(features.shape[0]):
            raise ValueError(
                "Invalid task evidence memory: num_atoms metadata "
                f"{num_atoms} does not match features.shape[0] {features.shape[0]}."
            )
        if feature_dim != int(features.shape[1]):
            raise ValueError(
                "Invalid task evidence memory: feature_dim metadata "
                f"{feature_dim} does not match features.shape[1] {features.shape[1]}."
            )

        labels = memory.get("labels")
        if memory.get("task_type") in {"classification", "multilabel"} and labels is None:
            raise ValueError(
                "Classification task evidence memory must include single-class labels."
            )
        if labels is not None:
            if not torch.is_tensor(labels):
                raise TypeError(
                    "Invalid task evidence memory: 'labels' must be a tensor when set."
                )
            if labels.shape[0] != features.shape[0]:
                raise ValueError(
                    "Invalid task evidence memory: labels length must match features, "
                    f"got {labels.shape[0]} and {features.shape[0]}."
                )
            labels = labels.detach().long()
            if device is not None:
                labels = labels.to(device)

        features = features.detach().float()
        if normalize:
            features = F.normalize(features, p=2, dim=-1, eps=1e-12)
        if device is not None:
            features = features.to(device)

        self.memory_path = memory_path
        self.normalize = bool(normalize)
        self.task_type = str(memory["task_type"])
        self.metadata = {key: value for key, value in memory.items() if key not in {"features", "labels"}}
        self.metadata["num_atoms"] = int(features.shape[0])
        self.metadata["feature_dim"] = int(features.shape[1])

        self.register_buffer("features", features, persistent=False)
        if labels is not None:
            self.register_buffer("labels", labels, persistent=False)
        else:
            self.labels = None

    def get_features(self) -> torch.Tensor:
        return self.features

    def get_labels(self) -> Optional[torch.Tensor]:
        return self.labels

    def get_metadata(self) -> dict:
        return dict(self.metadata)

    def get_task_type(self) -> str:
        return self.task_type
