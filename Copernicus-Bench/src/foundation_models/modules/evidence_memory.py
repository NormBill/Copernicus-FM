"""Task evidence memory loading utilities."""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn.functional as F


class TaskEvidenceMemory(torch.nn.Module):
    """Load frozen train-split task evidence features from a saved memory file."""

    def __init__(
        self,
        memory_path: str,
        device: Optional[torch.device] = None,
        normalize: bool = True,
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
            raise ValueError(
                "Task evidence memory must be a dict with a 'features' tensor and metadata."
            )

        created_from = memory.get("created_from")
        if created_from != "train_split_only":
            raise ValueError(
                "Invalid task evidence memory: expected "
                "'created_from' == 'train_split_only', got "
                f"{created_from!r}."
            )

        if "features" not in memory:
            raise ValueError("Invalid task evidence memory: missing 'features' tensor.")

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
        if features.numel() == 0 or features.shape[0] == 0 or features.shape[1] == 0:
            raise ValueError("Invalid task evidence memory: 'features' is empty.")
        if not torch.isfinite(features).all():
            raise ValueError("Invalid task evidence memory: 'features' contains NaN/Inf.")

        feature_dim = memory.get("feature_dim")
        if feature_dim is not None and int(feature_dim) != int(features.shape[1]):
            raise ValueError(
                "Invalid task evidence memory: metadata feature_dim "
                f"{feature_dim} does not match features.shape[1] {features.shape[1]}."
            )

        features = features.detach().float()
        if normalize:
            features = F.normalize(features, p=2, dim=-1, eps=1e-12)
        if device is not None:
            features = features.to(device)

        self.memory_path = memory_path
        self.normalize = normalize
        self.metadata = {
            key: value for key, value in memory.items() if key != "features"
        }
        self.metadata["num_features"] = int(features.shape[0])
        self.metadata["feature_dim"] = int(features.shape[1])
        self.register_buffer("features", features, persistent=False)

    def get_features(self) -> torch.Tensor:
        return self.features

    def get_metadata(self) -> dict:
        return dict(self.metadata)
