"""Validity-guided diagnostic and calibration helpers."""

from __future__ import annotations

import torch
import torch.nn as nn


class ValidityCalibrator(nn.Module):
    """Apply optional task evidence validity scaling."""

    SUPPORTED_MODES = {"none", "score_only", "feature_scale", "logit_scale"}

    def __init__(self, mode: str = "none"):
        super().__init__()
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported calibration mode {mode!r}. "
                f"Expected one of {sorted(self.SUPPORTED_MODES)}."
            )
        self.mode = mode

    def forward(self, x, validity: torch.Tensor):
        if self.mode in {"none", "score_only"}:
            return x
        if isinstance(x, list):
            return [self.forward(item, validity) for item in x]
        if isinstance(x, tuple):
            return tuple(self.forward(item, validity) for item in x)
        if not torch.is_tensor(x):
            raise TypeError("ValidityCalibrator can only scale tensors, lists, or tuples.")

        scale = self._broadcast_validity(validity, x)
        return x * scale

    @staticmethod
    def _broadcast_validity(validity: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(validity):
            raise TypeError("validity must be a tensor.")
        validity = validity.to(device=x.device, dtype=x.dtype)
        if validity.ndim > 1:
            validity = validity.reshape(validity.shape[0], -1).mean(dim=1)
        if validity.shape[0] != x.shape[0]:
            raise ValueError(
                "validity batch size must match x batch size: "
                f"{validity.shape[0]} vs {x.shape[0]}."
            )
        return validity.reshape(validity.shape[0], *([1] * (x.ndim - 1)))
