"""Representation calibration after task evidence routing."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class TransferCalibrator(nn.Module):
    """Calibrate universal EO features with routed task evidence."""

    SUPPORTED_MODES = {
        "none",
        "concat_mlp",
        "residual",
        "gated_residual",
        "feature_scale",
    }

    def __init__(
        self,
        feature_dim: int,
        mode: str = "residual",
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported transfer calibrator mode {mode!r}. "
                f"Expected one of {sorted(self.SUPPORTED_MODES)}."
            )
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive.")

        self.feature_dim = int(feature_dim)
        self.mode = mode
        hidden = int(hidden_dim) if hidden_dim is not None else max(feature_dim // 2, 64)
        drop = float(dropout)

        if mode == "concat_mlp":
            self.project = self._mlp(feature_dim * 2, hidden, feature_dim, drop)
        elif mode == "residual":
            self.delta = self._mlp(feature_dim * 2, hidden, feature_dim, drop)
        elif mode == "gated_residual":
            self.delta = self._mlp(feature_dim * 2, hidden, feature_dim, drop)
            self.gate = self._mlp(feature_dim * 3, hidden, feature_dim, drop)
        elif mode == "feature_scale":
            self.scale = self._mlp(feature_dim * 2, hidden, feature_dim, drop)

    def forward(
        self,
        features: torch.Tensor,
        routed_evidence: torch.Tensor,
        router_stats: Optional[dict] = None,
    ) -> torch.Tensor:
        if self.mode == "none":
            return features
        if features.shape != routed_evidence.shape:
            raise ValueError(
                "features and routed_evidence must have the same shape, got "
                f"{tuple(features.shape)} and {tuple(routed_evidence.shape)}."
            )

        pair = torch.cat([features, routed_evidence], dim=-1)
        if self.mode == "concat_mlp":
            return self.project(pair)
        if self.mode == "residual":
            return features + self.delta(pair)
        if self.mode == "feature_scale":
            scale = torch.sigmoid(self.scale(pair))
            self._store_stat(router_stats, "mean_gate_value", scale.mean())
            return features * (1.0 + scale)

        gate_input = torch.cat([features, routed_evidence, features - routed_evidence], dim=-1)
        gate = torch.sigmoid(self.gate(gate_input))
        delta = self.delta(pair)
        self._store_stat(router_stats, "mean_gate_value", gate.mean())
        return features + gate * delta

    @staticmethod
    def _mlp(input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    @staticmethod
    def _store_stat(router_stats: Optional[dict], key: str, value: torch.Tensor) -> None:
        if isinstance(router_stats, dict):
            router_stats[key] = value.detach()
