"""Lightweight Sinkhorn matcher for task evidence validity."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class OTMatcher(nn.Module):
    """Estimate task evidence validity with entropic optimal transport."""

    def __init__(
        self,
        ot_epsilon: float = 0.05,
        ot_iters: int = 30,
        validity_alpha: float = 1.0,
        validity_beta: float = 0.1,
        validity_bias: float = 1.0,
        validity_temperature: float = 1.0,
        normalize_features: bool = True,
    ):
        super().__init__()
        if ot_epsilon <= 0:
            raise ValueError("ot_epsilon must be positive.")
        if ot_iters <= 0:
            raise ValueError("ot_iters must be positive.")
        if validity_temperature <= 0:
            raise ValueError("validity_temperature must be positive.")

        self.ot_epsilon = float(ot_epsilon)
        self.ot_iters = int(ot_iters)
        self.validity_alpha = float(validity_alpha)
        self.validity_beta = float(validity_beta)
        self.validity_bias = float(validity_bias)
        self.validity_temperature = float(validity_temperature)
        self.normalize_features = bool(normalize_features)

    def forward(
        self, target_features: torch.Tensor, memory_features: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if not torch.is_tensor(target_features) or not torch.is_tensor(memory_features):
            raise TypeError("target_features and memory_features must be tensors.")
        if target_features.ndim not in (2, 3):
            raise ValueError(
                "target_features must have shape [B, D] or [B, N, D], "
                f"got {tuple(target_features.shape)}."
            )
        if memory_features.ndim != 2:
            raise ValueError(
                "memory_features must have shape [M, D], "
                f"got {tuple(memory_features.shape)}."
            )

        if target_features.ndim == 2:
            target_features = target_features.unsqueeze(1)

        batch_size, num_target, feature_dim = target_features.shape
        num_memory, memory_dim = memory_features.shape
        if batch_size == 0 or num_target == 0 or num_memory == 0:
            raise ValueError("OTMatcher received an empty target or memory tensor.")
        if feature_dim != memory_dim:
            raise ValueError(
                "Feature dimension mismatch: target has "
                f"{feature_dim}, memory has {memory_dim}."
            )

        target = target_features.float()
        memory = memory_features.to(device=target.device, dtype=target.dtype)
        if self.normalize_features:
            target = F.normalize(target, p=2, dim=-1, eps=1e-12)
            memory = F.normalize(memory, p=2, dim=-1, eps=1e-12)

        cosine = torch.matmul(target, memory.t()).clamp(-1.0, 1.0)
        cost = torch.nan_to_num(1.0 - cosine, nan=2.0, posinf=2.0, neginf=0.0)
        cost = cost.clamp(0.0, 2.0)

        transport = self._sinkhorn(cost)
        ot_cost = (transport * cost).sum(dim=(1, 2))

        tiny = torch.finfo(transport.dtype).eps
        log_transport = transport.clamp_min(tiny).log()
        transport_entropy = -(transport * log_transport).sum(dim=(1, 2))

        normalized_ot_cost = (ot_cost / 2.0).clamp_min(0.0)
        max_entropy = math.log(max(num_target * num_memory, 2))
        normalized_transport_entropy = (
            transport_entropy / max_entropy
        ).clamp_min(0.0)

        logits = (
            self.validity_bias
            - self.validity_alpha * normalized_ot_cost
            - self.validity_beta * normalized_transport_entropy
        ) / self.validity_temperature
        validity = torch.sigmoid(logits)

        return {
            "ot_cost": torch.nan_to_num(ot_cost),
            "transport_entropy": torch.nan_to_num(transport_entropy),
            "validity": torch.nan_to_num(validity).clamp(0.0, 1.0),
        }

    def _sinkhorn(self, cost: torch.Tensor) -> torch.Tensor:
        batch_size, num_target, num_memory = cost.shape
        eps = max(self.ot_epsilon, 1e-6)
        tiny = max(torch.finfo(cost.dtype).eps, 1e-8)

        kernel = torch.exp(-cost / eps).clamp_min(tiny)
        a = torch.full(
            (batch_size, num_target),
            1.0 / num_target,
            device=cost.device,
            dtype=cost.dtype,
        )
        b = torch.full(
            (batch_size, num_memory),
            1.0 / num_memory,
            device=cost.device,
            dtype=cost.dtype,
        )
        u = torch.ones_like(a)
        v = torch.ones_like(b)

        for _ in range(self.ot_iters):
            kv = torch.bmm(kernel.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1)
            v = b / kv.clamp_min(tiny)
            ku = torch.bmm(kernel, v.unsqueeze(-1)).squeeze(-1)
            u = a / ku.clamp_min(tiny)
            u = torch.nan_to_num(u, nan=0.0, posinf=1.0 / num_target)
            v = torch.nan_to_num(v, nan=0.0, posinf=1.0 / num_memory)

        transport = u.unsqueeze(-1) * kernel * v.unsqueeze(1)
        transport = torch.nan_to_num(transport).clamp_min(0.0)
        total_mass = transport.sum(dim=(1, 2), keepdim=True).clamp_min(tiny)
        return transport / total_mass
