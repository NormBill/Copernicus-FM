"""OT evidence routing for the universal-to-task transfer interface.

EO foundation models learn universal embeddings, but downstream tasks still use
them through task-specific predictors trained with direct task losses. This
leaves the application of universal representations largely implicit. We
introduce a universal-to-task transfer interface that assigns encoder features
to task evidence structures before prediction.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class OTEvidenceRouter(nn.Module):
    """Route universal EO features to local task evidence atoms."""

    SUPPORTED_MODES = {
        "class_conditional",
        "nearest_prototype_ablation",
        "global_ot_ablation",
    }

    def __init__(
        self,
        mode: str = "class_conditional",
        topk: int = 64,
        ot_epsilon: float = 0.05,
        ot_iters: int = 30,
        normalize_features: bool = True,
        entropy_weight: float = 0.01,
        use_ot: bool = True,
    ):
        super().__init__()
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported evidence router mode {mode!r}. "
                f"Expected one of {sorted(self.SUPPORTED_MODES)}."
            )
        if topk <= 0:
            raise ValueError("topk must be positive.")
        if ot_epsilon <= 0:
            raise ValueError("ot_epsilon must be positive.")
        if ot_iters <= 0:
            raise ValueError("ot_iters must be positive.")

        self.mode = mode
        self.topk = int(topk)
        self.ot_epsilon = float(ot_epsilon)
        self.ot_iters = int(ot_iters)
        self.normalize_features = bool(normalize_features)
        self.entropy_weight = float(entropy_weight)
        self.use_ot = bool(use_ot)

    def forward(
        self,
        query_features: torch.Tensor,
        memory_features: torch.Tensor,
        memory_labels: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
    ) -> dict:
        if query_features.ndim not in (2, 3):
            raise ValueError(
                "query_features must have shape [B, D] or [B, N, D], "
                f"got {tuple(query_features.shape)}."
            )
        if memory_features.ndim != 2:
            raise ValueError(
                "memory_features must have shape [M, D], "
                f"got {tuple(memory_features.shape)}."
            )
        if memory_features.shape[0] == 0:
            raise ValueError("memory_features must contain at least one atom.")

        original_shape = query_features.shape
        token_mode = query_features.ndim == 3
        if token_mode:
            batch_size, num_tokens, feature_dim = query_features.shape
            query = query_features.reshape(batch_size * num_tokens, feature_dim)
            if logits is not None:
                logits = logits.repeat_interleave(num_tokens, dim=0)
        else:
            batch_size, feature_dim = query_features.shape
            num_tokens = None
            query = query_features

        if feature_dim != memory_features.shape[1]:
            raise ValueError(
                "Feature dimension mismatch: query has "
                f"{feature_dim}, memory has {memory_features.shape[1]}."
            )

        query = query.float()
        memory = memory_features.to(device=query.device, dtype=query.dtype)
        if self.normalize_features:
            query = F.normalize(query, p=2, dim=-1, eps=1e-12)
            memory = F.normalize(memory, p=2, dim=-1, eps=1e-12)

        if self.mode == "global_ot_ablation":
            output = self._global_ot_ablation(query, memory)
        elif self.mode == "nearest_prototype_ablation":
            output = self._nearest_prototype_ablation(
                query, memory, memory_labels=memory_labels, logits=logits
            )
        else:
            output = self._class_conditional(query, memory, memory_labels, logits)

        if token_mode:
            for key in ("routed_evidence",):
                output[key] = output[key].reshape(*original_shape)
            for key in ("d_pos", "d_neg", "evidence_margin", "assignment_entropy"):
                output[key] = output[key].reshape(batch_size, num_tokens).mean(dim=1)
            if output.get("assignment_weights") is not None:
                output["assignment_weights"] = output["assignment_weights"].reshape(
                    batch_size, num_tokens, -1
                )
        return output

    def _class_conditional(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        memory_labels: Optional[torch.Tensor],
        logits: Optional[torch.Tensor],
    ) -> dict:
        if memory_labels is None:
            raise ValueError("class_conditional routing requires memory_labels.")
        if logits is None:
            raise ValueError("class_conditional routing requires preliminary logits.")

        labels = memory_labels.to(device=query.device).long().reshape(-1)
        if labels.shape[0] != memory.shape[0]:
            raise ValueError("memory_labels length must match memory_features.")

        preds = logits.argmax(dim=1).to(device=query.device).long()
        outputs = []
        weights = []
        pos_indices = []
        neg_indices = []
        d_pos_values = []
        d_neg_values = []
        entropies = []

        for i, pred in enumerate(preds):
            q = query[i : i + 1]
            pos_mask = labels == pred
            neg_mask = labels != pred
            pos_atoms, pos_global = self._select_atoms(memory, pos_mask)
            neg_atoms, neg_global = self._select_atoms(memory, neg_mask)

            pos_top_atoms, pos_top_dist, pos_top_global = self._nearest_atoms(
                q, pos_atoms, pos_global, self.topk
            )
            neg_top_atoms, neg_top_dist, neg_top_global = self._nearest_atoms(
                q, neg_atoms, neg_global, self.topk
            )

            assignment = self._assignment_weights(pos_top_dist)
            routed = assignment.unsqueeze(0).matmul(pos_top_atoms).squeeze(0)
            entropy = self._entropy(assignment)

            outputs.append(routed)
            weights.append(self._pad_1d(assignment, self.topk))
            pos_indices.append(self._pad_1d(pos_top_global, self.topk, pad_value=-1))
            neg_indices.append(self._pad_1d(neg_top_global, self.topk, pad_value=-1))
            d_pos_values.append(pos_top_dist.mean())
            d_neg_values.append(neg_top_dist.mean())
            entropies.append(entropy)

        d_pos = torch.stack(d_pos_values)
        d_neg = torch.stack(d_neg_values)
        assignment_entropy = torch.stack(entropies)
        return {
            "routed_evidence": torch.stack(outputs, dim=0),
            "d_pos": d_pos,
            "d_neg": d_neg,
            "evidence_margin": d_neg - d_pos,
            "assignment_entropy": assignment_entropy,
            "assignment_weights": torch.stack(weights, dim=0),
            "selected_positive_indices": torch.stack(pos_indices, dim=0),
            "selected_negative_indices": torch.stack(neg_indices, dim=0),
            "predicted_labels": preds.detach(),
        }

    def _nearest_prototype_ablation(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        memory_labels: Optional[torch.Tensor],
        logits: Optional[torch.Tensor],
    ) -> dict:
        if memory_labels is None:
            raise ValueError("nearest_prototype_ablation requires memory_labels.")
        labels = memory_labels.to(device=query.device).long().reshape(-1)
        classes = labels.unique(sorted=True)
        prototypes = []
        prototype_labels = []
        for cls in classes:
            prototypes.append(memory[labels == cls].mean(dim=0))
            prototype_labels.append(cls)
        prototypes = F.normalize(torch.stack(prototypes, dim=0), p=2, dim=-1, eps=1e-12)
        prototype_labels = torch.stack(prototype_labels)

        if logits is not None:
            preds = logits.argmax(dim=1).to(device=query.device).long()
            index = torch.stack(
                [
                    (prototype_labels == pred).nonzero(as_tuple=False)[0, 0]
                    if (prototype_labels == pred).any()
                    else torch.argmin(self._cosine_distance(query[i : i + 1], prototypes)[0])
                    for i, pred in enumerate(preds)
                ]
            )
        else:
            index = torch.argmin(self._cosine_distance(query, prototypes), dim=1)
            preds = prototype_labels[index]

        routed = prototypes[index]
        pos_dist = (1.0 - (query * routed).sum(dim=1).clamp(-1.0, 1.0)).clamp(0.0, 2.0)
        all_dist = self._cosine_distance(query, prototypes)
        d_neg = all_dist.masked_fill(prototype_labels.unsqueeze(0) == preds.unsqueeze(1), float("inf")).amin(dim=1)
        d_neg = torch.where(torch.isfinite(d_neg), d_neg, pos_dist)
        zeros = torch.zeros_like(pos_dist)
        return {
            "routed_evidence": routed,
            "d_pos": pos_dist,
            "d_neg": d_neg,
            "evidence_margin": d_neg - pos_dist,
            "assignment_entropy": zeros,
            "assignment_weights": None,
            "predicted_labels": preds.detach(),
        }

    def _global_ot_ablation(self, query: torch.Tensor, memory: torch.Tensor) -> dict:
        distance = self._cosine_distance(query, memory)
        k = min(self.topk, memory.shape[0])
        top_dist, top_index = torch.topk(distance, k=k, dim=1, largest=False)
        routed = []
        weights = []
        entropies = []
        for i in range(query.shape[0]):
            assignment = self._assignment_weights(top_dist[i])
            routed.append(assignment.unsqueeze(0).matmul(memory[top_index[i]]).squeeze(0))
            weights.append(self._pad_1d(assignment, self.topk))
            entropies.append(self._entropy(assignment))
        d_pos = top_dist.mean(dim=1)
        d_neg = distance.mean(dim=1)
        return {
            "routed_evidence": torch.stack(routed, dim=0),
            "d_pos": d_pos,
            "d_neg": d_neg,
            "evidence_margin": d_neg - d_pos,
            "assignment_entropy": torch.stack(entropies),
            "assignment_weights": torch.stack(weights, dim=0),
            "selected_positive_indices": self._pad_2d_indices(top_index, self.topk),
        }

    def _select_atoms(
        self, memory: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        indices = mask.nonzero(as_tuple=False).reshape(-1)
        if indices.numel() == 0:
            indices = torch.arange(memory.shape[0], device=memory.device)
        return memory[indices], indices

    def _nearest_atoms(
        self,
        q: torch.Tensor,
        atoms: torch.Tensor,
        global_indices: torch.Tensor,
        topk: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distance = self._cosine_distance(q, atoms).reshape(-1)
        k = min(topk, atoms.shape[0])
        top_dist, top_index = torch.topk(distance, k=k, largest=False)
        return atoms[top_index], top_dist, global_indices[top_index]

    def _assignment_weights(self, distance: torch.Tensor) -> torch.Tensor:
        if distance.numel() == 1:
            return torch.ones_like(distance)
        # With one pooled query feature, fixed-marginal Sinkhorn would assign a
        # uniform target marginal over atoms. The entropic assignment below keeps
        # the intended local evidence routing behavior while leaving _sinkhorn
        # available for future token-to-atom routing.
        weights = torch.softmax(-distance / max(self.ot_epsilon, 1e-6), dim=0)
        return torch.nan_to_num(weights, nan=1.0 / distance.numel())

    def _sinkhorn(self, cost: torch.Tensor) -> torch.Tensor:
        batch_size, num_query, num_atoms = cost.shape
        eps = max(self.ot_epsilon, 1e-6)
        tiny = max(torch.finfo(cost.dtype).eps, 1e-8)
        kernel = torch.exp(-cost / eps).clamp_min(tiny)
        a = torch.full(
            (batch_size, num_query),
            1.0 / num_query,
            device=cost.device,
            dtype=cost.dtype,
        )
        b = torch.full(
            (batch_size, num_atoms),
            1.0 / num_atoms,
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
            u = torch.nan_to_num(u, nan=0.0, posinf=1.0 / num_query)
            v = torch.nan_to_num(v, nan=0.0, posinf=1.0 / num_atoms)
        transport = u.unsqueeze(-1) * kernel * v.unsqueeze(1)
        transport = torch.nan_to_num(transport).clamp_min(0.0)
        return transport / transport.sum(dim=(1, 2), keepdim=True).clamp_min(tiny)

    @staticmethod
    def _cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        cosine = torch.matmul(a, b.t()).clamp(-1.0, 1.0)
        return torch.nan_to_num(1.0 - cosine, nan=2.0, posinf=2.0, neginf=0.0).clamp(0.0, 2.0)

    @staticmethod
    def _entropy(weights: torch.Tensor) -> torch.Tensor:
        tiny = torch.finfo(weights.dtype).eps
        entropy = -(weights * weights.clamp_min(tiny).log()).sum()
        max_entropy = math.log(max(int(weights.numel()), 2))
        return entropy / max_entropy

    @staticmethod
    def _pad_1d(values: torch.Tensor, length: int, pad_value: float = 0.0) -> torch.Tensor:
        if values.numel() >= length:
            return values[:length]
        pad = values.new_full((length - values.numel(),), pad_value)
        return torch.cat([values, pad], dim=0)

    @staticmethod
    def _pad_2d_indices(values: torch.Tensor, length: int) -> torch.Tensor:
        if values.shape[1] >= length:
            return values[:, :length]
        pad = values.new_full((values.shape[0], length - values.shape[1]), -1)
        return torch.cat([values, pad], dim=1)
