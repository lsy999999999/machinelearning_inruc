from __future__ import annotations

import torch


def relevance_regularization(
    relevance: torch.Tensor,
    sparse_weight: float = 0.0,
    tv_weight: float = 0.0,
) -> torch.Tensor:
    penalty = relevance.new_tensor(0.0)

    if sparse_weight:
        penalty = penalty + sparse_weight * relevance.mean()

    if tv_weight:
        tv_time = (relevance[:, :, 1:] - relevance[:, :, :-1]).abs().mean()
        tv_channel = (relevance[:, 1:, :] - relevance[:, :-1, :]).abs().mean()
        penalty = penalty + tv_weight * (tv_time + 0.25 * tv_channel)

    return penalty
