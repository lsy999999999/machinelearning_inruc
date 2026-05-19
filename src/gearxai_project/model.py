from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class GearXAICNNConfig:
    in_channels: int = 8
    num_classes: int = 9
    base_channels: int = 64
    depth: int = 4
    dropout: float = 0.10


class ConvBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class GearXAICNNGate(nn.Module):
    """Small 1D CNN that returns class probabilities and an input-shaped relevance map."""

    def __init__(self, cfg: GearXAICNNConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or GearXAICNNConfig()
        self.cfg = cfg

        self.gate = nn.Sequential(
            nn.Conv1d(cfg.in_channels, cfg.base_channels // 2, kernel_size=7, padding=3),
            nn.BatchNorm1d(cfg.base_channels // 2),
            nn.GELU(),
            nn.Conv1d(cfg.base_channels // 2, cfg.in_channels, kernel_size=1),
        )

        self.stem = nn.Sequential(
            nn.Conv1d(cfg.in_channels, cfg.base_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(cfg.base_channels),
            nn.GELU(),
        )
        blocks = []
        for i in range(cfg.depth):
            dilation = 2 ** min(i, 3)
            blocks.append(ConvBlock(cfg.base_channels, dilation=dilation, dropout=cfg.dropout))
        self.encoder = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Linear(cfg.base_channels * 2, cfg.base_channels),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.base_channels, cfg.num_classes),
        )

    def relevance(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.gate(x))

    def forward_train(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        relevance = self.relevance(x)
        gated_x = x * (0.5 + relevance)
        features = self.encoder(self.stem(gated_x))
        avg_pool = features.mean(dim=-1)
        max_pool = features.amax(dim=-1)
        logits = self.head(torch.cat([avg_pool, max_pool], dim=1))
        return logits, relevance

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, relevance = self.forward_train(x)
        probabilities = F.softmax(logits, dim=1)
        return probabilities, relevance


def build_model(model_cfg: dict) -> GearXAICNNGate:
    return GearXAICNNGate(GearXAICNNConfig(**model_cfg))
