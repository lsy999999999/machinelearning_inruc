from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class GearXAICNNConfig:
    in_channels: int = 8
    num_classes: int = 9
    base_channels: int = 96
    depth: int = 5
    dropout: float = 0.12
    energy_mix: float = 0.25
    use_spectral: bool = False
    spectral_bins: int = 50
    spectral_channels: int | None = None
    input_length: int = 100


class ChannelSE(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class ConvBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation, groups=channels),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation, groups=channels),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            ChannelSE(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class FixedDFTSpectrum(nn.Module):
    """Small ONNX-friendly magnitude spectrum for fixed-length vibration windows."""

    def __init__(self, input_length: int, bins: int) -> None:
        super().__init__()
        max_bins = input_length // 2
        bins = max(1, min(int(bins), max_bins))
        t = torch.arange(input_length, dtype=torch.float32)
        freqs = torch.arange(1, bins + 1, dtype=torch.float32)
        angles = 2.0 * math.pi * freqs[:, None] * t[None, :] / float(input_length)
        scale = math.sqrt(2.0 / float(input_length))
        self.register_buffer("cos_basis", torch.cos(angles) * scale)
        self.register_buffer("sin_basis", -torch.sin(angles) * scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        real = torch.matmul(x, self.cos_basis.transpose(0, 1))
        imag = torch.matmul(x, self.sin_basis.transpose(0, 1))
        magnitude = torch.sqrt(real * real + imag * imag + 1e-6)
        return torch.log1p(magnitude)


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
            nn.Conv1d(cfg.base_channels // 2, cfg.base_channels // 2, kernel_size=3, padding=2, dilation=2),
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
        self.relevance_decoder = nn.Sequential(
            nn.Conv1d(cfg.base_channels, cfg.base_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(cfg.base_channels // 2),
            nn.GELU(),
            nn.Conv1d(cfg.base_channels // 2, cfg.in_channels, kernel_size=1),
        )

        self.use_spectral = bool(cfg.use_spectral)
        spectral_channels = int(cfg.spectral_channels or max(cfg.base_channels // 2, 32))
        if self.use_spectral:
            self.spectrum = FixedDFTSpectrum(input_length=cfg.input_length, bins=cfg.spectral_bins)
            self.spectral_stem = nn.Sequential(
                nn.Conv1d(cfg.in_channels, spectral_channels, kernel_size=5, padding=2),
                nn.BatchNorm1d(spectral_channels),
                nn.GELU(),
            )
            self.spectral_encoder = nn.Sequential(
                ConvBlock(spectral_channels, dilation=1, dropout=cfg.dropout),
                ConvBlock(spectral_channels, dilation=2, dropout=cfg.dropout),
            )
            head_in = cfg.base_channels * 2 + spectral_channels * 2
        else:
            self.spectrum = None
            self.spectral_stem = None
            self.spectral_encoder = None
            head_in = cfg.base_channels * 2

        self.head = nn.Sequential(
            nn.Linear(head_in, cfg.base_channels),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.base_channels, cfg.num_classes),
        )

    def input_relevance(self, x: torch.Tensor) -> torch.Tensor:
        learned = torch.sigmoid(self.gate(x))
        centered = x - x.mean(dim=-1, keepdim=True)
        energy = centered.abs()
        energy = energy / energy.amax(dim=-1, keepdim=True).clamp_min(1e-6)
        mix = float(self.cfg.energy_mix)
        return (1.0 - mix) * learned + mix * energy

    def forward_train(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        input_relevance = self.input_relevance(x)
        gated_x = x * (0.5 + input_relevance)
        features = self.encoder(self.stem(gated_x))
        avg_pool = features.mean(dim=-1)
        max_pool = features.amax(dim=-1)
        pooled = [avg_pool, max_pool]
        if self.use_spectral:
            assert self.spectrum is not None
            assert self.spectral_stem is not None
            assert self.spectral_encoder is not None
            spectral = self.spectrum(gated_x)
            spectral_features = self.spectral_encoder(self.spectral_stem(spectral))
            pooled.extend([spectral_features.mean(dim=-1), spectral_features.amax(dim=-1)])
        logits = self.head(torch.cat(pooled, dim=1))
        decoded_relevance = torch.sigmoid(self.relevance_decoder(features))
        relevance = 0.7 * input_relevance + 0.3 * decoded_relevance
        return logits, relevance

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, relevance = self.forward_train(x)
        probabilities = F.softmax(logits, dim=1)
        return probabilities, relevance


def build_model(model_cfg: dict) -> GearXAICNNGate:
    return GearXAICNNGate(GearXAICNNConfig(**model_cfg))
