from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class LogicLSTMStudentConfig:
    """Baseline-inspired LSTM classifier with an ONNX-light relevance output.

    Defaults deliberately match the deployed official LogicLSTM graph observed in
    the devkit: one LSTM layer, input size 8 and hidden size 128.
    """

    in_channels: int = 8
    num_classes: int = 9
    hidden_size: int = 128
    num_layers: int = 1
    dropout: float = 0.0


class LogicLSTMStudent(nn.Module):
    """A self-trained LSTM student inspired by the official LogicLSTM baseline.

    The relevance output mirrors the lightweight ONNX pattern in the official
    baseline: absolute raw input normalised by total sample magnitude. No
    learned explanation decoder is included, keeping simplicity high.
    """

    def __init__(self, cfg: LogicLSTMStudentConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or LogicLSTMStudentConfig()
        lstm_dropout = float(self.cfg.dropout) if self.cfg.num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=self.cfg.in_channels,
            hidden_size=self.cfg.hidden_size,
            num_layers=self.cfg.num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(self.cfg.hidden_size, self.cfg.num_classes)

    @staticmethod
    def input_relevance(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        magnitude = x.abs()
        denominator = magnitude.sum(dim=(1, 2), keepdim=True).clamp_min(eps)
        return magnitude / denominator

    def forward_train(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = x.transpose(1, 2)  # [B, 8, 100] -> [B, 100, 8]
        outputs, _ = self.lstm(sequence)
        logits = self.fc(outputs[:, -1, :])
        relevance = self.input_relevance(x)
        return logits, relevance

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, relevance = self.forward_train(x)
        return F.softmax(logits, dim=1), relevance


def build_lstm_student(model_cfg: dict) -> LogicLSTMStudent:
    return LogicLSTMStudent(LogicLSTMStudentConfig(**model_cfg))
