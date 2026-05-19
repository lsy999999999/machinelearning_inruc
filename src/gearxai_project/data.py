from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset


FAULT_CODES = [
    "HEA",
    "CTF",
    "MTF",
    "RCF",
    "SWF",
    "BWF",
    "CWF",
    "IRF",
    "ORF",
]

FAULT_TO_LABEL = {code: idx for idx, code in enumerate(FAULT_CODES)}
LABEL_TO_FAULT = {idx: code for code, idx in FAULT_TO_LABEL.items()}
FAULT_NAME_TO_LABEL = {
    "healthy": 0,
    "chipped tooth fault": 1,
    "missing tooth fault": 2,
    "root crack fault": 3,
    "surface wear fault": 4,
    "ball fault": 5,
    "combination fault": 6,
    "inner race fault": 7,
    "outer race fault": 8,
}


@dataclass
class GearXAIDataConfig:
    dataset_name: str = "edi45/gearxai-dds-seu"
    config_name: str = "windows_100"
    cache_dir: str | None = None
    train_split: str = "train"
    val_split: str = "validation"
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    normalize: bool = True
    num_workers: int = 4


def _first_existing(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    raise KeyError(f"None of the expected keys exists: {keys}. Row keys: {sorted(row.keys())}")


def _as_signal_tensor(row: dict[str, Any], normalize: bool) -> torch.Tensor:
    raw = _first_existing(row, ("signal", "x", "window", "vibration", "data"))
    x = np.asarray(raw, dtype=np.float32)

    if x.shape == (100, 8):
        x = x.T
    elif x.shape != (8, 100):
        flat = x.reshape(-1)
        if flat.size != 800:
            raise ValueError(f"Expected 800 values for one window, got shape {x.shape}")
        x = flat.reshape(8, 100)

    if normalize:
        mean = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True)
        x = (x - mean) / np.maximum(std, 1e-6)

    return torch.from_numpy(np.ascontiguousarray(x))


def _as_label(row: dict[str, Any]) -> int:
    value = _first_existing(row, ("fault_code", "fault_name", "fault", "fault_type", "label", "y", "target"))
    if isinstance(value, str):
        normalized = value.strip()
        if normalized in FAULT_TO_LABEL:
            return FAULT_TO_LABEL[normalized]
        if normalized.lower() in FAULT_NAME_TO_LABEL:
            return FAULT_NAME_TO_LABEL[normalized.lower()]
        leading_digits = re.match(r"^\d+", normalized)
        if leading_digits:
            return int(leading_digits.group(0))
        try:
            return int(normalized)
        except ValueError as exc:
            raise ValueError(f"Unknown fault label: {value}") from exc
    return int(value)


class GearXAIWindows(Dataset):
    def __init__(
        self,
        split: str,
        dataset_name: str = "edi45/gearxai-dds-seu",
        config_name: str = "windows_100",
        cache_dir: str | None = None,
        max_samples: int | None = None,
        normalize: bool = True,
        seed: int = 42,
    ) -> None:
        self.normalize = normalize
        self.ds = load_dataset(dataset_name, config_name, split=split, cache_dir=cache_dir)
        if max_samples is not None:
            max_samples = min(max_samples, len(self.ds))
            self.ds = self.ds.shuffle(seed=seed).select(range(max_samples))

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.ds[index]
        x = _as_signal_tensor(row, normalize=self.normalize)
        y = torch.tensor(_as_label(row), dtype=torch.long)
        return x, y


def build_loaders(
    data_cfg: dict[str, Any],
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    cfg = GearXAIDataConfig(**data_cfg)
    train_set = GearXAIWindows(
        split=cfg.train_split,
        dataset_name=cfg.dataset_name,
        config_name=cfg.config_name,
        cache_dir=cfg.cache_dir,
        max_samples=cfg.max_train_samples,
        normalize=cfg.normalize,
        seed=seed,
    )
    val_set = GearXAIWindows(
        split=cfg.val_split,
        dataset_name=cfg.dataset_name,
        config_name=cfg.config_name,
        cache_dir=cfg.cache_dir,
        max_samples=cfg.max_val_samples,
        normalize=cfg.normalize,
        seed=seed + 1,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return train_loader, val_loader
