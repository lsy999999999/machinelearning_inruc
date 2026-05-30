from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import numpy as np
from datasets import load_dataset
from tqdm.auto import tqdm

from gearxai_project.data import FAULT_CODES, LABEL_TO_FAULT, _as_label
from gearxai_project.utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit GearXAI data distribution and signal ranges.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--output")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-shuffle", action="store_true")
    return parser.parse_args()


def _signal_array(row: dict[str, Any]) -> np.ndarray:
    x = np.asarray(row["signal"], dtype=np.float32)
    if x.shape == (100, 8):
        x = x.T
    elif x.shape != (8, 100):
        flat = x.reshape(-1)
        if flat.size != 800:
            raise ValueError(f"Expected 800 signal values, got shape {x.shape}")
        x = flat.reshape(8, 100)
    return np.ascontiguousarray(x)


def _counter_payload(counter: Counter, top_k: int | None = None) -> dict[str, int]:
    items = counter.items()
    if top_k is not None:
        items = counter.most_common(top_k)
    return {str(key): int(value) for key, value in items}


def _nested_counter_payload(counter: dict[Any, Counter], top_k: int) -> dict[str, dict[str, int]]:
    return {
        str(key): _counter_payload(value, top_k=top_k)
        for key, value in sorted(counter.items(), key=lambda kv: str(kv[0]))
    }


def _metadata_value(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    return "unknown" if value is None else str(value)


def audit_split(
    cfg: dict[str, Any],
    split: str,
    max_samples: int | None,
    top_k: int,
    seed: int,
    shuffle: bool,
) -> dict[str, Any]:
    data_cfg = cfg["data"]
    ds = load_dataset(
        data_cfg["dataset_name"],
        data_cfg["config_name"],
        split=split,
        cache_dir=data_cfg.get("cache_dir"),
    )
    total_rows = len(ds)
    if max_samples is not None:
        if shuffle:
            ds = ds.shuffle(seed=seed)
        ds = ds.select(range(min(max_samples, total_rows)))

    label_counts: Counter[int] = Counter()
    fault_counts: Counter[str] = Counter()
    regime_counts: Counter[str] = Counter()
    condition_counts: Counter[str] = Counter()
    speed_counts: Counter[int] = Counter()
    load_counts: Counter[int] = Counter()
    label_by_regime: dict[str, Counter] = defaultdict(Counter)
    label_by_condition: dict[str, Counter] = defaultdict(Counter)

    channel_sum = np.zeros(8, dtype=np.float64)
    channel_sumsq = np.zeros(8, dtype=np.float64)
    channel_min = np.full(8, np.inf, dtype=np.float64)
    channel_max = np.full(8, -np.inf, dtype=np.float64)
    channel_count = 0
    zero_std_windows = 0
    nonfinite_values = 0
    window_std_min = np.full(8, np.inf, dtype=np.float64)
    window_std_max = np.full(8, -np.inf, dtype=np.float64)

    for row in tqdm(ds, desc=f"audit-{split}", leave=False):
        label = _as_label(row)
        label_name = LABEL_TO_FAULT.get(label, str(label))
        regime = _metadata_value(row, "regime")
        condition = _metadata_value(row, "condition_id")

        label_counts[label] += 1
        fault_counts[str(row.get("fault_code", label_name))] += 1
        regime_counts[regime] += 1
        condition_counts[condition] += 1
        speed_counts[_metadata_value(row, "speed_hz")] += 1
        load_counts[_metadata_value(row, "load_nm")] += 1
        label_by_regime[regime][label_name] += 1
        label_by_condition[condition][label_name] += 1

        x = _signal_array(row)
        finite = np.isfinite(x)
        nonfinite_values += int((~finite).sum())
        x = np.where(finite, x, 0.0)

        channel_sum += x.sum(axis=1)
        channel_sumsq += np.square(x, dtype=np.float64).sum(axis=1)
        channel_min = np.minimum(channel_min, x.min(axis=1))
        channel_max = np.maximum(channel_max, x.max(axis=1))
        channel_count += x.shape[1]

        per_window_std = x.std(axis=1)
        zero_std_windows += int((per_window_std < 1e-6).sum())
        window_std_min = np.minimum(window_std_min, per_window_std)
        window_std_max = np.maximum(window_std_max, per_window_std)

    mean = channel_sum / max(channel_count, 1)
    var = channel_sumsq / max(channel_count, 1) - mean * mean
    std = np.sqrt(np.maximum(var, 0.0))
    label_values = [label_counts.get(i, 0) for i in range(len(FAULT_CODES))]
    imbalance_ratio = max(label_values) / max(min(v for v in label_values if v > 0), 1)

    warnings = []
    if nonfinite_values:
        warnings.append(f"found {nonfinite_values} non-finite values")
    if zero_std_windows:
        warnings.append(f"found {zero_std_windows} channel windows with near-zero std")
    if imbalance_ratio > 1.05:
        warnings.append(f"label imbalance ratio is {imbalance_ratio:.3f}")

    return {
        "dataset": data_cfg["dataset_name"],
        "config": data_cfg["config_name"],
        "split": split,
        "rows_total": int(total_rows),
        "rows_audited": int(len(ds)),
        "sample_shuffle": bool(shuffle and max_samples is not None),
        "sample_seed": int(seed),
        "label_counts": {LABEL_TO_FAULT[i]: int(label_counts.get(i, 0)) for i in range(len(FAULT_CODES))},
        "fault_code_counts": _counter_payload(fault_counts),
        "regime_counts": _counter_payload(regime_counts),
        "speed_hz_counts": _counter_payload(speed_counts),
        "load_nm_counts": _counter_payload(load_counts),
        "condition_id_top_counts": _counter_payload(condition_counts, top_k=top_k),
        "label_by_regime": _nested_counter_payload(label_by_regime, top_k=top_k),
        "label_by_condition_top": _nested_counter_payload(label_by_condition, top_k=top_k),
        "channel_raw_mean": [float(v) for v in mean],
        "channel_raw_std": [float(v) for v in std],
        "channel_raw_min": [float(v) for v in channel_min],
        "channel_raw_max": [float(v) for v in channel_max],
        "window_channel_std_min": [float(v) for v in window_std_min],
        "window_channel_std_max": [float(v) for v in window_std_max],
        "nonfinite_values": int(nonfinite_values),
        "zero_std_channel_windows": int(zero_std_windows),
        "label_imbalance_ratio": float(imbalance_ratio),
        "warnings": warnings,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    split = args.split or cfg["data"].get("train_split", "train")
    report = audit_split(
        cfg,
        split=split,
        max_samples=args.max_samples,
        top_k=args.top_k,
        seed=args.seed,
        shuffle=not args.no_shuffle,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
