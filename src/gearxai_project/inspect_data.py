from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from collections import Counter
from typing import Any

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import load_dataset

from gearxai_project.data import GearXAIWindows, LABEL_TO_FAULT, _as_label, _as_signal_tensor
from gearxai_project.utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect GearXAI dataset schema and label distribution.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--count-labels", action="store_true")
    parser.add_argument("--no-streaming", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=60)
    return parser.parse_args()


def _print_sample(row: dict[str, Any], normalize: bool) -> None:
    x = _as_signal_tensor(row, normalize=normalize)
    y = _as_label(row)
    print(f"First sample tensor shape: {tuple(x.shape)}")
    print(f"First sample label: {y} ({LABEL_TO_FAULT.get(y, 'unknown')})")
    print(
        "First sample channel mean/std range: "
        f"mean=[{float(x.mean(dim=1).min()):.4f}, {float(x.mean(dim=1).max()):.4f}], "
        f"std=[{float(x.std(dim=1).min()):.4f}, {float(x.std(dim=1).max()):.4f}]"
    )


def _inspect_streaming(args: argparse.Namespace, cfg: dict[str, Any], split: str) -> None:
    data_cfg: dict[str, Any] = cfg["data"]
    stream = load_dataset(
        data_cfg["dataset_name"],
        data_cfg["config_name"],
        split=split,
        cache_dir=data_cfg.get("cache_dir"),
        streaming=True,
    )
    limit = args.max_samples or 2
    rows = []
    for row in stream:
        rows.append(row)
        if len(rows) >= limit:
            break
    if not rows:
        raise RuntimeError(f"No rows read from split {split}")

    print(f"Dataset: {data_cfg['dataset_name']} / {data_cfg['config_name']}")
    print(f"Split: {split} (streaming)")
    print(f"Rows inspected: {len(rows)}")
    print(f"Columns: {sorted(rows[0].keys())}")
    print(f"Features: {getattr(stream, 'features', None)}")
    _print_sample(rows[0], normalize=bool(data_cfg.get("normalize", True)))

    if args.count_labels:
        counts = Counter(_as_label(row) for row in rows)
        print(f"Label counts in first {len(rows)} streamed rows:")
        for label in sorted(counts):
            print(f"  {label} ({LABEL_TO_FAULT.get(label, 'unknown')}): {counts[label]}")


def _inspect(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    data_cfg: dict[str, Any] = cfg["data"]
    split = args.split or data_cfg.get("train_split", "train")
    if not args.no_streaming:
        _inspect_streaming(args, cfg, split)
        return

    dataset = GearXAIWindows(
        split=split,
        dataset_name=data_cfg["dataset_name"],
        config_name=data_cfg["config_name"],
        cache_dir=data_cfg.get("cache_dir"),
        max_samples=args.max_samples,
        normalize=bool(data_cfg.get("normalize", True)),
        seed=int(cfg["training"].get("seed", 42)),
    )

    raw = dataset.ds
    print(f"Dataset: {data_cfg['dataset_name']} / {data_cfg['config_name']}")
    print(f"Split: {split}")
    print(f"Rows loaded: {len(dataset)}")
    print(f"Columns: {raw.column_names}")
    print(f"Features: {raw.features}")
    _print_sample(raw[0], normalize=bool(data_cfg.get("normalize", True)))

    if args.count_labels:
        counts = Counter(_as_label(row) for row in raw)
        print("Label counts:")
        for label in sorted(counts):
            print(f"  {label} ({LABEL_TO_FAULT.get(label, 'unknown')}): {counts[label]}")


def _run_with_timeout(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    timeout_seconds = int(args.timeout_seconds)
    if timeout_seconds <= 0:
        _inspect(args, cfg)
        return

    process = mp.Process(target=_inspect, args=(args, cfg))
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join()
        raise TimeoutError(f"Dataset inspection timed out after {timeout_seconds} seconds.")
    if process.exitcode:
        raise SystemExit(process.exitcode)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    try:
        _run_with_timeout(args, cfg)
    except TimeoutError as exc:
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    main()
