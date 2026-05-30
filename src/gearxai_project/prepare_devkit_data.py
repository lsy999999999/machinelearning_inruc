from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from gearxai_project.data import GearXAIWindows
from gearxai_project.utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export cached Hugging Face windows to GearXAI devkit NPY format.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-split", default=None)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def write_stats(output_dir: Path) -> None:
    stats = {
        "format": "[N, 8, 100]",
        "source": "Hugging Face edi45/gearxai-dds-seu windows_100 cache",
        "standardized_channel_mean": [0.0] * 8,
    }
    with (output_dir / "stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, sort_keys=True)


def main() -> None:
    args = parse_args()
    cfg: dict[str, Any] = load_config(args.config)
    data_cfg = cfg["data"]
    split = args.split or data_cfg["val_split"]
    output_split = args.output_split or split
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = GearXAIWindows(
        split=split,
        dataset_name=data_cfg["dataset_name"],
        config_name=data_cfg["config_name"],
        cache_dir=data_cfg.get("cache_dir"),
        max_samples=args.max_samples,
        normalize=False,
        seed=int(cfg["training"].get("seed", 42)) + 1,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    windows = np.lib.format.open_memmap(
        output_dir / f"{output_split}_windows.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(dataset), 8, 100),
    )
    labels = np.lib.format.open_memmap(
        output_dir / f"{output_split}_labels.npy",
        mode="w+",
        dtype=np.int64,
        shape=(len(dataset),),
    )

    offset = 0
    for x, y in tqdm(loader, desc=f"export-{output_split}", leave=False):
        batch = int(x.shape[0])
        end = offset + batch
        windows[offset:end] = x.numpy().astype(np.float32, copy=False)
        labels[offset:end] = y.numpy().astype(np.int64, copy=False)
        offset = end

    windows.flush()
    labels.flush()

    with (output_dir / f"{output_split}_metadata.jsonl").open("w", encoding="utf-8") as handle:
        for index, label in enumerate(np.asarray(labels)):
            handle.write(json.dumps({"source_split": split, "index": index, "label": int(label)}) + "\n")
    write_stats(output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "split": output_split,
                "samples": int(len(dataset)),
                "windows": str(output_dir / f"{output_split}_windows.npy"),
                "labels": str(output_dir / f"{output_split}_labels.npy"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
