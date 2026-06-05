from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from gearxai_project.data import GearXAIWindows
from gearxai_project.utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache official LogicLSTM teacher probabilities for distillation.")
    parser.add_argument("--config", required=True, help="Student YAML config; its data sample selection is reused exactly.")
    parser.add_argument("--teacher", required=True, help="Official or chosen teacher ONNX model.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--ort-threads", type=int, default=8)
    return parser.parse_args()


def make_dataset(data_cfg: dict[str, Any], split: str, max_samples: int | None, seed: int) -> GearXAIWindows:
    return GearXAIWindows(
        split=split,
        dataset_name=data_cfg["dataset_name"],
        config_name=data_cfg["config_name"],
        cache_dir=data_cfg.get("cache_dir"),
        max_samples=max_samples,
        normalize=False,  # Official LogicLSTM ONNX consumes raw windows.
        augment=False,
        seed=seed,
    )


def infer_probs(session: ort.InferenceSession, dataset: GearXAIWindows, batch_size: int) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    input_name = session.get_inputs()[0].name
    chunks: list[np.ndarray] = []
    for windows, _ in tqdm(loader, desc="teacher inference", leave=False):
        outputs = session.run(None, {input_name: windows.numpy().astype(np.float32, copy=False)})
        chunks.append(np.asarray(outputs[0], dtype=np.float32))
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0, 9), dtype=np.float32)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    seed = int(cfg.get("training", {}).get("seed", 42))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = int(args.ort_threads)
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.teacher), sess_options=opts, providers=["CPUExecutionProvider"])

    train_set = make_dataset(data_cfg, data_cfg["train_split"], data_cfg.get("max_train_samples"), seed)
    val_set = make_dataset(data_cfg, data_cfg["val_split"], data_cfg.get("max_val_samples"), seed + 1)

    train_probs = infer_probs(session, train_set, args.batch_size)
    val_probs = infer_probs(session, val_set, args.batch_size)
    np.save(output_dir / "train_teacher_probs.npy", train_probs)
    np.save(output_dir / "val_teacher_probs.npy", val_probs)

    metadata = {
        "teacher": str(args.teacher),
        "config": str(args.config),
        "seed": seed,
        "train_examples": int(len(train_set)),
        "val_examples": int(len(val_set)),
        "train_shape": list(train_probs.shape),
        "val_shape": list(val_probs.shape),
        "normalization": False,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
