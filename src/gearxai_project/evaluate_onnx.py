from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from gearxai_project.data import FAULT_CODES, GearXAIWindows
from gearxai_project.utils import load_config, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an exported GearXAI ONNX model.")
    parser.add_argument("--model", required=True, help="Path to exported ONNX model.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--full-val", action="store_true")
    parser.add_argument("--no-normalize", action="store_true", help="Disable the project loader's per-window normalization.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output")
    return parser.parse_args()


def build_val_loader(cfg: dict[str, Any], args: argparse.Namespace) -> DataLoader:
    data_cfg = dict(cfg["data"])
    max_samples = None if args.full_val else args.max_val_samples
    if max_samples is None and not args.full_val:
        max_samples = data_cfg.get("max_val_samples")

    dataset = GearXAIWindows(
        split=args.split or data_cfg["val_split"],
        dataset_name=data_cfg["dataset_name"],
        config_name=data_cfg["config_name"],
        cache_dir=data_cfg.get("cache_dir"),
        max_samples=max_samples,
        normalize=bool(data_cfg.get("normalize", True)) and not args.no_normalize,
        seed=int(cfg["training"].get("seed", 42)) + 1,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )


def evaluate_onnx(model_path: Path, loader: DataLoader) -> dict[str, Any]:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]

    preds: list[int] = []
    labels: list[int] = []
    total_seen = 0
    total_correct = 0

    for x, y in tqdm(loader, desc="onnx-val", leave=False):
        probabilities, relevance = session.run(output_names, {input_name: x.numpy().astype(np.float32)})
        if probabilities.shape[1] != len(FAULT_CODES):
            raise RuntimeError(f"Unexpected probability shape: {probabilities.shape}")
        if relevance.shape != tuple(x.shape):
            raise RuntimeError(f"Unexpected relevance shape: {relevance.shape}, expected {tuple(x.shape)}")

        pred = probabilities.argmax(axis=1)
        y_np = y.numpy()
        total_seen += int(y_np.shape[0])
        total_correct += int((pred == y_np).sum())
        preds.extend(pred.tolist())
        labels.extend(y_np.tolist())

    label_ids = list(range(len(FAULT_CODES)))
    per_class_f1 = f1_score(labels, preds, average=None, labels=label_ids, zero_division=0)
    cm = confusion_matrix(labels, preds, labels=label_ids)
    return {
        "accuracy": total_correct / max(total_seen, 1),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "per_class_f1": {FAULT_CODES[i]: float(v) for i, v in enumerate(per_class_f1)},
        "confusion_matrix": cm.astype(int).tolist(),
    }


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model does not exist: {model_path}")

    cfg = load_config(args.config)
    loader = build_val_loader(cfg, args)
    metrics = evaluate_onnx(model_path, loader)
    print(metrics)
    if args.output:
        save_json(args.output, metrics)


if __name__ == "__main__":
    main()
