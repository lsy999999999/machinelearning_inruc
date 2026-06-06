from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.run_logic_timereise_search import (
    compute_stats as compute_global_stats,
    load_dataset,
    make_variant,
    normalize_importance,
    time_blocks,
)
from tools.timereise_branch_utils import load_base_model, score_and_package_manifest


GROUP_NAMES = (
    "rms_low",
    "rms_mid",
    "rms_high",
    "dom_low",
    "dom_mid",
    "dom_high",
    "crest_low",
    "crest_mid",
    "crest_high",
)


def robust_stats_path(output_dir: Path) -> Path:
    return output_dir / "timereise_robust_stats.npz"


def signal_features(x_np: np.ndarray) -> np.ndarray:
    abs_x = np.abs(x_np)
    rms = np.sqrt(np.mean(np.square(x_np), axis=(1, 2)) + 1e-12)
    crest = np.max(abs_x, axis=(1, 2)) / np.maximum(rms, 1e-6)
    spectrum = np.abs(np.fft.rfft(x_np, axis=2))
    spectrum[:, :, 0] = 0.0
    dominant = spectrum.mean(axis=1).argmax(axis=1).astype(np.float32)
    return np.stack([rms.astype(np.float32), dominant, crest.astype(np.float32)], axis=1)


def collect_thresholds(args: argparse.Namespace) -> np.ndarray:
    loader = load_dataset(args.config, args.split, args.max_samples, args.stats_batch_size)
    chunks = []
    for x, _y in tqdm(loader, desc="robust-thresholds", leave=True):
        chunks.append(signal_features(x.numpy().astype(np.float32)))
    features = np.concatenate(chunks, axis=0)
    return np.quantile(features, [1.0 / 3.0, 2.0 / 3.0], axis=0).astype(np.float32)


def group_memberships(features: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    memberships = np.zeros((features.shape[0], 3), dtype=np.int64)
    for metric_id in range(3):
        memberships[:, metric_id] = np.digitize(features[:, metric_id], thresholds[:, metric_id], right=False) + metric_id * 3
    return memberships


def compute_robust_stats(args: argparse.Namespace, output_dir: Path) -> None:
    path = robust_stats_path(output_dir)
    if path.exists():
        print(f"Robust stats already exist: {path}", flush=True)
        return

    thresholds = collect_thresholds(args)
    loader = load_dataset(args.config, args.split, args.max_samples, args.stats_batch_size)
    session = ort.InferenceSession(args.base_model, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    prob_output_name = None
    for output in session.get_outputs():
        shape = output.shape
        if len(shape) == 2 and shape[-1] == 9:
            prob_output_name = output.name
            break
    if prob_output_name is None:
        raise RuntimeError("Could not identify probability output.")
    prob_index = output_names.index(prob_output_name)

    blocks = time_blocks(args.num_bins)
    drop_sums = np.zeros((len(GROUP_NAMES), 9, 8, args.num_bins), dtype=np.float64)
    keep_sums = np.zeros((len(GROUP_NAMES), 9, 8, args.num_bins), dtype=np.float64)
    counts = np.zeros((len(GROUP_NAMES), 9), dtype=np.float64)
    processed = 0

    for x, _y in tqdm(loader, desc="robust-timereise-stats", leave=True):
        x_np = x.numpy().astype(np.float32)
        memberships = group_memberships(signal_features(x_np), thresholds)
        probabilities = session.run(output_names, {input_name: x_np})[prob_index]
        pred = probabilities.argmax(axis=1).astype(np.int64)
        base_conf = probabilities[np.arange(probabilities.shape[0]), pred]

        for group_id in range(len(GROUP_NAMES)):
            group_mask = np.any(memberships == group_id, axis=1)
            for class_id in range(9):
                counts[group_id, class_id] += float(np.logical_and(group_mask, pred == class_id).sum())

        for channel in range(8):
            for bin_id, (start, end) in enumerate(blocks):
                deleted = x_np.copy()
                deleted[:, channel, start:end] = 0.0
                deleted_prob = session.run(output_names, {input_name: deleted})[prob_index]
                deleted_conf = deleted_prob[np.arange(deleted_prob.shape[0]), pred]
                drop = np.maximum(base_conf - deleted_conf, 0.0)

                kept = np.zeros_like(x_np)
                kept[:, channel, start:end] = x_np[:, channel, start:end]
                kept_prob = session.run(output_names, {input_name: kept})[prob_index]
                keep_conf = kept_prob[np.arange(kept_prob.shape[0]), pred]

                for group_id in range(len(GROUP_NAMES)):
                    group_mask = np.any(memberships == group_id, axis=1)
                    for class_id in range(9):
                        mask = np.logical_and(group_mask, pred == class_id)
                        if np.any(mask):
                            drop_sums[group_id, class_id, channel, bin_id] += float(drop[mask].sum())
                            keep_sums[group_id, class_id, channel, bin_id] += float(keep_conf[mask].sum())
        processed += int(x_np.shape[0])

    np.savez_compressed(
        path,
        drop_sums=drop_sums,
        keep_sums=keep_sums,
        counts=counts,
        thresholds=thresholds,
        group_names=np.asarray(GROUP_NAMES),
        processed=np.asarray(processed, dtype=np.int64),
        num_bins=np.asarray(args.num_bins, dtype=np.int64),
        blocks=np.asarray(blocks, dtype=np.int64),
    )
    print(f"Saved robust stats: {path} processed={processed}", flush=True)


def aggregate_group_factors(group_factors: np.ndarray, aggregate: str) -> np.ndarray:
    if aggregate == "mean":
        return group_factors.mean(axis=0)
    if aggregate == "median":
        return np.median(group_factors, axis=0)
    if aggregate == "trimmed_mean":
        ordered = np.sort(group_factors, axis=0)
        return ordered[1:-1].mean(axis=0)
    raise ValueError(f"Unknown aggregate: {aggregate}")


def build_robust_specs(args: argparse.Namespace, output_dir: Path) -> list[tuple[str, np.ndarray]]:
    stats = np.load(robust_stats_path(output_dir))
    global_stats = np.load(args.global_stats)
    global_counts = np.maximum(global_stats["counts"].astype(np.float64), 1.0)
    global_drop = global_stats["drop_sums"].astype(np.float64) / global_counts[:, None, None]
    global_keep = global_stats["keep_sums"].astype(np.float64) / global_counts[:, None, None]

    counts = stats["counts"].astype(np.float64)
    safe_counts = np.maximum(counts, 1.0)
    drop = stats["drop_sums"].astype(np.float64) / safe_counts[:, :, None, None]
    keep = stats["keep_sums"].astype(np.float64) / safe_counts[:, :, None, None]
    missing = counts <= 0
    drop[missing] = global_drop[np.where(missing)[1]]
    keep[missing] = global_keep[np.where(missing)[1]]

    group_factors = []
    for group_id in range(drop.shape[0]):
        mix = normalize_importance(0.5 * normalize_importance(drop[group_id]) + 0.5 * normalize_importance(keep[group_id]))
        group_factors.append(mix)
    group_factors_np = np.stack(group_factors, axis=0)

    specs = []
    for aggregate in args.aggregate:
        binned = normalize_importance(aggregate_group_factors(group_factors_np, aggregate))
        expanded = np.zeros((9, 8, 100), dtype=np.float32)
        for bin_id, (start, end) in enumerate(stats["blocks"].tolist()):
            expanded[:, :, int(start) : int(end)] = binned[:, :, bin_id : bin_id + 1]
        expanded = expanded / np.maximum(expanded.mean(axis=(1, 2), keepdims=True), 1e-6)
        for beta in args.time_beta:
            weights = np.clip(1.0 + beta * (expanded - 1.0), 0.05, None)
            weights = weights / np.maximum(weights.mean(axis=(1, 2), keepdims=True), 1e-6)
            specs.append((f"robust_{aggregate}_mix50_tb{int(beta * 100):03d}", weights.astype(np.float32)))
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robust TimeREISE pseudo-condition aggregation search.")
    parser.add_argument("--base-model", default="external/gearxai_devkit/baselines/onnx/logic_lstm.onnx")
    parser.add_argument("--config", default="configs/spectral_lite_c.yaml")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=50000)
    parser.add_argument("--stats-batch-size", type=int, default=256)
    parser.add_argument("--num-bins", type=int, default=20)
    parser.add_argument("--output-dir", default="runs/logic_timereise_robust_search_50k_b20")
    parser.add_argument("--global-stats", default="runs/logic_timereise_search_50k_b20_refine/timereise_stats.npz")
    parser.add_argument("--data-dir", default="prepared_hf_val5k")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--time-beta", type=float, nargs="+", default=[0.34, 0.35, 0.36])
    parser.add_argument("--aggregate", nargs="+", default=["mean", "median", "trimmed_mean"])
    parser.add_argument("--copy-prefix", default="logic_timereise_robust")
    parser.add_argument("--include-hard", action="store_true")
    parser.add_argument("--no-package", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not Path(args.global_stats).exists():
        compute_global_stats(args, output_dir)
        args.global_stats = str(output_dir / "timereise_stats.npz")
    compute_robust_stats(args, output_dir)

    base_model = load_base_model(args.base_model)
    manifest = []
    for tag, weights in build_robust_specs(args, output_dir):
        for hard in ([False, True] if args.include_hard else [False]):
            suffix = f"{tag}_hard" if hard else tag
            model_path = output_dir / f"logic_timereise_{suffix}.onnx"
            if not model_path.exists():
                make_variant(base_model, output_dir, tag, weights, hard)
            manifest.append({"tag": suffix, "model": str(model_path), "branch": "robust"})
    print(f"Prepared {len(manifest)} robust TimeREISE variants", flush=True)
    score_and_package_manifest(manifest, output_dir, args, copy_prefix=args.copy_prefix)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
