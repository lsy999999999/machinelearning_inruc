from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import onnxruntime as ort
import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.run_logic_timereise_marginal_search import (
    iter_stat_batches,
    probability_output_index,
)
from tools.run_logic_timereise_offline_innovation_search import add_offline_contrast
from tools.run_logic_timereise_search import expand_bins, make_variant, normalize_importance, time_blocks
from tools.timereise_branch_utils import load_base_model, score_and_package_manifest


def sample_gains_path(output_dir: Path) -> Path:
    return output_dir / "timereise_marginal_sample_gains.npz"


def compute_sample_gains(args: argparse.Namespace, output_dir: Path) -> None:
    path = sample_gains_path(output_dir)
    if path.exists():
        print(f"Sample marginal gains already exist: {path}", flush=True)
        return

    batch_iter, source_description = iter_stat_batches(args)
    session = ort.InferenceSession(args.base_model, providers=["CPUExecutionProvider"])
    input_name, output_names, prob_index = probability_output_index(session)
    blocks = time_blocks(args.num_bins)
    baseline_channel = np.asarray(args.baseline_channel_mean, dtype=np.float32).reshape(1, 8, 1)

    pred_chunks: list[np.ndarray] = []
    drop_chunks: list[np.ndarray] = []
    insert_chunks: list[np.ndarray] = []
    base_conf_chunks: list[np.ndarray] = []
    baseline_conf_chunks: list[np.ndarray] = []
    processed = 0

    for x_np in tqdm(batch_iter, desc="marginal-sample-gains", leave=True):
        probabilities = session.run(output_names, {input_name: x_np})[prob_index]
        pred = probabilities.argmax(axis=1).astype(np.int64)
        rows = np.arange(probabilities.shape[0])
        base_conf = probabilities[rows, pred].astype(np.float32)

        baseline = np.broadcast_to(baseline_channel, x_np.shape).astype(np.float32).copy()
        baseline_prob = session.run(output_names, {input_name: baseline})[prob_index]
        baseline_conf = baseline_prob[rows, pred].astype(np.float32)

        batch_drop = np.zeros((x_np.shape[0], 8, args.num_bins), dtype=np.float32)
        batch_insert = np.zeros_like(batch_drop)

        for channel in range(8):
            for bin_id, (start, end) in enumerate(blocks):
                deleted = x_np.copy()
                deleted[:, channel, start:end] = baseline_channel[:, channel, :]
                deleted_prob = session.run(output_names, {input_name: deleted})[prob_index]
                deleted_conf = deleted_prob[rows, pred]
                batch_drop[:, channel, bin_id] = base_conf - deleted_conf

                inserted = baseline.copy()
                inserted[:, channel, start:end] = x_np[:, channel, start:end]
                inserted_prob = session.run(output_names, {input_name: inserted})[prob_index]
                inserted_conf = inserted_prob[rows, pred]
                batch_insert[:, channel, bin_id] = inserted_conf - baseline_conf

        pred_chunks.append(pred)
        drop_chunks.append(batch_drop)
        insert_chunks.append(batch_insert)
        base_conf_chunks.append(base_conf)
        baseline_conf_chunks.append(baseline_conf)
        processed += int(x_np.shape[0])

    np.savez_compressed(
        path,
        pred=np.concatenate(pred_chunks, axis=0),
        drop_gains=np.concatenate(drop_chunks, axis=0),
        insert_gains=np.concatenate(insert_chunks, axis=0),
        base_conf=np.concatenate(base_conf_chunks, axis=0),
        baseline_conf=np.concatenate(baseline_conf_chunks, axis=0),
        blocks=np.asarray(blocks, dtype=np.int64),
        processed=np.asarray(processed, dtype=np.int64),
        num_bins=np.asarray(args.num_bins, dtype=np.int64),
        baseline_channel_mean=np.asarray(args.baseline_channel_mean, dtype=np.float32),
        source_description=np.asarray(source_description),
    )
    print(f"Saved sample marginal gains: {path} processed={processed} source={source_description}", flush=True)


def classwise_aggregate(values: np.ndarray, pred: np.ndarray, *, mode: str, value: float) -> np.ndarray:
    out = np.zeros((9, values.shape[1], values.shape[2]), dtype=np.float64)
    for class_id in range(9):
        class_values = values[pred == class_id]
        if class_values.shape[0] == 0:
            continue
        if mode == "quantile":
            out[class_id] = np.quantile(class_values, value, axis=0)
        elif mode == "trimmed":
            trim = int(np.floor(class_values.shape[0] * value))
            if trim <= 0 or trim * 2 >= class_values.shape[0]:
                out[class_id] = class_values.mean(axis=0)
            else:
                ordered = np.sort(class_values, axis=0)
                out[class_id] = ordered[trim:-trim].mean(axis=0)
        elif mode == "mean":
            out[class_id] = class_values.mean(axis=0)
        else:
            raise ValueError(f"Unknown aggregate mode: {mode}")
    return out


def classwise_positive(values: np.ndarray) -> np.ndarray:
    return normalize_importance(np.maximum(values, 0.0))


def combine_gain(
    *,
    drop_pos: np.ndarray,
    insert_pos: np.ndarray,
    drop_neg: np.ndarray,
    insert_neg: np.ndarray,
    drop_weight: float,
    penalty: float,
) -> np.ndarray:
    score = drop_weight * classwise_positive(drop_pos) + (1.0 - drop_weight) * classwise_positive(insert_pos)
    if penalty > 0:
        penalty_score = classwise_positive(drop_neg + insert_neg)
        score = np.maximum(score - penalty * penalty_score, 0.0)
    return classwise_positive(score)


def apply_power(values: np.ndarray, power: float) -> np.ndarray:
    if abs(power - 1.0) < 1e-9:
        return values.astype(np.float32)
    powered = np.power(np.maximum(values, 1e-6), power)
    powered = powered / np.maximum(powered.mean(axis=(1, 2), keepdims=True), 1e-6)
    return np.clip(powered, 0.05, 10.0).astype(np.float32)


def expand_to_weights(binned: np.ndarray, blocks: np.ndarray, beta: float, power: float) -> np.ndarray:
    expanded = expand_bins(apply_power(binned, power), blocks)
    expanded = expanded / np.maximum(expanded.mean(axis=(1, 2), keepdims=True), 1e-6)
    weights = np.clip(1.0 + beta * (expanded - 1.0), 0.03, None)
    weights = weights / np.maximum(weights.mean(axis=(1, 2), keepdims=True), 1e-6)
    return weights.astype(np.float32)


def aggregate_components(
    drop_gains: np.ndarray,
    insert_gains: np.ndarray,
    pred: np.ndarray,
    *,
    mode: str,
    value: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    drop_pos = classwise_aggregate(np.maximum(drop_gains, 0.0), pred, mode=mode, value=value)
    insert_pos = classwise_aggregate(np.maximum(insert_gains, 0.0), pred, mode=mode, value=value)
    drop_neg = classwise_aggregate(np.maximum(-drop_gains, 0.0), pred, mode=mode, value=value)
    insert_neg = classwise_aggregate(np.maximum(-insert_gains, 0.0), pred, mode=mode, value=value)
    return drop_pos, insert_pos, drop_neg, insert_neg


def build_specs(args: argparse.Namespace, output_dir: Path) -> list[tuple[str, np.ndarray]]:
    stats = np.load(sample_gains_path(output_dir))
    pred = stats["pred"].astype(np.int64)
    drop_gains = stats["drop_gains"].astype(np.float32)
    insert_gains = stats["insert_gains"].astype(np.float32)
    blocks = stats["blocks"]

    aggregate_specs: list[tuple[str, str, float]] = []
    for quantile in args.quantile:
        aggregate_specs.append((f"q{int(quantile * 100):02d}", "quantile", quantile))
    for trim_ratio in args.trim_ratio:
        aggregate_specs.append((f"tr{int(trim_ratio * 100):02d}", "trimmed", trim_ratio))
    if args.include_mean:
        aggregate_specs.append(("mean", "mean", 0.0))

    specs: list[tuple[str, np.ndarray]] = []
    for agg_tag, mode, value in aggregate_specs:
        drop_pos, insert_pos, drop_neg, insert_neg = aggregate_components(
            drop_gains,
            insert_gains,
            pred,
            mode=mode,
            value=value,
        )
        for drop_weight in args.drop_weight:
            for penalty in args.penalty:
                binned = combine_gain(
                    drop_pos=drop_pos,
                    insert_pos=insert_pos,
                    drop_neg=drop_neg,
                    insert_neg=insert_neg,
                    drop_weight=drop_weight,
                    penalty=penalty,
                )
                score_tag = f"{agg_tag}_dw{int(drop_weight * 100):03d}"
                if penalty > 0:
                    score_tag += f"_p{int(penalty * 100):03d}"
                for power in args.power:
                    power_tag = "" if abs(power - 1.0) < 1e-9 else f"_pw{int(power * 100):03d}"
                    for beta in args.time_beta:
                        weights = expand_to_weights(binned, blocks, beta, power)
                        beta_tag = f"tb{int(beta * 100):03d}"
                        base_tag = f"{score_tag}{power_tag}_{beta_tag}"
                        specs.append((base_tag, weights))
                        for lam in args.contrast_lambda:
                            if lam <= 0:
                                continue
                            contrast = add_offline_contrast(weights, lam)
                            specs.append((f"{base_tag}_l{int(lam * 1000):03d}", contrast))
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantile and trimmed marginal TimeREISE search.")
    parser.add_argument("--base-model", default="external/gearxai_devkit/baselines/onnx/logic_lstm.onnx")
    parser.add_argument("--config", default="configs/spectral_lite_c.yaml")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--stats-batch-size", type=int, default=512)
    parser.add_argument("--num-bins", type=int, default=10)
    parser.add_argument("--stats-source", choices=["hf", "devkit"], default="devkit")
    parser.add_argument("--stats-data-dir", default="prepared_hf_val5k")
    parser.add_argument("--stats-split", default="validation")
    parser.add_argument("--normalize-stats", action="store_true")
    parser.add_argument("--baseline-channel-mean", type=float, nargs=8, default=[0.0] * 8)
    parser.add_argument("--output-dir", default="runs/logic_timereise_quantile_marginal_val5k_b10")
    parser.add_argument("--data-dir", default="prepared_hf_val5k")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--quantile", type=float, nargs="+", default=[0.60, 0.70, 0.75, 0.80, 0.85])
    parser.add_argument("--trim-ratio", type=float, nargs="+", default=[0.05, 0.10, 0.15])
    parser.add_argument("--drop-weight", type=float, nargs="+", default=[0.45, 0.50, 0.55, 0.60])
    parser.add_argument("--penalty", type=float, nargs="+", default=[0.05, 0.10, 0.15])
    parser.add_argument("--time-beta", type=float, nargs="+", default=[0.10, 0.15, 0.20])
    parser.add_argument("--power", type=float, nargs="+", default=[1.0])
    parser.add_argument("--contrast-lambda", type=float, nargs="+", default=[0.0])
    parser.add_argument("--include-mean", action="store_true")
    parser.add_argument("--copy-prefix", default="logic_timereise_quantile_marginal")
    parser.add_argument("--no-package", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    compute_sample_gains(args, output_dir)

    base_model = load_base_model(args.base_model)
    manifest = []
    for tag, weights in build_specs(args, output_dir):
        model_path = output_dir / f"logic_timereise_{tag}.onnx"
        if not model_path.exists():
            make_variant(base_model, output_dir, tag, weights, hard=False)
        manifest.append({"tag": tag, "model": str(model_path), "branch": "quantile_marginal"})
    print(f"Prepared {len(manifest)} quantile/trimmed marginal TimeREISE variants", flush=True)
    score_and_package_manifest(manifest, output_dir, args, copy_prefix=args.copy_prefix)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
