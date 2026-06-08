from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gearxai_project.data import GearXAIWindows
from gearxai_project.utils import load_config
from tools.run_logic_timereise_offline_innovation_search import add_offline_contrast
from tools.run_logic_timereise_search import expand_bins, make_variant, normalize_importance, time_blocks
from tools.timereise_branch_utils import load_base_model, score_and_package_manifest


def marginal_stats_path(output_dir: Path) -> Path:
    return output_dir / "timereise_marginal_stats.npz"


def load_hf_batches(args: argparse.Namespace) -> Iterator[np.ndarray]:
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    dataset = GearXAIWindows(
        split=args.split,
        dataset_name=data_cfg["dataset_name"],
        config_name=data_cfg["config_name"],
        cache_dir=data_cfg.get("cache_dir"),
        max_samples=args.max_samples,
        normalize=args.normalize_stats,
        seed=int(cfg["training"].get("seed", 42)) + 29,
    )
    loader = DataLoader(dataset, batch_size=args.stats_batch_size, shuffle=False, num_workers=0, drop_last=False)
    for x, _y in loader:
        yield x.numpy().astype(np.float32, copy=False)


def load_devkit_batches(args: argparse.Namespace) -> Iterator[np.ndarray]:
    path = Path(args.stats_data_dir) / f"{args.stats_split}_windows.npy"
    windows = np.load(path, mmap_mode="r")
    max_samples = len(windows) if args.max_samples is None else min(int(args.max_samples), len(windows))
    for start in range(0, max_samples, args.stats_batch_size):
        end = min(start + args.stats_batch_size, max_samples)
        yield np.asarray(windows[start:end], dtype=np.float32)


def iter_stat_batches(args: argparse.Namespace) -> tuple[Iterator[np.ndarray], str]:
    if args.stats_source == "devkit":
        return load_devkit_batches(args), f"devkit:{args.stats_data_dir}:{args.stats_split}"
    return load_hf_batches(args), f"hf:{args.split}:normalize={args.normalize_stats}"


def probability_output_index(session: ort.InferenceSession) -> tuple[str, list[str], int]:
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    for index, output in enumerate(session.get_outputs()):
        shape = output.shape
        if len(shape) == 2 and shape[-1] == 9:
            return input_name, output_names, index
    raise RuntimeError("Could not identify probability output.")


def class_bincount(pred: np.ndarray, values: np.ndarray | None = None) -> np.ndarray:
    return np.bincount(pred.astype(np.int64), weights=values, minlength=9).astype(np.float64)


def compute_marginal_stats(args: argparse.Namespace, output_dir: Path) -> None:
    path = marginal_stats_path(output_dir)
    if path.exists():
        print(f"Marginal stats already exist: {path}", flush=True)
        return

    batch_iter, source_description = iter_stat_batches(args)
    session = ort.InferenceSession(args.base_model, providers=["CPUExecutionProvider"])
    input_name, output_names, prob_index = probability_output_index(session)

    blocks = time_blocks(args.num_bins)
    shape = (9, 8, args.num_bins)
    drop_pos_sums = np.zeros(shape, dtype=np.float64)
    drop_neg_sums = np.zeros(shape, dtype=np.float64)
    drop_signed_sums = np.zeros(shape, dtype=np.float64)
    insert_pos_sums = np.zeros(shape, dtype=np.float64)
    insert_neg_sums = np.zeros(shape, dtype=np.float64)
    insert_signed_sums = np.zeros(shape, dtype=np.float64)
    counts = np.zeros((9,), dtype=np.float64)
    base_conf_sums = np.zeros((9,), dtype=np.float64)
    baseline_conf_sums = np.zeros((9,), dtype=np.float64)
    processed = 0

    baseline_channel = np.asarray(args.baseline_channel_mean, dtype=np.float32).reshape(1, 8, 1)

    for x_np in tqdm(batch_iter, desc="marginal-faith-stats", leave=True):
        probabilities = session.run(output_names, {input_name: x_np})[prob_index]
        pred = probabilities.argmax(axis=1).astype(np.int64)
        rows = np.arange(probabilities.shape[0])
        base_conf = probabilities[rows, pred]

        baseline = np.broadcast_to(baseline_channel, x_np.shape).astype(np.float32).copy()
        baseline_prob = session.run(output_names, {input_name: baseline})[prob_index]
        baseline_conf = baseline_prob[rows, pred]

        counts += class_bincount(pred)
        base_conf_sums += class_bincount(pred, base_conf)
        baseline_conf_sums += class_bincount(pred, baseline_conf)

        for channel in range(8):
            for bin_id, (start, end) in enumerate(blocks):
                deleted = x_np.copy()
                deleted[:, channel, start:end] = baseline_channel[:, channel, :]
                deleted_prob = session.run(output_names, {input_name: deleted})[prob_index]
                deleted_conf = deleted_prob[rows, pred]
                drop_gain = base_conf - deleted_conf

                inserted = baseline.copy()
                inserted[:, channel, start:end] = x_np[:, channel, start:end]
                inserted_prob = session.run(output_names, {input_name: inserted})[prob_index]
                inserted_conf = inserted_prob[rows, pred]
                insert_gain = inserted_conf - baseline_conf

                drop_pos_sums[:, channel, bin_id] += class_bincount(pred, np.maximum(drop_gain, 0.0))
                drop_neg_sums[:, channel, bin_id] += class_bincount(pred, np.maximum(-drop_gain, 0.0))
                drop_signed_sums[:, channel, bin_id] += class_bincount(pred, drop_gain)
                insert_pos_sums[:, channel, bin_id] += class_bincount(pred, np.maximum(insert_gain, 0.0))
                insert_neg_sums[:, channel, bin_id] += class_bincount(pred, np.maximum(-insert_gain, 0.0))
                insert_signed_sums[:, channel, bin_id] += class_bincount(pred, insert_gain)

        processed += int(x_np.shape[0])

    np.savez_compressed(
        path,
        drop_pos_sums=drop_pos_sums,
        drop_neg_sums=drop_neg_sums,
        drop_signed_sums=drop_signed_sums,
        insert_pos_sums=insert_pos_sums,
        insert_neg_sums=insert_neg_sums,
        insert_signed_sums=insert_signed_sums,
        counts=counts,
        base_conf_sums=base_conf_sums,
        baseline_conf_sums=baseline_conf_sums,
        processed=np.asarray(processed, dtype=np.int64),
        num_bins=np.asarray(args.num_bins, dtype=np.int64),
        blocks=np.asarray(blocks, dtype=np.int64),
        baseline_channel_mean=np.asarray(args.baseline_channel_mean, dtype=np.float32),
        source_description=np.asarray(source_description),
    )
    print(f"Saved marginal stats: {path} processed={processed} source={source_description}", flush=True)


def positive_mean(stats: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    counts = np.maximum(stats["counts"].astype(np.float64), 1.0)
    return stats[key].astype(np.float64) / counts[:, None, None]


def classwise_positive(values: np.ndarray) -> np.ndarray:
    values = np.maximum(values, 0.0)
    return normalize_importance(values)


def combine_gain(
    *,
    drop_pos: np.ndarray,
    insert_pos: np.ndarray,
    drop_signed: np.ndarray,
    insert_signed: np.ndarray,
    drop_neg: np.ndarray,
    insert_neg: np.ndarray,
    drop_weight: float,
    penalty: float,
    signed: bool,
) -> np.ndarray:
    if signed:
        raw_drop = np.maximum(drop_signed, 0.0)
        raw_insert = np.maximum(insert_signed, 0.0)
    else:
        raw_drop = drop_pos
        raw_insert = insert_pos
    score = drop_weight * classwise_positive(raw_drop) + (1.0 - drop_weight) * classwise_positive(raw_insert)
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


def build_specs(args: argparse.Namespace, output_dir: Path) -> list[tuple[str, np.ndarray]]:
    stats = np.load(marginal_stats_path(output_dir))
    drop_pos = positive_mean(stats, "drop_pos_sums")
    insert_pos = positive_mean(stats, "insert_pos_sums")
    drop_neg = positive_mean(stats, "drop_neg_sums")
    insert_neg = positive_mean(stats, "insert_neg_sums")
    drop_signed = positive_mean(stats, "drop_signed_sums")
    insert_signed = positive_mean(stats, "insert_signed_sums")
    blocks = stats["blocks"]

    score_specs: list[tuple[str, np.ndarray]] = []
    if args.include_components:
        score_specs.extend(
            [
                ("drop", classwise_positive(drop_pos)),
                ("insert", classwise_positive(insert_pos)),
                ("signed_drop", classwise_positive(np.maximum(drop_signed, 0.0))),
                ("signed_insert", classwise_positive(np.maximum(insert_signed, 0.0))),
            ]
        )

    for drop_weight in args.drop_weight:
        for penalty in args.penalty:
            for signed in ([False, True] if args.include_signed else [False]):
                tag = f"marg_dw{int(drop_weight * 100):03d}"
                if penalty > 0:
                    tag += f"_p{int(penalty * 100):03d}"
                if signed:
                    tag += "_signed"
                score_specs.append(
                    (
                        tag,
                        combine_gain(
                            drop_pos=drop_pos,
                            insert_pos=insert_pos,
                            drop_signed=drop_signed,
                            insert_signed=insert_signed,
                            drop_neg=drop_neg,
                            insert_neg=insert_neg,
                            drop_weight=drop_weight,
                            penalty=penalty,
                            signed=signed,
                        ),
                    )
                )

    specs: list[tuple[str, np.ndarray]] = []
    for score_tag, binned in score_specs:
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
    parser = argparse.ArgumentParser(description="Faith-targeted marginal TimeREISE search.")
    parser.add_argument("--base-model", default="external/gearxai_devkit/baselines/onnx/logic_lstm.onnx")
    parser.add_argument("--config", default="configs/spectral_lite_c.yaml")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=50000)
    parser.add_argument("--stats-batch-size", type=int, default=256)
    parser.add_argument("--num-bins", type=int, default=20)
    parser.add_argument("--stats-source", choices=["hf", "devkit"], default="hf")
    parser.add_argument("--stats-data-dir", default="prepared_hf_val5k")
    parser.add_argument("--stats-split", default="validation")
    parser.add_argument("--normalize-stats", action="store_true")
    parser.add_argument("--baseline-channel-mean", type=float, nargs=8, default=[0.0] * 8)
    parser.add_argument("--output-dir", default="runs/logic_timereise_marginal_search")
    parser.add_argument("--data-dir", default="prepared_hf_val5k")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--time-beta", type=float, nargs="+", default=[0.25, 0.35, 0.50, 0.80])
    parser.add_argument("--drop-weight", type=float, nargs="+", default=[0.4, 0.5, 0.6])
    parser.add_argument("--penalty", type=float, nargs="+", default=[0.0, 0.25])
    parser.add_argument("--power", type=float, nargs="+", default=[1.0])
    parser.add_argument("--contrast-lambda", type=float, nargs="+", default=[0.0, 0.05])
    parser.add_argument("--include-components", action="store_true")
    parser.add_argument("--include-signed", action="store_true")
    parser.add_argument("--include-hard", action="store_true")
    parser.add_argument("--copy-prefix", default="logic_timereise_marginal")
    parser.add_argument("--no-package", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    compute_marginal_stats(args, output_dir)

    base_model = load_base_model(args.base_model)
    manifest = []
    for tag, weights in build_specs(args, output_dir):
        for hard in ([False, True] if args.include_hard else [False]):
            suffix = f"{tag}_hard" if hard else tag
            model_path = output_dir / f"logic_timereise_{suffix}.onnx"
            if not model_path.exists():
                make_variant(base_model, output_dir, tag, weights, hard)
            manifest.append({"tag": suffix, "model": str(model_path), "branch": "marginal"})
    print(f"Prepared {len(manifest)} marginal TimeREISE variants", flush=True)
    score_and_package_manifest(manifest, output_dir, args, copy_prefix=args.copy_prefix)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
