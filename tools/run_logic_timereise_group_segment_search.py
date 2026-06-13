from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.run_logic_timereise_power_ensemble_search import extract_timereise_weights, normalize_weights
from tools.run_logic_timereise_search import expand_bins, make_variant, normalize_importance, time_blocks
from tools.timereise_branch_utils import load_base_model
from tools.timereise_teacher_utils import score_and_package_teacher_manifest


def parse_groups(value: str) -> list[list[int]]:
    groups: list[list[int]] = []
    for group_text in value.split("|"):
        group = [int(part) for part in group_text.split(",") if part.strip()]
        if not group:
            raise argparse.ArgumentTypeError("empty channel group")
        if any(channel < 0 or channel > 7 for channel in group):
            raise argparse.ArgumentTypeError("channel ids must be in [0, 7]")
        groups.append(group)
    flat = [channel for group in groups for channel in group]
    if sorted(flat) != list(range(8)):
        raise argparse.ArgumentTypeError("channel groups must cover each channel exactly once")
    return groups


def load_windows(data_dir: str | Path, split: str) -> np.ndarray:
    return np.load(Path(data_dir) / f"{split}_windows.npy", mmap_mode="r")


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


def stats_path(output_dir: Path) -> Path:
    return output_dir / "group_segment_stats.npz"


def iter_batches(windows: np.ndarray, batch_size: int, max_samples: int | None) -> tuple[int, int]:
    total = len(windows) if max_samples is None else min(int(max_samples), len(windows))
    for start in range(0, total, batch_size):
        yield start, min(start + batch_size, total)


def compute_group_stats(args: argparse.Namespace, output_dir: Path, groups: list[list[int]]) -> None:
    path = stats_path(output_dir)
    if path.exists():
        print(f"Group-segment stats already exist: {path}", flush=True)
        return

    windows = load_windows(args.data_dir, args.stats_split)
    session = ort.InferenceSession(args.stats_model, providers=["CPUExecutionProvider"])
    input_name, output_names, prob_index = probability_output_index(session)
    blocks = time_blocks(args.num_bins)

    shape = (9, len(groups), args.num_bins)
    drop_pos_sums = np.zeros(shape, dtype=np.float64)
    drop_neg_sums = np.zeros(shape, dtype=np.float64)
    insert_pos_sums = np.zeros(shape, dtype=np.float64)
    insert_neg_sums = np.zeros(shape, dtype=np.float64)
    counts = np.zeros((9,), dtype=np.float64)
    abs_bin_sums = np.zeros((9, 8, args.num_bins), dtype=np.float64)
    processed = 0
    baseline_channel = np.asarray(args.baseline_channel_mean, dtype=np.float32).reshape(1, 8, 1)

    batches = list(iter_batches(windows, args.stats_batch_size, args.max_samples))
    for start_idx, end_idx in tqdm(batches, desc="group-segment-stats", leave=True):
        x_np = np.asarray(windows[start_idx:end_idx], dtype=np.float32)
        probabilities = session.run(output_names, {input_name: x_np})[prob_index]
        pred = probabilities.argmax(axis=1).astype(np.int64)
        rows = np.arange(probabilities.shape[0])
        base_conf = probabilities[rows, pred]

        baseline = np.broadcast_to(baseline_channel, x_np.shape).astype(np.float32).copy()
        baseline_prob = session.run(output_names, {input_name: baseline})[prob_index]
        baseline_conf = baseline_prob[rows, pred]
        counts += class_bincount(pred)

        abs_x = np.abs(x_np).astype(np.float32)
        for bin_id, (block_start, block_end) in enumerate(blocks):
            block_abs = abs_x[:, :, block_start:block_end].mean(axis=2)
            for class_id in range(9):
                mask = pred == class_id
                if np.any(mask):
                    abs_bin_sums[class_id, :, bin_id] += block_abs[mask].sum(axis=0)

        for group_id, channels in enumerate(groups):
            channel_index = np.asarray(channels, dtype=np.int64)
            for bin_id, (block_start, block_end) in enumerate(blocks):
                deleted = x_np.copy()
                deleted[:, channel_index, block_start:block_end] = baseline_channel[:, channel_index, :]
                deleted_prob = session.run(output_names, {input_name: deleted})[prob_index]
                deleted_conf = deleted_prob[rows, pred]
                drop_gain = base_conf - deleted_conf

                inserted = baseline.copy()
                inserted[:, channel_index, block_start:block_end] = x_np[:, channel_index, block_start:block_end]
                inserted_prob = session.run(output_names, {input_name: inserted})[prob_index]
                inserted_conf = inserted_prob[rows, pred]
                insert_gain = inserted_conf - baseline_conf

                drop_pos_sums[:, group_id, bin_id] += class_bincount(pred, np.maximum(drop_gain, 0.0))
                drop_neg_sums[:, group_id, bin_id] += class_bincount(pred, np.maximum(-drop_gain, 0.0))
                insert_pos_sums[:, group_id, bin_id] += class_bincount(pred, np.maximum(insert_gain, 0.0))
                insert_neg_sums[:, group_id, bin_id] += class_bincount(pred, np.maximum(-insert_gain, 0.0))

        processed += int(x_np.shape[0])

    np.savez_compressed(
        path,
        drop_pos_sums=drop_pos_sums,
        drop_neg_sums=drop_neg_sums,
        insert_pos_sums=insert_pos_sums,
        insert_neg_sums=insert_neg_sums,
        abs_bin_sums=abs_bin_sums,
        counts=counts,
        processed=np.asarray(processed, dtype=np.int64),
        blocks=np.asarray(blocks, dtype=np.int64),
        groups=np.asarray([",".join(map(str, group)) for group in groups]),
        baseline_channel_mean=np.asarray(args.baseline_channel_mean, dtype=np.float32),
    )
    print(f"Saved group-segment stats: {path} processed={processed}", flush=True)


def positive_mean(stats: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    counts = np.maximum(stats["counts"].astype(np.float64), 1.0)
    return stats[key].astype(np.float64) / counts[:, None, None]


def combine_gain(
    drop_pos: np.ndarray,
    insert_pos: np.ndarray,
    drop_neg: np.ndarray,
    insert_neg: np.ndarray,
    drop_weight: float,
    penalty: float,
) -> np.ndarray:
    score = drop_weight * normalize_importance(drop_pos) + (1.0 - drop_weight) * normalize_importance(insert_pos)
    if penalty > 0:
        score = np.maximum(score - penalty * normalize_importance(drop_neg + insert_neg), 0.0)
    return normalize_importance(score)


def group_to_channel(
    group_scores: np.ndarray,
    groups: list[list[int]],
    abs_prior: np.ndarray,
    mode: str,
) -> np.ndarray:
    out = np.ones((9, 8, group_scores.shape[2]), dtype=np.float64)
    for group_id, channels in enumerate(groups):
        for channel in channels:
            out[:, channel, :] = group_scores[:, group_id, :]
    if mode == "uniform":
        return normalize_importance(out)
    if mode == "absmod":
        return normalize_importance(out * abs_prior)
    raise ValueError(f"Unknown channel mode: {mode}")


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


def geometric_blend(source: np.ndarray, candidate: np.ndarray, alpha: float) -> np.ndarray:
    if alpha <= 0:
        return normalize_weights(source)
    if alpha >= 1:
        return normalize_weights(candidate)
    fused = np.exp((1.0 - alpha) * np.log(np.maximum(source, 1e-6)) + alpha * np.log(np.maximum(candidate, 1e-6)))
    return normalize_weights(fused)


def build_specs(args: argparse.Namespace, output_dir: Path, groups: list[list[int]]) -> list[tuple[str, np.ndarray]]:
    stats = np.load(stats_path(output_dir))
    drop_pos = positive_mean(stats, "drop_pos_sums")
    insert_pos = positive_mean(stats, "insert_pos_sums")
    drop_neg = positive_mean(stats, "drop_neg_sums")
    insert_neg = positive_mean(stats, "insert_neg_sums")
    abs_prior = normalize_importance(positive_mean(stats, "abs_bin_sums"))
    blocks = stats["blocks"]
    source_weights = extract_timereise_weights(args.source_model)

    specs: list[tuple[str, np.ndarray]] = [("start", source_weights)]
    for drop_weight in args.drop_weight:
        for penalty in args.penalty:
            group_score = combine_gain(drop_pos, insert_pos, drop_neg, insert_neg, drop_weight, penalty)
            for channel_mode in args.channel_mode:
                binned = group_to_channel(group_score, groups, abs_prior, channel_mode)
                for power in args.power:
                    for beta in args.time_beta:
                        candidate = expand_to_weights(binned, blocks, beta, power)
                        base_tag = (
                            f"gs_{channel_mode}_dw{int(drop_weight * 100):03d}"
                            f"_p{int(penalty * 100):03d}_pw{int(power * 100):03d}_tb{int(beta * 100):03d}"
                        )
                        if args.include_pure:
                            specs.append((f"{base_tag}_pure", candidate))
                        for alpha in args.fuse_alpha:
                            if alpha <= 0:
                                continue
                            specs.append((f"{base_tag}_a{int(alpha * 100):03d}", geometric_blend(source_weights, candidate, alpha)))
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Group-segment TimeREISE search with teacher-score ranking.")
    parser.add_argument("--base-model", default="runs/candidates/logic_timereise_logit_folded_gemm_bestteacher.onnx")
    parser.add_argument("--source-model", default="runs/candidates/logic_timereise_logit_folded_gemm_bestteacher.onnx")
    parser.add_argument("--stats-model", default="")
    parser.add_argument("--output-dir", default="runs/logic_timereise_group_segment_search")
    parser.add_argument("--data-dir", default="prepared_hf_val5k")
    parser.add_argument("--stats-split", default="validation")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--stats-batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--channel-groups", type=parse_groups, default=parse_groups("0|1,2,3|4|5,6,7"))
    parser.add_argument("--num-bins", type=int, default=10)
    parser.add_argument("--baseline-channel-mean", type=float, nargs=8, default=[0.0] * 8)
    parser.add_argument("--drop-weight", type=float, nargs="+", default=[0.4, 0.5, 0.6])
    parser.add_argument("--penalty", type=float, nargs="+", default=[0.05, 0.10])
    parser.add_argument("--time-beta", type=float, nargs="+", default=[0.05, 0.10, 0.15, 0.20])
    parser.add_argument("--power", type=float, nargs="+", default=[0.90, 1.00, 1.15])
    parser.add_argument("--channel-mode", choices=["uniform", "absmod"], nargs="+", default=["uniform", "absmod"])
    parser.add_argument("--fuse-alpha", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.35])
    parser.add_argument("--include-pure", action="store_true")
    parser.add_argument("--copy-prefix", default="logic_timereise_group_segment")
    parser.add_argument("--no-package", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.stats_model:
        args.stats_model = args.source_model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = args.channel_groups
    compute_group_stats(args, output_dir, groups)

    base_model = load_base_model(args.base_model)
    manifest = []
    seen_tags: set[str] = set()
    for tag, weights in build_specs(args, output_dir, groups):
        if tag in seen_tags:
            continue
        seen_tags.add(tag)
        model_path = output_dir / f"logic_timereise_{tag}.onnx"
        if not model_path.exists():
            make_variant(base_model, output_dir, tag, weights, hard=False)
        manifest.append({"tag": tag, "model": str(model_path), "branch": "group_segment"})

    print(f"Prepared {len(manifest)} group-segment variants", flush=True)
    score_and_package_teacher_manifest(manifest, output_dir, args, copy_prefix=args.copy_prefix)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
