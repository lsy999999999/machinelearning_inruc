from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.run_logic_timereise_robust_search import build_robust_specs
from tools.run_logic_timereise_search import load_dataset, make_variant, normalize_importance, time_blocks
from tools.timereise_branch_utils import load_base_model, score_and_package_manifest, timereise_weights_from_stats


def mech_stats_path(output_dir: Path) -> Path:
    return output_dir / "offline_mech_stats.npz"


def local_mean_abs(abs_x: np.ndarray, kernel: int = 5) -> np.ndarray:
    pad = kernel // 2
    padded = np.pad(abs_x, ((0, 0), (0, 0), (pad, pad)), mode="edge")
    out = np.zeros_like(abs_x, dtype=np.float32)
    for offset in range(kernel):
        out += padded[:, :, offset : offset + abs_x.shape[2]]
    return out / float(kernel)


def mechanical_proxy(x_np: np.ndarray) -> np.ndarray:
    abs_x = np.abs(x_np).astype(np.float32)
    diff = np.zeros_like(abs_x, dtype=np.float32)
    diff[:, :, :-1] = np.abs(x_np[:, :, 1:] - x_np[:, :, :-1])
    peak = np.maximum(abs_x - local_mean_abs(abs_x, kernel=5), 0.0)

    rms = np.sqrt(np.mean(np.square(x_np), axis=2, keepdims=True) + 1e-12)
    crest = np.max(abs_x, axis=2, keepdims=True) / np.maximum(rms, 1e-6)
    crest = crest / np.maximum(crest.mean(axis=1, keepdims=True), 1e-6)

    return (0.45 * diff + 0.35 * peak + 0.20 * abs_x) * np.clip(crest, 0.5, 2.0)


def compute_mech_stats(args: argparse.Namespace, output_dir: Path) -> None:
    path = mech_stats_path(output_dir)
    if path.exists():
        print(f"Offline mechanical stats already exist: {path}", flush=True)
        return

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
    proxy_sums = np.zeros((9, 8, args.num_bins), dtype=np.float64)
    counts = np.zeros((9,), dtype=np.float64)
    processed = 0

    for x, _y in tqdm(loader, desc="offline-mech-stats", leave=True):
        x_np = x.numpy().astype(np.float32)
        probabilities = session.run(output_names, {input_name: x_np})[prob_index]
        pred = probabilities.argmax(axis=1).astype(np.int64)
        proxy = mechanical_proxy(x_np)

        for class_id in range(9):
            mask = pred == class_id
            counts[class_id] += float(mask.sum())
            if not np.any(mask):
                continue
            for bin_id, (start, end) in enumerate(blocks):
                proxy_sums[class_id, :, bin_id] += proxy[mask, :, start:end].mean(axis=2).sum(axis=0)
        processed += int(x_np.shape[0])

    np.savez_compressed(
        path,
        proxy_sums=proxy_sums,
        counts=counts,
        processed=np.asarray(processed, dtype=np.int64),
        num_bins=np.asarray(args.num_bins, dtype=np.int64),
        blocks=np.asarray(blocks, dtype=np.int64),
    )
    print(f"Saved offline mechanical stats: {path} processed={processed}", flush=True)


def expanded_mechanical_prior(stats_file: str | Path) -> np.ndarray:
    stats = np.load(stats_file)
    counts = np.maximum(stats["counts"].astype(np.float64), 1.0)
    proxy = stats["proxy_sums"].astype(np.float64) / counts[:, None, None]
    binned = normalize_importance(proxy)
    expanded = np.zeros((9, 8, 100), dtype=np.float32)
    for bin_id, (start, end) in enumerate(stats["blocks"].tolist()):
        expanded[:, :, int(start) : int(end)] = binned[:, :, bin_id : bin_id + 1]
    expanded = expanded / np.maximum(expanded.mean(axis=(1, 2), keepdims=True), 1e-6)
    return expanded.astype(np.float32)


def add_mechanical_prior(weights: np.ndarray, prior: np.ndarray, gamma: float) -> np.ndarray:
    fused = weights * np.clip(1.0 + gamma * (prior - 1.0), 0.05, None)
    fused = np.clip(fused, 0.05, None)
    fused = fused / np.maximum(fused.mean(axis=(1, 2), keepdims=True), 1e-6)
    return fused.astype(np.float32)


def add_offline_contrast(weights: np.ndarray, lam: float) -> np.ndarray:
    sum_all = weights.sum(axis=0, keepdims=True)
    mean_other = (sum_all - weights) / 8.0
    shared_other_evidence = np.maximum(mean_other - 1.0, 0.0)
    contrasted = np.clip(weights - lam * shared_other_evidence, 0.05, None)
    contrasted = contrasted / np.maximum(contrasted.mean(axis=(1, 2), keepdims=True), 1e-6)
    return contrasted.astype(np.float32)


def robust_mean_weights(args: argparse.Namespace) -> np.ndarray:
    robust_args = argparse.Namespace(
        global_stats=args.stats,
        aggregate=["mean"],
        time_beta=[args.time_beta],
    )
    specs = build_robust_specs(robust_args, Path(args.robust_dir))
    for tag, weights in specs:
        if tag == f"robust_mean_mix50_tb{int(args.time_beta * 100):03d}":
            return weights
    raise RuntimeError("Could not build robust mean weights.")


def build_specs(args: argparse.Namespace, output_dir: Path) -> list[tuple[str, np.ndarray]]:
    base_weights = timereise_weights_from_stats(args.stats, args.time_beta)
    robust_weights = robust_mean_weights(args)
    prior = expanded_mechanical_prior(mech_stats_path(output_dir))

    specs: list[tuple[str, np.ndarray]] = []
    beta_tag = f"tb{int(args.time_beta * 100):03d}"

    for gamma in args.mech_gamma:
        specs.append((f"offline_mech_mix50_{beta_tag}_g{int(gamma * 1000):03d}", add_mechanical_prior(base_weights, prior, gamma)))
        specs.append((f"robust_offline_mech_mix50_{beta_tag}_g{int(gamma * 1000):03d}", add_mechanical_prior(robust_weights, prior, gamma)))

    for lam in args.contrast_lambda:
        specs.append((f"offline_contrast_mix50_{beta_tag}_l{int(lam * 1000):03d}", add_offline_contrast(base_weights, lam)))
        specs.append((f"robust_offline_contrast_mix50_{beta_tag}_l{int(lam * 1000):03d}", add_offline_contrast(robust_weights, lam)))

    for gamma in args.combo_mech_gamma:
        for lam in args.combo_contrast_lambda:
            mech = add_mechanical_prior(robust_weights, prior, gamma)
            combo = add_offline_contrast(mech, lam)
            tag = f"robust_offline_combo_mix50_{beta_tag}_g{int(gamma * 1000):03d}_l{int(lam * 1000):03d}"
            specs.append((tag, combo))

    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Low-complexity offline TimeREISE innovation search.")
    parser.add_argument("--base-model", default="external/gearxai_devkit/baselines/onnx/logic_lstm.onnx")
    parser.add_argument("--config", default="configs/spectral_lite_c.yaml")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=50000)
    parser.add_argument("--stats-batch-size", type=int, default=256)
    parser.add_argument("--num-bins", type=int, default=20)
    parser.add_argument("--stats", default="runs/logic_timereise_search_50k_b20_refine/timereise_stats.npz")
    parser.add_argument("--robust-dir", default="runs/logic_timereise_robust_search_50k_b20")
    parser.add_argument("--output-dir", default="runs/logic_timereise_offline_innovation_search")
    parser.add_argument("--data-dir", default="prepared_hf_val5k")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--time-beta", type=float, default=0.35)
    parser.add_argument("--mech-gamma", type=float, nargs="+", default=[0.02, 0.05, 0.08])
    parser.add_argument("--contrast-lambda", type=float, nargs="+", default=[0.03, 0.05, 0.08])
    parser.add_argument("--combo-mech-gamma", type=float, nargs="+", default=[0.02, 0.05])
    parser.add_argument("--combo-contrast-lambda", type=float, nargs="+", default=[0.03, 0.05])
    parser.add_argument("--copy-prefix", default="logic_timereise_offline_innovation")
    parser.add_argument("--no-package", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    compute_mech_stats(args, output_dir)

    base_model = load_base_model(args.base_model)
    manifest = []
    for tag, weights in build_specs(args, output_dir):
        model_path = output_dir / f"logic_timereise_{tag}.onnx"
        if not model_path.exists():
            make_variant(base_model, output_dir, tag, weights, hard=False)
        manifest.append({"tag": tag, "model": str(model_path), "branch": "offline_innovation"})
    print(f"Prepared {len(manifest)} offline innovation TimeREISE variants", flush=True)
    score_and_package_manifest(manifest, output_dir, args, copy_prefix=args.copy_prefix)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
