from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.run_logic_timereise_power_ensemble_search import extract_timereise_weights
from tools.run_logic_timereise_search import make_variant, normalize_importance, time_blocks
from tools.timereise_branch_utils import load_base_model, score_and_package_manifest


def load_split(data_dir: str | Path, split: str) -> np.ndarray:
    data_dir = Path(data_dir)
    return np.load(data_dir / f"{split}_windows.npy").astype(np.float32)


def run_probabilities(model_path: str | Path, windows: np.ndarray, batch_size: int) -> np.ndarray:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    chunks: list[np.ndarray] = []
    for start in range(0, len(windows), batch_size):
        end = min(start + batch_size, len(windows))
        outputs = session.run(output_names, {input_name: windows[start:end].astype(np.float32, copy=False)})
        prob = None
        for output in outputs:
            if output.ndim == 2 and output.shape[-1] == 9:
                prob = output
                break
        if prob is None:
            raise RuntimeError("Could not identify probability output.")
        chunks.append(prob.astype(np.float32))
    return np.concatenate(chunks, axis=0)


def local_mean(values: np.ndarray, kernel: int) -> np.ndarray:
    pad = kernel // 2
    padded = np.pad(values, ((0, 0), (0, 0), (pad, pad)), mode="edge")
    out = np.zeros_like(values, dtype=np.float32)
    for offset in range(kernel):
        out += padded[:, :, offset : offset + values.shape[2]]
    return out / float(kernel)


def lag_autocorr_proxy(x: np.ndarray, max_lag: int = 8) -> np.ndarray:
    centered = x - x.mean(axis=2, keepdims=True)
    denom = np.mean(centered * centered, axis=2, keepdims=True) + 1e-6
    best = np.zeros_like(x, dtype=np.float32)
    for lag in range(1, max_lag + 1):
        corr = np.zeros_like(x, dtype=np.float32)
        prod = centered[:, :, lag:] * centered[:, :, :-lag]
        corr[:, :, lag:] = np.maximum(prod / denom, 0.0)
        best = np.maximum(best, corr)
    return best


def proxy_maps(windows: np.ndarray) -> dict[str, np.ndarray]:
    abs_x = np.abs(windows).astype(np.float32)
    energy = np.square(windows).astype(np.float32)

    diff = np.zeros_like(abs_x, dtype=np.float32)
    diff[:, :, 1:] = np.abs(windows[:, :, 1:] - windows[:, :, :-1])

    smooth_abs = local_mean(abs_x, kernel=5)
    peak = np.maximum(abs_x - smooth_abs, 0.0)
    envelope = local_mean(abs_x, kernel=9)

    rms = np.sqrt(np.mean(np.square(windows), axis=2, keepdims=True) + 1e-12)
    crest = np.max(abs_x, axis=2, keepdims=True) / np.maximum(rms, 1e-6)
    crest = crest / np.maximum(crest.mean(axis=1, keepdims=True), 1e-6)
    crest_abs = abs_x * np.clip(crest, 0.5, 2.0)

    mean = windows.mean(axis=2, keepdims=True)
    centered = windows - mean
    var = np.mean(centered * centered, axis=2, keepdims=True) + 1e-6
    kurt_scale = np.mean(centered**4, axis=2, keepdims=True) / (var * var)
    kurt_scale = kurt_scale / np.maximum(kurt_scale.mean(axis=1, keepdims=True), 1e-6)
    kurt_abs = abs_x * np.clip(kurt_scale, 0.5, 3.0)

    autocorr = lag_autocorr_proxy(windows)

    torque_energy = energy[:, 4:5, :]
    torque_scale = torque_energy / np.maximum(torque_energy.mean(axis=2, keepdims=True), 1e-6)
    torque_coupled = abs_x * np.clip(torque_scale, 0.5, 2.0)

    combo_impulse = 0.45 * diff + 0.35 * peak + 0.20 * abs_x
    combo_bearing = 0.40 * envelope + 0.35 * kurt_abs + 0.25 * diff
    combo_periodic = 0.45 * autocorr + 0.35 * envelope + 0.20 * abs_x

    return {
        "abs": abs_x,
        "energy": energy,
        "diff": diff,
        "peak": peak,
        "envelope": envelope,
        "crest_abs": crest_abs,
        "kurt_abs": kurt_abs,
        "autocorr": autocorr,
        "torque_coupled": torque_coupled,
        "combo_impulse": combo_impulse,
        "combo_bearing": combo_bearing,
        "combo_periodic": combo_periodic,
    }


def binned_prior(
    proxy: np.ndarray,
    pred: np.ndarray,
    num_bins: int,
) -> np.ndarray:
    blocks = time_blocks(num_bins)
    sums = np.zeros((9, 8, num_bins), dtype=np.float64)
    counts = np.zeros((9,), dtype=np.float64)
    for class_id in range(9):
        mask = pred == class_id
        counts[class_id] = float(mask.sum())
        if not np.any(mask):
            continue
        for bin_id, (start, end) in enumerate(blocks):
            sums[class_id, :, bin_id] = proxy[mask, :, start:end].mean(axis=2).sum(axis=0)
    prior = sums / np.maximum(counts[:, None, None], 1.0)
    prior = normalize_importance(prior)
    expanded = np.zeros((9, 8, 100), dtype=np.float32)
    for bin_id, (start, end) in enumerate(blocks):
        expanded[:, :, start:end] = prior[:, :, bin_id : bin_id + 1]
    expanded = expanded / np.maximum(expanded.mean(axis=(1, 2), keepdims=True), 1e-6)
    return expanded.astype(np.float32)


def fuse_linear(weights: np.ndarray, prior: np.ndarray, gamma: float) -> np.ndarray:
    fused = weights * np.clip(1.0 + gamma * (prior - 1.0), 0.05, None)
    fused = np.clip(fused, 0.05, None)
    fused = fused / np.maximum(fused.mean(axis=(1, 2), keepdims=True), 1e-6)
    return fused.astype(np.float32)


def fuse_power(weights: np.ndarray, prior: np.ndarray, gamma: float) -> np.ndarray:
    fused = weights * np.power(np.maximum(prior, 0.05), gamma)
    fused = np.clip(fused, 0.05, None)
    fused = fused / np.maximum(fused.mean(axis=(1, 2), keepdims=True), 1e-6)
    return fused.astype(np.float32)


def gamma_tag(gamma: float) -> str:
    sign = "p" if gamma >= 0 else "m"
    return f"{sign}{int(round(abs(gamma) * 10000)):04d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search folded TimeREISE fusion with validation-aligned physical proxies.")
    parser.add_argument("--base-model", default="external/gearxai_devkit/baselines/onnx/logic_lstm.onnx")
    parser.add_argument("--source-model", default="runs/candidates/logic_timereise_class_candidate_selection_bestproxy.onnx")
    parser.add_argument("--output-dir", default="runs/logic_timereise_phys_proxy_search")
    parser.add_argument("--data-dir", default="prepared_hf_val5k")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--num-bins", type=int, default=10)
    parser.add_argument("--gamma", type=float, nargs="+", default=[-0.002, -0.001, -0.0005, 0.0002, 0.0004, 0.0006, 0.0008, 0.001, 0.0015, 0.002, 0.003, 0.005])
    parser.add_argument("--mode", choices=["linear", "power"], nargs="+", default=["linear", "power"])
    parser.add_argument("--copy-prefix", default="logic_timereise_phys_proxy_search")
    parser.add_argument("--no-package", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    windows = load_split(args.data_dir, args.eval_split)
    probabilities = run_probabilities(args.source_model, windows, args.eval_batch_size)
    pred = probabilities.argmax(axis=1).astype(np.int64)
    source_weights = extract_timereise_weights(args.source_model)

    cache_path = output_dir / "phys_priors.npz"
    if cache_path.exists():
        loaded = np.load(cache_path)
        priors = {name: loaded[name].astype(np.float32) for name in loaded.files}
        print(f"Loaded physical priors: {cache_path}", flush=True)
    else:
        priors = {
            name: binned_prior(proxy, pred, args.num_bins)
            for name, proxy in proxy_maps(windows).items()
        }
        np.savez_compressed(cache_path, **priors)
        print(f"Saved physical priors: {cache_path}", flush=True)

    base_model = load_base_model(args.base_model)
    manifest = []
    for proxy_name, prior in priors.items():
        for mode in args.mode:
            for gamma in args.gamma:
                if abs(gamma) < 1e-12:
                    continue
                if mode == "linear":
                    weights = fuse_linear(source_weights, prior, gamma)
                else:
                    weights = fuse_power(source_weights, prior, gamma)
                tag = f"{proxy_name}_{mode}_{gamma_tag(gamma)}"
                model_path = output_dir / f"logic_timereise_{tag}.onnx"
                if not model_path.exists():
                    make_variant(base_model, output_dir, tag, weights, hard=False)
                manifest.append({"tag": tag, "model": str(model_path), "branch": "phys_proxy", "proxy_name": proxy_name, "mode": mode, "gamma": gamma})

    (output_dir / "phys_proxy_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Prepared {len(manifest)} physical-proxy variants", flush=True)
    score_and_package_manifest(manifest, output_dir, args, copy_prefix=args.copy_prefix)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
