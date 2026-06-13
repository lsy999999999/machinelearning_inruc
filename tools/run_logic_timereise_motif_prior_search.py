from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.run_logic_timereise_power_ensemble_search import extract_timereise_weights, normalize_weights
from tools.run_logic_timereise_search import expand_bins, make_variant, normalize_importance, time_blocks
from tools.timereise_branch_utils import load_base_model
from tools.timereise_teacher_utils import score_and_package_teacher_manifest


def load_windows(data_dir: str | Path, split: str) -> np.ndarray:
    return np.load(Path(data_dir) / f"{split}_windows.npy").astype(np.float32)


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


def feature_map(windows: np.ndarray, feature: str) -> np.ndarray:
    if feature == "raw":
        return windows.astype(np.float32)
    if feature == "abs":
        return np.abs(windows).astype(np.float32)
    if feature == "diff":
        diff = np.zeros_like(windows, dtype=np.float32)
        diff[:, :, 1:] = np.abs(windows[:, :, 1:] - windows[:, :, :-1])
        return diff
    if feature == "absdiff":
        diff = np.zeros_like(windows, dtype=np.float32)
        diff[:, :, 1:] = np.abs(windows[:, :, 1:] - windows[:, :, :-1])
        return (0.65 * np.abs(windows) + 0.35 * diff).astype(np.float32)
    raise ValueError(f"Unknown feature: {feature}")


def l2_normalize_segments(segments: np.ndarray, center: bool) -> np.ndarray:
    values = segments.astype(np.float64, copy=False)
    if center:
        values = values - values.mean(axis=1, keepdims=True)
    denom = np.sqrt(np.sum(values * values, axis=1, keepdims=True)) + 1e-8
    return values / denom


def class_thresholds(confidence: np.ndarray, pred: np.ndarray, quantile: float) -> np.ndarray:
    thresholds = np.zeros((9,), dtype=np.float32)
    for class_id in range(9):
        values = confidence[pred == class_id]
        if len(values) == 0:
            thresholds[class_id] = 1.0
        else:
            thresholds[class_id] = float(np.quantile(values, quantile))
    return thresholds


def motif_prior(
    windows: np.ndarray,
    probabilities: np.ndarray,
    *,
    num_bins: int,
    feature: str,
    quantile: float,
    score_mode: str,
) -> np.ndarray:
    pred = probabilities.argmax(axis=1).astype(np.int64)
    confidence = probabilities[np.arange(len(probabilities)), pred]
    thresholds = class_thresholds(confidence, pred, quantile)
    features = feature_map(windows, feature)
    abs_x = np.abs(windows).astype(np.float32)
    blocks = time_blocks(num_bins)
    scores = np.ones((9, 8, num_bins), dtype=np.float64)

    center = feature == "raw"
    for class_id in range(9):
        mask = (pred == class_id) & (confidence >= thresholds[class_id])
        if mask.sum() < 8:
            mask = pred == class_id
        if mask.sum() == 0:
            continue
        feat_c = features[mask]
        abs_c = abs_x[mask]
        for channel in range(8):
            for bin_id, (start, end) in enumerate(blocks):
                segments = feat_c[:, channel, start:end]
                normalized = l2_normalize_segments(segments, center=center)
                motif = normalized.mean(axis=0, keepdims=True)
                motif = motif / (np.sqrt(np.sum(motif * motif)) + 1e-8)
                coherence = np.maximum((normalized * motif).sum(axis=1), 0.0)
                energy = abs_c[:, channel, start:end].mean(axis=1)
                energy = energy / (energy.mean() + 1e-8)
                if score_mode == "coherence":
                    score = coherence.mean()
                elif score_mode == "energy":
                    score = energy.mean()
                elif score_mode == "coh_energy":
                    score = (coherence * energy).mean()
                elif score_mode == "coh_peak":
                    peak = abs_c[:, channel, start:end].max(axis=1)
                    peak = peak / (peak.mean() + 1e-8)
                    score = (coherence * peak).mean()
                else:
                    raise ValueError(f"Unknown score mode: {score_mode}")
                scores[class_id, channel, bin_id] = float(score)
    return normalize_importance(scores)


def fuse_power(source: np.ndarray, prior: np.ndarray, gamma: float) -> np.ndarray:
    fused = source * np.power(np.maximum(prior, 0.05), gamma)
    return normalize_weights(fused)


def fuse_linear(source: np.ndarray, prior: np.ndarray, gamma: float) -> np.ndarray:
    fused = source * np.clip(1.0 + gamma * (prior - 1.0), 0.03, None)
    return normalize_weights(fused)


def write_variant(base_model: onnx.ModelProto, output_dir: Path, tag: str, weights: np.ndarray) -> Path:
    model_path = output_dir / f"logic_timereise_{tag}.onnx"
    if model_path.exists():
        try:
            onnx.checker.check_model(onnx.load(model_path))
            return model_path
        except Exception:
            model_path.unlink()
    return make_variant(base_model, output_dir, tag, weights, hard=False)


def gamma_tag(value: float) -> str:
    sign = "p" if value >= 0 else "m"
    return f"{sign}{int(round(abs(value) * 1000)):03d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search folded motif/shapelet priors for TimeREISE weights.")
    parser.add_argument("--base-model", default="runs/candidates/logic_timereise_multiscale_logit_rowcoord_bestproxy.onnx")
    parser.add_argument("--source-model", default="runs/candidates/logic_timereise_multiscale_logit_rowcoord_bestproxy.onnx")
    parser.add_argument("--output-dir", default="runs/logic_timereise_motif_prior_search")
    parser.add_argument("--data-dir", default="prepared_hf_val5k")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--num-bins", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--feature", choices=["raw", "abs", "diff", "absdiff"], nargs="+", default=["raw", "abs", "diff", "absdiff"])
    parser.add_argument("--quantile", type=float, nargs="+", default=[0.70, 0.80, 0.90])
    parser.add_argument("--score-mode", choices=["coherence", "energy", "coh_energy", "coh_peak"], nargs="+", default=["coherence", "coh_energy", "coh_peak"])
    parser.add_argument("--mode", choices=["power", "linear"], nargs="+", default=["power"])
    parser.add_argument("--gamma", type=float, nargs="+", default=[0.02, 0.05, 0.10, -0.02])
    parser.add_argument("--copy-prefix", default="logic_timereise_motif_prior")
    parser.add_argument("--no-package", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    windows = load_windows(args.data_dir, args.eval_split)
    probabilities = run_probabilities(args.source_model, windows, args.eval_batch_size)
    source_weights = extract_timereise_weights(args.source_model)
    base_model = load_base_model(args.base_model)

    manifest = [{"tag": "start", "model": str(write_variant(base_model, output_dir, "start", source_weights)), "branch": "motif_prior"}]
    cache_dir = output_dir / "priors"
    cache_dir.mkdir(exist_ok=True)
    for num_bins in args.num_bins:
        blocks = np.asarray(time_blocks(num_bins), dtype=np.int64)
        for feature in args.feature:
            for quantile in args.quantile:
                for score_mode in args.score_mode:
                    prior_tag = f"b{num_bins}_{feature}_q{int(quantile * 100):02d}_{score_mode}"
                    cache_path = cache_dir / f"{prior_tag}.npy"
                    if cache_path.exists():
                        binned = np.load(cache_path).astype(np.float32)
                    else:
                        binned = motif_prior(
                            windows,
                            probabilities,
                            num_bins=num_bins,
                            feature=feature,
                            quantile=quantile,
                            score_mode=score_mode,
                        )
                        np.save(cache_path, binned)
                    expanded = expand_bins(binned, blocks)
                    expanded = expanded / np.maximum(expanded.mean(axis=(1, 2), keepdims=True), 1e-6)
                    for mode in args.mode:
                        for gamma in args.gamma:
                            if mode == "power":
                                weights = fuse_power(source_weights, expanded, gamma)
                            else:
                                weights = fuse_linear(source_weights, expanded, gamma)
                            tag = f"{prior_tag}_{mode}_{gamma_tag(gamma)}"
                            model_path = write_variant(base_model, output_dir, tag, weights)
                            manifest.append({"tag": tag, "model": str(model_path), "branch": "motif_prior"})

    (output_dir / "motif_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Prepared {len(manifest)} motif-prior variants", flush=True)
    score_and_package_teacher_manifest(manifest, output_dir, args, copy_prefix=args.copy_prefix)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
