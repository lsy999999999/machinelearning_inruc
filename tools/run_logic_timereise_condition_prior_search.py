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
from tools.run_logic_timereise_search import make_variant
from tools.run_logic_timereise_periodic_impulse_search import proxy_maps, smooth_time
from tools.timereise_branch_utils import load_base_model
from tools.timereise_teacher_utils import score_and_package_teacher_manifest


def load_split(data_dir: str | Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    data_dir = Path(data_dir)
    windows = np.load(data_dir / f"{split}_windows.npy").astype(np.float32)
    labels = np.load(data_dir / f"{split}_labels.npy").astype(np.int64)
    return windows, labels


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


def normalize_prior(values: np.ndarray) -> np.ndarray:
    values = np.maximum(values.astype(np.float64, copy=False), 0.0)
    means = values.mean(axis=(1, 2), keepdims=True)
    out = values / np.maximum(means, 1e-8)
    return np.clip(out, 0.05, 10.0).astype(np.float32)


def quantile_bins(values: np.ndarray, bins: int) -> np.ndarray:
    if bins <= 1:
        return np.zeros_like(values, dtype=np.int64)
    edges = np.quantile(values.astype(np.float64), np.linspace(0.0, 1.0, bins + 1)[1:-1])
    return np.digitize(values, edges).astype(np.int64)


def condition_features(windows: np.ndarray, lags: np.ndarray) -> dict[str, np.ndarray]:
    abs_x = np.abs(windows).astype(np.float32)
    rms = np.sqrt(np.mean(windows * windows, axis=(1, 2)) + 1e-8)
    crest = abs_x.max(axis=(1, 2)) / np.maximum(rms, 1e-6)
    torque_rms = np.sqrt(np.mean(windows[:, 4:5, :] * windows[:, 4:5, :], axis=(1, 2)) + 1e-8)
    rgb_rms = np.sqrt(np.mean(windows[:, 1:4, :] * windows[:, 1:4, :], axis=(1, 2)) + 1e-8)
    pgb_rms = np.sqrt(np.mean(windows[:, 5:8, :] * windows[:, 5:8, :], axis=(1, 2)) + 1e-8)
    spectrum = np.abs(np.fft.rfft(windows, axis=2))
    spectrum[:, :, 0] = 0.0
    dominant = spectrum.mean(axis=1).argmax(axis=1).astype(np.float32)
    return {
        "rms": rms.astype(np.float32),
        "crest": crest.astype(np.float32),
        "torque": torque_rms.astype(np.float32),
        "rgb": rgb_rms.astype(np.float32),
        "pgb": pgb_rms.astype(np.float32),
        "dom": dominant,
        "lag": lags.astype(np.float32),
    }


def make_condition_ids(features: dict[str, np.ndarray], spec: str, bins: int) -> np.ndarray:
    parts = [part for part in spec.split("+") if part]
    if not parts:
        raise ValueError("condition spec cannot be empty")
    ids = np.zeros_like(next(iter(features.values())), dtype=np.int64)
    multiplier = 1
    for part in parts:
        if part not in features:
            raise ValueError(f"Unknown condition feature: {part}")
        bin_id = quantile_bins(features[part], bins)
        ids += multiplier * bin_id
        multiplier *= bins
    return ids.astype(np.int64)


def aggregate_group_maps(group_maps: list[np.ndarray], aggregate: str) -> np.ndarray:
    stack = np.stack(group_maps, axis=0)
    if aggregate == "mean":
        return stack.mean(axis=0)
    if aggregate == "median":
        return np.median(stack, axis=0)
    if aggregate == "trimmed_mean":
        if stack.shape[0] <= 2:
            return stack.mean(axis=0)
        ordered = np.sort(stack, axis=0)
        return ordered[1:-1].mean(axis=0)
    raise ValueError(f"Unknown aggregate: {aggregate}")


def class_pred_fallback(proxy: np.ndarray, pred: np.ndarray, conf: np.ndarray) -> np.ndarray:
    out = np.zeros((9, proxy.shape[1], proxy.shape[2]), dtype=np.float64)
    for class_id in range(9):
        mask = pred == class_id
        if not np.any(mask):
            out[class_id] = 1.0
            continue
        weights = conf[mask].astype(np.float64)
        out[class_id] = (proxy[mask] * weights[:, None, None]).sum(axis=0) / max(float(weights.sum()), 1e-12)
    return normalize_prior(smooth_time(out.astype(np.float32), passes=1))


def condition_prior(
    proxy: np.ndarray,
    pred: np.ndarray,
    conf: np.ndarray,
    labels: np.ndarray,
    condition_ids: np.ndarray,
    *,
    aggregate: str,
    sample_mode: str,
    min_samples: int,
    normalize_groups: bool,
    fallback: np.ndarray,
) -> np.ndarray:
    out = np.zeros((9, proxy.shape[1], proxy.shape[2]), dtype=np.float64)
    unique_groups = np.unique(condition_ids)
    for class_id in range(9):
        rows: list[np.ndarray] = []
        for group_id in unique_groups:
            mask = (condition_ids == group_id) & (pred == class_id)
            if sample_mode == "correct":
                mask &= labels == class_id
            elif sample_mode == "highconf":
                class_conf = conf[pred == class_id]
                if len(class_conf) > 0:
                    threshold = float(np.quantile(class_conf, 0.65))
                    mask &= conf >= threshold
            elif sample_mode != "pred":
                raise ValueError(f"Unknown sample mode: {sample_mode}")
            if int(mask.sum()) < min_samples:
                continue
            weights = conf[mask].astype(np.float64)
            row = (proxy[mask] * weights[:, None, None]).sum(axis=0) / max(float(weights.sum()), 1e-12)
            if normalize_groups:
                row = normalize_prior(row[None, :, :])[0]
            rows.append(row)
        if rows:
            out[class_id] = aggregate_group_maps(rows, aggregate)
        else:
            out[class_id] = fallback[class_id]
    return normalize_prior(smooth_time(out.astype(np.float32), passes=1))


def fuse_weights(source: np.ndarray, prior: np.ndarray, gamma: float, mode: str, classes: list[int]) -> np.ndarray:
    out = source.astype(np.float64, copy=True)
    class_set = set(classes)
    for class_id in range(source.shape[0]):
        if class_id not in class_set:
            continue
        if mode == "power":
            row = source[class_id] * np.power(np.maximum(prior[class_id], 0.05), gamma)
        elif mode == "linear":
            row = source[class_id] * np.clip(1.0 + gamma * (prior[class_id] - 1.0), 0.03, None)
        else:
            raise ValueError(f"Unknown fuse mode: {mode}")
        out[class_id] = row / np.maximum(row.mean(), 1e-8)
    return normalize_weights(out)


def parse_classes(value: str) -> list[int]:
    if value.strip().lower() in {"all", "*"}:
        return list(range(9))
    return [int(part) for part in value.split(",") if part.strip()]


def gamma_tag(value: float) -> str:
    sign = "p" if value >= 0 else "m"
    return f"{sign}{int(round(abs(value) * 10000)):04d}"


def safe_write_variant(base_model: onnx.ModelProto, output_dir: Path, tag: str, weights: np.ndarray) -> Path:
    model_path = output_dir / f"logic_timereise_{tag}.onnx"
    if model_path.exists():
        try:
            onnx.checker.check_model(onnx.load(model_path))
            return model_path
        except Exception:
            model_path.unlink()
    return make_variant(base_model, output_dir, tag, weights, hard=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Condition-aware robust aggregation search for folded TimeREISE weights.")
    parser.add_argument("--base-model", default="runs/logic_timereise_final_micro_morph/logic_timereise_start.onnx")
    parser.add_argument("--source-model", default="runs/logic_timereise_final_micro_morph/logic_timereise_start.onnx")
    parser.add_argument("--output-dir", default="runs/logic_timereise_condition_prior_search")
    parser.add_argument("--data-dir", default="prepared_hf_val5k")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--min-lag", type=int, default=3)
    parser.add_argument("--max-lag", type=int, default=30)
    parser.add_argument("--condition", nargs="+", default=["rms+lag", "crest+lag", "torque+lag", "rms+crest", "pgb+lag"])
    parser.add_argument("--condition-bins", type=int, default=3)
    parser.add_argument("--prior", nargs="+", default=["impulse", "periodic_impulse", "hybrid"])
    parser.add_argument("--aggregate", choices=["mean", "median", "trimmed_mean"], nargs="+", default=["median", "trimmed_mean"])
    parser.add_argument("--sample-mode", choices=["pred", "highconf", "correct"], nargs="+", default=["pred"])
    parser.add_argument("--mode", choices=["power", "linear"], nargs="+", default=["power"])
    parser.add_argument("--gamma", type=float, nargs="+", default=[-0.001, 0.001, 0.002, 0.003, 0.005])
    parser.add_argument("--classes", type=parse_classes, default=parse_classes("all"))
    parser.add_argument("--min-samples", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--copy-prefix", default="logic_timereise_condition_prior")
    parser.add_argument("--no-package", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "priors"
    cache_dir.mkdir(exist_ok=True)

    windows, labels = load_split(args.data_dir, args.eval_split)
    probabilities = run_probabilities(args.source_model, windows, args.eval_batch_size)
    pred = probabilities.argmax(axis=1).astype(np.int64)
    conf = probabilities[np.arange(len(probabilities)), pred].astype(np.float32)
    maps, lags, periodicity = proxy_maps(windows, args.min_lag, args.max_lag)
    features = condition_features(windows, lags)

    (output_dir / "condition_feature_summary.json").write_text(
        json.dumps(
            {
                name: {
                    "min": float(np.min(values)),
                    "q33": float(np.quantile(values, 1.0 / 3.0)),
                    "q67": float(np.quantile(values, 2.0 / 3.0)),
                    "max": float(np.max(values)),
                }
                for name, values in {**features, "periodicity": periodicity.astype(np.float32)}.items()
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    source_weights = extract_timereise_weights(args.source_model)
    base_model = load_base_model(args.base_model)

    manifest: list[dict[str, object]] = [
        {
            "tag": "condition_start",
            "model": str(safe_write_variant(base_model, output_dir, "condition_start", source_weights)),
            "branch": "condition_prior",
        }
    ]
    priors_written: dict[str, Path] = {}
    stop = False
    for prior_name in args.prior:
        if prior_name not in maps:
            raise ValueError(f"Unknown prior map: {prior_name}")
        fallback = class_pred_fallback(maps[prior_name], pred, conf)
        for condition in args.condition:
            condition_ids = make_condition_ids(features, condition, args.condition_bins)
            condition_tag = condition.replace("+", "")
            for aggregate in args.aggregate:
                for sample_mode in args.sample_mode:
                    for normalize_groups in (True, False):
                        norm_tag = "ng" if normalize_groups else "raw"
                        prior_tag = f"{prior_name}_{condition_tag}_{aggregate}_{sample_mode}_{norm_tag}"
                        prior_path = cache_dir / f"{prior_tag}.npy"
                        if prior_path.exists():
                            prior = np.load(prior_path).astype(np.float32)
                        else:
                            prior = condition_prior(
                                maps[prior_name],
                                pred,
                                conf,
                                labels,
                                condition_ids,
                                aggregate=aggregate,
                                sample_mode=sample_mode,
                                min_samples=args.min_samples,
                                normalize_groups=normalize_groups,
                                fallback=fallback,
                            )
                            np.save(prior_path, prior)
                        priors_written[prior_tag] = prior_path
                        for mode in args.mode:
                            for gamma in args.gamma:
                                if abs(gamma) < 1e-12:
                                    continue
                                tag = f"{prior_tag}_{mode}_{gamma_tag(gamma)}"
                                weights = fuse_weights(source_weights, prior, gamma, mode, args.classes)
                                model_path = safe_write_variant(base_model, output_dir, tag, weights)
                                manifest.append(
                                    {
                                        "tag": tag,
                                        "model": str(model_path),
                                        "branch": "condition_prior",
                                        "prior": prior_name,
                                        "condition": condition,
                                        "aggregate": aggregate,
                                        "sample_mode": sample_mode,
                                        "normalize_groups": normalize_groups,
                                        "mode": mode,
                                        "gamma": gamma,
                                    }
                                )
                                if args.limit is not None and len(manifest) >= args.limit:
                                    stop = True
                                    break
                            if stop:
                                break
                        if stop:
                            break
                    if stop:
                        break
                if stop:
                    break
            if stop:
                break
        if stop:
            break

    (output_dir / "condition_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "prior_cache_manifest.json").write_text(
        json.dumps({key: str(value) for key, value in sorted(priors_written.items())}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Prepared {len(manifest)} condition-prior variants", flush=True)
    score_and_package_teacher_manifest(manifest, output_dir, args, copy_prefix=args.copy_prefix)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
