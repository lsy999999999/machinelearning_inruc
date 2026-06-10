from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.run_logic_timereise_power_ensemble_search import extract_timereise_weights, normalize_weights
from tools.run_logic_timereise_search import make_variant, package_model, report_scores, run_eval_subprocess
from tools.timereise_branch_utils import load_base_model


TARGET_CLASSES = (6, 7, 8)


def parse_classes(value: str) -> list[int]:
    if value.strip().lower() in {"all", "*"}:
        return list(range(9))
    return [int(part) for part in value.split(",") if part.strip()]


def gamma_tag(value: float) -> str:
    sign = "p" if value >= 0 else "m"
    return f"{sign}{int(round(abs(value) * 10000)):04d}"


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


def local_mean(values: np.ndarray, kernel: int) -> np.ndarray:
    pad = kernel // 2
    padded = np.pad(values, ((0, 0), (0, 0), (pad, pad)), mode="edge")
    out = np.zeros_like(values, dtype=np.float32)
    for offset in range(kernel):
        out += padded[:, :, offset : offset + values.shape[2]]
    return out / float(kernel)


def smooth_time(values: np.ndarray, passes: int = 1) -> np.ndarray:
    out = values.astype(np.float32, copy=True)
    for _ in range(passes):
        padded = np.pad(out, ((0, 0), (0, 0), (1, 1)), mode="edge")
        out = 0.25 * padded[:, :, :-2] + 0.5 * padded[:, :, 1:-1] + 0.25 * padded[:, :, 2:]
    return out


def normalize_prior(values: np.ndarray) -> np.ndarray:
    values = np.maximum(values.astype(np.float64, copy=False), 0.0)
    out = np.ones_like(values, dtype=np.float64)
    means = values.mean(axis=(1, 2), keepdims=True)
    valid = means[:, 0, 0] > 1e-12
    if np.any(valid):
        out[valid] = values[valid] / np.maximum(means[valid], 1e-12)
    return np.clip(out, 0.05, 10.0).astype(np.float32)


def combine_priors(parts: list[tuple[np.ndarray, float]]) -> np.ndarray:
    total = np.zeros_like(parts[0][0], dtype=np.float64)
    weight_sum = 0.0
    for prior, weight in parts:
        total += float(weight) * normalize_prior(prior)
        weight_sum += float(weight)
    return normalize_prior(total / max(weight_sum, 1e-12))


def teager_energy(windows: np.ndarray) -> np.ndarray:
    out = np.zeros_like(windows, dtype=np.float32)
    out[:, :, 1:-1] = np.abs(windows[:, :, 1:-1] ** 2 - windows[:, :, :-2] * windows[:, :, 2:])
    out[:, :, 0] = np.abs(windows[:, :, 0])
    out[:, :, -1] = np.abs(windows[:, :, -1])
    return out


def lag_autocorr_proxy(windows: np.ndarray, max_lag: int = 12) -> np.ndarray:
    centered = windows - windows.mean(axis=2, keepdims=True)
    denom = np.mean(centered * centered, axis=2, keepdims=True) + 1e-6
    best = np.zeros_like(windows, dtype=np.float32)
    for lag in range(2, max_lag + 1):
        corr = np.zeros_like(windows, dtype=np.float32)
        prod = centered[:, :, lag:] * centered[:, :, :-lag]
        corr[:, :, lag:] = np.maximum(prod / denom, 0.0)
        best = np.maximum(best, corr)
    return best


def estimate_lags(abs_x: np.ndarray, min_lag: int, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    envelope = local_mean(abs_x, kernel=5)
    centered = envelope - envelope.mean(axis=2, keepdims=True)
    denom = np.mean(centered * centered, axis=2) + 1e-6
    scores: list[np.ndarray] = []
    lags = np.arange(min_lag, max_lag + 1, dtype=np.int64)
    for lag in lags:
        prod = centered[:, :, lag:] * centered[:, :, :-lag]
        corr = np.mean(prod, axis=2) / denom
        scores.append(np.maximum(corr, 0.0).mean(axis=1))
    score_matrix = np.stack(scores, axis=1)
    best = score_matrix.argmax(axis=1)
    return lags[best], score_matrix[np.arange(score_matrix.shape[0]), best].astype(np.float32)


def periodic_comb_proxy(abs_x: np.ndarray, lags: np.ndarray, periodicity: np.ndarray, carrier: np.ndarray) -> np.ndarray:
    envelope = local_mean(abs_x, kernel=5)
    out = np.zeros_like(abs_x, dtype=np.float32)
    time = np.arange(abs_x.shape[2], dtype=np.float32)
    for index, lag_value in enumerate(lags.tolist()):
        lag = max(int(lag_value), 2)
        offset_window = envelope[index].mean(axis=0)[:lag]
        offset = int(offset_window.argmax()) if offset_window.size else 0
        phase = np.mod(time - float(offset), float(lag))
        distance = np.minimum(phase, float(lag) - phase)
        sigma = max(1.0, 0.16 * float(lag))
        comb = np.exp(-0.5 * (distance / sigma) ** 2).astype(np.float32)
        scale = 0.5 + 2.0 * float(np.clip(periodicity[index], 0.0, 1.0))
        out[index] = carrier[index] * comb[None, :] * scale
    return out


def proxy_maps(windows: np.ndarray, min_lag: int, max_lag: int) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    abs_x = np.abs(windows).astype(np.float32)
    diff = np.zeros_like(abs_x, dtype=np.float32)
    diff[:, :, 1:] = np.abs(windows[:, :, 1:] - windows[:, :, :-1])
    diff_energy = diff * diff

    peak = np.maximum(abs_x - local_mean(abs_x, kernel=5), 0.0)
    envelope = local_mean(abs_x, kernel=9)
    teager = teager_energy(windows)

    l2 = np.sqrt(np.mean(windows * windows, axis=2, keepdims=True) + 1e-6)
    l4 = np.mean(np.abs(windows) ** 4, axis=2, keepdims=True) ** 0.25
    l2l4_scale = l4 / np.maximum(l2, 1e-6)
    l2l4_scale = l2l4_scale / np.maximum(l2l4_scale.mean(axis=1, keepdims=True), 1e-6)
    l2l4_abs = abs_x * np.clip(l2l4_scale, 0.5, 3.0)

    impulse = 0.40 * teager + 0.30 * diff_energy + 0.20 * peak + 0.10 * abs_x
    lags, periodicity = estimate_lags(abs_x, min_lag=min_lag, max_lag=max_lag)
    periodic_env = periodic_comb_proxy(abs_x, lags, periodicity, envelope)
    periodic_impulse = periodic_comb_proxy(abs_x, lags, periodicity, impulse)
    cyclo_light = 0.45 * periodic_env + 0.35 * lag_autocorr_proxy(windows, max_lag=12) + 0.20 * envelope
    hybrid = 0.40 * periodic_impulse + 0.35 * impulse + 0.25 * l2l4_abs

    maps = {
        "teager": teager,
        "impulse": impulse,
        "l2l4": l2l4_abs,
        "periodic_env": periodic_env,
        "periodic_impulse": periodic_impulse,
        "cyclo_light": cyclo_light,
        "hybrid": hybrid,
    }
    return {name: smooth_time(value, passes=1) for name, value in maps.items()}, lags, periodicity


def aggregate_weighted(proxy: np.ndarray, weights_by_class: list[np.ndarray], fallback: np.ndarray) -> np.ndarray:
    out = np.zeros((9, proxy.shape[1], proxy.shape[2]), dtype=np.float64)
    for class_id, sample_weights in enumerate(weights_by_class):
        weights = np.maximum(sample_weights.astype(np.float64, copy=False), 0.0)
        total = float(weights.sum())
        if total <= 1e-12:
            out[class_id] = fallback[class_id]
            continue
        out[class_id] = (proxy * weights[:, None, None]).sum(axis=0) / total
    return normalize_prior(smooth_time(out.astype(np.float32), passes=1))


def pred_weights(probabilities: np.ndarray, pred: np.ndarray) -> list[np.ndarray]:
    conf = probabilities[np.arange(len(probabilities)), pred]
    return [np.where(pred == class_id, conf, 0.0).astype(np.float32) for class_id in range(9)]


def high_conf_weights(
    probabilities: np.ndarray,
    pred: np.ndarray,
    labels: np.ndarray,
    quantile: float,
    min_samples: int,
) -> list[np.ndarray]:
    conf = probabilities[np.arange(len(probabilities)), pred]
    rows: list[np.ndarray] = []
    for class_id in range(9):
        mask = (pred == class_id) & (labels == class_id)
        if int(mask.sum()) >= min_samples:
            threshold = float(np.quantile(conf[mask], quantile))
            mask = mask & (conf >= threshold)
        weights = np.where(mask, conf * conf, 0.0).astype(np.float32)
        rows.append(weights)
    return rows


def boundary_weights(
    probabilities: np.ndarray,
    pred: np.ndarray,
    labels: np.ndarray,
    target_classes: list[int],
    margin_threshold: float,
) -> list[np.ndarray]:
    order = np.argsort(probabilities, axis=1)
    top2 = order[:, -2:]
    top1_prob = probabilities[np.arange(len(probabilities)), order[:, -1]]
    top2_prob = probabilities[np.arange(len(probabilities)), order[:, -2]]
    margin = top1_prob - top2_prob
    wrong = pred != labels
    boundary_strength = np.clip((margin_threshold - margin) / max(margin_threshold, 1e-6), 0.0, 1.0)

    rows: list[np.ndarray] = []
    target_set = set(target_classes)
    for class_id in range(9):
        if class_id not in target_set:
            rows.append(np.zeros((len(pred),), dtype=np.float32))
            continue
        involved = (pred == class_id) | (labels == class_id) | np.any(top2 == class_id, axis=1)
        mask = involved & (wrong | (margin <= margin_threshold))
        weights = probabilities[:, class_id] * (1.0 + wrong.astype(np.float32) + boundary_strength)
        rows.append(np.where(mask, weights, 0.0).astype(np.float32))
    return rows


def guided_weights(
    pred_rows: list[np.ndarray],
    high_rows: list[np.ndarray],
    boundary_rows: list[np.ndarray],
    target_classes: list[int],
) -> list[np.ndarray]:
    target_set = set(target_classes)
    rows: list[np.ndarray] = []
    for class_id in range(9):
        if class_id in target_set and float(boundary_rows[class_id].sum()) > 1e-12:
            rows.append((0.55 * high_rows[class_id] + 0.45 * boundary_rows[class_id]).astype(np.float32))
        elif float(high_rows[class_id].sum()) > 1e-12:
            rows.append(high_rows[class_id])
        else:
            rows.append(pred_rows[class_id])
    return rows


def pseudo_condition_ids(windows: np.ndarray, lags: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(windows * windows, axis=(1, 2)))
    lag_values = lags.astype(np.float32)
    rms_edges = np.quantile(rms, [1.0 / 3.0, 2.0 / 3.0])
    lag_edges = np.quantile(lag_values, [1.0 / 3.0, 2.0 / 3.0])
    rms_bin = np.digitize(rms, rms_edges)
    lag_bin = np.digitize(lag_values, lag_edges)
    return (3 * rms_bin + lag_bin).astype(np.int64)


def robust_condition_prior(
    proxy: np.ndarray,
    pred: np.ndarray,
    probabilities: np.ndarray,
    condition_ids: np.ndarray,
    fallback: np.ndarray,
    min_samples: int,
) -> np.ndarray:
    conf = probabilities[np.arange(len(probabilities)), pred]
    out = np.zeros((9, proxy.shape[1], proxy.shape[2]), dtype=np.float64)
    for class_id in range(9):
        rows: list[np.ndarray] = []
        for group_id in np.unique(condition_ids):
            mask = (condition_ids == group_id) & (pred == class_id)
            if int(mask.sum()) < min_samples:
                continue
            weights = conf[mask].astype(np.float64)
            rows.append((proxy[mask] * weights[:, None, None]).sum(axis=0) / max(float(weights.sum()), 1e-12))
        if rows:
            out[class_id] = np.median(np.stack(rows, axis=0), axis=0)
        else:
            out[class_id] = fallback[class_id]
    return normalize_prior(smooth_time(out.astype(np.float32), passes=1))


def build_priors(
    windows: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    pred = probabilities.argmax(axis=1).astype(np.int64)
    maps, lags, _periodicity = proxy_maps(windows, args.min_lag, args.max_lag)
    pred_rows = pred_weights(probabilities, pred)
    high_rows = high_conf_weights(probabilities, pred, labels, args.high_conf_quantile, args.min_samples)
    boundary_rows = boundary_weights(probabilities, pred, labels, args.target_classes, args.boundary_margin)
    guided_rows = guided_weights(pred_rows, high_rows, boundary_rows, args.target_classes)
    condition_ids = pseudo_condition_ids(windows, lags)

    priors: dict[str, np.ndarray] = {}
    pred_fallbacks: dict[str, np.ndarray] = {}
    for name, proxy in maps.items():
        pred_prior = aggregate_weighted(proxy, pred_rows, normalize_prior(np.ones((9, 8, 100), dtype=np.float32)))
        pred_fallbacks[name] = pred_prior

    for name in ("teager", "impulse", "l2l4", "periodic_impulse", "cyclo_light", "hybrid"):
        proxy = maps[name]
        fallback = pred_fallbacks[name]
        priors[f"{name}_guided"] = aggregate_weighted(proxy, guided_rows, fallback)

    for name in ("periodic_env", "periodic_impulse", "hybrid"):
        proxy = maps[name]
        priors[f"{name}_pred"] = pred_fallbacks[name]
        priors[f"{name}_boundary"] = aggregate_weighted(proxy, boundary_rows, pred_fallbacks[name])

    for name in ("impulse", "periodic_impulse", "hybrid"):
        priors[f"{name}_robust"] = robust_condition_prior(
            maps[name],
            pred,
            probabilities,
            condition_ids,
            pred_fallbacks[name],
            args.min_samples,
        )

    priors["paper_hybrid_guided"] = combine_priors(
        [
            (priors["periodic_impulse_guided"], 0.40),
            (priors["impulse_guided"], 0.35),
            (priors["l2l4_guided"], 0.25),
        ]
    )
    priors["paper_hybrid_robust"] = combine_priors(
        [
            (priors["periodic_impulse_robust"], 0.45),
            (priors["impulse_robust"], 0.35),
            (priors["cyclo_light_guided"], 0.20),
        ]
    )
    return priors


def fuse_weights(
    source: np.ndarray,
    prior: np.ndarray,
    gamma: float,
    mode: str,
    classes: list[int],
) -> np.ndarray:
    out = source.astype(np.float64, copy=True)
    class_set = set(classes)
    for class_id in range(source.shape[0]):
        if class_id not in class_set:
            continue
        if mode == "linear":
            row = source[class_id] * np.clip(1.0 + gamma * (prior[class_id] - 1.0), 0.03, None)
        elif mode == "power":
            row = source[class_id] * np.power(np.maximum(prior[class_id], 0.05), gamma)
        else:
            raise ValueError(f"Unknown fuse mode: {mode}")
        row = row / np.maximum(row.mean(), 1e-6)
        out[class_id] = np.clip(row, 0.03, 20.0)
    return normalize_weights(out)


def evaluate_model(model_path: Path, report_path: Path, args: argparse.Namespace) -> dict[str, float]:
    if not report_path.exists():
        run_eval_subprocess(str(model_path), report_path, args)
    faith, deletion, insertion, f1, simplicity, proxy = report_scores(report_path)
    teacher_score = 0.6 * f1 + 0.3 * faith + 0.1 * simplicity
    return {
        "faith": faith,
        "deletion": deletion,
        "insertion": insertion,
        "f1": f1,
        "simplicity": simplicity,
        "proxy": proxy,
        "teacher_score": teacher_score,
    }


def score_and_package_teacher_manifest(
    manifest: list[dict[str, object]],
    output_dir: Path,
    args: argparse.Namespace,
    copy_prefix: str,
) -> list[dict[str, object]]:
    eval_dir = output_dir / "eval"
    candidate_dir = Path("runs/candidates")
    final_dir = Path("runs/final_candidates")
    eval_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    rows: list[dict[str, object]] = []
    for index, row in enumerate(manifest, 1):
        tag = str(row["tag"])
        report_path = eval_dir / f"{tag}.json"
        metrics = evaluate_model(Path(str(row["model"])), report_path, args)
        scored = {**row, "report": str(report_path), **metrics}
        rows.append(scored)
        print(
            f"{index}/{len(manifest)} {tag} faith={metrics['faith']:.6f} "
            f"f1={metrics['f1']:.6f} simp={metrics['simplicity']:.6f} "
            f"teacher={metrics['teacher_score']:.9f}",
            flush=True,
        )

    rows_by_teacher = sorted(rows, key=lambda item: float(item["teacher_score"]), reverse=True)
    rows_by_faith = sorted(rows, key=lambda item: float(item["faith"]), reverse=True)
    rows_by_proxy = sorted(rows, key=lambda item: float(item["proxy"]), reverse=True)
    (output_dir / "summary_top.json").write_text(
        json.dumps(
            {
                "best_teacher": rows_by_teacher[:20],
                "best_faith": rows_by_faith[:20],
                "best_proxy": rows_by_proxy[:20],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    for kind, row in (("bestteacher", rows_by_teacher[0]), ("bestfaith", rows_by_faith[0])):
        model_dst = candidate_dir / f"{copy_prefix}_{kind}.onnx"
        report_dst = candidate_dir / f"{copy_prefix}_{kind}_devkit.json"
        shutil.copy2(str(row["model"]), model_dst)
        shutil.copy2(str(row["report"]), report_dst)
        print(f"Copied {kind}: {model_dst}", flush=True)
        if not args.no_package:
            package_dst = final_dir / f"{copy_prefix}_{kind}_submission.zip"
            package_model(str(model_dst), package_dst, args)
            print(f"Packaged {kind}: {package_dst}", flush=True)

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search periodic-comb and classifier-guided impulse TimeREISE weights.")
    parser.add_argument("--base-model", default="", help="Classification graph to preserve; defaults to --source-model.")
    parser.add_argument("--source-model", default="runs/candidates/logic_timereise_weight_morph_long_rowcoord_full_bestproxy.onnx")
    parser.add_argument("--output-dir", default="runs/logic_timereise_periodic_impulse_search")
    parser.add_argument("--data-dir", default="prepared_hf_val5k")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--target-classes", type=parse_classes, default=parse_classes("6,7,8"))
    parser.add_argument("--row-scope", choices=["target", "all", "both"], default="both")
    parser.add_argument("--mode", choices=["linear", "power"], nargs="+", default=["linear", "power"])
    parser.add_argument("--gamma", type=float, nargs="+", default=[-0.002, -0.001, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.008])
    parser.add_argument("--min-lag", type=int, default=3)
    parser.add_argument("--max-lag", type=int, default=30)
    parser.add_argument("--high-conf-quantile", type=float, default=0.65)
    parser.add_argument("--boundary-margin", type=float, default=0.08)
    parser.add_argument("--min-samples", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--copy-prefix", default="logic_timereise_periodic_impulse")
    parser.add_argument("--no-package", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    windows, labels = load_split(args.data_dir, args.eval_split)
    probabilities = run_probabilities(args.source_model, windows, args.eval_batch_size)
    priors = build_priors(windows, labels, probabilities, args)
    np.savez_compressed(output_dir / "periodic_impulse_priors.npz", **priors)
    print(f"Prepared {len(priors)} periodic/impulse priors", flush=True)

    source_weights = extract_timereise_weights(args.source_model)
    base_model = load_base_model(args.base_model or args.source_model)
    scopes: list[tuple[str, list[int]]] = []
    if args.row_scope in {"target", "both"}:
        scopes.append(("c" + "".join(str(item) for item in args.target_classes), args.target_classes))
    if args.row_scope in {"all", "both"}:
        scopes.append(("all", list(range(9))))

    manifest: list[dict[str, object]] = [
        {
            "tag": "start",
            "model": args.source_model,
            "branch": "periodic_impulse",
            "prior": "start",
            "scope": "none",
            "mode": "none",
            "gamma": 0.0,
        }
    ]
    seen_tags: set[str] = set()
    for prior_name, prior in priors.items():
        for scope_name, classes in scopes:
            for mode in args.mode:
                for gamma in args.gamma:
                    if abs(gamma) < 1e-12:
                        continue
                    weights = fuse_weights(source_weights, prior, gamma, mode, classes)
                    tag = f"{prior_name}_{scope_name}_{mode}_{gamma_tag(gamma)}"
                    if tag in seen_tags:
                        continue
                    seen_tags.add(tag)
                    model_path = output_dir / f"logic_timereise_{tag}.onnx"
                    if not model_path.exists():
                        make_variant(base_model, output_dir, tag, weights, hard=False)
                    manifest.append(
                        {
                            "tag": tag,
                            "model": str(model_path),
                            "branch": "periodic_impulse",
                            "prior": prior_name,
                            "scope": scope_name,
                            "mode": mode,
                            "gamma": gamma,
                        }
                    )
                    if args.limit is not None and len(manifest) >= args.limit:
                        break
                if args.limit is not None and len(manifest) >= args.limit:
                    break
            if args.limit is not None and len(manifest) >= args.limit:
                break
        if args.limit is not None and len(manifest) >= args.limit:
            break

    print(f"Prepared {len(manifest)} periodic/impulse TimeREISE variants", flush=True)
    score_and_package_teacher_manifest(manifest, output_dir, args, copy_prefix=args.copy_prefix)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
