from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from tqdm.auto import tqdm

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid  # type: ignore[attr-defined]

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.run_logic_timereise_power_ensemble_search import extract_timereise_weights
from tools.run_logic_timereise_search import make_variant
from tools.timereise_branch_utils import load_base_model, score_and_package_manifest


DEFAULT_CANDIDATES = (
    ("marg_b10", "runs/candidates/logic_timereise_marginal_val5k_b10_bestproxy.onnx"),
    ("robust", "runs/candidates/logic_timereise_robust_bestfaith.onnx"),
    ("pow_b030", "runs/logic_timereise_power_ensemble_robust_refine/logic_timereise_b030.onnx"),
    ("pow_b035", "runs/logic_timereise_power_ensemble_robust_refine/logic_timereise_b035.onnx"),
    ("pow_b040", "runs/logic_timereise_power_ensemble_robust_refine/logic_timereise_b040.onnx"),
    ("pow_b041", "runs/logic_timereise_power_ensemble_robust_fine/logic_timereise_b041.onnx"),
    ("pow_b042", "runs/logic_timereise_power_ensemble_robust_fine/logic_timereise_b042.onnx"),
    ("pow_b043", "runs/logic_timereise_power_ensemble_robust_fine/logic_timereise_b043.onnx"),
    ("pow_b044", "runs/logic_timereise_power_ensemble_robust_fine/logic_timereise_b044.onnx"),
    ("pow_b045", "runs/logic_timereise_power_ensemble_robust_fine/logic_timereise_b045.onnx"),
    ("pow_b046", "runs/logic_timereise_power_ensemble_robust_fine/logic_timereise_b046.onnx"),
    ("pow_b047", "runs/logic_timereise_power_ensemble_robust_fine/logic_timereise_b047.onnx"),
    ("pow_b048", "runs/logic_timereise_power_ensemble_robust_fine/logic_timereise_b048.onnx"),
    ("trimmed", "runs/candidates/logic_timereise_quantile_trimmed_val5k_b10_bestproxy.onnx"),
)


def parse_candidate(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("candidate must be name=path")
    name, path = value.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise argparse.ArgumentTypeError("candidate must be name=path")
    return name, path


def load_split(data_dir: str | Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    data_dir = Path(data_dir)
    windows = np.load(data_dir / f"{split}_windows.npy").astype(np.float32)
    labels = np.load(data_dir / f"{split}_labels.npy").astype(np.int64)
    return windows, labels


def load_channel_mean(data_dir: str | Path) -> np.ndarray:
    stats_path = Path(data_dir) / "stats.json"
    if not stats_path.exists():
        return np.zeros((8,), dtype=np.float32)
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    if "standardized_channel_mean" not in stats:
        return np.zeros((8,), dtype=np.float32)
    return np.asarray(stats["standardized_channel_mean"], dtype=np.float32)


def run_model(session: ort.InferenceSession, windows: np.ndarray, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    prob_chunks: list[np.ndarray] = []
    rel_chunks: list[np.ndarray] = []
    for start in range(0, len(windows), batch_size):
        end = min(start + batch_size, len(windows))
        outputs = session.run(output_names, {input_name: windows[start:end].astype(np.float32, copy=False)})
        prob = None
        rel = None
        for output in outputs:
            if output.ndim == 2 and output.shape[-1] == 9:
                prob = output
            elif output.ndim == 3 and output.shape[-2:] == (8, 100):
                rel = output
        if prob is None or rel is None:
            raise RuntimeError("Could not identify model outputs.")
        prob_chunks.append(prob.astype(np.float32))
        rel_chunks.append(rel.astype(np.float32))
    return np.concatenate(prob_chunks, axis=0), np.concatenate(rel_chunks, axis=0)


def predict_prob(session: ort.InferenceSession, windows: np.ndarray, batch_size: int) -> np.ndarray:
    probabilities, _relevance = run_model(session, windows, batch_size)
    return probabilities


def normalize_relevance(relevance: np.ndarray) -> np.ndarray:
    relevance = np.maximum(relevance.astype(np.float32), 0.0)
    total = relevance.sum(axis=(1, 2), keepdims=True)
    return relevance / np.maximum(total, 1e-12)


def topk_mask(relevance: np.ndarray, fraction: float) -> np.ndarray:
    flat = relevance.reshape(relevance.shape[0], -1)
    total_cells = flat.shape[1]
    k = int(round(fraction * total_cells))
    mask = np.zeros_like(flat, dtype=bool)
    if k <= 0:
        return mask.reshape(relevance.shape)
    order = np.argpartition(-flat, kth=min(k - 1, total_cells - 1), axis=1)[:, :k]
    rows = np.arange(flat.shape[0])[:, None]
    mask[rows, order] = True
    return mask.reshape(relevance.shape)


def classwise_faith(
    session: ort.InferenceSession,
    windows: np.ndarray,
    relevance: np.ndarray,
    class_ids: np.ndarray,
    groups: np.ndarray,
    channel_mean: np.ndarray,
    batch_size: int,
) -> dict[str, object]:
    relevance = normalize_relevance(relevance)
    baseline = channel_mean.reshape(1, 8, 1).astype(np.float32)
    baseline_full = np.broadcast_to(baseline, windows.shape).astype(np.float32)
    base_input = baseline_full.copy()
    rows = np.arange(len(windows))
    fractions = np.linspace(0.0, 1.0, 11)
    deletion_by_fraction = []
    insertion_by_fraction = []

    for fraction in fractions:
        mask = topk_mask(relevance, float(fraction))
        deleted = windows.copy()
        deleted[mask] = baseline_full[mask]
        inserted = base_input.copy()
        inserted[mask] = windows[mask]
        deleted_probs = predict_prob(session, deleted, batch_size)
        inserted_probs = predict_prob(session, inserted, batch_size)
        deletion_by_fraction.append(deleted_probs[rows, class_ids])
        insertion_by_fraction.append(inserted_probs[rows, class_ids])

    deletion = np.stack(deletion_by_fraction, axis=1)
    insertion = np.stack(insertion_by_fraction, axis=1)
    class_scores = np.zeros((9,), dtype=np.float64)
    class_deletion = np.zeros((9,), dtype=np.float64)
    class_insertion = np.zeros((9,), dtype=np.float64)
    class_counts = np.zeros((9,), dtype=np.int64)
    for class_id in range(9):
        mask = groups == class_id
        class_counts[class_id] = int(mask.sum())
        if not np.any(mask):
            continue
        deletion_curve = deletion[mask].mean(axis=0)
        insertion_curve = insertion[mask].mean(axis=0)
        deletion_auc = float(np.trapz(deletion_curve, fractions))
        insertion_auc = float(np.trapz(insertion_curve, fractions))
        class_deletion[class_id] = deletion_auc
        class_insertion[class_id] = insertion_auc
        class_scores[class_id] = float(np.clip((insertion_auc + (1.0 - deletion_auc)) / 2.0, 0.0, 1.0))
    return {
        "faith": class_scores.tolist(),
        "deletion": class_deletion.tolist(),
        "insertion": class_insertion.tolist(),
        "counts": class_counts.tolist(),
    }


def select_rows(
    candidate_weights: dict[str, np.ndarray],
    class_scores: dict[str, list[float]],
) -> tuple[np.ndarray, list[dict[str, object]]]:
    names = list(candidate_weights)
    selected = np.zeros((9, 8, 100), dtype=np.float32)
    rows: list[dict[str, object]] = []
    for class_id in range(9):
        best_name = max(names, key=lambda name: float(class_scores[name][class_id]))
        selected[class_id] = candidate_weights[best_name][class_id]
        rows.append({"class_id": class_id, "candidate": best_name, "faith": float(class_scores[best_name][class_id])})
    return selected, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Class-specific candidate selection for TimeREISE weights.")
    parser.add_argument("--base-model", default="external/gearxai_devkit/baselines/onnx/logic_lstm.onnx")
    parser.add_argument("--candidate", type=parse_candidate, nargs="+", default=list(DEFAULT_CANDIDATES))
    parser.add_argument("--candidate-manifest", nargs="*", default=[], help="JSON or JSONL manifest rows with tag/model fields.")
    parser.add_argument("--output-dir", default="runs/logic_timereise_class_candidate_selection")
    parser.add_argument("--data-dir", default="prepared_hf_val5k")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--group-by", choices=["true", "pred"], nargs="+", default=["true", "pred"])
    parser.add_argument("--copy-prefix", default="logic_timereise_class_candidate_selection")
    parser.add_argument("--no-package", action="store_true")
    return parser.parse_args()


def load_manifest_candidates(paths: list[str]) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    for path in paths:
        manifest_path = Path(path)
        text = manifest_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        if text.startswith("["):
            rows = json.loads(text)
        else:
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        for row in rows:
            tag = str(row.get("tag") or row.get("name") or "").strip()
            model = str(row.get("model") or "").strip()
            if not tag or not model:
                raise ValueError(f"Manifest row must contain tag/model: {row}")
            candidates.append((tag, model))
    return candidates


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    windows, labels = load_split(args.data_dir, args.eval_split)
    channel_mean = load_channel_mean(args.data_dir)

    args.candidate = list(args.candidate) + load_manifest_candidates(args.candidate_manifest)

    loaded_candidates: list[tuple[str, Path]] = []
    candidate_weights: dict[str, np.ndarray] = {}
    for name, path in args.candidate:
        model_path = Path(path)
        if not model_path.exists():
            print(f"Skipping missing candidate {name}: {model_path}", flush=True)
            continue
        loaded_candidates.append((name, model_path))
        candidate_weights[name] = extract_timereise_weights(model_path)
    if not loaded_candidates:
        raise RuntimeError("No candidates were loaded.")

    base_session = ort.InferenceSession(str(loaded_candidates[0][1]), providers=["CPUExecutionProvider"])
    base_prob, _base_rel = run_model(base_session, windows, args.eval_batch_size)
    pred = base_prob.argmax(axis=1).astype(np.int64)

    score_table: dict[str, dict[str, object]] = {}
    for name, model_path in tqdm(loaded_candidates, desc="classwise-candidate-faith", leave=True):
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        probabilities, relevance = run_model(session, windows, args.eval_batch_size)
        class_ids = probabilities.argmax(axis=1).astype(np.int64)
        if not np.array_equal(class_ids, pred):
            raise RuntimeError(f"Candidate {name} changed predictions.")
        score_table[name] = {}
        for group_by in args.group_by:
            groups = labels if group_by == "true" else pred
            score_table[name][group_by] = classwise_faith(
                session,
                windows,
                relevance,
                class_ids,
                groups,
                channel_mean,
                args.eval_batch_size,
            )

    (output_dir / "classwise_scores.json").write_text(json.dumps(score_table, indent=2, sort_keys=True), encoding="utf-8")

    base_model = load_base_model(args.base_model)
    manifest = []
    selections: dict[str, list[dict[str, object]]] = {}
    for group_by in args.group_by:
        class_scores = {
            name: score_table[name][group_by]["faith"]  # type: ignore[index]
            for name, _path in loaded_candidates
        }
        selected_weights, rows = select_rows(candidate_weights, class_scores)
        tag = f"classsel_{group_by}"
        model_path = output_dir / f"logic_timereise_{tag}.onnx"
        if not model_path.exists():
            make_variant(base_model, output_dir, tag, selected_weights, hard=False)
        selections[group_by] = rows
        manifest.append({"tag": tag, "model": str(model_path), "branch": "class_candidate_selection", "selection": rows})
        print(f"Selection {group_by}: {rows}", flush=True)

    (output_dir / "selections.json").write_text(json.dumps(selections, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Prepared {len(manifest)} class-specific candidate-selection variants", flush=True)
    score_and_package_manifest(manifest, output_dir, args, copy_prefix=args.copy_prefix)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
