from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gearxai_devkit.data import load_split
from gearxai_devkit.metrics import confusion_matrix, macro_f1_score
from gearxai_devkit.runtime import run_submission
from tools.run_logic_logit_bias_calibration import make_biased_model


CLASS_LABELS = ("HEA", "CTF", "MTF", "RCF", "SWF", "BWF", "CWF", "IRF", "ORF")


def parse_float_list(value: str) -> list[float]:
    return [float(part) for part in value.split(",") if part.strip()]


def parse_classes(value: str) -> list[int]:
    if value.strip().lower() in {"all", "*"}:
        return list(range(9))
    out: list[int] = []
    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        if item.isdigit():
            out.append(int(item))
        else:
            out.append(CLASS_LABELS.index(item.upper()))
    return out


def biased_predictions(log_prob: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return (log_prob + bias.reshape(1, -1)).argmax(axis=1).astype(np.int64)


def per_class_f1(matrix: np.ndarray) -> list[float]:
    values: list[float] = []
    for class_id in range(matrix.shape[0]):
        tp = float(matrix[class_id, class_id])
        fp = float(matrix[:, class_id].sum() - tp)
        fn = float(matrix[class_id, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        values.append(2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0)
    return values


def confusion_summary(matrix: np.ndarray) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for true_id in range(matrix.shape[0]):
        for pred_id in range(matrix.shape[1]):
            if true_id == pred_id:
                continue
            count = int(matrix[true_id, pred_id])
            if count:
                rows.append({"true": CLASS_LABELS[true_id], "pred": CLASS_LABELS[pred_id], "count": count})
    return sorted(rows, key=lambda row: int(row["count"]), reverse=True)


def evaluate_bias(log_prob: np.ndarray, labels: np.ndarray, bias: np.ndarray) -> tuple[float, np.ndarray]:
    pred = biased_predictions(log_prob, bias)
    return float(macro_f1_score(labels, pred)), pred


def coordinate_search(
    log_prob: np.ndarray,
    labels: np.ndarray,
    base_pred: np.ndarray,
    classes: list[int],
    values: list[float],
    passes: int,
    min_delta: float,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    current = np.zeros((9,), dtype=np.float32)
    current_score, current_pred = evaluate_bias(log_prob, labels, current)
    history: list[dict[str, object]] = [
        {
            "step": 0,
            "class": "start",
            "macro_f1": current_score,
            "changed_predictions": int(np.count_nonzero(current_pred != base_pred)),
            "bias": {},
        }
    ]
    step = 0
    for pass_id in range(passes):
        improved = False
        for class_id in classes:
            best_value = float(current[class_id])
            best_score = current_score
            best_pred = current_pred
            for value in values:
                trial = current.copy()
                trial[class_id] = float(value)
                score, pred = evaluate_bias(log_prob, labels, trial)
                if score > best_score + min_delta:
                    best_value = float(value)
                    best_score = score
                    best_pred = pred
            if best_score > current_score + min_delta:
                current[class_id] = best_value
                current_score = best_score
                current_pred = best_pred
                step += 1
                improved = True
                row = {
                    "step": step,
                    "pass": pass_id,
                    "class": CLASS_LABELS[class_id],
                    "value": best_value,
                    "macro_f1": current_score,
                    "changed_predictions": int(np.count_nonzero(current_pred != base_pred)),
                    "bias": {
                        CLASS_LABELS[index]: float(value)
                        for index, value in enumerate(current)
                        if abs(float(value)) > 1e-12
                    },
                }
                history.append(row)
                print(
                    f"step={step} pass={pass_id} class={CLASS_LABELS[class_id]} "
                    f"value={best_value:.6f} macro_f1={current_score:.9f} "
                    f"changed={row['changed_predictions']}",
                    flush=True,
                )
        if not improved:
            print(f"No improvement in pass {pass_id}; stopping.", flush=True)
            break
    return current, history


def run_devkit(model_path: Path, report_path: Path, args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        "-m",
        "gearxai_project.evaluate_devkit",
        "--model",
        str(model_path),
        "--data-dir",
        args.data_dir,
        "--split",
        args.split,
        "--batch-size",
        str(args.eval_batch_size),
        "--output",
        str(report_path),
    ]
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Coordinate-search additive logit biases for all classes.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-dir", default="prepared_hf_val5k")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--output-dir", default="runs/logic_logit_bias_coordinate_search")
    parser.add_argument("--classes", type=parse_classes, default=parse_classes("all"))
    parser.add_argument(
        "--values",
        type=parse_float_list,
        default=parse_float_list("-3,-2.5,-2,-1.5,-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1,1.5,2,2.5,3"),
    )
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=1e-12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split = load_split(args.data_dir, args.split)
    runtime = run_submission(args.model, split.windows, batch_size=args.batch_size)
    labels = split.labels.astype(np.int64)
    probabilities = runtime.probabilities.astype(np.float64)
    log_prob = np.log(np.maximum(probabilities, 1e-12))
    base_pred = probabilities.argmax(axis=1).astype(np.int64)
    base_score = float(macro_f1_score(labels, base_pred))
    print(f"base macro_f1={base_score:.9f}", flush=True)

    bias, history = coordinate_search(
        log_prob=log_prob,
        labels=labels,
        base_pred=base_pred,
        classes=args.classes,
        values=args.values,
        passes=args.passes,
        min_delta=args.min_delta,
    )
    final_score, final_pred = evaluate_bias(log_prob, labels, bias)
    matrix = confusion_matrix(labels, final_pred)
    summary = {
        "base_macro_f1": base_score,
        "best_macro_f1": final_score,
        "delta_macro_f1": final_score - base_score,
        "changed_predictions": int(np.count_nonzero(final_pred != base_pred)),
        "bias": {
            CLASS_LABELS[index]: float(value)
            for index, value in enumerate(bias)
            if abs(float(value)) > 1e-12
        },
        "history": history,
        "confusion_matrix": matrix.tolist(),
        "per_class_f1": {CLASS_LABELS[index]: value for index, value in enumerate(per_class_f1(matrix))},
        "top_confusions": confusion_summary(matrix)[:20],
    }
    (output_dir / "coordinate_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"best macro_f1={final_score:.9f}; delta={final_score - base_score:.9f}; "
        f"changed={summary['changed_predictions']}",
        flush=True,
    )
    print(f"best bias={json.dumps(summary['bias'], sort_keys=True)}", flush=True)

    if final_score <= base_score + args.min_delta:
        print("No positive coordinate bias found; not writing a biased ONNX.", flush=True)
        return

    model_path = output_dir / "logic_logit_bias_coordinate.onnx"
    report_path = output_dir / "logic_logit_bias_coordinate_devkit.json"
    make_biased_model(Path(args.model), model_path, bias)
    run_devkit(model_path, report_path, args)
    print(f"Wrote {model_path}", flush=True)
    print(f"Wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
