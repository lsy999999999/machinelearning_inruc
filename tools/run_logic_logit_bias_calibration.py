from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gearxai_devkit.data import load_split
from gearxai_devkit.metrics import confusion_matrix, macro_f1_score
from gearxai_devkit.runtime import run_submission


CLASS_LABELS = ("HEA", "CTF", "MTF", "RCF", "SWF", "BWF", "CWF", "IRF", "ORF")


def parse_float_list(value: str) -> list[float]:
    return [float(part) for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Targeted logit-bias calibration for the current LogicLSTM candidate.")
    parser.add_argument("--model", default="runs/candidates/logic_timereise_row_coordinate_ext_bestproxy.onnx")
    parser.add_argument("--data-dir", default="prepared_hf_val5k")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--output-dir", default="runs/logic_logit_bias_calibration")
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--package", action="store_true")
    parser.add_argument("--orf", type=parse_float_list, default=parse_float_list("-0.12,-0.10,-0.08,-0.06,-0.04,-0.02,0"))
    parser.add_argument("--cwf", type=parse_float_list, default=parse_float_list("0,0.02,0.04,0.06,0.08,0.10,0.12"))
    parser.add_argument("--irf", type=parse_float_list, default=parse_float_list("0,0.01,0.02,0.03,0.04,0.05,0.06"))
    parser.add_argument("--rcf", type=parse_float_list, default=parse_float_list("-0.04,-0.02,0,0.02"))
    parser.add_argument("--hea", type=parse_float_list, default=parse_float_list("0,0.01,0.02,0.03"))
    return parser.parse_args()


def calibrate_probabilities(probabilities: np.ndarray, bias: np.ndarray) -> np.ndarray:
    eps = np.float32(1e-12)
    logits = np.log(np.maximum(probabilities.astype(np.float64), eps))
    logits = logits + bias.reshape(1, -1)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return (exp_logits / exp_logits.sum(axis=1, keepdims=True)).astype(np.float32)


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
                rows.append(
                    {
                        "true": CLASS_LABELS[true_id],
                        "pred": CLASS_LABELS[pred_id],
                        "count": count,
                    }
                )
    return sorted(rows, key=lambda row: int(row["count"]), reverse=True)


def make_biased_model(model_path: Path, output_path: Path, bias: np.ndarray) -> None:
    model = onnx.load(model_path)
    graph = model.graph
    softmax_nodes = [node for node in graph.node if node.op_type == "Softmax" and "probabilities" in node.output]
    if len(softmax_nodes) != 1:
        raise RuntimeError(f"Expected exactly one Softmax node producing probabilities, found {len(softmax_nodes)}")

    softmax = softmax_nodes[0]
    logits_name = softmax.input[0]
    used_names = {initializer.name for initializer in graph.initializer}
    used_names.update(value.name for value in graph.value_info)
    used_names.update(value.name for value in graph.input)
    used_names.update(value.name for value in graph.output)
    used_names.update(output for node in graph.node for output in node.output)
    used_node_names = {node.name for node in graph.node}

    def unique_name(base: str, used: set[str]) -> str:
        if base not in used:
            used.add(base)
            return base
        index = 2
        while f"{base}_{index}" in used:
            index += 1
        name = f"{base}_{index}"
        used.add(name)
        return name

    bias_name = unique_name("logit_bias_calibration_bias", used_names)
    biased_logits_name = unique_name("logit_bias_calibration_logits", used_names)
    add_name = unique_name("logit_bias_calibration_add", used_node_names)
    graph.initializer.append(numpy_helper.from_array(bias.astype(np.float32), name=bias_name))

    add_node = helper.make_node("Add", [logits_name, bias_name], [biased_logits_name], name=add_name)
    softmax_index = next(index for index, node in enumerate(graph.node) if node is softmax)
    graph.node.insert(softmax_index, add_node)
    softmax.input[0] = biased_logits_name

    onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)


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


def package_model(model_path: Path, package_path: Path, args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        "-m",
        "gearxai_project.package_devkit",
        "package",
        "--model",
        str(model_path),
        "--data-dir",
        args.data_dir,
        "--split",
        args.split,
        "--batch-size",
        str(args.eval_batch_size),
        "--out",
        str(package_path),
    ]
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split = load_split(args.data_dir, args.split)
    runtime = run_submission(args.model, split.windows, batch_size=args.batch_size)
    labels = split.labels.astype(np.int64)
    base_pred = runtime.probabilities.argmax(axis=1).astype(np.int64)
    base_matrix = confusion_matrix(labels, base_pred)
    base_f1 = macro_f1_score(labels, base_pred)

    rows: list[dict[str, object]] = []
    for orf, cwf, irf, rcf, hea in itertools.product(args.orf, args.cwf, args.irf, args.rcf, args.hea):
        bias = np.zeros(9, dtype=np.float32)
        bias[8] = orf
        bias[6] = cwf
        bias[7] = irf
        bias[3] = rcf
        bias[0] = hea
        calibrated = calibrate_probabilities(runtime.probabilities, bias)
        pred = calibrated.argmax(axis=1).astype(np.int64)
        matrix = confusion_matrix(labels, pred)
        f1 = macro_f1_score(labels, pred)
        rows.append(
            {
                "macro_f1": float(f1),
                "delta_macro_f1": float(f1 - base_f1),
                "changed_predictions": int(np.count_nonzero(pred != base_pred)),
                "bias": {CLASS_LABELS[index]: float(value) for index, value in enumerate(bias) if abs(float(value)) > 1e-12},
                "per_class_f1": {CLASS_LABELS[index]: value for index, value in enumerate(per_class_f1(matrix))},
                "top_confusions": confusion_summary(matrix)[:12],
            }
        )

    rows.sort(key=lambda row: (float(row["macro_f1"]), -int(row["changed_predictions"])), reverse=True)
    best = rows[0]
    summary = {
        "base": {
            "macro_f1": float(base_f1),
            "confusion_matrix": base_matrix.tolist(),
            "per_class_f1": {CLASS_LABELS[index]: value for index, value in enumerate(per_class_f1(base_matrix))},
            "top_confusions": confusion_summary(base_matrix)[:12],
        },
        "grid_size": len(rows),
        "best": best,
        "top": rows[:30],
    }
    (output_dir / "calibration_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(
        f"base macro_f1={base_f1:.9f}; best macro_f1={best['macro_f1']:.9f}; "
        f"delta={best['delta_macro_f1']:.9f}; changed={best['changed_predictions']}",
        flush=True,
    )
    print(f"best bias={json.dumps(best['bias'], sort_keys=True)}", flush=True)

    if float(best["delta_macro_f1"]) <= 0.0:
        print("No positive calibration found; not writing a biased ONNX.", flush=True)
        return

    bias = np.zeros(9, dtype=np.float32)
    for label, value in dict(best["bias"]).items():
        bias[CLASS_LABELS.index(label)] = float(value)

    model_path = output_dir / "logic_timereise_row_coordinate_ext_logit_bias.onnx"
    report_path = output_dir / "logic_timereise_row_coordinate_ext_logit_bias_devkit.json"
    make_biased_model(Path(args.model), model_path, bias)
    run_devkit(model_path, report_path, args)
    print(f"Wrote {model_path}", flush=True)
    print(f"Wrote {report_path}", flush=True)

    if args.package:
        package_path = Path("runs/final_candidates/logic_timereise_row_coordinate_ext_logit_bias_submission.zip")
        package_model(model_path, package_path, args)
        print(f"Packaged {package_path}", flush=True)


if __name__ == "__main__":
    main()
