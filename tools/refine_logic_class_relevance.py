from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.search_logic_relevance_variants import add_initializer, find_outputs, prune_unused_graph_parts


def normalized_matrices(class_means: np.ndarray) -> dict[str, np.ndarray]:
    eps = 1e-6
    global_mean = class_means.mean(axis=0, keepdims=True)
    ratio = class_means / np.maximum(global_mean, eps)
    ratio = ratio / np.maximum(ratio.mean(axis=1, keepdims=True), eps)

    center = class_means / np.maximum(class_means.mean(axis=1, keepdims=True), eps)
    center = center / np.maximum(center.mean(axis=1, keepdims=True), eps)

    z = class_means - class_means.mean(axis=1, keepdims=True)
    z = z / np.maximum(class_means.std(axis=1, keepdims=True), eps)
    return {"ratio": ratio, "center": center, "z": z}


def build_weights(class_means: np.ndarray) -> list[tuple[str, np.ndarray]]:
    bases = normalized_matrices(class_means)
    specs: list[tuple[str, np.ndarray]] = []

    for alpha in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
        matrix = 1.0 + alpha * (bases["ratio"] - 1.0)
        specs.append((f"ratio_a{int(alpha * 100):03d}", np.clip(matrix, 0.05, None).astype(np.float32)))

    for alpha in (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
        matrix = 1.0 + alpha * (bases["center"] - 1.0)
        specs.append((f"center_a{int(alpha * 100):03d}", np.clip(matrix, 0.05, None).astype(np.float32)))

    for alpha in (0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26):
        matrix = 1.0 + alpha * bases["z"]
        specs.append((f"z_a{int(alpha * 100):03d}", np.clip(matrix, 0.05, None).astype(np.float32)))

    return specs


def add_power_feature(graph: onnx.GraphProto, input_name: str, prefix: str, power: str) -> str:
    abs_name = f"{prefix}_abs"
    graph.node.append(helper.make_node("Abs", [input_name], [abs_name], name=abs_name))
    if power == "abs":
        return abs_name

    sqrt_name = f"{prefix}_sqrt"
    graph.node.append(helper.make_node("Sqrt", [abs_name], [sqrt_name], name=sqrt_name))
    if power == "sqrt":
        return sqrt_name

    fourth_name = f"{prefix}_fourth"
    graph.node.append(helper.make_node("Sqrt", [sqrt_name], [fourth_name], name=fourth_name))
    if power == "fourth":
        return fourth_name

    raise ValueError(f"unknown power feature: {power}")


def add_class_factor(
    model: onnx.ModelProto,
    prob_name: str,
    weights: np.ndarray,
    prefix: str,
    *,
    hard: bool,
) -> str:
    graph = model.graph
    weights_name = add_initializer(graph, f"{prefix}_weights", weights.astype(np.float32))
    class_factor_2d = f"{prefix}_class_factor_2d"
    class_factor = f"{prefix}_class_factor"
    axes_name = add_initializer(graph, f"{prefix}_unsqueeze_axes", [2], dtype=np.int64)

    if hard:
        argmax_name = f"{prefix}_argmax"
        graph.node.append(helper.make_node("ArgMax", [prob_name], [argmax_name], name=argmax_name, axis=1, keepdims=0))
        graph.node.append(helper.make_node("Gather", [weights_name, argmax_name], [class_factor_2d], name=class_factor_2d, axis=0))
    else:
        graph.node.append(helper.make_node("MatMul", [prob_name, weights_name], [class_factor_2d], name=class_factor_2d))

    graph.node.append(helper.make_node("Unsqueeze", [class_factor_2d, axes_name], [class_factor], name=class_factor))
    return class_factor


def make_variant(
    base_model: onnx.ModelProto,
    output_dir: Path,
    tag: str,
    weights: np.ndarray,
    *,
    power: str,
    hard: bool,
) -> Path:
    model = onnx.ModelProto()
    model.CopyFrom(base_model)
    graph = model.graph
    input_name = graph.input[0].name
    prob_output, rel_output = find_outputs(model)
    prefix = f"class_refine_{tag}"

    feature_name = add_power_feature(graph, input_name, prefix, power)
    factor_name = add_class_factor(model, prob_output.name, weights, prefix, hard=hard)
    relevance_name = f"{prefix}_relevance"
    graph.node.append(helper.make_node("Mul", [feature_name, factor_name], [relevance_name], name=relevance_name))

    new_rel_output = onnx.ValueInfoProto()
    new_rel_output.CopyFrom(rel_output)
    new_rel_output.name = relevance_name
    graph.ClearField("output")
    graph.output.extend([prob_output, new_rel_output])
    prune_unused_graph_parts(model)

    output_path = output_dir / f"logic_class_refine_{tag}.onnx"
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine class-specific LogicLSTM relevance variants.")
    parser.add_argument("--base-model", default="external/gearxai_devkit/baselines/onnx/logic_lstm.onnx")
    parser.add_argument("--stats", default="runs/logic_class_xai_search/class_channel_abs_means.npy")
    parser.add_argument("--output-dir", default="runs/logic_class_xai_refine")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    class_means = np.load(args.stats)
    base_model = onnx.load(args.base_model)

    manifest = []
    for matrix_name, weights in build_weights(class_means):
        for power in ("sqrt", "fourth"):
            for hard in (False, True):
                tag = matrix_name
                if power != "sqrt":
                    tag += f"_{power}"
                if hard:
                    tag += "_hard"
                model_path = make_variant(base_model, output_dir, tag, weights, power=power, hard=hard)
                row = {
                    "model": str(model_path),
                    "tag": tag,
                    "matrix": matrix_name,
                    "power": power,
                    "hard": hard,
                }
                manifest.append(row)
                print(f"Saved {model_path}")

    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"Wrote {output_dir / 'manifest.jsonl'} ({len(manifest)} variants)")


if __name__ == "__main__":
    main()
