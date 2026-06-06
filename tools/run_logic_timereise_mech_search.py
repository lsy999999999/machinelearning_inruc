from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import helper

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.search_logic_relevance_variants import add_diff_feature, add_initializer, find_outputs, prune_unused_graph_parts
from tools.timereise_branch_utils import load_base_model, score_and_package_manifest, timereise_weights_from_stats


def add_scalar(graph: onnx.GraphProto, name: str, value: float) -> str:
    return add_initializer(graph, name, np.asarray(value, dtype=np.float32))


def make_mech_variant(
    base_model: onnx.ModelProto,
    output_dir: Path,
    tag: str,
    weights_9x8x100: np.ndarray,
    gamma: float,
) -> Path:
    model = onnx.ModelProto()
    model.CopyFrom(base_model)
    graph = model.graph
    input_name = graph.input[0].name
    prob_output, rel_output = find_outputs(model)
    prefix = f"timereise_{tag}"

    abs_name = f"{prefix}_abs"
    sqrt_name = f"{prefix}_sqrt"
    graph.node.append(helper.make_node("Abs", [input_name], [abs_name], name=abs_name))
    graph.node.append(helper.make_node("Sqrt", [abs_name], [sqrt_name], name=sqrt_name))

    weights_flat_name = add_initializer(graph, f"{prefix}_weights_flat", weights_9x8x100.reshape(9, 800).astype(np.float32))
    factor_flat = f"{prefix}_factor_flat"
    factor_name = f"{prefix}_factor"
    shape_name = add_initializer(graph, f"{prefix}_shape", [-1, 8, 100], dtype=np.int64)
    graph.node.append(helper.make_node("MatMul", [prob_output.name, weights_flat_name], [factor_flat], name=factor_flat))
    graph.node.append(helper.make_node("Reshape", [factor_flat, shape_name], [factor_name], name=factor_name))

    diff_name = add_diff_feature(graph, input_name, prefix)
    local_name = f"{prefix}_local5"
    peak_raw = f"{prefix}_peak5_raw"
    peak_name = f"{prefix}_peak5"
    graph.node.append(
        helper.make_node(
            "AveragePool",
            [abs_name],
            [local_name],
            name=local_name,
            kernel_shape=[5],
            strides=[1],
            pads=[2, 2],
            count_include_pad=0,
        )
    )
    graph.node.append(helper.make_node("Sub", [abs_name, local_name], [peak_raw], name=peak_raw))
    graph.node.append(helper.make_node("Relu", [peak_raw], [peak_name], name=peak_name))

    diff_weight = add_scalar(graph, f"{prefix}_diff_w", 0.55)
    peak_weight = add_scalar(graph, f"{prefix}_peak_w", 0.45)
    diff_scaled = f"{prefix}_diff_scaled"
    peak_scaled = f"{prefix}_peak_scaled"
    proxy_name = f"{prefix}_proxy"
    graph.node.append(helper.make_node("Mul", [diff_name, diff_weight], [diff_scaled], name=diff_scaled))
    graph.node.append(helper.make_node("Mul", [peak_name, peak_weight], [peak_scaled], name=peak_scaled))
    graph.node.append(helper.make_node("Add", [diff_scaled, peak_scaled], [proxy_name], name=proxy_name))

    base_rel = f"{prefix}_base_relevance"
    proxy_scaled = f"{prefix}_proxy_scaled"
    multiplier = f"{prefix}_multiplier"
    relevance_name = f"{prefix}_relevance"
    gamma_name = add_scalar(graph, f"{prefix}_gamma", gamma)
    one_name = add_scalar(graph, f"{prefix}_one", 1.0)
    graph.node.append(helper.make_node("Mul", [sqrt_name, factor_name], [base_rel], name=base_rel))
    graph.node.append(helper.make_node("Mul", [proxy_name, gamma_name], [proxy_scaled], name=proxy_scaled))
    graph.node.append(helper.make_node("Add", [one_name, proxy_scaled], [multiplier], name=multiplier))
    graph.node.append(helper.make_node("Mul", [base_rel, multiplier], [relevance_name], name=relevance_name))

    new_rel_output = onnx.ValueInfoProto()
    new_rel_output.CopyFrom(rel_output)
    new_rel_output.name = relevance_name
    graph.ClearField("output")
    graph.output.extend([prob_output, new_rel_output])
    prune_unused_graph_parts(model)

    output_path = output_dir / f"logic_timereise_{tag}.onnx"
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mechanical-aware TimeREISE relevance search.")
    parser.add_argument("--base-model", default="external/gearxai_devkit/baselines/onnx/logic_lstm.onnx")
    parser.add_argument("--stats", default="runs/logic_timereise_search_50k_b20_refine/timereise_stats.npz")
    parser.add_argument("--output-dir", default="runs/logic_timereise_mech_search")
    parser.add_argument("--data-dir", default="prepared_hf_val5k")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--time-beta", type=float, default=0.35)
    parser.add_argument("--gamma", type=float, nargs="+", default=[0.05, 0.10, 0.15])
    parser.add_argument("--copy-prefix", default="logic_timereise_mech")
    parser.add_argument("--no-package", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_model = load_base_model(args.base_model)
    weights = timereise_weights_from_stats(args.stats, args.time_beta)

    manifest = []
    for gamma in args.gamma:
        tag = f"mech_mix50_tb{int(args.time_beta * 100):03d}_g{int(gamma * 100):03d}"
        model_path = output_dir / f"logic_timereise_{tag}.onnx"
        if not model_path.exists():
            make_mech_variant(base_model, output_dir, tag, weights, gamma)
        manifest.append({"tag": tag, "model": str(model_path), "branch": "mechanical", "gamma": gamma})
    print(f"Prepared {len(manifest)} mechanical-aware TimeREISE variants", flush=True)
    score_and_package_manifest(manifest, output_dir, args, copy_prefix=args.copy_prefix)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
