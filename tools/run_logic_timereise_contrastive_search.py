from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import helper

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.search_logic_relevance_variants import add_initializer, find_outputs, prune_unused_graph_parts
from tools.timereise_branch_utils import load_base_model, score_and_package_manifest, timereise_weights_from_stats


def add_scalar(graph: onnx.GraphProto, name: str, value: float) -> str:
    return add_initializer(graph, name, np.asarray(value, dtype=np.float32))


def make_contrastive_variant(
    base_model: onnx.ModelProto,
    output_dir: Path,
    tag: str,
    weights_9x8x100: np.ndarray,
    lam: float,
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
    graph.node.append(helper.make_node("MatMul", [prob_output.name, weights_flat_name], [factor_flat], name=factor_flat))

    k_name = add_initializer(graph, f"{prefix}_topk_k", [2], dtype=np.int64)
    topk_values = f"{prefix}_topk_values"
    topk_indices = f"{prefix}_topk_indices"
    graph.node.append(helper.make_node("TopK", [prob_output.name, k_name], [topk_values, topk_indices], name=f"{prefix}_topk", axis=1, largest=1, sorted=1))

    starts = add_initializer(graph, f"{prefix}_slice_starts", [1], dtype=np.int64)
    ends = add_initializer(graph, f"{prefix}_slice_ends", [2], dtype=np.int64)
    axes = add_initializer(graph, f"{prefix}_slice_axes", [1], dtype=np.int64)
    steps = add_initializer(graph, f"{prefix}_slice_steps", [1], dtype=np.int64)
    top2_slice = f"{prefix}_top2_slice"
    top2_index = f"{prefix}_top2_index"
    squeeze_axes = add_initializer(graph, f"{prefix}_squeeze_axes", [1], dtype=np.int64)
    graph.node.append(helper.make_node("Slice", [topk_indices, starts, ends, axes, steps], [top2_slice], name=top2_slice))
    graph.node.append(helper.make_node("Squeeze", [top2_slice, squeeze_axes], [top2_index], name=top2_index))

    top2_factor = f"{prefix}_top2_factor"
    graph.node.append(helper.make_node("Gather", [weights_flat_name, top2_index], [top2_factor], name=top2_factor, axis=0))

    one_name = add_scalar(graph, f"{prefix}_one", 1.0)
    top2_over_raw = f"{prefix}_top2_over_raw"
    top2_over = f"{prefix}_top2_over"
    scaled_contrast = f"{prefix}_scaled_contrast"
    raw_factor = f"{prefix}_raw_factor"
    clipped_factor_flat = f"{prefix}_clipped_factor_flat"
    lam_name = add_scalar(graph, f"{prefix}_lambda", lam)
    clip_min = add_scalar(graph, f"{prefix}_clip_min", 0.05)
    graph.node.append(helper.make_node("Sub", [top2_factor, one_name], [top2_over_raw], name=top2_over_raw))
    graph.node.append(helper.make_node("Relu", [top2_over_raw], [top2_over], name=top2_over))
    graph.node.append(helper.make_node("Mul", [top2_over, lam_name], [scaled_contrast], name=scaled_contrast))
    graph.node.append(helper.make_node("Sub", [factor_flat, scaled_contrast], [raw_factor], name=raw_factor))
    graph.node.append(helper.make_node("Clip", [raw_factor, clip_min], [clipped_factor_flat], name=clipped_factor_flat))

    shape_name = add_initializer(graph, f"{prefix}_shape", [-1, 8, 100], dtype=np.int64)
    factor_name = f"{prefix}_factor"
    relevance_name = f"{prefix}_relevance"
    graph.node.append(helper.make_node("Reshape", [clipped_factor_flat, shape_name], [factor_name], name=factor_name))
    graph.node.append(helper.make_node("Mul", [sqrt_name, factor_name], [relevance_name], name=relevance_name))

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
    parser = argparse.ArgumentParser(description="Contrastive TimeREISE top-2 suppression search.")
    parser.add_argument("--base-model", default="external/gearxai_devkit/baselines/onnx/logic_lstm.onnx")
    parser.add_argument("--stats", default="runs/logic_timereise_search_50k_b20_refine/timereise_stats.npz")
    parser.add_argument("--output-dir", default="runs/logic_timereise_contrastive_search")
    parser.add_argument("--data-dir", default="prepared_hf_val5k")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--time-beta", type=float, default=0.35)
    parser.add_argument("--lambda-value", type=float, nargs="+", default=[0.05, 0.10, 0.15])
    parser.add_argument("--copy-prefix", default="logic_timereise_contrastive")
    parser.add_argument("--no-package", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_model = load_base_model(args.base_model)
    weights = timereise_weights_from_stats(args.stats, args.time_beta)

    manifest = []
    for lam in args.lambda_value:
        tag = f"contrast_mix50_tb{int(args.time_beta * 100):03d}_l{int(lam * 100):03d}"
        model_path = output_dir / f"logic_timereise_{tag}.onnx"
        if not model_path.exists():
            make_contrastive_variant(base_model, output_dir, tag, weights, lam)
        manifest.append({"tag": tag, "model": str(model_path), "branch": "contrastive", "lambda": lam})
    print(f"Prepared {len(manifest)} contrastive TimeREISE variants", flush=True)
    score_and_package_manifest(manifest, output_dir, args, copy_prefix=args.copy_prefix)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
