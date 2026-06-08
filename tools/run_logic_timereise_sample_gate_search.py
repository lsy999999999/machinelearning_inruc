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
from tools.timereise_branch_utils import score_and_package_manifest


def add_group_conv_feature(graph: onnx.GraphProto, input_name: str, prefix: str, kind: str) -> str:
    if kind == "diff":
        weight = np.zeros((8, 1, 2), dtype=np.float32)
        weight[:, 0, 0] = -1.0
        weight[:, 0, 1] = 1.0
        weight_name = add_initializer(graph, f"{prefix}_diff_kernel", weight)
        conv_name = f"{prefix}_diff_conv"
        out_name = f"{prefix}_diff_abs"
        graph.node.append(
            helper.make_node("Conv", [input_name, weight_name], [conv_name], name=conv_name, group=8, pads=[1, 0])
        )
        graph.node.append(helper.make_node("Abs", [conv_name], [out_name], name=out_name))
        return out_name
    raise ValueError(f"Unknown grouped conv feature: {kind}")


def add_abs_feature(graph: onnx.GraphProto, input_name: str, prefix: str) -> str:
    out_name = f"{prefix}_abs"
    graph.node.append(helper.make_node("Abs", [input_name], [out_name], name=out_name))
    return out_name


def add_local_feature(graph: onnx.GraphProto, abs_name: str, prefix: str, kernel: int) -> str:
    out_name = f"{prefix}_local{kernel}"
    pad = kernel // 2
    graph.node.append(
        helper.make_node(
            "AveragePool",
            [abs_name],
            [out_name],
            name=out_name,
            kernel_shape=[kernel],
            strides=[1],
            pads=[pad, pad],
            count_include_pad=0,
        )
    )
    return out_name


def add_peak_feature(graph: onnx.GraphProto, abs_name: str, local_name: str, prefix: str) -> str:
    raw_name = f"{prefix}_peak_raw"
    out_name = f"{prefix}_peak"
    graph.node.append(helper.make_node("Sub", [abs_name, local_name], [raw_name], name=raw_name))
    graph.node.append(helper.make_node("Relu", [raw_name], [out_name], name=out_name))
    return out_name


def add_weighted_sum(graph: onnx.GraphProto, prefix: str, terms: list[tuple[str, float]]) -> str:
    current = ""
    for index, (name, weight) in enumerate(terms):
        scaled = name
        if abs(weight - 1.0) > 1e-9:
            weight_name = add_initializer(graph, f"{prefix}_term_weight_{index}", np.asarray(weight, dtype=np.float32))
            scaled = f"{prefix}_term_scaled_{index}"
            graph.node.append(helper.make_node("Mul", [name, weight_name], [scaled], name=scaled))
        if not current:
            current = scaled
        else:
            out_name = f"{prefix}_term_sum_{index}"
            graph.node.append(helper.make_node("Add", [current, scaled], [out_name], name=out_name))
            current = out_name
    if not current:
        raise ValueError("No terms for weighted sum.")
    return current


def add_sample_gate(graph: onnx.GraphProto, prefix: str, relevance_name: str, proxy_name: str, gamma: float) -> str:
    mean_name = f"{prefix}_proxy_mean"
    eps_name = add_initializer(graph, f"{prefix}_eps", np.asarray(1e-6, dtype=np.float32))
    denom_name = f"{prefix}_proxy_denom"
    norm_name = f"{prefix}_proxy_norm"
    gamma_name = add_initializer(graph, f"{prefix}_gamma", np.asarray(gamma, dtype=np.float32))
    base_name = add_initializer(graph, f"{prefix}_base", np.asarray(1.0 - gamma, dtype=np.float32))
    scaled_name = f"{prefix}_gate_scaled"
    gate_name = f"{prefix}_gate"
    out_name = f"{prefix}_relevance"

    graph.node.append(helper.make_node("ReduceMean", [proxy_name], [mean_name], name=mean_name, axes=[1, 2], keepdims=1))
    graph.node.append(helper.make_node("Add", [mean_name, eps_name], [denom_name], name=denom_name))
    graph.node.append(helper.make_node("Div", [proxy_name, denom_name], [norm_name], name=norm_name))
    graph.node.append(helper.make_node("Mul", [norm_name, gamma_name], [scaled_name], name=scaled_name))
    graph.node.append(helper.make_node("Add", [scaled_name, base_name], [gate_name], name=gate_name))
    graph.node.append(helper.make_node("Mul", [relevance_name, gate_name], [out_name], name=out_name))
    return out_name


def make_variant(base_model: onnx.ModelProto, output_dir: Path, tag: str, terms: list[tuple[str, float]], gamma: float) -> Path:
    model = onnx.ModelProto()
    model.CopyFrom(base_model)
    graph = model.graph
    input_name = graph.input[0].name
    prob_output, rel_output = find_outputs(model)
    prefix = f"sample_gate_{tag}"

    abs_name = add_abs_feature(graph, input_name, prefix)
    features: dict[str, str] = {"abs": abs_name}
    features["diff"] = add_group_conv_feature(graph, input_name, prefix, "diff")
    features["local3"] = add_local_feature(graph, abs_name, prefix, 3)
    features["local5"] = add_local_feature(graph, abs_name, prefix, 5)
    features["peak3"] = add_peak_feature(graph, abs_name, features["local3"], prefix)

    proxy = add_weighted_sum(graph, prefix, [(features[name], weight) for name, weight in terms])
    new_relevance = add_sample_gate(graph, prefix, rel_output.name, proxy, gamma)

    new_rel_output = onnx.ValueInfoProto()
    new_rel_output.CopyFrom(rel_output)
    new_rel_output.name = new_relevance
    graph.ClearField("output")
    graph.output.extend([prob_output, new_rel_output])
    prune_unused_graph_parts(model)

    output_path = output_dir / f"logic_timereise_{tag}.onnx"
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    return output_path


def build_specs() -> list[tuple[str, list[tuple[str, float]], float]]:
    specs: list[tuple[str, list[tuple[str, float]], float]] = []
    for feature in ("abs", "diff", "peak3", "local5"):
        for gamma in (0.03, 0.05, 0.10, 0.15):
            specs.append((f"gate_{feature}_g{int(gamma * 1000):03d}", [(feature, 1.0)], gamma))
    combos = {
        "diff_peak": [("diff", 0.6), ("peak3", 0.4)],
        "abs_diff": [("abs", 0.5), ("diff", 0.5)],
        "abs_peak": [("abs", 0.5), ("peak3", 0.5)],
        "local_peak": [("local5", 0.5), ("peak3", 0.5)],
    }
    for name, terms in combos.items():
        for gamma in (0.03, 0.05, 0.10):
            specs.append((f"gate_{name}_g{int(gamma * 1000):03d}", terms, gamma))
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample-level gate search on top of best marginal TimeREISE relevance.")
    parser.add_argument("--base-model", default="runs/candidates/logic_timereise_marginal_val5k_b10_bestproxy.onnx")
    parser.add_argument("--output-dir", default="runs/logic_timereise_sample_gate_search")
    parser.add_argument("--data-dir", default="prepared_hf_val5k")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--copy-prefix", default="logic_timereise_sample_gate")
    parser.add_argument("--no-package", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_model = onnx.load(args.base_model)
    onnx.checker.check_model(base_model)

    manifest = []
    for tag, terms, gamma in build_specs():
        model_path = output_dir / f"logic_timereise_{tag}.onnx"
        if not model_path.exists():
            model_path = make_variant(base_model, output_dir, tag, terms, gamma)
        manifest.append({"tag": tag, "model": str(model_path), "branch": "sample_gate"})

    print(f"Prepared {len(manifest)} sample-gated TimeREISE variants", flush=True)
    rows = score_and_package_manifest(manifest, output_dir, args, copy_prefix=args.copy_prefix)
    print(json.dumps({"best": sorted(rows, key=lambda item: float(item["proxy"]), reverse=True)[:5]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
