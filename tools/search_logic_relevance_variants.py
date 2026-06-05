from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import helper, numpy_helper


def tensor_shape(value_info: onnx.ValueInfoProto) -> list[int | None]:
    dims: list[int | None] = []
    for dim in value_info.type.tensor_type.shape.dim:
        dims.append(dim.dim_value if dim.HasField("dim_value") else None)
    return dims


def find_outputs(model: onnx.ModelProto) -> tuple[onnx.ValueInfoProto, onnx.ValueInfoProto]:
    prob_output = None
    rel_output = None
    for output in model.graph.output:
        shape = tensor_shape(output)
        if len(shape) == 2 and shape[-1] == 9:
            prob_output = output
        elif len(shape) == 3 and shape[-2:] == [8, 100]:
            rel_output = output
    if prob_output is None or rel_output is None:
        observed = [(output.name, tensor_shape(output)) for output in model.graph.output]
        raise RuntimeError(f"Cannot identify probability/relevance outputs. Observed: {observed}")
    return prob_output, rel_output


def add_initializer(graph: onnx.GraphProto, name: str, value: Any, dtype=np.float32) -> str:
    graph.initializer.append(numpy_helper.from_array(np.asarray(value, dtype=dtype), name=name))
    return name


def add_scalar(graph: onnx.GraphProto, name: str, value: float) -> str:
    return add_initializer(graph, name, np.array(value, dtype=np.float32))


def add_weighted_term(
    graph: onnx.GraphProto,
    prefix: str,
    current: str | None,
    term: str,
    weight: float,
    index: int,
) -> str | None:
    if weight == 0.0:
        return current
    weight_name = add_scalar(graph, f"{prefix}_w_{index}", weight)
    scaled_name = f"{prefix}_scaled_{index}"
    graph.node.append(helper.make_node("Mul", [term, weight_name], [scaled_name], name=scaled_name))
    if current is None:
        return scaled_name
    out_name = f"{prefix}_sum_{index}"
    graph.node.append(helper.make_node("Add", [current, scaled_name], [out_name], name=out_name))
    return out_name


def reduce_mean_time(model: onnx.ModelProto, value: str, output: str, prefix: str) -> None:
    graph = model.graph
    opset = max(imp.version for imp in model.opset_import if imp.domain in ("", "ai.onnx"))
    if opset >= 18:
        axes = add_initializer(graph, f"{prefix}_mean_axes", [2], dtype=np.int64)
        graph.node.append(helper.make_node("ReduceMean", [value, axes], [output], name=output, keepdims=1))
    else:
        graph.node.append(helper.make_node("ReduceMean", [value], [output], name=output, axes=[2], keepdims=1))


def add_diff_feature(graph: onnx.GraphProto, input_name: str, prefix: str) -> str:
    starts_prev = add_initializer(graph, f"{prefix}_starts_prev", [0], dtype=np.int64)
    ends_prev = add_initializer(graph, f"{prefix}_ends_prev", [99], dtype=np.int64)
    starts_next = add_initializer(graph, f"{prefix}_starts_next", [1], dtype=np.int64)
    ends_next = add_initializer(graph, f"{prefix}_ends_next", [100], dtype=np.int64)
    axes = add_initializer(graph, f"{prefix}_slice_axes", [2], dtype=np.int64)
    steps = add_initializer(graph, f"{prefix}_slice_steps", [1], dtype=np.int64)

    prev_name = f"{prefix}_prev"
    next_name = f"{prefix}_next"
    diff_raw_name = f"{prefix}_diff_raw"
    diff_abs_name = f"{prefix}_diff_abs"
    diff_pad_name = f"{prefix}_diff_pad"

    graph.node.append(helper.make_node("Slice", [input_name, starts_prev, ends_prev, axes, steps], [prev_name], name=prev_name))
    graph.node.append(helper.make_node("Slice", [input_name, starts_next, ends_next, axes, steps], [next_name], name=next_name))
    graph.node.append(helper.make_node("Sub", [next_name, prev_name], [diff_raw_name], name=diff_raw_name))
    graph.node.append(helper.make_node("Abs", [diff_raw_name], [diff_abs_name], name=diff_abs_name))

    pads = add_initializer(graph, f"{prefix}_pads", [0, 0, 0, 0, 0, 1], dtype=np.int64)
    zero = add_initializer(graph, f"{prefix}_zero", 0.0)
    graph.node.append(helper.make_node("Pad", [diff_abs_name, pads, zero], [diff_pad_name], name=diff_pad_name, mode="constant"))
    return diff_pad_name


def make_variant(base_model: onnx.ModelProto, output_dir: Path, spec: dict[str, Any]) -> Path:
    model = copy.deepcopy(base_model)
    graph = model.graph
    input_name = graph.input[0].name
    prob_output, rel_output = find_outputs(model)
    tag = str(spec["tag"])
    prefix = f"xai_{tag}"

    abs_name = f"{prefix}_abs"
    graph.node.append(helper.make_node("Abs", [input_name], [abs_name], name=abs_name))

    features: dict[str, str] = {"abs": abs_name}

    square_name = f"{prefix}_square"
    graph.node.append(helper.make_node("Mul", [abs_name, abs_name], [square_name], name=square_name))
    features["square"] = square_name

    sqrt_name = f"{prefix}_sqrt"
    graph.node.append(helper.make_node("Sqrt", [abs_name], [sqrt_name], name=sqrt_name))
    features["sqrt"] = sqrt_name

    for kernel in (3, 5, 7):
        local_name = f"{prefix}_local{kernel}"
        pad = kernel // 2
        graph.node.append(
            helper.make_node(
                "AveragePool",
                [abs_name],
                [local_name],
                name=local_name,
                kernel_shape=[kernel],
                strides=[1],
                pads=[pad, pad],
                count_include_pad=0,
            )
        )
        features[f"local{kernel}"] = local_name

        peak_raw = f"{prefix}_peak{kernel}_raw"
        peak_name = f"{prefix}_peak{kernel}"
        graph.node.append(helper.make_node("Sub", [abs_name, local_name], [peak_raw], name=peak_raw))
        graph.node.append(helper.make_node("Relu", [peak_raw], [peak_name], name=peak_name))
        features[f"peak{kernel}"] = peak_name

    diff_name = add_diff_feature(graph, input_name, prefix)
    features["diff"] = diff_name

    channel_mean_name = f"{prefix}_channel_mean"
    reduce_mean_time(model, abs_name, channel_mean_name, prefix)
    for source in ("abs", "square", "sqrt", "local3", "local5", "diff"):
        channel_name = f"{prefix}_{source}_channel"
        graph.node.append(helper.make_node("Mul", [features[source], channel_mean_name], [channel_name], name=channel_name))
        features[f"{source}_channel"] = channel_name

    final_name: str | None = None
    for index, (feature_name, weight) in enumerate(spec["terms"], start=1):
        final_name = add_weighted_term(graph, prefix, final_name, features[feature_name], float(weight), index)
    if final_name is None:
        raise ValueError(f"Variant {tag} has no nonzero terms.")

    new_rel_output = copy.deepcopy(rel_output)
    new_rel_output.name = final_name
    graph.ClearField("output")
    graph.output.extend([copy.deepcopy(prob_output), new_rel_output])
    prune_unused_graph_parts(model)

    output_path = output_dir / f"logic_search_{tag}.onnx"
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    return output_path


def prune_unused_graph_parts(model: onnx.ModelProto) -> None:
    graph = model.graph
    required = {output.name for output in graph.output}
    kept_reversed = []
    for node in reversed(graph.node):
        if any(output in required for output in node.output):
            kept_reversed.append(node)
            required.update(input_name for input_name in node.input if input_name)

    kept_nodes = list(reversed(kept_reversed))
    kept_node_outputs = {output for node in kept_nodes for output in node.output}
    graph.ClearField("node")
    graph.node.extend(kept_nodes)

    graph_inputs = {value.name for value in graph.input}
    required_initializers = required - graph_inputs - kept_node_outputs
    kept_initializers = [initializer for initializer in graph.initializer if initializer.name in required_initializers]
    graph.ClearField("initializer")
    graph.initializer.extend(kept_initializers)


def build_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []

    def add(tag: str, terms: list[tuple[str, float]]) -> None:
        specs.append({"tag": tag, "terms": terms})

    for feature in ("abs", "square", "sqrt", "local3", "local5", "local7", "diff", "abs_channel", "square_channel", "sqrt_channel", "local3_channel", "local5_channel", "diff_channel"):
        add(feature, [(feature, 1.0)])

    for channel_weight in (0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 5.00):
        tag = f"abs_ch{int(channel_weight * 100):03d}"
        add(tag, [("abs", 1.0), ("abs_channel", channel_weight)])

    for base in ("abs", "sqrt", "local3", "local5"):
        for channel_weight in (0.25, 0.50, 1.00, 2.00):
            for peak_weight in (0.05, 0.10, 0.20):
                tag = f"{base}_ch{int(channel_weight * 100):03d}_pk{int(peak_weight * 100):02d}"
                add(tag, [(base, 1.0), ("abs_channel", channel_weight), ("peak3", peak_weight)])

    for base in ("abs", "sqrt", "local3"):
        for channel_weight in (0.25, 0.50, 1.00):
            for diff_weight in (0.02, 0.05, 0.10):
                tag = f"{base}_ch{int(channel_weight * 100):03d}_df{int(diff_weight * 100):02d}"
                add(tag, [(base, 1.0), ("abs_channel", channel_weight), ("diff", diff_weight)])

    for square_weight in (0.05, 0.10, 0.20, 0.35):
        for channel_weight in (0.25, 0.50, 1.00):
            tag = f"abs_sq{int(square_weight * 100):02d}_ch{int(channel_weight * 100):03d}"
            add(tag, [("abs", 1.0), ("square", square_weight), ("abs_channel", channel_weight)])

    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ONNX relevance variants for official LogicLSTM.")
    parser.add_argument("--base-model", default="external/gearxai_devkit/baselines/onnx/logic_lstm.onnx")
    parser.add_argument("--output-dir", default="runs/logic_xai_search")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_model = onnx.load(args.base_model)
    specs = build_specs()
    if args.limit is not None:
        specs = specs[: args.limit]

    manifest = []
    for spec in specs:
        output_path = make_variant(base_model, output_dir, spec)
        manifest.append({"model": str(output_path), **spec})
        print(f"Saved {output_path}")

    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"Wrote {output_dir / 'manifest.jsonl'} ({len(manifest)} variants)")


if __name__ == "__main__":
    main()
