from __future__ import annotations

import copy
import shutil
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto


BASE_MODEL = Path("external/gearxai_devkit/baselines/onnx/logic_lstm.onnx")
OUT_DIR = Path("runs/logic_xai_variants")


def tensor_shape(value_info):
    dims = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(dim.dim_value)
        else:
            dims.append(None)
    return dims


def find_outputs(model):
    prob_output = None
    rel_output = None

    for output in model.graph.output:
        shape = tensor_shape(output)
        if len(shape) == 2 and shape[-1] == 9:
            prob_output = output
        elif len(shape) == 3 and shape[-2:] == [8, 100]:
            rel_output = output

    if prob_output is None or rel_output is None:
        observed = [(o.name, tensor_shape(o)) for o in model.graph.output]
        raise RuntimeError(f"Cannot identify official outputs. Observed: {observed}")

    return prob_output, rel_output


def add_initializer(graph, name: str, value, dtype=np.float32):
    arr = np.asarray(value, dtype=dtype)
    graph.initializer.append(numpy_helper.from_array(arr, name=name))
    return name


def add_scalar(graph, name: str, value: float):
    return add_initializer(graph, name, np.array(value, dtype=np.float32))


def weighted_add(graph, prefix: str, current: str, term: str, weight: float, idx: int):
    if weight == 0:
        return current
    weight_name = add_scalar(graph, f"{prefix}_weight_{idx}", weight)
    scaled_name = f"{prefix}_scaled_{idx}"
    out_name = f"{prefix}_sum_{idx}"
    graph.node.append(helper.make_node("Mul", [term, weight_name], [scaled_name], name=scaled_name))
    graph.node.append(helper.make_node("Add", [current, scaled_name], [out_name], name=out_name))
    return out_name


def make_variant(
    base: onnx.ModelProto,
    tag: str,
    *,
    local_weight: float = 0.0,
    diff_weight: float = 0.0,
    peak_weight: float = 0.0,
    channel_weight: float = 0.0,
):
    model = copy.deepcopy(base)
    graph = model.graph
    input_name = graph.input[0].name
    prob_output, rel_output = find_outputs(model)
    prefix = f"xai_{tag}"

    abs_name = f"{prefix}_abs"
    graph.node.append(helper.make_node("Abs", [input_name], [abs_name], name=abs_name))

    final_name = abs_name
    term_index = 0

    local_name = None
    if local_weight != 0.0 or peak_weight != 0.0:
        local_name = f"{prefix}_local3"
        graph.node.append(
            helper.make_node(
                "AveragePool",
                [abs_name],
                [local_name],
                name=local_name,
                kernel_shape=[3],
                strides=[1],
                pads=[1, 1],
                count_include_pad=0,
            )
        )

    if local_weight != 0.0:
        term_index += 1
        final_name = weighted_add(graph, prefix, final_name, local_name, local_weight, term_index)

    if diff_weight != 0.0:
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

        graph.node.append(
            helper.make_node(
                "Slice",
                [input_name, starts_prev, ends_prev, axes, steps],
                [prev_name],
                name=prev_name,
            )
        )
        graph.node.append(
            helper.make_node(
                "Slice",
                [input_name, starts_next, ends_next, axes, steps],
                [next_name],
                name=next_name,
            )
        )
        graph.node.append(helper.make_node("Sub", [next_name, prev_name], [diff_raw_name], name=diff_raw_name))
        graph.node.append(helper.make_node("Abs", [diff_raw_name], [diff_abs_name], name=diff_abs_name))

        pads = add_initializer(graph, f"{prefix}_pads", [0, 0, 0, 0, 0, 1], dtype=np.int64)
        zero = add_initializer(graph, f"{prefix}_zero", np.array(0.0, dtype=np.float32))
        graph.node.append(
            helper.make_node(
                "Pad",
                [diff_abs_name, pads, zero],
                [diff_pad_name],
                name=diff_pad_name,
                mode="constant",
            )
        )

        term_index += 1
        final_name = weighted_add(graph, prefix, final_name, diff_pad_name, diff_weight, term_index)

    if peak_weight != 0.0:
        peak_raw_name = f"{prefix}_peak_raw"
        peak_name = f"{prefix}_peak"
        graph.node.append(helper.make_node("Sub", [abs_name, local_name], [peak_raw_name], name=peak_raw_name))
        graph.node.append(helper.make_node("Relu", [peak_raw_name], [peak_name], name=peak_name))

        term_index += 1
        final_name = weighted_add(graph, prefix, final_name, peak_name, peak_weight, term_index)

    if channel_weight != 0.0:
        opset = max(
            imp.version for imp in model.opset_import
            if imp.domain in ("", "ai.onnx")
        )
        channel_mean_name = f"{prefix}_channel_mean"
        channel_term_name = f"{prefix}_channel_term"

        if opset >= 18:
            reduce_axes = add_initializer(graph, f"{prefix}_reduce_axes", [2], dtype=np.int64)
            graph.node.append(
                helper.make_node(
                    "ReduceMean",
                    [abs_name, reduce_axes],
                    [channel_mean_name],
                    name=channel_mean_name,
                    keepdims=1,
                )
            )
        else:
            graph.node.append(
                helper.make_node(
                    "ReduceMean",
                    [abs_name],
                    [channel_mean_name],
                    name=channel_mean_name,
                    axes=[2],
                    keepdims=1,
                )
            )

        graph.node.append(
            helper.make_node(
                "Mul",
                [abs_name, channel_mean_name],
                [channel_term_name],
                name=channel_term_name,
            )
        )

        term_index += 1
        final_name = weighted_add(graph, prefix, final_name, channel_term_name, channel_weight, term_index)

    new_rel_output = copy.deepcopy(rel_output)
    new_rel_output.name = final_name

    graph.ClearField("output")
    graph.output.extend([copy.deepcopy(prob_output), new_rel_output])

    out_path = OUT_DIR / f"logic_{tag}.onnx"
    onnx.checker.check_model(model)
    onnx.save(model, out_path)
    print(f"Saved {out_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = onnx.load(BASE_MODEL)

    shutil.copy2(BASE_MODEL, OUT_DIR / "logic_original.onnx")
    print(f"Copied {OUT_DIR / 'logic_original.onnx'}")

    variants = [
        ("abs_only", {}),
        ("diff10", {"diff_weight": 0.10}),
        ("diff25", {"diff_weight": 0.25}),
        ("diff50", {"diff_weight": 0.50}),
        ("peak25", {"peak_weight": 0.25}),
        ("peak50", {"peak_weight": 0.50}),
        ("channel25", {"channel_weight": 0.25}),
        ("channel50", {"channel_weight": 0.50}),
        ("diff25_peak25", {"diff_weight": 0.25, "peak_weight": 0.25}),
        ("diff25_channel25", {"diff_weight": 0.25, "channel_weight": 0.25}),
        ("local25_diff25", {"local_weight": 0.25, "diff_weight": 0.25}),
    ]

    for tag, kwargs in variants:
        make_variant(base, tag, **kwargs)


if __name__ == "__main__":
    main()
