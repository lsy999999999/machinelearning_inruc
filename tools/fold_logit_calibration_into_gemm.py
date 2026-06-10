from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def initializer_map(model: onnx.ModelProto) -> dict[str, onnx.TensorProto]:
    return {initializer.name: initializer for initializer in model.graph.initializer}


def node_by_output(model: onnx.ModelProto) -> dict[str, onnx.NodeProto]:
    return {output: node for node in model.graph.node for output in node.output}


def array_for(initializers: dict[str, onnx.TensorProto], name: str) -> np.ndarray | None:
    if name not in initializers:
        return None
    return numpy_helper.to_array(initializers[name]).astype(np.float32)


def fold_path(
    model: onnx.ModelProto,
    class_count: int,
) -> tuple[onnx.NodeProto, np.ndarray, np.ndarray, set[str], set[str]]:
    initializers = initializer_map(model)
    producers = node_by_output(model)
    softmax_nodes = [node for node in model.graph.node if node.op_type == "Softmax" and "probabilities" in node.output]
    if len(softmax_nodes) != 1:
        raise RuntimeError(f"Expected one probabilities Softmax, found {len(softmax_nodes)}")

    scale = np.ones((class_count,), dtype=np.float32)
    bias = np.zeros((class_count,), dtype=np.float32)
    removed_nodes: set[str] = set()
    removed_initializers: set[str] = set()

    current = softmax_nodes[0].input[0]
    while current in producers:
        node = producers[current]
        if node.op_type == "Gemm":
            softmax_nodes[0].input[0] = node.output[0]
            return node, scale, bias, removed_nodes, removed_initializers
        if node.op_type not in {"Add", "Mul"}:
            break

        if len(node.input) != 2:
            break
        left, right = node.input
        left_array = array_for(initializers, left)
        right_array = array_for(initializers, right)
        if left_array is not None and right_array is None:
            const_name = left
            const = left_array
            previous = right
        elif right_array is not None and left_array is None:
            const_name = right
            const = right_array
            previous = left
        else:
            break

        const = const.reshape(-1)
        if const.shape != (class_count,):
            break
        if node.op_type == "Add":
            bias = bias + const * scale
        else:
            scale = scale * const
        removed_nodes.add(node.name)
        removed_initializers.add(const_name)
        current = previous

    raise RuntimeError("Could not trace Add/Mul calibration path back to Gemm.")


def fold_into_gemm(gemm: onnx.NodeProto, model: onnx.ModelProto, scale: np.ndarray, bias_delta: np.ndarray) -> None:
    initializers = initializer_map(model)
    weight_name = gemm.input[1]
    bias_name = gemm.input[2]
    weights = numpy_helper.to_array(initializers[weight_name]).astype(np.float32)
    bias = numpy_helper.to_array(initializers[bias_name]).astype(np.float32)
    attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in gemm.attribute}
    trans_b = int(attrs.get("transB", 0))

    if trans_b:
        if weights.shape[0] != scale.shape[0]:
            raise RuntimeError(f"Unexpected transposed Gemm weight shape: {weights.shape}")
        folded_weights = weights * scale[:, None]
    else:
        if weights.shape[-1] != scale.shape[0]:
            raise RuntimeError(f"Unexpected Gemm weight shape: {weights.shape}")
        folded_weights = weights * scale[None, :]
    folded_bias = bias * scale + bias_delta

    initializers[weight_name].CopyFrom(numpy_helper.from_array(folded_weights.astype(np.float32), name=weight_name))
    initializers[bias_name].CopyFrom(numpy_helper.from_array(folded_bias.astype(np.float32), name=bias_name))


def prune_removed(model: onnx.ModelProto, removed_nodes: set[str], removed_initializers: set[str]) -> None:
    graph = model.graph
    keep_nodes = [node for node in graph.node if node.name not in removed_nodes]
    graph.ClearField("node")
    graph.node.extend(keep_nodes)

    keep_initializers = [initializer for initializer in graph.initializer if initializer.name not in removed_initializers]
    graph.ClearField("initializer")
    graph.initializer.extend(keep_initializers)

    live_outputs = {output for node in graph.node for output in node.output} | {output.name for output in graph.output}
    keep_value_info = [value_info for value_info in graph.value_info if value_info.name in live_outputs]
    graph.ClearField("value_info")
    graph.value_info.extend(keep_value_info)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fold vector Add/Mul logit calibration into the final Gemm.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--class-count", type=int, default=9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = onnx.load(args.model)
    gemm, scale, bias, removed_nodes, removed_initializers = fold_path(model, args.class_count)
    fold_into_gemm(gemm, model, scale, bias)
    prune_removed(model, removed_nodes, removed_initializers)
    onnx.checker.check_model(model)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output)
    print(f"Wrote {output}")
    print(f"folded_nodes={sorted(removed_nodes)}")
    print(f"folded_initializers={sorted(removed_initializers)}")


if __name__ == "__main__":
    main()
