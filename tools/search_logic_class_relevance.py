from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import helper
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from gearxai_project.data import GearXAIWindows
from gearxai_project.utils import load_config

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.search_logic_relevance_variants import (
    add_initializer,
    add_weighted_term,
    find_outputs,
    prune_unused_graph_parts,
    reduce_mean_time,
)


def compute_class_channel_stats(config_path: str, split: str, max_samples: int | None, batch_size: int) -> np.ndarray:
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    dataset = GearXAIWindows(
        split=split,
        dataset_name=data_cfg["dataset_name"],
        config_name=data_cfg["config_name"],
        cache_dir=data_cfg.get("cache_dir"),
        max_samples=max_samples,
        normalize=False,
        seed=int(cfg["training"].get("seed", 42)),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    sums = np.zeros((9, 8), dtype=np.float64)
    counts = np.zeros((9,), dtype=np.float64)
    for x, y in tqdm(loader, desc=f"stats-{split}", leave=False):
        channel_abs = x.abs().mean(dim=2).numpy()
        labels = y.numpy().astype(np.int64)
        for class_id in range(9):
            mask = labels == class_id
            if np.any(mask):
                sums[class_id] += channel_abs[mask].sum(axis=0)
                counts[class_id] += float(mask.sum())
    means = sums / np.maximum(counts[:, None], 1.0)
    return means.astype(np.float32)


def build_weight_matrices(class_means: np.ndarray) -> dict[str, np.ndarray]:
    eps = 1e-6
    global_mean = class_means.mean(axis=0, keepdims=True)
    ratio = class_means / np.maximum(global_mean, eps)
    ratio = ratio / np.maximum(ratio.mean(axis=1, keepdims=True), eps)

    centered = class_means / np.maximum(class_means.mean(axis=1, keepdims=True), eps)
    centered = centered / np.maximum(centered.mean(axis=1, keepdims=True), eps)

    z = class_means - class_means.mean(axis=1, keepdims=True)
    z = z / np.maximum(class_means.std(axis=1, keepdims=True), eps)

    matrices: dict[str, np.ndarray] = {}
    for name, base in {"ratio": ratio, "center": centered}.items():
        for alpha in (0.10, 0.20, 0.35, 0.50, 0.75, 1.00):
            matrix = 1.0 + alpha * (base - 1.0)
            matrices[f"{name}_a{int(alpha * 100):03d}"] = np.clip(matrix, 0.05, None).astype(np.float32)
    for alpha in (0.05, 0.10, 0.20, 0.35):
        matrix = 1.0 + alpha * z
        matrices[f"z_a{int(alpha * 100):03d}"] = np.clip(matrix, 0.05, None).astype(np.float32)
    return matrices


def make_variant(
    base_model: onnx.ModelProto,
    output_dir: Path,
    tag: str,
    class_channel_weights: np.ndarray,
    *,
    sample_channel_weight: float = 0.0,
    use_sqrt: bool = False,
) -> Path:
    model = onnx.ModelProto()
    model.CopyFrom(base_model)
    graph = model.graph
    input_name = graph.input[0].name
    prob_output, rel_output = find_outputs(model)
    prefix = f"class_xai_{tag}"

    abs_name = f"{prefix}_abs"
    graph.node.append(helper.make_node("Abs", [input_name], [abs_name], name=abs_name))
    base_feature = abs_name
    if use_sqrt:
        sqrt_name = f"{prefix}_sqrt"
        graph.node.append(helper.make_node("Sqrt", [abs_name], [sqrt_name], name=sqrt_name))
        base_feature = sqrt_name

    combined_feature: str | None = base_feature
    if sample_channel_weight != 0.0:
        channel_mean = f"{prefix}_channel_mean"
        channel_term = f"{prefix}_channel_term"
        reduce_mean_time(model, abs_name, channel_mean, prefix)
        graph.node.append(helper.make_node("Mul", [base_feature, channel_mean], [channel_term], name=channel_term))
        combined_feature = add_weighted_term(
            graph,
            prefix,
            base_feature,
            channel_term,
            sample_channel_weight,
            1,
        )

    weights_name = add_initializer(graph, f"{prefix}_weights", class_channel_weights.astype(np.float32))
    class_factor_2d = f"{prefix}_class_factor_2d"
    class_factor = f"{prefix}_class_factor"
    axes_name = add_initializer(graph, f"{prefix}_unsqueeze_axes", [2], dtype=np.int64)
    graph.node.append(helper.make_node("MatMul", [prob_output.name, weights_name], [class_factor_2d], name=class_factor_2d))
    graph.node.append(helper.make_node("Unsqueeze", [class_factor_2d, axes_name], [class_factor], name=class_factor))

    relevance_name = f"{prefix}_relevance"
    graph.node.append(helper.make_node("Mul", [combined_feature, class_factor], [relevance_name], name=relevance_name))

    new_rel_output = onnx.ValueInfoProto()
    new_rel_output.CopyFrom(rel_output)
    new_rel_output.name = relevance_name
    graph.ClearField("output")
    graph.output.extend([prob_output, new_rel_output])
    prune_unused_graph_parts(model)

    output_path = output_dir / f"logic_class_{tag}.onnx"
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate class-specific LogicLSTM relevance variants.")
    parser.add_argument("--base-model", default="external/gearxai_devkit/baselines/onnx/logic_lstm.onnx")
    parser.add_argument("--config", default="configs/spectral_lite_c.yaml")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--output-dir", default="runs/logic_class_xai_search")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    class_means = compute_class_channel_stats(args.config, args.split, args.max_samples, args.batch_size)
    np.save(output_dir / "class_channel_abs_means.npy", class_means)
    matrices = build_weight_matrices(class_means)
    base_model = onnx.load(args.base_model)

    manifest = []
    for matrix_name, matrix in matrices.items():
        for sample_channel_weight in (0.0, 0.10, 0.20, 0.35):
            for use_sqrt in (False, True):
                tag = matrix_name
                if sample_channel_weight:
                    tag += f"_sch{int(sample_channel_weight * 100):03d}"
                if use_sqrt:
                    tag += "_sqrt"
                model_path = make_variant(
                    base_model,
                    output_dir,
                    tag,
                    matrix,
                    sample_channel_weight=sample_channel_weight,
                    use_sqrt=use_sqrt,
                )
                row = {
                    "model": str(model_path),
                    "tag": tag,
                    "matrix": matrix_name,
                    "sample_channel_weight": sample_channel_weight,
                    "use_sqrt": use_sqrt,
                }
                manifest.append(row)
                print(f"Saved {model_path}")
    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"Wrote {output_dir / 'manifest.jsonl'} ({len(manifest)} variants)")


if __name__ == "__main__":
    main()
