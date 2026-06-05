from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid  # type: ignore[attr-defined]

from gearxai_devkit.evaluator import evaluate_submission

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.search_logic_relevance_variants import (
    add_initializer,
    find_outputs,
    prune_unused_graph_parts,
    reduce_mean_time,
)


def ratio_weights(class_means: np.ndarray, alpha: float, clip_min: float) -> np.ndarray:
    eps = 1e-6
    global_mean = class_means.mean(axis=0, keepdims=True)
    ratio = class_means / np.maximum(global_mean, eps)
    ratio = ratio / np.maximum(ratio.mean(axis=1, keepdims=True), eps)
    matrix = 1.0 + alpha * (ratio - 1.0)
    return np.clip(matrix, clip_min, None).astype(np.float32)


def add_scalar(graph: onnx.GraphProto, name: str, value: float) -> str:
    return add_initializer(graph, name, np.asarray(value, dtype=np.float32))


def make_variant(
    base_model: onnx.ModelProto,
    output_dir: Path,
    tag: str,
    weights: np.ndarray,
    *,
    hard: bool,
    sample_channel_weight: float,
    clip_rel: float | None,
) -> Path:
    model = onnx.ModelProto()
    model.CopyFrom(base_model)
    graph = model.graph
    input_name = graph.input[0].name
    prob_output, rel_output = find_outputs(model)
    prefix = f"ratio_long_{tag}"

    abs_name = f"{prefix}_abs"
    sqrt_name = f"{prefix}_sqrt"
    graph.node.append(helper.make_node("Abs", [input_name], [abs_name], name=abs_name))
    graph.node.append(helper.make_node("Sqrt", [abs_name], [sqrt_name], name=sqrt_name))
    feature_name = sqrt_name

    if sample_channel_weight:
        channel_mean = f"{prefix}_channel_mean"
        channel_term = f"{prefix}_channel_term"
        channel_scale = f"{prefix}_channel_scale"
        scaled = f"{prefix}_channel_scaled"
        reduce_mean_time(model, abs_name, channel_mean, prefix)
        graph.node.append(helper.make_node("Mul", [sqrt_name, channel_mean], [channel_term], name=channel_term))
        weight_name = add_scalar(graph, f"{prefix}_sample_channel_weight", sample_channel_weight)
        graph.node.append(helper.make_node("Mul", [channel_term, weight_name], [scaled], name=scaled))
        graph.node.append(helper.make_node("Add", [sqrt_name, scaled], [channel_scale], name=channel_scale))
        feature_name = channel_scale

    weights_name = add_initializer(graph, f"{prefix}_weights", weights)
    class_factor_2d = f"{prefix}_class_factor_2d"
    class_factor = f"{prefix}_class_factor"
    axes_name = add_initializer(graph, f"{prefix}_unsqueeze_axes", [2], dtype=np.int64)
    if hard:
        argmax_name = f"{prefix}_argmax"
        graph.node.append(helper.make_node("ArgMax", [prob_output.name], [argmax_name], name=argmax_name, axis=1, keepdims=0))
        graph.node.append(helper.make_node("Gather", [weights_name, argmax_name], [class_factor_2d], name=class_factor_2d, axis=0))
    else:
        graph.node.append(helper.make_node("MatMul", [prob_output.name, weights_name], [class_factor_2d], name=class_factor_2d))
    graph.node.append(helper.make_node("Unsqueeze", [class_factor_2d, axes_name], [class_factor], name=class_factor))

    relevance_name = f"{prefix}_relevance_raw"
    graph.node.append(helper.make_node("Mul", [feature_name, class_factor], [relevance_name], name=relevance_name))
    final_name = relevance_name

    if clip_rel is not None:
        clip_min_name = add_scalar(graph, f"{prefix}_clip_min", 0.0)
        clip_max_name = add_scalar(graph, f"{prefix}_clip_max", clip_rel)
        final_name = f"{prefix}_relevance"
        graph.node.append(helper.make_node("Clip", [relevance_name, clip_min_name, clip_max_name], [final_name], name=final_name))

    new_rel_output = onnx.ValueInfoProto()
    new_rel_output.CopyFrom(rel_output)
    new_rel_output.name = final_name
    graph.ClearField("output")
    graph.output.extend([prob_output, new_rel_output])
    prune_unused_graph_parts(model)

    output_path = output_dir / f"logic_ratio_long_{tag}.onnx"
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    return output_path


def build_manifest(args: argparse.Namespace, output_dir: Path) -> list[dict[str, object]]:
    class_means = np.load(args.stats)
    base_model = onnx.load(args.base_model)
    manifest: list[dict[str, object]] = []

    alphas = [round(v, 3) for v in np.arange(args.alpha_min, args.alpha_max + 1e-9, args.alpha_step)]
    for alpha in alphas:
        for clip_min in args.clip_min:
            weights = ratio_weights(class_means, alpha, clip_min)
            for sample_channel_weight in args.sample_channel_weight:
                for hard in (False, True):
                    for clip_rel in args.clip_rel:
                        tag = f"a{int(round(alpha * 1000)):04d}_cm{int(round(clip_min * 100)):02d}"
                        if sample_channel_weight:
                            tag += f"_sch{int(round(sample_channel_weight * 1000)):03d}"
                        if hard:
                            tag += "_hard"
                        if clip_rel is not None:
                            tag += f"_rclip{int(round(clip_rel * 100)):03d}"

                        model_path = output_dir / f"logic_ratio_long_{tag}.onnx"
                        if not model_path.exists():
                            make_variant(
                                base_model,
                                output_dir,
                                tag,
                                weights,
                                hard=hard,
                                sample_channel_weight=sample_channel_weight,
                                clip_rel=clip_rel,
                            )
                        manifest.append(
                            {
                                "tag": tag,
                                "model": str(model_path),
                                "alpha": alpha,
                                "clip_min": clip_min,
                                "sample_channel_weight": sample_channel_weight,
                                "hard": hard,
                                "clip_rel": clip_rel,
                            }
                        )
    return manifest


def score_row(report: dict[str, object]) -> tuple[float, float, float, float, float, float]:
    faith = report["faithfulness"]  # type: ignore[index]
    classification = report["classification"]  # type: ignore[index]
    simplicity = report["simplicity"]  # type: ignore[index]
    faith_score = float(faith["faith_score"])  # type: ignore[index]
    simplicity_score = float(simplicity["simplicity_score"])  # type: ignore[index]
    proxy = 0.4 * faith_score + 0.2 * simplicity_score
    return (
        faith_score,
        float(faith["deletion_auc"]),  # type: ignore[index]
        float(faith["insertion_auc"]),  # type: ignore[index]
        float(classification["macro_f1"]),  # type: ignore[index]
        simplicity_score,
        proxy,
    )


def package_model(model_path: str, out_path: Path, args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        "-m",
        "gearxai_project.package_devkit",
        "package",
        "--model",
        model_path,
        "--data-dir",
        args.data_dir,
        "--split",
        args.split,
        "--out",
        str(out_path),
        "--batch-size",
        str(args.batch_size),
    ]
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Long local search around the best LogicLSTM ratio class relevance.")
    parser.add_argument("--base-model", default="external/gearxai_devkit/baselines/onnx/logic_lstm.onnx")
    parser.add_argument("--stats", default="runs/logic_class_xai_search/class_channel_abs_means.npy")
    parser.add_argument("--output-dir", default="runs/logic_ratio_long_search")
    parser.add_argument("--data-dir", default="prepared_hf_val5k")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--alpha-min", type=float, default=0.76)
    parser.add_argument("--alpha-max", type=float, default=0.84)
    parser.add_argument("--alpha-step", type=float, default=0.0025)
    parser.add_argument("--clip-min", type=float, nargs="+", default=[0.03, 0.05, 0.08])
    parser.add_argument("--sample-channel-weight", type=float, nargs="+", default=[0.0, 0.02, 0.04, 0.06])
    parser.add_argument("--clip-rel", type=float, nargs="+", default=[None])
    parser.add_argument("--copy-prefix", default="logic_ratio_long")
    parser.add_argument("--no-package", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    eval_dir = output_dir / "eval"
    candidate_dir = Path("runs/candidates")
    final_dir = Path("runs/final_candidates")
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(args, output_dir)
    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"Prepared {len(manifest)} variants under {output_dir}", flush=True)

    rows = []
    for index, row in enumerate(manifest, 1):
        tag = str(row["tag"])
        report_path = eval_dir / f"{tag}.json"
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
        else:
            report = evaluate_submission(
                model_path=str(row["model"]),
                data_dir=args.data_dir,
                split=args.split,
                batch_size=args.batch_size,
                output_path=str(report_path),
            )
        faith, deletion, insertion, f1, simplicity, proxy = score_row(report)
        rows.append(
            {
                **row,
                "report": str(report_path),
                "faith": faith,
                "deletion": deletion,
                "insertion": insertion,
                "f1": f1,
                "simplicity": simplicity,
                "proxy": proxy,
            }
        )
        print(
            f"{index}/{len(manifest)} {tag} faith={faith:.6f} del={deletion:.6f} "
            f"ins={insertion:.6f} f1={f1:.6f} simp={simplicity:.6f} proxy={proxy:.6f}",
            flush=True,
        )

    rows_by_proxy = sorted(rows, key=lambda item: float(item["proxy"]), reverse=True)
    rows_by_faith = sorted(rows, key=lambda item: float(item["faith"]), reverse=True)
    summary_path = output_dir / "summary_top.json"
    summary_path.write_text(
        json.dumps({"best_proxy": rows_by_proxy[:20], "best_faith": rows_by_faith[:20]}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    for kind, row in (("bestproxy", rows_by_proxy[0]), ("bestfaith", rows_by_faith[0])):
        model_src = Path(str(row["model"]))
        report_src = Path(str(row["report"]))
        model_dst = candidate_dir / f"{args.copy_prefix}_{kind}.onnx"
        report_dst = candidate_dir / f"{args.copy_prefix}_{kind}_devkit.json"
        shutil.copy2(model_src, model_dst)
        shutil.copy2(report_src, report_dst)
        print(f"Copied {kind}: {model_dst}", flush=True)
        if not args.no_package:
            package_dst = final_dir / f"{args.copy_prefix}_{kind}_submission.zip"
            package_model(str(model_dst), package_dst, args)
            print(f"Packaged {kind}: {package_dst}", flush=True)

    print(f"Wrote summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
