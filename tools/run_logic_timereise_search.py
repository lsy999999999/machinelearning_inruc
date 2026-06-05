from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import helper
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gearxai_project.data import GearXAIWindows
from gearxai_project.utils import load_config
from tools.search_logic_relevance_variants import add_initializer, find_outputs, prune_unused_graph_parts


def load_dataset(config_path: str, split: str, max_samples: int | None, batch_size: int) -> DataLoader:
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    dataset = GearXAIWindows(
        split=split,
        dataset_name=data_cfg["dataset_name"],
        config_name=data_cfg["config_name"],
        cache_dir=data_cfg.get("cache_dir"),
        max_samples=max_samples,
        normalize=True,
        seed=int(cfg["training"].get("seed", 42)) + 17,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)


def time_blocks(num_bins: int, length: int = 100) -> list[tuple[int, int]]:
    edges = np.linspace(0, length, num_bins + 1).round().astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(num_bins)]


def stats_path(output_dir: Path) -> Path:
    return output_dir / "timereise_stats.npz"


def compute_stats(args: argparse.Namespace, output_dir: Path) -> None:
    path = stats_path(output_dir)
    if path.exists():
        print(f"Stats already exist: {path}", flush=True)
        return

    loader = load_dataset(args.config, args.split, args.max_samples, args.stats_batch_size)
    session = ort.InferenceSession(args.base_model, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    prob_output_name = None
    for output in session.get_outputs():
        shape = output.shape
        if len(shape) == 2 and shape[-1] == 9:
            prob_output_name = output.name
            break
    if prob_output_name is None:
        raise RuntimeError("Could not identify probability output.")
    prob_index = output_names.index(prob_output_name)

    blocks = time_blocks(args.num_bins)
    drop_sums = np.zeros((9, 8, args.num_bins), dtype=np.float64)
    keep_sums = np.zeros((9, 8, args.num_bins), dtype=np.float64)
    counts = np.zeros((9,), dtype=np.float64)
    processed = 0

    for x, _y in tqdm(loader, desc="timereise-stats", leave=True):
        x_np = x.numpy().astype(np.float32)
        probabilities = session.run(output_names, {input_name: x_np})[prob_index]
        pred = probabilities.argmax(axis=1).astype(np.int64)
        base_conf = probabilities[np.arange(probabilities.shape[0]), pred]
        for class_id in range(9):
            counts[class_id] += float((pred == class_id).sum())

        for channel in range(8):
            for bin_id, (start, end) in enumerate(blocks):
                deleted = x_np.copy()
                deleted[:, channel, start:end] = 0.0
                deleted_prob = session.run(output_names, {input_name: deleted})[prob_index]
                deleted_conf = deleted_prob[np.arange(deleted_prob.shape[0]), pred]
                drop = np.maximum(base_conf - deleted_conf, 0.0)

                kept = np.zeros_like(x_np)
                kept[:, channel, start:end] = x_np[:, channel, start:end]
                kept_prob = session.run(output_names, {input_name: kept})[prob_index]
                keep_conf = kept_prob[np.arange(kept_prob.shape[0]), pred]

                for class_id in range(9):
                    mask = pred == class_id
                    if np.any(mask):
                        drop_sums[class_id, channel, bin_id] += float(drop[mask].sum())
                        keep_sums[class_id, channel, bin_id] += float(keep_conf[mask].sum())
        processed += int(x_np.shape[0])

    np.savez_compressed(
        path,
        drop_sums=drop_sums,
        keep_sums=keep_sums,
        counts=counts,
        processed=np.asarray(processed, dtype=np.int64),
        num_bins=np.asarray(args.num_bins, dtype=np.int64),
        blocks=np.asarray(blocks, dtype=np.int64),
    )
    print(f"Saved stats: {path} processed={processed}", flush=True)


def normalize_importance(values: np.ndarray) -> np.ndarray:
    eps = 1e-6
    centered = values / np.maximum(values.mean(axis=(1, 2), keepdims=True), eps)
    return np.clip(centered, 0.05, 10.0).astype(np.float32)


def expand_bins(values: np.ndarray, blocks: np.ndarray, length: int = 100) -> np.ndarray:
    out = np.zeros((values.shape[0], values.shape[1], length), dtype=np.float32)
    for bin_id, (start, end) in enumerate(blocks.tolist()):
        out[:, :, int(start) : int(end)] = values[:, :, bin_id : bin_id + 1]
    return out


def ratio_channel_weights(class_means_path: str, alpha: float) -> np.ndarray:
    class_means = np.load(class_means_path)
    eps = 1e-6
    global_mean = class_means.mean(axis=0, keepdims=True)
    ratio = class_means / np.maximum(global_mean, eps)
    ratio = ratio / np.maximum(ratio.mean(axis=1, keepdims=True), eps)
    return np.clip(1.0 + alpha * (ratio - 1.0), 0.05, None).astype(np.float32)


def build_weight_specs(args: argparse.Namespace, output_dir: Path) -> list[tuple[str, np.ndarray]]:
    stats = np.load(stats_path(output_dir))
    counts = np.maximum(stats["counts"].astype(np.float64), 1.0)
    drop = stats["drop_sums"].astype(np.float64) / counts[:, None, None]
    keep = stats["keep_sums"].astype(np.float64) / counts[:, None, None]
    blocks = stats["blocks"]

    modes: dict[str, np.ndarray] = {
        "drop": normalize_importance(drop),
        "keep": normalize_importance(keep),
        "mix50": normalize_importance(0.5 * normalize_importance(drop) + 0.5 * normalize_importance(keep)),
    }
    ratio = ratio_channel_weights(args.class_channel_stats, args.ratio_alpha)[:, :, None]

    specs: list[tuple[str, np.ndarray]] = []
    for mode_name, binned in modes.items():
        expanded = expand_bins(binned, blocks)
        expanded = expanded / np.maximum(expanded.mean(axis=(1, 2), keepdims=True), 1e-6)
        for beta in args.time_beta:
            time_factor = np.clip(1.0 + beta * (expanded - 1.0), 0.05, None)
            pure = time_factor / np.maximum(time_factor.mean(axis=(1, 2), keepdims=True), 1e-6)
            specs.append((f"{mode_name}_tb{int(beta * 100):03d}", pure.astype(np.float32)))

            fused = ratio * time_factor
            fused = fused / np.maximum(fused.mean(axis=(1, 2), keepdims=True), 1e-6)
            specs.append((f"ratio{int(args.ratio_alpha * 1000):04d}_{mode_name}_tb{int(beta * 100):03d}", fused.astype(np.float32)))
    return specs


def make_variant(base_model: onnx.ModelProto, output_dir: Path, tag: str, weights_9x8x100: np.ndarray, hard: bool) -> Path:
    model = onnx.ModelProto()
    model.CopyFrom(base_model)
    graph = model.graph
    input_name = graph.input[0].name
    prob_output, rel_output = find_outputs(model)
    suffix = f"{tag}_hard" if hard else tag
    prefix = f"timereise_{suffix}"

    abs_name = f"{prefix}_abs"
    sqrt_name = f"{prefix}_sqrt"
    graph.node.append(helper.make_node("Abs", [input_name], [abs_name], name=abs_name))
    graph.node.append(helper.make_node("Sqrt", [abs_name], [sqrt_name], name=sqrt_name))

    if hard:
        weights_name = add_initializer(graph, f"{prefix}_weights", weights_9x8x100.astype(np.float32))
        argmax_name = f"{prefix}_argmax"
        factor_name = f"{prefix}_factor"
        graph.node.append(helper.make_node("ArgMax", [prob_output.name], [argmax_name], name=argmax_name, axis=1, keepdims=0))
        graph.node.append(helper.make_node("Gather", [weights_name, argmax_name], [factor_name], name=factor_name, axis=0))
    else:
        weights_flat_name = add_initializer(graph, f"{prefix}_weights_flat", weights_9x8x100.reshape(9, 800).astype(np.float32))
        factor_flat = f"{prefix}_factor_flat"
        factor_name = f"{prefix}_factor"
        shape_name = add_initializer(graph, f"{prefix}_shape", [-1, 8, 100], dtype=np.int64)
        graph.node.append(helper.make_node("MatMul", [prob_output.name, weights_flat_name], [factor_flat], name=factor_flat))
        graph.node.append(helper.make_node("Reshape", [factor_flat, shape_name], [factor_name], name=factor_name))

    relevance_name = f"{prefix}_relevance"
    graph.node.append(helper.make_node("Mul", [sqrt_name, factor_name], [relevance_name], name=relevance_name))

    new_rel_output = onnx.ValueInfoProto()
    new_rel_output.CopyFrom(rel_output)
    new_rel_output.name = relevance_name
    graph.ClearField("output")
    graph.output.extend([prob_output, new_rel_output])
    prune_unused_graph_parts(model)

    output_path = output_dir / f"logic_timereise_{suffix}.onnx"
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    return output_path


def run_eval_subprocess(model: str, output: Path, args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        "-m",
        "gearxai_project.evaluate_devkit",
        "--model",
        model,
        "--data-dir",
        args.data_dir,
        "--split",
        args.eval_split,
        "--batch-size",
        str(args.eval_batch_size),
        "--output",
        str(output),
    ]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def report_scores(report_path: Path) -> tuple[float, float, float, float, float, float]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    faith = report["faithfulness"]
    classification = report["classification"]
    simplicity = report["simplicity"]
    faith_score = float(faith["faith_score"])
    simplicity_score = float(simplicity["simplicity_score"])
    proxy = 0.4 * faith_score + 0.2 * simplicity_score
    return (
        faith_score,
        float(faith["deletion_auc"]),
        float(faith["insertion_auc"]),
        float(classification["macro_f1"]),
        simplicity_score,
        proxy,
    )


def package_model(model: str, out_path: Path, args: argparse.Namespace) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "gearxai_project.package_devkit",
            "package",
            "--model",
            model,
            "--data-dir",
            args.data_dir,
            "--split",
            args.eval_split,
            "--out",
            str(out_path),
            "--batch-size",
            str(args.eval_batch_size),
        ],
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TimeREISE-style perturbation distillation for LogicLSTM relevance.")
    parser.add_argument("--base-model", default="external/gearxai_devkit/baselines/onnx/logic_lstm.onnx")
    parser.add_argument("--config", default="configs/spectral_lite_c.yaml")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=20000)
    parser.add_argument("--stats-batch-size", type=int, default=256)
    parser.add_argument("--num-bins", type=int, default=20)
    parser.add_argument("--output-dir", default="runs/logic_timereise_search")
    parser.add_argument("--data-dir", default="prepared_hf_val5k")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--class-channel-stats", default="runs/logic_class_xai_search/class_channel_abs_means.npy")
    parser.add_argument("--ratio-alpha", type=float, default=0.7875)
    parser.add_argument("--time-beta", type=float, nargs="+", default=[0.15, 0.25, 0.40, 0.60])
    parser.add_argument("--copy-prefix", default="logic_timereise")
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

    compute_stats(args, output_dir)
    specs = build_weight_specs(args, output_dir)
    base_model = onnx.load(args.base_model)
    manifest = []
    for tag, weights in specs:
        for hard in (False, True):
            suffix = f"{tag}_hard" if hard else tag
            model_path = output_dir / f"logic_timereise_{suffix}.onnx"
            if not model_path.exists():
                model_path = make_variant(base_model, output_dir, tag, weights, hard)
            manifest.append({"tag": suffix, "model": str(model_path)})
    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"Prepared {len(manifest)} TimeREISE-distilled variants", flush=True)

    rows = []
    for index, row in enumerate(manifest, 1):
        report_path = eval_dir / f"{row['tag']}.json"
        if not report_path.exists():
            run_eval_subprocess(str(row["model"]), report_path, args)
        faith, deletion, insertion, f1, simplicity, proxy = report_scores(report_path)
        rows.append({**row, "report": str(report_path), "faith": faith, "deletion": deletion, "insertion": insertion, "f1": f1, "simplicity": simplicity, "proxy": proxy})
        print(
            f"{index}/{len(manifest)} {row['tag']} faith={faith:.6f} del={deletion:.6f} "
            f"ins={insertion:.6f} f1={f1:.6f} simp={simplicity:.6f} proxy={proxy:.6f}",
            flush=True,
        )

    by_proxy = sorted(rows, key=lambda item: float(item["proxy"]), reverse=True)
    by_faith = sorted(rows, key=lambda item: float(item["faith"]), reverse=True)
    (output_dir / "summary_top.json").write_text(
        json.dumps({"best_proxy": by_proxy[:20], "best_faith": by_faith[:20]}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    for kind, row in (("bestproxy", by_proxy[0]), ("bestfaith", by_faith[0])):
        model_dst = candidate_dir / f"{args.copy_prefix}_{kind}.onnx"
        report_dst = candidate_dir / f"{args.copy_prefix}_{kind}_devkit.json"
        shutil.copy2(str(row["model"]), model_dst)
        shutil.copy2(str(row["report"]), report_dst)
        print(f"Copied {kind}: {model_dst}", flush=True)
        if not args.no_package:
            package_dst = final_dir / f"{args.copy_prefix}_{kind}_submission.zip"
            package_model(str(model_dst), package_dst, args)
            print(f"Packaged {kind}: {package_dst}", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
