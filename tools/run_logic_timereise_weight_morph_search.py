from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.run_logic_timereise_power_ensemble_search import extract_timereise_weights, normalize_weights, parse_source
from tools.run_logic_timereise_search import make_variant
from tools.timereise_branch_utils import load_base_model, score_and_package_manifest


def parse_classes(value: str) -> list[int]:
    if value.strip().lower() in {"all", "*"}:
        return list(range(9))
    return [int(part) for part in value.split(",") if part.strip()]


def smooth_time(weights: np.ndarray, passes: int) -> np.ndarray:
    out = weights.astype(np.float64, copy=True)
    for _ in range(passes):
        padded = np.pad(out, ((0, 0), (0, 0), (1, 1)), mode="edge")
        out = 0.25 * padded[:, :, :-2] + 0.5 * padded[:, :, 1:-1] + 0.25 * padded[:, :, 2:]
        out = out / np.maximum(out.mean(axis=(1, 2), keepdims=True), 1e-6)
    return normalize_weights(out)


def power_rows(weights: np.ndarray, power: float, classes: list[int]) -> np.ndarray:
    out = weights.copy()
    for class_id in classes:
        row = np.power(np.maximum(out[class_id], 1e-6), power)
        row = row / np.maximum(row.mean(), 1e-6)
        out[class_id] = np.clip(row, 0.03, 20.0)
    return normalize_weights(out)


def blend_rows(base: np.ndarray, other: np.ndarray, alpha: float, classes: list[int]) -> np.ndarray:
    out = base.copy()
    for class_id in classes:
        row = (1.0 - alpha) * base[class_id] + alpha * other[class_id]
        row = row / np.maximum(row.mean(), 1e-6)
        out[class_id] = np.clip(row, 0.03, 20.0)
    return normalize_weights(out)


def format_float(value: float) -> str:
    return f"{int(round(value * 100)):03d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate folded TimeREISE weight morph candidates.")
    parser.add_argument("--base-model", default="runs/candidates/logic_timereise_teacherbest_rowcoord_upgrade_bestproxy.onnx")
    parser.add_argument(
        "--source",
        type=parse_source,
        nargs="+",
        default=[
            ("start", "runs/candidates/logic_timereise_teacherbest_rowcoord_upgrade_bestproxy.onnx"),
            ("stable", "runs/candidates/logic_timereise_row_coordinate_ext_bestproxy.onnx"),
            ("phys_all", "runs/logic_timereise_classsel_phys_proxy_all/logic_timereise_classsel_pred.onnx"),
            ("phys_top", "runs/logic_timereise_classsel_phys_proxy_top/logic_timereise_classsel_pred.onnx"),
            ("sharp126", "runs/logic_timereise_classsel_all_sharp_ultrafine/logic_timereise_sharp126.onnx"),
            ("pow047", "runs/logic_timereise_power_ensemble_robust_fine/logic_timereise_b047.onnx"),
            ("pow048", "runs/logic_timereise_power_ensemble_robust_fine/logic_timereise_b048.onnx"),
        ],
    )
    parser.add_argument("--output-dir", default="runs/logic_timereise_weight_morph_search")
    parser.add_argument("--data-dir", default="prepared_hf_val5k")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--blend-source", nargs="+", default=["phys_all", "phys_top", "pow047", "pow048"])
    parser.add_argument("--blend-alpha", type=float, nargs="+", default=[0.20, 0.35, 0.50, 0.65, 0.80])
    parser.add_argument("--blend-classes", type=parse_classes, default=parse_classes("8"))
    parser.add_argument("--smooth-passes", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--smooth-classes", type=parse_classes, default=parse_classes("6,7,8"))
    parser.add_argument("--power", type=float, nargs="+", default=[0.85, 0.95, 1.05, 1.15, 1.30])
    parser.add_argument("--power-classes", type=parse_classes, default=parse_classes("6,7,8"))
    parser.add_argument("--combo", action="store_true", help="Also combine blend, smoothing, and power transforms.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--copy-prefix", default="logic_timereise_weight_morph")
    parser.add_argument("--no-package", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_weights: dict[str, np.ndarray] = {}
    for name, path in args.source:
        model_path = Path(path)
        if not model_path.exists():
            print(f"Skipping missing source {name}: {model_path}", flush=True)
            continue
        source_weights[name] = extract_timereise_weights(model_path)
        print(f"Loaded source {name}: {model_path}", flush=True)
    if "start" not in source_weights:
        raise RuntimeError("A source named 'start' is required.")

    start = source_weights["start"]
    specs: list[tuple[str, np.ndarray]] = [("start", start)]

    for passes in args.smooth_passes:
        smoothed = start.copy()
        smoothed_rows = smooth_time(start, passes)
        for class_id in args.smooth_classes:
            smoothed[class_id] = smoothed_rows[class_id]
        specs.append((f"smooth{passes}_c{''.join(map(str, args.smooth_classes))}", normalize_weights(smoothed)))

    for power in args.power:
        specs.append((f"power{format_float(power)}_c{''.join(map(str, args.power_classes))}", power_rows(start, power, args.power_classes)))

    for source_name, alpha in itertools.product(args.blend_source, args.blend_alpha):
        if source_name not in source_weights:
            continue
        specs.append(
            (
                f"blend_{source_name}_a{format_float(alpha)}_c{''.join(map(str, args.blend_classes))}",
                blend_rows(start, source_weights[source_name], alpha, args.blend_classes),
            )
        )

    if args.combo:
        for source_name, alpha, passes, power in itertools.product(
            args.blend_source,
            args.blend_alpha,
            args.smooth_passes,
            args.power,
        ):
            if source_name not in source_weights:
                continue
            weights = blend_rows(start, source_weights[source_name], alpha, args.blend_classes)
            smoothed_rows = smooth_time(weights, passes)
            for class_id in args.smooth_classes:
                weights[class_id] = smoothed_rows[class_id]
            weights = power_rows(weights, power, args.power_classes)
            specs.append((f"combo_{source_name}_a{format_float(alpha)}_s{passes}_p{format_float(power)}", weights))

    if args.limit is not None:
        specs = specs[: args.limit]

    base_model = load_base_model(args.base_model)
    manifest = []
    seen_tags: set[str] = set()
    for tag, weights in specs:
        if tag in seen_tags:
            continue
        seen_tags.add(tag)
        model_path = output_dir / f"logic_timereise_{tag}.onnx"
        if not model_path.exists():
            make_variant(base_model, output_dir, tag, weights, hard=False)
        manifest.append({"tag": tag, "model": str(model_path), "branch": "weight_morph"})

    print(f"Prepared {len(manifest)} weight-morph TimeREISE variants", flush=True)
    score_and_package_manifest(manifest, output_dir, args, copy_prefix=args.copy_prefix)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
