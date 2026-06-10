from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.run_logic_timereise_search import make_variant
from tools.timereise_branch_utils import load_base_model, score_and_package_manifest


DEFAULT_SOURCES = (
    ("marg", "runs/candidates/logic_timereise_marginal_val5k_b10_bestproxy.onnx"),
    ("robust", "runs/candidates/logic_timereise_robust_bestfaith.onnx"),
    ("mix", "runs/candidates/logic_timereise_50k_b20_mix50_tb035_bestproxy.onnx"),
    ("contrast", "runs/candidates/logic_timereise_contrastive_bestfaith.onnx"),
    ("offline", "runs/candidates/logic_timereise_offline_innovation_bestfaith.onnx"),
)


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    weights = weights.astype(np.float64, copy=False)
    weights = np.clip(weights, 1e-6, None)
    weights = weights / np.maximum(weights.mean(axis=(1, 2), keepdims=True), 1e-6)
    return np.clip(weights, 0.03, 20.0).astype(np.float32)


def extract_timereise_weights(model_path: str | Path) -> np.ndarray:
    model = onnx.load(str(model_path))
    candidates: list[np.ndarray] = []
    for initializer in model.graph.initializer:
        name = initializer.name.lower()
        if "weight" not in name:
            continue
        array = numpy_helper.to_array(initializer)
        if array.shape == (9, 800):
            candidates.append(array.reshape(9, 8, 100))
        elif array.shape == (9, 8, 100):
            candidates.append(array)

    if not candidates:
        raise RuntimeError(f"Could not find TimeREISE weights in {model_path}")
    if len(candidates) > 1:
        shapes = [candidate.shape for candidate in candidates]
        raise RuntimeError(f"Multiple TimeREISE weight tensors in {model_path}: {shapes}")
    return normalize_weights(candidates[0])


def parse_source(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("source must be name=path")
    name, path = value.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise argparse.ArgumentTypeError("source must be name=path")
    return name, path


def format_exp(value: float) -> str:
    return f"{int(round(value * 1000)):04d}"


def tag_for(exponents: dict[str, float], prefix: str = "pow") -> str:
    parts = [prefix]
    for name in sorted(exponents):
        parts.append(f"{name[:3]}{format_exp(exponents[name])}")
    return "_".join(parts)


def default_exponent_specs(source_names: set[str]) -> list[tuple[str, dict[str, float]]]:
    specs: list[tuple[str, dict[str, float]]] = []

    def add(exponents: dict[str, float], prefix: str = "pow") -> None:
        if all(name in source_names for name in exponents):
            specs.append((tag_for(exponents, prefix), exponents))

    for robust_exp in (0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30):
        add({"marg": 1.0 - robust_exp, "robust": robust_exp})
    for robust_exp in (0.03, 0.05, 0.08, 0.10, 0.15):
        add({"marg": 1.0, "robust": robust_exp}, prefix="boost")

    for other in ("mix", "contrast", "offline"):
        for other_exp in (0.03, 0.05, 0.08, 0.10, 0.15):
            add({"marg": 1.0 - other_exp, other: other_exp})
        for other_exp in (0.05, 0.10):
            add({"marg": 1.0, other: other_exp}, prefix="boost")

    for other in ("mix", "contrast", "offline"):
        for robust_exp, other_exp in ((0.05, 0.05), (0.10, 0.05), (0.10, 0.10), (0.15, 0.05)):
            add({"marg": 1.0 - robust_exp - other_exp, "robust": robust_exp, other: other_exp})

    seen: set[str] = set()
    unique: list[tuple[str, dict[str, float]]] = []
    for tag, exponents in specs:
        if tag in seen:
            continue
        seen.add(tag)
        unique.append((tag, exponents))
    return unique


def parse_manual_spec(value: str) -> tuple[str, dict[str, float]]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("spec must be tag=name:exp,name:exp")
    tag, body = value.split("=", 1)
    tag = tag.strip()
    if not tag:
        raise argparse.ArgumentTypeError("spec tag cannot be empty")
    exponents: dict[str, float] = {}
    for part in body.split(","):
        if ":" not in part:
            raise argparse.ArgumentTypeError("spec entries must be name:exp")
        name, exp = part.split(":", 1)
        name = name.strip()
        if not name:
            raise argparse.ArgumentTypeError("spec source name cannot be empty")
        exponents[name] = float(exp)
    return tag, exponents


def geometric_fusion(source_weights: dict[str, np.ndarray], exponents: dict[str, float]) -> np.ndarray:
    total = np.zeros((9, 8, 100), dtype=np.float64)
    used = 0
    for name, exponent in exponents.items():
        if name not in source_weights:
            raise RuntimeError(f"Unknown source in spec: {name}")
        if exponent == 0.0:
            continue
        total += float(exponent) * np.log(np.maximum(source_weights[name], 1e-6))
        used += 1
    if used == 0:
        raise RuntimeError("Fusion spec has no non-zero exponents.")
    return normalize_weights(np.exp(total))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Power/geometric ensemble search for folded TimeREISE weights.")
    parser.add_argument("--base-model", default="external/gearxai_devkit/baselines/onnx/logic_lstm.onnx")
    parser.add_argument("--source", type=parse_source, nargs="+", default=list(DEFAULT_SOURCES))
    parser.add_argument("--spec", type=parse_manual_spec, nargs="*", default=[])
    parser.add_argument("--output-dir", default="runs/logic_timereise_power_ensemble_search")
    parser.add_argument("--data-dir", default="prepared_hf_val5k")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--copy-prefix", default="logic_timereise_power_ensemble")
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

    if "marg" not in source_weights:
        raise RuntimeError("The default search needs a 'marg' source.")

    specs = args.spec if args.spec else default_exponent_specs(set(source_weights))
    base_model = load_base_model(args.base_model)
    manifest = []
    for tag, exponents in specs:
        weights = geometric_fusion(source_weights, exponents)
        model_path = output_dir / f"logic_timereise_{tag}.onnx"
        if not model_path.exists():
            make_variant(base_model, output_dir, tag, weights, hard=False)
        manifest.append(
            {
                "tag": tag,
                "model": str(model_path),
                "branch": "power_ensemble",
                "exponents": exponents,
            }
        )

    print(f"Prepared {len(manifest)} power ensemble TimeREISE variants", flush=True)
    score_and_package_manifest(manifest, output_dir, args, copy_prefix=args.copy_prefix)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
