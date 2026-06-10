from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.run_logic_timereise_offline_innovation_search import add_mechanical_prior, expanded_mechanical_prior
from tools.run_logic_timereise_power_ensemble_search import extract_timereise_weights
from tools.run_logic_timereise_search import make_variant
from tools.timereise_branch_utils import load_base_model, score_and_package_manifest


def gamma_tag(gamma: float) -> str:
    """Use 1e-4 precision so tiny mechanical-prior sweeps do not collide."""
    return f"g{int(round(gamma * 10000)):04d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse a folded TimeREISE weight map with an offline mechanical prior.")
    parser.add_argument("--base-model", default="external/gearxai_devkit/baselines/onnx/logic_lstm.onnx")
    parser.add_argument("--source-model", default="runs/candidates/logic_timereise_class_candidate_selection_bestproxy.onnx")
    parser.add_argument("--mech-stats", default="runs/logic_timereise_offline_innovation_search/offline_mech_stats.npz")
    parser.add_argument("--output-dir", default="runs/logic_timereise_mech_prior_fusion")
    parser.add_argument("--data-dir", default="prepared_hf_val5k")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--gamma", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.05, 0.08])
    parser.add_argument("--copy-prefix", default="logic_timereise_mech_prior_fusion")
    parser.add_argument("--no-package", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_weights = extract_timereise_weights(args.source_model)
    prior = expanded_mechanical_prior(args.mech_stats)
    base_model = load_base_model(args.base_model)

    manifest = []
    for gamma in args.gamma:
        weights = add_mechanical_prior(source_weights, prior, gamma)
        tag = f"mech_{gamma_tag(gamma)}"
        model_path = output_dir / f"logic_timereise_{tag}.onnx"
        if not model_path.exists():
            make_variant(base_model, output_dir, tag, weights, hard=False)
        manifest.append({"tag": tag, "model": str(model_path), "branch": "mech_prior_fusion", "gamma": gamma})

    print(f"Prepared {len(manifest)} mechanical-prior fusion variants", flush=True)
    score_and_package_manifest(manifest, output_dir, args, copy_prefix=args.copy_prefix)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
