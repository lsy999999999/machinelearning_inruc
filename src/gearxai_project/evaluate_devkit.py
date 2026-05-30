from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid  # type: ignore[attr-defined]

from gearxai_devkit.evaluator import evaluate_submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GearXAI devkit evaluator with NumPy 2.x compatibility.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--band-config")
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate_submission(
        model_path=args.model,
        data_dir=args.data_dir,
        split=args.split,
        band_config_path=args.band_config,
        batch_size=args.batch_size,
        output_path=args.output,
    )
    summary = {
        "macro_f1": result["classification"]["macro_f1"],
        "faith_score": result["faithfulness"]["faith_score"],
        "deletion_auc": result["faithfulness"]["deletion_auc"],
        "insertion_auc": result["faithfulness"]["insertion_auc"],
        "mechanical_score": result["mechanical"]["mechanical_score"],
        "simplicity_score": result["simplicity"]["simplicity_score"],
        "eligible": result["score"]["eligible"],
        "explainability_score": result["score"]["explainability_score"],
        "reason": result["score"]["reason"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.output:
        print(f"Saved full devkit report to {Path(args.output)}")


if __name__ == "__main__":
    main()
