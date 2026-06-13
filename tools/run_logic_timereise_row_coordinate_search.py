from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from json import JSONDecodeError

import numpy as np
import onnx
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.run_logic_timereise_class_candidate_selection import load_manifest_candidates, parse_candidate
from tools.run_logic_timereise_power_ensemble_search import extract_timereise_weights
from tools.run_logic_timereise_search import make_variant, package_model, report_scores, run_eval_subprocess
from tools.timereise_branch_utils import load_base_model


def write_variant(base_model, output_dir: Path, tag: str, weights: np.ndarray) -> Path:
    model_path = output_dir / f"logic_timereise_{tag}.onnx"
    if model_path.exists():
        try:
            onnx.checker.check_model(onnx.load(model_path))
            return model_path
        except Exception:
            model_path.unlink()
    return make_variant(base_model, output_dir, tag, weights, hard=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Coordinate-search class rows with full devkit evaluation.")
    parser.add_argument("--base-model", default="external/gearxai_devkit/baselines/onnx/logic_lstm.onnx")
    parser.add_argument("--start-model", required=True)
    parser.add_argument("--candidate", type=parse_candidate, nargs="+", default=[])
    parser.add_argument("--candidate-manifest", nargs="*", default=[])
    parser.add_argument("--output-dir", default="runs/logic_timereise_row_coordinate_search")
    parser.add_argument("--data-dir", default="prepared_hf_val5k")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--min-delta", type=float, default=1e-7)
    parser.add_argument("--tag-prefix", default="coord_search")
    parser.add_argument("--copy-prefix", default="logic_timereise_row_coordinate")
    parser.add_argument("--no-package", action="store_true")
    return parser.parse_args()


def evaluate_model(model_path: Path, report_path: Path, args: argparse.Namespace) -> dict[str, float]:
    if report_path.exists():
        try:
            faith, deletion, insertion, f1, simplicity, proxy = report_scores(report_path)
        except (JSONDecodeError, KeyError):
            report_path.unlink()
            run_eval_subprocess(str(model_path), report_path, args)
            faith, deletion, insertion, f1, simplicity, proxy = report_scores(report_path)
    else:
        run_eval_subprocess(str(model_path), report_path, args)
        faith, deletion, insertion, f1, simplicity, proxy = report_scores(report_path)
    new_score = 0.6 * f1 + 0.3 * faith + 0.1 * simplicity
    return {
        "faith": faith,
        "deletion": deletion,
        "insertion": insertion,
        "f1": f1,
        "simplicity": simplicity,
        "proxy": proxy,
        "new_score": new_score,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    eval_dir = output_dir / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    Path("runs/candidates").mkdir(parents=True, exist_ok=True)
    Path("runs/final_candidates").mkdir(parents=True, exist_ok=True)

    candidate_pairs = list(args.candidate) + load_manifest_candidates(args.candidate_manifest)
    if not candidate_pairs:
        raise RuntimeError("Provide at least one candidate row source.")

    base_model = load_base_model(args.base_model)
    current = extract_timereise_weights(args.start_model)
    candidate_weights: dict[str, np.ndarray] = {
        "start": current,
    }
    for name, path in candidate_pairs:
        model_path = Path(path)
        if not model_path.exists():
            print(f"Skipping missing candidate {name}: {model_path}", flush=True)
            continue
        candidate_weights[name] = extract_timereise_weights(model_path)

    start_tag = f"{args.tag_prefix}_start"
    current_path = write_variant(base_model, output_dir, start_tag, current)
    current_report = eval_dir / f"{start_tag}.json"
    current_metrics = evaluate_model(current_path, current_report, args)
    history: list[dict[str, object]] = [
        {"tag": start_tag, "model": str(current_path), **current_metrics},
    ]
    print(
        f"start faith={current_metrics['faith']:.6f} del={current_metrics['deletion']:.6f} "
        f"ins={current_metrics['insertion']:.6f} score={current_metrics['new_score']:.9f}",
        flush=True,
    )

    step = 0
    for pass_id in range(args.passes):
        improved_in_pass = False
        for class_id in range(9):
            best_name = "current"
            best_weights = current
            best_metrics = current_metrics
            best_path = current_path
            for name, weights in candidate_weights.items():
                trial = current.copy()
                trial[class_id] = weights[class_id]
                tag = f"{args.tag_prefix}_p{pass_id}_c{class_id}_{name}"
                model_path = write_variant(base_model, output_dir, tag, trial)
                metrics = evaluate_model(model_path, eval_dir / f"{tag}.json", args)
                if metrics["new_score"] > best_metrics["new_score"] + args.min_delta:
                    best_name = name
                    best_weights = trial
                    best_metrics = metrics
                    best_path = model_path
            if best_name != "current":
                step += 1
                current = best_weights
                current_metrics = best_metrics
                step_tag = f"{args.tag_prefix}_update{step}"
                current_path = write_variant(base_model, output_dir, step_tag, current)
                current_report = eval_dir / f"{step_tag}.json"
                current_metrics = evaluate_model(current_path, current_report, args)
                improved_in_pass = True
                row = {
                    "step": step,
                    "pass": pass_id,
                    "class_id": class_id,
                    "source": best_name,
                    "model": str(current_path),
                    **current_metrics,
                }
                history.append(row)
                print(
                    f"step={step} pass={pass_id} class={class_id} source={best_name} "
                    f"faith={current_metrics['faith']:.6f} del={current_metrics['deletion']:.6f} "
                    f"ins={current_metrics['insertion']:.6f} score={current_metrics['new_score']:.9f}",
                    flush=True,
                )
        if not improved_in_pass:
            print(f"No improvement in pass {pass_id}; stopping.", flush=True)
            break

    summary = {
        "history": history,
        "best": history[-1],
    }
    (output_dir / "summary_top.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    candidate_model = Path("runs/candidates") / f"{args.copy_prefix}_bestproxy.onnx"
    candidate_report = Path("runs/candidates") / f"{args.copy_prefix}_bestproxy_devkit.json"
    candidate_model.write_bytes(current_path.read_bytes())
    candidate_report.write_bytes(current_report.read_bytes())
    print(f"Copied best: {candidate_model}", flush=True)

    if not args.no_package:
        package_path = Path("runs/final_candidates") / f"{args.copy_prefix}_bestproxy_submission.zip"
        package_model(str(candidate_model), package_path, args)
        print(f"Packaged best: {package_path}", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
