from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate manifest models via isolated subprocesses.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--data-dir", default="prepared_hf_val5k")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--retries", type=int, default=2)
    return parser.parse_args()


def metrics_line(index: int, total: int, tag: str, report_path: Path) -> str:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    faith = report["faithfulness"]
    classification = report["classification"]
    simplicity = report["simplicity"]
    faith_score = faith["faith_score"]
    proxy = 0.4 * faith_score + 0.2 * simplicity["simplicity_score"]
    return (
        f"{index}/{total} {tag} faith={faith_score:.6f} "
        f"del={faith['deletion_auc']:.6f} ins={faith['insertion_auc']:.6f} "
        f"f1={classification['macro_f1']:.6f} simp={simplicity['simplicity_score']:.6f} "
        f"proxy={proxy:.6f}"
    )


def main() -> None:
    args = parse_args()
    manifest = [json.loads(line) for line in Path(args.manifest).read_text(encoding="utf-8").splitlines()]
    eval_dir = Path(args.eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    total = len(manifest)

    for index, row in enumerate(manifest, 1):
        if index < args.start_index:
            continue
        tag = row["tag"]
        report_path = eval_dir / f"{tag}.json"
        if report_path.exists():
            print(metrics_line(index, total, tag, report_path), flush=True)
            continue

        command = [
            sys.executable,
            "-m",
            "gearxai_project.evaluate_devkit",
            "--model",
            row["model"],
            "--data-dir",
            args.data_dir,
            "--split",
            args.split,
            "--batch-size",
            str(args.batch_size),
            "--output",
            str(report_path),
        ]
        for attempt in range(1, args.retries + 2):
            completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            if completed.returncode == 0 and report_path.exists():
                print(metrics_line(index, total, tag, report_path), flush=True)
                break
            if report_path.exists():
                report_path.unlink()
            print(
                f"FAILED {index}/{total} {tag} attempt={attempt} returncode={completed.returncode}",
                flush=True,
            )
            if completed.stdout:
                print(completed.stdout[-2000:], flush=True)
        else:
            raise SystemExit(f"Could not evaluate {tag} after retries.")


if __name__ == "__main__":
    main()
