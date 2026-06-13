from __future__ import annotations

import argparse
import json
from json import JSONDecodeError
import shutil
from pathlib import Path

from tools.run_logic_timereise_search import package_model, report_scores, run_eval_subprocess


def evaluate_teacher_model(model_path: Path, report_path: Path, args: argparse.Namespace) -> dict[str, float]:
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
    teacher_score = 0.6 * f1 + 0.3 * faith + 0.1 * simplicity
    return {
        "faith": faith,
        "deletion": deletion,
        "insertion": insertion,
        "f1": f1,
        "simplicity": simplicity,
        "proxy": proxy,
        "teacher_score": teacher_score,
    }


def score_and_package_teacher_manifest(
    manifest: list[dict[str, object]],
    output_dir: Path,
    args: argparse.Namespace,
    *,
    copy_prefix: str,
) -> list[dict[str, object]]:
    eval_dir = output_dir / "eval"
    candidate_dir = Path("runs/candidates")
    final_dir = Path("runs/final_candidates")
    eval_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    rows: list[dict[str, object]] = []
    for index, row in enumerate(manifest, 1):
        tag = str(row["tag"])
        report_path = eval_dir / f"{tag}.json"
        metrics = evaluate_teacher_model(Path(str(row["model"])), report_path, args)
        scored = {**row, "report": str(report_path), **metrics}
        rows.append(scored)
        print(
            f"{index}/{len(manifest)} {tag} faith={metrics['faith']:.6f} "
            f"del={metrics['deletion']:.6f} ins={metrics['insertion']:.6f} "
            f"f1={metrics['f1']:.6f} simp={metrics['simplicity']:.6f} "
            f"teacher={metrics['teacher_score']:.9f}",
            flush=True,
        )

    rows_by_teacher = sorted(rows, key=lambda item: float(item["teacher_score"]), reverse=True)
    rows_by_faith = sorted(rows, key=lambda item: float(item["faith"]), reverse=True)
    rows_by_proxy = sorted(rows, key=lambda item: float(item["proxy"]), reverse=True)
    (output_dir / "summary_top.json").write_text(
        json.dumps(
            {
                "best_teacher": rows_by_teacher[:20],
                "best_faith": rows_by_faith[:20],
                "best_proxy": rows_by_proxy[:20],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    if rows_by_teacher:
        for kind, row in (("bestteacher", rows_by_teacher[0]), ("bestfaith", rows_by_faith[0])):
            model_dst = candidate_dir / f"{copy_prefix}_{kind}.onnx"
            report_dst = candidate_dir / f"{copy_prefix}_{kind}_devkit.json"
            shutil.copy2(str(row["model"]), model_dst)
            shutil.copy2(str(row["report"]), report_dst)
            print(f"Copied {kind}: {model_dst}", flush=True)
            if not args.no_package:
                package_dst = final_dir / f"{copy_prefix}_{kind}_submission.zip"
                package_model(str(model_dst), package_dst, args)
                print(f"Packaged {kind}: {package_dst}", flush=True)

    return rows
