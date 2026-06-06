from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import onnx

from tools.run_logic_timereise_search import (
    expand_bins,
    normalize_importance,
    package_model,
    report_scores,
    run_eval_subprocess,
)


def timereise_weights_from_stats(stats_file: str | Path, beta: float) -> np.ndarray:
    stats = np.load(stats_file)
    counts = np.maximum(stats["counts"].astype(np.float64), 1.0)
    drop = stats["drop_sums"].astype(np.float64) / counts[:, None, None]
    keep = stats["keep_sums"].astype(np.float64) / counts[:, None, None]
    mix = normalize_importance(0.5 * normalize_importance(drop) + 0.5 * normalize_importance(keep))
    expanded = expand_bins(mix, stats["blocks"])
    expanded = expanded / np.maximum(expanded.mean(axis=(1, 2), keepdims=True), 1e-6)
    weights = np.clip(1.0 + beta * (expanded - 1.0), 0.05, None)
    weights = weights / np.maximum(weights.mean(axis=(1, 2), keepdims=True), 1e-6)
    return weights.astype(np.float32)


def score_and_package_manifest(
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
        if not report_path.exists():
            run_eval_subprocess(str(row["model"]), report_path, args)
        faith, deletion, insertion, f1, simplicity, proxy = report_scores(report_path)
        scored = {
            **row,
            "report": str(report_path),
            "faith": faith,
            "deletion": deletion,
            "insertion": insertion,
            "f1": f1,
            "simplicity": simplicity,
            "proxy": proxy,
        }
        rows.append(scored)
        print(
            f"{index}/{len(manifest)} {tag} faith={faith:.6f} del={deletion:.6f} "
            f"ins={insertion:.6f} f1={f1:.6f} simp={simplicity:.6f} proxy={proxy:.6f}",
            flush=True,
        )

    rows_by_proxy = sorted(rows, key=lambda item: float(item["proxy"]), reverse=True)
    rows_by_faith = sorted(rows, key=lambda item: float(item["faith"]), reverse=True)
    (output_dir / "summary_top.json").write_text(
        json.dumps({"best_proxy": rows_by_proxy[:20], "best_faith": rows_by_faith[:20]}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if rows_by_proxy:
        for kind, row in (("bestproxy", rows_by_proxy[0]), ("bestfaith", rows_by_faith[0])):
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


def load_base_model(path: str) -> onnx.ModelProto:
    model = onnx.load(path)
    onnx.checker.check_model(model)
    return model
