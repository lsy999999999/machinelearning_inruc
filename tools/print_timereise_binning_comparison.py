from __future__ import annotations

import json
from pathlib import Path


RUNS = [
    ("b10", Path("runs/logic_timereise_marginal_val5k_b10/summary_top.json")),
    ("b25", Path("runs/logic_timereise_marginal_val5k_b25/summary_top.json")),
    ("b50", Path("runs/logic_timereise_marginal_val5k_b50_narrow/summary_top.json")),
]


def best_row(path: Path) -> dict:
    data = json.loads(path.read_text())
    rows = data.get("best_faith") or data.get("best_proxy")
    if not rows:
        raise RuntimeError(f"No best rows found in {path}")
    return rows[0]


def main() -> None:
    print("TimeREISE binning comparison on val5k")
    print("bin  tag                         macro_f1   faith      deletion   insertion  simplicity")
    print("---  --------------------------  ---------  ---------  ---------  ---------  ----------")
    for label, path in RUNS:
        row = best_row(path)
        print(
            f"{label:<4} "
            f"{row['tag']:<26} "
            f"{row['f1']:.6f}  "
            f"{row['faith']:.6f}  "
            f"{row['deletion']:.6f}  "
            f"{row['insertion']:.6f}  "
            f"{row['simplicity']:.6f}"
        )


if __name__ == "__main__":
    main()
