from __future__ import annotations

import argparse
import json

import numpy as np

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid  # type: ignore[attr-defined]

from gearxai_devkit.submission import create_submission_package, inspect_submission_package


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a GearXAI submission package with NumPy 2.x compatibility.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    package = subparsers.add_parser("package")
    package.add_argument("--model", required=True)
    package.add_argument("--data-dir")
    package.add_argument("--split", default="validation")
    package.add_argument("--samples", type=int, default=8)
    package.add_argument("--out", required=True)
    package.add_argument("--band-config")
    package.add_argument("--batch-size", type=int, default=256)
    package.add_argument("--readme")

    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("package")
    inspect.add_argument("--data-dir")
    inspect.add_argument("--split", default="validation")
    inspect.add_argument("--samples", type=int, default=8)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "package":
        payload = create_submission_package(
            model_path=args.model,
            data_dir=args.data_dir,
            output_path=args.out,
            split=args.split,
            samples=args.samples,
            band_config_path=args.band_config,
            batch_size=args.batch_size,
            readme_path=args.readme,
        )
    elif args.command == "inspect":
        payload = inspect_submission_package(
            package_path=args.package,
            data_dir=args.data_dir,
            split=args.split,
            samples=args.samples,
        )
    else:
        raise ValueError(f"Unknown command: {args.command}")

    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload.get("valid", True):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
