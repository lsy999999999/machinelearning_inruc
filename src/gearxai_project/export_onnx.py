from __future__ import annotations

import argparse
from pathlib import Path

import torch

from gearxai_project.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export GearXAI model to ONNX.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--opset", type=int, default=18)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg = checkpoint["config"]
    model = build_model(cfg["model"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dummy = torch.randn(1, cfg["model"]["in_channels"], 100, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        output_path,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["probabilities", "relevance_map"],
        dynamic_axes={
            "input": {0: "batch"},
            "probabilities": {0: "batch"},
            "relevance_map": {0: "batch"},
        },
    )
    print(f"Exported ONNX model to {output_path}")


if __name__ == "__main__":
    main()
