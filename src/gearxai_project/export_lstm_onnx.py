from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from gearxai_project.lstm_model import build_lstm_student


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a self-trained LogicLSTM student to ONNX.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--skip-verify", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = build_lstm_student(checkpoint["config"]["model"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    dummy = torch.randn(2, checkpoint["config"]["model"].get("in_channels", 8), 100, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(output),
        opset_version=args.opset,
        export_params=True,
        do_constant_folding=True,
        input_names=["windows"],
        output_names=["probabilities", "relevance"],
        dynamic_axes={"windows": {0: "N"}, "probabilities": {0: "N"}, "relevance": {0: "N"}},
        dynamo=False,
    )
    if not args.skip_verify:
        onnx.checker.check_model(onnx.load(str(output)))
        session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
        probs, rel = session.run(None, {session.get_inputs()[0].name: dummy.numpy()})
        if probs.shape != (2, 9) or rel.shape != (2, 8, 100):
            raise RuntimeError(f"Unexpected output shapes: {probs.shape}, {rel.shape}")
        if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-5):
            raise RuntimeError("Probabilities do not sum to one.")
        if not np.allclose(rel.sum(axis=(1, 2)), 1.0, atol=1e-5):
            raise RuntimeError("Relevance is not normalised by sample magnitude.")
    print(f"Exported baseline-inspired LSTM ONNX to {output} (raw-input abs/sum relevance).")


if __name__ == "__main__":
    main()
