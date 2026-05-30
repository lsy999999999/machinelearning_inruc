from __future__ import annotations

import argparse
import inspect
import warnings
from pathlib import Path

import torch
from torch import nn

from gearxai_project.model import build_model


class ExportWrapper(nn.Module):
    """Embed input normalization and optional relevance-map variants into the exported graph."""

    def __init__(self, model: nn.Module, normalize_input: bool = True, relevance_mode: str = "model", eps: float = 1e-6) -> None:
        super().__init__()
        self.model = model
        self.normalize_input = normalize_input
        self.relevance_mode = relevance_mode
        self.eps = eps

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        centered = x - x.mean(dim=-1, keepdim=True)
        var = (centered * centered).mean(dim=-1, keepdim=True)
        return centered / torch.sqrt(var).clamp_min(self.eps)

    def _energy_relevance(self, x: torch.Tensor) -> torch.Tensor:
        centered = x - x.mean(dim=-1, keepdim=True)
        energy = centered.abs()
        return energy / energy.amax(dim=-1, keepdim=True).clamp_min(self.eps)

    def _abs_relevance(self, x: torch.Tensor) -> torch.Tensor:
        magnitude = x.abs()
        return magnitude / magnitude.amax(dim=-1, keepdim=True).clamp_min(self.eps)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        model_input = self._normalize(x) if self.normalize_input else x
        probabilities, model_relevance = self.model(model_input)
        if self.relevance_mode == "model":
            relevance = model_relevance
        elif self.relevance_mode == "input":
            relevance = self.model.input_relevance(model_input)
        elif self.relevance_mode == "energy":
            relevance = self._energy_relevance(model_input)
        elif self.relevance_mode == "abs":
            relevance = self._abs_relevance(model_input)
        else:
            raise ValueError(f"Unknown relevance mode: {self.relevance_mode}")
        return probabilities, relevance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export GearXAI model to ONNX.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--input-length", type=int, default=100)
    parser.add_argument(
        "--no-input-normalize",
        action="store_true",
        help="Do not embed per-window input normalization even if the checkpoint was trained with it.",
    )
    parser.add_argument(
        "--relevance-mode",
        choices=["model", "input", "energy", "abs"],
        default="model",
        help="Relevance map exported with the probabilities. Classification probabilities are unchanged.",
    )
    parser.add_argument("--skip-verify", action="store_true")
    return parser.parse_args()


def verify_onnx(output_path: Path, dummy: torch.Tensor) -> None:
    import numpy as np
    import onnx
    import onnxruntime as ort

    model_proto = onnx.load(output_path)
    onnx.checker.check_model(model_proto)

    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    probabilities, relevance = session.run(None, {"input": dummy.numpy()})

    expected_prob_shape = (dummy.shape[0], 9)
    expected_rel_shape = tuple(dummy.shape)
    if probabilities.shape != expected_prob_shape:
        raise RuntimeError(f"Unexpected probabilities shape: {probabilities.shape}, expected {expected_prob_shape}")
    if relevance.shape != expected_rel_shape:
        raise RuntimeError(f"Unexpected relevance map shape: {relevance.shape}, expected {expected_rel_shape}")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise RuntimeError("ONNX probabilities do not sum to 1.")
    if relevance.min() < -1e-5 or relevance.max() > 1.0 + 1e-5:
        raise RuntimeError(f"ONNX relevance map is outside [0, 1]: min={relevance.min()}, max={relevance.max()}")


def export_model_to_onnx(model: nn.Module, dummy: torch.Tensor, output_path: Path, opset: int) -> None:
    export_kwargs = {
        "export_params": True,
        "opset_version": opset,
        "do_constant_folding": True,
        "input_names": ["input"],
        "output_names": ["probabilities", "relevance_map"],
        "dynamic_axes": {
            "input": {0: "batch"},
            "probabilities": {0: "batch"},
            "relevance_map": {0: "batch"},
        },
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*legacy TorchScript-based ONNX export.*")
        torch.onnx.export(model, dummy, output_path, **export_kwargs)


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg = checkpoint["config"]
    model = build_model(cfg["model"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    embed_normalize = bool(cfg.get("data", {}).get("normalize", False)) and not args.no_input_normalize
    export_model: nn.Module = model
    if embed_normalize or args.relevance_mode != "model":
        export_model = ExportWrapper(model, normalize_input=embed_normalize, relevance_mode=args.relevance_mode)
    export_model.eval()

    dummy = torch.randn(2, cfg["model"]["in_channels"], args.input_length, dtype=torch.float32)
    export_model_to_onnx(export_model, dummy, output_path, opset=args.opset)
    if not args.skip_verify:
        verify_onnx(output_path, dummy)
    normalization_note = "with embedded input normalization" if embed_normalize else "without embedded input normalization"
    print(f"Exported ONNX model to {output_path} ({normalization_note}, relevance_mode={args.relevance_mode})")


if __name__ == "__main__":
    main()
