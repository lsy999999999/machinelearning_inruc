from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from gearxai_project.export_onnx import export_model_to_onnx, verify_onnx
from gearxai_project.model import GearXAICNNConfig, GearXAICNNGate


def main() -> None:
    model = GearXAICNNGate(GearXAICNNConfig(use_spectral=True, spectral_channels=32))
    model.eval()
    x = torch.randn(2, 8, 100)

    with torch.no_grad():
        probabilities, relevance = model(x)

    assert probabilities.shape == (2, 9)
    assert relevance.shape == (2, 8, 100)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(2), atol=1e-5)
    assert relevance.min() >= 0
    assert relevance.max() <= 1

    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "model.onnx"
        export_model_to_onnx(model, torch.randn(2, 8, 100), output, opset=18)
        assert output.exists()
        verify_onnx(output, torch.randn(2, 8, 100))

    print("Smoke test passed.")


if __name__ == "__main__":
    main()
