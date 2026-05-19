from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from gearxai_project.model import GearXAICNNConfig, GearXAICNNGate


def main() -> None:
    model = GearXAICNNGate(GearXAICNNConfig())
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
        torch.onnx.export(
            model,
            torch.randn(1, 8, 100),
            output,
            opset_version=18,
            input_names=["input"],
            output_names=["probabilities", "relevance_map"],
            dynamic_axes={
                "input": {0: "batch"},
                "probabilities": {0: "batch"},
                "relevance_map": {0: "batch"},
            },
        )
        assert output.exists()

    print("Smoke test passed.")


if __name__ == "__main__":
    main()
