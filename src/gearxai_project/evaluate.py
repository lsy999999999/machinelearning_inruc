from __future__ import annotations

import argparse

import torch
from torch import nn

from gearxai_project.data import build_loaders
from gearxai_project.model import build_model
from gearxai_project.train import evaluate
from gearxai_project.utils import choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a GearXAI checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--full-val", action="store_true", help="Evaluate the full validation split saved in the dataset cache.")
    parser.add_argument("--num-workers", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    cfg = checkpoint["config"]

    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.full_val:
        cfg["data"]["max_val_samples"] = None
    elif args.max_val_samples is not None:
        cfg["data"]["max_val_samples"] = args.max_val_samples
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers

    _, val_loader = build_loaders(
        cfg["data"],
        batch_size=int(cfg["training"]["batch_size"]),
        seed=int(cfg["training"].get("seed", 42)),
    )
    device = choose_device()
    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(checkpoint["model_state"])

    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg["training"].get("label_smoothing", 0.0)))
    metrics = evaluate(model, val_loader, criterion, device, num_classes=int(cfg["model"]["num_classes"]))
    print(metrics)


if __name__ == "__main__":
    main()
