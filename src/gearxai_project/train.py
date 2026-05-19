from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import AdamW
from tqdm.auto import tqdm

from gearxai_project.data import build_loaders
from gearxai_project.losses import relevance_regularization
from gearxai_project.model import build_model
from gearxai_project.utils import append_jsonl, choose_device, count_parameters, load_config, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GearXAI starter model.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--output-dir")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    return parser.parse_args()


def apply_cli_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg = dict(cfg)
    cfg["data"] = dict(cfg["data"])
    cfg["training"] = dict(cfg["training"])

    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["training"]["lr"] = args.lr
    if args.output_dir is not None:
        cfg["training"]["output_dir"] = args.output_dir
    if args.max_train_samples is not None:
        cfg["data"]["max_train_samples"] = args.max_train_samples
    if args.max_val_samples is not None:
        cfg["data"]["max_val_samples"] = args.max_val_samples
    return cfg


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    train_cfg: dict[str, Any],
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    grad_clip_norm = train_cfg.get("grad_clip_norm")
    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"

    progress = tqdm(loader, desc="train", leave=False)
    for x, y in progress:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits, relevance = model.forward_train(x)
            cls_loss = criterion(logits, y)
            reg_loss = relevance_regularization(
                relevance,
                sparse_weight=float(train_cfg.get("sparse_weight", 0.0)),
                tv_weight=float(train_cfg.get("tv_weight", 0.0)),
            )
            loss = cls_loss + reg_loss

        scaler.scale(loss).backward()
        if grad_clip_norm:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
        scaler.step(optimizer)
        scaler.update()

        batch_size = x.size(0)
        total_loss += float(loss.detach()) * batch_size
        total_correct += int((logits.argmax(dim=1) == y).sum())
        total_seen += batch_size
        progress.set_postfix(loss=total_loss / total_seen, acc=total_correct / total_seen)

    return {
        "loss": total_loss / max(total_seen, 1),
        "accuracy": total_correct / max(total_seen, 1),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    preds: list[int] = []
    labels: list[int] = []

    for x, y in tqdm(loader, desc="val", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits, _ = model.forward_train(x)
        loss = criterion(logits, y)

        pred = logits.argmax(dim=1)
        batch_size = x.size(0)
        total_loss += float(loss.detach()) * batch_size
        total_correct += int((pred == y).sum())
        total_seen += batch_size
        preds.extend(pred.cpu().tolist())
        labels.extend(y.cpu().tolist())

    return {
        "loss": total_loss / max(total_seen, 1),
        "accuracy": total_correct / max(total_seen, 1),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    epoch: int,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": cfg,
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    cfg = apply_cli_overrides(load_config(args.config), args)
    train_cfg = cfg["training"]
    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "config.resolved.json", cfg)

    set_seed(int(train_cfg.get("seed", 42)))
    device = choose_device()

    train_loader, val_loader = build_loaders(
        cfg["data"],
        batch_size=int(train_cfg["batch_size"]),
        seed=int(train_cfg.get("seed", 42)),
    )
    model = build_model(cfg["model"]).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=float(train_cfg.get("label_smoothing", 0.0)))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(train_cfg.get("amp", True)) and device.type == "cuda")

    print(f"Device: {device}")
    print(f"Train batches: {len(train_loader)}, val batches: {len(val_loader)}")
    print(f"Trainable parameters: {count_parameters(model):,}")

    best_f1 = -1.0
    epochs = int(train_cfg["epochs"])
    for epoch in range(1, epochs + 1):
        start = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, train_cfg)
        val_metrics = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - start

        row = {
            "epoch": epoch,
            "seconds": round(elapsed, 2),
            "train": train_metrics,
            "val": val_metrics,
        }
        append_jsonl(output_dir / "metrics.jsonl", row)
        print(
            f"epoch {epoch:03d}/{epochs} "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['macro_f1']:.4f}"
        )

        save_checkpoint(output_dir / "last.pt", model, optimizer, cfg, epoch, val_metrics)
        if bool(train_cfg.get("save_every_epoch", False)):
            save_checkpoint(output_dir / f"epoch_{epoch:03d}.pt", model, optimizer, cfg, epoch, val_metrics)
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            save_checkpoint(output_dir / "best.pt", model, optimizer, cfg, epoch, val_metrics)

    print(f"Best validation macro F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
