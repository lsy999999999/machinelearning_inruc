from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

from gearxai_project.data import build_loaders
from gearxai_project.losses import relevance_regularization
from gearxai_project.model import build_model
from gearxai_project.utils import append_jsonl, choose_device, count_parameters, load_config, save_json, set_seed


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.updates = 0
        self.shadow = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }
        self.backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        decay = min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))
        current = model.state_dict()
        for name, value in current.items():
            if value.is_floating_point():
                self.shadow[name].mul_(decay).add_(value.detach(), alpha=1.0 - decay)
            else:
                self.shadow[name].copy_(value.detach())

    def store(self, model: nn.Module) -> None:
        self.backup = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)

    def restore(self, model: nn.Module) -> None:
        if self.backup:
            model.load_state_dict(self.backup, strict=True)
            self.backup = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GearXAI starter model.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--label-smoothing", type=float)
    parser.add_argument("--output-dir")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--noise-std", type=float)
    parser.add_argument("--scale-range", type=float)
    parser.add_argument("--time-shift", type=int)
    parser.add_argument("--channel-dropout", type=float)
    parser.add_argument("--ema-decay", type=float)
    return parser.parse_args()


def apply_cli_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg = dict(cfg)
    cfg["data"] = dict(cfg["data"])
    cfg["model"] = dict(cfg["model"])
    cfg["training"] = dict(cfg["training"])

    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["training"]["lr"] = args.lr
    if args.dropout is not None:
        cfg["model"]["dropout"] = args.dropout
    if args.weight_decay is not None:
        cfg["training"]["weight_decay"] = args.weight_decay
    if args.label_smoothing is not None:
        cfg["training"]["label_smoothing"] = args.label_smoothing
    if args.output_dir is not None:
        cfg["training"]["output_dir"] = args.output_dir
    if args.max_train_samples is not None:
        cfg["data"]["max_train_samples"] = args.max_train_samples
    if args.max_val_samples is not None:
        cfg["data"]["max_val_samples"] = args.max_val_samples
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers
    if args.patience is not None:
        cfg["training"]["patience"] = args.patience
    if args.augment:
        cfg["data"]["augment"] = True
    if args.noise_std is not None:
        cfg["data"]["noise_std"] = args.noise_std
    if args.scale_range is not None:
        cfg["data"]["scale_range"] = args.scale_range
    if args.time_shift is not None:
        cfg["data"]["time_shift"] = args.time_shift
    if args.channel_dropout is not None:
        cfg["data"]["channel_dropout"] = args.channel_dropout
    if args.ema_decay is not None:
        cfg["training"]["ema_decay"] = args.ema_decay
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
        ema = train_cfg.get("_ema")
        if ema is not None:
            ema.update(model)

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
    num_classes: int,
) -> dict[str, Any]:
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

    label_ids = list(range(num_classes))
    per_class_f1 = f1_score(labels, preds, average=None, labels=label_ids, zero_division=0)
    cm = confusion_matrix(labels, preds, labels=label_ids)

    return {
        "loss": total_loss / max(total_seen, 1),
        "accuracy": total_correct / max(total_seen, 1),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "per_class_f1": [float(v) for v in per_class_f1],
        "confusion_matrix": cm.astype(int).tolist(),
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    epoch: int,
    metrics: dict[str, Any],
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
    ema_decay = float(train_cfg.get("ema_decay") or 0.0)
    ema = ModelEMA(model, decay=ema_decay) if ema_decay > 0 else None
    optimizer = AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(int(train_cfg["epochs"]), 1),
        eta_min=float(train_cfg["lr"]) * float(train_cfg.get("cosine_min_lr_ratio", 0.0)),
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=float(train_cfg.get("label_smoothing", 0.0)))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(train_cfg.get("amp", True)) and device.type == "cuda")

    print(f"Device: {device}")
    print(f"Train batches: {len(train_loader)}, val batches: {len(val_loader)}")
    print(f"Trainable parameters: {count_parameters(model):,}")

    best_f1 = -1.0
    epochs_since_best = 0
    epochs = int(train_cfg["epochs"])
    patience = train_cfg.get("patience")
    for epoch in range(1, epochs + 1):
        start = time.time()
        lr = float(optimizer.param_groups[0]["lr"])
        if ema is not None:
            train_cfg["_ema"] = ema
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, train_cfg)
        train_cfg.pop("_ema", None)
        if ema is not None:
            ema.store(model)
            ema.copy_to(model)
        val_metrics = evaluate(model, val_loader, criterion, device, num_classes=int(cfg["model"]["num_classes"]))
        scheduler.step()
        elapsed = time.time() - start

        row = {
            "epoch": epoch,
            "seconds": round(elapsed, 2),
            "lr": lr,
            "train": train_metrics,
            "val": val_metrics,
        }
        append_jsonl(output_dir / "metrics.jsonl", row)
        save_json(output_dir / "last_metrics.json", row)
        print(
            f"epoch {epoch:03d}/{epochs} "
            f"lr={lr:.2e} "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['macro_f1']:.4f}"
        )

        save_checkpoint(output_dir / "last.pt", model, optimizer, cfg, epoch, val_metrics)
        if bool(train_cfg.get("save_every_epoch", False)):
            save_checkpoint(output_dir / f"epoch_{epoch:03d}.pt", model, optimizer, cfg, epoch, val_metrics)
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            epochs_since_best = 0
            save_checkpoint(output_dir / "best.pt", model, optimizer, cfg, epoch, val_metrics)
            save_json(output_dir / "best_metrics.json", row)
        else:
            epochs_since_best += 1
        if ema is not None:
            ema.restore(model)

        if patience and epochs_since_best >= int(patience):
            print(f"Early stopping after {epoch} epochs without improving for {patience} epochs.")
            break

    print(f"Best validation macro F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
