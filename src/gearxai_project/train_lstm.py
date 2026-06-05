from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from gearxai_project.data import GearXAIWindows
from gearxai_project.lstm_model import build_lstm_student
from gearxai_project.utils import append_jsonl, choose_device, count_parameters, load_config, save_json, set_seed


class DistillationDataset(Dataset):
    def __init__(self, base: GearXAIWindows, teacher_probs: np.ndarray | None) -> None:
        self.base = base
        self.teacher_probs = teacher_probs
        if teacher_probs is not None and len(teacher_probs) != len(base):
            raise ValueError(f"Teacher probability length {len(teacher_probs)} != dataset length {len(base)}")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        x, y = self.base[index]
        if self.teacher_probs is None:
            return x, y
        return x, y, torch.from_numpy(np.asarray(self.teacher_probs[index], dtype=np.float32))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a baseline-inspired LogicLSTM student with optional distillation.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    return parser.parse_args()


def apply_cli_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg = {k: dict(v) if isinstance(v, dict) else v for k, v in cfg.items()}
    if args.output_dir is not None:
        cfg["training"]["output_dir"] = args.output_dir
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers
    return cfg


def build_dataset(data_cfg: dict[str, Any], split: str, max_samples: int | None, seed: int, train: bool) -> GearXAIWindows:
    return GearXAIWindows(
        split=split,
        dataset_name=data_cfg["dataset_name"],
        config_name=data_cfg["config_name"],
        cache_dir=data_cfg.get("cache_dir"),
        max_samples=max_samples,
        normalize=False,  # Match official LogicLSTM deployment input.
        augment=bool(data_cfg.get("augment", False)) if train else False,
        noise_std=float(data_cfg.get("noise_std", 0.0)) if train else 0.0,
        scale_range=float(data_cfg.get("scale_range", 0.0)) if train else 0.0,
        time_shift=int(data_cfg.get("time_shift", 0)) if train else 0,
        channel_dropout=float(data_cfg.get("channel_dropout", 0.0)) if train else 0.0,
        seed=seed,
    )


def load_teacher_cache(cache_dir: str | None, train_len: int, val_len: int, alpha: float) -> tuple[np.ndarray | None, np.ndarray | None]:
    if alpha <= 0:
        return None, None
    if not cache_dir:
        raise ValueError("distill_alpha > 0 requires training.teacher_cache_dir")
    cache_path = Path(cache_dir)
    train_probs = np.load(cache_path / "train_teacher_probs.npy", mmap_mode="r")
    val_probs = np.load(cache_path / "val_teacher_probs.npy", mmap_mode="r")
    if train_probs.shape != (train_len, 9) or val_probs.shape != (val_len, 9):
        raise ValueError(
            f"Teacher cache shapes do not match datasets: train={train_probs.shape}/{train_len}, val={val_probs.shape}/{val_len}. "
            "Regenerate cache with this exact config."
        )
    return train_probs, val_probs


def distillation_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    teacher_probs: torch.Tensor | None,
    alpha: float,
    temperature: float,
    label_smoothing: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ce = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
    if alpha <= 0 or teacher_probs is None:
        zero = ce.new_tensor(0.0)
        return ce, ce, zero
    t = float(temperature)
    teacher_logits_proxy = teacher_probs.clamp_min(1e-8).log()
    soft_teacher = F.softmax(teacher_logits_proxy / t, dim=1)
    kd = F.kl_div(F.log_softmax(logits / t, dim=1), soft_teacher, reduction="batchmean") * (t * t)
    total = (1.0 - alpha) * ce + alpha * kd
    return total, ce, kd


def train_one_epoch(model, loader, optimizer, scaler, device, train_cfg):
    model.train()
    total_loss = total_ce = total_kd = 0.0
    total_correct = total_seen = 0
    alpha = float(train_cfg.get("distill_alpha", 0.0))
    temperature = float(train_cfg.get("distill_temperature", 2.0))
    smoothing = float(train_cfg.get("label_smoothing", 0.0))
    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"

    for batch in tqdm(loader, desc="train", leave=False):
        x, y = batch[:2]
        teacher = batch[2] if len(batch) == 3 else None
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        teacher = teacher.to(device, non_blocking=True) if teacher is not None else None
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits, _ = model.forward_train(x)
            loss, ce, kd = distillation_loss(logits, y, teacher, alpha, temperature, smoothing)
        scaler.scale(loss).backward()
        if train_cfg.get("grad_clip_norm"):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip_norm"]))
        scaler.step(optimizer)
        scaler.update()
        n = x.size(0)
        total_loss += float(loss.detach()) * n
        total_ce += float(ce.detach()) * n
        total_kd += float(kd.detach()) * n
        total_correct += int((logits.argmax(dim=1) == y).sum())
        total_seen += n
    return {"loss": total_loss / total_seen, "ce": total_ce / total_seen, "kd": total_kd / total_seen, "accuracy": total_correct / total_seen}


@torch.no_grad()
def evaluate(model, loader, device, train_cfg, num_classes):
    model.eval()
    total_loss = total_ce = total_kd = 0.0
    total_correct = total_seen = 0
    preds: list[int] = []
    labels: list[int] = []
    alpha = float(train_cfg.get("distill_alpha", 0.0))
    temperature = float(train_cfg.get("distill_temperature", 2.0))
    smoothing = float(train_cfg.get("label_smoothing", 0.0))
    for batch in tqdm(loader, desc="val", leave=False):
        x, y = batch[:2]
        teacher = batch[2] if len(batch) == 3 else None
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        teacher = teacher.to(device, non_blocking=True) if teacher is not None else None
        logits, _ = model.forward_train(x)
        loss, ce, kd = distillation_loss(logits, y, teacher, alpha, temperature, smoothing)
        pred = logits.argmax(dim=1)
        n = x.size(0)
        total_loss += float(loss) * n
        total_ce += float(ce) * n
        total_kd += float(kd) * n
        total_correct += int((pred == y).sum())
        total_seen += n
        preds.extend(pred.cpu().tolist())
        labels.extend(y.cpu().tolist())
    ids = list(range(num_classes))
    return {
        "loss": total_loss / total_seen,
        "ce": total_ce / total_seen,
        "kd": total_kd / total_seen,
        "accuracy": total_correct / total_seen,
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "per_class_f1": [float(v) for v in f1_score(labels, preds, average=None, labels=ids, zero_division=0)],
        "confusion_matrix": confusion_matrix(labels, preds, labels=ids).astype(int).tolist(),
    }


def save_checkpoint(path: Path, model, optimizer, cfg, epoch, metrics) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "config": cfg, "epoch": epoch, "metrics": metrics}, path)


def main() -> None:
    args = parse_args()
    cfg = apply_cli_overrides(load_config(args.config), args)
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    out = Path(train_cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "config.resolved.json", cfg)
    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)
    device = choose_device()

    train_base = build_dataset(data_cfg, data_cfg["train_split"], data_cfg.get("max_train_samples"), seed, train=True)
    val_base = build_dataset(data_cfg, data_cfg["val_split"], data_cfg.get("max_val_samples"), seed + 1, train=False)
    alpha = float(train_cfg.get("distill_alpha", 0.0))
    train_probs, val_probs = load_teacher_cache(train_cfg.get("teacher_cache_dir"), len(train_base), len(val_base), alpha)
    train_set = DistillationDataset(train_base, train_probs)
    val_set = DistillationDataset(val_base, val_probs)
    workers = int(data_cfg.get("num_workers", 0))
    batch_size = int(train_cfg["batch_size"])
    loader_kwargs = {"batch_size": batch_size, "num_workers": workers, "pin_memory": device.type == "cuda", "drop_last": False}
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)

    model = build_lstm_student(cfg["model"]).to(device)
    optimizer = AdamW(model.parameters(), lr=float(train_cfg["lr"]), weight_decay=float(train_cfg.get("weight_decay", 0.0)))
    epochs = int(train_cfg["epochs"])
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=float(train_cfg["lr"]) * float(train_cfg.get("cosine_min_lr_ratio", 0.05)))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(train_cfg.get("amp", True)) and device.type == "cuda")

    print(f"Device: {device}")
    print(f"Trainable parameters: {count_parameters(model):,}")
    print(f"Train examples: {len(train_set):,}, val examples: {len(val_set):,}")
    print(f"Distillation alpha={alpha:.2f}, temperature={float(train_cfg.get('distill_temperature', 2.0)):.2f}")
    best_f1 = -1.0
    for epoch in range(1, epochs + 1):
        start = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, device, train_cfg)
        val_metrics = evaluate(model, val_loader, device, train_cfg, int(cfg["model"]["num_classes"]))
        scheduler.step()
        row = {"epoch": epoch, "seconds": round(time.time() - start, 2), "lr": float(optimizer.param_groups[0]["lr"]), "train": train_metrics, "val": val_metrics}
        append_jsonl(out / "metrics.jsonl", row)
        save_json(out / "last_metrics.json", row)
        save_checkpoint(out / "last.pt", model, optimizer, cfg, epoch, val_metrics)
        if bool(train_cfg.get("save_every_epoch", True)):
            save_checkpoint(out / f"epoch_{epoch:03d}.pt", model, optimizer, cfg, epoch, val_metrics)
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            save_checkpoint(out / "best.pt", model, optimizer, cfg, epoch, val_metrics)
            save_json(out / "best_metrics.json", row)
        print(
            f"epoch {epoch:03d}/{epochs} train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['macro_f1']:.4f}"
        )
    print(f"Best validation macro F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
