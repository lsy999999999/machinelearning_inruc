from pathlib import Path

path = Path("src/gearxai_project/train_causal.py")
text = path.read_text(encoding="utf-8")

# 1. 增加从已有自研 checkpoint 初始化的参数
old_arg = '    parser.add_argument("--ema-decay", type=float)\n'
new_arg = (
    '    parser.add_argument("--ema-decay", type=float)\n'
    '    parser.add_argument("--init-checkpoint", help="Initialize the causal model from an existing self-trained checkpoint.")\n'
)
if old_arg not in text:
    raise RuntimeError("Cannot find parser insertion point.")
text = text.replace(old_arg, new_arg, 1)

# 2. 替换训练 epoch：加入可微 causal deletion / insertion loss
start = text.index("def train_one_epoch(")
end = text.index("@torch.no_grad()\ndef evaluate", start)

new_train_code = r'''
def _normalise_perturbed_windows(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Mimic the per-window normalization embedded in the exported ONNX graph.

    During official faithfulness evaluation, deleted/inserted raw windows are
    passed through the exported input normalization again. Re-normalising here
    makes causal training closer to that evaluation process.
    """
    centered = x - x.mean(dim=-1, keepdim=True)
    variance = (centered * centered).mean(dim=-1, keepdim=True)
    return centered / torch.sqrt(variance).clamp_min(eps)


def _soft_top_fraction_mask(
    relevance: torch.Tensor,
    fraction: float,
    temperature: float,
) -> torch.Tensor:
    """Differentiable approximation to evaluator-style top-k masking.

    The evaluator ranks relevance cells and masks the most important portion.
    A hard ranking would block gradients to the relevance branch, therefore
    this function uses a detached top-k threshold plus a sigmoid transition.
    """
    batch_size = relevance.size(0)
    flat = relevance.reshape(batch_size, -1)
    total_cells = flat.size(1)

    k = max(1, min(int(round(float(fraction) * total_cells)), total_cells - 1))
    threshold = torch.topk(flat.detach(), k=k, dim=1).values[:, -1]
    threshold = threshold.view(batch_size, 1, 1)

    scale = flat.detach().std(dim=1, keepdim=True).view(batch_size, 1, 1)
    scale = scale.clamp_min(1e-3)

    return torch.sigmoid((relevance - threshold) / (float(temperature) * scale))


def _causal_relevance_loss(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    logits: torch.Tensor,
    relevance: torch.Tensor,
    train_cfg: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Optimize the relevance map in the same direction as faithfulness scoring.

    Important locations should satisfy:
    - deletion: removing them decreases true-class confidence;
    - insertion: keeping them preserves true-class confidence.
    """
    delete_weight = float(train_cfg.get("causal_delete_weight", 0.0))
    insert_weight = float(train_cfg.get("causal_insert_weight", 0.0))

    if delete_weight <= 0.0 and insert_weight <= 0.0:
        zero = logits.new_tensor(0.0)
        return zero, zero, zero, 0.0

    fractions = train_cfg.get("causal_fractions", [0.10])
    if not fractions:
        fractions = [0.10]
    fraction = float(fractions[np.random.randint(0, len(fractions))])

    temperature = float(train_cfg.get("causal_temperature", 0.15))
    delete_margin = float(train_cfg.get("causal_delete_margin", 0.08))
    insert_tolerance = float(train_cfg.get("causal_insert_tolerance", 0.10))
    renormalise = bool(train_cfg.get("causal_renormalize", True))

    mask = _soft_top_fraction_mask(
        relevance=relevance,
        fraction=fraction,
        temperature=temperature,
    )

    deleted_x = x * (1.0 - mask)
    inserted_x = x * mask

    if renormalise:
        deleted_x = _normalise_perturbed_windows(deleted_x)
        inserted_x = _normalise_perturbed_windows(inserted_x)

    deleted_logits, _ = model.forward_train(deleted_x)
    inserted_logits, _ = model.forward_train(inserted_x)

    label_index = y.unsqueeze(1)
    full_conf = F.softmax(logits.detach(), dim=1).gather(1, label_index).squeeze(1)
    deleted_conf = F.softmax(deleted_logits, dim=1).gather(1, label_index).squeeze(1)
    inserted_conf = F.softmax(inserted_logits, dim=1).gather(1, label_index).squeeze(1)

    # Deleted confidence should be lower than full confidence by at least margin.
    delete_loss = F.relu(deleted_conf - full_conf + delete_margin).mean()

    # Inserted confidence should stay close to full confidence.
    insert_loss = F.relu(full_conf - inserted_conf - insert_tolerance).mean()

    total = delete_weight * delete_loss + insert_weight * insert_loss
    return total, delete_loss, insert_loss, fraction


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
    total_cls = 0.0
    total_reg = 0.0
    total_causal = 0.0
    total_delete = 0.0
    total_insert = 0.0
    total_correct = 0
    total_seen = 0

    grad_clip_norm = train_cfg.get("grad_clip_norm")
    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"

    progress = tqdm(loader, desc="train-causal", leave=False)
    for x, y in progress:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits, _ = model.forward_train(x)

            # Align training with the ONNX export mode used for spectral_lite_c:
            # the final exported relevance is model.input_relevance(...).
            relevance = model.input_relevance(x)

            cls_loss = criterion(logits, y)
            reg_loss = relevance_regularization(
                relevance,
                sparse_weight=float(train_cfg.get("sparse_weight", 0.0)),
                tv_weight=float(train_cfg.get("tv_weight", 0.0)),
            )
            causal_loss, delete_loss, insert_loss, fraction = _causal_relevance_loss(
                model=model,
                x=x,
                y=y,
                logits=logits,
                relevance=relevance,
                train_cfg=train_cfg,
            )

            loss = cls_loss + reg_loss + causal_loss

        scaler.scale(loss).backward()

        if grad_clip_norm:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))

        scaler.step(optimizer)
        scaler.update()

        ema = train_cfg.get("_ema")
        if ema is not None:
            ema.update(model)

        n = x.size(0)
        total_loss += float(loss.detach()) * n
        total_cls += float(cls_loss.detach()) * n
        total_reg += float(reg_loss.detach()) * n
        total_causal += float(causal_loss.detach()) * n
        total_delete += float(delete_loss.detach()) * n
        total_insert += float(insert_loss.detach()) * n
        total_correct += int((logits.argmax(dim=1) == y).sum())
        total_seen += n

        progress.set_postfix(
            loss=total_loss / max(total_seen, 1),
            acc=total_correct / max(total_seen, 1),
            causal=total_causal / max(total_seen, 1),
            frac=fraction,
        )

    denom = max(total_seen, 1)
    return {
        "loss": total_loss / denom,
        "classification_loss": total_cls / denom,
        "regularization_loss": total_reg / denom,
        "causal_loss": total_causal / denom,
        "deletion_loss": total_delete / denom,
        "insertion_loss": total_insert / denom,
        "accuracy": total_correct / denom,
    }


'''

text = text[:start] + new_train_code + text[end:]

# 3. 在构建自研模型后加载 spectral_lite_c 的已有 checkpoint
old_model_line = '    model = build_model(cfg["model"]).to(device)\n'
new_model_line = (
    '    model = build_model(cfg["model"]).to(device)\n'
    '    if args.init_checkpoint is not None:\n'
    '        init_payload = torch.load(args.init_checkpoint, map_location="cpu")\n'
    '        model.load_state_dict(init_payload["model_state"], strict=True)\n'
    '        print(f"Initialized causal fine-tuning from: {args.init_checkpoint}")\n'
)

if old_model_line not in text:
    raise RuntimeError("Cannot find model construction insertion point.")
text = text.replace(old_model_line, new_model_line, 1)

path.write_text(text, encoding="utf-8")
print(f"Patched: {path}")
