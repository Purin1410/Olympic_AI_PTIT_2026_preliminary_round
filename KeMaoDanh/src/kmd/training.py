"""Single-fold training derived from trainer.fit; no queue or holdout entrypoint."""
from dataclasses import asdict, replace
from pathlib import Path
import hashlib
import json
import math
import time
import uuid

import numpy as np
import torch

from .config import Config
from .core import PACKAGE, split_fold, write_json, sha256, metric, read_csv, environment
from .data import Pairs, make_loader, fit_normalization
from .models import model_for, train_mode, head_name, logits, objective, predict_pairs


def atomic_save(path, value):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    torch.save(value, temp)
    temp.replace(path)


def train_step(model, x, y, optimizer, c, target_batch=None, clip_grad=5.0, perform_update=True, device="cuda"):
    """Single forward-backward computation step with optional gradient step.

    Shared between visible teaching cells (02_pretrained_and_finetune) and the full trainer.
    Gradient accumulation scales each microbatch loss by len(y) / target_batch.
    """
    target = target_batch if target_batch is not None else len(y)
    dev = torch.device(device) if isinstance(device, str) else device
    x = x.to(dev, non_blocking=True)
    y = y.to(dev, non_blocking=True)

    use_autocast = c.amp and dev.type == "cuda" and torch.cuda.is_bf16_supported()
    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=use_autocast):
        pred_logits = logits(model, x, c)
        loss = objective(pred_logits, y, c)

    scaled_loss = loss * (len(y) / target)
    scaled_loss.backward()

    grad_norm = None
    if perform_update:
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return float(loss.item()), grad_norm


def train_one_fold(c, runtime, run_name=None, train_frame=None, valid_frame=None, purpose="teaching_full_fold"):
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA. Use the LR notebook on CPU, or select a CUDA runtime.")
    if c.kind != "deep" or c.effective_batch % c.microbatch or c.mode not in {"full", "frozen", "partial"}:
        raise ValueError("Invalid CNN configuration.")
    if c.amp and not torch.cuda.is_bf16_supported():
        print("GPU does not support bfloat16; training in float32.", flush=True)
        c = replace(c, amp=False)
    data_root = runtime.require_data()
    if train_frame is None or valid_frame is None:
        raise ValueError("Pass training and validation frames loaded from actual pairs.csv.")
    tr, va = train_frame.reset_index(drop=True), valid_frame.reset_index(drop=True)
    for frame in (tr, va):
        if not len(frame) or not frame.pair_id.is_unique or not frame.fake_position.isin([0, 1]).all():
            raise ValueError("Empty, duplicated or incorrectly labeled input frame.")
    if (set(tr.image_0) | set(tr.image_1)) & (set(va.image_0) | set(va.image_1)):
        raise ValueError("Train/validation image overlap.")
    if set(tr.pair_id) & set(va.pair_id):
        raise ValueError("Train/validation overlap.")
    if purpose != "smoke" and (not tr.inner_fold.ne(c.fold).all() or not va.inner_fold.eq(c.fold).all()):
        raise ValueError("Full-fold run must honor the frozen validation fold.")
    name = run_name or time.strftime("full_%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    if Path(name).name != name:
        raise ValueError("run_name must be a single directory name.")
    folder = runtime.run_root / name
    folder.mkdir(parents=True, exist_ok=False)
    source_hashes = {p.name: sha256(p) for p in (PACKAGE / "src/kmd").glob("*.py")}
    contract = dict(id=name, config=asdict(c), train_ids=tr.pair_id.tolist(), validation_ids=va.pair_id.tolist(),
                    purpose=purpose, result_kind="new_run", code_sha256=source_hashes, holdout_evaluated=False)
    job_hash = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    contract["job_hash"] = job_hash
    write_json(folder / "job.json", contract)
    write_json(folder / "environment.json", environment())
    try:
        torch.set_num_threads(4)
        torch.cuda.reset_peak_memory_stats()
        wall_start = time.monotonic()
        model, weight_hash = model_for(c, device="cuda", pretrained=True)
        norm = fit_normalization(tr, data_root, c) if c.normalization == "train" else None
        head = head_name(c)
        opt = torch.optim.AdamW([
            dict(params=[p for n, p in model.named_parameters() if not n.startswith(head)], lr=c.backbone_lr),
            dict(params=[p for n, p in model.named_parameters() if n.startswith(head)], lr=c.head_lr),
        ], weight_decay=c.weight_decay)
        bn_before = {n: v.detach().cpu().clone() for n, v in model.named_buffers() if "running_" in n or "num_batches_tracked" in n}
        frozen_before = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if not n.startswith(head)} if c.mode == "frozen" else {}
        history = []
        best = -1.0
        best_epoch = 0
        bad = 0
        steps = 0
        cap = math.ceil(c.fixed_updates / math.ceil(len(tr) / c.effective_batch)) if c.fixed_updates else c.epochs
        begin = time.monotonic()
        for epoch in range(1, cap + 1):
            train_mode(model, c, epoch)
            start = time.monotonic()
            total = 0.0
            n = 0
            within = 0
            epoch_steps = 0
            ds = Pairs(tr, data_root, c, True, epoch, norm)
            loader = make_loader(ds, True)
            opt.zero_grad(set_to_none=True)
            target = min(c.effective_batch, len(tr))
            for x, y in loader:
                is_update = (within + len(y) == target)
                loss_val, _ = train_step(
                    model=model,
                    x=x,
                    y=y,
                    optimizer=opt,
                    c=c,
                    target_batch=target,
                    clip_grad=5.0,
                    perform_update=is_update,
                    device="cuda"
                )
                total += loss_val * len(y)
                n += len(y)
                within += len(y)
                if is_update:
                    steps += 1
                    epoch_steps += 1
                    within = 0
                    target = min(c.effective_batch, len(tr) - n)
                    if c.fixed_updates and steps >= c.fixed_updates:
                        break
            del loader, ds
            if not (c.fixed_updates and steps >= c.fixed_updates):
                assert epoch_steps == math.ceil(len(tr) / c.effective_batch)
            if not np.isfinite(total):
                raise FloatingPointError("Nonfinite training loss.")
            pred = predict_pairs(model, va, c, data_root, norm, device="cuda")
            score = metric(pred.y, pred.p)
            history.append(dict(epoch=epoch, train_loss=total / n, optimizer_steps=steps, epoch_updates=epoch_steps,
                                dev_f1=score["macro_f1"], seconds=time.monotonic() - start))
            terminal = c.fixed_epochs or bool(c.fixed_updates)
            improved = terminal or score["macro_f1"] > best + 1e-10
            if improved:
                best = score["macro_f1"]
                best_epoch = epoch
                bad = 0
                atomic_save(folder / "best.pt", dict(state_dict={n: v.detach().cpu() for n, v in model.state_dict().items()}, config=asdict(c), epoch=epoch, norm=norm, job_hash=job_hash))
                pred.to_csv(folder / "development.csv", index=False)
            else:
                bad += 1
            import pandas as pd
            pd.DataFrame(history).to_csv(folder / "history.csv", index=False)
            write_json(folder / "status.json", dict(status="running", epoch=epoch, best_epoch=best_epoch, best_dev_f1=best))
            print(f"{name}: epoch {epoch}/{cap}, loss={total/n:.4f}, current_F1={score['macro_f1']:.4f}, best={best:.4f}", flush=True)
            if (not terminal and bad >= c.patience and epoch >= c.warmup + 3) or (c.fixed_updates and steps >= c.fixed_updates):
                break
        for name_, before in bn_before.items():
            assert torch.equal(dict(model.named_buffers())[name_].cpu(), before), f"BatchNorm changed: {name_}"
        for name_, before in frozen_before.items():
            assert torch.equal(dict(model.named_parameters())[name_].cpu(), before), f"Frozen backbone changed: {name_}"
        state = torch.load(folder / "best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state["state_dict"])
        model.eval()
        sample = Pairs(va, data_root, c, norm=norm)[0][0].unsqueeze(0).cuda()
        with torch.inference_mode():
            a, b = logits(model, sample, c), logits(model, sample.flip(1), c)
            swap = float((torch.sigmoid(a[:, 1] - a[:, 0]) + torch.sigmoid(b[:, 1] - b[:, 0]) - 1).abs().max())
        assert swap <= 1e-6
        reloaded = predict_pairs(model, va, c, data_root, norm, device="cuda")
        recorded = read_csv(folder / "development.csv")
        reload_error = float(np.abs(reloaded.p - recorded.p).max())
        assert reload_error < 1e-6
        result = dict(status="passed", result_kind="new_run", purpose=purpose, best_epoch=best_epoch, completed_epochs=len(history),
                      train_pairs=len(tr), validation_pairs=len(va), optimizer_steps=steps,
                      elapsed_seconds=time.monotonic() - begin, wall_seconds_including_setup=time.monotonic() - wall_start,
                      peak_allocated_MiB=torch.cuda.max_memory_allocated() / 2**20, peak_reserved_MiB=torch.cuda.max_memory_reserved() / 2**20,
                      pair_swap_error=swap, reload_probability_max_abs_error=reload_error, bn_unchanged=True,
                      frozen_backbone_unchanged=True if c.mode == "frozen" else None,
                      pretrained_sha256=weight_hash, checkpoint_sha256=sha256(folder / "best.pt"), job_hash=job_hash,
                      holdout_evaluated=False, metrics=metric(recorded.y, recorded.p))
        write_json(folder / "metrics.json", result)
        write_json(folder / "status.json", dict(status="complete"))
        del model, opt
        torch.cuda.empty_cache()
        return folder, result
    except BaseException as exc:
        write_json(folder / "status.json", dict(status="failed", error=f"{type(exc).__name__}: {exc}"))
        raise
