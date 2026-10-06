"""Training, checkpoint loading, and prediction using injected notebook callables."""

from __future__ import annotations

import hashlib
import inspect
import json
import random
import subprocess
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import TrainConfig, Workspace, write_json
from .data import FaceDataset, make_loader
from .metrics import classification_report
from .models import optimizer_parameter_groups
from .predictions import PredictionTable
from .resources import sha256_file

ModelFactory = Callable[..., nn.Module]
ViewFunction = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True, slots=True)
class RunInfo:
    """Paths and validation results for a completed branch run."""

    run_id: str
    branch: str
    run_dir: Path
    checkpoint_path: Path
    curves: pd.DataFrame
    val_predictions: PredictionTable
    metadata: Mapping[str, Any]

    def summary(self) -> dict[str, Any]:
        """Return compact run information for a notebook."""
        return {"run_id": self.run_id, "branch": self.branch, "epochs": len(self.curves),
                "checkpoint": str(self.checkpoint_path),
                "validation_rows": len(self.val_predictions.rows),
                "last_validation_macro_f1": float(self.curves.val_macro_f1.iloc[-1]) if len(self.curves) else None}


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _code_payload(code: types.CodeType) -> Any:
    """Describe executable code without notebook filename/line-number noise."""
    constants = [
        _code_payload(value) if isinstance(value, types.CodeType) else repr(value)
        for value in code.co_consts
    ]
    return {"bytecode": code.co_code.hex(), "constants": constants, "names": code.co_names,
            "variables": code.co_varnames, "freevars": code.co_freevars, "cellvars": code.co_cellvars,
            "argcount": code.co_argcount, "kwonly": code.co_kwonlyargcount, "flags": code.co_flags}


def _stable_value(value: Any, seen: set[int] | None = None) -> Any:
    seen = seen or set()
    if id(value) in seen:
        return "<recursive>"
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if inspect.isfunction(value):
        return {"function": value.__qualname__, "code": _code_payload(value.__code__)}
    if isinstance(value, types.ModuleType):
        return {"module": value.__name__}
    if isinstance(value, type):
        return {"type": f"{value.__module__}.{value.__qualname__}"}
    if isinstance(value, Mapping):
        nested = seen | {id(value)}
        return {str(key): _stable_value(value[key], nested) for key in sorted(value, key=str)}
    if isinstance(value, (tuple, list)):
        nested = seen | {id(value)}
        return [_stable_value(item, nested) for item in value]
    return repr(value)


def _callable_digest(function: Callable[..., Any], spec: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    payload: dict[str, Any] = {"spec": _stable_value(spec)}
    if hasattr(function, "__code__"):
        payload["code"] = _code_payload(function.__code__)
        dependencies = {}
        namespace = getattr(function, "__globals__", {})
        for name in function.__code__.co_names:
            if name in namespace:
                value = namespace[name]
                if inspect.isfunction(value) or isinstance(value, (Mapping, types.ModuleType, type)):
                    dependencies[name] = _stable_value(value)
        payload["dependencies"] = dependencies
    else:
        payload["callable"] = repr(function)
    digest.update(json.dumps(payload, sort_keys=True, default=str).encode())
    for cell in getattr(function, "__closure__", ()) or ():
        try:
            digest.update(json.dumps(_stable_value(cell.cell_contents), sort_keys=True, default=str).encode())
        except ValueError:
            digest.update(b"<empty>")
    return digest.hexdigest()


def _rows_digest(rows: pd.DataFrame) -> str:
    ordered = rows.sort_values("file_name").reset_index(drop=True)
    payload = ordered[[name for name in ("file_name", "label", "fold") if name in ordered]].to_csv(index=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _split_digest(train_rows: pd.DataFrame, val_rows: pd.DataFrame) -> str:
    payload = f"train={_rows_digest(train_rows)}\nvalidation={_rows_digest(val_rows)}".encode()
    return hashlib.sha256(payload).hexdigest()


def _source_revision(task_root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(task_root), "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _resolved_config(
    workspace: Workspace,
    cfg: TrainConfig,
    branch: str,
    train_rows: pd.DataFrame,
    val_rows: pd.DataFrame,
    model_factory: ModelFactory,
    view_fn: ViewFunction,
    view_spec: Mapping[str, Any],
    model_spec: Mapping[str, Any],
    sample_weights: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    if dict(cfg.model) != dict(model_spec):
        raise ValueError("TrainConfig.model and model_spec must describe the same model.")
    expected_view = dict(cfg.view)
    if any(key in view_spec and view_spec.get(key) != value for key, value in expected_view.items()):
        raise ValueError("TrainConfig.view and view_spec must describe the same input transform.")
    recipe = {"view": dict(view_spec), "model": dict(model_spec)}
    if sample_weights is not None:
        if set(sample_weights) != set(train_rows.file_name):
            raise ValueError("Sample weights must cover exactly the training file names.")
        values = {name: float(sample_weights[name]) for name in sorted(sample_weights)}
        if any(not np.isfinite(value) or value <= 0 for value in values.values()):
            raise ValueError("Sample weights must be finite and positive.")
        recipe["sample_weights"] = values
    return {
        "schema_version": 1,
        "run_id": workspace.run_id,
        "branch": branch,
        "train_config": cfg.to_dict(),
        "recipe": recipe,
        "model_factory_sha256": _callable_digest(model_factory, model_spec),
        "view_fn_sha256": _callable_digest(view_fn, view_spec),
        "train_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "train_ids_sha256": _rows_digest(train_rows),
        "validation_ids_sha256": _rows_digest(val_rows),
        "split_sha256": _split_digest(train_rows, val_rows),
        "environment": {"torch": torch.__version__, "cuda": torch.version.cuda},
        "source_revision": _source_revision(workspace.root),
    }


def _save_curves(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".partial")
    pd.DataFrame(records).to_csv(temporary, index=False)
    temporary.replace(path)


def _normalize(images: torch.Tensor, cfg: TrainConfig) -> torch.Tensor:
    mean = images.new_tensor(cfg.normalize_mean).view(1, 3, 1, 1)
    std = images.new_tensor(cfg.normalize_std).view(1, 3, 1, 1)
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"view_fn must return BCHW RGB, received {tuple(images.shape)}")
    if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
        raise ValueError("view_fn must return finite values in [0, 1] before normalization.")
    return (images - mean) / std


def _view_batch(batch: torch.Tensor, view_fn: ViewFunction, cfg: TrainConfig, augment: bool) -> torch.Tensor:
    if augment:
        probability = float(cfg.augmentation.get("horizontal_flip_probability", 0.0))
        if not 0 <= probability <= 1:
            raise ValueError("horizontal_flip_probability must be in [0, 1].")
        flips = torch.rand(batch.shape[0], device=batch.device) < probability
        batch = torch.where(flips[:, None, None, None], batch.flip(-1), batch)
    viewed = view_fn(batch)
    if not isinstance(viewed, torch.Tensor) or viewed.shape[0] != batch.shape[0]:
        raise ValueError("view_fn must return a tensor with the same batch size.")
    return _normalize(viewed, cfg)


def _validation_predictions(
    model: nn.Module,
    loader: DataLoader,
    cfg: TrainConfig,
    view_fn: ViewFunction,
    device: torch.device,
) -> tuple[PredictionTable, float, float]:
    model.eval()
    names: list[str] = []
    labels: list[int] = []
    probabilities: list[float] = []
    losses = 0.0
    with torch.no_grad():
        for images, target, file_names in loader:
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            prepared = _view_batch(images, view_fn, cfg, augment=False)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(prepared)
                loss = nn.functional.cross_entropy(logits, target)
            probability = logits.float().softmax(dim=1)[:, 1]
            names.extend(file_names)
            labels.extend(target.cpu().tolist())
            probabilities.extend(probability.cpu().tolist())
            losses += float(loss.detach()) * len(target)
    table = PredictionTable(pd.DataFrame({"file_name": names, "label": labels, "prob": probabilities}))
    report = classification_report(labels, probabilities)
    return table, losses / len(labels), float(report["macro_f1"])


def _checkpoint_state(model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: Any,
                      scaler: Any, epoch: int) -> dict[str, Any]:
    return {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch,
        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
    }


def _optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    groups = optimizer_parameter_groups(model, cfg.backbone_lr, cfg.head_lr)
    return torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)


def _run_directory(workspace: Workspace, branch: str, resolved: dict[str, Any]) -> Path:
    base = workspace.artifact_root / branch
    config = base / "config.json"
    if not config.is_file() or json.loads(config.read_text()) == resolved:
        return base
    suffix = hashlib.sha256(json.dumps(resolved, sort_keys=True).encode()).hexdigest()[:12]
    return workspace.artifact_root / f"{branch}_{suffix}"


def fit_fold(
    workspace: Workspace,
    cfg: TrainConfig,
    branch: str,
    train_rows: pd.DataFrame,
    val_rows: pd.DataFrame,
    model_factory: ModelFactory,
    view_fn: ViewFunction,
    view_spec: Mapping[str, Any],
    model_spec: Mapping[str, Any],
    sample_weights: Mapping[str, float] | None = None,
) -> RunInfo:
    """Fit one branch on the provided training rows and validate each epoch."""
    if not torch.cuda.is_available():
        raise RuntimeError("fit_fold needs a CUDA GPU. Use load_run for a released checkpoint.")
    if branch not in {"rgb", "highpass", "resampled", "wavelet", "edge_weighted"}:
        raise ValueError("Unknown training branch.")
    if train_rows.empty or val_rows.empty or set(train_rows.file_name) & set(val_rows.file_name):
        raise ValueError("Training and validation rows must be nonempty and disjoint.")
    resolved = _resolved_config(workspace, cfg, branch, train_rows, val_rows, model_factory, view_fn, view_spec, model_spec, sample_weights)
    sample_weights = resolved["recipe"].get("sample_weights")
    run_dir = _run_directory(workspace, branch, resolved)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Lượt chạy {branch}: {run_dir}", flush=True)
    config_path = run_dir / "config.json"
    checkpoint_path = run_dir / "last.pt"
    if checkpoint_path.exists() and not config_path.exists():
        raise ValueError(f"Checkpoint has no saved recipe; move it aside or use a new run_id: {run_dir}")
    if config_path.exists() and json.loads(config_path.read_text()) != resolved:
        raise ValueError(f"Run configuration changed; choose a new run_id. Existing run: {run_dir}")
    status_path = run_dir / "run_status.json"
    status = json.loads(status_path.read_text()) if status_path.is_file() else {}
    if status.get("status") == "complete" and checkpoint_path.is_file():
        print("Đã có lượt chạy hoàn tất; đang nạp kết quả.", flush=True)
        return load_run(workspace, cfg, branch, train_rows, val_rows, model_factory,
                        view_fn, view_spec, model_spec, checkpoint_source=checkpoint_path, sample_weights=sample_weights)
    write_json(config_path, resolved)
    device = torch.device("cuda")
    _seed(cfg.seed)
    model = model_factory(initialize=not checkpoint_path.is_file()).to(device)
    optimizer = _optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler("cuda")
    validation_loader = make_loader(val_rows, cfg.batch_size, labeled=True, shuffle=False, seed=cfg.seed)
    curves_path = run_dir / "curves.csv"
    start_epoch = 0
    records = pd.read_csv(curves_path).to_dict("records") if curves_path.is_file() else []
    if checkpoint_path.is_file():
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        torch.set_rng_state(state["torch_rng"].cpu())
        if state.get("cuda_rng"):
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        start_epoch = int(state["epoch"])
        records = [record for record in records if int(record["epoch"]) <= start_epoch]
    write_json(run_dir / "run_status.json", {"status": "running", "start_epoch": start_epoch, "config_sha256": sha256_file(config_path)})
    for epoch in range(start_epoch, cfg.epochs):
        tick = time.monotonic()
        model.train()
        epoch_train = make_loader(train_rows, cfg.batch_size, labeled=True, shuffle=True, seed=cfg.seed + epoch)
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        correct = 0
        seen = 0
        accumulation = cfg.accumulation
        for step, (images, target, file_names) in enumerate(epoch_train):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            prepared = _view_batch(images, view_fn, cfg, augment=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                logits = model(prepared)
                if sample_weights is not None:
                    losses = nn.functional.cross_entropy(logits, target, label_smoothing=cfg.label_smoothing, reduction="none")
                    weights = losses.new_tensor([sample_weights[name] for name in file_names])
                    loss = (losses * weights).mean()
                else:
                    loss = nn.functional.cross_entropy(logits, target, label_smoothing=cfg.label_smoothing)
            loss_value = float(loss.detach())
            group_start = (step // accumulation) * accumulation
            batches_per_group = min(accumulation, len(epoch_train) - group_start)
            scaler.scale(loss / batches_per_group).backward()
            boundary = (step + 1) % accumulation == 0 or step + 1 == len(epoch_train)
            if boundary:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            total_loss += loss_value * len(target)
            correct += int((logits.detach().argmax(1) == target).sum())
            seen += len(target)
        scheduler.step()
        predictions, val_loss, val_f1 = _validation_predictions(model, validation_loader, cfg, view_fn, device)
        records.append({"epoch": epoch + 1, "train_loss": total_loss / seen, "train_accuracy": correct / seen,
                        "val_loss": val_loss, "val_macro_f1": val_f1, "seconds": time.monotonic() - tick,
                        "backbone_lr": optimizer.param_groups[0]["lr"], "head_lr": optimizer.param_groups[1]["lr"]})
        _save_curves(curves_path, records)
        pending_checkpoint = checkpoint_path.with_suffix(".pt.partial")
        torch.save(_checkpoint_state(model, optimizer, scheduler, scaler, epoch + 1), pending_checkpoint)
        pending_checkpoint.replace(checkpoint_path)
        print(f"{branch} | epoch {epoch + 1}/{cfg.epochs} | loss {total_loss / seen:.4f} | "
              f"val F1 {val_f1:.4f} | {records[-1]['seconds']:.1f}s", flush=True)
    if len(records) < cfg.epochs:
        raise RuntimeError("Training ended before the configured terminal epoch.")
    prediction_meta = {
        "source": "learner_run", "run_id": workspace.run_id, "branch": branch,
        "checkpoint_sha256": sha256_file(checkpoint_path), "config_sha256": sha256_file(config_path),
        "split_sha256": resolved["split_sha256"], "validation_ids_sha256": resolved["validation_ids_sha256"],
        "validation_manifest": "validation_manifest.csv",
        "model_spec": dict(model_spec), "view_spec": dict(view_spec), "label_map": {"0": "Real", "1": "Fake"},
    }
    predictions, _, _ = _validation_predictions(model, validation_loader, cfg, view_fn, device)
    predictions = PredictionTable(predictions.rows, prediction_meta)
    predictions.save(run_dir / "val_predictions.csv")
    pd.DataFrame(val_rows).to_csv(run_dir / "validation_manifest.csv", index=False)
    write_json(run_dir / "run_status.json", {"status": "complete", "epochs": cfg.epochs,
                                             "checkpoint_sha256": prediction_meta["checkpoint_sha256"]})
    write_json(workspace.artifact_root / f"{branch}_active.json", {"directory": run_dir.name})
    result = RunInfo(workspace.run_id, branch, run_dir, checkpoint_path, pd.DataFrame(records), predictions, prediction_meta)
    del model, optimizer, scheduler, scaler
    torch.cuda.empty_cache()
    return result


def _load_checkpoint(
    workspace: Workspace,
    cfg: TrainConfig,
    branch: str,
    train_rows: pd.DataFrame,
    val_rows: pd.DataFrame,
    model_factory: ModelFactory,
    view_fn: ViewFunction,
    view_spec: Mapping[str, Any],
    model_spec: Mapping[str, Any],
    checkpoint_source: str | Path | None,
    sample_weights: Mapping[str, float] | None = None,
) -> tuple[RunInfo, nn.Module, torch.device]:
    resolved = _resolved_config(workspace, cfg, branch, train_rows, val_rows, model_factory, view_fn, view_spec, model_spec, sample_weights)
    run_dir = _run_directory(workspace, branch, resolved)
    checkpoint = Path(checkpoint_source) if checkpoint_source else run_dir / "last.pt"
    if not checkpoint.is_absolute():
        checkpoint = workspace.root / checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(f"No checkpoint for {branch}: {checkpoint}; train this branch or configure its release asset.")
    resolved = _resolved_config(workspace, cfg, branch, train_rows, val_rows, model_factory, view_fn, view_spec, model_spec, sample_weights)
    config_path = checkpoint.parent / "config.json"
    if not config_path.is_file() or json.loads(config_path.read_text()) != resolved:
        raise ValueError("Checkpoint recipe/split differs from the current notebook; choose the matching config and callables.")
    status_path = checkpoint.parent / "run_status.json"
    status = json.loads(status_path.read_text()) if status_path.is_file() else {}
    if (status.get("status") != "complete" or int(status.get("epochs", -1)) != cfg.epochs
            or status.get("checkpoint_sha256") != sha256_file(checkpoint)):
        raise ValueError("Checkpoint is incomplete or was not saved at the configured terminal epoch.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_factory(initialize=False).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    metadata = {"checkpoint_sha256": sha256_file(checkpoint), "config_sha256": sha256_file(config_path),
                "view_spec": dict(view_spec), "model_spec": dict(model_spec), "split_sha256": resolved["split_sha256"]}
    return RunInfo(workspace.run_id, branch, checkpoint.parent, checkpoint,
                   pd.read_csv(checkpoint.parent / "curves.csv"),
                   PredictionTable.load(checkpoint.parent / "val_predictions.csv"),
                   metadata), model, device


def load_run(
    workspace: Workspace,
    cfg: TrainConfig,
    branch: str,
    train_rows: pd.DataFrame,
    val_rows: pd.DataFrame,
    model_factory: ModelFactory,
    view_fn: ViewFunction,
    view_spec: Mapping[str, Any],
    model_spec: Mapping[str, Any],
    checkpoint_source: str | Path | None = None,
    sample_weights: Mapping[str, float] | None = None,
) -> RunInfo:
    """Load a completed branch after exact config and callable checks."""
    run, model, device = _load_checkpoint(workspace, cfg, branch, train_rows, val_rows,
                                          model_factory, view_fn, view_spec, model_spec, checkpoint_source, sample_weights)
    if not run.val_predictions.rows.file_name.isin(val_rows.file_name).all() or len(run.val_predictions.rows) != len(val_rows):
        raise ValueError("Stored validation predictions do not cover the current fold.")
    write_json(workspace.artifact_root / f"{branch}_active.json", {"directory": run.run_dir.name})
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return run


def predict(
    run: RunInfo,
    rows: pd.DataFrame,
    model_factory: ModelFactory,
    view_fn: ViewFunction,
    view_spec: Mapping[str, Any],
    model_spec: Mapping[str, Any],
    decision: Any = None,
) -> PredictionTable:
    """Predict P(Fake) for each row using a completed checkpoint and injected view."""
    if rows.empty or not {"file_name", "path"}.issubset(rows.columns) or not rows.file_name.is_unique:
        raise ValueError("Prediction rows need unique file_name and path columns.")
    saved = json.loads((run.run_dir / "config.json").read_text())
    if saved["model_factory_sha256"] != _callable_digest(model_factory, model_spec):
        raise ValueError("model_factory differs from the trained checkpoint recipe.")
    if saved["view_fn_sha256"] != _callable_digest(view_fn, view_spec):
        raise ValueError("view_fn or its declared parameters differ from the trained recipe.")
    if decision is not None:
        branch = decision.weights.get(run.branch)
        recorded = decision.path
        payload = json.loads(recorded.read_text()) if recorded.is_file() else {}
        receipt = payload.get("branches", {}).get(run.branch, {})
        if branch is None or receipt.get("checkpoint_sha256") != sha256_file(run.checkpoint_path):
            raise ValueError("The decision does not select this branch checkpoint.")
    cfg = TrainConfig(**{**saved["train_config"], "normalize_mean": tuple(saved["train_config"]["normalize_mean"]),
                         "normalize_std": tuple(saved["train_config"]["normalize_std"])})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_factory(initialize=False).to(device)
    state = torch.load(run.checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    loader = make_loader(rows, cfg.batch_size, labeled=False, shuffle=False, seed=cfg.seed)
    model.eval()
    names: list[str] = []
    probs: list[float] = []
    with torch.no_grad():
        for images, _, file_names in loader:
            images = images.to(device, non_blocking=True)
            prepared = _view_batch(images, view_fn, cfg, augment=False)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(prepared).float()
                probability = logits.softmax(1)[:, 1]
                if cfg.tta:
                    flipped = _view_batch(images.flip(-1), view_fn, cfg, augment=False)
                    probability = 0.5 * (probability + model(flipped).float().softmax(1)[:, 1])
            names.extend(file_names)
            probs.extend(probability.cpu().tolist())
    metadata = {"source": "learner_run", "run_id": run.run_id, "branch": run.branch,
                "checkpoint_sha256": sha256_file(run.checkpoint_path), "model_spec": dict(model_spec),
                "view_spec": dict(view_spec), "decision_id": getattr(decision, "decision_id", None),
                "label_map": {"0": "Real", "1": "Fake"}}
    table = PredictionTable(pd.DataFrame({"file_name": names, "prob": probs}), metadata)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return table
