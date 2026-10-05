"""Configuration objects and workspace paths for the AILAAI lessons."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_thaw(v) for v in value]
    return value


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON object from disk."""
    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {source}.")
    return value


def write_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    """Write JSON atomically, creating parent directories."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    temporary.write_text(json.dumps(_thaw(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Resolved settings for one training or inference branch."""

    seed: int = 2026
    epochs: int = 15
    batch_size: int = 16
    accumulation: int = 2
    backbone_lr: float = 1.5e-4
    head_lr: float = 7.5e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    checkpoint_policy: str = "terminal"
    model: Mapping[str, Any] = field(default_factory=lambda: {"backbone": "resnet34", "weights": "IMAGENET1K_V1"})
    view: Mapping[str, Any] = field(default_factory=lambda: {"name": "native", "crop": 358})
    normalize_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    augmentation: Mapping[str, Any] = field(default_factory=lambda: {"horizontal_flip_probability": 0.5})
    tta: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "model", _freeze(self.model))
        object.__setattr__(self, "view", _freeze(self.view))
        object.__setattr__(self, "augmentation", _freeze(self.augmentation))
        object.__setattr__(self, "normalize_mean", tuple(float(v) for v in self.normalize_mean))
        object.__setattr__(self, "normalize_std", tuple(float(v) for v in self.normalize_std))
        if self.seed < 0 or self.epochs < 1 or self.batch_size < 1 or self.accumulation < 1:
            raise ValueError("seed must be nonnegative; epochs, batch_size and accumulation must be positive.")
        if min(self.backbone_lr, self.head_lr) <= 0 or self.weight_decay < 0:
            raise ValueError("Learning rates must be positive and weight decay cannot be negative.")
        if not 0 <= self.label_smoothing < 1:
            raise ValueError("label_smoothing must be in [0, 1).")
        if len(self.normalize_mean) != 3 or len(self.normalize_std) != 3 or min(self.normalize_std) <= 0:
            raise ValueError("Image normalization needs three means and three positive standard deviations.")
        if self.optimizer != "adamw" or self.scheduler != "cosine":
            raise ValueError("This lesson currently supports AdamW and cosine scheduling.")
        if self.checkpoint_policy != "terminal":
            raise ValueError("Only the terminal checkpoint policy is implemented.")

    @property
    def effective_batch(self) -> int:
        """Return the number of examples contributing to one optimizer step."""
        return self.batch_size * self.accumulation

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {field.name: _thaw(getattr(self, field.name)) for field in fields(self)}


def load_config(path: str | Path) -> TrainConfig:
    """Load and validate a branch preset JSON file."""
    raw = read_json(path)
    raw.pop("schema_version", None)
    allowed = {field.name for field in fields(TrainConfig)}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown TrainConfig fields in {path}: {sorted(unknown)}")
    for name in ("normalize_mean", "normalize_std"):
        if name in raw:
            raw[name] = tuple(raw[name])
    return TrainConfig(**raw)


def save_config(config: TrainConfig, path: str | Path) -> Path:
    """Save a resolved training preset as schema-versioned JSON."""
    return write_json(path, {"schema_version": 1, **config.to_dict()})


@dataclass(frozen=True, slots=True)
class Workspace:
    """Stable paths for data, reference predictions, runs, and outputs."""

    root: Path
    run_id: str
    data_root: Path
    reference_root: Path
    artifact_root: Path
    output_root: Path

    @classmethod
    def from_root(cls, root: str | Path, run_id: str = "student_v1") -> "Workspace":
        """Create a workspace, honoring explicit task-specific path overrides."""
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", run_id):
            raise ValueError("run_id may contain only letters, digits, underscores, and hyphens.")
        task_root = Path(root).expanduser().resolve()
        if not task_root.is_dir():
            raise FileNotFoundError(f"Task root does not exist: {task_root}")
        data = Path(os.environ.get("AILAAI_DATA_ROOT", task_root / "data")).expanduser().resolve()
        reference = Path(os.environ.get("AILAAI_REFERENCE_ROOT", task_root / "reference_artifacts")).expanduser().resolve()
        artifacts = Path(os.environ.get("AILAAI_ARTIFACT_ROOT", task_root / "artifacts")).expanduser().resolve()
        outputs = Path(os.environ.get("AILAAI_OUTPUT_ROOT", task_root / "outputs")).expanduser().resolve()
        for directory in (data, artifacts / run_id, outputs / run_id):
            directory.mkdir(parents=True, exist_ok=True)
        return cls(task_root, run_id, data, reference, artifacts / run_id, outputs / run_id)

    def summary(self) -> dict[str, str]:
        """Return workspace locations for notebook display."""
        return {name: str(getattr(self, name)) for name in ("root", "data_root", "reference_root", "artifact_root", "output_root")}


def config_with(config: TrainConfig, **changes: Any) -> TrainConfig:
    """Return a validated immutable preset with explicit notebook overrides."""
    return replace(config, **changes)
