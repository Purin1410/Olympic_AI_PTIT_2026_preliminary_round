"""Small helpers for recording analysis notes and the final inference decision."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .config import Workspace, write_json
from .resources import sha256_file


@dataclass(frozen=True, slots=True)
class Decision:
    """Recorded branch checkpoints, probability weights, and threshold."""

    decision_id: str
    path: Path
    threshold: float
    weights: Mapping[str, float]

    def summary(self) -> dict[str, Any]:
        return {"decision_id": self.decision_id, "path": str(self.path), "threshold": self.threshold,
                "weights": dict(self.weights)}


def save_decision(
    workspace: Workspace,
    runs: Mapping[str, Any],
    method: str,
    weights: Mapping[str, float],
    threshold: float,
    label_map: Mapping[int | str, str],
    test_rows: pd.DataFrame,
    validation_scores: pd.DataFrame,
) -> Decision:
    """Save a reviewable inference recipe after checking both complete branch runs."""
    if set(runs) != {"rgb", "highpass"} or set(weights) != {"rgb", "highpass"}:
        raise ValueError("A decision needs complete RGB and High-pass runs and weights.")
    values = {key: float(value) for key, value in weights.items()}
    if any(value < 0 for value in values.values()) or abs(sum(values.values()) - 1.0) > 1e-12:
        raise ValueError("Ensemble weights must be nonnegative and sum to one.")
    if not 0 <= threshold <= 1 or method != "probability_mean":
        raise ValueError("Supported decision: probability_mean with a threshold in [0, 1].")
    if test_rows.empty or not test_rows.file_name.is_unique:
        raise ValueError("Test rows must be nonempty and have unique file names.")
    branches: dict[str, Any] = {}
    for name, run in runs.items():
        if run.branch != name or run.run_id != workspace.run_id:
            raise ValueError(f"Run {name} does not belong to this workspace/branch.")
        status_path = run.run_dir / "run_status.json"
        if not status_path.is_file():
            raise ValueError(f"Run {name} has no completion receipt.")
        status = json.loads(status_path.read_text())
        checkpoint_hash = sha256_file(run.checkpoint_path)
        if status.get("status") != "complete" or status.get("checkpoint_sha256") != checkpoint_hash:
            raise ValueError(f"Run {name} is incomplete or its checkpoint changed.")
        branches[name] = {
            "run_id": run.run_id,
            "checkpoint": os.path.relpath(run.checkpoint_path, workspace.root),
            "checkpoint_sha256": checkpoint_hash,
            "config_sha256": run.metadata.get("config_sha256"),
            "view_spec": dict(run.metadata.get("view_spec", {})),
            "model_spec": dict(run.metadata.get("model_spec", {})),
            "validation_rows": len(run.val_predictions.rows),
        }
    payload = {
        "schema_version": 1,
        "run_id": workspace.run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "weights": values,
        "threshold": float(threshold),
        "label_map": {str(key): value for key, value in label_map.items()},
        "test_file_names_sha256": hashlib.sha256("\n".join(sorted(test_rows.file_name.astype(str))).encode()).hexdigest(),
        "test_count": len(test_rows),
        "branches": branches,
        "validation_scores": validation_scores.to_dict(orient="records"),
        "validation_score_scope": "validation only; not test performance",
    }
    digest = hashlib.sha256(repr(sorted(payload.items())).encode()).hexdigest()[:16]
    payload["decision_id"] = digest
    path = workspace.output_root / "decision.json"
    write_json(path, payload)
    return Decision(digest, path, float(threshold), values)


def save_analysis_note(
    workspace: Workspace,
    rgb: Any,
    highpass: Any,
    threshold: float,
    method: str,
    note: str,
) -> Path:
    """Save a validation-only NB3 note without turning it into a test decision."""
    if not 0 <= threshold <= 1 or not note.strip():
        raise ValueError("Provide a threshold in [0, 1] and a nonempty note.")
    path = workspace.output_root / "analysis_note.json"
    write_json(path, {
        "run_id": workspace.run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"rgb": rgb.summary(), "highpass": highpass.summary()},
        "threshold_under_discussion": float(threshold),
        "method": method,
        "note": note,
        "scope": "validation analysis only",
    })
    return path
