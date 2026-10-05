"""Prediction tables, provenance metadata, and strict ID alignment."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .config import Workspace, read_json, write_json
from .resources import sha256_file


@dataclass(frozen=True, slots=True)
class PredictionTable:
    """A probability table paired with its source and split metadata."""

    rows: pd.DataFrame
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        frame = self.rows.copy()
        if "positive_probability" in frame and "prob" not in frame:
            frame = frame.rename(columns={"positive_probability": "prob"})
        if not {"file_name", "prob"}.issubset(frame.columns):
            raise ValueError("Prediction rows must include file_name and prob.")
        if frame.file_name.isna().any():
            raise ValueError("Prediction names cannot be missing.")
        frame["file_name"] = frame.file_name.astype(str)
        frame["prob"] = pd.to_numeric(frame.prob, errors="coerce")
        if not frame.file_name.is_unique:
            raise ValueError("Prediction names must be nonmissing and unique.")
        if frame.prob.isna().any() or not np.isfinite(frame.prob).all() or not frame.prob.between(0, 1).all():
            raise ValueError("Prediction probabilities must be finite values in [0, 1].")
        if "label" in frame:
            labels = pd.to_numeric(frame.label, errors="coerce")
            if labels.isna().any() or not labels.isin([0, 1]).all():
                raise ValueError("Prediction labels, when present, must be 0 or 1.")
            frame["label"] = labels.astype("int64")
        object.__setattr__(self, "rows", frame.reset_index(drop=True))
        object.__setattr__(self, "meta", dict(self.meta))

    def save(self, path: str | Path) -> tuple[Path, Path]:
        """Write CSV and a metadata sidecar atomically."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".partial")
        self.rows.to_csv(temporary, index=False)
        temporary.replace(destination)
        sidecar = destination.with_suffix(destination.suffix + ".meta.json")
        write_json(sidecar, {**self.meta, "prediction_sha256": sha256_file(destination), "rows": len(self.rows)})
        return destination, sidecar

    @classmethod
    def load(cls, path: str | Path, meta_path: str | Path | None = None) -> "PredictionTable":
        """Load a CSV and verify its optional metadata sidecar hash."""
        source = Path(path)
        sidecar = Path(meta_path) if meta_path else source.with_suffix(source.suffix + ".meta.json")
        metadata: dict[str, Any] = {}
        if sidecar.is_file():
            metadata = read_json(sidecar)
            expected = metadata.pop("prediction_sha256", None)
            rows_expected = metadata.pop("rows", None)
            if expected and sha256_file(source) != expected:
                raise ValueError(f"Prediction CSV hash mismatch: {source}")
            rows = pd.read_csv(source)
            if rows_expected is not None and len(rows) != int(rows_expected):
                raise ValueError(f"Prediction CSV row count differs from its receipt: {source}")
        else:
            rows = pd.read_csv(source)
        return cls(rows, metadata)

    def summary(self) -> dict[str, Any]:
        """Return a small provenance summary suitable for notebook display."""
        return {"source": self.meta.get("source", "unknown"), "branch": self.meta.get("branch"),
                "rows": len(self.rows), "split_sha256": self.meta.get("split_sha256"),
                "checkpoint_sha256": self.meta.get("checkpoint_sha256")}


def align_predictions(
    rgb: PredictionTable | pd.DataFrame,
    highpass: PredictionTable | pd.DataFrame,
    expected_rows: pd.DataFrame | list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Join by unique file IDs and require exact coverage, labels, folds, and split."""
    rgb_table = rgb if isinstance(rgb, PredictionTable) else PredictionTable(rgb)
    hp_table = highpass if isinstance(highpass, PredictionTable) else PredictionTable(highpass)
    rgb_meta, hp_meta = rgb_table.meta, hp_table.meta
    rgb_split, hp_split = rgb_meta.get("split_sha256"), hp_meta.get("split_sha256")
    if rgb_split and hp_split and rgb_split != hp_split:
        raise ValueError("RGB and High-pass predictions come from different train splits.")
    left = rgb_table.rows.copy()
    right = hp_table.rows.copy()
    shared = sorted(set(left.columns) & set(right.columns) - {"file_name", "prob"})
    left = left.rename(columns={"prob": "p_rgb"})
    right = right.rename(columns={"prob": "p_hp"})
    merged = left.merge(right[["file_name", "p_hp", *shared]], on="file_name", how="outer",
                        validate="one_to_one", indicator=True, suffixes=("", "_hp"))
    if not (merged._merge == "both").all():
        missing = merged.loc[merged._merge != "both", "file_name"].head(5).tolist()
        raise ValueError(f"Prediction file_name coverage differs between branches: {missing}")
    merged = merged.drop(columns="_merge")
    for column in shared:
        other = f"{column}_hp"
        if other in merged:
            if not merged[column].equals(merged[other]):
                raise ValueError(f"RGB and High-pass prediction metadata differ in {column}.")
            merged = merged.drop(columns=other)
    if expected_rows is not None:
        if isinstance(expected_rows, pd.DataFrame):
            expected = expected_rows.copy()
            if "file_name" not in expected:
                raise ValueError("expected_rows must contain file_name.")
        else:
            expected = pd.DataFrame({"file_name": list(expected_rows)})
        expected["file_name"] = expected.file_name.astype(str)
        if not expected.file_name.is_unique:
            raise ValueError("Expected file names are not unique.")
        if set(merged.file_name) != set(expected.file_name):
            missing = sorted(set(expected.file_name) - set(merged.file_name))[:5]
            extra = sorted(set(merged.file_name) - set(expected.file_name))[:5]
            raise ValueError(f"Prediction coverage differs from expected rows; missing={missing}, extra={extra}.")
        if "label" in expected:
            if "label" not in merged:
                raise ValueError("Expected validation rows have labels but predictions do not.")
            labels = expected.set_index("file_name").label
            actual = merged.set_index("file_name").label
            if not labels.sort_index().equals(actual.reindex(labels.index).sort_index()):
                raise ValueError("Prediction labels do not match the requested validation rows.")
        if "fold" in expected and "fold" in merged:
            folds = expected.set_index("file_name").fold
            actual_folds = merged.set_index("file_name").fold
            if not folds.sort_index().equals(actual_folds.reindex(folds.index).sort_index()):
                raise ValueError("Prediction folds do not match the requested validation rows.")
        if "path" in expected:
            merged = merged.merge(expected[["file_name", "path"]], on="file_name", how="left", validate="one_to_one")
    return merged.sort_values("file_name").reset_index(drop=True)


def load_validation_pair(workspace: Workspace, source: str = "reference") -> tuple[PredictionTable, PredictionTable, pd.DataFrame]:
    """Load either the packaged historical pair or both branches of one learner run."""
    if source == "reference":
        lesson = read_json(workspace.root / "configs/lesson.json")
        paths = lesson["reference_pair"]
        def reference_path(value: str) -> Path:
            path = Path(value)
            if path.is_absolute():
                return path
            if path.parts and path.parts[0] == "reference_artifacts":
                return workspace.reference_root.joinpath(*path.parts[1:])
            return workspace.root / path

        provenance = read_json(reference_path(paths["provenance"]))
        tables = []
        for branch in ("rgb", "highpass"):
            expected_asset = provenance.get("assets", {}).get(f"{branch}_oof.csv", {})
            prediction_path = reference_path(paths[branch])
            if expected_asset.get("sha256") and sha256_file(prediction_path) != expected_asset["sha256"]:
                raise ValueError(f"Reference prediction hash mismatch: {prediction_path}")
            table = PredictionTable.load(prediction_path)
            table = PredictionTable(table.rows, {**table.meta, **provenance, "source": "historical_reference", "branch": branch})
            tables.append(table)
        split_path = workspace.root / "assets/splits/train_folds.csv"
        expected = pd.read_csv(split_path)
        expected["path"] = expected.path.map(lambda path: str((workspace.data_root / "train" / str(path)).resolve()))
        if provenance.get("split_sha256"):
            if sha256_file(split_path) != provenance["split_sha256"]:
                raise ValueError("Reference predictions were created against a different train split.")
        return tables[0], tables[1], expected
    if source == "learner":
        branches = []
        expected: pd.DataFrame | None = None
        for branch in ("rgb", "highpass"):
            active = workspace.artifact_root / f"{branch}_active.json"
            directory = read_json(active)["directory"] if active.is_file() else branch
            prediction_path = workspace.artifact_root / directory / "val_predictions.csv"
            table = PredictionTable.load(prediction_path)
            if table.meta.get("run_id") != workspace.run_id:
                raise ValueError(f"{branch} validation predictions belong to another run.")
            branches.append(table)
            split_path = Path(table.meta["validation_manifest"])
            if not split_path.is_absolute():
                split_path = prediction_path.parent / split_path
            if split_path:
                expected = pd.read_csv(split_path)
        if expected is None:
            raise ValueError("Learner prediction sidecars do not identify validation rows.")
        return branches[0], branches[1], expected
    raise ValueError("source must be 'reference' or 'learner'.")
