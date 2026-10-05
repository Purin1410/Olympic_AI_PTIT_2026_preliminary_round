"""Submission CSV/ZIP export and read-back checks."""

from __future__ import annotations

import hashlib
import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import write_json
from .resources import sha256_file


@dataclass(frozen=True, slots=True)
class SubmissionReceipt:
    """Validation details for a written submission archive."""

    path: Path
    sha256: str
    row_count: int
    member: str
    columns: tuple[str, ...]

    def summary(self) -> dict[str, Any]:
        return {"path": str(self.path), "sha256": self.sha256, "row_count": self.row_count,
                "member": self.member, "columns": list(self.columns), "valid": True}


def _validate_frame(frame: pd.DataFrame, expected_names: Iterable[str], expected_count: int) -> pd.DataFrame:
    names = list(expected_names)
    if len(names) != expected_count or len(set(names)) != expected_count:
        raise ValueError(f"Expected test names must contain exactly {expected_count} unique entries.")
    if list(frame.columns) != ["file_name", "category_id"]:
        raise ValueError("Submission columns and order must be exactly file_name,category_id.")
    if len(frame) != expected_count:
        raise ValueError(f"Submission must have exactly {expected_count} data rows; found {len(frame)}.")
    result = frame.copy()
    if result.isna().any().any():
        raise ValueError("Submission contains missing values.")
    result["file_name"] = result.file_name.astype(str)
    if not result.file_name.is_unique:
        raise ValueError("Submission contains duplicate file_name values.")
    if set(result.file_name) != set(names):
        missing = sorted(set(names) - set(result.file_name))[:5]
        extra = sorted(set(result.file_name) - set(names))[:5]
        raise ValueError(f"Submission file names differ from test; missing={missing}, extra={extra}.")
    labels = pd.to_numeric(result.category_id, errors="coerce")
    if labels.isna().any() or not np.equal(labels, np.floor(labels)).all() or not labels.isin([0, 1]).all():
        raise ValueError("category_id values must be integer labels 0 or 1.")
    result["category_id"] = labels.astype("int64")
    return result


def validate_submission(
    zip_path: str | Path,
    expected_names: Iterable[str],
    expected_count: int = 200,
    expected_frame: pd.DataFrame | None = None,
    receipt_path: str | Path | None = None,
) -> SubmissionReceipt:
    """Read a ZIP back and check member, CRC, schema, IDs, labels, and optional values."""
    source = Path(zip_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        with zipfile.ZipFile(source) as archive:
            if archive.testzip() is not None:
                raise ValueError("Submission ZIP CRC check failed.")
            members = archive.namelist()
            if members != ["submission.csv"]:
                raise ValueError("ZIP must contain exactly one root member named submission.csv.")
            with archive.open("submission.csv") as handle:
                returned = pd.read_csv(handle)
    except zipfile.BadZipFile as exc:
        raise ValueError("Submission is not a readable ZIP archive.") from exc
    returned = _validate_frame(returned, expected_names, expected_count)
    if expected_frame is not None:
        expected = _validate_frame(expected_frame, expected_names, expected_count)
        left = returned.sort_values("file_name").reset_index(drop=True)
        right = expected.sort_values("file_name").reset_index(drop=True)
        if not left.equals(right):
            raise ValueError("Read-back submission values do not match the frame that was exported.")
    receipt = SubmissionReceipt(source.resolve(), sha256_file(source), len(returned), "submission.csv", tuple(returned.columns))
    if receipt_path:
        write_json(receipt_path, receipt.summary())
    return receipt


def export_submission(
    frame: pd.DataFrame,
    zip_path: str | Path,
    expected_names: Iterable[str],
    expected_count: int = 200,
) -> Path:
    """Validate and atomically create a ZIP with one root-level submission.csv."""
    expected_list = list(expected_names)
    normalized = _validate_frame(frame, expected_list, expected_count)
    destination = Path(zip_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent)
    os.close(fd)
    temporary = Path(temp_name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("submission.csv", normalized.to_csv(index=False, lineterminator="\n"))
        validate_submission(temporary, expected_list, expected_count, expected_frame=normalized)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination
