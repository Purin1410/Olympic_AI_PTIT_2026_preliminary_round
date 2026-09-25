"""Utilities for extracting and safely structuring official competition datasets.

Follows official ZIP specifications:
- Preserves raw image bytes without re-encoding, resizing or compressing
- Validates against Zip Slip path traversal and malicious filenames
- Rejects symbolic links and special files; skips bundled notebooks
- Performs full preflight validation before writing any files to disk
- Resolves official nested directories, including extra data/ wrappers and private_test/private_test
- Separates evaluation ground truth (pairs_results.csv) from inference inputs
- Validates manifests meaningfully while gracefully accepting absent test sets
"""
from pathlib import Path
import shutil
import stat
import zipfile
from typing import Dict, Optional, Any

from .core import read_csv


def is_safe_zip_path(target_dir: Path, member_name: str) -> bool:
    """Validate that zip member does not attempt directory traversal outside target_dir."""
    p = Path(member_name)
    if p.is_absolute() or ".." in p.parts:
        return False
    try:
        resolved = (target_dir / member_name).resolve()
        return resolved.is_relative_to(target_dir.resolve())
    except (ValueError, RuntimeError):
        return False


def is_ignored_zip_entry(member_name: str) -> bool:
    """Ignore OS metadata and junk entries."""
    parts = Path(member_name).parts
    for part in parts:
        if part.startswith("__MACOSX") or part in {".DS_Store", "Thumbs.db"}:
            return True
    return False


def is_special_or_forbidden_zip_member(member: zipfile.ZipInfo) -> tuple[bool, str]:
    """Check if zip member is a symlink, special file (FIFO, device, socket), or forbidden file (.ipynb)."""
    name = member.filename

    mode = member.external_attr >> 16
    if mode != 0:
        # Check for symbolic link
        if stat.S_ISLNK(mode):
            return True, f"Forbidden symbolic link in dataset archive: {name}"
        # Check for FIFO, character device, block device, or socket
        if stat.S_ISFIFO(mode) or stat.S_ISCHR(mode) or stat.S_ISBLK(mode) or stat.S_ISSOCK(mode):
            return True, f"Forbidden special file in dataset archive: {name}"

    return False, ""


def extract_official_zip(
    zip_path: Path | str,
    dest_dir: Path | str,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Safely extract official dataset ZIP preserving raw image bytes.

    Performs complete safe preflight check before writing any file to disk.
    Refuses to silently overwrite existing files unless overwrite=True.
    Rejects path traversal, symlinks and special files. Bundled notebooks are skipped.
    """
    zip_path = Path(zip_path).expanduser().resolve()
    dest_dir = Path(dest_dir).expanduser().resolve()

    if not zip_path.is_file():
        raise FileNotFoundError(f"ZIP archive not found: {zip_path}")

    # --- Preflight check before creating directories or writing any files ---
    with zipfile.ZipFile(zip_path, "r") as archive:
        members_to_extract = []
        skipped_files = []
        seen_paths = set()
        for member in archive.infolist():
            name = member.filename
            if is_ignored_zip_entry(name):
                continue

            if not is_safe_zip_path(dest_dir, name):
                raise ValueError(f"Dangerous zip entry detected (path traversal attempt): {name}")

            is_forbidden, reason = is_special_or_forbidden_zip_member(member)
            if is_forbidden:
                raise ValueError(f"Dangerous zip entry detected ({reason})")

            # Classroom ZIP contains a baseline notebook, which is not dataset input.
            # Do not execute or extract it; retain only the image/manifest package.
            if name.lower().endswith('.ipynb'):
                skipped_files.append(name)
                continue
            out_path = dest_dir / name
            key = str(out_path.resolve())
            if key in seen_paths:
                raise ValueError(f"Duplicate zip destination: {name}")
            seen_paths.add(key)
            if not member.is_dir() and out_path.exists() and not overwrite:
                raise FileExistsError(
                    f"Destination file '{out_path}' already exists. Pass overwrite=True to replace."
                )

            members_to_extract.append((member, out_path))

    # --- Extraction execution after successful preflight ---
    dest_dir.mkdir(parents=True, exist_ok=True)
    extracted_files = []

    with zipfile.ZipFile(zip_path, "r") as archive:
        for member, out_path in members_to_extract:
            if member.is_dir():
                out_path.mkdir(parents=True, exist_ok=True)
                continue

            out_path.parent.mkdir(parents=True, exist_ok=True)
            # Stream raw bytes directly to ensure no re-encoding or corruption
            with archive.open(member, "r") as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted_files.append(out_path)

    roots = discover_dataset_roots(dest_dir)
    return {
        "status": "extracted",
        "dest_dir": str(dest_dir),
        "total_files": len(extracted_files),
        "skipped_files": skipped_files,
        "roots": {k: str(v) if v else None for k, v in roots.items()},
    }


def discover_dataset_roots(base_dir: Path | str) -> Dict[str, Optional[Path]]:
    """Discover train, public_test, and nested private_test directories with pairs.csv."""
    base = Path(base_dir).expanduser().resolve()

    candidates_train = [
        base / "train",
        base / "data" / "train",
        base / "data" / "data" / "train",
        base,
    ]
    candidates_public = [
        base / "public_test",
        base / "data" / "public_test",
        base / "data" / "data" / "public_test",
        base / "test",
    ]
    candidates_private = [
        # Official archives may add one or two data/ wrappers before private_test.
        base / "private_test" / "private_test",
        base / "data" / "private_test" / "private_test",
        base / "data" / "data" / "private_test" / "private_test",
        base / "private_test",
        base / "data" / "private_test",
        base / "data" / "data" / "private_test",
    ]

    train_root = next((p for p in candidates_train if (p / "pairs.csv").is_file()), None)
    public_root = next((p for p in candidates_public if (p / "pairs.csv").is_file()), None)
    private_root = next((p for p in candidates_private if (p / "pairs.csv").is_file()), None)

    return {
        "train": train_root,
        "public_test": public_root,
        "private_test": private_root,
    }


def validate_official_layout(roots: Dict[str, Optional[Path]]) -> Dict[str, Any]:
    """Validate sample counts and schemas of official manifests.

    Fails meaningfully on malformed manifests; accepts optional absence of test sets.
    Never reads or expects evaluation ground truth (pairs_results.csv).
    """
    report = {}
    train_root = roots.get("train")
    if train_root and (train_root / "pairs.csv").is_file():
        df_tr = read_csv(train_root / "pairs.csv")
        req_cols = ["pair_id", "image_0", "image_1", "fake_position"]
        missing = [c for c in req_cols if c not in df_tr.columns]
        if missing:
            raise ValueError(f"Malformed train pairs manifest at {train_root / 'pairs.csv'}: missing columns {missing}")

        if not df_tr["pair_id"].is_unique:
            raise ValueError(f"Malformed train pairs manifest: duplicate pair_id values found in {train_root / 'pairs.csv'}")

        valid_positions = {0, 1, "0", "1"}
        if not set(df_tr["fake_position"]).issubset(valid_positions):
            raise ValueError(f"Malformed train pairs manifest: invalid fake_position labels in {train_root / 'pairs.csv'}")

        report["train_pairs"] = len(df_tr)
        report["train_valid"] = len(df_tr) == 1000
    else:
        report["train_pairs"] = 0
        report["train_valid"] = False

    for test_key in ("public_test", "private_test"):
        t_root = roots.get(test_key)
        if t_root and (t_root / "pairs.csv").is_file():
            df_te = read_csv(t_root / "pairs.csv")
            req_cols = ["pair_id", "image_0", "image_1"]
            missing = [c for c in req_cols if c not in df_te.columns]
            if missing:
                raise ValueError(f"Malformed {test_key} pairs manifest at {t_root / 'pairs.csv'}: missing columns {missing}")

            if not df_te["pair_id"].is_unique:
                raise ValueError(f"Malformed {test_key} pairs manifest: duplicate pair_id values in {t_root / 'pairs.csv'}")

            report[f"{test_key}_present"] = True
            report[f"{test_key}_pairs"] = len(df_te)
            report[f"{test_key}_valid"] = len(df_te) == 100
        else:
            # Optional test absence is completely okay
            report[f"{test_key}_present"] = False
            report[f"{test_key}_pairs"] = 0
            report[f"{test_key}_valid"] = False

    return report
