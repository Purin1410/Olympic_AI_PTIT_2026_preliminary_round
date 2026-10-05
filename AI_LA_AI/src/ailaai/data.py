"""Image manifests, dataset decoding, and deterministic train/validation splits."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset

from .config import Workspace, read_json
from .resources import sha256_file

ImageFile.LOAD_TRUNCATED_IMAGES = False


def decode_rgb(path: str | Path) -> torch.Tensor:
    """Decode an image as contiguous RGB float CHW values in [0, 1]."""
    with Image.open(path) as image:
        image.load()
        array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)


def _read_manifest(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if "file_name" not in frame:
        if "path" in frame:
            frame["file_name"] = frame["path"].astype(str).apply(lambda p: Path(p).name)
        elif "filename" in frame:
            frame["file_name"] = frame["filename"].astype(str).apply(lambda p: Path(p).name)
        elif "image" in frame:
            frame["file_name"] = frame["image"].astype(str).apply(lambda p: Path(p).name)
        else:
            raise ValueError(f"Manifest must contain file_name: {path}")
    if frame.file_name.isna().any() or not frame.file_name.is_unique:
        raise ValueError(f"Manifest file_name values must be present and unique: {path}")
    frame["file_name"] = frame.file_name.astype(str)
    if "label" not in frame:
        if "category_id" in frame:
            frame["label"] = frame["category_id"]
        elif "target" in frame:
            frame["label"] = frame["target"]
    return frame


def _resolve_paths(frame: pd.DataFrame, image_root: Path) -> pd.DataFrame:
    result = frame.copy()
    paths: list[str] = []
    for row in result.itertuples(index=False):
        file_name = str(getattr(row, "file_name"))
        relative = str(getattr(row, "path", file_name))
        candidates = [
            image_root / relative,
            image_root.parent / relative,
            image_root / file_name,
            image_root / Path(relative).name,
        ]
        selected = next((candidate.resolve() for candidate in candidates if candidate.is_file()), candidates[0].resolve())
        paths.append(str(selected))
    result["path"] = paths
    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing image file: {missing[0]}")
    return result


def _find_test_image_dir(workspace: Workspace) -> Path | None:
    candidates = [
        workspace.data_root / "test" / "images",
        workspace.data_root / "test",
        workspace.data_root / "private_test" / "private_test" / "images",
        workspace.data_root / "private_test" / "images",
        workspace.data_root / "public_test" / "images",
        workspace.data_root / "who_is_AI" / "data" / "private_test" / "private_test" / "images",
        workspace.data_root / "who_is_AI" / "data" / "private_test" / "images",
        workspace.data_root / "who_is_AI" / "private_test" / "private_test" / "images",
        workspace.data_root / "who_is_AI" / "private_test" / "images",
        workspace.data_root.parent / "who_is_AI" / "data" / "private_test" / "private_test" / "images",
        workspace.data_root.parent / "who_is_AI" / "data" / "private_test" / "images",
    ]
    for cand in candidates:
        if cand.is_dir() and any(p.suffix.lower() in {".jpg", ".jpeg", ".png"} for p in cand.iterdir() if p.is_file()):
            return cand.resolve()
    return None


def load_train_manifest(
    workspace: Workspace,
    manifest_path: str | Path | None = None,
    split_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load a labeled CSV, or use the distributed five-fold split as its manifest."""
    train_root = workspace.data_root / "train"
    image_root = train_root / "images"
    candidates = [Path(manifest_path)] if manifest_path else [
        train_root / "manifest.csv",
        train_root / "train.csv",
        workspace.data_root / "manifest.csv",
        workspace.data_root / "train.csv",
        workspace.data_root / "who_is_AI" / "data" / "train" / "manifest.csv",
        workspace.data_root / "who_is_AI" / "train" / "manifest.csv",
        workspace.data_root.parent / "who_is_AI" / "data" / "train" / "manifest.csv",
    ]
    if manifest_path is not None:
        explicit = Path(manifest_path)
        if not explicit.is_absolute():
            explicit = workspace.root / explicit
        if not explicit.is_file():
            raise FileNotFoundError(explicit)
    source = next((p if p.is_absolute() else workspace.root / p for p in candidates if (p if p.is_absolute() else workspace.root / p).is_file()), None)
    if source is not None:
        frame = _read_manifest(source)
    else:
        split = Path(split_path) if split_path else workspace.root / "assets/splits/train_folds.csv"
        if not split.is_absolute():
            split = workspace.root / split
        frame = _read_manifest(split)
    if "label" not in frame:
        raise ValueError("Train manifest must contain label values 0 (Real) or 1 (Fake).")
    labels = pd.to_numeric(frame.label, errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("Train labels must be integers 0 or 1 with no missing values.")
    frame["label"] = labels.astype("int64")
    return _resolve_paths(frame, image_root)


def load_test_manifest(workspace: Workspace, manifest_path: str | Path | None = None) -> pd.DataFrame:
    """Load test file names, or list a prepared test image directory."""
    test_root = workspace.data_root / "test"
    image_root = _find_test_image_dir(workspace) or (test_root / "images")
    candidates = [Path(manifest_path)] if manifest_path else [
        test_root / "manifest.csv",
        test_root / "test.csv",
        workspace.data_root / "private_test" / "private_test" / "pair_results.csv",
        workspace.data_root / "private_test" / "pair_results.csv",
    ]
    source = next((p if p.is_absolute() else workspace.root / p for p in candidates if (p if p.is_absolute() else workspace.root / p).is_file()), None)
    if source is not None:
        frame = _read_manifest(source)
    else:
        if not image_root.is_dir():
            raise FileNotFoundError(f"Test manifest or image directory is missing: {source} / {image_root}")
        names = sorted(path.name for path in image_root.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"})
        frame = pd.DataFrame({"file_name": names})
    frame = frame.drop(columns=[c for c in ("label", "category_id", "target") if c in frame.columns], errors="ignore")
    return _resolve_paths(frame, image_root)


def load_fold_split(
    train: pd.DataFrame,
    split_path: str | Path,
    fold: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply a fixed fold while checking names and labels against the train manifest."""
    split = _read_manifest(Path(split_path))
    required = {"file_name", "label", "fold"}
    if not required.issubset(split.columns):
        raise ValueError(f"Fold split must include {sorted(required)}.")
    split["label"] = pd.to_numeric(split.label, errors="raise").astype("int64")
    split["fold"] = pd.to_numeric(split.fold, errors="raise").astype("int64")
    if not split.label.isin([0, 1]).all() or not split.file_name.is_unique:
        raise ValueError("Split labels or file names are invalid.")
    expected = train.set_index("file_name").label.to_dict()
    got = split.set_index("file_name").label.to_dict()
    if expected.keys() != got.keys() or any(expected[name] != got[name] for name in expected):
        raise ValueError("Split file names/labels do not match the loaded train manifest.")
    if fold not in set(split.fold):
        raise ValueError(f"Fold {fold} is not present in {split_path}.")
    joined = split[["file_name", "label", "fold"]].merge(
        train.drop(columns=["label", "fold"], errors="ignore"), on="file_name", validate="one_to_one")
    validation = joined[joined.fold == fold].reset_index(drop=True)
    fitting = joined[joined.fold != fold].reset_index(drop=True)
    if fitting.empty or validation.empty or set(fitting.file_name) & set(validation.file_name):
        raise ValueError("The selected fold has an empty or overlapping split.")
    return fitting, validation


def validate_data(train: pd.DataFrame, test: pd.DataFrame, resource_manifest: str | Path | None = None) -> dict[str, Any]:
    """Check required columns, labels, image readability, and train/test identity."""
    if train.empty or test.empty:
        raise ValueError("Train and test manifests must both contain images.")
    if not {"file_name", "path", "label"}.issubset(train.columns) or not {"file_name", "path"}.issubset(test.columns):
        raise ValueError("Train/test manifests have missing required columns.")
    if train.label.isna().any() or not train.label.isin([0, 1]).all():
        raise ValueError("Train labels must be nonmissing binary integers.")
    if resource_manifest is not None:
        manifest_path = Path(resource_manifest)
        if not manifest_path.is_absolute():
            manifest_path = Path.cwd() / manifest_path
        assets = read_json(manifest_path).get("assets", {})
        for frame, asset_name, label in ((train, "train_images", "train"), (test, "test_images", "test")):
            expected = assets.get(asset_name, {}).get("expected_count")
            if expected is not None and len(frame) != int(expected):
                raise ValueError(f"{label} manifest has {len(frame)} rows; resource config expects {expected}.")
    train_hashes: dict[str, str] = {}
    for frame in (train, test):
        for row in frame.itertuples(index=False):
            with Image.open(row.path) as image:
                image.verify()
                if min(image.size) < 358:
                    raise ValueError(f"Image is smaller than the required 358-pixel crop: {row.file_name}")
            image_hash = sha256_file(row.path)
            if frame is train:
                train_hashes[image_hash] = str(row.file_name)
            elif image_hash in train_hashes:
                raise ValueError(f"Train/test contain identical image bytes: {train_hashes[image_hash]} and {row.file_name}.")
    return {"train_count": len(train), "test_count": len(test), "label_counts": train.label.value_counts().sort_index().to_dict()}


class FaceDataset(Dataset[tuple[torch.Tensor, int, str]]):
    """Decode images without resizing; the engine applies the injected view."""

    def __init__(self, rows: pd.DataFrame, labeled: bool = True) -> None:
        self.rows = rows.reset_index(drop=True).copy()
        self.labeled = labeled

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        row = self.rows.iloc[index]
        label = int(row.label) if self.labeled else -1
        return decode_rgb(row.path), label, str(row.file_name)


def make_loader(
    rows: pd.DataFrame,
    batch_size: int,
    labeled: bool = True,
    shuffle: bool = False,
    seed: int = 2026,
    num_workers: int = 0,
) -> DataLoader:
    """Construct a worker-safe loader for notebook-defined transform functions."""
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(FaceDataset(rows, labeled=labeled), batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available(), generator=generator)
