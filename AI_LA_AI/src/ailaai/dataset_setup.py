"""Locate and prepare the published dataset using only the standard library."""
from __future__ import annotations

import csv
import shutil
import tempfile
import zipfile
from pathlib import Path

DATASET_ID = "1g_43_Xn-DWYB-k7Yq4XQTr5ZQXZ0UdXq"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def image_names(directory: Path) -> set[str]:
    if not directory.is_dir():
        return set()
    return {p.name for p in directory.iterdir() if p.is_file() and p.stat().st_size > 0 and p.suffix.lower() in IMAGE_SUFFIXES}


def expected_names(workspace, split: str) -> set[str]:
    if split == "train":
        with (workspace.root / "assets/splits/train_folds.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        names = {row["file_name"] for row in rows}
        if len(rows) != 2000 or len(names) != 2000:
            raise ValueError("Bảng chia fold phải có đúng 2.000 tên ảnh train khác nhau.")
        return names
    if split == "test":
        return {f"{i:05d}.jpg" for i in range(1, 201)}
    raise ValueError(f"Unknown dataset split: {split}")


def find_images(root: Path, split: str, expected: set[str]) -> Path | None:
    """Recognize published and legacy layouts; never substitute public test."""
    bases = [root, root / "data", root / "who_is_AI", root / "who_is_AI/data"]
    relatives = ["train/images", "images"] if split == "train" else [
        "test/images", "private_test/private_test/images", "private_test/images"]
    for base in bases:
        for relative in relatives:
            candidate = base / relative
            if "public_test" in candidate.resolve().parts:
                continue
            if image_names(candidate) == expected:
                return candidate
    return None


def prepare_dataset(workspace, *, required=("train", "test"), file_id=DATASET_ID, force=False):
    # Lazy import avoids a cycle and keeps locating data independent of ML libraries.
    from .resources import _safe_extract

    root = workspace.data_root
    root.mkdir(parents=True, exist_ok=True)
    targets = {split: root / split / "images" for split in required}
    if "test" in targets and "public_test" in targets["test"].resolve().parts:
        raise ValueError("Thư mục test đang trỏ vào Public Test; cần dùng ảnh Private Test của bài.")
    expected = {split: expected_names(workspace, split) for split in required}
    pending = [split for split in required if force or image_names(targets[split]) != expected[split]]
    if not pending:
        return root
    sources = {split: find_images(root, split, expected[split]) for split in pending}
    staging = None
    try:
        if force or any(source is None for source in sources.values()):
            archive = root / "who_is_AI.zip"
            if force or not zipfile.is_zipfile(archive):
                import gdown
                partial = archive.with_suffix(".zip.partial")
                print("Đang tải bộ dữ liệu từ Google Drive...", flush=True)
                try:
                    result = gdown.download(id=file_id, output=str(partial), quiet=False)
                    if not result or not zipfile.is_zipfile(partial):
                        raise ValueError("Tệp tải về chưa phải ZIP hợp lệ.")
                    with zipfile.ZipFile(partial) as z:
                        if z.testzip() is not None:
                            raise ValueError("Tệp ZIP tải về bị thiếu hoặc hỏng.")
                    partial.replace(archive)
                except Exception as exc:
                    partial.unlink(missing_ok=True)
                    raise RuntimeError(
                        "Chưa tải được dataset từ Google Drive. Kiểm tra mạng hoặc giới hạn tải "
                        "của Drive rồi chạy lại cell. "
                        f"Nguồn: https://drive.google.com/file/d/{file_id}/view"
                    ) from exc
            staging = Path(tempfile.mkdtemp(prefix=".dataset-", dir=root))
            unpacked = staging / "unpacked"
            print("Đang giải nén và kiểm tra dữ liệu...", flush=True)
            try:
                _safe_extract(archive, unpacked)
            except zipfile.BadZipFile as exc:
                if not force:
                    print("ZIP đang lưu bị hỏng; đang tải lại một lần.", flush=True)
                    return prepare_dataset(workspace, required=required, file_id=file_id, force=True)
                raise RuntimeError("ZIP tải lại vẫn bị hỏng. Kiểm tra nguồn dữ liệu rồi chạy lại cell.") from exc
            for split in pending:
                if force or sources[split] is None:
                    sources[split] = find_images(unpacked, split, expected[split])
        for split in pending:
            source = sources[split]
            if source is None:
                raise FileNotFoundError(f"Không tìm thấy đủ ảnh {split} trong dataset đã tải.")
            destination = targets[split]
            if source.resolve() != destination.resolve():
                extra = image_names(destination) - expected[split]
                if extra:
                    raise ValueError(f"Thư mục {destination} có ảnh ngoài danh sách: {sorted(extra)[:3]}")
                destination.mkdir(parents=True, exist_ok=True)
                for name in sorted(expected[split]):
                    target = destination / name
                    if force or not target.is_file() or target.stat().st_size == 0:
                        shutil.copy2(source / name, target)
                if split == "train":
                    manifest = source.parent / "manifest.csv"
                    target_manifest = destination.parent / "manifest.csv"
                    if manifest.is_file() and not target_manifest.exists():
                        shutil.copy2(manifest, target_manifest)
            if image_names(destination) != expected[split]:
                raise ValueError(f"Danh sách ảnh {split} chưa khớp dữ liệu của bài.")
        print("Dữ liệu đã sẵn sàng: " + ", ".join(f"{split} {len(expected[split])} ảnh" for split in required))
        return root
    finally:
        if staging is not None:
            shutil.rmtree(staging)
