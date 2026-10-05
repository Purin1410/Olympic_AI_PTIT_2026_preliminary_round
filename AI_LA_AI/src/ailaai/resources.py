"""Verified resource preparation and runtime checks."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .config import Workspace, read_json, write_json


def sha256_file(path: str | Path) -> str:
    """Compute a file's SHA-256 without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_child(root: Path, relative: str) -> Path:
    child = (root / relative).resolve()
    if child != root and root not in child.parents:
        raise ValueError(f"Resource destination escapes task root: {relative}")
    return child


def _count_files(path: Path, suffixes: list[str] | None = None) -> int:
    extensions = {suffix.lower() for suffix in (suffixes or [])}
    return sum(1 for item in path.rglob("*") if item.is_file() and (not extensions or item.suffix.lower() in extensions))


def _validate_existing(path: Path, spec: Mapping[str, Any], asset_name: str) -> dict[str, Any]:
    kind = spec.get("kind", "file")
    if kind == "directory":
        if not path.is_dir():
            raise FileNotFoundError(f"Required resource {asset_name} is not a directory: {path}")
        count = _count_files(path, spec.get("suffixes"))
        expected = spec.get("expected_count")
        if expected is not None and count != int(expected):
            raise ValueError(f"{asset_name}: expected {expected} files, found {count} in {path}.")
        return {"asset": asset_name, "path": str(path), "files": count, "kind": kind}
    if not path.is_file():
        raise FileNotFoundError(f"Required resource {asset_name} is missing: {path}")
    digest = sha256_file(path)
    expected_hash = spec.get("sha256")
    expected_size = spec.get("size_bytes")
    if expected_hash and digest != expected_hash:
        raise ValueError(f"SHA-256 mismatch for {asset_name}: expected {expected_hash}, got {digest}.")
    if expected_size is not None and path.stat().st_size != int(expected_size):
        raise ValueError(f"Size mismatch for {asset_name}: expected {expected_size}, got {path.stat().st_size}.")
    return {"asset": asset_name, "path": str(path), "sha256": digest, "size_bytes": path.stat().st_size, "kind": kind}


def _download(url: str, destination: Path, expected_hash: str | None, expected_size: int | None) -> Path:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https":
        raise ValueError(f"Only HTTPS resource URLs are accepted: {url}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "ailaai-resource-loader/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        size = partial.stat().st_size
        digest = sha256_file(partial)
        if expected_size is not None and size != int(expected_size):
            raise ValueError(f"Downloaded size mismatch for {url}: expected {expected_size}, got {size}.")
        if expected_hash and digest != expected_hash:
            raise ValueError(f"Downloaded SHA-256 mismatch for {url}.")
        partial.replace(destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return destination


def _safe_extract(archive_path: Path, destination: Path, max_unpacked_bytes: int = 8_000_000_000) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".ailaai-extract-", dir=destination.parent))
    total = 0
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                relative = PurePosixPath(member.filename)
                mode = member.external_attr >> 16
                if relative.is_absolute() or ".." in relative.parts or stat.S_ISLNK(mode):
                    raise ValueError(f"Unsafe path in archive: {member.filename}")
                total += member.file_size
                if total > max_unpacked_bytes:
                    raise ValueError("Archive exceeds the configured extraction size limit.")
                archive.extract(member, staging)
        if destination.exists():
            raise FileExistsError(f"Extraction destination already exists: {destination}")
        staging.replace(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def download_dataset(
    workspace: Workspace,
    file_id: str = "1g_43_Xn-DWYB-k7Yq4XQTr5ZQXZ0UdXq",
    force: bool = False,
) -> Path:
    """Download and extract who_is_AI dataset from Google Drive if not present."""
    data_root = workspace.data_root
    check_dirs = [
        data_root / "train" / "images",
        data_root / "images",
        data_root / "who_is_AI" / "data" / "train" / "images",
        data_root / "who_is_AI" / "train" / "images",
        data_root.parent / "who_is_AI" / "data" / "train" / "images",
    ]
    if not force and any(d.is_dir() and any(d.glob("*.jpg")) for d in check_dirs):
        return data_root

    data_root.mkdir(parents=True, exist_ok=True)
    zip_path = data_root / "who_is_AI.zip"

    if not zip_path.is_file() or zip_path.stat().st_size == 0 or force:
        print("Đang tải bộ dữ liệu who_is_AI từ Google Drive...")
        url = f"https://drive.google.com/uc?id={file_id}"
        downloaded = False
        try:
            import gdown
            gdown.download(url, str(zip_path), quiet=False)
            downloaded = zip_path.is_file() and zip_path.stat().st_size > 0
        except Exception:
            pass

        if not downloaded:
            try:
                subprocess.run(["gdown", "--id", file_id, "-O", str(zip_path)], check=True)
                downloaded = zip_path.is_file() and zip_path.stat().st_size > 0
            except Exception:
                pass

        if not downloaded:
            print(
                f"[Cảnh báo] Không thể tải tự động qua gdown. Bạn có thể tải thủ công từ:\n"
                f"https://drive.google.com/file/d/{file_id}/view?usp=drive_link\n"
                f"và đặt tệp ZIP hoặc thư mục giải nén vào {data_root}."
            )
            return data_root

    if zip_path.is_file() and zip_path.stat().st_size > 0:
        print("Đang giải nén bộ dữ liệu...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_root)
        print("Giải nén hoàn tất!")

    return data_root


def prepare_resources(
    workspace: Workspace,
    manifest_path: str | Path,
    profile: str = "e2e",
) -> list[dict[str, Any]]:
    """Verify local resources or fetch configured HTTPS assets for one profile."""
    manifest_file = Path(manifest_path)
    if not manifest_file.is_absolute():
        manifest_file = workspace.root / manifest_file
    manifest = read_json(manifest_file)
    try:
        names = manifest["profiles"][profile]
        assets = manifest["assets"]
    except KeyError as exc:
        raise ValueError(f"Unknown resource profile or missing section: {profile}") from exc
    receipts: list[dict[str, Any]] = []
    for name in names:
        if name not in assets:
            raise ValueError(f"Resource profile {profile} references unknown asset {name}.")
        spec = assets[name]
        destination_root = workspace.data_root if spec.get("root") == "data" else workspace.root
        destination = _safe_child(destination_root, str(spec["destination"]))
        if not destination.exists():
            if name in ("train_images", "test_images"):
                download_dataset(workspace)
            if name == "test_images":
                candidates = [
                    destination_root / "private_test" / "private_test" / "images",
                    destination_root / "private_test" / "images",
                    destination_root / "public_test" / "images",
                    destination_root / "who_is_AI" / "data" / "private_test" / "private_test" / "images",
                    destination_root / "who_is_AI" / "data" / "private_test" / "images",
                    destination_root / "who_is_AI" / "private_test" / "private_test" / "images",
                    destination_root / "who_is_AI" / "private_test" / "images",
                    destination_root.parent / "who_is_AI" / "data" / "private_test" / "private_test" / "images",
                    destination_root.parent / "who_is_AI" / "data" / "private_test" / "images",
                ]
                cand = next((c.resolve() for c in candidates if c.is_dir()), None)
                if cand is not None:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        destination.symlink_to(cand)
                    except OSError:
                        shutil.copytree(cand, destination)
            elif name == "train_images":
                candidates = [
                    destination_root / "images",
                    destination_root / "who_is_AI" / "data" / "train" / "images",
                    destination_root / "who_is_AI" / "train" / "images",
                    destination_root.parent / "who_is_AI" / "data" / "train" / "images",
                ]
                cand = next((c.resolve() for c in candidates if c.is_dir()), None)
                if cand is not None:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        destination.symlink_to(cand)
                    except OSError:
                        shutil.copytree(cand, destination)
        if destination.exists():
            receipts.append(_validate_existing(destination, spec, name))
            continue
        sources = list(spec.get("sources", []))
        if not sources:
            raise FileNotFoundError(
                f"Resource {name} is not available at {destination}; add an approved HTTPS source "
                "with SHA-256 to configs/resources.json or place the files at the documented path."
            )
        source = sources[0]
        if isinstance(source, str):
            source = {"url": source}
        url = str(source["url"])
        cache_dir = workspace.data_root / ".downloads"
        archive_name = str(source.get("filename") or Path(urllib.parse.urlparse(url).path).name or f"{name}.asset")
        cached = _download(url, cache_dir / archive_name, spec.get("sha256"), spec.get("size_bytes"))
        if spec.get("archive", False):
            _safe_extract(cached, destination, int(spec.get("max_unpacked_bytes", 8_000_000_000)))
            receipts.append(_validate_existing(destination, spec, name))
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cached, destination)
            receipts.append(_validate_existing(destination, spec, name))
    receipt_path = workspace.data_root / "receipts" / f"{profile}.json"
    write_json(receipt_path, {"profile": profile, "assets": receipts})
    return receipts


def check_environment(profile: str = "e2e") -> dict[str, Any]:
    """Report library/device versions and require CUDA only for new training."""
    try:
        import torch
        import torchvision
    except Exception as exc:
        raise RuntimeError("Install the package dependencies and restart the runtime before continuing.") from exc
    cuda_available = bool(torch.cuda.is_available())
    if profile == "train" and not cuda_available:
        raise RuntimeError("Training requires a CUDA GPU; choose load mode or attach a GPU runtime.")
    if profile not in {"e2e", "train", "infer", "replay", "smoke"}:
        raise ValueError(f"Unknown environment profile: {profile}")
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": cuda_available,
        "device": torch.cuda.get_device_name(0) if cuda_available else "cpu",
        "is_colab": "google.colab" in sys.modules or Path("/content").exists(),
    }


def verify_checkout(repository: str | Path, expected_sha: str | None = None) -> dict[str, str]:
    """Read Git revision and dirty state without changing the checkout."""
    root = Path(repository)
    try:
        head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Could not inspect the package checkout at {root}.") from exc
    if expected_sha and head != expected_sha:
        raise RuntimeError(f"Checkout revision mismatch: expected {expected_sha}, got {head}.")
    if dirty.strip():
        raise RuntimeError("The code checkout has local changes; use a fresh task folder before running.")
    return {"head": head, "dirty": "false"}


def asset_path(workspace: Workspace, manifest_path: str | Path, asset_name: str) -> Path:
    """Resolve one declared asset path and ensure it remains inside the task root."""
    manifest_file = Path(manifest_path)
    if not manifest_file.is_absolute():
        manifest_file = workspace.root / manifest_file
    spec = read_json(manifest_file)["assets"][asset_name]
    destination_root = workspace.data_root if spec.get("root") == "data" else workspace.root
    return _safe_child(destination_root, str(spec["destination"]))
