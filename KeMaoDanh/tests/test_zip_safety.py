"""Tests for dataset zip safety, symlink rejection, preflight validation, byte preservation, and ground-truth isolation."""
import os
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import numpy as np
    import pandas as pd
    from kmd.dataset_prep import (
        is_safe_zip_path,
        is_ignored_zip_entry,
        is_special_or_forbidden_zip_member,
        extract_official_zip,
        discover_dataset_roots,
        validate_official_layout,
    )
    from kmd.pipeline import load_pairs
    HAS_DEPS = True
    MISSING_ERR = ""
except ImportError as e:
    HAS_DEPS = False
    MISSING_ERR = str(e)


class TestZipSafety(unittest.TestCase):
    def setUp(self):
        if not HAS_DEPS:
            raise unittest.SkipTest(f"Dependencies unavailable ({MISSING_ERR}); marked NOTRUN for root realenv")
        self.temp_dir = tempfile.mkdtemp(prefix="kmd_test_zip_")

    def tearDown(self):
        if hasattr(self, "temp_dir") and os.path.isdir(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_zip_path_traversal_prevention(self):
        """Verify Zip Slip path traversal attempts are detected and rejected."""
        target = Path(self.temp_dir) / "dest"
        target.mkdir(parents=True, exist_ok=True)

        # Dangerous paths
        self.assertFalse(is_safe_zip_path(target, "../evil.txt"))
        self.assertFalse(is_safe_zip_path(target, "../../etc/passwd"))
        self.assertFalse(is_safe_zip_path(target, "/absolute/path/file.jpg"))
        self.assertFalse(is_safe_zip_path(target, "sub/../../outside.py"))

        # Safe paths
        self.assertTrue(is_safe_zip_path(target, "train/pairs.csv"))
        self.assertTrue(is_safe_zip_path(target, "data/train/images/img_001.jpg"))
        self.assertTrue(is_safe_zip_path(target, "private_test/private_test/pairs.csv"))

    def test_extract_rejects_malicious_traversal(self):
        """Verify extract_official_zip raises ValueError on archives with traversal entries."""
        zip_path = Path(self.temp_dir) / "malicious.zip"
        dest_dir = Path(self.temp_dir) / "extracted"

        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("../traversal.txt", b"malicious content")

        with self.assertRaises(ValueError) as ctx:
            extract_official_zip(zip_path, dest_dir)
        self.assertIn("Dangerous zip entry detected", str(ctx.exception))

    def test_extract_rejects_symlinks_and_skips_notebooks(self):
        """Verify extract_official_zip rejects symlinks and forbidden notebook files."""
        # 1. Reject notebook file in dataset zip
        nb_zip = Path(self.temp_dir) / "with_notebook.zip"
        with zipfile.ZipFile(nb_zip, "w") as zf:
            zf.writestr("baseline.ipynb", b'{"cells": []}')

        result = extract_official_zip(nb_zip, Path(self.temp_dir) / "extracted_nb")
        self.assertEqual(result["skipped_files"], ["baseline.ipynb"])
        self.assertFalse((Path(self.temp_dir) / "extracted_nb/baseline.ipynb").exists())

        # 2. Reject symbolic link member
        link_zip = Path(self.temp_dir) / "with_symlink.zip"
        with zipfile.ZipFile(link_zip, "w") as zf:
            info = zipfile.ZipInfo("symlink_entry")
            info.create_system = 3  # Unix
            info.external_attr = 0o120777 << 16  # S_IFLNK
            zf.writestr(info, b"/etc/passwd")

        with self.assertRaises(ValueError) as ctx:
            extract_official_zip(link_zip, Path(self.temp_dir) / "extracted_link")
        self.assertIn("Forbidden symbolic link", str(ctx.exception))

    def test_preflight_check_prevents_partial_writes(self):
        """Verify safe preflight check ensures no files are extracted if a later member is unsafe."""
        bad_zip = Path(self.temp_dir) / "partial_test.zip"
        dest_dir = Path(self.temp_dir) / "dest_preflight"

        with zipfile.ZipFile(bad_zip, "w") as zf:
            zf.writestr("valid_file.txt", b"valid")
            zf.writestr("../bad_file.txt", b"invalid")

        with self.assertRaises(ValueError):
            extract_official_zip(bad_zip, dest_dir)

        # Preflight should have failed before writing valid_file.txt
        self.assertFalse((dest_dir / "valid_file.txt").exists())

    def test_byte_preservation(self):
        """Verify raw image bytes are preserved exactly bit-for-bit without re-encoding."""
        zip_path = Path(self.temp_dir) / "valid.zip"
        dest_dir = Path(self.temp_dir) / "extracted"

        test_binary = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00" + os.urandom(1024)

        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("train/images/img_0001.jpg", test_binary)
            zf.writestr("train/pairs.csv", "pair_id,image_0,image_1,fake_position\n0,a.jpg,b.jpg,1\n")

        res = extract_official_zip(zip_path, dest_dir)
        self.assertEqual(res["status"], "extracted")

        extracted_file = dest_dir / "train/images/img_0001.jpg"
        self.assertTrue(extracted_file.is_file())
        self.assertEqual(extracted_file.read_bytes(), test_binary)

    def test_nested_private_test_discovery(self):
        """Verify discovery of nested data/private_test/private_test structure."""
        base = Path(self.temp_dir) / "official_layout"
        train_dir = base / "data" / "train"
        public_dir = base / "data" / "public_test"
        private_nested = base / "data" / "private_test" / "private_test"

        for d in (train_dir, public_dir, private_nested):
            d.mkdir(parents=True, exist_ok=True)
            (d / "pairs.csv").write_text("pair_id,image_0,image_1\n")

        roots = discover_dataset_roots(base)
        self.assertEqual(roots["train"], train_dir)
        self.assertEqual(roots["public_test"], public_dir)
        self.assertEqual(roots["private_test"], private_nested)

    def test_refuse_silent_overwrite(self):
        """Verify extract_official_zip refuses to overwrite unless overwrite=True."""
        zip_path = Path(self.temp_dir) / "sample.zip"
        dest_dir = Path(self.temp_dir) / "dest_overwrite"

        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("pairs.csv", "dummy")

        extract_official_zip(zip_path, dest_dir, overwrite=False)

        with self.assertRaises(FileExistsError):
            extract_official_zip(zip_path, dest_dir, overwrite=False)

        res = extract_official_zip(zip_path, dest_dir, overwrite=True)
        self.assertEqual(res["status"], "extracted")

    def test_validate_official_layout_meaningful_failure_and_optional_test(self):
        """Verify layout check fails meaningfully on malformed manifests and accepts absent test sets."""
        base = Path(self.temp_dir) / "layout_test"
        train_dir = base / "train"
        train_dir.mkdir(parents=True, exist_ok=True)

        # Missing required fake_position column in train
        (train_dir / "pairs.csv").write_text("pair_id,image_0,image_1\n001,a.jpg,b.jpg\n")
        roots = {"train": train_dir, "public_test": None, "private_test": None}

        with self.assertRaises(ValueError) as ctx:
            validate_official_layout(roots)
        self.assertIn("missing columns", str(ctx.exception))

        # Valid train, optional test absent -> should succeed without error
        (train_dir / "pairs.csv").write_text(
            "pair_id,image_0,image_1,fake_position\n"
            + "\n".join(f"{i:05d},img_{i}a.jpg,img_{i}b.jpg,{i % 2}" for i in range(1000))
        )
        report = validate_official_layout(roots)
        self.assertTrue(report["train_valid"])
        self.assertEqual(report["train_pairs"], 1000)
        self.assertFalse(report["public_test_present"])
        self.assertFalse(report["private_test_present"])

    def test_pairs_results_isolation(self):
        """Verify pairs_results.csv is strictly ignored during inference and cannot be leaked."""
        test_dir = Path(self.temp_dir) / "test_set"
        test_dir.mkdir(parents=True, exist_ok=True)

        pairs_csv = test_dir / "pairs.csv"
        pairs_csv.write_text(
            "pair_id,image_0,image_1\n"
            "00001,img_a.jpg,img_b.jpg\n"
            "00002,img_c.jpg,img_d.jpg\n"
        )

        results_csv = test_dir / "pairs_results.csv"
        results_csv.write_text(
            "pair_id,fake_position,ground_truth\n"
            "00001,1,1\n"
            "00002,0,0\n"
        )

        frame = load_pairs(test_dir, labeled=False)
        self.assertIn("pair_id", frame)
        self.assertIn("image_0", frame)
        self.assertIn("image_1", frame)
        self.assertNotIn("fake_position", frame)
        self.assertNotIn("ground_truth", frame)

        pairs_csv.unlink()
        with self.assertRaises(FileNotFoundError):
            load_pairs(test_dir, labeled=False)


if __name__ == "__main__":
    unittest.main()
