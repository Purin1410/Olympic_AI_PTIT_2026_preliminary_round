"""Tests for canonical run namespacing, seed collision resistance, legacy symlinks, and checkpoint validation calling production code."""
import tempfile
import shutil
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import numpy as np
    import pandas as pd
    from kmd.pipeline import (
        fold_run_name,
        find_fold_folder,
        validate_fold_checkpoint,
        compare_oof,
    )
    from kmd.presets import get_preset_config, canonical_preset_name, list_registered_presets
    from kmd.core import write_json
    from kmd.config import Config
    HAS_DEPS = True
    MISSING_ERR = ""
except ImportError as e:
    HAS_DEPS = False
    MISSING_ERR = str(e)


class TestNamespaces(unittest.TestCase):
    def setUp(self):
        if not HAS_DEPS:
            raise unittest.SkipTest(f"Dependencies unavailable ({MISSING_ERR}); marked NOTRUN for root realenv")

        self.temp_dir = tempfile.mkdtemp(prefix="kmd_test_ns_")
        self.session_dir = Path(self.temp_dir) / "test_session"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        from kmd.core import PACKAGE, sha256
        from kmd.pipeline import current_code_hashes
        write_json(self.session_dir / 'session.json', {
            'split_sha256': sha256(PACKAGE / 'configs/development_split.csv'),
            'code_sha256': current_code_hashes(),
        })
        pd.DataFrame({'sha256': []}).to_csv(self.session_dir / 'input_images.csv', index=False)
        pd.DataFrame({'pair_id': [], 'image_0': [], 'image_1': []}).to_csv(self.session_dir / 'development.csv', index=False)
        for name in ('a.jpg', 'b.jpg', 't0.jpg', 't1.jpg'):
            (Path(self.temp_dir) / name).write_bytes(name.encode())


    def tearDown(self):
        if hasattr(self, "temp_dir") and Path(self.temp_dir).is_dir():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_presets_match_original_jobs_and_center60_json(self):
        """Verify presets center60, center60_micro8, B_cap48, frozen_lr match exact specifications."""
        # 1. center60 matches configs/center60.json and A_full_batch24: microbatch 24, effective_batch 24, epochs 24
        c60 = get_preset_config("center60")
        self.assertEqual(c60.microbatch, 24)
        self.assertEqual(c60.effective_batch, 24)
        self.assertEqual(c60.epochs, 24)
        self.assertEqual(c60.workers, 4)

        c60_b24 = get_preset_config("center60_batch24")
        self.assertEqual(c60_b24, c60)

        # 2. A_full_micro8 has microbatch 8, effective 24, epochs 24
        c60_u8 = get_preset_config("A_full_micro8")
        self.assertEqual(c60_u8.microbatch, 8)
        self.assertEqual(c60_u8.effective_batch, 24)
        self.assertEqual(c60_u8.epochs, 24)
        self.assertEqual(c60_u8.workers, 4)

        # 3. B_cap48 has microbatch 8, effective 24, epochs 48
        b_cap48 = get_preset_config("B_cap48")
        self.assertEqual(b_cap48.microbatch, 8)
        self.assertEqual(b_cap48.effective_batch, 24)
        self.assertEqual(b_cap48.epochs, 48)
        self.assertEqual(b_cap48.workers, 4)

        # 4. frozen_lr has kind="ml", feature="embedding", microbatch 24, effective 24, epochs 24
        flr = get_preset_config("frozen_lr")
        self.assertEqual(flr.kind, "ml")
        self.assertEqual(flr.feature, "embedding")
        self.assertEqual(flr.microbatch, 24)
        self.assertEqual(flr.effective_batch, 24)
        self.assertEqual(flr.epochs, 24)
        self.assertEqual(flr.workers, 4)

        # 5. Smoke preserves workers=0
        c60_smoke = get_preset_config("center60", smoke=True)
        self.assertEqual(c60_smoke.workers, 0)
        self.assertEqual(c60_smoke.epochs, 2)
        self.assertEqual(c60_smoke.warmup, 1)

    def test_fold_run_name_formatting(self):
        """Verify canonical fold run name formatting for various presets, seeds, and smoke modes."""
        self.assertEqual(fold_run_name("b2", 0, 20260917, smoke=False), "b2_s20260917_f0")
        self.assertEqual(fold_run_name("native", 1, 20260918, smoke=False), "native_s20260918_f1")
        self.assertEqual(fold_run_name("b2", 0, 20260917, smoke=True), "b2_smoke_s20260917_f0")
        self.assertEqual(fold_run_name("partial_native", 0, 20260917, smoke=False), "partial_native_s20260917_f0")

    def test_seed_collision_resistance(self):
        """Verify runs with different seeds produce distinct directory paths that do not collide."""
        seeds = [20260917, 20260918, 20260919]
        names = [fold_run_name("b2", 0, seed, smoke=False) for seed in seeds]
        self.assertEqual(len(set(names)), 3)

    def test_find_fold_folder_canonical_and_legacy(self):
        """Verify find_fold_folder resolves canonical names and backwards-compatible legacy symlinks."""
        c_dir = self.session_dir / "b2_s20260917_f0"
        c_dir.mkdir(parents=True, exist_ok=True)
        found = find_fold_folder(self.session_dir, "b2", 0, seed=20260917, smoke=False)
        self.assertEqual(found, c_dir)

        leg_dir = self.session_dir / "native_fold1"
        leg_dir.mkdir(parents=True, exist_ok=True)
        found_leg = find_fold_folder(self.session_dir, "native", 1, seed=20260917, smoke=False)
        self.assertEqual(found_leg, leg_dir)

        found_other = find_fold_folder(self.session_dir, "native", 1, seed=20260918, smoke=False)
        self.assertIsNone(found_other)

    def test_compare_oof_mode_separation_and_missing_rejection(self):
        """Verify compare_oof strictly separates smoke/non-smoke and raises FileNotFoundError on missing models."""
        dev = pd.DataFrame({
            "pair_id": ["001", "002"],
            "image_0": ["a.jpg", "c.jpg"],
            "image_1": ["b.jpg", "d.jpg"],
            "inner_fold": [0, 1],
            "fake_position": [1, 0],
        })
        dev.to_csv(self.session_dir / "development.csv", index=False)

        # Missing model must raise FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            compare_oof(self.session_dir, ["b2"], seed=20260917, smoke=False)

        # Create non-smoke OOF for CNN
        oof_full = dev.copy()
        oof_full["p"] = 0.8
        oof_full.to_csv(self.session_dir / "b2_s20260917_oof.csv", index=False)

        # Requesting smoke CNN must NOT fallback to non-smoke
        with self.assertRaises(FileNotFoundError):
            compare_oof(self.session_dir, ["b2"], seed=20260917, smoke=True)

    def test_compare_oof_lr_resolves_in_smoke_mode(self):
        """Verify compare_oof resolves LR to full-dev deterministic artifacts even under smoke mode."""
        dev = pd.DataFrame({
            "pair_id": ["001", "002"],
            "image_0": ["a.jpg", "c.jpg"],
            "image_1": ["b.jpg", "d.jpg"],
            "inner_fold": [0, 1],
            "fake_position": [1, 0],
        })
        dev.to_csv(self.session_dir / "development.csv", index=False)

        # Missing LR raises FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            compare_oof(self.session_dir, ["lr"], seed=20260917, smoke=True)

        # Create full-dev LR OOF
        lr_oof = dev.copy()
        lr_oof["p"] = 0.6
        lr_oof.to_csv(self.session_dir / "lr_full32_oof.csv", index=False)

        # In smoke mode, compare_oof successfully finds LR full-dev artifact
        table, preds = compare_oof(self.session_dir, ["lr"], seed=20260917, smoke=True)
        self.assertIn("lr", preds)
        self.assertEqual(len(table), 1)
        self.assertEqual(table.iloc[0]["model"], "lr")

    def test_infer_cnn_rejects_missing_job_json(self):
        """Verify infer_cnn rejects checkpoints that lack the mandatory job.json contract."""
        from kmd.pipeline import infer_cnn
        records = [
            {"fold": 0, "checkpoint": "b2_s20260917_f0/best.pt", "sha256": "fake1"},
            {"fold": 1, "checkpoint": "b2_s20260917_f1/best.pt", "sha256": "fake2"},
            {"fold": 2, "checkpoint": "b2_s20260917_f2/best.pt", "sha256": "fake3"},
        ]
        write_json(self.session_dir / "b2_s20260917_models.json", records)

        # Create dummy checkpoint files without job.json
        for rec in records:
            p = self.session_dir / rec["checkpoint"]
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"dummy")
            from kmd.core import sha256
            rec["sha256"] = sha256(p)
        write_json(self.session_dir / "b2_s20260917_models.json", records)

        # input_images.csv for separation check
        pd.DataFrame({"sha256": []}).to_csv(self.session_dir / "input_images.csv", index=False)

        dummy_test = pd.DataFrame({
            "pair_id": ["t1"],
            "image_0": ["t0.jpg"],
            "image_1": ["t1.jpg"],
        })
        dummy_test_root = self.temp_dir

        with self.assertRaises(FileNotFoundError) as ctx:
            infer_cnn("b2", dummy_test, dummy_test_root, self.session_dir, seed=20260917, smoke=False)
        self.assertIn("job.json", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
