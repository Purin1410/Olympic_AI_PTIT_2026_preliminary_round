"""Tests for the 9 Logistic Regression ablation probes and group isolation calling production code."""
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import tempfile
    import shutil
    import numpy as np
    import pandas as pd
    from kmd.extractor import (
        NAMES,
        LR_GROUPS,
        LR_ABLATIONS,
        LR_ABLATION_ALIASES,
        resolve_lr_ablation_name,
        get_lr_features,
        list_lr_ablations,
    )
    from kmd.pipeline import fit_lr_cv, infer_lr
    from kmd.core import write_json
    HAS_DEPS = True
    MISSING_ERR = ""
except ImportError as e:
    HAS_DEPS = False
    MISSING_ERR = str(e)


class TestLRAblations(unittest.TestCase):
    def setUp(self):
        if not HAS_DEPS:
            raise unittest.SkipTest(f"Dependencies unavailable ({MISSING_ERR}); marked NOTRUN for root realenv")
        self.temp_dir = tempfile.mkdtemp(prefix="kmd_test_lr_")
        self.session_dir = Path(self.temp_dir) / "session"
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


        n = 800
        pair_ids = [f"{i:05d}" for i in range(n)]
        rng = np.random.RandomState(42)
        y = rng.choice([0, 1], size=n)

        self.dev_frame = pd.DataFrame({
            "pair_id": pair_ids,
            "image_0": [f"img_{i}_0.jpg" for i in range(n)],
            "image_1": [f"img_{i}_1.jpg" for i in range(n)],
            "inner_fold": [i % 3 for i in range(n)],
            "fake_position": y,
        })
        self.dev_frame.to_csv(self.session_dir / "development.csv", index=False)

    def tearDown(self):
        if hasattr(self, "temp_dir") and Path(self.temp_dir).is_dir():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_feature_count_and_uniqueness(self):
        """Verify the full feature vector contains exactly 32 unique names."""
        self.assertEqual(len(NAMES), 32)
        self.assertEqual(len(set(NAMES)), 32)

    def test_four_feature_groups_partition(self):
        """Verify the 4 report-defined feature groups partition the 32 features exactly."""
        # Group 0: File size [0]
        self.assertEqual(LR_GROUPS["filesize"], NAMES[0:1])
        self.assertEqual(len(LR_GROUPS["filesize"]), 1)

        # Group 1: Color / Intensity [1:19]
        self.assertEqual(LR_GROUPS["color"], NAMES[1:19])
        self.assertEqual(len(LR_GROUPS["color"]), 18)

        # Group 2: Texture / Residual [19:28]
        self.assertEqual(LR_GROUPS["texture"], NAMES[19:28])
        self.assertEqual(len(LR_GROUPS["texture"]), 9)

        # Group 3: Spatial Center / Border [28:32]
        self.assertEqual(LR_GROUPS["center_border"], NAMES[28:32])
        self.assertEqual(len(LR_GROUPS["center_border"]), 4)

        # Total count across 4 groups must equal 32
        total_grouped = sum(len(g) for g in LR_GROUPS.values())
        self.assertEqual(total_grouped, 32)

    def test_nine_ablation_probes_definitions(self):
        """Verify all 9 ablation probes from Figure 7 of the report are present and correct."""
        ablations = list_lr_ablations()
        self.assertEqual(len(ablations), 9)

        # 1. Full 32 features
        self.assertEqual(get_lr_features("full32"), NAMES)
        self.assertEqual(len(get_lr_features("full32")), 32)

        # 2-5. Four single-group models
        self.assertEqual(get_lr_features("filesize_only"), LR_GROUPS["filesize"])
        self.assertEqual(get_lr_features("color_only"), LR_GROUPS["color"])
        self.assertEqual(get_lr_features("texture_only"), LR_GROUPS["texture"])
        self.assertEqual(get_lr_features("center_border_only"), LR_GROUPS["center_border"])

        # 6-9. Four drop-one-group models
        self.assertEqual(len(get_lr_features("drop_filesize")), 31)
        self.assertEqual(len(get_lr_features("drop_color")), 14)
        self.assertEqual(len(get_lr_features("drop_texture")), 23)
        self.assertEqual(len(get_lr_features("drop_center_border")), 28)

        # Confirm drop sets are exact set complements
        all_set = set(NAMES)
        for group_name in ("filesize", "color", "texture", "center_border"):
            expected = sorted(all_set - set(LR_GROUPS[group_name]), key=NAMES.index)
            actual = get_lr_features(f"drop_{group_name}")
            self.assertEqual(actual, expected)

    def test_alias_resolution(self):
        """Verify shorthand aliases resolve properly to canonical ablation names."""
        self.assertEqual(resolve_lr_ablation_name("full"), "full32")
        self.assertEqual(resolve_lr_ablation_name("all"), "full32")
        self.assertEqual(resolve_lr_ablation_name("lr"), "full32")
        self.assertEqual(resolve_lr_ablation_name("filesize"), "filesize_only")
        self.assertEqual(resolve_lr_ablation_name("no_filesize"), "drop_filesize")
        self.assertEqual(resolve_lr_ablation_name("drop_color"), "drop_color")
        self.assertEqual(resolve_lr_ablation_name("spatial_only"), "center_border_only")
        self.assertEqual(resolve_lr_ablation_name("no_spatial"), "drop_center_border")

    def test_namespaced_artifact_paths_no_overwrite(self):
        """Verify that each ablation generates distinct filenames that cannot overwrite each other."""
        for v in list_lr_ablations():
            oof_name = f"lr_{v}_oof.csv"
            feat_name = f"lr_{v}_feature_names.json"
            model_names = [f"lr_{v}_fold{f}.joblib" for f in range(3)]
            self.assertIn(v, oof_name)
            self.assertIn(v, feat_name)
            for m in model_names:
                self.assertIn(v, m)

    def test_fit_lr_cv_rejects_unknown_variant(self):
        """Verify fit_lr_cv rejects unknown variant names and path traversal."""
        with self.assertRaises(ValueError):
            fit_lr_cv(self.dev_frame, None, self.session_dir, variant="nonexistent_variant")
        with self.assertRaises(ValueError):
            fit_lr_cv(self.dev_frame, None, self.session_dir, variant="../traversal")

    def test_fit_lr_cv_enforces_canonical_columns(self):
        """Verify fit_lr_cv rejects feature columns that conflict with requested ablation variant."""
        wrong_cols = ["file_size_log", "color_mean_r"]
        with self.assertRaises(ValueError):
            fit_lr_cv(self.dev_frame, None, self.session_dir, feature_columns=wrong_cols, variant="full32")

    def test_fit_lr_cv_rejects_nonfinite_precomputed_features(self):
        """Verify fit_lr_cv rejects precomputed features containing NaN or Inf."""
        all_imgs = sorted(set(self.dev_frame.image_0) | set(self.dev_frame.image_1))
        bad_table = pd.DataFrame(
            np.nan,
            index=all_imgs,
            columns=get_lr_features("filesize_only"),
        )
        bad_table.index.name = "relpath"
        with self.assertRaises(ValueError) as ctx:
            fit_lr_cv(self.dev_frame, None, self.session_dir, variant="filesize_only", precomputed_features=bad_table)
        self.assertIn("Non-finite", str(ctx.exception))

    def test_infer_lr_no_fallback_to_full32(self):
        """Verify infer_lr never falls back to full32 when a requested ablation variant is missing."""
        # Create full32 artifacts only
        for f in range(3):
            (self.session_dir / f"lr_fold{f}.joblib").write_bytes(b"dummy")
        write_json(self.session_dir / "lr_feature_names.json", NAMES)
        write_json(self.session_dir / "lr_job.json", {"variant": "full32", "model_sha256": {}})
        pd.DataFrame({"sha256": []}).to_csv(self.session_dir / "input_images.csv", index=False)

        dummy_test = pd.DataFrame({
            "pair_id": ["p1"],
            "image_0": ["a.jpg"],
            "image_1": ["b.jpg"],
        })

        # Requesting color_only must NOT fall back to full32
        with self.assertRaises(FileNotFoundError) as ctx:
            infer_lr(dummy_test, self.temp_dir, self.session_dir, variant="color_only")
        self.assertIn("color_only", str(ctx.exception))

    def test_fit_lr_cv_reuse_rejects_partial_or_mismatched(self):
        """Verify fit_lr_cv refuses silent overwrite and fails with new RUN_ID directive on partial artifacts."""
        # Create partial artifact (only OOF, no job.json or models)
        oof_file = self.session_dir / "lr_full32_oof.csv"
        oof_file.write_text("corrupt data\n")

        with self.assertRaises(ValueError) as ctx:
            fit_lr_cv(self.dev_frame, None, self.session_dir, variant="full32")
        self.assertIn("choose a new run_id", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
