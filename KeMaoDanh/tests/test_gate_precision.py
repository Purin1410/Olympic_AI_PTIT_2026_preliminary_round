"""Tests for Blend gate decision logic, exact floating-point precision, and smoke rejection calling production code."""
import tempfile
import shutil
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import numpy as np
    import pandas as pd
    from kmd.gate import evaluate_blend_gate, compute_seed_delta
    from kmd.core import write_json
    HAS_DEPS = True
    MISSING_ERR = ""
except ImportError as e:
    HAS_DEPS = False
    MISSING_ERR = str(e)


class TestGatePrecision(unittest.TestCase):
    def setUp(self):
        if not HAS_DEPS:
            raise unittest.SkipTest(f"Dependencies unavailable ({MISSING_ERR}); marked NOTRUN for root realenv")

        self.temp_dir = tempfile.mkdtemp(prefix="kmd_test_gate_")
        self.session_dir = Path(self.temp_dir) / "session"
        self.session_dir.mkdir(parents=True, exist_ok=True)

        n = 800
        pair_ids = [f"{i:05d}" for i in range(n)]
        # Deterministic labels
        rng = np.random.RandomState(42)
        y = rng.choice([0, 1], size=n)

        self.dev_frame = pd.DataFrame({
            "pair_id": pair_ids,
            "image_0": [f"{p}a.jpg" for p in pair_ids],
            "image_1": [f"{p}b.jpg" for p in pair_ids],
            "inner_fold": [i % 3 for i in range(n)],
            "fake_position": y,
        })
        self.dev_frame.to_csv(self.session_dir / "development.csv", index=False)

    def tearDown(self):
        if hasattr(self, "temp_dir") and Path(self.temp_dir).is_dir():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_unrounded_delta_boundary_precision(self):
        """Verify the gate evaluates exact unrounded delta against the 0.005 threshold."""
        from kmd.gate import summarize_deltas
        mean_pass, accepted = summarize_deltas([.005001, .005000, .005002])
        self.assertTrue(accepted)
        mean_fail, accepted = summarize_deltas([.004999, .004998, .005000])
        self.assertFalse(accepted)
        self.assertEqual(round(mean_pass, 4), round(mean_fail, 4))

    def test_smoke_mode_refusal_to_conclude(self):
        """Verify the gate returns decision=None and mean_delta_percent=None in smoke mode."""
        res = evaluate_blend_gate(self.session_dir, is_smoke=True, dev_frame=self.dev_frame)
        self.assertIsNone(res["decision"])
        self.assertEqual(res["status"], "smoke_run_no_conclusion")
        self.assertFalse(res["complete"])
        self.assertFalse(res["gate_passed"])
        self.assertIsNone(res["mean_delta_unrounded"])
        self.assertIsNone(res["mean_delta_percent"])

    def test_missing_seeds_returns_none_decision(self):
        """Verify the gate returns decision=None when seeds are missing."""
        res = evaluate_blend_gate(self.session_dir, is_smoke=False, dev_frame=self.dev_frame)
        self.assertIsNone(res["decision"])
        self.assertEqual(res["status"], "incomplete_seeds")
        self.assertFalse(res["complete"])
        self.assertFalse(res["gate_passed"])
        self.assertIsNone(res["mean_delta_percent"])

    def test_invalid_seeds_count_rejected(self):
        """Verify the gate requires exactly 3 unique expected seeds (len == 3)."""
        res_two = evaluate_blend_gate(self.session_dir, seeds=[20260917, 20260918], dev_frame=self.dev_frame)
        self.assertIsNone(res_two["decision"])
        self.assertEqual(res_two["status"], "invalid_seeds")
        self.assertFalse(res_two["complete"])

        res_dup = evaluate_blend_gate(self.session_dir, seeds=[20260917, 20260917, 20260918], dev_frame=self.dev_frame)
        self.assertIsNone(res_dup["decision"])
        self.assertEqual(res_dup["status"], "invalid_seeds")

    def test_loose_csvs_without_models_json_rejected(self):
        """Verify the gate fails closed when loose CSVs exist without mandatory models.json provenance."""
        seeds = (20260917, 20260918, 20260919)
        for s in seeds:
            oof = self.dev_frame.copy()
            oof["p"] = 0.5
            oof.to_csv(self.session_dir / f"b2_s{s}_oof.csv", index=False)
            oof.to_csv(self.session_dir / f"native_s{s}_oof.csv", index=False)

        res = evaluate_blend_gate(self.session_dir, seeds=seeds, is_smoke=False, dev_frame=self.dev_frame)
        self.assertIsNone(res["decision"])
        self.assertEqual(res["status"], "missing_provenance")
        self.assertFalse(res["complete"])
        self.assertIn("loose CSVs", res["reason"])

    def test_compute_seed_delta_requires_aligned_predictions(self):
        """Verify compute_seed_delta enforces strict aligned_predictions validation on inputs."""
        oof_bad = self.dev_frame.iloc[:-5].copy()
        oof_bad["p"] = 0.5
        oof_good = self.dev_frame.copy()
        oof_good["p"] = 0.5

        with self.assertRaises(ValueError):
            compute_seed_delta(oof_bad, oof_good, self.dev_frame)


if __name__ == "__main__":
    unittest.main()
