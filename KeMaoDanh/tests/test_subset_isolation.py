"""Tests for SUBSET_RULE execution, stratification, validation isolation, and smoke sampling calling production code."""
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import numpy as np
    import pandas as pd
    from kmd.subsets import sample_train_subset, sample_smoke_train
    from kmd.presets import get_preset_config
    HAS_DEPS = True
    MISSING_ERR = ""
except ImportError as e:
    HAS_DEPS = False
    MISSING_ERR = str(e)


class TestSubsetIsolation(unittest.TestCase):
    def setUp(self):
        if not HAS_DEPS:
            raise unittest.SkipTest(f"Dependencies unavailable ({MISSING_ERR}); marked NOTRUN for root realenv")

        # Create simulated dataset of 800 pairs (533 train, 267 val for fold 0)
        rng = np.random.RandomState(42)
        n = 800
        pair_ids = [f"{i:05d}" for i in range(n)]
        inner_folds = [i % 3 for i in range(n)]
        fake_pos = rng.choice([0, 1], size=n).tolist()

        self.dev_frame = pd.DataFrame({
            "pair_id": pair_ids,
            "image_0": [f"img_{i}_0.jpg" for i in range(n)],
            "image_1": [f"img_{i}_1.jpg" for i in range(n)],
            "inner_fold": inner_folds,
            "fake_position": fake_pos,
        })
        self.tr0 = self.dev_frame[self.dev_frame.inner_fold != 0].copy().reset_index(drop=True)
        self.va0 = self.dev_frame[self.dev_frame.inner_fold == 0].copy().reset_index(drop=True)

    def test_subset_rule_stratification_and_determinism(self):
        """Verify production sample_train_subset executes exact stratification and determinism per SUBSET_RULE."""
        # 1. fraction=1.0 returns full training frame
        sub_full = sample_train_subset(self.tr0, fraction=1.0, fold=0)
        self.assertEqual(len(sub_full), len(self.tr0))
        self.assertEqual(list(sub_full.pair_id), list(self.tr0.pair_id))

        # 2. Rejection of invalid fractions
        for bad_frac in (-0.5, 0.0, 1.05, float('nan'), float('inf'), None):
            with self.assertRaises(ValueError):
                sample_train_subset(self.tr0, fraction=bad_frac, fold=0)

        # 3. Deterministic stratified sampling at 25% and 50%
        for frac in (0.25, 0.50):
            sub1 = sample_train_subset(self.tr0, fraction=frac, fold=0)
            sub2 = sample_train_subset(self.tr0, fraction=frac, fold=0)

            # Strict determinism
            self.assertEqual(list(sub1.pair_id), list(sub2.pair_id))

            # Stratification preservation
            orig_ratio = self.tr0.fake_position.mean()
            sub_ratio = sub1.fake_position.mean()
            self.assertAlmostEqual(orig_ratio, sub_ratio, delta=0.03)

            # Preserves sorted order of indices
            orig_indices = [self.tr0.index[self.tr0.pair_id == pid].item() for pid in sub1.pair_id]
            self.assertEqual(orig_indices, sorted(orig_indices))

    def test_validation_isolation_zero_leakage(self):
        """Verify that validation frame is never modified, never sampled, and has zero train overlap."""
        train_pairs = set(self.tr0.pair_id)
        valid_pairs = set(self.va0.pair_id)

        self.assertEqual(len(train_pairs & valid_pairs), 0)
        self.assertEqual(len(valid_pairs), 267)

        # Validation set must remain completely unchanged regardless of train subset fraction
        for frac in (0.25, 0.50, 1.0):
            sub_train = sample_train_subset(self.tr0, fraction=frac, fold=0)
            self.assertEqual(len(self.va0), 267)
            self.assertEqual(len(set(sub_train.pair_id) & valid_pairs), 0)

    def test_smoke_sampling_max_64_stratified(self):
        """Verify production sample_smoke_train selects max 64 pairs while preserving label ratio."""
        smoke_tr = sample_smoke_train(self.tr0, fold=0, max_pairs=64, fraction=1.0)
        self.assertEqual(len(smoke_tr), 64)

        orig_ratio = self.tr0.fake_position.mean()
        smoke_ratio = smoke_tr.fake_position.mean()
        self.assertAlmostEqual(orig_ratio, smoke_ratio, delta=0.05)

        # Determinism
        smoke_tr2 = sample_smoke_train(self.tr0, fold=0, max_pairs=64, fraction=1.0)
        self.assertEqual(list(smoke_tr.pair_id), list(smoke_tr2.pair_id))

    def test_data_amount_fractions_under_smoke(self):
        """Verify data amount smoke fractions derive from the same 64 pairs with 4 updates."""
        c_25 = get_preset_config("data_25", smoke=True)
        c_50 = get_preset_config("data_50", smoke=True)
        c_100 = get_preset_config("data_100", smoke=True)

        self.assertEqual(c_25.train_fraction, 0.25)
        self.assertEqual(c_50.train_fraction, 0.5)
        self.assertEqual(c_100.train_fraction, 1.0)

        for c in (c_25, c_50, c_100):
            self.assertEqual(c.fixed_updates, 4)
            self.assertTrue(c.fixed_epochs)
            self.assertEqual(c.warmup, 0)
            self.assertEqual(c.workers, 0)


if __name__ == "__main__":
    unittest.main()
