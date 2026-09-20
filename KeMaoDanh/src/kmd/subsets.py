"""Data subsetting and smoke sampling rules matching the verified report engine.

Adheres strictly to SUBSET_RULE:
- Training fold is subsetted using train_test_split with random_state = 20260917 + fold
- Stratification by train.fake_position
- Indices are sorted to preserve original order
- Validation fold is strictly unchanged (800 development pairs preserved across 3 folds)
- Fraction 1.0 is a no-op
"""
from typing import Tuple
import numpy as np
import pandas as pd


def sample_train_subset(
    train_frame: pd.DataFrame,
    fraction: float,
    fold: int,
    base_seed: int = 20260917,
) -> pd.DataFrame:
    """Sample a deterministic stratified subset of the training frame according to SUBSET_RULE.

    Validation is untouched. Fraction must be a finite float in (0, 1].
    Fraction 1.0 returns the full training frame without subsetting.
    """
    if fraction is None or not np.isfinite(fraction):
        raise ValueError(f"train_fraction must be a finite float, got {fraction}")

    if fraction <= 0.0 or fraction > 1.0:
        raise ValueError(f"train_fraction must be in (0, 1], got {fraction}")

    if fraction == 1.0:
        return train_frame.reset_index(drop=True)

    from sklearn.model_selection import train_test_split

    n_samples = len(train_frame)
    indices = np.arange(n_samples)
    y = train_frame["fake_position"].to_numpy()

    # Apply train_test_split with exact rule: random_state = base_seed + fold
    sub_indices, _ = train_test_split(
        indices,
        train_size=fraction,
        random_state=base_seed + fold,
        stratify=y,
    )
    sorted_indices = np.sort(sub_indices)
    return train_frame.iloc[sorted_indices].reset_index(drop=True)


def sample_smoke_train(
    train_frame: pd.DataFrame,
    fold: int,
    max_pairs: int = 64,
    fraction: float = 1.0,
    base_seed: int = 20260917,
) -> pd.DataFrame:
    """Produce unified smoke training set: max 64 pairs stratified by fake_position.

    For data amount experiments (fraction < 1.0), fractions are taken from the
    exact same 64 pairs to allow meaningful miniature training.
    """
    n_samples = len(train_frame)
    if n_samples <= max_pairs:
        base_64 = train_frame.reset_index(drop=True)
    else:
        from sklearn.model_selection import train_test_split
        indices = np.arange(n_samples)
        y = train_frame["fake_position"].to_numpy()
        sub_indices, _ = train_test_split(
            indices,
            train_size=max_pairs,
            random_state=base_seed + fold,
            stratify=y,
        )
        base_64 = train_frame.iloc[np.sort(sub_indices)].reset_index(drop=True)

    if fraction is None or not np.isfinite(fraction):
        raise ValueError(f"train_fraction must be a finite float, got {fraction}")
    if fraction <= 0.0 or fraction > 1.0:
        raise ValueError(f"train_fraction must be in (0, 1], got {fraction}")

    if fraction < 1.0:
        from sklearn.model_selection import train_test_split
        target_size = max(2, int(round(len(base_64) * fraction)))
        indices = np.arange(len(base_64))
        y = base_64["fake_position"].to_numpy()
        frac_indices, _ = train_test_split(
            indices,
            train_size=target_size,
            random_state=base_seed + fold,
            stratify=y,
        )
        return base_64.iloc[np.sort(frac_indices)].reset_index(drop=True)

    return base_64
