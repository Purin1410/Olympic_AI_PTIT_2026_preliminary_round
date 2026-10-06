"""Explicit teaching recipes for Haar views, sample weighting, and calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .data import decode_rgb
from .metrics import classification_report
from .transforms import native_view


def haar_view(images: torch.Tensor, crop: int = 358) -> torch.Tensor:
    """Tile four Haar bands into each RGB channel, keeping a three-channel view.

    LL is divided by two; signed details are scaled and centered at 0.5.
    Each quadrant is 179x179 for a 358 crop; no per-image normalization is used.
    """
    if crop < 2 or crop % 2:
        raise ValueError("Haar crop must be a positive even integer.")
    x = native_view(images, crop)
    a, b = x[..., 0::2, 0::2], x[..., 0::2, 1::2]
    c, d = x[..., 1::2, 0::2], x[..., 1::2, 1::2]
    ll = (a + b + c + d) / 4
    lh = (a - b + c - d) / 4 + 0.5
    hl = (a + b - c - d) / 4 + 0.5
    hh = (a - b - c + d) / 4 + 0.5
    return torch.cat((torch.cat((ll, lh), dim=-1), torch.cat((hl, hh), dim=-1)), dim=-2)


def image_features(rows: pd.DataFrame, crop: int = 358, edge_threshold: float = 0.08) -> pd.DataFrame:
    """Compute gray flag and Sobel edge fraction directly from the current images."""
    records = []
    sobel = torch.tensor([[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                          [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]])[:, None] / 8
    for i, row in enumerate(rows.itertuples(), 1):
        rgb = native_view(decode_rgb(row.path), crop)
        is_gray = bool((rgb.max(0).values - rgb.min(0).values).mean() <= 1 / 255)
        gray = (rgb * rgb.new_tensor([0.299, 0.587, 0.114])[:, None, None]).sum(0)[None, None]
        gradients = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="reflect"), sobel)
        fraction = float((gradients.square().sum(1).sqrt() > edge_threshold).float().mean())
        records.append({"file_name": row.file_name, "is_gray": int(is_gray), "edge_ratio": fraction})
        if i % 200 == 0 or i == len(rows):
            print(f"Đã tính đặc trưng ảnh {i}/{len(rows)}", flush=True)
    return pd.DataFrame(records)


def low_edge_weights(fit_rows: pd.DataFrame, multiplier: float = 1.5) -> tuple[dict[str, float], float]:
    """Choose the bottom fake quartile using training rows only, preserving class mass.

    Strict inequality at the cutoff keeps tied/constant features from overweighting
    the entire class. No low-edge group means all weights stay one.
    """
    if not np.isfinite(multiplier) or multiplier < 1:
        raise ValueError("Weight multiplier must be finite and at least one.")
    fake = fit_rows[fit_rows.label == 1]
    if fake.empty or not np.isfinite(fit_rows.edge_ratio).all():
        raise ValueError("Training data needs finite edge ratios and Fake examples.")
    cutoff = float(fake.edge_ratio.quantile(0.25))
    low = (fit_rows.label == 1) & (fit_rows.edge_ratio < cutoff)
    n_low, n_fake = int(low.sum()), len(fake)
    other = (n_fake - multiplier * n_low) / (n_fake - n_low)
    if other <= 0:
        raise ValueError("Multiplier would make the remaining Fake weights nonpositive.")
    values = np.ones(len(fit_rows))
    values[fit_rows.label.to_numpy() == 1] = other
    values[low.to_numpy()] = multiplier
    return dict(zip(fit_rows.file_name, values)), cutoff


def calibration_split(frame: pd.DataFrame, seed: int = 2026) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split each held-out fold by label and gray flag, without crossing train rows."""
    if len(frame) < 4 or not frame.file_name.is_unique:
        raise ValueError("Calibration requires unique names and at least four held-out rows.")
    rng = np.random.default_rng(seed)
    calibration = []
    for _, group in frame.groupby(["label", "is_gray"], sort=True):
        indices = rng.permutation(group.index.to_numpy())
        calibration.extend(indices[:len(indices) // 2])
    mask = frame.index.isin(calibration)
    if not mask.any() or mask.all():
        raise ValueError("Calibration and evaluation must both be nonempty.")
    return frame.loc[mask].copy(), frame.loc[~mask].copy()


def choose_group_thresholds(calibration: pd.DataFrame, probability: str = "rgb_prob") -> dict[int, float]:
    """Grid search on calibration only; missing/single-class groups keep 0.5."""
    chosen = {}
    grid = np.round(np.arange(0.30, 0.701, 0.01), 2)
    for flag in (0, 1):
        group = calibration[calibration.is_gray == flag]
        if group.label.nunique() < 2:
            chosen[flag] = 0.5
            continue
        scores = [(float(classification_report(group.label, group[probability], threshold=float(t))["macro_f1"]),
                   -abs(float(t) - 0.5), -float(t), float(t)) for t in grid]
        chosen[flag] = max(scores)[-1]
    return chosen


def paired_report(frame: pd.DataFrame, a: str, b: str) -> dict:
    """Compare paired predictions without inserting historical expected scores."""
    y = frame.label.to_numpy()
    pa, pb = frame[a].to_numpy() >= 0.5, frame[b].to_numpy() >= 0.5
    ca, cb = pa == y, pb == y
    return {"A": a, "B": b, "n": len(frame),
            "A Macro-F1": classification_report(y, frame[a])["macro_f1"],
            "B Macro-F1": classification_report(y, frame[b])["macro_f1"],
            "A errors": int((~ca).sum()), "B errors": int((~cb).sum()),
            "fixes": int((~ca & cb).sum()), "breaks": int((ca & ~cb).sum()),
            "net errors (B - A)": int((~cb).sum() - (~ca).sum())}
