"""Small implementations embedded as editable code cells in the lessons."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torchvision import models
from .models import attach_optimizer_groups


def image_census(rows):
    """Read every image; record size, exact gray flag and duplicate content."""
    records = []
    for row in rows.itertuples():
        path = Path(row.path)
        with Image.open(path) as image:
            original_mode = image.mode
            rgb = np.asarray(image.convert("RGB"))
        delta = rgb.max(axis=2).astype(float) - rgb.min(axis=2).astype(float)
        records.append({"file_name": row.file_name, "label": int(row.label),
                        "path": str(path), "width": rgb.shape[1], "height": rgb.shape[0],
                        "mode": original_mode, "size_kib": path.stat().st_size / 1024,
                        "is_gray": bool((delta == 0).all()),
                        "near_gray": bool(delta.mean() <= 1),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    census = pd.DataFrame(records)
    if census.empty or not census.file_name.is_unique:
        raise ValueError("Image census requires nonempty rows with unique file names.")
    return census


def choose_size_rule(rows):
    """Select a JPEG-size threshold and direction on the supplied training rows."""
    values = np.sort(rows.size_kib.unique())
    candidates = np.r_[values[0] - 1, (values[:-1] + values[1:]) / 2, values[-1] + 1]
    best = None
    for fake_is_small in (True, False):
        for threshold in candidates:
            prediction = (rows.size_kib < threshold) if fake_is_small else (rows.size_kib >= threshold)
            accuracy = float((prediction.astype(int) == rows.label).mean())
            if best is None or accuracy > best["accuracy"]:
                best = {"threshold_kib": float(threshold), "fake_is_small": fake_is_small,
                        "accuracy": accuracy}
    return best


class BaselineCNN2(nn.Module):
    """The two Conv-BatchNorm-ReLU-Pool blocks printed in Reading III.1."""
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16),
            nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, num_classes))
        attach_optimizer_groups(self, self.classifier)

    def forward(self, images):
        return self.classifier(self.features(images))


def build_resnet34(pretrained=True, freeze_backbone=False):
    weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet34(weights=weights)
    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 2)
    model._freeze_backbone = freeze_backbone
    return attach_optimizer_groups(model, model.fc)


def crossfit_threshold(frame, probability="p_mean", grid=None):
    """Fit a threshold on other held-out folds, then score only the excluded fold."""
    from sklearn.metrics import f1_score
    if frame.fold.nunique() < 2 or not frame.file_name.is_unique:
        raise ValueError("Cross-fit needs at least two folds and unique file names.")
    grid = np.asarray(grid if grid is not None else np.linspace(.1, .9, 161))
    parts, choices = [], []
    for fold in sorted(frame.fold.unique()):
        calibration = frame[frame.fold != fold]
        evaluation = frame[frame.fold == fold].copy()
        scores = [f1_score(calibration.label, calibration[probability] >= t,
                           labels=[0, 1], average="macro", zero_division=0) for t in grid]
        # Deterministic tie break: prefer the threshold closest to 0.5.
        best = np.flatnonzero(np.isclose(scores, np.max(scores), rtol=0, atol=1e-12))
        threshold = float(grid[best[np.argmin(abs(grid[best] - .5))]])
        evaluation["chosen_threshold"] = threshold
        evaluation["crossfit_prediction"] = (evaluation[probability] >= threshold).astype(int)
        parts.append(evaluation)
        choices.append({"fold": int(fold), "threshold": threshold,
                        "calibration_n": len(calibration), "evaluation_n": len(evaluation)})
    return pd.concat(parts, ignore_index=True), pd.DataFrame(choices)


def crossfit_stack(frame, columns=("p_rgb", "p_hp")):
    """Meta-CV on a fixed OOF feature matrix; this is not nested base-model CV."""
    from sklearn.linear_model import LogisticRegression
    if frame.fold.nunique() < 2 or not frame.file_name.is_unique:
        raise ValueError("Meta-CV needs at least two folds and unique file names.")
    parts = []
    for fold in sorted(frame.fold.unique()):
        fit = frame[frame.fold != fold]
        heldout = frame[frame.fold == fold].copy()
        meta = LogisticRegression(C=1.0, max_iter=1000, random_state=2026)
        meta.fit(fit[list(columns)], fit.label)
        heldout["p_stack"] = meta.predict_proba(heldout[list(columns)])[:, 1]
        parts.append(heldout)
    return pd.concat(parts, ignore_index=True)


def crossfit_logit_stack(frame, columns):
    """Fit standardized-logit LR per held-out fold, following the saved Stack6 report."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    if frame.fold.nunique() < 2 or not frame.file_name.is_unique:
        raise ValueError("Meta-CV needs multiple folds and unique file names.")
    probability = frame[list(columns)].to_numpy(dtype=float)
    if not np.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any():
        raise ValueError("Stacking inputs must be finite probabilities.")
    probability = np.clip(probability, 1e-6, 1-1e-6)
    logits = np.log(probability / (1-probability))
    output = frame.copy()
    output["p_stack_refit"] = np.nan
    for fold in sorted(frame.fold.unique()):
        heldout = (frame.fold == fold).to_numpy()
        # Both scaler and LR are fit on calibration folds only.
        meta = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=2026))
        meta.fit(logits[~heldout], frame.label.to_numpy()[~heldout])
        output.loc[heldout, "p_stack_refit"] = meta.predict_proba(logits[heldout])[:, 1]
    return output
