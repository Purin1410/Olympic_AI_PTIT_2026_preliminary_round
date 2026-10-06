"""Notebook figures for image previews, spectra, curves, and prediction changes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .data import decode_rgb
from .forensics import spectrum


def _array(image: Any) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        value = image.detach().float().cpu().numpy()
    else:
        value = np.asarray(image)
    if value.ndim == 3 and value.shape[0] in (1, 3):
        value = value.transpose(1, 2, 0)
    return np.clip(value, 0, 1) if np.issubdtype(value.dtype, np.floating) else value


def show_samples(rows: pd.DataFrame, label_col: str = "label", n: int = 6) -> plt.Figure:
    """Display a small deterministic set of manifest images."""
    selected = rows.head(n)
    columns = min(3, max(1, len(selected)))
    figure, axes = plt.subplots(max(1, int(np.ceil(len(selected) / columns))), columns,
                                figsize=(4 * columns, 4 * max(1, int(np.ceil(len(selected) / columns)))), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, row in zip(axes.flat, selected.itertuples(index=False)):
        axis.imshow(_array(decode_rgb(row.path)))
        label = getattr(row, label_col, "") if label_col else ""
        axis.set_title(f"{row.file_name} | {label}")
    figure.tight_layout()
    return figure


def show_views(original: Any, views: Mapping[str, Any]) -> plt.Figure:
    """Display one source image and named views in a shared figure."""
    items = [("RGB", original), *views.items()]
    figure, axes = plt.subplots(1, len(items), figsize=(4 * len(items), 4), squeeze=False)
    for axis, (title, image) in zip(axes[0], items):
        axis.imshow(_array(image))
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    return figure


def plot_spectra(first: Any, second: Any, names: tuple[str, str] = ("Native", "Resampled")) -> plt.Figure:
    """Plot two log spectra with one shared color scale and a symmetric difference."""
    p1, log1 = spectrum(first)
    p2, log2 = spectrum(second)
    if log1.shape != log2.shape:
        raise ValueError("Spectrum inputs must have matching height and width.")
    lower, upper = float(min(log1.min(), log2.min())), float(max(log1.max(), log2.max()))
    delta = log2 - log1
    extent = float(np.abs(delta).max()) or 1.0
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis, values, title in zip(axes[:2], (log1, log2), names):
        image = axis.imshow(values, cmap="magma", vmin=lower, vmax=upper)
        axis.set_title(f"{title} spectrum (dB)")
        figure.colorbar(image, ax=axis, fraction=0.046)
    image = axes[2].imshow(delta, cmap="coolwarm", vmin=-extent, vmax=extent)
    axes[2].set_title("Second − first (dB)")
    figure.colorbar(image, ax=axes[2], fraction=0.046)
    figure.tight_layout()
    return figure


def show_learning_curves(curves: pd.DataFrame) -> plt.Figure:
    """Plot training/validation loss and validation Macro-F1 over epochs."""
    required = {"epoch", "train_loss", "val_loss", "val_macro_f1"}
    if not required.issubset(curves.columns):
        raise ValueError(f"Learning curves are missing: {sorted(required - set(curves.columns))}")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(curves.epoch, curves.train_loss, label="Train")
    axes[0].plot(curves.epoch, curves.val_loss, label="Validation")
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy", title="Loss")
    axes[1].plot(curves.epoch, curves.val_macro_f1, marker="o", label="Validation")
    axes[1].set(xlabel="Epoch", ylabel="Macro-F1", ylim=(0, 1), title="Validation score")
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.2)
    figure.tight_layout()
    return figure


def show_prediction_examples(rows: pd.DataFrame, title: str, max_examples: int = 3) -> plt.Figure | None:
    """Show images with true labels and branch/mean probabilities; handle empty groups."""
    selected = rows.head(max_examples)
    if selected.empty:
        print(f"Không có ảnh thuộc nhóm: {title}.")
        return None
    figure, axes = plt.subplots(1, len(selected), figsize=(4 * len(selected), 4), squeeze=False)
    for axis, row in zip(axes[0], selected.itertuples(index=False)):
        path = getattr(row, "path", None)
        if path is not None and Path(path).is_file():
            axis.imshow(_array(decode_rgb(path)))
        else:
            axis.text(0.5, 0.5, "Ảnh tham khảo chưa có trong runtime", ha="center", va="center")
            axis.set_xlim(0, 1)
            axis.set_ylim(0, 1)
        axis.set_title(f"{row.file_name}\ny={row.label} | RGB={row.p_rgb:.2f} HP={row.p_hp:.2f} mean={row.p_mean:.2f}")
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    return figure


def error_montage(
    rows: pd.DataFrame,
    rgb_correct: pd.Series,
    mean_correct: pd.Series,
    max_examples: int = 3,
) -> tuple[plt.Figure | None, plt.Figure | None]:
    """Show RGB-to-mean fixes and breaks as separate groups."""
    if len(rows) != len(rgb_correct) or len(rows) != len(mean_correct):
        raise ValueError("Error masks must have one entry for each prediction row.")
    return (
        show_prediction_examples(rows.loc[~rgb_correct & mean_correct], "RGB sai → mean đúng", max_examples),
        show_prediction_examples(rows.loc[rgb_correct & ~mean_correct], "RGB đúng → mean sai", max_examples),
    )
