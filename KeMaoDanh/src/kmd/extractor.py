"""32-feature statistical extractor with readable, named modular steps.

Semantics, ordering, and numerical definitions match the development experiments.
"""
from pathlib import Path
import numpy as np
import cv2

NAMES = [
    'log_bytes',
    'mean_r', 'mean_g', 'mean_b',
    'std_r', 'std_g', 'std_b',
    'q10_r', 'q10_g', 'q10_b',
    'q90_r', 'q90_g', 'q90_b',
    'sat_mean', 'sat_std',
    'val_mean', 'val_std',
    'gray_mean', 'gray_std',
    'entropy',
    'grad_mean', 'grad_std',
    'lap_std', 'lap_abs',
    'residual_std', 'residual_abs',
    'block_x', 'block_y',
    'center_mean', 'center_std',
    'border_mean', 'border_std',
]

# Report-defined feature groups (Report pp. 8, 10-11)
# Group 0: File size [0]
# Group 1: Color / Intensity [1:19]
# Group 2: Texture / Residual [19:28]
# Group 3: Spatial Center / Border [28:32]
LR_GROUPS = {
    'filesize': NAMES[0:1],
    'color': NAMES[1:19],
    'texture': NAMES[19:28],
    'center_border': NAMES[28:32],
}

# The 9 ablation probes tested in Figure 7 of the report
LR_ABLATIONS = {
    'full32': NAMES,
    'filesize_only': NAMES[0:1],
    'color_only': NAMES[1:19],
    'texture_only': NAMES[19:28],
    'center_border_only': NAMES[28:32],
    'drop_filesize': NAMES[1:32],
    'drop_color': NAMES[0:1] + NAMES[19:32],
    'drop_texture': NAMES[0:19] + NAMES[28:32],
    'drop_center_border': NAMES[0:28],
}

LR_ABLATION_ALIASES = {
    'full': 'full32',
    'lr': 'full32',
    'no_filesize': 'drop_filesize',
    'no_spatial': 'drop_center_border',
    'all': 'full32',
    'filesize': 'filesize_only',
    'color': 'color_only',
    'texture': 'texture_only',
    'center_border': 'center_border_only',
    'spatial': 'center_border_only',
    'spatial_only': 'center_border_only',
    'drop_spatial': 'drop_center_border',
}


def resolve_lr_ablation_name(variant: str) -> str:
    """Resolve shorthand alias to canonical LR ablation name."""
    v = str(variant).strip()
    return LR_ABLATION_ALIASES.get(v, v)


def get_lr_features(variant: str) -> list[str]:
    """Retrieve the exact list of feature names for an LR ablation probe."""
    canonical = resolve_lr_ablation_name(variant)
    if canonical not in LR_ABLATIONS:
        valid = ', '.join(sorted(LR_ABLATIONS.keys()))
        raise ValueError(f"Unknown LR ablation '{variant}'. Available: {valid}")
    return list(LR_ABLATIONS[canonical])


def list_lr_ablations() -> list[str]:
    """Return sorted list of all 9 canonical LR ablation probe names."""
    return sorted(LR_ABLATIONS.keys())


def extract_file_size_features(path: Path) -> list[float]:
    """Feature 0: log-transformed file size in bytes."""
    size_bytes = Path(path).stat().st_size
    return [float(np.log1p(size_bytes))]


def extract_color_stats(rgb: np.ndarray) -> list[float]:
    """Features 1-12: Mean, std, 10th percentile, 90th percentile for RGB channels."""
    mean_rgb = rgb.mean(axis=(0, 1)).tolist()
    std_rgb = rgb.std(axis=(0, 1)).tolist()
    q10_rgb = np.quantile(rgb, 0.1, axis=(0, 1)).tolist()
    q90_rgb = np.quantile(rgb, 0.9, axis=(0, 1)).tolist()
    return mean_rgb + std_rgb + q10_rgb + q90_rgb


def extract_hsv_stats(hsv: np.ndarray) -> list[float]:
    """Features 13-16: Mean and std for Saturation and Value channels."""
    sat = hsv[..., 1]
    val = hsv[..., 2]
    return [float(sat.mean()), float(sat.std()), float(val.mean()), float(val.std())]


def extract_gray_stats(gray: np.ndarray) -> list[float]:
    """Features 17-19: Grayscale mean, standard deviation, and intensity entropy."""
    g_mean = float(gray.mean())
    g_std = float(gray.std())
    hist, _ = np.histogram(gray, bins=64, range=(0, 1))
    hist = hist.astype(float)
    hist_sum = hist.sum()
    if hist_sum > 0:
        hist /= hist_sum
        nonzero = hist[hist > 0]
        entropy = float(-(nonzero * np.log2(nonzero)).sum())
    else:
        entropy = 0.0
    return [g_mean, g_std, entropy]


def extract_gradient_stats(gray: np.ndarray) -> list[float]:
    """Features 20-23: Sobel gradient magnitude and Laplacian 2nd derivative stats."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    grad_mean = float(grad_mag.mean())
    grad_std = float(grad_mag.std())

    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lap_std = float(lap.std())
    lap_abs = float(np.abs(lap).mean())
    return [grad_mean, grad_std, lap_std, lap_abs]


def extract_residual_stats(gray: np.ndarray) -> list[float]:
    """Features 24-25: High-frequency residual (image minus Gaussian blur)."""
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.2, sigmaY=1.2)
    residual = gray - blurred
    return [float(residual.std()), float(np.abs(residual).mean())]


def extract_blockiness_stats(gray: np.ndarray) -> list[float]:
    """Features 26-27: Mean absolute 8x8 JPEG grid boundary differences."""
    diff_x = np.abs(np.diff(gray, axis=1)[:, 7::8]).mean()
    diff_y = np.abs(np.diff(gray, axis=0)[7::8, :]).mean()
    return [float(diff_x), float(diff_y)]


def extract_center_border_stats(gray: np.ndarray) -> list[float]:
    """Features 28-31: Intensity distribution in central 60% box vs outer border."""
    h, w = gray.shape
    r_start, r_end = int(0.2 * h), int(0.8 * h)
    c_start, c_end = int(0.2 * w), int(0.8 * w)

    center = gray[r_start:r_end, c_start:c_end]
    mask = np.ones_like(gray, dtype=bool)
    mask[r_start:r_end, c_start:c_end] = False
    border = gray[mask]

    return [
        float(center.mean()), float(center.std()),
        float(border.mean()), float(border.std()),
    ]


def features(path: Path | str) -> np.ndarray:
    """Extract full 32-feature vector for an image file."""
    path = Path(path)
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"Cannot load image from {path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0

    values: list[float] = []
    values.extend(extract_file_size_features(path))
    values.extend(extract_color_stats(rgb))
    values.extend(extract_hsv_stats(hsv))
    values.extend(extract_gray_stats(gray))
    values.extend(extract_gradient_stats(gray))
    values.extend(extract_residual_stats(gray))
    values.extend(extract_blockiness_stats(gray))
    values.extend(extract_center_border_stats(gray))

    assert len(values) == 32, f"Expected 32 features, got {len(values)}"
    return np.array(values, dtype=np.float64)
