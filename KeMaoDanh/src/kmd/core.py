"""Metrics, pair scores and file utilities; no precomputed results."""
from pathlib import Path
import hashlib
import json
import platform
import time
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, log_loss
PACKAGE = Path(__file__).resolve().parents[2]

def read_json(path):
    return json.loads(Path(path).read_text())

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def read_csv(path):
    return pd.read_csv(path, dtype={"pair_id": str})

def split_fold(frame, fold):
    if fold not in [0, 1, 2]:
        raise ValueError("fold must be 0, 1, or 2")
    train = frame.loc[frame.inner_fold != fold].reset_index(drop=True)
    valid = frame.loc[frame.inner_fold == fold].reset_index(drop=True)
    tr_images = set(train.image_0) | set(train.image_1)
    va_images = set(valid.image_0) | set(valid.image_1)
    if set(train.pair_id) & set(valid.pair_id) or tr_images & va_images:
        raise ValueError("Pair/image leakage between training and validation.")
    return train, valid

def metric(y, p):
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=float)
    if y.shape != p.shape or y.ndim != 1 or not len(y):
        raise ValueError("Labels/probabilities must be nonempty aligned vectors.")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid binary labels/probabilities.")
    return dict(n=len(y), macro_f1=float(f1_score(y, p >= .5, average="macro", labels=[0, 1], zero_division=0)),
                accuracy=float(np.mean((p >= .5) == y)), auc=float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
                log_loss=float(log_loss(y, np.clip(p, 1e-7, 1-1e-7), labels=[0, 1])), errors=int(np.sum((p >= .5) != y)))

def pair_table(frame, scores):
    scores = np.asarray(scores)
    if scores.shape != (len(frame), 2) or not np.isfinite(scores).all():
        raise ValueError("Expected two finite image logits per pair.")
    result = frame[["pair_id", "image_0", "image_1"]].copy()
    if "fake_position" in frame:
        result["y"] = frame.fake_position.astype(int).to_numpy()
    result["logit_0"], result["logit_1"] = scores[:, 0], scores[:, 1]
    result["p"] = 1 / (1 + np.exp(-np.clip(scores[:, 1] - scores[:, 0], -60, 60)))
    result["prediction"] = (result.p >= .5).astype(int)
    if "inner_fold" in frame:
        result["fold"] = frame.inner_fold.to_numpy()
    return result

def export_submission(predictions, path):
    if not predictions.pair_id.is_unique or not predictions.pair_id.map(lambda x: isinstance(x, str)).all():
        raise ValueError("pair_id must be unique strings, preserving leading zeroes.")
    p = predictions.p.to_numpy(dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid prediction probabilities.")
    result = pd.DataFrame({"pair_id": predictions.pair_id, "fake_position": (p >= .5).astype(int)})
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); result.to_csv(path, index=False)
    return result

def environment():
    import importlib.metadata as md
    packages = {}
    for name in ["numpy", "pandas", "scikit-learn", "opencv-python", "Pillow", "torch", "torchvision", "nbclient", "nbconvert", "nbformat", "ipykernel"]:
        try: packages[name] = md.version(name)
        except md.PackageNotFoundError: packages[name] = None
    return dict(host=platform.node(), python=platform.python_version(), platform=platform.platform(), packages=packages, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
