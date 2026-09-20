"""Live inputs, fresh features, fold training, and submission inference."""
from dataclasses import dataclass, replace, asdict
from pathlib import Path
import hashlib
import json
import uuid
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from .core import PACKAGE, read_csv, read_json, write_json, sha256, split_fold, metric
from .config import Config
from .extractor import NAMES, features


@dataclass
class Runtime:
    data_root: Path
    run_root: Path

    def require_data(self):
        root = Path(self.data_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Missing data directory: {root}")
        return root


def current_code_hashes():
    """Compute SHA256 hashes of all Python source files in src/kmd."""
    src_dir = PACKAGE / 'src/kmd'
    return {p.name: sha256(p) for p in sorted(src_dir.glob('*.py'))}


def validate_run_id(run_id):
    """Validate that run_id is a clean, single directory name and not . or .."""
    if run_id is None:
        return 'lesson_session'
    s = str(run_id).strip()
    if not s or s in {'.', '..'} or Path(s).name != s or '/' in s or '\\' in s:
        raise ValueError(f"Invalid run_id: '{run_id}'. Must be a non-empty single directory name, not '.' or '..'.")
    return s


def load_pairs(root, labeled=True):
    """Load actual manifest, preserving string IDs. Never infer labels from filenames."""
    root = Path(root).expanduser().resolve()
    manifest_path = root / 'pairs.csv'
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing pairs manifest at {manifest_path}")
    frame = read_csv(manifest_path)
    needed = ['pair_id', 'image_0', 'image_1'] + (['fake_position'] if labeled else [])
    if not set(needed) <= set(frame):
        raise ValueError(f"pairs.csv needs columns {needed}")
    frame = frame[needed].copy()
    if frame.empty or frame.isna().any().any() or not frame.pair_id.is_unique:
        raise ValueError('Empty manifest, missing values, or duplicate pair_id.')
    if labeled:
        if not frame.fake_position.isin([0, 1]).all():
            raise ValueError('fake_position must be 0 (left fake) or 1 (right fake).')
        frame.fake_position = frame.fake_position.astype(int)
    for rel in set(frame.image_0) | set(frame.image_1):
        p = Path(rel)
        if p.is_absolute() or '..' in p.parts or not (root / p).resolve().is_relative_to(root):
            raise ValueError(f'Image path escapes data root: {rel}')
    return frame


def prepare_development(root):
    """Read real labels, then keep only the fixed development IDs (never holdout)."""
    all_pairs = load_pairs(root)
    split = read_csv(PACKAGE / 'configs/development_split.csv')
    if len(split) != 800 or not split.pair_id.is_unique or set(split.inner_fold) != {0, 1, 2}:
        raise ValueError('Invalid development split configuration.')
    if not set(split.pair_id) <= set(all_pairs.pair_id):
        raise ValueError('Training data does not contain all development IDs. Use the original Kẻ mạo danh dataset.')
    frame = split.merge(all_pairs, on='pair_id', how='left', validate='one_to_one')
    for fold in range(3):
        tr, va = split_fold(frame, fold)
        if tr.fake_position.nunique() != 2 or va.fake_position.nunique() != 2:
            raise ValueError('Both classes must occur in every training/validation split.')
    return frame.sort_values('pair_id').reset_index(drop=True)


def inventory(frame, root):
    """Validate actual selected pixels and record their hashes; not a cached inventory."""
    from PIL import Image
    root = Path(root)
    records = []
    for rel in sorted(set(frame.image_0) | set(frame.image_1)):
        path = root / rel
        with Image.open(path) as image:
            w, h = image.size
            image.verify()
        if min(w, h) < 224:
            raise ValueError(f'Native-patch tutorial requires images at least 224px: {rel}')
        records.append(dict(relpath=rel, width=w, height=h, bytes=path.stat().st_size, sha256=sha256(path)))
    table = pd.DataFrame(records).sort_values('relpath').reset_index(drop=True)
    fold_by_path = {}
    for row in frame.itertuples():
        for rel in (row.image_0, row.image_1):
            fold_by_path.setdefault(rel, set()).add(int(row.inner_fold))
    hash_folds = {}
    for row in table.itertuples():
        hash_folds.setdefault(row.sha256, set()).update(fold_by_path[row.relpath])
    if any(len(folds) > 1 for folds in hash_folds.values()):
        raise ValueError('Identical image bytes appear across folds.')
    return table


def validate_session_contents(folder, root=None, frame=None):
    """Deeply validate an existing session folder against data, split, and code."""
    folder = Path(folder)
    needed = ['session.json', 'development.csv', 'input_images.csv']
    for fname in needed:
        if not (folder / fname).is_file():
            raise FileNotFoundError(
                f"Session directory '{folder.name}' is incomplete (missing '{fname}'). "
                "Please choose a new RUN_ID."
            )

    sess = read_json(folder / 'session.json')
    split_sha = sha256(PACKAGE / 'configs/development_split.csv')
    if sess.get('split_sha256') != split_sha:
        raise ValueError(
            f"Session '{folder.name}' was created with a different split configuration. "
            "Please choose a new RUN_ID."
        )

    # Check code hashes
    curr_code = current_code_hashes()
    recorded_code = sess.get('code_sha256', {})
    if curr_code != recorded_code:
        raise ValueError(
            f"Session '{folder.name}' code hashes do not match current src/kmd source files. "
            "Please choose a new RUN_ID."
        )

    # Check saved development frame
    saved_dev = read_csv(folder / 'development.csv')
    if frame is not None:
        dev_to_check = frame.sort_values('pair_id').reset_index(drop=True)
        if len(saved_dev) != len(dev_to_check):
            raise ValueError(f"Saved development frame length ({len(saved_dev)}) does not match current ({len(dev_to_check)}).")
        for col in ['pair_id', 'inner_fold', 'image_0', 'image_1', 'fake_position']:
            if col in dev_to_check:
                if col not in saved_dev or not np.array_equal(saved_dev[col].astype(str), dev_to_check[col].astype(str)):
                    raise ValueError(f"Saved development frame mismatch on column '{col}'. Please choose a new RUN_ID.")

    if root is not None:
        data_path = Path(root).expanduser().resolve()
        manifest_file = data_path / 'pairs.csv'
        if not manifest_file.is_file():
            raise FileNotFoundError(f"Missing pairs manifest at {manifest_file}")
        current_pairs_sha = sha256(manifest_file)
        if sess.get('pairs_sha256') != current_pairs_sha:
            raise ValueError(
                f"Dataset pairs.csv hash at '{data_path}' does not match session recorded hash. "
                "Please choose a new RUN_ID."
            )

        # Validate inventory
        if frame is not None:
            curr_inv = inventory(frame, data_path)
            saved_inv = read_csv(folder / 'input_images.csv').sort_values('relpath').reset_index(drop=True)
            if not curr_inv.astype(str).equals(saved_inv[curr_inv.columns].astype(str)):
                raise ValueError(
                    f"Current dataset image byte hashes do not match session input_images.csv. "
                    "Please choose a new RUN_ID."
                )

    return True


def start_session(frame, root, run_id='lesson_session'):
    """Start or validate a shared session directory for learning notebooks."""
    run_id = validate_run_id(run_id)
    root = Path(root).expanduser().resolve()
    manifest_path = root / 'pairs.csv'
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing pairs.csv at {manifest_path}")

    folder = PACKAGE / 'artifacts/models' / run_id
    dev_frame = prepare_development(root)
    if not frame.reset_index(drop=True).equals(dev_frame):
        raise ValueError('Use the canonical frame returned by prepare_development().')
    current_pairs_sha = sha256(manifest_path)
    current_split_sha = sha256(PACKAGE / 'configs/development_split.csv')

    if folder.is_dir():
        validate_session_contents(folder, root=root, frame=dev_frame)
        return folder

    folder.mkdir(parents=True, exist_ok=False)
    images = inventory(dev_frame, root)
    dev_frame.to_csv(folder / 'development.csv', index=False)
    images.to_csv(folder / 'input_images.csv', index=False)
    write_json(folder / 'session.json', {
        'run_id': run_id,
        'pairs_sha256': current_pairs_sha,
        'split_sha256': current_split_sha,
        'code_sha256': current_code_hashes(),
        'blend': {'b2': 0.5, 'native': 0.5},
        'holdout_evaluated': False,
    })
    return folder


def load_session(session_path_or_id, data_root=None):
    """Load an existing session directory, verifying prerequisite artifacts and data hashes."""
    target = Path(session_path_or_id).expanduser()
    if target.name == str(target):
        run_id = validate_run_id(str(target))
        target = PACKAGE / 'artifacts/models' / run_id
    if not target.is_dir():
        raise FileNotFoundError(f"Session '{target}' not found. Run notebook 01 first, keeping the same RUN_ID.")

    curr_frame = None
    if data_root is not None:
        data_path = Path(data_root).expanduser().resolve()
        if (data_path / 'pairs.csv').is_file():
            curr_frame = prepare_development(data_path)

    validate_session_contents(target, root=data_root, frame=curr_frame)
    return target


def extract_pair_features(frame, root, feature_columns=None):
    """Extract statistical features from images, returning pair differences (image_1 - image_0)."""
    paths = sorted(set(frame.image_0) | set(frame.image_1))
    table = pd.DataFrame([features(Path(root) / p) for p in paths], index=paths, columns=NAMES)
    table.index.name = 'relpath'
    if not np.isfinite(table.to_numpy()).all():
        raise ValueError('Non-finite feature values.')
    cols = feature_columns if feature_columns is not None else NAMES
    if not len(cols) or len(set(cols)) != len(cols) or not set(cols) <= set(NAMES):
        raise ValueError(f"Unknown feature column in {cols}")
    sub_table = table[cols]
    x = sub_table.loc[frame.image_1].to_numpy() - sub_table.loc[frame.image_0].to_numpy()
    return x, table


def fit_lr_fold(x, frame, fold):
    train = frame.inner_fold.to_numpy() != fold
    valid = ~train
    # Scale fit on train only. No mean/intercept preserves p(-x) = 1-p(x).
    model = make_pipeline(StandardScaler(with_mean=False),
                          LogisticRegression(C=.1, fit_intercept=False, max_iter=5000, random_state=20260917))
    model.fit(x[train], frame.fake_position.to_numpy()[train])
    return model, model.predict_proba(x[valid])[:, 1]


def fit_lr_cv(frame, root, session, feature_columns=None):
    session = load_session(session, data_root=root)
    dev_frame = prepare_development(root) if root else frame
    selected_cols = list(feature_columns) if feature_columns is not None else NAMES
    x, table = extract_pair_features(dev_frame, root, feature_columns=selected_cols)
    table.to_csv(session / 'features.csv')
    write_json(session / 'lr_feature_names.json', selected_cols)
    prediction = dev_frame.copy()
    prediction['p'] = np.nan
    for fold in range(3):
        model, p = fit_lr_fold(x, dev_frame, fold)
        prediction.loc[dev_frame.inner_fold == fold, 'p'] = p
        joblib.dump(model, session / f'lr_fold{fold}.joblib')
    prediction.to_csv(session / 'lr_oof.csv', index=False)
    return prediction


def validate_fold_checkpoint(folder, c, tr, va, session):
    """Strictly validate an existing fold directory before safe reuse."""
    folder = Path(folder)
    status_file = folder / 'status.json'
    job_file = folder / 'job.json'
    ckpt_file = folder / 'best.pt'
    metrics_file = folder / 'metrics.json'
    dev_file = folder / 'development.csv'

    for f in [status_file, job_file, ckpt_file, metrics_file, dev_file]:
        if not f.is_file():
            return False, f"Missing {f.name}"

    status = read_json(status_file)
    if status.get('status') != 'complete':
        return False, "status is not complete"

    job = read_json(job_file)
    job_config = job.get('config', {})
    expected_config = asdict(c)
    for k, v in expected_config.items():
        job_val = job_config.get(k)
        if k == 'amp':
            # Permit amp mismatch ONLY when requested is True and recorded is False (bfloat16 fallback on older GPUs)
            if job_val != v and not (v is True and job_val is False):
                return False, f"Config mismatch on field 'amp': requested {v}, found {job_val}"
        elif job_val != v:
            return False, f"Config mismatch on field '{k}': requested {v}, found {job_val}"

    if job.get('train_ids') != tr.pair_id.tolist():
        return False, "train_ids mismatch"
    if job.get('validation_ids') != va.pair_id.tolist():
        return False, "validation_ids mismatch"

    if job.get('code_sha256') != current_code_hashes():
        return False, 'Fold code hashes differ from current code'

    # Verify job hash
    contract_copy = dict(job)
    recorded_job_hash = contract_copy.pop('job_hash', None)
    expected_job_hash = hashlib.sha256(json.dumps(contract_copy, sort_keys=True).encode()).hexdigest()
    if recorded_job_hash != expected_job_hash:
        return False, "job.json contract hash mismatch"

    metrics = read_json(metrics_file)
    if metrics.get('status') != 'passed':
        return False, "metrics status not passed"

    if metrics.get('job_hash') != recorded_job_hash:
        return False, 'metrics job hash mismatch'
    recorded_sha = metrics.get('checkpoint_sha256')
    if sha256(ckpt_file) != recorded_sha:
        return False, "checkpoint SHA256 mismatch with metrics.json"

    if metrics.get('reload_probability_max_abs_error', 1.0) >= 1e-6:
        return False, "reload probability error exceeds tolerance"
    if metrics.get('pair_swap_error', 1.0) > 1e-6:
        return False, "pair swap error exceeds tolerance"
    if metrics.get('holdout_evaluated') is not False:
        return False, "holdout was improperly evaluated"

    dev_df = read_csv(dev_file)
    if list(dev_df.pair_id) != list(va.pair_id):
        return False, "development.csv pair_id order/content mismatch"

    for col in ['image_0', 'image_1']:
        if not np.array_equal(dev_df[col], va[col]):
            return False, f"development.csv metadata mismatch on '{col}'"

    if 'y' not in dev_df or not np.array_equal(dev_df['y'], va.fake_position):
        return False, 'development.csv label mismatch'
    if 'fold' not in dev_df or not np.array_equal(dev_df['fold'], va.inner_fold):
        return False, 'development.csv fold mismatch'

    p = dev_df.p.to_numpy(float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        return False, "development.csv contains non-finite or out-of-bounds probabilities"

    return True, metrics


def train_cnn_fold(name, frame, root, session, fold=0):
    """Train a single fold of a CNN configuration, safely reusing matching completed runs."""
    from .training import train_one_fold
    session = load_session(session, data_root=root)
    if name not in {'frozen', 'center60', 'native', 'resampled', 'b2'}:
        raise ValueError(f"Unknown CNN configuration: {name}")
    run_name = f'{name}_fold{fold}'
    folder = session / run_name

    c = Config(**read_json(PACKAGE / f'configs/{name}.json'))
    c = replace(c, fold=fold)
    dev_frame = prepare_development(root) if root else frame
    tr, va = split_fold(dev_frame, fold)

    if folder.is_dir():
        is_valid, info_or_metrics = validate_fold_checkpoint(folder, c, tr, va, session)
        if is_valid:
            return folder, info_or_metrics
        raise RuntimeError(
            f"Phát hiện thư mục fold '{folder.name}' chưa hoàn thành hoặc không khớp ({info_or_metrics}). "
            "Vui lòng chọn một RUN_ID mới."
        )

    runtime = Runtime(Path(root), session)
    folder, result = train_one_fold(c, runtime, run_name=run_name, train_frame=tr, valid_frame=va)
    return folder, result


def train_cnn_cv(name, frame, root, session, folds=(0, 1, 2)):
    """Train multiple folds (or all 3 folds) for a CNN architecture."""
    session = load_session(session, data_root=root)
    dev_frame = prepare_development(root) if root else frame
    if name not in {'frozen', 'center60', 'native', 'resampled', 'b2'}:
        raise ValueError(f"Unknown CNN configuration: {name}")

    for fold in folds:
        train_cnn_fold(name, dev_frame, root, session, fold=fold)

    # Validate and assemble all 3 folds if all exist
    c = Config(**read_json(PACKAGE / f'configs/{name}.json'))
    all_done = True
    records, predictions = [], []
    for fold in range(3):
        folder = session / f'{name}_fold{fold}'
        tr, va = split_fold(dev_frame, fold)
        if not folder.is_dir():
            all_done = False
            break
        is_valid, metrics = validate_fold_checkpoint(folder, replace(c, fold=fold), tr, va, session)
        if not is_valid:
            raise ValueError(f'Invalid existing fold {folder.name}: {metrics}. Choose a new RUN_ID.')
        p = read_csv(folder / 'development.csv').rename(columns={'y': 'fake_position', 'fold': 'inner_fold'})
        predictions.append(p)
        records.append({'fold': fold, 'checkpoint': str((folder / 'best.pt').relative_to(session)),
                        'sha256': metrics['checkpoint_sha256']})

    if all_done:
        combined = pd.concat(predictions, ignore_index=True).sort_values('pair_id').reset_index(drop=True)
        if len(combined) != 800 or not combined.pair_id.is_unique or set(combined.pair_id) != set(dev_frame.pair_id):
            raise ValueError(f"Assembled OOF coverage for {name} is incomplete or corrupted.")
        combined.to_csv(session / f'{name}_oof.csv', index=False)
        write_json(session / f'{name}_models.json', records)
        return combined
    return None


def complete_method_cv(method, frame, root, session):
    """Complete 3-fold cross validation for the user-selected method."""
    session = load_session(session, data_root=root)
    dev_frame = prepare_development(root) if root else frame
    if method == 'lr':
        return fit_lr_cv(dev_frame, root, session)
    if method in {'center60', 'native', 'b2', 'frozen', 'resampled'}:
        return train_cnn_cv(method, dev_frame, root, session, folds=(0, 1, 2))
    if method == 'blend':
        train_cnn_cv('b2', dev_frame, root, session, folds=(0, 1, 2))
        train_cnn_cv('native', dev_frame, root, session, folds=(0, 1, 2))
        b2_oof = read_csv(session / 'b2_oof.csv')
        nat_oof = read_csv(session / 'native_oof.csv')
        blend_df = blend(b2_oof, nat_oof, dev_frame)
        blend_df.to_csv(session / 'blend_oof.csv', index=False)
        return blend_df
    raise ValueError(f"Unknown method '{method}'. Supported: 'lr', 'center60', 'native', 'b2', 'blend'.")


def aligned_predictions(predictions, frame):
    if not predictions.pair_id.is_unique or set(predictions.pair_id) != set(frame.pair_id):
        raise ValueError('Predictions must cover exactly the requested pair IDs.')
    result = predictions.set_index('pair_id').loc[frame.pair_id].reset_index()
    for key in ['image_0', 'image_1', 'fake_position', 'inner_fold']:
        if key in frame and (key not in result or not np.array_equal(result[key].astype(str), frame[key].astype(str))):
            raise ValueError(f'Prediction metadata mismatch: {key}')
    p = result.p.to_numpy(float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Invalid probabilities.')
    return result


def compare_oof(session, names):
    session = Path(session)
    frame = read_csv(session / 'development.csv')
    predictions, rows = {}, []
    for name in names:
        oof_path = session / f'{name}_oof.csv'
        if not oof_path.is_file():
            continue
        pred = aligned_predictions(read_csv(oof_path), frame)
        predictions[name] = pred
        rows.append({'model': name, **metric(frame.fake_position, pred.p)})
    return pd.DataFrame(rows), predictions


def blend(a, b, frame):
    a, b = aligned_predictions(a, frame), aligned_predictions(b, frame)
    result = frame.copy()
    result['p'] = .5 * a.p.to_numpy() + .5 * b.p.to_numpy()
    return result


def infer_lr(frame, root, session, feature_columns=None):
    session = load_session(session, data_root=root)
    recorded_columns = read_json(session / 'lr_feature_names.json')
    if feature_columns is not None and list(feature_columns) != recorded_columns:
        raise ValueError('Inference feature columns must match fitted LR columns.')
    feature_columns = recorded_columns
    x, _ = extract_pair_features(frame, root, feature_columns=feature_columns)
    result = frame.copy()
    result['p'] = np.mean([joblib.load(session / f'lr_fold{f}.joblib').predict_proba(x)[:, 1]
                           for f in range(3)], axis=0)
    return result


def infer_cnn(name, frame, root, session, device='cuda'):
    import torch
    from .models import model_for, predict_pairs
    session = load_session(session, data_root=root)
    records_file = session / f'{name}_models.json'
    if not records_file.is_file():
        raise FileNotFoundError(
            f"Model index '{records_file.name}' not found. "
            f"3-fold CV training for '{name}' must be completed before inference."
        )
    records = read_json(records_file)
    if sorted(r['fold'] for r in records) != [0, 1, 2]:
        raise ValueError('Need exactly three fold checkpoints for CV inference.')
    probabilities = []
    for record in records:
        checkpoint = session / record['checkpoint']
        if sha256(checkpoint) != record['sha256']:
            raise ValueError('Checkpoint hash changed.')
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        c = Config(**state['config'])
        model, _ = model_for(c, device=device, pretrained=False)
        model.load_state_dict(state['state_dict'], strict=True)
        p = predict_pairs(model, frame, c, Path(root), state.get('norm'), device=device)
        probabilities.append(aligned_predictions(p, frame).p.to_numpy())
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()
    result = frame.copy()
    result['p'] = np.mean(probabilities, axis=0)
    return result


def check_test_separation(test, test_root, session):
    """Prevent exporting disguised development predictions as unseen test results."""
    session = Path(session)
    hashes = set(pd.read_csv(session / 'input_images.csv').sha256)
    for rel in sorted(set(test.image_0) | set(test.image_1)):
        if sha256(Path(test_root) / rel) in hashes:
            raise ValueError(f'Test image duplicates development bytes: {rel}')
