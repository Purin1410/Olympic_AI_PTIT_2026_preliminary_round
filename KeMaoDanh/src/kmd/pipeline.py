"""Live inputs, fresh features, fold training, and submission inference."""
from dataclasses import dataclass, replace, asdict
from pathlib import Path
from typing import Optional
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
from .extractor import NAMES, features, resolve_lr_ablation_name, get_lr_features, list_lr_ablations
from .presets import get_preset_config, is_registered_preset, canonical_preset_name
from .subsets import sample_train_subset, sample_smoke_train


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


def fit_lr_cv(frame, root, session, feature_columns=None, variant='full32', precomputed_features=None):
    session = load_session(session, data_root=root)
    dev_frame = prepare_development(root) if root else frame
    if len(dev_frame) != 800 or not dev_frame['pair_id'].is_unique:
        raise ValueError("Development frame must contain exactly 800 unique pairs.")

    if not variant or "/" in str(variant) or "\\" in str(variant) or ".." in str(variant):
        raise ValueError(f"Invalid variant name '{variant}' (path traversal detected)")
    v_name = resolve_lr_ablation_name(variant)
    if v_name not in list_lr_ablations():
        raise ValueError(f"Unknown LR variant '{variant}'. Available: {list_lr_ablations()}")

    expected_cols = get_lr_features(v_name)
    if feature_columns is not None:
        selected_cols = list(feature_columns)
        if selected_cols != expected_cols:
            raise ValueError(f"Feature columns do not agree with requested ablation variant '{v_name}': expected {expected_cols}, got {selected_cols}")
    else:
        selected_cols = list(expected_cols)

    oof_target = session / f'lr_{v_name}_oof.csv'
    job_target = session / f'lr_{v_name}_job.json'
    feat_target = session / f'lr_{v_name}_feature_names.json'
    fold_model_files = [session / f'lr_{v_name}_fold{f}.joblib' for f in range(3)]

    # Validate exact metadata and hashes on reuse, fail with new RUN_ID on mismatch/partial
    existing_artifacts = [p for p in [oof_target, job_target, feat_target] + fold_model_files if p.is_file()]
    if existing_artifacts:
        all_exist = (
            job_target.is_file()
            and oof_target.is_file()
            and feat_target.is_file()
            and all(p.is_file() for p in fold_model_files)
        )
        if not all_exist:
            missing = [p.name for p in [job_target, oof_target, feat_target] + fold_model_files if not p.is_file()]
            raise ValueError(
                f"Partial existing LR artifacts found for '{v_name}' in {session}: missing {missing}. "
                "Mismatched or partial runs cannot be reused; choose a new RUN_ID."
            )

        try:
            job = read_json(job_target)
        except Exception as e:
            raise ValueError(f"Corrupt {job_target.name}: {e}. Choose a new RUN_ID.")

        if job.get('variant') != v_name:
            raise ValueError(f"Variant mismatch in {job_target.name}: expected {v_name}, found {job.get('variant')}. Choose a new RUN_ID.")
        if job.get('feature_columns') != selected_cols:
            raise ValueError(f"Feature columns mismatch in {job_target.name}. Choose a new RUN_ID.")
        if job.get('code_sha256') != current_code_hashes():
            raise ValueError(f"Code SHA256 mismatch in {job_target.name}. Code has changed; choose a new RUN_ID.")
        if job.get('pair_ids') != dev_frame.pair_id.tolist():
            raise ValueError(f"Pair IDs mismatch in {job_target.name}. Choose a new RUN_ID.")
        if job.get('folds') != [0, 1, 2]:
            raise ValueError(f"Folds mismatch in {job_target.name}. Choose a new RUN_ID.")

        recorded_models = job.get('model_sha256', {})
        for f, f_path in enumerate(fold_model_files):
            expected_sha = recorded_models.get(f'fold{f}')
            if not expected_sha or sha256(f_path) != expected_sha:
                raise ValueError(f"Model SHA256 mismatch for fold {f} in {job_target.name}. Choose a new RUN_ID.")

        if sha256(oof_target) != job.get('oof_sha256'):
            raise ValueError(f"OOF CSV SHA256 mismatch with {job_target.name}. Choose a new RUN_ID.")

        saved_cols = read_json(feat_target)
        if saved_cols != selected_cols:
            raise ValueError(f"Saved feature names mismatch in {feat_target.name}. Choose a new RUN_ID.")

        saved_oof = read_csv(oof_target)
        if len(saved_oof) != len(dev_frame) or not saved_oof.pair_id.is_unique or set(saved_oof.pair_id) != set(dev_frame.pair_id):
            raise ValueError(f"OOF predictions corrupted or incomplete in {oof_target.name}. Choose a new RUN_ID.")

        p_vals = saved_oof.p.to_numpy(float)
        if not np.isfinite(p_vals).all() or ((p_vals < 0) | (p_vals > 1)).any():
            raise ValueError(f"OOF predictions contain non-finite or out-of-bounds probabilities in {oof_target.name}. Choose a new RUN_ID.")

        contract_copy = dict(job)
        stored_hash = contract_copy.pop('job_hash', None)
        if stored_hash != hashlib.sha256(json.dumps(contract_copy, sort_keys=True).encode()).hexdigest():
            raise ValueError('LR contract hash changed. Choose a new RUN_ID.')
        for f in range(3):
            tr, va = split_fold(dev_frame, f)
            if job['fold_train_ids'].get(str(f)) != tr.pair_id.tolist() or job['fold_val_ids'].get(str(f)) != va.pair_id.tolist():
                raise ValueError('LR fold IDs changed. Choose a new RUN_ID.')
        return aligned_predictions(saved_oof, dev_frame)

    # Extract or use precomputed features with finite validation
    if precomputed_features is not None:
        table = precomputed_features
        all_images = list(set(dev_frame.image_0) | set(dev_frame.image_1))
        for p in all_images:
            if p not in table.index:
                raise ValueError(f"Precomputed features missing image '{p}'")
        for col in selected_cols:
            if col not in table.columns:
                raise ValueError(f"Precomputed features missing column '{col}'")
        feats_matrix = table.loc[all_images, selected_cols].to_numpy(dtype=float)
        if not np.isfinite(feats_matrix).all():
            raise ValueError('Non-finite feature values in precomputed features.')
        if not table.index.is_unique:
            raise ValueError('Precomputed image feature index must be unique.')
        sub_table = table[selected_cols]
        x = sub_table.loc[dev_frame.image_1].to_numpy() - sub_table.loc[dev_frame.image_0].to_numpy()
    else:
        x, table = extract_pair_features(dev_frame, root, feature_columns=selected_cols)
        if not (session / 'features.csv').is_file():
            table.to_csv(session / 'features.csv')

    # Save variant-specific artifacts to prevent cross-ablation collisions
    write_json(feat_target, selected_cols)
    prediction = dev_frame.copy()
    prediction['p'] = np.nan
    model_shas = {}
    for fold in range(3):
        model, p = fit_lr_fold(x, dev_frame, fold)
        prediction.loc[dev_frame.inner_fold == fold, 'p'] = p
        model_path = fold_model_files[fold]
        joblib.dump(model, model_path)
        model_shas[f'fold{fold}'] = sha256(model_path)

    p_all = prediction['p'].to_numpy(float)
    if not np.isfinite(p_all).all() or ((p_all < 0) | (p_all > 1)).any():
        raise ValueError("Fitted LR produced non-finite or out-of-bounds probabilities.")

    prediction.to_csv(oof_target, index=False)
    oof_sha = sha256(oof_target)

    # Persist job contract with code hashes, IDs/folds/columns and SHA model+OOF
    contract = {
        "id": f"lr_{v_name}",
        "variant": v_name,
        "feature_columns": selected_cols,
        "code_sha256": current_code_hashes(),
        "pair_ids": dev_frame.pair_id.tolist(),
        "folds": [0, 1, 2],
        "fold_train_ids": {str(f): split_fold(dev_frame, f)[0].pair_id.tolist() for f in range(3)},
        "fold_val_ids": {str(f): split_fold(dev_frame, f)[1].pair_id.tolist() for f in range(3)},
        "model_files": [f"lr_{v_name}_fold{f}.joblib" for f in range(3)],
        "model_sha256": model_shas,
        "oof_file": f"lr_{v_name}_oof.csv",
        "oof_sha256": oof_sha,
    }
    contract_copy = dict(contract)
    contract["job_hash"] = hashlib.sha256(json.dumps(contract_copy, sort_keys=True).encode()).hexdigest()
    write_json(job_target, contract)

    # ONLY full32 with exact NAMES may write base aliases
    if v_name == 'full32' and selected_cols == NAMES:
        write_json(session / 'lr_feature_names.json', selected_cols)
        base_model_shas = {}
        for fold in range(3):
            base_m = session / f'lr_fold{fold}.joblib'
            joblib.dump(joblib.load(fold_model_files[fold]), base_m)
            base_model_shas[f'fold{fold}'] = sha256(base_m)
        prediction.to_csv(session / 'lr_oof.csv', index=False)
        base_contract = dict(contract)
        base_contract["id"] = "lr"
        base_contract["model_files"] = [f"lr_fold{f}.joblib" for f in range(3)]
        base_contract["model_sha256"] = base_model_shas
        base_contract["oof_file"] = "lr_oof.csv"
        base_contract["oof_sha256"] = sha256(session / 'lr_oof.csv')
        base_copy = dict(base_contract)
        base_copy.pop('job_hash', None)
        base_contract["job_hash"] = hashlib.sha256(json.dumps(base_copy, sort_keys=True).encode()).hexdigest()
        write_json(session / 'lr_job.json', base_contract)

    return prediction


def fold_run_name(name: str, fold: int, seed: int = 20260917, smoke: bool = False) -> str:
    """Generate canonical immutable directory name for a preset fold run."""
    canonical = canonical_preset_name(name)
    prefix = f"{canonical}_smoke" if smoke else canonical
    return f"{prefix}_s{seed}_f{fold}"


def find_fold_folder(session: Path, name: str, fold: int, seed: int = 20260917, smoke: bool = False) -> Optional[Path]:
    """Find existing fold directory, checking canonical name and backwards-compatible legacy name."""
    session = Path(session)
    canonical = canonical_preset_name(name)
    canonical_folder = session / fold_run_name(canonical, fold, seed, smoke=smoke)
    if canonical_folder.is_dir():
        return canonical_folder
    if not smoke and seed == 20260917:
        legacy = session / f"{canonical}_fold{fold}"
        if legacy.is_dir():
            return legacy
    return None


def validate_fold_checkpoint(folder, c, tr, va, session, purpose="teaching_full_fold"):
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
    job_purpose = job.get('purpose')
    if purpose == 'smoke' and job_purpose != 'smoke':
        return False, "requested smoke run but checkpoint is non-smoke"
    if purpose != 'smoke' and job_purpose == 'smoke':
        return False, "smoke run cannot satisfy non-smoke validation"

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


def train_cnn_fold(name, frame, root, session, fold=0, seed=None, smoke=False):
    """Train a single fold of a CNN configuration, safely reusing matching completed runs."""
    from .training import train_one_fold
    session = load_session(session, data_root=root)
    canonical = canonical_preset_name(name)

    if is_registered_preset(canonical):
        c = get_preset_config(canonical, seed=seed, fold=fold, smoke=smoke)
    elif (PACKAGE / f'configs/{name}.json').is_file():
        c = Config(**read_json(PACKAGE / f'configs/{name}.json'))
        c = replace(c, fold=fold)
        if seed is not None:
            c = replace(c, seed=int(seed))
        if smoke:
            c = replace(c, epochs=2, warmup=1, patience=2)
    else:
        raise ValueError(f"Unknown CNN configuration: {name}")

    run_name = fold_run_name(canonical, fold, seed=c.seed, smoke=smoke)
    folder = session / run_name
    legacy_folder = session / f"{canonical}_fold{fold}"

    dev_frame = prepare_development(root) if root else frame
    tr, va = split_fold(dev_frame, fold)
    purpose = "smoke" if smoke else "teaching_full_fold"

    # Pre-sample expected training set to match contract validation
    if purpose == "smoke":
        expected_tr = sample_smoke_train(tr, fold=c.fold, max_pairs=64, fraction=c.train_fraction)
    elif c.train_fraction < 1.0:
        expected_tr = sample_train_subset(tr, fraction=c.train_fraction, fold=c.fold)
    else:
        expected_tr = tr

    # 1. Check canonical namespaced folder
    if folder.is_dir():
        is_valid, info_or_metrics = validate_fold_checkpoint(folder, c, expected_tr, va, session, purpose=purpose)
        if is_valid:
            return folder, info_or_metrics
        raise RuntimeError(
            f"Phát hiện thư mục fold '{folder.name}' chưa hoàn thành hoặc không khớp ({info_or_metrics}). "
            "Vui lòng chọn một RUN_ID mới."
        )

    # 2. Check legacy unadorned folder for backwards compatibility (default non-smoke seed 20260917)
    if not smoke and c.seed == 20260917 and legacy_folder.is_dir() and not legacy_folder.is_symlink():
        is_valid, info_or_metrics = validate_fold_checkpoint(legacy_folder, c, expected_tr, va, session, purpose=purpose)
        if is_valid:
            return legacy_folder, info_or_metrics

    runtime = Runtime(Path(root), session)
    folder, result = train_one_fold(c, runtime, run_name=run_name, train_frame=tr, valid_frame=va, purpose=purpose)

    # Create backwards-compatible relative symlink for default non-smoke seed
    if not smoke and c.seed == 20260917 and not legacy_folder.exists():
        try:
            import os
            os.symlink(folder.name, legacy_folder)
        except Exception:
            pass

    return folder, result


def train_cnn_cv(name, frame, root, session, folds=(0, 1, 2), seed=None, smoke=False):
    """Train multiple folds (or all 3 folds) for a CNN architecture."""
    session = load_session(session, data_root=root)
    dev_frame = prepare_development(root) if root else frame
    canonical = canonical_preset_name(name)

    if not is_registered_preset(canonical) and not (PACKAGE / f'configs/{name}.json').is_file():
        raise ValueError(f"Unknown CNN configuration: {name}")

    c_sample = (get_preset_config(canonical, seed=seed, fold=0, smoke=smoke)
                if is_registered_preset(canonical)
                else Config(**read_json(PACKAGE / f'configs/{name}.json')))
    c_seed = seed if seed is not None else c_sample.seed
    purpose = "smoke" if smoke else "teaching_full_fold"

    # Special handling for frozen embedding + LR (A_frozen_LR / kind="ml")
    if c_sample.kind == "ml" or canonical in ("frozen_lr", "A_frozen_LR"):
        return train_frozen_lr_cv(canonical, dev_frame, root, session, folds=folds, seed=c_seed, smoke=smoke)

    for fold in folds:
        train_cnn_fold(name, dev_frame, root, session, fold=fold, seed=c_seed, smoke=smoke)

    # Validate and assemble all 3 folds if all exist
    all_done = True
    records, predictions = [], []
    for fold in range(3):
        c_fold = (get_preset_config(canonical, seed=c_seed, fold=fold, smoke=smoke)
                  if is_registered_preset(canonical)
                  else replace(c_sample, fold=fold, seed=c_seed))
        folder = find_fold_folder(session, canonical, fold, seed=c_seed, smoke=smoke)
        tr, va = split_fold(dev_frame, fold)
        if folder is None or not folder.is_dir():
            all_done = False
            break

        if purpose == "smoke":
            expected_tr = sample_smoke_train(tr, fold=c_fold.fold, max_pairs=64, fraction=c_fold.train_fraction)
        elif c_fold.train_fraction < 1.0:
            expected_tr = sample_train_subset(tr, fraction=c_fold.train_fraction, fold=c_fold.fold)
        else:
            expected_tr = tr

        is_valid, metrics = validate_fold_checkpoint(folder, c_fold, expected_tr, va, session, purpose=purpose)
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

        oof_name = f"{canonical}_smoke_s{c_seed}_oof.csv" if smoke else f"{canonical}_s{c_seed}_oof.csv"
        models_name = f"{canonical}_smoke_s{c_seed}_models.json" if smoke else f"{canonical}_s{c_seed}_models.json"
        combined.to_csv(session / oof_name, index=False)
        write_json(session / models_name, records)

        if not smoke and c_seed == 20260917:
            combined.to_csv(session / f'{canonical}_oof.csv', index=False)
            write_json(session / f'{canonical}_models.json', records)

        return combined
    return None


def train_frozen_lr_cv(name, frame, root, session, folds=(0, 1, 2), seed=20260917, smoke=False):
    """Train frozen embedding extraction + Logistic Regression across folds.

    Protocol (historical frozen embedding classification):
    1. Extract per-image features from backbone (embedding=True) for all images in development.
    2. In each fold, decompose pairs into individual images:
       x_train = np.vstack([emb0_train, emb1_train])
       y_train = np.concatenate([1 - pair_y_train, pair_y_train])
    3. Fit on training set only:
       StandardScaler() -> PCA(n_components=min(128, emb_dim, len(x_train)-1), whiten=True, random_state=seed)
       -> LogisticRegression(C=0.03, max_iter=5000, random_state=seed, fit_intercept=True)
    4. Inference per image on validation set:
       proba0, proba1 clipped to [1e-7, 1 - 1e-7]
       log_odds0 = log(proba0 / (1 - proba0))
       log_odds1 = log(proba1 / (1 - proba1))
       score = log_odds1 - log_odds0
       p_pair = 1 / (1 + exp(-clip(score, -60, 60)))
    5. Maintain complete 800-pair OOF coverage without collisions.
    """
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from .models import model_for, embedding_pairs

    session = load_session(session, data_root=root)
    dev_frame = prepare_development(root) if root else frame
    if len(dev_frame) != 800 or not dev_frame['pair_id'].is_unique:
        raise ValueError("Development frame must contain exactly 800 unique pairs.")
    for req_col in ['pair_id', 'image_0', 'image_1', 'fake_position', 'inner_fold']:
        if req_col not in dev_frame:
            raise ValueError(f"dev_frame missing required column '{req_col}'")

    purpose = "smoke" if smoke else "teaching_full_fold"
    prefix = "frozen_lr_smoke" if smoke else "frozen_lr"

    completed_folds = {}
    records = []

    # Check existing completed folds for safe reuse or mismatch errors
    for fold in range(3):
        c_fold = get_preset_config("frozen_lr", seed=seed, fold=fold, smoke=smoke)
        fold_run = f"{prefix}_s{seed}_f{fold}"
        fold_folder = session / fold_run
        tr, va = split_fold(dev_frame, fold)
        if smoke:
            expected_tr = sample_smoke_train(tr, fold=fold, max_pairs=64, fraction=c_fold.train_fraction, base_seed=20260917)
        else:
            expected_tr = tr

        status_file = fold_folder / 'status.json'
        job_file = fold_folder / 'job.json'
        metrics_file = fold_folder / 'metrics.json'
        dev_file = fold_folder / 'development.csv'
        model_artifact = fold_folder / f"{fold_run}.joblib"

        if fold_folder.is_dir():
            if not (status_file.is_file() and job_file.is_file() and metrics_file.is_file() and dev_file.is_file() and model_artifact.is_file()):
                raise ValueError(f"Partial existing fold in {fold_folder.name}. Choose a new RUN_ID.")

            status = read_json(status_file)
            if status.get('status') != 'complete':
                raise ValueError(f"Existing fold {fold_folder.name} status is incomplete. Choose a new RUN_ID.")

            job = read_json(job_file)
            if job.get('purpose') != purpose:
                raise ValueError(f"Existing fold {fold_folder.name} purpose mismatch ({job.get('purpose')} vs {purpose}). Choose a new RUN_ID.")
            if job.get('config') != asdict(c_fold):
                raise ValueError(f"Existing fold {fold_folder.name} config mismatch. Choose a new RUN_ID.")
            if job.get('train_ids') != expected_tr.pair_id.tolist():
                raise ValueError(f"Existing fold {fold_folder.name} train_ids mismatch. Choose a new RUN_ID.")
            if job.get('validation_ids') != va.pair_id.tolist():
                raise ValueError(f"Existing fold {fold_folder.name} validation_ids mismatch. Choose a new RUN_ID.")
            if job.get('code_sha256') != current_code_hashes():
                raise ValueError(f"Existing fold {fold_folder.name} code SHA256 mismatch. Choose a new RUN_ID.")
            m_sha = sha256(model_artifact)
            if job.get('model_sha256') != m_sha:
                raise ValueError(f"Existing fold {fold_folder.name} model SHA256 mismatch. Choose a new RUN_ID.")

            metrics = read_json(metrics_file)
            if metrics.get('status') != 'passed':
                raise ValueError(f"Existing fold {fold_folder.name} metrics status is not passed. Choose a new RUN_ID.")
            if metrics.get('checkpoint_sha256') != m_sha:
                raise ValueError(f"Existing fold {fold_folder.name} metrics checkpoint SHA mismatch. Choose a new RUN_ID.")
            if metrics.get('reload_probability_max_abs_error', 1.0) >= 1e-6:
                raise ValueError(f"Existing fold {fold_folder.name} reload error exceeds tolerance. Choose a new RUN_ID.")

            dev_df = read_csv(dev_file)
            if list(dev_df.pair_id) != list(va.pair_id):
                raise ValueError(f"Existing fold {fold_folder.name} development.csv pair_id mismatch. Choose a new RUN_ID.")

            contract_copy = dict(job)
            stored_hash = contract_copy.pop('job_hash', None)
            if stored_hash != hashlib.sha256(json.dumps(contract_copy, sort_keys=True).encode()).hexdigest():
                raise ValueError('Frozen LR contract hash changed. Choose a new RUN_ID.')
            if sha256(dev_file) != job.get('development_sha256') or metrics.get('job_hash') != stored_hash:
                raise ValueError('Frozen LR predictions or metadata changed. Choose a new RUN_ID.')
            aligned_predictions(dev_df.rename(columns={'y':'fake_position', 'fold':'inner_fold'}), va)
            completed_folds[fold] = dev_df
            records.append({
                "fold": fold,
                "checkpoint": str(model_artifact.relative_to(session)),
                "sha256": m_sha,
                "purpose": purpose,
            })

    # Train any requested folds not yet completed
    needs_training = [f for f in folds if f not in completed_folds]
    if needs_training:
        c_sample = get_preset_config("frozen_lr", seed=seed, fold=0, smoke=smoke)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, weight_hash = model_for(c_sample, device=device, pretrained=True, embedding=True)
        embeddings = embedding_pairs(model, dev_frame, c_sample, Path(root), device=device)
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()
        emb0 = embeddings[:, 0, :]
        emb1 = embeddings[:, 1, :]
        emb_dim = emb0.shape[1]

        for fold in needs_training:
            c_fold = get_preset_config("frozen_lr", seed=seed, fold=fold, smoke=smoke)
            fold_run = f"{prefix}_s{seed}_f{fold}"
            fold_folder = session / fold_run
            fold_folder.mkdir(parents=True, exist_ok=True)

            tr, va = split_fold(dev_frame, fold)
            if smoke:
                expected_tr = sample_smoke_train(tr, fold=fold, max_pairs=64, fraction=c_fold.train_fraction, base_seed=20260917)
            else:
                expected_tr = tr

            train_indices = [dev_frame.index[dev_frame.pair_id == pid].item() for pid in expected_tr.pair_id]
            valid_indices = [dev_frame.index[dev_frame.pair_id == pid].item() for pid in va.pair_id]

            x_tr0 = emb0[train_indices]
            x_tr1 = emb1[train_indices]
            y_pairs = expected_tr.fake_position.to_numpy().astype(int)

            x_train = np.vstack([x_tr0, x_tr1])
            y_train = np.concatenate([1 - y_pairs, y_pairs])

            scaler = StandardScaler()
            x_train_scaled = scaler.fit_transform(x_train)

            n_comp = min(128, emb_dim, len(x_train) - 1)
            pca = PCA(n_components=n_comp, whiten=True, random_state=seed)
            x_train_pca = pca.fit_transform(x_train_scaled)

            clf = LogisticRegression(C=0.03, max_iter=5000, random_state=seed, fit_intercept=True)
            clf.fit(x_train_pca, y_train)

            x_va0 = emb0[valid_indices]
            x_va1 = emb1[valid_indices]

            p0 = clf.predict_proba(pca.transform(scaler.transform(x_va0)))[:, 1]
            p1 = clf.predict_proba(pca.transform(scaler.transform(x_va1)))[:, 1]

            p0 = np.clip(p0, 1e-7, 1.0 - 1e-7)
            p1 = np.clip(p1, 1e-7, 1.0 - 1e-7)

            log_odds0 = np.log(p0 / (1.0 - p0))
            log_odds1 = np.log(p1 / (1.0 - p1))
            score = log_odds1 - log_odds0
            p_pairs = 1.0 / (1.0 + np.exp(-np.clip(score, -60.0, 60.0)))

            model_artifact = fold_folder / f"{fold_run}.joblib"
            fold_pipeline = {"scaler": scaler, "pca": pca, "clf": clf, "config": asdict(c_fold)}
            joblib.dump(fold_pipeline, model_artifact)

            # Reload parity check
            reloaded = joblib.load(model_artifact)
            p0_r = reloaded["clf"].predict_proba(reloaded["pca"].transform(reloaded["scaler"].transform(x_va0)))[:, 1]
            p1_r = reloaded["clf"].predict_proba(reloaded["pca"].transform(reloaded["scaler"].transform(x_va1)))[:, 1]
            p0_r = np.clip(p0_r, 1e-7, 1.0 - 1e-7)
            p1_r = np.clip(p1_r, 1e-7, 1.0 - 1e-7)
            lo0_r = np.log(p0_r / (1.0 - p0_r))
            lo1_r = np.log(p1_r / (1.0 - p1_r))
            p_pairs_r = 1.0 / (1.0 + np.exp(-np.clip(lo1_r - lo0_r, -60.0, 60.0)))
            reload_error = float(np.max(np.abs(p_pairs - p_pairs_r)))
            if reload_error >= 1e-6:
                raise ValueError(f"Model reload verification failed: error {reload_error} exceeds tolerance 1e-6")

            dev_fold = va.copy().rename(columns={'fake_position': 'y', 'inner_fold': 'fold'})
            dev_fold['p'] = p_pairs
            dev_fold.to_csv(fold_folder / 'development.csv', index=False)

            m_sha = sha256(model_artifact)
            contract = {
                "id": fold_run,
                "purpose": purpose,
                "is_smoke": smoke,
                "config": asdict(c_fold),
                "train_ids": expected_tr.pair_id.tolist(),
                "validation_ids": va.pair_id.tolist(),
                "code_sha256": current_code_hashes(),
                "model_file": model_artifact.name,
                "model_sha256": m_sha,
                "development_sha256": sha256(fold_folder / "development.csv"),
                "pretrained_weight_sha256": weight_hash,
            }
            contract_copy = dict(contract)
            job_hash = hashlib.sha256(json.dumps(contract_copy, sort_keys=True).encode()).hexdigest()
            contract["job_hash"] = job_hash
            write_json(fold_folder / "job.json", contract)

            metrics = {
                "status": "passed",
                "job_hash": job_hash,
                "checkpoint_sha256": m_sha,
                "reload_probability_max_abs_error": reload_error,
                "holdout_evaluated": False,
            }
            write_json(fold_folder / "metrics.json", metrics)
            write_json(fold_folder / "status.json", {"status": "complete"})

            completed_folds[fold] = dev_fold
            records.append({
                "fold": fold,
                "checkpoint": str(model_artifact.relative_to(session)),
                "sha256": m_sha,
                "purpose": purpose,
            })

    # Assemble OOF only when all 3 folds are complete and verified; don't write NaN or partial
    if sorted(completed_folds.keys()) == [0, 1, 2]:
        predictions = []
        for f in range(3):
            df = completed_folds[f].rename(columns={'y': 'fake_position', 'fold': 'inner_fold'})
            predictions.append(df)
        combined = pd.concat(predictions, ignore_index=True).sort_values('pair_id').reset_index(drop=True)

        if len(combined) != 800 or not combined.pair_id.is_unique or set(combined.pair_id) != set(dev_frame.pair_id):
            raise ValueError("Assembled OOF coverage for frozen_lr is incomplete.")
        p_vals = combined['p'].to_numpy(float)
        if not np.isfinite(p_vals).all() or ((p_vals < 0) | (p_vals > 1)).any():
            raise ValueError("Assembled OOF contains NaNs or out-of-bounds probabilities.")

        oof_name = f"frozen_lr_smoke_s{seed}_oof.csv" if smoke else f"frozen_lr_s{seed}_oof.csv"
        models_name = f"frozen_lr_smoke_s{seed}_models.json" if smoke else f"frozen_lr_s{seed}_models.json"
        combined.to_csv(session / oof_name, index=False)
        write_json(session / models_name, sorted(records, key=lambda r: r['fold']))

        if not smoke and seed == 20260917:
            combined.to_csv(session / "frozen_lr_oof.csv", index=False)
            write_json(session / "frozen_lr_models.json", sorted(records, key=lambda r: r['fold']))

        return combined
    return None


def complete_method_cv(method, frame, root, session, seed=None, smoke=False):
    """Complete 3-fold cross validation for the user-selected method."""
    session = load_session(session, data_root=root)
    dev_frame = prepare_development(root) if root else frame
    canonical = canonical_preset_name(method)
    c_seed = seed if seed is not None else 20260917

    if canonical == 'lr':
        return fit_lr_cv(dev_frame, root, session)
    if canonical == 'blend':
        b2_oof = train_cnn_cv('b2', dev_frame, root, session, folds=(0, 1, 2), seed=c_seed, smoke=smoke)
        nat_oof = train_cnn_cv('native', dev_frame, root, session, folds=(0, 1, 2), seed=c_seed, smoke=smoke)
        blend_df = blend(b2_oof, nat_oof, dev_frame)
        oof_name = f'blend_smoke_s{c_seed}_oof.csv' if smoke else f'blend_s{c_seed}_oof.csv'
        blend_df.to_csv(session / oof_name, index=False)
        if not smoke and c_seed == 20260917:
            blend_df.to_csv(session / 'blend_oof.csv', index=False)
        return blend_df

    if is_registered_preset(canonical) or canonical in {'center60', 'native', 'b2', 'frozen', 'resampled'}:
        return train_cnn_cv(canonical, dev_frame, root, session, folds=(0, 1, 2), seed=c_seed, smoke=smoke)

    raise ValueError(f"Unknown method '{method}'. Supported: 'lr', 'center60', 'native', 'b2', 'blend', etc.")


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


def compare_oof(session, names, seed=20260917, smoke=False):
    """Compare multiple OOF predictions against development labels.

    Strictly adheres to requested seed and smoke mode; never mixes modes or skips missing models.

    Exception:
        Logistic Regression (names 'lr', 'lr_full32', or any ablation variant) is non-iterative tabular CV
        that operates deterministically on the full 800 development set in all execution modes, producing
        canonical full-dev artifacts ('lr_full32_oof.csv' / 'lr_{variant}_oof.csv'). As an explicit design rule,
        `compare_oof` resolves LR models to these canonical deterministic full-dev artifacts even under smoke mode.
        For all CNN models, mode fallback between smoke and non-smoke is strictly prohibited.
    """
    session = Path(session)
    frame = read_csv(session / 'development.csv')
    predictions, rows = {}, []
    for name in names:
        canonical = canonical_preset_name(name)
        # Check if this model is an LR tabular model / ablation variant
        is_lr = False
        lr_variant = None
        if canonical in ('lr', 'lr_full32') or name in ('lr', 'lr_full32', 'full32'):
            is_lr = True
            lr_variant = 'full32'
        else:
            cand = name[3:] if name.startswith('lr_') else name
            resolved_v = resolve_lr_ablation_name(cand)
            if resolved_v in list_lr_ablations():
                is_lr = True
                lr_variant = resolved_v

        if is_lr:
            # Deterministic full-dev LR artifacts for all modes (smoke and non-smoke)
            candidates = [
                session / f'lr_{lr_variant}_oof.csv',
            ]
            if lr_variant == 'full32':
                candidates.append(session / 'lr_oof.csv')
        elif smoke:
            # CNN smoke mode: strictly look for smoke OOF, NEVER fallback to non-smoke
            candidates = [
                session / f'{canonical}_smoke_s{seed}_oof.csv',
            ]
        else:
            # CNN non-smoke mode: strictly look for non-smoke OOF, NEVER fallback to smoke
            candidates = [
                session / f'{canonical}_s{seed}_oof.csv',
            ]
            if seed == 20260917:
                candidates.extend([
                    session / f'{canonical}_oof.csv',
                    session / f'{name}_oof.csv',
                ])

        oof_path = next((p for p in candidates if p.is_file()), None)
        if oof_path is None:
            mode_desc = "smoke" if smoke else "non-smoke"
            raise FileNotFoundError(
                f"Missing requested OOF predictions for '{name}' (seed={seed}, mode={mode_desc}) in {session}"
            )
        pred = aligned_predictions(read_csv(oof_path), frame)
        predictions[name] = pred
        rows.append({'model': name, **metric(frame.fake_position, pred.p)})

    table = pd.DataFrame(rows)
    table_name = f'comparison_smoke_s{seed}.csv' if smoke else ('comparison.csv' if seed == 20260917 else f'comparison_s{seed}.csv')
    table.to_csv(session / table_name, index=False)
    return table, predictions


def blend(a, b, frame):
    a, b = aligned_predictions(a, frame), aligned_predictions(b, frame)
    result = frame.copy()
    result['p'] = .5 * a.p.to_numpy() + .5 * b.p.to_numpy()
    return result


def infer_lr(frame, root, session, feature_columns=None, variant='full32'):
    """Inference for 3-fold Logistic Regression.

    Validates session without treating test_root as train_root, and checks test separation.
    Uses strictly the requested canonical variant with manifest hash validation, with no fallback to full32.
    """
    session = load_session(session)
    check_test_separation(frame, root, session)

    if not variant or "/" in str(variant) or "\\" in str(variant) or ".." in str(variant):
        raise ValueError(f"Invalid variant name '{variant}' (path traversal detected)")
    v_name = resolve_lr_ablation_name(variant)
    if v_name not in list_lr_ablations():
        raise ValueError(f"Unknown LR variant '{variant}'. Available: {list_lr_ablations()}")

    # Find job contract for variant - NO fallback to full32 for other variants
    if v_name == 'full32':
        job_candidates = [
            session / f'lr_{v_name}_job.json',
            session / 'lr_job.json',
        ]
        feat_candidates = [
            session / f'lr_{v_name}_feature_names.json',
            session / 'lr_feature_names.json',
        ]
    else:
        job_candidates = [
            session / f'lr_{v_name}_job.json',
        ]
        feat_candidates = [
            session / f'lr_{v_name}_feature_names.json',
        ]

    job_file = next((p for p in job_candidates if p.is_file()), None)
    if job_file is None:
        raise FileNotFoundError(f"Missing job contract for LR variant '{v_name}' in {session}")
    job = read_json(job_file)

    feat_file = next((p for p in feat_candidates if p.is_file()), None)
    if feat_file is None:
        raise FileNotFoundError(f"Missing feature names for LR variant '{v_name}' in {session}")
    recorded_columns = read_json(feat_file)

    if feature_columns is not None and list(feature_columns) != recorded_columns:
        raise ValueError(f"Inference feature columns must match fitted LR columns for '{v_name}'.")
    feature_columns = recorded_columns

    models = []
    recorded_model_sha = job.get('model_sha256', {})
    for f in range(3):
        if v_name == 'full32':
            m_candidates = [
                session / f'lr_{v_name}_fold{f}.joblib',
                session / f'lr_fold{f}.joblib',
            ]
        else:
            m_candidates = [
                session / f'lr_{v_name}_fold{f}.joblib',
            ]
        m_file = next((p for p in m_candidates if p.is_file()), None)
        if m_file is None:
            raise FileNotFoundError(f"Missing LR model fold {f} for variant '{v_name}' in {session}")

        expected_sha = recorded_model_sha.get(f'fold{f}')
        if expected_sha and sha256(m_file) != expected_sha:
            raise ValueError(f"LR model fold {f} SHA256 mismatch with job contract for '{v_name}'")

        models.append(joblib.load(m_file))

    x, _ = extract_pair_features(frame, root, feature_columns=feature_columns)
    result = frame.copy()
    result['p'] = np.mean([m.predict_proba(x)[:, 1] for m in models], axis=0)
    return result


def infer_cnn(name, frame, root, session, device='cuda', seed=None, smoke=False):
    """Inference for 3-fold CNN models.

    Validates session without treating test_root as train_root, and checks test separation.
    Requires validated 3-fold contracts and hashes for selected preset/seed/mode even when seed is None.
    Verifies checkpoint config and canonical names.
    """
    import torch
    from .models import model_for, predict_pairs
    session = load_session(session)
    check_test_separation(frame, root, session)

    canonical = canonical_preset_name(name)
    c_seed = seed if seed is not None else 20260917

    if seed is not None:
        models_file_name = f"{canonical}_smoke_s{c_seed}_models.json" if smoke else f"{canonical}_s{c_seed}_models.json"
        records_file = session / models_file_name
    else:
        if smoke:
            records_file = session / f"{canonical}_smoke_s{c_seed}_models.json"
        else:
            candidates = [
                session / f"{canonical}_s{c_seed}_models.json",
                session / f"{canonical}_models.json",
            ]
            records_file = next((p for p in candidates if p.is_file()), None)

    if records_file is None or not records_file.is_file():
        raise FileNotFoundError(
            f"Model index for '{name}' (seed={c_seed}, smoke={smoke}) not found in session."
        )

    records = read_json(records_file)
    if sorted(r['fold'] for r in records) != [0, 1, 2]:
        raise ValueError('Need exactly three fold checkpoints for CV inference.')

    # Require validated 3-fold contracts and hashes for selected preset/seed/mode
    for record in records:
        ckpt_path = session / record['checkpoint']
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
        if sha256(ckpt_path) != record['sha256']:
            raise ValueError(f"Checkpoint hash mismatch for {ckpt_path.name}.")

        job_file = ckpt_path.parent / "job.json"
        if not job_file.is_file():
            raise FileNotFoundError(f"Missing mandatory job.json contract for checkpoint {ckpt_path} in {ckpt_path.parent}")

        job_meta = read_json(job_file)
        job_is_smoke = job_meta.get("is_smoke") is True or job_meta.get("purpose") == "smoke"
        if smoke and not job_is_smoke:
            raise ValueError(f"Requested smoke inference, but checkpoint {ckpt_path.name} was trained without smoke.")
        if not smoke and job_is_smoke:
            raise ValueError(f"Checkpoint {ckpt_path.name} was trained in smoke mode and cannot satisfy non-smoke inference.")

        # Verify seed even when seed argument was None
        ckpt_seed = job_meta.get("config", {}).get("seed")
        if ckpt_seed != c_seed:
            raise ValueError(f"Checkpoint {ckpt_path.name} seed ({ckpt_seed}) does not match required seed ({c_seed}).")

        # Verify fold
        ckpt_fold = job_meta.get("config", {}).get("fold")
        if ckpt_fold != record['fold']:
            raise ValueError(f"Checkpoint {ckpt_path.name} fold ({ckpt_fold}) does not match record fold ({record['fold']}).")

        expected_c = get_preset_config(canonical, seed=c_seed, fold=record['fold'], smoke=smoke)
        dev_frame = read_csv(session / 'development.csv')
        tr, va = split_fold(dev_frame, record['fold'])
        if smoke:
            tr = sample_smoke_train(tr, fold=record['fold'], max_pairs=64, fraction=expected_c.train_fraction)
        elif expected_c.train_fraction < 1:
            tr = sample_train_subset(tr, fold=record['fold'], fraction=expected_c.train_fraction)
        valid, reason = validate_fold_checkpoint(ckpt_path.parent, expected_c, tr, va, session,
                                                purpose='smoke' if smoke else 'teaching_full_fold')
        if not valid:
            raise ValueError(f'Invalid inference checkpoint: {reason}')

    probabilities = []
    for record in records:
        checkpoint = session / record['checkpoint']
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
