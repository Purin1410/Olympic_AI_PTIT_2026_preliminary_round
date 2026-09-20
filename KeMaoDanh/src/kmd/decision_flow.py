"""Dynamic backbone screening, confirmation, and selected-native blend.

Legacy suite/gate/select_candidate/blend APIs stay unchanged. These helpers
train only when a suite is completing missing fold runs; verification of
already-written artifacts never fits a model.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .core import metric, read_csv, read_json, sha256, split_fold, write_json
from .pipeline import (
    aligned_predictions,
    blend,
    current_code_hashes,
    load_session,
    validate_fold_checkpoint,
)
from .presets import canonical_preset_name, get_preset_config

PROTOCOL = 'backbone_flow_v1'
PRIMARY_SEED = 20260917
CONFIRMATION_SEEDS = (20260917, 20260918, 20260919)
MINIMUM_GAIN = 0.005
BLEND_METHOD = 'blend_selected_native'
SCREENING_METHODS = (
    'center60_cap48', 'dense288', 'center72', 'b0_224', 'b2_224', 'b2',
    'center60', 'native',
)
CHALLENGERS = ('dense288', 'center72', 'b0_224', 'b2_224', 'b2')
ANCHORS = ('center60', 'center60_cap48', 'native')
COMPARISON_METHODS = (
    'lr_full32', 'frozen', 'center60', 'center60_cap48', 'dense288',
    'center72', 'b0_224', 'b2_224', 'b2', 'native', 'resampled',
)
TABLE_COLUMNS = (
    'model', 'arch', 'seed', 'smoke', 'n', 'macro_f1', 'accuracy', 'auc',
    'log_loss', 'errors',
)
SELECTION_FIELDS = (
    'protocol', 'method', 'confirmed_methods', 'shortlist', 'seeds',
    'mean_macro_f1', 'seed_for_submission', 'partition', 'private_used',
)
DECISION_FIELDS = SELECTION_FIELDS + (
    'best_single', 'blend_delta_vs_best_single', 'minimum_gain',
    'components', 'weights', 'comparison_methods', 'blend_status',
)
SCREENING_CSV = 'backbone_screening_comparison.csv'
SCREENING_SMOKE_CSV = 'backbone_screening_smoke_comparison.csv'
SHORTLIST_JSON = 'screening_shortlist.json'
CONFIRMATION_CSV = 'single_confirmation_comparison.csv'
SELECTION_JSON = 'single_selection.json'
BLEND_CSV = 'selected_blend_comparison.csv'
DECISION_JSON = 'submission_decision.json'
BACKBONE_FLOW_FILES = (
    SCREENING_CSV, SHORTLIST_JSON, CONFIRMATION_CSV, SELECTION_JSON,
    BLEND_CSV, DECISION_JSON,
)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _public_record(record, fields):
    return {key: _jsonable(record[key]) for key in fields}


def _empty_table(extra=()):
    return pd.DataFrame(columns=list(TABLE_COLUMNS) + list(extra))


def _model_name(arch, seed, smoke):
    return f'{arch}_smoke_s{seed}' if smoke else f'{arch}_s{seed}'


def _score_row(arch, seed, smoke, scores, extra=None):
    row = {
        'model': _model_name(arch, seed, smoke),
        'arch': arch,
        'seed': int(seed),
        'smoke': bool(smoke),
        'n': scores['n'],
        'macro_f1': float(scores['macro_f1']),
        'accuracy': float(scores['accuracy']),
        'auc': None if scores['auc'] is None else float(scores['auc']),
        'log_loss': float(scores['log_loss']),
        'errors': int(scores['errors']),
    }
    if extra:
        row.update(extra)
    return row


def _require_development_frame(frame, smoke=False):
    for col in ('pair_id', 'image_0', 'image_1', 'fake_position', 'inner_fold'):
        if col not in frame.columns:
            raise ValueError(f'development frame missing {col}.')
    if not frame.pair_id.is_unique:
        raise ValueError('development frame pair_id values must be unique.')
    folds = set(int(v) for v in frame.inner_fold.tolist())
    if not folds <= {0, 1, 2}:
        raise ValueError('inner_fold must use only 0, 1, and 2.')
    if not smoke:
        if len(frame) != 800:
            raise ValueError('Need the canonical 800 unique development pairs.')
        if folds != {0, 1, 2}:
            raise ValueError('Need folds 0, 1, and 2.')


def _models_path(session, method, seed, smoke=False):
    canonical = canonical_preset_name(method)
    if smoke:
        return session / f'{canonical}_smoke_s{seed}_models.json'
    path = session / f'{canonical}_s{seed}_models.json'
    if path.is_file():
        return path
    if int(seed) == PRIMARY_SEED:
        legacy = session / f'{canonical}_models.json'
        if legacy.is_file():
            return legacy
    return path


def _oof_path(session, method, seed, smoke=False):
    canonical = canonical_preset_name(method)
    if smoke:
        return session / f'{canonical}_smoke_s{seed}_oof.csv'
    path = session / f'{canonical}_s{seed}_oof.csv'
    if path.is_file():
        return path
    if int(seed) == PRIMARY_SEED:
        legacy = session / f'{canonical}_oof.csv'
        if legacy.is_file():
            return legacy
    return path


def _tables_match(left, right):
    if list(left.columns) != list(right.columns) or len(left) != len(right):
        return False
    if left.empty and right.empty:
        return True
    keys = [c for c in ('arch', 'seed', 'model') if c in left.columns]
    a = left.sort_values(keys).reset_index(drop=True) if keys else left.reset_index(drop=True)
    b = right.sort_values(keys).reset_index(drop=True) if keys else right.reset_index(drop=True)
    for col in a.columns:
        sa, sb = a[col], b[col]
        if pd.api.types.is_numeric_dtype(sa) or pd.api.types.is_numeric_dtype(sb):
            aa = pd.to_numeric(sa, errors='coerce').to_numpy(float)
            bb = pd.to_numeric(sb, errors='coerce').to_numpy(float)
            if not np.array_equal(np.isnan(aa), np.isnan(bb)):
                return False
            if not np.allclose(aa, bb, atol=1e-12, rtol=0, equal_nan=True):
                return False
        elif sa.astype(str).tolist() != sb.astype(str).tolist():
            return False
    return True


def _write_csv_immutable(path, table):
    path = Path(path)
    if path.is_file():
        existing = read_csv(path)
        if not _tables_match(existing, table):
            raise ValueError(f'{path.name} changed. Keep prior evidence; use a new RUN_ID.')
        return existing
    table.to_csv(path, index=False)
    return table


def _write_json_immutable(path, record):
    path = Path(path)
    payload = _jsonable(record)
    if path.is_file():
        existing = read_json(path)
        if existing != payload:
            raise ValueError(f'{path.name} changed. Keep prior evidence; use a new RUN_ID.')
        return existing
    write_json(path, payload)
    return payload


def _assert_bound_hashes(record, session):
    session = Path(session)
    if record.get('code_sha256') != current_code_hashes():
        raise ValueError('Source changed. Use a new RUN_ID.')
    files = record.get('files') or {}
    if not files:
        raise ValueError('Decision records must bind upstream file hashes.')
    for rel, digest in files.items():
        target = session / rel
        if not target.is_file() or sha256(target) != digest:
            raise ValueError(f'Upstream artifact changed: {rel}')


def _hash_files(session, paths):
    files = {}
    for path in paths:
        path = Path(path)
        if path.is_file():
            files[str(path.relative_to(session))] = sha256(path)
    return files


def shortlist_challengers(primary_macro_f1):
    """Top two challengers by primary-seed Macro-F1; ties are alphabetical."""
    missing = [name for name in CHALLENGERS if name not in primary_macro_f1]
    if missing:
        raise ValueError(f'Screening is missing challengers: {missing}.')
    values = {name: float(primary_macro_f1[name]) for name in CHALLENGERS}
    if any(not np.isfinite(v) for v in values.values()):
        raise ValueError('Nonfinite screening Macro-F1.')
    ranked = sorted(CHALLENGERS, key=lambda name: (-values[name], name))
    return ranked[:2]


def choose_confirmed_winner(mean_macro_f1, confirmed_methods):
    """Best mean Macro-F1 among confirmed methods; ties are alphabetical."""
    if not confirmed_methods:
        raise ValueError('Need confirmed methods before choosing a winner.')
    missing = [name for name in confirmed_methods if name not in mean_macro_f1]
    if missing:
        raise ValueError(f'Confirmation is missing methods: {missing}.')
    values = {name: float(mean_macro_f1[name]) for name in confirmed_methods}
    if any(not np.isfinite(v) for v in values.values()):
        raise ValueError('Nonfinite confirmation Macro-F1.')
    return sorted(confirmed_methods, key=lambda name: (-values[name], name))[0]


def decide_blend(winner, seed_gains, threshold=MINIMUM_GAIN):
    """Promote a 50/50 winner+native blend only on unrounded mean gain."""
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError('Invalid minimum improvement.')
    if winner == 'native':
        return {
            'method': 'native',
            'blend_status': 'skipped_self_blend',
            'blend_delta_vs_best_single': None,
            'components': [],
            'weights': [],
        }
    values = np.asarray(list(seed_gains), dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError('Need exactly three finite seed gains.')
    mean = float(values.mean())
    promoted = mean >= float(threshold)
    return {
        'method': BLEND_METHOD if promoted else winner,
        'blend_status': 'tested',
        'blend_delta_vs_best_single': mean,
        'components': [winner, 'native'],
        'weights': [0.5, 0.5],
    }


def comparison_methods_for(blend_status):
    methods = list(COMPARISON_METHODS)
    if blend_status == 'tested':
        methods.append(BLEND_METHOD)
    return list(dict.fromkeys(methods))


def verify_method_artifacts(session, method, seed, frame, smoke=False):
    """Recompute OOF scores from verified fold artifacts. Never trains."""
    session = Path(session)
    canonical = canonical_preset_name(method)
    index_path = _models_path(session, canonical, seed, smoke=smoke)
    oof_path = _oof_path(session, canonical, seed, smoke=smoke)
    if not index_path.is_file() or not oof_path.is_file():
        raise ValueError(f'Missing verified artifacts for {canonical} seed {seed}.')
    records = read_json(index_path)
    if sorted(r.get('fold') for r in records) != [0, 1, 2]:
        raise ValueError(f'Incomplete folds for {canonical} seed {seed}.')
    aligned = aligned_predictions(read_csv(oof_path), frame)
    p = aligned.p.to_numpy(float)
    if not np.isfinite(p).all():
        raise ValueError(f'Nonfinite OOF probabilities for {canonical} seed {seed}.')
    purpose = 'smoke' if smoke else 'teaching_full_fold'
    for record in records:
        fold = record.get('fold')
        if fold not in (0, 1, 2):
            raise ValueError(f'Invalid fold {fold} for {canonical} seed {seed}.')
        rel = record.get('checkpoint')
        checkpoint = session / (rel or '')
        if not rel or not checkpoint.is_file():
            raise ValueError(f'Missing checkpoint for {canonical} seed {seed} fold {fold}.')
        if sha256(checkpoint) != record.get('sha256'):
            raise ValueError(f'Checkpoint hash mismatch for {canonical} seed {seed} fold {fold}.')
        folder = checkpoint.parent
        config = get_preset_config(canonical, seed=int(seed), fold=int(fold), smoke=smoke)
        train, valid = split_fold(frame, fold)
        if smoke:
            from .subsets import sample_smoke_train
            train = sample_smoke_train(train, fold=fold, max_pairs=64, fraction=config.train_fraction)
        ok, info = validate_fold_checkpoint(folder, config, train, valid, session, purpose=purpose)
        if not ok:
            raise ValueError(
                f'Unverified artifact for {canonical} seed {seed} fold {fold}: {info}'
            )
        job = read_json(folder / 'job.json')
        if job.get('config', {}).get('seed') != int(seed):
            raise ValueError(f'Job seed mismatch for {canonical} seed {seed} fold {fold}.')
        fold_oof = aligned[aligned.inner_fold == fold].sort_values('pair_id')
        fold_dev = read_csv(folder / 'development.csv').sort_values('pair_id')
        if (fold_oof.pair_id.tolist() != fold_dev.pair_id.tolist()
                or not np.allclose(fold_oof.p.to_numpy(float), fold_dev.p.to_numpy(float),
                                   atol=1e-12, rtol=0)):
            raise ValueError(
                f'OOF does not match fold predictions for {canonical} seed {seed} fold {fold}.'
            )
    scores = metric(aligned.fake_position, aligned.p)
    if not np.isfinite(scores['macro_f1']):
        raise ValueError(f'Nonfinite Macro-F1 for {canonical} seed {seed}.')
    return aligned, scores


def _train_cnn_cv(name, frame, root, session, seed, smoke):
    from .pipeline import train_cnn_cv
    return train_cnn_cv(name, frame, root, session, folds=(0, 1, 2), seed=seed, smoke=smoke)


def _complete_method(name, frame, root, session, seed, smoke):
    oof = _train_cnn_cv(name, frame, root, session, seed=seed, smoke=smoke)
    if oof is None:
        raise ValueError(f'Incomplete folds for {name} seed {seed}.')
    return verify_method_artifacts(session, name, seed, frame, smoke=smoke)


def _primary_f1_from_table(table):
    primary = table[(table.seed == PRIMARY_SEED) & (table.smoke == False)]
    values = {}
    for arch, group in primary.groupby('arch'):
        if len(group) != 1 or not np.isfinite(group.macro_f1.iloc[0]):
            raise ValueError(f'Need one finite primary-seed score for {arch}.')
        values[arch] = float(group.macro_f1.iloc[0])
    return values


def _mean_f1_from_table(table, methods, seeds=CONFIRMATION_SEEDS):
    means = {}
    for arch in methods:
        rows = table[(table.arch == arch) & (table.smoke == False)]
        if set(int(s) for s in rows.seed) != set(seeds) or len(rows) != len(seeds):
            raise ValueError(f'Incomplete seed coverage for {arch}.')
        if not np.isfinite(rows.macro_f1).all():
            raise ValueError(f'Nonfinite confirmation Macro-F1 for {arch}.')
        means[arch] = float(rows.macro_f1.mean())
    return means


def _screening_csv_name(smoke):
    return SCREENING_SMOKE_CSV if smoke else SCREENING_CSV


def _load_verified_screening(session, frame, smoke=False):
    session = Path(session)
    path = session / _screening_csv_name(smoke)
    if not path.is_file():
        raise ValueError('Need a verified backbone screening table before confirmation.')
    stored = read_csv(path)
    rows = []
    predictions = {}
    for arch in SCREENING_METHODS:
        aligned, scores = verify_method_artifacts(
            session, arch, PRIMARY_SEED, frame, smoke=smoke
        )
        predictions[arch] = aligned
        rows.append(_score_row(arch, PRIMARY_SEED, smoke, scores))
    table = pd.DataFrame(rows, columns=list(TABLE_COLUMNS))
    if not _tables_match(stored[list(TABLE_COLUMNS)], table):
        raise ValueError('Screening table does not match verified OOF.')
    if set(table.arch) != set(SCREENING_METHODS) or len(table) != len(SCREENING_METHODS):
        raise ValueError('Screening must contain the eight declared configs once.')
    return table, predictions


def _confirmed_methods(shortlist):
    return list(shortlist) + [name for name in ANCHORS if name not in shortlist]


def _shortlist_record(session, shortlist, primary_f1, smoke=False):
    session = Path(session)
    files = _hash_files(session, [
        session / _screening_csv_name(smoke),
        *(_oof_path(session, arch, PRIMARY_SEED, smoke=smoke) for arch in SCREENING_METHODS),
        *(_models_path(session, arch, PRIMARY_SEED, smoke=smoke) for arch in SCREENING_METHODS),
    ])
    return {
        'protocol': PROTOCOL,
        'shortlist': list(shortlist),
        'challengers': list(CHALLENGERS),
        'anchors': list(ANCHORS),
        'primary_seed': PRIMARY_SEED,
        'primary_macro_f1': _jsonable(primary_f1),
        'code_sha256': current_code_hashes(),
        'files': files,
        'partition': 'development',
        'private_used': False,
        'smoke': bool(smoke),
    }


def _selection_record(shortlist, confirmed, winner, mean_f1, session, extra_files=()):
    session = Path(session)
    files = _hash_files(session, [
        session / SHORTLIST_JSON,
        session / CONFIRMATION_CSV,
        *extra_files,
    ])
    record = {
        'protocol': PROTOCOL,
        'method': winner,
        'confirmed_methods': list(confirmed),
        'shortlist': list(shortlist),
        'seeds': list(CONFIRMATION_SEEDS),
        'mean_macro_f1': _jsonable(mean_f1),
        'seed_for_submission': PRIMARY_SEED,
        'partition': 'development',
        'private_used': False,
        'code_sha256': current_code_hashes(),
        'files': files,
        'shortlist_sha256': sha256(session / SHORTLIST_JSON) if (session / SHORTLIST_JSON).is_file() else None,
    }
    return record


def _decision_record(selection, blend_choice, seed_gains, session):
    session = Path(session)
    files = _hash_files(session, [
        session / SELECTION_JSON,
        session / CONFIRMATION_CSV,
        session / BLEND_CSV,
        session / SHORTLIST_JSON,
    ])
    record = dict(selection)
    record.update(blend_choice)
    record['best_single'] = selection['method']
    record['minimum_gain'] = float(MINIMUM_GAIN)
    record['comparison_methods'] = comparison_methods_for(blend_choice['blend_status'])
    record['seed_gains'] = _jsonable(list(seed_gains)) if seed_gains is not None else None
    record['code_sha256'] = current_code_hashes()
    record['files'] = files
    if (session / SELECTION_JSON).is_file():
        record['selection_sha256'] = sha256(session / SELECTION_JSON)
    return record


def _error_groups(y, winner_p, blend_p):
    y = np.asarray(y, dtype=int)
    w_hat = np.asarray(winner_p) >= 0.5
    b_hat = np.asarray(blend_p) >= 0.5
    rescued = int(((w_hat != y) & (b_hat == y)).sum())
    broken = int(((w_hat == y) & (b_hat != y)).sum())
    return rescued, broken


def run_backbone_screening_suite(frame, root, session, smoke=False):
    """Train or reuse the eight primary-seed screening configs."""
    session = load_session(session, data_root=root)
    _require_development_frame(frame, smoke=smoke)
    predictions = {}
    rows = []
    for arch in SCREENING_METHODS:
        aligned, scores = _complete_method(
            arch, frame, root, session, seed=PRIMARY_SEED, smoke=smoke
        )
        predictions[arch] = aligned
        rows.append(_score_row(arch, PRIMARY_SEED, smoke, scores))
    table = pd.DataFrame(rows, columns=list(TABLE_COLUMNS))
    _write_csv_immutable(session / _screening_csv_name(smoke), table)
    return table, predictions


def run_single_confirmation_suite(frame, root, session, smoke=False):
    """Shortlist from verified screening, then confirm at three seeds."""
    session = load_session(session, data_root=root)
    if smoke:
        return _empty_table(), None
    _require_development_frame(frame, smoke=False)
    screening, _ = _load_verified_screening(session, frame, smoke=False)
    primary_f1 = _primary_f1_from_table(screening)
    shortlist = shortlist_challengers(primary_f1)
    _write_json_immutable(
        session / SHORTLIST_JSON,
        _shortlist_record(session, shortlist, primary_f1, smoke=False),
    )
    confirmed = _confirmed_methods(shortlist)
    rows = []
    for arch in confirmed:
        for seed in CONFIRMATION_SEEDS:
            if int(seed) == PRIMARY_SEED:
                aligned, scores = verify_method_artifacts(
                    session, arch, seed, frame, smoke=False
                )
            else:
                aligned, scores = _complete_method(
                    arch, frame, root, session, seed=seed, smoke=False
                )
            rows.append(_score_row(arch, seed, False, scores))
    table = pd.DataFrame(rows, columns=list(TABLE_COLUMNS))
    mean_f1 = _mean_f1_from_table(table, confirmed)
    winner = choose_confirmed_winner(mean_f1, confirmed)
    _write_csv_immutable(session / CONFIRMATION_CSV, table)
    extra = []
    for arch in confirmed:
        for seed in CONFIRMATION_SEEDS:
            extra.extend([
                _oof_path(session, arch, seed, smoke=False),
                _models_path(session, arch, seed, smoke=False),
            ])
    selection = _selection_record(shortlist, confirmed, winner, mean_f1, session, extra)
    stored = _write_json_immutable(session / SELECTION_JSON, selection)
    return table, _public_record(stored, SELECTION_FIELDS)


def _load_verified_selection(session, frame):
    session = Path(session)
    path = session / SELECTION_JSON
    shortlist_path = session / SHORTLIST_JSON
    if not path.is_file() or not shortlist_path.is_file():
        raise ValueError('Need verified screening_shortlist and single_selection records.')
    screening, _ = _load_verified_screening(session, frame, smoke=False)
    primary_f1 = _primary_f1_from_table(screening)
    shortlist = shortlist_challengers(primary_f1)
    stored_shortlist = read_json(shortlist_path)
    expected_shortlist = _shortlist_record(session, shortlist, primary_f1, smoke=False)
    if stored_shortlist != expected_shortlist:
        raise ValueError('screening_shortlist does not match verified screening.')
    _assert_bound_hashes(stored_shortlist, session)
    if stored_shortlist.get('code_sha256') != expected_shortlist['code_sha256']:
        raise ValueError('screening_shortlist code hash does not match current source.')
    confirmed = _confirmed_methods(shortlist)
    rows = []
    for arch in confirmed:
        for seed in CONFIRMATION_SEEDS:
            _aligned, scores = verify_method_artifacts(
                session, arch, seed, frame, smoke=False
            )
            rows.append(_score_row(arch, seed, False, scores))
    table = pd.DataFrame(rows, columns=list(TABLE_COLUMNS))
    stored_table_path = session / CONFIRMATION_CSV
    if not stored_table_path.is_file():
        raise ValueError('Need a verified single_confirmation_comparison table.')
    stored_table = read_csv(stored_table_path)
    if not _tables_match(stored_table[list(TABLE_COLUMNS)], table):
        raise ValueError('Confirmation table does not match verified OOF.')
    mean_f1 = _mean_f1_from_table(table, confirmed)
    winner = choose_confirmed_winner(mean_f1, confirmed)
    stored = read_json(path)
    extra = [path for arch in confirmed for seed in CONFIRMATION_SEEDS
             for path in (_oof_path(session, arch, seed), _models_path(session, arch, seed))]
    expected = _selection_record(shortlist, confirmed, winner, mean_f1, session, extra)
    if stored != expected:
        raise ValueError('single_selection does not match verified confirmation.')
    _assert_bound_hashes(stored, session)
    return stored, table


def _blend_rows(frame, winner, native_by_seed, winner_by_seed):
    rows = []
    seed_gains = []
    for seed in CONFIRMATION_SEEDS:
        winner_oof = winner_by_seed[seed]
        native_oof = native_by_seed[seed]
        blended = blend(winner_oof, native_oof, frame)
        winner_scores = metric(winner_oof.fake_position, winner_oof.p)
        native_scores = metric(native_oof.fake_position, native_oof.p)
        blend_scores = metric(blended.fake_position, blended.p)
        rescued, broken = _error_groups(frame.fake_position, winner_oof.p, blended.p)
        seed_gains.append(float(blend_scores['macro_f1'] - winner_scores['macro_f1']))
        rows.append(_score_row(winner, seed, False, winner_scores))
        rows.append(_score_row('native', seed, False, native_scores))
        rows.append(_score_row(
            BLEND_METHOD, seed, False, blend_scores,
            extra={'rescued': rescued, 'broken': broken},
        ))
    columns = list(TABLE_COLUMNS) + ['rescued', 'broken']
    table = pd.DataFrame(rows)
    for col in ('rescued', 'broken'):
        if col not in table.columns:
            table[col] = np.nan
    return table[columns], seed_gains


def run_selected_blend_suite(frame, root, session, smoke=False):
    """50/50 blend of the confirmed winner and native. Never trains."""
    session = load_session(session, data_root=root)
    if smoke:
        return _empty_table(('rescued', 'broken')), None
    _require_development_frame(frame, smoke=False)
    selection, _confirmation = _load_verified_selection(session, frame)
    winner = selection['method']
    native_by_seed = {}
    winner_by_seed = {}
    for seed in CONFIRMATION_SEEDS:
        native_by_seed[seed], _ = verify_method_artifacts(
            session, 'native', seed, frame, smoke=False
        )
        winner_by_seed[seed], _ = verify_method_artifacts(
            session, winner, seed, frame, smoke=False
        )
    if winner == 'native':
        rows = [_score_row('native', seed, False, metric(native_by_seed[seed].fake_position,
                                                         native_by_seed[seed].p))
                for seed in CONFIRMATION_SEEDS]
        table = pd.DataFrame(rows, columns=list(TABLE_COLUMNS))
        table['rescued'] = np.nan
        table['broken'] = np.nan
        seed_gains = None
    else:
        table, seed_gains = _blend_rows(frame, winner, native_by_seed, winner_by_seed)
    blend_choice = decide_blend(winner, seed_gains if seed_gains is not None else (), threshold=MINIMUM_GAIN)
    _write_csv_immutable(session / BLEND_CSV, table)
    decision = _decision_record(selection, blend_choice, seed_gains, session)
    stored = _write_json_immutable(session / DECISION_JSON, decision)
    return table, _public_record(stored, DECISION_FIELDS)


def verify_submission_decision(session, frame=None):
    """Recompute the blend decision from current artifacts. Never trains."""
    session = Path(session)
    if frame is None:
        frame = read_csv(session / 'development.csv')
    _require_development_frame(frame, smoke=False)
    selection, _table = _load_verified_selection(session, frame)
    winner = selection['method']
    path = session / DECISION_JSON
    if not path.is_file():
        raise ValueError('Missing submission_decision.json.')
    stored = read_json(path)
    native_by_seed = {}
    winner_by_seed = {}
    for seed in CONFIRMATION_SEEDS:
        native_by_seed[seed], _ = verify_method_artifacts(
            session, 'native', seed, frame, smoke=False
        )
        winner_by_seed[seed], _ = verify_method_artifacts(
            session, winner, seed, frame, smoke=False
        )
    if winner == 'native':
        seed_gains = None
        rows = [_score_row('native', seed, False, metric(native_by_seed[seed].fake_position,
                                                         native_by_seed[seed].p))
                for seed in CONFIRMATION_SEEDS]
        table = pd.DataFrame(rows, columns=list(TABLE_COLUMNS))
        table['rescued'] = np.nan
        table['broken'] = np.nan
    else:
        table, seed_gains = _blend_rows(frame, winner, native_by_seed, winner_by_seed)
    blend_path = session / BLEND_CSV
    if not blend_path.is_file():
        raise ValueError('Missing selected_blend_comparison.csv.')
    stored_table = read_csv(blend_path)
    if not _tables_match(stored_table, table):
        raise ValueError('selected_blend_comparison does not match verified OOF blend.')
    blend_choice = decide_blend(winner, seed_gains if seed_gains is not None else (), threshold=MINIMUM_GAIN)
    expected = _decision_record(selection, blend_choice, seed_gains, session)
    if stored != expected:
        raise ValueError('submission_decision does not match verified artifacts.')
    _assert_bound_hashes(stored, session)
    return stored
