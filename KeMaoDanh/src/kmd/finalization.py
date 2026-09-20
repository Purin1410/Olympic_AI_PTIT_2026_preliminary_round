"""Freeze development choices, score test predictions, and refit without test labels.

These entrypoints are separate from fold training. No private label is read by
selection, prediction, or refitting. A refit has no validation checkpoint search.
"""
from dataclasses import asdict, replace
from pathlib import Path
import math
import time

import numpy as np
import pandas as pd

from .core import read_csv, read_json, write_json, sha256, metric, split_fold
from .pipeline import (load_session, load_pairs, aligned_predictions, current_code_hashes,
                       infer_lr, infer_cnn, check_test_separation, extract_pair_features,
                       validate_fold_checkpoint, inventory)
from .presets import get_preset_config

PRIMARY_SEED = 20260917
MAIN_METHODS = [
    'lr_full32', 'frozen', 'center60', 'center60_cap48', 'dense288',
    'center72', 'b0_224', 'b2_224', 'b2', 'native', 'resampled',
    'blend', 'blend_selected_native',
]
EXTENSION_METHODS = [
    'lr_filesize_only', 'lr_color_only', 'lr_texture_only', 'lr_center_border_only',
    'lr_drop_filesize', 'lr_drop_color', 'lr_drop_texture', 'lr_drop_center_border',
    'frozen_lr', 'partial_native', 'partial_resampled', 'top2',
    'resnet18_rgb', 'resnet18_gaussian', 'resnet18_npr',
    'loss_image', 'loss_pair', 'loss_mixed', 'repair',
    'data_25', 'data_50', 'data_100', 'b2_resize',
]
REFIT_METHODS = {
    'lr_full32', 'b2', 'center60', 'dense288', 'native',
    'b0_224', 'b2_224', 'center72', 'center60_cap48',
}


def select_candidate(table, threshold=0.005):
    """Propose a method from complete three-seed DEVELOPMENT results only.

    Prefer the highest mean single-model F1; ties use alphabetical method order.
    Blend must exceed that single model by the declared extra-compute margin.
    This selection is distinct from the existing B2-specific blend gate.
    """
    expected = {'center60', 'dense288', 'b2', 'native', 'blend'}
    seeds = {20260917, 20260918, 20260919}
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError('Invalid minimum improvement.')
    if set(table.arch) != expected or len(table) != 15 or table[['arch', 'seed']].duplicated().any():
        raise ValueError('Need five candidates, each measured at three distinct seeds.')
    if 'smoke' not in table or not table.smoke.eq(False).all():
        raise ValueError('Selection requires full-budget results, not smoke.')
    if not all(set(g.seed) == seeds for _, g in table.groupby('arch')):
        raise ValueError('Incomplete seed coverage.')
    if not np.isfinite(table.macro_f1).all() or not table.macro_f1.between(0, 1).all():
        raise ValueError('Invalid F1 values.')
    means = table.groupby('arch').macro_f1.mean()
    single = sorted(expected - {'blend'}, key=lambda a: (-means[a], a))[0]
    delta = float(means['blend'] - means[single])
    chosen = 'blend' if delta >= threshold else single
    return {'method': chosen, 'best_single': single, 'mean_macro_f1': means.to_dict(),
            'blend_delta_vs_best_single': delta, 'minimum_gain': float(threshold),
            'seed_for_submission': PRIMARY_SEED,
            'rule': 'highest_mean_single; blend_if_gain_at_least_margin; alphabetical_ties',
            'partition': 'development', 'private_used': False}


def _dynamic_blend_recipe(session):
    """Read the recorded winner+native recipe, never a live mutable config.

    During freeze the recipe comes from verified submission_decision.json.
    After freeze it comes from the immutable review_freeze record.
    """
    session = Path(session)
    if (session / 'review_freeze.json').is_file():
        decision = verify_freeze(session)['selection']
    else:
        from .decision_flow import verify_submission_decision
        decision = verify_submission_decision(session)
    winner = decision.get('best_single')
    if (decision.get('protocol') != 'backbone_flow_v1'
            or winner not in REFIT_METHODS - {'lr_full32', 'native'}
            or decision.get('blend_status') != 'tested'
            or decision.get('components') != [winner, 'native']
            or decision.get('weights') != [0.5, 0.5]):
        raise ValueError('Dynamic blend recipe must match the verified winner+native decision.')
    return decision


def _paths(session, method, seed=PRIMARY_SEED):
    if method == 'blend':
        return _paths(session, 'b2', seed) + _paths(session, 'native', seed)
    if method == 'blend_selected_native':
        recipe = _dynamic_blend_recipe(session)
        components = list(recipe.get('components') or [])
        if len(components) != 2:
            raise ValueError('Dynamic blend requires a recorded two-component recipe.')
        paths = []
        for component in components:
            paths.extend(_paths(session, component, seed))
        return paths
    if method.startswith('lr_'):
        variant = method[3:]
        job = read_json(session / f'lr_{variant}_job.json')
        if job.get('code_sha256') != current_code_hashes():
            raise ValueError('LR source mismatch.')
        for fold, name in enumerate(job['model_files']):
            if sha256(session/name) != job['model_sha256'][f'fold{fold}']:
                raise ValueError('LR checkpoint changed.')
        if sha256(session/job['oof_file']) != job['oof_sha256']:
            raise ValueError('LR OOF changed.')
        return [session / job['oof_file'], session / f'lr_{variant}_job.json',
                session / f'lr_{variant}_feature_names.json',
                *[session / f for f in job['model_files']]]
    index = session / f'{method}_s{seed}_models.json'
    records = read_json(index)
    if sorted(r['fold'] for r in records) != [0, 1, 2]:
        raise ValueError(f'Incomplete folds for {method}')
    paths = [index, session / f'{method}_s{seed}_oof.csv']
    for record in records:
        checkpoint = session / record['checkpoint']
        if sha256(checkpoint) != record['sha256']:
            raise ValueError(f'Checkpoint changed: {checkpoint}')
        job = read_json(checkpoint.parent / 'job.json')
        if job.get('is_smoke') is not False or job.get('config', {}).get('seed') != seed:
            raise ValueError('Need full-budget checkpoints of the declared seed.')
        frame = read_csv(session/'development.csv')
        tr, va = split_fold(frame, record['fold'])
        c = get_preset_config(method, seed=seed, fold=record['fold'])
        from .subsets import sample_train_subset
        tr = sample_train_subset(tr, c.train_fraction, c.fold)
        if method != 'frozen_lr':
            ok, reason = validate_fold_checkpoint(checkpoint.parent, c, tr, va, session)
            if not ok: raise ValueError(reason)
        elif (job.get('train_ids') != tr.pair_id.tolist()
              or job.get('validation_ids') != va.pair_id.tolist()
              or job.get('code_sha256') != current_code_hashes()
              or job.get('config') != asdict(c)
              or job.get('development_sha256') != sha256(checkpoint.parent/'development.csv')
              or read_json(checkpoint.parent/'metrics.json').get('status') != 'passed'):
            raise ValueError('Frozen embedding fold contract mismatch.')
        paths.extend([checkpoint, checkpoint.parent/'job.json', checkpoint.parent/'metrics.json',
                      checkpoint.parent/'development.csv'])
    return paths


def method_oof(session, method):
    session = Path(session)
    frame = read_csv(session/'development.csv')
    if method == 'blend':
        a, b = method_oof(session, 'b2'), method_oof(session, 'native')
        out = frame.copy(); out['p'] = (a.p.to_numpy() + b.p.to_numpy()) / 2
        return out
    if method == 'blend_selected_native':
        recipe = _dynamic_blend_recipe(session)
        components = list(recipe.get('components') or [])
        weights = list(recipe.get('weights') or [])
        if len(components) != 2 or len(weights) != 2:
            raise ValueError('Dynamic blend requires a recorded two-component recipe.')
        parts = [method_oof(session, component) for component in components]
        out = frame.copy()
        out['p'] = float(weights[0]) * parts[0].p.to_numpy() + float(weights[1]) * parts[1].p.to_numpy()
        return out
    name = f'{method}_oof.csv' if method.startswith('lr_') else f'{method}_s{PRIMARY_SEED}_oof.csv'
    return aligned_predictions(read_csv(session/name), frame)


def freeze_review(session, methods, selection):
    """Write an immutable comparison list before creating/scoring private outputs."""
    session = load_session(session)
    if len(methods) != len(set(methods)) or not methods:
        raise ValueError('Methods must be unique and nonempty.')
    if selection.get('private_used') is not False or selection.get('method') not in methods:
        raise ValueError('Selection must come from development and be in the frozen list.')
    allowed = set(MAIN_METHODS + EXTENSION_METHODS)
    if not set(methods) <= allowed:
        raise ValueError('Unknown method in comparison list.')
    paths = [session/'development.csv', session/'session.json', session/'input_images.csv']
    if selection.get('protocol') == 'backbone_flow_v1':
        from .decision_flow import BACKBONE_FLOW_FILES, DECISION_FIELDS, verify_submission_decision
        frame = read_csv(session/'development.csv')
        verified = verify_submission_decision(session, frame)
        if any(selection.get(k) != verified.get(k) for k in DECISION_FIELDS):
            raise ValueError('Selection differs from the verified submission decision.')
        for name in BACKBONE_FLOW_FILES:
            path = session / name
            if path.is_file():
                paths.append(path)
        for method in verified.get('confirmed_methods') or []:
            for seed in verified.get('seeds') or (20260917, 20260918, 20260919):
                paths.extend(_paths(session, method, seed))
        if verified.get('blend_status') == 'tested':
            for seed in verified.get('seeds') or (20260917, 20260918, 20260919):
                for component in verified.get('components') or []:
                    paths.extend(_paths(session, component, seed))
        for method in methods:
            method_oof(session, method)
            paths.extend(_paths(session, method))
    else:
        for method in methods:
            method_oof(session, method)  # validates complete coverage and label alignment
            paths.extend(_paths(session, method))
        candidate_table = session/'candidate_seeds_comparison.csv'
        if selection['method'] != 'lr_full32':
            table = read_csv(candidate_table)
            verified_selection = select_candidate(table, threshold=selection['minimum_gain'])
            if any(selection.get(k) != v for k,v in verified_selection.items()):
                raise ValueError('Selection differs from the development table.')
            paths.append(candidate_table)
            frame = read_csv(session/'development.csv')
            for seed in (20260917, 20260918, 20260919):
                by_method = {}
                for method in ('center60', 'dense288', 'b2', 'native'):
                    paths.extend(_paths(session, method, seed))
                    by_method[method] = aligned_predictions(
                        read_csv(session/f'{method}_s{seed}_oof.csv'), frame).p.to_numpy()
                by_method['blend'] = (by_method['b2'] + by_method['native'])/2
                for method, probabilities in by_method.items():
                    observed = metric(frame.fake_position, probabilities)['macro_f1']
                    declared = float(table.loc[(table.arch == method) & (table.seed == seed), 'macro_f1'].iloc[0])
                    if abs(observed - declared) > 1e-12:
                        raise ValueError('Candidate table does not match verified OOF.')
    record = {'methods': list(methods), 'selection': selection,
              'code_sha256': current_code_hashes(), 'strategy': 'fold_ensemble',
              'seed': PRIMARY_SEED, 'private_labels_read': False,
              'files': {str(p.relative_to(session)): sha256(p) for p in paths}}
    target = session/'review_freeze.json'
    if target.exists():
        if read_json(target) != record:
            raise ValueError('Frozen comparison changed. Keep prior evidence; use a new session.')
    else:
        write_json(target, record)
    return record


def verify_freeze(session):
    session = load_session(session)
    frozen = read_json(session/'review_freeze.json')
    if frozen['code_sha256'] != current_code_hashes():
        raise ValueError('Source changed after freeze.')
    for rel, digest in frozen['files'].items():
        if sha256(session/rel) != digest:
            raise ValueError(f'Frozen artifact changed: {rel}')
    if frozen.get('selection', {}).get('protocol') == 'backbone_flow_v1':
        from .decision_flow import DECISION_FIELDS
        decision = read_json(session/'submission_decision.json')
        if any(frozen['selection'].get(k) != decision.get(k) for k in DECISION_FIELDS):
            raise ValueError('Frozen recipe differs from submission_decision.')
    return frozen


def infer_method(method, frame, root, session, device='cuda'):
    """Predict without reading any test labels. Frozen embedding uses its own adapter."""
    session = load_session(session)
    check_test_separation(frame, root, session)
    if 'fake_position' in frame or 'y' in frame:
        raise ValueError('Inference expects an unlabeled manifest.')
    if method.startswith('lr_'):
        return infer_lr(frame, root, session, variant=method[3:])
    if method == 'blend':
        a = infer_method('b2', frame, root, session, device)
        b = infer_method('native', frame, root, session, device)
        result = frame.copy(); result['p'] = (a.p.to_numpy()+b.p.to_numpy())/2
        return result
    if method == 'blend_selected_native':
        recipe = _dynamic_blend_recipe(session)
        components = list(recipe.get('components') or [])
        weights = list(recipe.get('weights') or [])
        if len(components) != 2 or len(weights) != 2:
            raise ValueError('Dynamic blend requires a recorded two-component recipe.')
        parts = [infer_method(component, frame, root, session, device) for component in components]
        result = frame.copy()
        result['p'] = float(weights[0]) * parts[0].p.to_numpy() + float(weights[1]) * parts[1].p.to_numpy()
        return result
    if method != 'frozen_lr':
        return infer_cnn(method, frame, root, session, device=device, seed=PRIMARY_SEED, smoke=False)
    _paths(session, 'frozen_lr')
    import joblib
    import torch
    from .models import model_for, embedding_pairs
    c = get_preset_config('frozen_lr', seed=PRIMARY_SEED)
    model, weight_hash = model_for(c, device=device, pretrained=True, embedding=True)
    emb = embedding_pairs(model, frame, c, Path(root), device=device)
    del model
    probabilities = []
    for record in read_json(session/f'frozen_lr_s{PRIMARY_SEED}_models.json'):
        p = session/record['checkpoint']; job = read_json(p.parent/'job.json')
        if (sha256(p) != record['sha256'] or sha256(p) != job['model_sha256']
                or weight_hash != job['pretrained_weight_sha256']
                or job['code_sha256'] != current_code_hashes() or job['is_smoke']):
            raise ValueError('Frozen embedding artifact failed verification.')
        fitted = joblib.load(p)
        q = fitted['clf'].predict_proba(fitted['pca'].transform(
            fitted['scaler'].transform(emb.reshape(-1, emb.shape[-1]))))[:, 1]
        q = np.clip(q.reshape(len(frame), 2), 1e-7, 1-1e-7)
        scores = np.log(q/(1-q))
        probabilities.append(1/(1+np.exp(-np.clip(scores[:, 1]-scores[:, 0], -60, 60))))
    if len(probabilities) != 3:
        raise ValueError('Frozen embedding requires three folds.')
    if device == 'cuda': torch.cuda.empty_cache()
    result = frame.copy(); result['p'] = np.mean(probabilities, axis=0)
    return result


def predict_frozen_comparison(session, test_root, output_dir, device='cuda'):
    """Save every declared prediction BEFORE a separate label-scoring step."""
    session = Path(session); frozen = verify_freeze(session)
    frame = load_pairs(test_root, labeled=False)
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    manifest = {'freeze_sha256': sha256(session/'review_freeze.json'),
                'test_pairs_sha256': sha256(Path(test_root)/'pairs.csv'),
                'test_images': {rel: sha256(Path(test_root)/rel)
                                for rel in sorted(set(frame.image_0)|set(frame.image_1))},
                'predictions': {}}
    for method in frozen['methods']:
        pred = aligned_predictions(infer_method(method, frame, test_root, session, device), frame)
        target = out/f'{method}.csv'
        pred[['pair_id', 'p']].to_csv(target, index=False)
        manifest['predictions'][method] = sha256(target)
    write_json(out/'prediction_manifest.json', manifest)
    return manifest


def score_private(session, prediction_dir, labels_csv, label_source):
    """Only this function reads private labels; source description is mandatory."""
    if not str(label_source).strip():
        raise ValueError('Describe where the private labels came from before scoring.')
    session = Path(session); frozen = verify_freeze(session); out = Path(prediction_dir)
    saved = read_json(out/'prediction_manifest.json')
    if saved['freeze_sha256'] != sha256(session/'review_freeze.json'):
        raise ValueError('Predictions do not belong to the frozen review.')
    if set(saved['predictions']) != set(frozen['methods']):
        raise ValueError('All frozen predictions must exist before labels are opened.')
    for method, digest in saved['predictions'].items():
        if sha256(out/f'{method}.csv') != digest:
            raise ValueError('Prediction file changed before scoring.')
    labels = read_csv(labels_csv)
    if ('fake_position' not in labels or not labels.pair_id.is_unique
            or not labels.fake_position.isin([0, 1]).all()):
        raise ValueError('Labels need unique pair_id and binary fake_position.')
    rows = []
    for method in frozen['methods']:
        p = read_csv(out/f'{method}.csv')
        if not p.pair_id.is_unique or set(p.pair_id) != set(labels.pair_id):
            raise ValueError('Private prediction/label IDs differ.')
        p = p.set_index('pair_id').loc[labels.pair_id]
        test = metric(labels.fake_position, p.p)
        oof = method_oof(session, method)
        dev = metric(oof.fake_position, oof.p)
        rows.append({'method': method, 'development_f1': dev['macro_f1'],
                     'private_f1': test['macro_f1'], 'private_errors': test['errors'],
                     'private_n': test['n'], 'protocol': '3-fold ensemble, seed 20260917'})
    table = pd.DataFrame(rows); table.to_csv(out/'private_comparison.csv', index=False)
    write_json(out/'label_provenance.json', {'source': str(label_source),
               'labels_sha256': sha256(labels_csv), 'freeze_sha256': saved['freeze_sha256'],
               'label_column': 'fake_position', 'official_status': 'not_inferred_from_filename'})
    return table


def refit_config(method, session):
    """Derive a fixed budget from verified primary-seed fold checkpoints."""
    c = get_preset_config(method, seed=PRIMARY_SEED)
    if c.kind != 'deep' or c.train_fraction != 1:
        raise ValueError('Refit entrypoint supports full-data deep candidates only.')
    session = load_session(session); frame = read_csv(session/'development.csv')
    records = read_json(session/f'{method}_s{PRIMARY_SEED}_models.json')
    if sorted(r['fold'] for r in records) != [0, 1, 2]:
        raise ValueError('Need all three verified folds for the refit budget.')
    best_epochs = []
    for r in records:
        folder = (session/r['checkpoint']).parent
        tr, va = split_fold(frame, r['fold'])
        cf = get_preset_config(method, seed=PRIMARY_SEED, fold=r['fold'])
        ok, reason = validate_fold_checkpoint(folder, cf, tr, va, session)
        if not ok: raise ValueError(reason)
        best_epochs.append(int(read_json(folder/'metrics.json')['best_epoch']))
    epochs = c.epochs if c.fixed_epochs or c.fixed_updates else int(np.median(best_epochs))
    return replace(c, epochs=epochs, fixed_epochs=True), best_epochs


def refit_all(method, train_root, session):
    """Train from pretrained initialization on all1000 pairs, no validation loop.

    Saves only the terminal model at the already-fixed budget. Does not use test
    images, private labels, or held-out scores. Incomplete runs require manual review.
    """
    session = load_session(session, data_root=train_root)
    frozen = verify_freeze(session)
    selected = frozen['selection']['method']
    recipe_components = (list(_dynamic_blend_recipe(session)['components'])
                         if selected == 'blend_selected_native' else [])
    if selected == 'blend':
        allowed_branches = {'b2', 'native'}
    elif selected == 'blend_selected_native':
        allowed_branches = set(recipe_components)
    else:
        allowed_branches = {selected}
    if method != selected and method not in allowed_branches:
        raise ValueError('Refit method must match the frozen development choice.')
    if frozen['selection'].get('submission_strategy') != 'refit_all':
        raise ValueError('Refit strategy must be declared before private evaluation.')
    if method == 'blend':
        return {m: refit_all(m, train_root, session) for m in ('b2', 'native')}
    if method == 'blend_selected_native':
        if len(recipe_components) != 2:
            raise ValueError('Frozen dynamic blend recipe is incomplete.')
        return {m: refit_all(m, train_root, session) for m in recipe_components}
    if method not in REFIT_METHODS: raise ValueError('Unsupported final refit candidate.')
    import joblib
    import torch
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from .data import Pairs, make_loader, fit_normalization
    from .models import model_for, train_mode, head_name
    from .training import train_step, atomic_save
    frame = load_pairs(train_root, labeled=True)
    if len(frame) != 1000: raise ValueError('Refit expects all1000 labeled pairs.')
    root = Path(train_root)
    images = inventory(frame.assign(inner_fold=-1), root)  # One training set, no CV split.
    c, epochs = (None, None) if method == 'lr_full32' else refit_config(method, session)
    contract = {'method': method, 'strategy': 'refit_all', 'train_pairs': 1000,
                'train_ids': frame.pair_id.tolist(), 'validation_ids': [],
                'train_manifest_sha256': sha256(root/'pairs.csv'),
                'image_sha256': dict(zip(images.relpath, images.sha256)),
                'code_sha256': current_code_hashes(), 'source_best_epochs': epochs,
                'config': asdict(c) if c else None, 'previous_holdout_used_for_training': True}
    folder = session/'refit_all'/method
    if folder.exists():
        result = read_json(folder/'result.json')
        if read_json(folder/'contract.json') != contract or result['status'] != 'complete':
            raise ValueError('Refit artifact does not match current request.')
        if sha256(folder/result['checkpoint']) != result['checkpoint_sha256']:
            raise ValueError('Refit checkpoint changed.')
        return folder
    folder.mkdir(parents=True); write_json(folder/'contract.json', contract)
    started = time.monotonic()
    try:
        if c is None:
            x, _ = extract_pair_features(frame, root)
            fitted = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(
                C=.1, fit_intercept=False, max_iter=5000, random_state=PRIMARY_SEED))
            fitted.fit(x, frame.fake_position)
            target = folder/'terminal.joblib'; joblib.dump(fitted, target)
            updates = None
        else:
            if not torch.cuda.is_available(): raise RuntimeError('Refit CNN requires CUDA.')
            model, weight_hash = model_for(c, device='cuda', pretrained=True)
            norm = fit_normalization(frame, root, c) if c.normalization == 'train' else None
            head = head_name(c)
            optimizer = torch.optim.AdamW([
                {'params': [p for n,p in model.named_parameters() if not n.startswith(head)], 'lr': c.backbone_lr},
                {'params': [p for n,p in model.named_parameters() if n.startswith(head)], 'lr': c.head_lr},
            ], weight_decay=c.weight_decay)
            bn = {n:b.detach().cpu().clone() for n,b in model.named_buffers() if 'running_' in n or 'num_batches_tracked' in n}
            history = []; updates = 0
            cap = math.ceil(c.fixed_updates/math.ceil(len(frame)/c.effective_batch)) if c.fixed_updates else c.epochs
            for epoch in range(1, cap+1):
                train_mode(model, c, epoch)
                dataset = Pairs(frame, root, c, True, epoch, norm)
                loader = make_loader(dataset, True)
                optimizer.zero_grad(set_to_none=True)
                within = seen = 0; total = 0.; target_batch = min(c.effective_batch, len(frame))
                for x,y in loader:
                    update = within+len(y) == target_batch
                    loss,_ = train_step(model,x,y,optimizer,c,target_batch=target_batch,perform_update=update)
                    total += loss*len(y); seen += len(y); within += len(y)
                    if update:
                        updates += 1; within = 0
                        target_batch = min(c.effective_batch, len(frame)-seen)
                        if c.fixed_updates and updates >= c.fixed_updates: break
                if not np.isfinite(total) or within: raise ValueError('Invalid refit accumulation/loss.')
                history.append({'epoch':epoch,'train_loss':total/seen,'optimizer_steps':updates})
                pd.DataFrame(history).to_csv(folder/'history.csv', index=False)
                print(f'Refit {method}: epoch {epoch}/{cap}, train loss {total/seen:.5f}', flush=True)
                del loader, dataset
                if c.fixed_updates and updates >= c.fixed_updates: break
            for n,b in bn.items():
                if not torch.equal(dict(model.named_buffers())[n].cpu(),b): raise ValueError('BatchNorm changed.')
            target = folder/'terminal.pt'
            atomic_save(target, {'state_dict': {n:v.detach().cpu() for n,v in model.state_dict().items()},
                                'config': asdict(c), 'norm':norm, 'pretrained_sha256':weight_hash})
            del model, optimizer; torch.cuda.empty_cache()
        write_json(folder/'result.json', {'status':'complete','checkpoint':target.name,
                   'checkpoint_sha256':sha256(target),'optimizer_steps':updates,
                   'elapsed_seconds':time.monotonic()-started,'validation_used':False})
    except BaseException as exc:
        write_json(folder/'result.json', {'status':'failed','error':str(exc)})
        raise
    return folder


def infer_refit(method, frame, test_root, session, device='cuda'):
    session = load_session(session)
    if method == 'blend':
        a = infer_refit('b2',frame,test_root,session,device)
        b = infer_refit('native',frame,test_root,session,device)
        result = frame.copy(); result['p'] = (a.p.to_numpy()+b.p.to_numpy())/2
        return result
    if method == 'blend_selected_native':
        recipe = _dynamic_blend_recipe(session)
        components = list(recipe.get('components') or [])
        weights = list(recipe.get('weights') or [])
        if len(components) != 2 or len(weights) != 2:
            raise ValueError('Dynamic blend requires a recorded two-component recipe.')
        parts = [infer_refit(component, frame, test_root, session, device) for component in components]
        result = frame.copy()
        result['p'] = float(weights[0]) * parts[0].p.to_numpy() + float(weights[1]) * parts[1].p.to_numpy()
        return result
    if method not in REFIT_METHODS:
        raise ValueError('Unknown refit model.')
    import joblib
    import torch
    from .config import Config
    from .models import model_for, predict_pairs
    if 'fake_position' in frame or 'y' in frame: raise ValueError('Use unlabeled test manifest.')
    folder = session/'refit_all'/method
    contract, result = read_json(folder/'contract.json'), read_json(folder/'result.json')
    if contract['code_sha256'] != current_code_hashes() or result['status'] != 'complete':
        raise ValueError('Refit source/status mismatch.')
    checkpoint = folder/result['checkpoint']
    if sha256(checkpoint) != result['checkpoint_sha256']: raise ValueError('Refit hash mismatch.')
    # Check against ALL1000 training pairs, including the former200holdout.
    train_hashes = set(contract['image_sha256'].values())
    if any(sha256(Path(test_root)/p) in train_hashes for p in set(frame.image_0)|set(frame.image_1)):
        raise ValueError('Test image duplicates a refit training image.')
    if method == 'lr_full32':
        x,_ = extract_pair_features(frame, test_root)
        result = frame.copy(); result['p'] = joblib.load(checkpoint).predict_proba(x)[:,1]
        return result
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    c = Config(**state['config']); model,_ = model_for(c,device=device,pretrained=False)
    model.load_state_dict(state['state_dict'])
    pred = predict_pairs(model,frame,c,Path(test_root),state.get('norm'),device=device)
    del model
    if device == 'cuda': torch.cuda.empty_cache()
    return aligned_predictions(pred,frame)
