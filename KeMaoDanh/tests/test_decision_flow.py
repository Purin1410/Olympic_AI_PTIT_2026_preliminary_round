"""No-training checks for dynamic backbone screening, confirmation, and blend."""
import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

try:
    import numpy as np
    import pandas as pd
    from kmd.core import metric, sha256, write_json
    from kmd.decision_flow import (
        ANCHORS, BLEND_METHOD, CHALLENGERS, CONFIRMATION_SEEDS, DECISION_FIELDS,
        MINIMUM_GAIN, PRIMARY_SEED, PROTOCOL, SCREENING_METHODS, SELECTION_FIELDS,
        TABLE_COLUMNS, choose_confirmed_winner, comparison_methods_for,
        decide_blend, run_backbone_screening_suite, run_selected_blend_suite,
        run_single_confirmation_suite, shortlist_challengers,
        verify_method_artifacts,
    )
    from kmd.presets import canonical_preset_name, get_preset_config
    HAS_DEPS = True
    MISSING_ERR = ''
except ImportError as exc:
    HAS_DEPS = False
    MISSING_ERR = str(exc)




def _frame(n=800):
    return pd.DataFrame({
        'pair_id': [f'{i:05d}' for i in range(n)],
        'image_0': [f'{i:05d}a.jpg' for i in range(n)],
        'image_1': [f'{i:05d}b.jpg' for i in range(n)],
        'inner_fold': [i % 3 for i in range(n)],
        'fake_position': [i % 2 for i in range(n)],
    })


def _probs(frame, errors=0, correct=0.91, wrong=0.09):
    y = frame.fake_position.to_numpy()
    p = np.where(y == 1, correct, 1.0 - correct)
    if errors:
        p[:errors] = np.where(y[:errors] == 1, wrong, 1.0 - wrong)
    return p


def _config_core(name):
    data = asdict(get_preset_config(name))
    data.pop('seed')
    data.pop('fold')
    return data


def _diff(a, b):
    left, right = _config_core(a), _config_core(b)
    return {key for key in left if left[key] != right[key]}


def _write_cnn_artifacts(session, method, seed, frame, p, smoke=False, folds=(0, 1, 2),
                         digest=None):
    oof = frame.copy()
    oof['p'] = p
    prefix = f'{method}_smoke_s{seed}' if smoke else f'{method}_s{seed}'
    oof.to_csv(session / f'{prefix}_oof.csv', index=False)
    records = []
    for fold in folds:
        folder = session / f'{prefix}_f{fold}'
        folder.mkdir(parents=True, exist_ok=True)
        checkpoint = folder / 'best.pt'
        checkpoint.write_bytes(b'ckpt')
        oof.loc[oof.inner_fold == fold].to_csv(folder / 'development.csv', index=False)
        write_json(folder / 'job.json', {
            'config': {'seed': int(seed)},
            'purpose': 'smoke' if smoke else 'teaching_full_fold',
        })
        records.append({
            'fold': fold,
            'checkpoint': str(checkpoint.relative_to(session)),
            'sha256': digest if digest is not None else sha256(checkpoint),
        })
    write_json(session / f'{prefix}_models.json', records)
    return oof


class DecisionFlowTests(unittest.TestCase):
    def setUp(self):
        if not HAS_DEPS:
            raise unittest.SkipTest(f'Dependencies unavailable ({MISSING_ERR}); marked NOTRUN for root realenv')
        self.temp = tempfile.TemporaryDirectory(prefix='kmd_decision_')
        self.session = Path(self.temp.name) / 'session'
        self.session.mkdir()
        self.frame = _frame()
        self.frame.to_csv(self.session / 'development.csv', index=False)

    def tearDown(self):
        self.temp.cleanup()

    def test_original_configs_and_single_factor_deltas(self):
        cap = get_preset_config('center60_cap48')
        self.assertEqual((cap.microbatch, cap.effective_batch, cap.epochs), (8, 24, 48))
        self.assertEqual(_diff('dense288', 'center60_cap48'), {'size'})
        self.assertEqual(_diff('center72', 'center60_cap48'), {'view'})
        self.assertEqual(_diff('b0_224', 'center60_cap48'), {'backbone'})
        self.assertEqual(_diff('b2_224', 'center60_cap48'), {'backbone'})
        self.assertEqual(_diff('b2', 'dense288'), {'backbone'})
        self.assertEqual(get_preset_config('dense288').size, 288)
        self.assertEqual(get_preset_config('center72').view, 'center72')
        self.assertEqual(get_preset_config('b0_224').backbone, 'efficientnet_b0')
        self.assertEqual(get_preset_config('b2_224').backbone, 'efficientnet_b2')
        self.assertEqual(canonical_preset_name('E_center72'), 'center72')
        self.assertEqual(canonical_preset_name('E_efficientnet_b0_224'), 'b0_224')
        self.assertEqual(canonical_preset_name('E_efficientnet_b2_224'), 'b2_224')

    def test_shortlist_top2_and_alphabetical_ties(self):
        scores = {
            'dense288': 0.97, 'center72': 0.95, 'b0_224': 0.96,
            'b2_224': 0.94, 'b2': 0.973,
        }
        self.assertEqual(shortlist_challengers(scores), ['b2', 'dense288'])
        tied = dict(scores)
        tied['b2'] = 0.97
        self.assertEqual(shortlist_challengers(tied), ['b2', 'dense288'])
        tied['b0_224'] = 0.97
        self.assertEqual(shortlist_challengers(tied), ['b0_224', 'b2'])
        with self.assertRaises(ValueError):
            shortlist_challengers({'dense288': 0.9})
        with self.assertRaises(ValueError):
            shortlist_challengers({name: float('nan') for name in CHALLENGERS})

    def test_confirmation_winner_every_method_including_native(self):
        confirmed = list(CHALLENGERS[:2]) + list(ANCHORS)
        for winner in ('dense288', 'center72', 'b0_224', 'b2_224', 'b2',
                       'center60', 'center60_cap48', 'native'):
            methods = list(dict.fromkeys([winner] + confirmed))
            means = {name: 0.90 for name in methods}
            means[winner] = 0.94
            self.assertEqual(choose_confirmed_winner(means, methods), winner)
        means = {name: 0.91 for name in ('b2', 'center60', 'center60_cap48', 'dense288', 'native')}
        self.assertEqual(
            choose_confirmed_winner(means, ['b2', 'dense288', 'center60', 'center60_cap48', 'native']),
            'b2',
        )

    def test_blend_is_probability_average_not_oracle(self):
        y = np.array([1, 1, 1, 0])
        winner = np.array([0.6, 0.4, 0.05, 0.2])
        native = np.array([0.1, 0.8, 0.75, 0.2])
        blended = 0.5 * winner + 0.5 * native
        np.testing.assert_allclose(blended, [0.35, 0.6, 0.4, 0.2])
        oracle = np.where((winner >= 0.5) == y, winner, native)
        self.assertGreater(int(((oracle >= 0.5) == y).sum()), int(((blended >= 0.5) == y).sum()))
        self.assertEqual(int(((blended >= 0.5) == y).sum()), 2)
        self.assertEqual(int((((winner >= 0.5) != y) & ((blended >= 0.5) == y)).sum()), 1)
        self.assertEqual(int((((winner >= 0.5) == y) & ((blended >= 0.5) != y)).sum()), 1)

    def test_blend_threshold_unrounded_boundary(self):
        keep = decide_blend('b2', [0.004999, 0.005000, 0.004999])
        self.assertEqual(keep['method'], 'b2')
        self.assertEqual(keep['blend_status'], 'tested')
        self.assertLess(keep['blend_delta_vs_best_single'], MINIMUM_GAIN)
        promote = decide_blend('center72', [0.005001, 0.005000, 0.005000])
        self.assertEqual(promote['method'], BLEND_METHOD)
        self.assertEqual(promote['components'], ['center72', 'native'])
        self.assertEqual(promote['weights'], [0.5, 0.5])
        skipped = decide_blend('native', [0.9, 0.9, 0.9])
        self.assertEqual(skipped['method'], 'native')
        self.assertEqual(skipped['blend_status'], 'skipped_self_blend')
        self.assertEqual(skipped['components'], [])
        self.assertIsNone(skipped['blend_delta_vs_best_single'])
        self.assertEqual(comparison_methods_for('tested')[-1], BLEND_METHOD)
        self.assertNotIn(BLEND_METHOD, comparison_methods_for('skipped_self_blend'))

    def test_screening_writes_eight_primary_seed_rows(self):
        scores = {arch: metric(self.frame.fake_position, _probs(self.frame, errors=20 + i))
                  for i, arch in enumerate(SCREENING_METHODS)}

        def fake_complete(name, frame, root, session, seed, smoke):
            oof = frame.copy()
            oof['p'] = _probs(frame, errors=20)
            return oof, scores[name]

        with patch('kmd.decision_flow.load_session', return_value=self.session), \
             patch('kmd.decision_flow._complete_method', side_effect=fake_complete) as complete:
            table, predictions = run_backbone_screening_suite(
                self.frame, self.session, self.session, smoke=False
            )
        self.assertEqual(complete.call_count, len(SCREENING_METHODS))
        self.assertEqual(list(table.columns), list(TABLE_COLUMNS))
        self.assertEqual(list(table.arch), list(SCREENING_METHODS))
        self.assertTrue((table.seed == PRIMARY_SEED).all())
        self.assertEqual(set(predictions), set(SCREENING_METHODS))
        written = pd.read_csv(self.session / 'backbone_screening_comparison.csv')
        self.assertEqual(list(written.arch), list(SCREENING_METHODS))

        def fake_smoke(name, frame, root, session, seed, smoke):
            self.assertTrue(smoke)
            oof = frame.copy()
            oof['p'] = _probs(frame, errors=20)
            return oof, scores[name]

        with patch('kmd.decision_flow.load_session', return_value=self.session), \
             patch('kmd.decision_flow._complete_method', side_effect=fake_smoke):
            run_backbone_screening_suite(self.frame, self.session, self.session, smoke=True)
        self.assertTrue((self.session / 'backbone_screening_smoke_comparison.csv').is_file())

    def test_confirmation_smoke_and_shortlist_before_extra_seeds(self):
        with patch('kmd.decision_flow.load_session', return_value=self.session):
            table, selection = run_single_confirmation_suite(
                self.frame, self.session, self.session, smoke=True
            )
        self.assertIsNone(selection)
        self.assertTrue(table.empty)
        self.assertFalse((self.session / 'screening_shortlist.json').is_file())

        f1 = {
            'center60_cap48': 0.968, 'dense288': 0.974, 'center72': 0.955,
            'b0_224': 0.961, 'b2_224': 0.965, 'b2': 0.972,
            'center60': 0.960, 'native': 0.966,
        }
        screening_rows = []
        for arch in SCREENING_METHODS:
            screening_rows.append({
                'model': f'{arch}_s{PRIMARY_SEED}', 'arch': arch, 'seed': PRIMARY_SEED,
                'smoke': False, 'n': 800, 'macro_f1': f1[arch], 'accuracy': 0.97,
                'auc': 0.99, 'log_loss': 0.1, 'errors': 20,
            })
        screening = pd.DataFrame(screening_rows)
        screening.to_csv(self.session / 'backbone_screening_comparison.csv', index=False)
        trained = []

        def fake_verify(session, method, seed, frame, smoke=False):
            oof = frame.copy()
            oof['p'] = _probs(frame, errors=10)
            scores = dict(n=800, macro_f1=f1.get(method, 0.96) - 0.0001 * (int(seed) - PRIMARY_SEED),
                          accuracy=0.97, auc=0.99, log_loss=0.1, errors=10)
            return oof, scores

        def fake_complete(name, frame, root, session, seed, smoke):
            self.assertTrue((Path(session) / 'screening_shortlist.json').is_file())
            trained.append((name, int(seed), smoke))
            return fake_verify(session, name, seed, frame, smoke)

        with patch('kmd.decision_flow.load_session', return_value=self.session), \
             patch('kmd.decision_flow._load_verified_screening', return_value=(screening, {})), \
             patch('kmd.decision_flow.verify_method_artifacts', side_effect=fake_verify), \
             patch('kmd.decision_flow._complete_method', side_effect=fake_complete), \
             patch('kmd.decision_flow.current_code_hashes', return_value={'k': 'hash'}):
            table, selection = run_single_confirmation_suite(
                self.frame, self.session, self.session, smoke=False
            )
        self.assertEqual(selection['protocol'], PROTOCOL)
        self.assertEqual(selection['shortlist'], ['dense288', 'b2'])
        self.assertEqual(selection['seed_for_submission'], PRIMARY_SEED)
        self.assertFalse(selection['private_used'])
        self.assertEqual(set(selection), set(SELECTION_FIELDS))
        self.assertTrue(all(seed != PRIMARY_SEED for _, seed, _ in trained))
        self.assertEqual({name for name, _, _ in trained},
                         set(selection['confirmed_methods']))
        stored = json.loads((self.session / 'screening_shortlist.json').read_text())
        self.assertEqual(stored['shortlist'], ['dense288', 'b2'])

    def test_blend_suite_does_not_train_and_handles_native_winner(self):
        selection = {
            'protocol': PROTOCOL,
            'method': 'b2',
            'confirmed_methods': ['dense288', 'b2', 'center60', 'center60_cap48', 'native'],
            'shortlist': ['dense288', 'b2'],
            'seeds': list(CONFIRMATION_SEEDS),
            'mean_macro_f1': {'b2': 0.97, 'native': 0.96},
            'seed_for_submission': PRIMARY_SEED,
            'partition': 'development',
            'private_used': False,
        }
        write_json(self.session / 'single_selection.json', selection)
        write_json(self.session / 'screening_shortlist.json', {'protocol': PROTOCOL})
        pd.DataFrame(columns=list(TABLE_COLUMNS)).to_csv(
            self.session / 'single_confirmation_comparison.csv', index=False
        )
        winner_p = {seed: _probs(self.frame, errors=30) for seed in CONFIRMATION_SEEDS}
        native_p = {seed: _probs(self.frame, errors=40) for seed in CONFIRMATION_SEEDS}
        # One pair where averaging breaks a correct winner call.
        for seed in CONFIRMATION_SEEDS:
            winner_p[seed] = winner_p[seed].copy()
            native_p[seed] = native_p[seed].copy()
            winner_p[seed][0] = 0.91 if self.frame.fake_position.iloc[0] == 1 else 0.09
            native_p[seed][0] = 0.01 if self.frame.fake_position.iloc[0] == 1 else 0.99

        def fake_verify(session, method, seed, frame, smoke=False):
            oof = frame.copy()
            oof['p'] = native_p[int(seed)] if method == 'native' else winner_p[int(seed)]
            return oof, metric(oof.fake_position, oof.p)

        with patch('kmd.decision_flow.load_session', return_value=self.session), \
             patch('kmd.decision_flow._load_verified_selection', return_value=(selection, pd.DataFrame())), \
             patch('kmd.decision_flow.verify_method_artifacts', side_effect=fake_verify), \
             patch('kmd.decision_flow._train_cnn_cv') as train, \
             patch('kmd.decision_flow.current_code_hashes', return_value={'k': 'hash'}):
            table, decision = run_selected_blend_suite(
                self.frame, self.session, self.session, smoke=False
            )
        train.assert_not_called()
        self.assertEqual(set(decision), set(DECISION_FIELDS))
        self.assertEqual(decision['best_single'], 'b2')
        self.assertEqual(decision['components'], ['b2', 'native'])
        self.assertEqual(decision['weights'], [0.5, 0.5])
        self.assertEqual(decision['blend_status'], 'tested')
        blend_rows = table[table.arch == BLEND_METHOD]
        self.assertEqual(len(blend_rows), 3)
        self.assertTrue({'rescued', 'broken'} <= set(table.columns))
        seed = PRIMARY_SEED
        expected = 0.5 * winner_p[seed] + 0.5 * native_p[seed]
        self.assertEqual(int(blend_rows.iloc[0].broken), int((((winner_p[seed] >= .5) == self.frame.fake_position) & ((expected >= .5) != self.frame.fake_position)).sum()))
        self.assertEqual(float(blend_rows.iloc[0].macro_f1), metric(self.frame.fake_position, expected)['macro_f1'])
        # An immutable decision cannot be changed inside the same session.
        # The independent native-winner scenario therefore uses a fresh session.
        self.session = Path(self.temp.name) / 'native_session'
        self.session.mkdir()
        write_json(self.session / 'single_selection.json', {'method': 'native'})
        train.reset_mock()
        native_selection = dict(selection)
        native_selection['method'] = 'native'
        with patch('kmd.decision_flow.load_session', return_value=self.session), \
             patch('kmd.decision_flow._load_verified_selection',
                   return_value=(native_selection, pd.DataFrame())), \
             patch('kmd.decision_flow.verify_method_artifacts', side_effect=fake_verify), \
             patch('kmd.decision_flow._train_cnn_cv') as train, \
             patch('kmd.decision_flow.current_code_hashes', return_value={'k': 'hash'}):
            table, decision = run_selected_blend_suite(
                self.frame, self.session, self.session, smoke=False
            )
        train.assert_not_called()
        self.assertEqual(decision['method'], 'native')
        self.assertEqual(decision['blend_status'], 'skipped_self_blend')
        self.assertEqual(decision['components'], [])
        self.assertIsNone(decision['blend_delta_vs_best_single'])
        self.assertEqual(set(table.arch.unique()), {'native'})
        self.assertNotIn(BLEND_METHOD, decision['comparison_methods'])

        with patch('kmd.decision_flow.load_session', return_value=self.session):
            smoke_table, smoke_decision = run_selected_blend_suite(
                self.frame, self.session, self.session, smoke=True
            )
        self.assertIsNone(smoke_decision)
        self.assertTrue(smoke_table.empty)

    def test_verify_rejects_incomplete_hash_and_smoke_artifacts(self):
        p = _probs(self.frame, errors=12)
        _write_cnn_artifacts(self.session, 'b2', PRIMARY_SEED, self.frame, p, folds=(0, 1))
        with self.assertRaises(ValueError):
            verify_method_artifacts(self.session, 'b2', PRIMARY_SEED, self.frame)
        self.temp.cleanup()
        self.setUp()
        _write_cnn_artifacts(self.session, 'b2', PRIMARY_SEED, self.frame, p, digest='0' * 64)
        with patch('kmd.decision_flow.validate_fold_checkpoint', return_value=(True, {})):
            with self.assertRaises(ValueError):
                verify_method_artifacts(self.session, 'b2', PRIMARY_SEED, self.frame)
        self.temp.cleanup()
        self.setUp()
        _write_cnn_artifacts(self.session, 'b2', PRIMARY_SEED, self.frame, p, smoke=True)
        with patch('kmd.decision_flow.validate_fold_checkpoint',
                   return_value=(False, 'smoke run cannot satisfy non-smoke validation')):
            with self.assertRaises(ValueError):
                verify_method_artifacts(self.session, 'b2', PRIMARY_SEED, self.frame, smoke=False)
        with self.assertRaises(ValueError):
            verify_method_artifacts(self.session, 'native', 1999, self.frame)
        with patch('kmd.decision_flow.validate_fold_checkpoint', return_value=(True, {})), \
             patch('kmd.decision_flow.get_preset_config', side_effect=get_preset_config):
            _write_cnn_artifacts(self.session, 'native', PRIMARY_SEED, self.frame, p)
            oof, scores = verify_method_artifacts(self.session, 'native', PRIMARY_SEED, self.frame)
        self.assertEqual(len(oof), 800)
        self.assertTrue(np.isfinite(scores['macro_f1']))

    def test_complete_protocol_reuse_tampering_and_native_with_synthetic_artifacts(self):
        # Only checkpoint bytes and prediction tables are fabricated. The production
        # fold-contract validator, shortlist, confirmation, blend and freeze all run.
        import hashlib
        import kmd.decision_flow as d
        import kmd.finalization as f
        from kmd.core import split_fold
        from kmd.pipeline import current_code_hashes

        def fake_train(name, frame, root, session, seed, smoke):
            errors = {'dense288': 16, 'b2': 24, 'native': 32}.get(name, 40)
            p = _probs(frame, errors=errors, correct=.8, wrong=.4)
            if name == 'native':
                # Strong complementary native predictions rescue the winner's
                # weak mistakes; weak native errors preserve winner successes.
                p = np.where(frame.fake_position == 1, .99, .01)
                p[32:64] = np.where(frame.fake_position.iloc[32:64] == 1, .4, .6)
            if session.name == 'native_winner' and name == 'native':
                p = _probs(frame, errors=0)
            oof = _write_cnn_artifacts(session, name, seed, frame, p)
            for fold in range(3):
                folder = session / f'{name}_s{seed}_f{fold}'
                tr, va = split_fold(frame, fold)
                oof[oof.inner_fold == fold].rename(columns={
                    'inner_fold': 'fold', 'fake_position': 'y'
                }).to_csv(folder / 'development.csv', index=False)
                job = {'config': asdict(get_preset_config(name, seed=seed, fold=fold)),
                       'train_ids': tr.pair_id.tolist(), 'validation_ids': va.pair_id.tolist(),
                       'code_sha256': current_code_hashes(), 'is_smoke': False,
                       'purpose': 'teaching_full_fold'}
                job['job_hash'] = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()
                write_json(folder / 'job.json', job)
                write_json(folder / 'status.json', {'status': 'complete'})
                write_json(folder / 'metrics.json', {
                    'status': 'passed', 'job_hash': job['job_hash'],
                    'checkpoint_sha256': sha256(folder/'best.pt'),
                    'reload_probability_max_abs_error': 0, 'pair_swap_error': 0,
                    'holdout_evaluated': False, 'best_epoch': 8})
            return oof

        for case, winner in [('non_b2_winner', 'dense288'), ('native_winner', 'native')]:
            session = Path(self.temp.name) / case
            session.mkdir()
            self.frame.to_csv(session/'development.csv', index=False)
            write_json(session/'session.json', {})
            (session/'input_images.csv').write_text('relpath,sha256\n')
            with patch.object(d, 'load_session', return_value=session), \
                 patch.object(d, '_train_cnn_cv', side_effect=fake_train), \
                 patch.object(f, 'load_session', return_value=session):
                screen, _ = d.run_backbone_screening_suite(self.frame, session, session)
                confirmed, selected = d.run_single_confirmation_suite(self.frame, session, session)
                self.assertEqual(selected['method'], winner)
                self.assertEqual(len(confirmed), 15)
                table, choice = d.run_selected_blend_suite(self.frame, session, session)
                self.assertEqual(choice['best_single'], winner)
                self.assertEqual(choice['method'], 'native' if winner == 'native' else d.BLEND_METHOD)
                # Idempotent rerun and verification cannot fit anything.
                with patch.object(d, '_train_cnn_cv', side_effect=AssertionError('unexpected training')):
                    table2, choice2 = d.run_selected_blend_suite(self.frame, session, session)
                    self.assertEqual(choice, choice2)
                    self.assertEqual(d.verify_submission_decision(session)['method'], choice['method'])
                frozen_choice = dict(choice, submission_strategy='fold_ensemble')
                methods = ['native'] if winner == 'native' else ['dense288', 'native', d.BLEND_METHOD]
                f.freeze_review(session, methods, frozen_choice)
                self.assertEqual(f.verify_freeze(session)['selection'], frozen_choice)
                if winner != 'native':
                    got = f.method_oof(session, d.BLEND_METHOD)
                    expected = .5 * f.method_oof(session, winner).p + .5 * f.method_oof(session, 'native').p
                    np.testing.assert_allclose(got.p, expected, rtol=0, atol=1e-12)
                # Mutating the recipe must fail both freeze and fresh verification.
                decision = json.loads((session/d.DECISION_JSON).read_text())
                decision['weights'] = [.9, .1]
                write_json(session/d.DECISION_JSON, decision)
                with self.assertRaises(ValueError):
                    f.verify_freeze(session)
                with self.assertRaises(ValueError):
                    d.verify_submission_decision(session)

    def test_table_comparison_rejects_missing_zero_and_close_but_different_score(self):
        from kmd.decision_flow import _tables_match
        table = pd.DataFrame({'arch': ['native'], 'macro_f1': [.95], 'rescued': [np.nan]})
        other = table.copy(); other['rescued'] = 0
        self.assertFalse(_tables_match(table, other))
        other = table.copy(); other['macro_f1'] += .000001
        self.assertFalse(_tables_match(table, other))
        self.assertTrue(_tables_match(table, table.copy()))


if __name__ == '__main__':
    unittest.main()
