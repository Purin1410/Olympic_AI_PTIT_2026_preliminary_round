"""Protocol checks only: no model fitting, notebook execution, or GPU access."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from kmd import finalization as f
from kmd.core import write_json, sha256

class FinalizationChecks(unittest.TestCase):
    def table(self, blend=.91):
        scores={'center60':.9,'dense288':.92,'b2':.90,'native':.91,'blend':blend}
        return pd.DataFrame([dict(arch=a,seed=s,smoke=False,macro_f1=v)
                             for a,v in scores.items() for s in (20260917,20260918,20260919)])

    def test_blend_must_beat_best_single_not_only_b2(self):
        self.assertEqual(f.select_candidate(self.table(.924))['method'],'dense288')
        self.assertEqual(f.select_candidate(self.table(.926))['method'],'blend')

    def test_rejects_smoke_missing_duplicate_and_nonfinite(self):
        variants=[self.table().iloc[:-1],pd.concat([self.table().iloc[:-1],self.table().iloc[:1]])]
        t=self.table();t.loc[0,'smoke']=True;variants.append(t)
        t=self.table();t.loc[0,'macro_f1']=float('nan');variants.append(t)
        for t in variants:
            with self.assertRaises(ValueError):f.select_candidate(t)

    def test_gate_uses_unrounded_gain(self):
        self.assertEqual(f.select_candidate(self.table(.92499))['method'],'dense288')
        self.assertEqual(f.select_candidate(self.table(.92501))['method'],'blend')

    def test_predictions_verified_before_reading_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'review_freeze.json').write_text('{}')
            write_json(root/'prediction_manifest.json',dict(freeze_sha256=sha256(root/'review_freeze.json'),predictions={}))
            with patch.object(f,'verify_freeze',return_value={'methods':['lr_full32']}),patch.object(f,'read_csv') as read:
                with self.assertRaises(ValueError):f.score_private(root,root,root/'labels.csv','test fixture')
                read.assert_not_called()

    def test_scoring_aligns_string_ids_and_rejects_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'review_freeze.json').write_text('{}')
            pd.DataFrame({'pair_id':['00002','00001'],'p':[.9,.1]}).to_csv(root/'lr_full32.csv',index=False)
            write_json(root/'prediction_manifest.json',dict(freeze_sha256=sha256(root/'review_freeze.json'),predictions={'lr_full32':sha256(root/'lr_full32.csv')}))
            labels=pd.DataFrame({'pair_id':['00001','00002'],'fake_position':[0,1]});labels.to_csv(root/'labels.csv',index=False)
            oof=pd.DataFrame({'fake_position':[0,1],'p':[.1,.9]})
            with patch.object(f,'verify_freeze',return_value={'methods':['lr_full32']}),patch.object(f,'method_oof',return_value=oof):
                t=f.score_private(root,root,root/'labels.csv','Synthetic unit-test labels, not dataset results')
                self.assertEqual(t.private_f1.iloc[0],1)
                labels.iloc[:1].to_csv(root/'labels.csv',index=False)
                with self.assertRaises(ValueError):f.score_private(root,root,root/'labels.csv','fixture')

    def test_lr_contract_uses_fold_hash_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            files=[f'lr_full32_fold{i}.joblib' for i in range(3)]
            for name in files:(root/name).write_text('fixture, never loaded')
            (root/'lr_full32_oof.csv').write_text('fixture')
            write_json(root/'lr_full32_job.json',{
                'code_sha256':f.current_code_hashes(), 'model_files':files,
                'model_sha256':{f'fold{i}':sha256(root/name) for i,name in enumerate(files)},
                'oof_file':'lr_full32_oof.csv','oof_sha256':sha256(root/'lr_full32_oof.csv')})
            self.assertEqual(len(f._paths(root,'lr_full32')),6)
            (root/files[0]).write_text('changed')
            with self.assertRaises(ValueError):f._paths(root,'lr_full32')

    def test_refit_budget_median_and_terminal(self):
        frame=pd.DataFrame({'pair_id':['a','b','c'],'image_0':['a0','b0','c0'],'image_1':['a1','b1','c1'],'inner_fold':[0,1,2]})
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            records=[{'fold':i,'checkpoint':f'f{i}/best.pt'} for i in range(3)]
            for method in ('b2','native'):
                write_json(root/f'{method}_s20260917_models.json',records)
            for i,e in enumerate([8,12,10]):write_json(root/f'f{i}/metrics.json',{'best_epoch':e})
            with patch.object(f,'load_session',return_value=root),patch.object(f,'read_csv',return_value=frame),patch.object(f,'validate_fold_checkpoint',return_value=(True,{})):
                cfg,epochs=f.refit_config('b2',root)
                self.assertEqual(cfg.epochs,10);self.assertTrue(cfg.fixed_epochs)
                self.assertEqual(f.refit_config('native',root)[0].epochs,19)

    def test_dynamic_inference_and_legacy_blend_dispatch(self):
        frame = pd.DataFrame({'pair_id': ['x', 'y'], 'image_0': ['x0', 'y0'], 'image_1': ['x1', 'y1']})
        recipe = {'protocol': 'backbone_flow_v1', 'method': 'blend_selected_native',
                  'best_single': 'center72', 'components': ['center72', 'native'],
                  'weights': [.5, .5], 'blend_status': 'tested'}
        def infer(name, frame, *args, **kwargs):
            out = frame.copy(); out['p'] = [.2, .8] if name != 'native' else [.6, .4]
            return out
        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp)
            write_json(session/'review_freeze.json', {})
            with patch.object(f, 'load_session', return_value=session), \
                 patch.object(f, 'check_test_separation'), \
                 patch.object(f, 'verify_freeze', return_value={'selection': recipe}), \
                 patch.object(f, 'infer_cnn', side_effect=infer) as cnn:
                out = f.infer_method('blend_selected_native', frame, session, session)
                self.assertEqual([c.args[0] for c in cnn.call_args_list], ['center72', 'native'])
                np.testing.assert_allclose(out.p, [.4, .6])
                cnn.reset_mock()
                f.infer_method('blend', frame, session, session)
                self.assertEqual([c.args[0] for c in cnn.call_args_list], ['b2', 'native'])
                recipe['weights'] = [.9, .1]
                with self.assertRaises(ValueError):
                    f.infer_method('blend_selected_native', frame, session, session)

    def test_dynamic_refit_dispatch_before_loading_training_dependencies(self):
        recipe = {'protocol': 'backbone_flow_v1', 'method': 'blend_selected_native',
                  'best_single': 'b0_224', 'components': ['b0_224', 'native'],
                  'weights': [.5, .5], 'blend_status': 'tested', 'submission_strategy': 'refit_all'}
        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp)
            write_json(session/'review_freeze.json', {})
            original_refit = f.refit_all
            original_infer = f.infer_refit
            with patch.object(f, 'load_session', return_value=session), \
                 patch.object(f, 'verify_freeze', return_value={'selection': recipe}):
                with patch.object(f, 'refit_all', return_value='mock_training_branch') as recurse:
                    original_refit('blend_selected_native', session, session)
                    self.assertEqual([c.args[0] for c in recurse.call_args_list], ['b0_224', 'native'])
                frame = pd.DataFrame({'pair_id': ['x'], 'image_0': ['x0'], 'image_1': ['x1']})
                def result(name, frame, *args, **kwargs):
                    out = frame.copy(); out['p'] = .2 if name == 'b0_224' else .6
                    return out
                with patch.object(f, 'infer_refit', side_effect=result) as recurse:
                    out = original_infer('blend_selected_native', frame, session, session)
                    self.assertEqual([c.args[0] for c in recurse.call_args_list], ['b0_224', 'native'])
                    np.testing.assert_allclose(out.p, [.4])
                with self.assertRaises(ValueError):
                    original_refit('b2', session, session)

    def test_every_new_single_winner_has_refit_budget(self):
        frame = pd.DataFrame({'pair_id': ['a','b','c'], 'image_0': ['a0','b0','c0'],
                              'image_1': ['a1','b1','c1'], 'inner_fold': [0,1,2]})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = [{'fold': i, 'checkpoint': f'f{i}/best.pt'} for i in range(3)]
            for i, epoch in enumerate([8, 12, 10]):
                write_json(root/f'f{i}/metrics.json', {'best_epoch': epoch})
            with patch.object(f, 'load_session', return_value=root), \
                 patch.object(f, 'read_csv', return_value=frame), \
                 patch.object(f, 'validate_fold_checkpoint', return_value=(True, {})):
                for method in ('center60_cap48', 'center72', 'b0_224', 'b2_224', 'dense288'):
                    self.assertIn(method, f.REFIT_METHODS)
                    write_json(root/f'{method}_s20260917_models.json', records)
                    cfg, epochs = f.refit_config(method, root)
                    self.assertEqual(cfg.epochs, 10)
                    self.assertTrue(cfg.fixed_epochs)
                    self.assertEqual(epochs, [8, 12, 10])

if __name__=='__main__':unittest.main()
