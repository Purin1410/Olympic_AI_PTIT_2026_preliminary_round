import numpy as np
import pandas as pd
import pytest
import torch
from ailaai.ablation import haar_view, low_edge_weights, calibration_split, choose_group_thresholds
from ailaai.config import TrainConfig, Workspace
from ailaai.engine import _resolved_config


def test_haar_coefficients_shapes_and_gradient():
    # Analytical 2x2 signal: verify band signs, scale and quadrant placement.
    x = torch.tensor([[[1., 0.], [0., 0.]]]).repeat(3, 1, 1).requires_grad_()
    expected = torch.tensor([[.25, .75], [.75, .75]])
    assert torch.equal(haar_view(x, 2)[0], expected)
    haar_view(x, 2).sum().backward()
    assert torch.isfinite(x.grad).all()
    batch = torch.rand(2, 3, 512, 512)
    y = haar_view(batch)
    assert y.shape == (2, 3, 358, 358)
    assert y.min() >= 0 and y.max() <= 1
    with pytest.raises(ValueError):
        haar_view(x, 3)


def training_rows():
    return pd.DataFrame({'file_name':[f'{i}.jpg' for i in range(16)],
                         'label':[0]*8+[1]*8, 'edge_ratio':np.arange(16)/16,
                         'is_gray':[0,1]*8})


def test_weights_preserve_fake_mass_and_ties():
    frame = training_rows()
    weights, cutoff = low_edge_weights(frame)
    values = frame.file_name.map(weights)
    assert cutoff == pytest.approx(frame[frame.label == 1].edge_ratio.quantile(.25))
    assert values[frame.label == 0].eq(1).all()
    assert values[frame.label == 1].sum() == pytest.approx(8)
    assert values[frame.edge_ratio.lt(cutoff) & frame.label.eq(1)].eq(1.5).all()
    weights, _ = low_edge_weights(frame.assign(edge_ratio=0))
    assert set(weights.values()) == {1}


def test_calibration_does_not_reuse_evaluation_labels():
    frame = training_rows().assign(rgb_prob=np.linspace(.1,.9,16))
    calibration, evaluation = calibration_split(frame)
    assert set(calibration.file_name).isdisjoint(evaluation.file_name)
    assert set(calibration.file_name) | set(evaluation.file_name) == set(frame.file_name)
    chosen = choose_group_thresholds(calibration)
    evaluation['label'] = 1-evaluation.label
    assert choose_group_thresholds(calibration) == chosen
    assert choose_group_thresholds(calibration[calibration.label == 0]) == {0:.5,1:.5}


def test_sample_weights_are_part_of_checkpoint_recipe(tmp_path):
    fit = training_rows()
    val = fit.iloc[:2].assign(file_name=['val0.jpg','val1.jpg'])
    ws = Workspace.from_root(tmp_path)
    cfg = TrainConfig()
    weights, _ = low_edge_weights(fit)
    def factory(*, initialize):
        return None
    def view(x):
        return x
    args = (ws,cfg,'edge_weighted',fit,val,factory,view,dict(cfg.view),dict(cfg.model))
    first = _resolved_config(*args, weights)
    changed = dict(weights); changed['0.jpg']=2
    assert _resolved_config(*args, changed) != first
    with pytest.raises(ValueError, match='exactly'):
        _resolved_config(*args, {'0.jpg':1})
    changed['0.jpg']=float('nan')
    with pytest.raises(ValueError, match='finite'):
        _resolved_config(*args, changed)


def test_completed_weighted_checkpoint_loads_with_matching_weights(tmp_path):
    from ailaai.config import write_json
    from ailaai.engine import load_run
    from ailaai.predictions import PredictionTable
    from ailaai.resources import sha256_file
    fit = training_rows()
    val = fit.iloc[:2].assign(file_name=['val0.jpg','val1.jpg'])
    ws, cfg = Workspace.from_root(tmp_path), TrainConfig(epochs=1)
    weights, _ = low_edge_weights(fit)
    def factory(*, initialize):
        return torch.nn.Linear(3, 2)
    def view(x):
        return x
    args = (ws,cfg,'edge_weighted',fit,val,factory,view,dict(cfg.view),dict(cfg.model))
    folder = ws.artifact_root / 'edge_weighted'
    folder.mkdir()
    write_json(folder/'config.json', _resolved_config(*args, weights))
    torch.save({'model':factory(initialize=False).state_dict()}, folder/'last.pt')
    write_json(folder/'run_status.json', {'status':'complete','epochs':1,
                                         'checkpoint_sha256':sha256_file(folder/'last.pt')})
    pd.DataFrame({'epoch':[1],'val_macro_f1':[1.]}).to_csv(folder/'curves.csv',index=False)
    PredictionTable(val[['file_name','label']].assign(prob=[.1,.9])).save(folder/'val_predictions.csv')
    run = load_run(*args, sample_weights=weights)
    assert run.run_dir == folder
    assert len(run.val_predictions.rows) == len(val)
    changed = dict(weights); changed['0.jpg']=2
    with pytest.raises(ValueError, match='recipe/split'):
        load_run(*args, checkpoint_source=folder/'last.pt',sample_weights=changed)
