import copy
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from PIL import Image
from ailaai.config import TrainConfig, Workspace
from ailaai.engine import train_one_epoch, _validation_predictions, fit_fold
from ailaai.models import attach_optimizer_groups, optimizer_parameter_groups
from ailaai.teaching import image_census, choose_size_rule, build_resnet34, crossfit_threshold


def identity(x):
    return x


def tiny_factory(*, initialize):
    model = nn.Sequential(nn.Conv2d(3, 2, 1), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(2, 2))
    return attach_optimizer_groups(model, model[-1])


def test_census_reads_pixels_bytes_and_size_rule(tmp_path):
    rows = []
    for i, color in enumerate([(80, 80, 80), (80, 60, 20)]):
        path = tmp_path / f'{i}.png'
        Image.new('RGB', (12, 10), color).save(path)
        rows.append({'file_name': path.name, 'label': i, 'path': str(path)})
    census = image_census(pd.DataFrame(rows))
    assert census.is_gray.tolist() == [True, False]
    assert census.width.tolist() == [12, 12]
    assert census.size_kib.iloc[0] == (tmp_path / '0.png').stat().st_size / 1024
    # Threshold and direction both need fitting; flipped labels reverse direction.
    data = pd.DataFrame({'size_kib': [1., 2., 8., 9.], 'label': [1, 1, 0, 0]})
    assert choose_size_rule(data) == {'threshold_kib': 5., 'fake_is_small': True, 'accuracy': 1.}
    assert choose_size_rule(data.assign(label=1-data.label))['fake_is_small'] is False


def test_accumulation_final_group_matches_manual_optimizer_steps():
    torch.manual_seed(1)
    model = tiny_factory(initialize=False)
    control = copy.deepcopy(model)
    cfg = TrainConfig(accumulation=2, normalize_mean=(0, 0, 0), normalize_std=(1, 1, 1),
                      augmentation={'horizontal_flip_probability': 0})
    batches = [(torch.rand(2, 3, 8, 8), torch.tensor([0, 1]), [f'{k}a', f'{k}b']) for k in range(3)]
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    scaler = torch.amp.GradScaler('cpu', enabled=False)
    train_one_epoch(model, batches, optimizer, scaler, cfg, identity, torch.device('cpu'))
    manual = torch.optim.SGD(control.parameters(), lr=.1)
    for group in [batches[:2], batches[2:]]:
        manual.zero_grad()
        for images, y, _ in group:
            (nn.functional.cross_entropy(control(images), y) / len(group)).backward()
        manual.step()
    for left, right in zip(model.parameters(), control.parameters()):
        torch.testing.assert_close(left, right)


def test_frozen_backbone_keeps_parameters_and_batchnorm_statistics():
    model = build_resnet34(pretrained=False, freeze_backbone=True)
    before = {name: tensor.clone() for name, tensor in model.state_dict().items() if not name.startswith('fc.')}
    head_before = model.fc.weight.detach().clone()
    cfg = TrainConfig(accumulation=1, augmentation={'horizontal_flip_probability': 0})
    groups = optimizer_parameter_groups(model, cfg.backbone_lr, cfg.head_lr)
    assert len(groups) == 1 and groups[0]['lr'] == cfg.head_lr
    optimizer = torch.optim.AdamW(groups)
    train_one_epoch(model, [(torch.rand(2, 3, 64, 64), torch.tensor([0, 1]), ['a', 'b'])],
                    optimizer, torch.amp.GradScaler('cpu', enabled=False), cfg, identity, torch.device('cpu'))
    assert not torch.equal(head_before, model.fc.weight)
    for name, expected in before.items():
        assert torch.equal(model.state_dict()[name], expected), name


def test_tta_validation_is_average_of_probabilities():
    class Directional(nn.Module):
        def forward(self, x):
            v = x[:, 0, 0, 0] * 4
            return torch.stack([torch.zeros_like(v), v], 1)
    image = torch.zeros(1, 3, 4, 4); image[..., 0] = 1
    cfg = TrainConfig(tta=True, normalize_mean=(0, 0, 0), normalize_std=(1, 1, 1))
    model = Directional()
    pred, _, _ = _validation_predictions(model, [(image, torch.tensor([1]), ['a'])], cfg, identity, torch.device('cpu'))
    expected = .5 * (model(image).softmax(1)[:, 1] + model(image.flip(-1)).softmax(1)[:, 1])
    assert pred.rows.prob.iloc[0] == pytest.approx(expected.item())


def test_threshold_choice_does_not_read_excluded_fold_labels():
    rows = pd.DataFrame({'file_name': [str(i) for i in range(12)], 'fold': np.repeat([0, 1, 2], 4),
                         'label': [0, 0, 1, 1]*3, 'p_mean': [.1, .4, .45, .9]*3})
    _, first = crossfit_threshold(rows)
    changed = rows.copy(); changed.loc[changed.fold == 0, 'label'] = 1-changed.loc[changed.fold == 0, 'label']
    _, second = crossfit_threshold(changed)
    assert first.set_index('fold').loc[0, 'threshold'] == second.set_index('fold').loc[0, 'threshold']


def test_cpu_fit_writes_terminal_checkpoint_and_reuses_completed_run(tmp_path, monkeypatch):
    for key in ['AILAAI_DATA_ROOT', 'AILAAI_ARTIFACT_ROOT', 'AILAAI_OUTPUT_ROOT']:
        monkeypatch.delenv(key, raising=False)
    rows = []
    for i in range(8):
        path = tmp_path / f'{i}.png'; Image.new('RGB', (16, 16), (i*20, 80, 120)).save(path)
        rows.append({'file_name': path.name, 'label': i % 2, 'path': str(path)})
    rows = pd.DataFrame(rows)
    cfg = TrainConfig(epochs=1, batch_size=2, accumulation=2, model={'backbone': 'test_tiny'}, view={'name': 'identity'})
    ws = Workspace.from_root(tmp_path, 'smoke')
    args = (ws, cfg, 'rgb', rows.iloc[:6], rows.iloc[6:], tiny_factory, identity, dict(cfg.view), dict(cfg.model))
    run = fit_fold(*args, allow_cpu=True)
    assert run.checkpoint_path.is_file() and len(run.curves) == 1
    assert set(run.val_predictions.rows.file_name) == set(rows.iloc[6:].file_name)
    timestamp = run.checkpoint_path.stat().st_mtime_ns
    reused = fit_fold(*args, allow_cpu=True)
    assert reused.checkpoint_path.stat().st_mtime_ns == timestamp


def test_submission_rejects_decimal_label_and_wrong_column_order(tmp_path):
    import zipfile
    from ailaai.submission import validate_submission, export_submission
    frame = pd.DataFrame({'file_name': ['0001.jpg', '0002.jpg'], 'category_id': [0, 1]})
    path = export_submission(frame, tmp_path / 'ok.zip', frame.file_name, 2)
    assert validate_submission(path, frame.file_name, 2).row_count == 2
    for content in ['file_name,category_id\n0001.jpg,0\n0002.jpg,1.0\n',
                    'category_id,file_name\n0,0001.jpg\n1,0002.jpg\n']:
        with zipfile.ZipFile(tmp_path / 'bad.zip', 'w') as archive:
            archive.writestr('submission.csv', content)
        with pytest.raises(ValueError):
            validate_submission(tmp_path / 'bad.zip', frame.file_name, 2)
