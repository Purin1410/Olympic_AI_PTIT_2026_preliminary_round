import csv
import json
import zipfile
from pathlib import Path

import pytest

from ailaai.config import Workspace
from ailaai.dataset_setup import expected_names, find_images, image_names
from ailaai.resources import download_dataset


@pytest.fixture
def workspace(tmp_path):
    split = tmp_path / 'assets/splits/train_folds.csv'
    split.parent.mkdir(parents=True)
    with split.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['file_name', 'label', 'fold'])
        writer.writerows((f'{i:05d}.jpg', i % 2, i % 5) for i in range(1, 2001))
    return Workspace.from_root(tmp_path)


def write_archive(workspace):
    archive = workspace.data_root / 'who_is_AI.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        for name in expected_names(workspace, 'train'):
            z.writestr(f'data/train/images/{name}', b'train')
        for name in expected_names(workspace, 'test'):
            z.writestr(f'data/private_test/private_test/images/{name}', b'private')
            z.writestr(f'data/public_test/images/{name}', b'public')
    return archive


def test_real_layout_repair_and_rerun(workspace, monkeypatch):
    write_archive(workspace)
    download_dataset(workspace)
    assert len(image_names(workspace.data_root / 'train/images')) == 2000
    example = workspace.data_root / 'test/images/00001.jpg'
    assert example.read_bytes() == b'private'
    # Running again does not even need to open the archive.
    import ailaai.resources as resources
    extract = resources._safe_extract
    monkeypatch.setattr(resources, '_safe_extract', lambda *a: pytest.fail('Repeated extraction'))
    download_dataset(workspace)
    monkeypatch.setattr(resources, '_safe_extract', extract)
    example.unlink()
    download_dataset(workspace)
    assert example.read_bytes() == b'private'
    example.write_bytes(b'')
    download_dataset(workspace)
    assert example.read_bytes() == b'private'


def test_existing_double_data_is_recognized(workspace):
    archive = write_archive(workspace)
    with zipfile.ZipFile(archive) as z:
        z.extractall(workspace.data_root / 'data')
    # Legacy data/data layout is directly under data_root/data/train.
    with zipfile.ZipFile(archive) as z:
        z.extractall(workspace.data_root)
    archive.unlink()
    download_dataset(workspace)
    assert (workspace.data_root / 'train/images/00001.jpg').read_bytes() == b'train'


def test_public_test_is_not_a_private_test_fallback(workspace):
    folder = workspace.data_root / 'public_test/images'
    folder.mkdir(parents=True)
    for name in expected_names(workspace, 'test'):
        (folder / name).write_bytes(b'public')
    assert find_images(workspace.data_root, 'test', expected_names(workspace, 'test')) is None


def test_failed_download_does_not_publish_archive(workspace, monkeypatch):
    import gdown
    def fake_download(**kwargs):
        Path(kwargs['output']).write_text('<html>Quota exceeded</html>')
        return kwargs['output']
    monkeypatch.setattr(gdown, 'download', fake_download)
    with pytest.raises(RuntimeError, match='Google Drive'):
        download_dataset(workspace)
    assert not (workspace.data_root / 'who_is_AI.zip').exists()
    assert not (workspace.data_root / 'who_is_AI.zip.partial').exists()


def test_archive_traversal_is_rejected(workspace):
    with zipfile.ZipFile(workspace.data_root / 'who_is_AI.zip', 'w') as z:
        z.writestr('../outside.txt', b'bad')
    with pytest.raises(ValueError, match='Unsafe path'):
        download_dataset(workspace)
    assert not (workspace.data_root.parent / 'outside.txt').exists()


def test_changed_config_uses_separate_directory(workspace):
    from ailaai.engine import _run_directory
    first = _run_directory(workspace, 'rgb', {'epochs': 15})
    first.mkdir(parents=True)
    (first / 'config.json').write_text(json.dumps({'epochs': 15}))
    assert _run_directory(workspace, 'rgb', {'epochs': 15}) == first
    second = _run_directory(workspace, 'rgb', {'epochs': 16})
    assert second != first
    assert second == _run_directory(workspace, 'rgb', {'epochs': 16})
    assert json.loads((first / 'config.json').read_text()) == {'epochs': 15}
