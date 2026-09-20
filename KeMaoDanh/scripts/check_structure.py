"""Standard-library static checks only: never execute notebooks or import ML code."""
from pathlib import Path
import ast
import csv
import json

EXPECTED_NOTEBOOKS = [
    '00_problem_and_data.ipynb',
    '01_features_and_lr.ipynb',
    '02_pretrained_and_finetune.ipynb',
    '03_native_and_resampling.ipynb',
    '04_errors_and_blend.ipynb',
    '05_validation_and_submission.ipynb',
    'pipeline_end_to_end.ipynb',
]


def main():
    root = Path(__file__).resolve().parents[1]
    python_count = 0
    for path in sorted(root.rglob('*.py')):
        if '.venv' in path.parts:
            continue
        ast.parse(path.read_text(), filename=str(path))
        python_count += 1

    notebook_dir = root / 'notebooks'
    found_notebooks = sorted(p.name for p in notebook_dir.glob('*.ipynb'))
    assert found_notebooks == sorted(EXPECTED_NOTEBOOKS), (
        f"Expected notebooks {sorted(EXPECTED_NOTEBOOKS)}, but found {found_notebooks}"
    )

    counts = []
    for nb_name in EXPECTED_NOTEBOOKS:
        path = notebook_dir / nb_name
        notebook = json.loads(path.read_text())
        assert notebook.get('nbformat') == 4 and notebook.get('nbformat_minor') == 5, (
            f"{nb_name} must have nbformat=4 and nbformat_minor=5"
        )
        assert notebook.get('metadata', {}).get('kernelspec', {}).get('name') == 'python3', (
            f"{nb_name} must have kernelspec name 'python3'"
        )
        ids = set()
        code_count = 0
        for cell in notebook['cells']:
            assert cell['cell_type'] in {'code', 'markdown'}, f"Invalid cell type in {nb_name}"
            assert cell['id'] not in ids, f"Duplicate cell id {cell['id']} in {nb_name}"
            ids.add(cell['id'])
            source = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
            assert source.strip(), f"Empty cell {cell['id']} in {nb_name}"
            if cell['cell_type'] == 'code':
                assert cell.get('execution_count') is None, f"Executed cell {cell['id']} in {nb_name}"
                assert cell.get('outputs') == [], f"Non-empty output in {nb_name}:{cell['id']}"
                ast.parse(source, filename=f'{path.name}:{cell["id"]}')
                code_count += 1
        counts.append({'notebook': path.name, 'cells': len(notebook['cells']), 'code_cells': code_count})

    assert len(counts) == 7

    fields = {node.target.id for node in ast.parse((root / 'src/kmd/config.py').read_text()).body
              if isinstance(node, ast.ClassDef) for node in node.body if isinstance(node, ast.AnnAssign)}
    for path in sorted((root / 'configs').glob('*.json')):
        config = json.loads(path.read_text())
        assert set(config) <= fields, f"Unknown config field in {path.name}"
        assert config['effective_batch'] % config['microbatch'] == 0
        assert config['epochs'] > config['warmup']

    with (root / 'configs/development_split.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len({r['pair_id'] for r in rows}) == 800
    assert set(rows[0]) == {'pair_id', 'inner_fold'}
    assert {r['inner_fold'] for r in rows} == {'0', '1', '2'}

    print(json.dumps({'status': 'static_checks_passed', 'python_files': python_count,
                      'notebooks': counts, 'execution': 'not_run', 'gpu_access': 'not_used'}, indent=2))


if __name__ == '__main__':
    main()
