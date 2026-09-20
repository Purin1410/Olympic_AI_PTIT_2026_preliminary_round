"""Standard-library static checks only: never execute notebooks or import ML code."""
from pathlib import Path
import argparse
import ast
import csv
import json

EXPECTED_NOTEBOOKS = [
    '00_problem_and_data.ipynb',
    '01_eda_and_baseline.ipynb',
    '02_features_and_lr.ipynb',
    '03_pretrained_and_finetune.ipynb',
    '04_native_and_resampling.ipynb',
    '05_backbone_and_selection.ipynb',
    '06_errors_and_blend.ipynb',
    '07_private_and_submission.ipynb',
    'extension_a_representations.ipynb',
    'extension_b_objectives_and_repair.ipynb',
    'extension_c_data_scaling.ipynb',
    'extension_d_resize_augmentation.ipynb',
    'pipeline_end_to_end.ipynb',
]

EM_DASH_CHAR = chr(0x2014)
EM_DASH_ESCAPED = chr(92) + 'u2014'


def check_no_em_dash(text: str, location: str) -> None:
    """Enforce absolutely no em dash U+2014 or JSON-escaped U+2014."""
    assert EM_DASH_CHAR not in text, f"Em dash (U+2014) detected in {location}"
    assert EM_DASH_ESCAPED not in text, f"Escaped U+2014 detected in {location}"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Static structure and syntax checks for KeMaoDanh."
    )
    parser.add_argument(
        '--notebook-dir',
        type=Path,
        default=None,
        help='Directory containing notebooks to check (defaults to notebooks/).',
    )
    parser.add_argument(
        '--require-executed',
        action='store_true',
        help='Require all code cells to have sequential 1..N execution_count and no error outputs.',
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    python_count = 0
    for path in sorted(root.rglob('*.py')):
        if '.venv' in path.parts:
            continue
        text = path.read_text(encoding='utf-8')
        check_no_em_dash(text, str(path.relative_to(root)))
        ast.parse(text, filename=str(path))
        python_count += 1

    # Check Markdown files in scope
    md_paths = list(root.rglob('*.md'))
    root_repo = root.parent
    if (root_repo / 'README.md').is_file():
        md_paths.append(root_repo / 'README.md')
    for path in sorted(md_paths):
        if '.venv' in path.parts:
            continue
        text = path.read_text(encoding='utf-8')
        check_no_em_dash(text, str(path))

    notebook_dir = args.notebook_dir.resolve() if args.notebook_dir else (root / 'notebooks')
    found_notebooks = sorted(p.name for p in notebook_dir.glob('*.ipynb'))
    assert found_notebooks == sorted(EXPECTED_NOTEBOOKS), (
        f"Expected notebooks {sorted(EXPECTED_NOTEBOOKS)}, but found {found_notebooks}"
    )

    counts = []
    for nb_name in EXPECTED_NOTEBOOKS:
        path = notebook_dir / nb_name
        raw_text = path.read_text(encoding='utf-8')

        notebook = json.loads(raw_text)
        assert notebook.get('nbformat') == 4 and notebook.get('nbformat_minor') == 5, (
            f"{nb_name} must have nbformat=4 and nbformat_minor=5"
        )
        assert notebook.get('metadata', {}).get('kernelspec', {}).get('name') == 'python3', (
            f"{nb_name} must have kernelspec name 'python3'"
        )
        ids = set()
        code_count = 0
        executed_code_count = 0
        expected_ec = 1
        for cell in notebook['cells']:
            assert cell['cell_type'] in {'code', 'markdown'}, f"Invalid cell type in {nb_name}"
            assert cell['id'] not in ids, f"Duplicate cell id {cell['id']} in {nb_name}"
            ids.add(cell['id'])
            source = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
            assert source.strip(), f"Empty cell {cell['id']} in {nb_name}"

            # Check parsed cell sources for em dash U+2014 and escaped U+2014 (do not treat base64 outputs as prose)
            check_no_em_dash(source, f"{nb_name}:{cell['id']}")

            if cell['cell_type'] == 'code':
                code_count += 1
                ast.parse(source, filename=f'{path.name}:{cell["id"]}')
                exec_count = cell.get('execution_count')
                outputs = cell.get('outputs', [])

                if args.require_executed:
                    assert exec_count == expected_ec, (
                        f"Non-sequential execution_count in {nb_name}:{cell['id']}: expected {expected_ec}, got {exec_count}"
                    )
                    expected_ec += 1
                    executed_code_count += 1
                    for out in outputs:
                        out_type = out.get('output_type')
                        assert out_type != 'error', (
                            f"Error output in {nb_name}:{cell['id']}: {out.get('ename')}: {out.get('evalue')}"
                        )
                        assert 'ename' not in out and 'evalue' not in out, (
                            f"Error fields in {nb_name}:{cell['id']}: {out.get('ename')}"
                        )
                else:
                    if exec_count is not None and isinstance(exec_count, int) and exec_count > 0:
                        executed_code_count += 1

        counts.append({
            'notebook': path.name,
            'cells': len(notebook['cells']),
            'code_cells': code_count,
            'executed_code_cells': executed_code_count,
        })

    assert len(counts) == 13

    fields = {node.target.id for node in ast.parse((root / 'src/kmd/config.py').read_text(encoding='utf-8')).body
              if isinstance(node, ast.ClassDef) for node in node.body if isinstance(node, ast.AnnAssign)}
    for path in sorted((root / 'configs').glob('*.json')):
        config = json.loads(path.read_text(encoding='utf-8'))
        assert set(config) <= fields, f"Unknown config field in {path.name}"
        assert config['effective_batch'] % config['microbatch'] == 0
        assert config['epochs'] > config['warmup']

    with (root / 'configs/development_split.csv').open(encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len({r['pair_id'] for r in rows}) == 800
    assert set(rows[0]) == {'pair_id', 'inner_fold'}
    assert {r['inner_fold'] for r in rows} == {'0', '1', '2'}

    print(json.dumps({
        'status': 'static_checks_passed',
        'python_files': python_count,
        'notebooks': counts,
        'em_dash_enforced': True,
        'require_executed': bool(args.require_executed),
        'execution': 'verified_executed' if args.require_executed else 'allowed_or_unexecuted',
        'gpu_access': 'not_used',
    }, indent=2))


if __name__ == '__main__':
    main()
