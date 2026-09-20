"""Publish executed notebooks from evidence to main notebooks directory.

Enforces strict verification against revision_draft originals and execution metadata:
- Verifies execution_summary.json contains exactly 13 expected notebooks (normalizing optional .ipynb).
- Verifies status passed, mode full, smoke is boolean False.
- Verifies code_cells and executed_code_cells match actual notebook code count.
- Verifies source_sha256 explicitly matches revision_draft original bytes.
- Verifies sequential 1..N execution counts and no error outputs.
- Verifies every cell source, type, and ID matches draft exactly (no tampering).
- Preflights all archive collisions before any disk mutation.
- Archives old main notebooks outside repo before replacing/removing stale names.
- Refuses archive directories inside the repository.
- Refuses evidence inside main notebooks directory or overlapping archive/evidence dirs.
- Refuses unrecognized main notebook names.
- Copies executed .ipynb bytes unaltered to preserve inline PNG/SVG outputs.
- Preserves revision_draft completely unchanged.
- Supports --dry-run without disk mutations.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys

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

RECOGNIZED_MAIN_NOTEBOOKS = {
    # New 13 notebooks
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
    # Legacy main notebooks
    '01_features_and_lr.ipynb',
    '02_pretrained_and_finetune.ipynb',
    '03_native_and_resampling.ipynb',
    '04_errors_and_blend.ipynb',
    '05_validation_and_submission.ipynb',
    'extension_a_representations_and_losses.ipynb',
    'extension_b_candidate_seeds_and_gating.ipynb',
}

EM_DASH_CHAR = chr(0x2014)
EM_DASH_ESCAPED = chr(92) + 'u2014'


def check_no_em_dash(text: str, location: str) -> None:
    """Enforce absolutely no em dash U+2014 or JSON-escaped U+2014."""
    assert EM_DASH_CHAR not in text, f"Em dash (U+2014) detected in {location}"
    assert EM_DASH_ESCAPED not in text, f"Escaped U+2014 detected in {location}"


def find_task_root(explicit_root: Path | None = None) -> Path:
    if explicit_root:
        cand = explicit_root.resolve()
        if (cand / 'src/kmd').is_dir() and (cand / 'notebooks').is_dir():
            return cand
        if (cand / 'KeMaoDanh/src/kmd').is_dir() and (cand / 'KeMaoDanh/notebooks').is_dir():
            return (cand / 'KeMaoDanh').resolve()
        return cand
    script_dir = Path(__file__).resolve().parent
    return script_dir.parent


def is_inside_dir(target: Path, parent: Path) -> bool:
    try:
        target.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify and publish executed notebooks to main notebooks directory."
    )
    parser.add_argument(
        '--evidence-dir',
        type=Path,
        required=True,
        help='Directory containing execution_summary.json and executed notebooks.',
    )
    parser.add_argument(
        '--archive-dir',
        type=Path,
        required=True,
        help='Directory outside repository to archive old main notebooks.',
    )
    parser.add_argument(
        '--task-root',
        type=Path,
        default=None,
        help='Optional explicit KeMaoDanh root directory (for testing fixtures).',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Perform all verifications without modifying files on disk.',
    )
    return parser


def parse_summary_entries(summary_path: Path) -> dict[str, dict]:
    summary_raw = summary_path.read_text(encoding='utf-8')
    summary_data = json.loads(summary_raw)

    if isinstance(summary_data, dict):
        if 'notebooks' in summary_data and isinstance(summary_data['notebooks'], list):
            raw_entries = summary_data['notebooks']
        elif 'results' in summary_data and isinstance(summary_data['results'], list):
            raw_entries = summary_data['results']
        else:
            raw_entries = []
            for k, v in summary_data.items():
                if isinstance(v, dict):
                    item = dict(v)
                    item.setdefault('name', k)
                    raw_entries.append(item)
    elif isinstance(summary_data, list):
        raw_entries = summary_data
    else:
        raise ValueError(f"Unrecognized structure in {summary_path.name}")

    entries = {}
    for item in raw_entries:
        raw_name = item.get('name') or item.get('notebook') or item.get('file') or item.get('filename')
        if not raw_name:
            raise ValueError(f"Entry in {summary_path.name} missing notebook identifier: {item}")
        nb_name = Path(raw_name).name
        if not nb_name.endswith('.ipynb'):
            nb_name = f"{nb_name}.ipynb"

        if nb_name in entries:
            raise ValueError(f"Duplicate entry for notebook '{nb_name}' in {summary_path.name}")
        entries[nb_name] = item

    if set(entries.keys()) != set(EXPECTED_NOTEBOOKS):
        missing = sorted(set(EXPECTED_NOTEBOOKS) - set(entries.keys()))
        extra = sorted(set(entries.keys()) - set(EXPECTED_NOTEBOOKS))
        raise ValueError(
            f"{summary_path.name} must contain exactly 13 unique expected notebooks. "
            f"Missing: {missing}; Extra: {extra}"
        )

    return entries


def publish(
    evidence_dir: Path,
    archive_dir: Path,
    task_root: Path,
    dry_run: bool = False,
) -> dict:
    evidence_dir = evidence_dir.resolve()
    archive_dir = archive_dir.resolve()
    task_root = task_root.resolve()
    repo_root = task_root.parent.resolve()
    main_dir = (task_root / 'notebooks').resolve()

    if not evidence_dir.is_dir():
        raise FileNotFoundError(f"Evidence directory not found: {evidence_dir}")

    # 1. Refuse archive inside repo
    if is_inside_dir(archive_dir, repo_root) or is_inside_dir(archive_dir, task_root):
        raise ValueError(
            f"Refusing archive inside repository. Archive directory must be outside repo: {archive_dir}"
        )

    # 2. Refuse evidence inside main notebooks
    if is_inside_dir(evidence_dir, main_dir) or evidence_dir == main_dir:
        raise ValueError(f"Refusing evidence inside main notebooks directory: {evidence_dir}")

    # 3. Refuse dangerous overlap between archive and evidence
    if (archive_dir == evidence_dir or
            is_inside_dir(evidence_dir, archive_dir) or
            is_inside_dir(archive_dir, evidence_dir)):
        raise ValueError(
            f"Refusing dangerous archive and evidence directory overlap: {archive_dir} vs {evidence_dir}"
        )

    summary_path = evidence_dir / 'execution_summary.json'
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing execution_summary.json in {evidence_dir}")

    entries = parse_summary_entries(summary_path)

    draft_dir = task_root / 'notebooks' / 'revision_draft'
    if not draft_dir.is_dir():
        raise FileNotFoundError(f"Missing revision_draft directory at {draft_dir}")

    exec_paths: dict[str, Path] = {}
    initial_draft_hashes: dict[str, str] = {}

    for nb_name in EXPECTED_NOTEBOOKS:
        entry = entries[nb_name]

        # 1. Verify status passed
        status = str(entry.get('status', '')).strip().lower()
        if status != 'passed':
            raise ValueError(f"{nb_name}: status is '{status}', expected 'passed'")

        # 2. Verify mode full
        mode = str(entry.get('mode', '')).strip().lower()
        if mode != 'full':
            raise ValueError(f"{nb_name}: mode is '{mode}', expected 'full'")

        # 3. Verify smoke is actual boolean False
        smoke = entry.get('smoke')
        if smoke is not False:
            raise ValueError(f"{nb_name}: smoke must be boolean False, got {smoke!r}")

        # 4. Verify summary error count if present
        if 'errors' in entry and entry['errors'] != 0:
            raise ValueError(f"{nb_name}: summary reports errors={entry['errors']}")
        if 'error_count' in entry and entry['error_count'] != 0:
            raise ValueError(f"{nb_name}: summary reports error_count={entry['error_count']}")

        # 5. Locate executed notebook in evidence_dir
        cands = [
            evidence_dir / nb_name,
            evidence_dir / Path(str(entry.get('executed_file', ''))).name if entry.get('executed_file') else None,
            evidence_dir / Path(str(entry.get('executed_path', ''))).name if entry.get('executed_path') else None,
            evidence_dir / Path(str(entry.get('path', ''))).name if entry.get('path') else None,
            evidence_dir / f"{Path(nb_name).stem}.ipynb",
        ]
        chosen_exec = None
        for cand in cands:
            if cand and cand.is_file():
                chosen_exec = cand
                break
        if chosen_exec is None:
            raise FileNotFoundError(f"Could not locate executed notebook file for {nb_name} in {evidence_dir}")
        exec_paths[nb_name] = chosen_exec

        # 6. Verify revision_draft hash and require source_sha256 explicitly
        draft_path = draft_dir / nb_name
        if not draft_path.is_file():
            raise FileNotFoundError(f"Missing revision_draft file: {draft_path}")

        draft_bytes = draft_path.read_bytes()
        draft_sha = hashlib.sha256(draft_bytes).hexdigest()
        initial_draft_hashes[nb_name] = draft_sha

        if 'source_sha256' not in entry or not entry['source_sha256']:
            raise ValueError(f"{nb_name}: missing required field 'source_sha256' in execution summary")

        summary_hash = str(entry['source_sha256']).strip().lower()
        if summary_hash != draft_sha.lower():
            raise ValueError(
                f"{nb_name}: source_sha256 in summary ({summary_hash}) does not match draft file hash ({draft_sha})"
            )

        # 7. Compare cell by cell and verify sequential execution counts
        exec_bytes = chosen_exec.read_bytes()
        exec_nb = json.loads(exec_bytes.decode('utf-8'))
        draft_nb = json.loads(draft_bytes.decode('utf-8'))

        if len(exec_nb.get('cells', [])) != len(draft_nb.get('cells', [])):
            raise ValueError(
                f"{nb_name}: cell count mismatch ({len(exec_nb.get('cells', []))} vs draft {len(draft_nb.get('cells', []))})"
            )

        code_cells_count = 0
        expected_ec = 1

        for idx, (e_cell, d_cell) in enumerate(zip(exec_nb['cells'], draft_nb['cells'])):
            if e_cell.get('cell_type') != d_cell.get('cell_type'):
                raise ValueError(
                    f"{nb_name} cell {idx}: cell_type mismatch ('{e_cell.get('cell_type')}' vs '{d_cell.get('cell_type')}')"
                )
            if e_cell.get('id') != d_cell.get('id'):
                raise ValueError(
                    f"{nb_name} cell {idx}: id mismatch ('{e_cell.get('id')}' vs '{d_cell.get('id')}')"
                )

            e_src = e_cell.get('source', '')
            d_src = d_cell.get('source', '')
            if isinstance(e_src, list):
                e_src = ''.join(e_src)
            if isinstance(d_src, list):
                d_src = ''.join(d_src)
            if e_src != d_src:
                raise ValueError(
                    f"{nb_name} cell {idx} ({e_cell.get('id')}): source does not match revision_draft"
                )

            if e_cell['cell_type'] == 'code':
                code_cells_count += 1
                ec = e_cell.get('execution_count')
                if ec != expected_ec:
                    raise ValueError(
                        f"{nb_name} code cell {idx} ({e_cell.get('id')}): "
                        f"expected sequential execution_count {expected_ec}, got {ec}"
                    )
                expected_ec += 1

                for out in e_cell.get('outputs', []):
                    if out.get('output_type') == 'error' or 'ename' in out or 'evalue' in out:
                        raise ValueError(
                            f"{nb_name} code cell {idx} ({e_cell.get('id')}): error in outputs: "
                            f"{out.get('ename')}: {out.get('evalue')}"
                        )

        if code_cells_count == 0:
            raise ValueError(f"{nb_name}: has no code cells")

        # 8. Validate summary counts against actual code cells
        if 'code_cells' not in entry or 'executed_code_cells' not in entry:
            raise ValueError(f"{nb_name}: missing 'code_cells' or 'executed_code_cells' in execution summary")
        summary_code = entry['code_cells']
        summary_exec = entry['executed_code_cells']
        if summary_code != code_cells_count or summary_exec != code_cells_count:
            raise ValueError(
                f"{nb_name}: summary code_cells={summary_code}, executed_code_cells={summary_exec} "
                f"do not match actual notebook code cells ({code_cells_count})"
            )

    # 9. Check existing main notebooks and refuse unrecognized names
    existing_files = sorted(p for p in main_dir.glob('*.ipynb') if p.is_file())
    for p in existing_files:
        if p.name not in RECOGNIZED_MAIN_NOTEBOOKS:
            raise ValueError(
                f"Unrecognized notebook in main directory: {p.name}. Refusing publication."
            )

    # 10. Preflight archive collisions before any modification
    for p in existing_files:
        arch_target = archive_dir / p.name
        if arch_target.exists():
            if arch_target.read_bytes() != p.read_bytes():
                raise FileExistsError(
                    f"Archive collision: target file {arch_target} already exists with different contents. "
                    "Refusing to overwrite existing archive."
                )

    stale_files = [p for p in existing_files if p.name not in EXPECTED_NOTEBOOKS]

    # 11. If dry_run, return immediately without disk mutations
    if dry_run:
        return {
            'status': 'dry_run_completed',
            'verified_notebooks': len(EXPECTED_NOTEBOOKS),
            'archived_count': len(existing_files),
            'removed_stale_count': len(stale_files),
            'published_count': 0,
            'archive_dir': str(archive_dir),
            'dry_run': True,
        }

    # 12. Perform mutations
    archive_dir.mkdir(parents=True, exist_ok=True)
    for p in existing_files:
        arch_target = archive_dir / p.name
        if not arch_target.exists():
            shutil.copyfile(p, arch_target)

    for p in stale_files:
        p.unlink()

    for nb_name in EXPECTED_NOTEBOOKS:
        src_exec = exec_paths[nb_name]
        dst_main = main_dir / nb_name
        shutil.copyfile(src_exec, dst_main)

    # 13. Verify revision_draft remains completely unchanged
    for nb_name in EXPECTED_NOTEBOOKS:
        draft_path = draft_dir / nb_name
        current_hash = hashlib.sha256(draft_path.read_bytes()).hexdigest()
        if current_hash != initial_draft_hashes[nb_name]:
            raise RuntimeError(f"FATAL: revision_draft/{nb_name} was modified during publication!")

    return {
        'status': 'publication_completed',
        'verified_notebooks': len(EXPECTED_NOTEBOOKS),
        'archived_count': len(existing_files),
        'removed_stale_count': len(stale_files),
        'published_count': len(EXPECTED_NOTEBOOKS),
        'archive_dir': str(archive_dir),
        'dry_run': False,
    }


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    task_root = find_task_root(args.task_root)
    result = publish(
        evidence_dir=args.evidence_dir,
        archive_dir=args.archive_dir,
        task_root=task_root,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
