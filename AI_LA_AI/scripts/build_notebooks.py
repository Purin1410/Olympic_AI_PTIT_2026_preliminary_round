"""Build the teaching notebooks from small, reviewable cell-source Markdown files."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "notebook_sources"
OUTPUT_ROOT = ROOT / "notebooks"
CELL_MARKER = re.compile(r"^<!-- ailaai-cell:(\d+):(markdown|code) -->\s*$", re.MULTILINE)
BANNED_NB3 = (
    "rớt đài", "vũ khí tối thượng", "playbook thi đấu", "chiến thuật phòng thi",
    "cạm bẫy dò ngưỡng", "15% thí sinh", "4 tiếng", "thực chiến",
    "kiểm định hợp đồng dữ liệu nghiêm ngặt", "quản trị ngân sách tính toán",
    "cơ chế bù trừ sai số trực giao",
)


@dataclass
class Cell:
    index: int
    kind: Literal["markdown", "code"]
    source: str


NOTEBOOKS = {
    "00_pipeline_end_to_end": (65, 29),
    "01_eda_baseline_geometry": (55, 25),
    "02_forensic_specialist": (39, 18),
    "03_ensemble_threshold_submission": (41, 18),
    "04_negative_results_and_ablation": (51, 24),
}


def _read_cells(path: Path) -> list[Cell]:
    text = path.read_text(encoding="utf-8")
    markers = list(CELL_MARKER.finditer(text))
    if not markers:
        raise ValueError(f"No cell markers in {path}.")
    cells = []
    for position, marker in enumerate(markers):
        end = markers[position + 1].start() if position + 1 < len(markers) else len(text)
        source = text[marker.end():end].strip("\n")
        cells.append(Cell(int(marker.group(1)), marker.group(2), source))
    if [cell.index for cell in cells] != list(range(len(cells))):
        raise ValueError(f"Cell IDs in {path} must be zero-based and consecutive.")
    for cell in cells:
        if cell.kind == "code":
            ast.parse(cell.source, filename=f"{path.name}#cell-{cell.index}")
    return cells


def _write_cells(path: Path, cells: list[Cell]) -> None:
    blocks = [f"<!-- ailaai-cell:{cell.index:02d}:{cell.kind} -->\n{cell.source}" for cell in cells]
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def sync_source_cells(cells: list[Cell], write: bool) -> bool:
    """Embed real function bodies as editable cells, with a checked source pointer."""
    changed = False
    for cell in cells:
        if cell.kind != "code" or not cell.source.startswith("# ailaai-source: "):
            continue
        marker = cell.source.splitlines()[0]
        relative, names = marker.removeprefix("# ailaai-source: ").split("::")
        source_path = ROOT / relative
        if not source_path.resolve().is_relative_to((ROOT / "src/ailaai").resolve()):
            raise ValueError("Source cells must refer to src/ailaai.")
        source = source_path.read_text(encoding="utf-8")
        lines = source.splitlines()
        symbols = {node.name: node for node in ast.parse(source).body
                   if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        bodies = []
        for name in names.split(","):
            node = symbols[name]
            first = min([node.lineno] + [d.lineno for d in node.decorator_list])
            bodies.append("\n".join(lines[first - 1:node.end_lineno]))
        expected = marker + "\n" + "\n\n\n".join(bodies)
        if cell.source != expected:
            if not write:
                raise ValueError(f"Embedded source differs: {relative}::{names}")
            cell.source = expected
            changed = True
    return changed


def _notebook(path: Path, cells: list[Cell]) -> dict:
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    config_hashes = {}
    for config in sorted((ROOT / "configs").glob("*.json")):
        config_hashes[str(config.relative_to(ROOT))] = hashlib.sha256(config.read_bytes()).hexdigest()
    notebook_cells = []
    for cell in cells:
        common = {"cell_type": cell.kind, "metadata": {}, "source": cell.source.splitlines(keepends=True),
                  "id": hashlib.sha1(f"{path.stem}:{cell.index}".encode()).hexdigest()[:8]}
        if cell.kind == "code":
            common.update({"execution_count": None, "outputs": []})
        notebook_cells.append(common)
    return {
        "cells": notebook_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
            "ailaai": {"source_sha256": source_hash, "config_sha256": config_hashes,
                       "cell_count": len(cells)},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build(write: bool) -> list[Path]:
    written = []
    bootstrap = (SOURCE_ROOT / "shared/bootstrap.py").read_text(encoding="utf-8").strip()
    for name, (expected_cells, expected_markdown) in NOTEBOOKS.items():
        source_path = SOURCE_ROOT / f"{name}.md"
        cells = _read_cells(source_path)
        source_changed = sync_source_cells(cells, write)
        if len(cells) != expected_cells or sum(cell.kind == "markdown" for cell in cells) != expected_markdown:
            raise ValueError(f"{name}: expected {expected_cells} cells ({expected_markdown} Markdown).")
        # Every notebook shares the same Colab import and installation setup.
        if cells[1].kind != "code":
            raise ValueError(f"{name}: cell 01 must be the common setup code.")
        if cells[1].source != bootstrap:
            if not write:
                raise ValueError(f"{name}: cell 01 differs from notebook_sources/shared/bootstrap.py.")
            cells[1].source = bootstrap
            source_changed = True
        if source_changed:
            _write_cells(source_path, cells)
        notebook = _notebook(source_path, cells)
        if name == "03_ensemble_threshold_submission":
            content = "\n".join(cell.source for cell in cells if cell.kind == "markdown").casefold()
            hits = [term for term in BANNED_NB3 if term in content]
            if hits:
                raise ValueError(f"NB3 contains wording to remove: {hits}")
        destination = OUTPUT_ROOT / f"{name}.ipynb"
        rendered = json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"
        if write:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(rendered, encoding="utf-8")
        elif not destination.is_file() or destination.read_text(encoding="utf-8") != rendered:
            raise ValueError(f"{destination} is missing or differs from its notebook source.")
        written.append(destination)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true", help="regenerate committed notebooks")
    group.add_argument("--check", action="store_true", help="fail when generated content has drifted")
    args = parser.parse_args()
    try:
        paths = build(write=args.write)
    except (OSError, ValueError, SyntaxError, json.JSONDecodeError) as exc:
        print(f"Notebook build failed: {exc}", file=sys.stderr)
        return 1
    for path in paths:
        print(path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
