"""Execute notebook 04 reference mode in a fresh CPU kernel.

This checks all notebook cells with the shared setup. It does not train models.
Output paths are isolated; the committed historical bundle is left unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NAME = "04_negative_results_and_ablation"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/verification/nb4_reference_verification.json")
    args = parser.parse_args()
    bundle = ROOT / "data/negative_results"
    manifest = json.loads((bundle / "source_manifest.json").read_text())
    for name, expected in manifest["outputs_sha256"].items():
        if hashlib.sha256((bundle / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Historical bundle changed: {name}")
    notebook = nbformat.read(ROOT / "notebooks" / f"{NAME}.ipynb", as_version=4)
    nbformat.validate(notebook)
    config_cells = [cell for cell in notebook.cells if cell.cell_type == "code"
                    and 'MODE = "train"' in cell.source and 'FOLDS = [0]' in cell.source]
    if len(config_cells) != 1:
        raise ValueError("Expected one cell containing the reference-mode configuration.")
    config_cells[0].source = config_cells[0].source.replace('MODE = "train"', 'MODE = "reference"').replace('FOLDS = [0]', 'FOLDS = [0, 1, 2, 3, 4]')
    summary = 'print(json.dumps({"rows": len(results), "evaluation_rows": len(evaluation), "wavelet": wavelet_report, "models_trained": len(runs)}))'
    notebook.cells.append(nbformat.v4.new_code_cell(summary))
    with tempfile.TemporaryDirectory(prefix="ailaai04-reference-") as directory:
        names = ("AILAAI_ARTIFACT_ROOT", "AILAAI_OUTPUT_ROOT")
        previous = {name: os.environ.get(name) for name in names}
        try:
            for name in names:
                os.environ[name] = str(Path(directory) / name.lower())
            started = time.perf_counter()
            executed = NotebookClient(notebook, timeout=300, kernel_name="python3",
                                      resources={"metadata": {"path": str(ROOT / "notebooks")}}).execute()
            seconds = time.perf_counter() - started
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
    report = json.loads("".join(o.text for o in executed.cells[-1].outputs if o.output_type == "stream"))
    code = [c for c in executed.cells[:-1] if c.cell_type == "code"]
    png = sum("image/png" in o.get("data", {}) for c in code for o in c.outputs)
    if any(c.execution_count is None for c in code) or png < 3:
        raise ValueError("Not all reference cells or figures completed.")
    if report["rows"] != 2000 or report["models_trained"] != 0:
        raise ValueError("Reference coverage or mode is wrong.")
    if report["wavelet"]["fixes"] != 30 or report["wavelet"]["breaks"] != 183:
        raise ValueError("Historical paired comparison changed.")
    report.update(status="passed", scope="fresh local CPU kernel; reference predictions only",
                  executed_code_cells=len(code), png_display_outputs=png, wall_seconds=round(seconds, 3))
    destination = args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
