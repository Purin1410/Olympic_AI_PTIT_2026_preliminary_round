"""Execute appendix 04 and its committed bundle in an isolated fresh CPU kernel."""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NAME = "04_negative_results_and_ablation"


def main():
    bundle = ROOT / "data/negative_results"
    manifest = json.loads((bundle / "source_manifest.json").read_text())
    for name, expected in manifest["outputs_sha256"].items():
        observed = hashlib.sha256((bundle / name).read_bytes()).hexdigest()
        if observed != expected:
            raise ValueError(f"Bundle hash mismatch: {name}")
    notebook = nbformat.read(ROOT / "notebooks" / f"{NAME}.ipynb", as_version=4)
    nbformat.validate(notebook)
    if len(notebook.cells) != 18 or sum(c.cell_type == "markdown" for c in notebook.cells) != 10:
        raise ValueError("Expected 18 cells: 10 Markdown and 8 code")
    assertion_count = sum(isinstance(node, ast.Assert) for cell in notebook.cells
                          if cell.cell_type == "code" for node in ast.walk(ast.parse(cell.source)))
    summary = '''
import platform
import importlib.metadata
print(json.dumps({
    "rows": len(oof),
    "wavelet": {k: wavelet_result[k] for k in ["fixes", "breaks", "net errors (B − A)"]},
    "fake_low_edge": {"n": len(target_fake), "baseline_fn": int((fake_base == 0).sum()), "weighted_fn": int((fake_weighted == 0).sum())},
    "real_low_edge": {"n": len(target_real), "baseline_fp": int((real_base == 1).sum()), "weighted_fp": int((real_weighted == 1).sum())},
    "gates": {name: recorded_gates[name] for name in gate_names},
    "gray_median": gray_median, "color_median": color_median,
    "threshold_macro_f1_percent": threshold_results["Macro-F1 (%)"].tolist(),
    "crossfit_transitions": crossfit_transitions,
    "median_transitions": median_transitions,
    "python": platform.python_version(),
    "pandas": importlib.metadata.version("pandas")
}, ensure_ascii=False))
'''
    notebook.cells.append(nbformat.v4.new_code_cell(summary))
    with tempfile.TemporaryDirectory(prefix="ailaai04-replay-") as directory:
        isolated = Path(directory)
        shutil.copytree(bundle, isolated / "data/negative_results")
        (isolated / "notebooks").mkdir()
        started = time.perf_counter()
        executed = NotebookClient(notebook, timeout=60, kernel_name="python3",
                                  resources={"metadata": {"path": str(isolated / "notebooks")}}).execute()
        wall_seconds = time.perf_counter() - started
    report = json.loads("".join(output.text for output in executed.cells[-1].outputs
                               if output.output_type == "stream"))
    code_cells = [c for c in executed.cells[:-1] if c.cell_type == "code"]
    png_outputs = sum("image/png" in o.get("data", {}) for c in code_cells for o in c.outputs)
    if png_outputs != 3 or any(c.execution_count is None for c in code_cells):
        raise ValueError("Expected all 8 code cells executed and 3 PNG display outputs")
    code_seconds = 0.0
    for cell in code_cells:
        timing = cell.metadata["execution"]
        code_seconds += (datetime.fromisoformat(timing["shell.execute_reply"]) -
                         datetime.fromisoformat(timing["iopub.execute_input"])).total_seconds()
    report.update(status="passed", scope="isolated fresh local CPU kernel; historical replay only",
                  executed_code_cells=len(code_cells), notebook_assertions=assertion_count,
                  png_display_outputs=png_outputs, bundle_hashes="passed",
                  wall_seconds=round(wall_seconds, 4), code_seconds=round(code_seconds, 4),
                  source_sha256=executed.metadata.ailaai.source_sha256,
                  bundle_manifest_sha256=hashlib.sha256((bundle / "source_manifest.json").read_bytes()).hexdigest())
    destination = ROOT / "evidence/negative_results/replay_verification.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
