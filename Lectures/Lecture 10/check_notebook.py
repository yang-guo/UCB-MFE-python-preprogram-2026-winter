"""Execute the lecture from a fresh kernel and save a checked copy.

Usage: uv run --group notebook python check_notebook.py
Use --output 1_stock_research_harness.ipynb to refresh the lecture's saved output.
"""

import argparse
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/notebook-check.ipynb")
    args = parser.parse_args()
    notebook = nbformat.read(ROOT / "1_stock_research_harness.ipynb", as_version=4)
    nbformat.validate(notebook)
    # Keep the notebook kernel on the same interpreter as the pinned environment.
    kernel = KernelManager(kernel_name="python3")
    kernel.kernel_spec.argv[0] = sys.executable
    client = NotebookClient(notebook, km=kernel, timeout=240,
                            resources={"metadata": {"path": str(ROOT)}}, allow_errors=False)
    client.execute(cleanup_kc=True)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert all(cell.execution_count is not None for cell in code_cells)
    assert not any(output.output_type == "error" for cell in code_cells for output in cell.outputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, args.output)
    print(f"Executed {len(code_cells)} code cells in a fresh kernel: {args.output}")


if __name__ == "__main__":
    main()
