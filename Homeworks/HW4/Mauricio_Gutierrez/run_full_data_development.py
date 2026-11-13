"""Execute notebook 06 in order and save its outputs. Never execute notebook 04.

Run from the repository root with the course Python environment.
This runner uses IPython directly; it does not start a separate Jupyter kernel.
"""
from pathlib import Path
import os
import nbformat

HW = Path(__file__).resolve().parent
ROOT = HW.parents[2]
OUTPUT = HW / 'outputs/full_data_development'
OUTPUT.mkdir(parents=True, exist_ok=True)
os.environ['IPYTHONDIR'] = str(OUTPUT / '.ipython')
os.environ['MPLCONFIGDIR'] = str(OUTPUT / '.matplotlib')
os.environ['MPLBACKEND'] = 'Agg'

from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output


def main():
    os.chdir(ROOT)
    path = HW / '06_full_data_development.ipynb'
    notebook = nbformat.read(path, as_version=4)
    shell = InteractiveShell.instance()
    count = 0
    for cell in notebook.cells:
        if cell.cell_type != 'code':
            continue
        count += 1
        print(f'Running cell {count}', flush=True)
        with capture_output(stdout=True, stderr=True, display=True) as captured:
            result = shell.run_cell(cell.source, store_history=False)
        cell.execution_count = count
        cell.outputs = []
        for name, text in [('stdout', captured.stdout), ('stderr', captured.stderr)]:
            if text:
                cell.outputs.append(nbformat.v4.new_output('stream', name=name, text=text))
                print(text.strip(), flush=True)
        for output in captured.outputs:
            cell.outputs.append(nbformat.v4.new_output('display_data', data=output.data, metadata=output.metadata))
        nbformat.validate(notebook)
        nbformat.write(notebook, path)
        if not result.success:
            raise result.error_before_exec or result.error_in_exec
    print(f'Completed all {count} cells.', flush=True)


if __name__ == '__main__':
    main()
