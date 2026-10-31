"""Execute a project notebook using the current Python environment."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def execute_notebook(path: Path, *, working_dir: Path = ROOT, timeout: int = 3600):
    """Save executed cells only after the entire notebook succeeds."""
    path = Path(path).resolve()
    working_dir = Path(working_dir).resolve()
    if not path.is_file() or path.suffix != '.ipynb':
        raise ValueError(f'Notebook file required: {path}')
    if not working_dir.is_dir() or timeout < 1:
        raise ValueError('Need an existing working directory and a positive timeout')
    cache = Path(tempfile.gettempdir())/'infections-notebooks'
    for env, name in [('MPLCONFIGDIR', 'matplotlib'), ('JUPYTER_RUNTIME_DIR', 'jupyter'), ('IPYTHONDIR', 'ipython')]:
        directory = cache/name
        directory.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(env, str(directory))
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager

    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    manager = KernelManager(kernel_name='python3')
    manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
    NotebookClient(notebook, km=manager, timeout=timeout,
                   resources={'metadata': {'path': str(working_dir)}},
                   extra_arguments=['--HistoryManager.hist_file=:memory:']).execute()
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    print(f'Executed {sum(c.cell_type == "code" for c in notebook.cells)} code cells: {path}')
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notebook', type=Path)
    parser.add_argument('--working-dir', type=Path, default=ROOT)
    parser.add_argument('--timeout', type=int, default=3600, help='Seconds per cell')
    args = parser.parse_args()
    execute_notebook(args.notebook, working_dir=args.working_dir, timeout=args.timeout)
