"""Execute and save the EDA notebook using the current Python environment."""
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(tempfile.gettempdir())/'infections-forecasting-eda'
for env, directory in [('MPLCONFIGDIR','matplotlib'),('JUPYTER_RUNTIME_DIR','jupyter'),('IPYTHONDIR','ipython')]:
    location = CACHE/directory;location.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault(env,str(location))
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager


def main():
    path=Path(__file__).with_name('dataset_exploration.ipynb')
    notebook=nbformat.read(path,as_version=4)
    manager=KernelManager(kernel_name='python3')
    manager.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
    client=NotebookClient(notebook,km=manager,timeout=3600,resources={'metadata':{'path':str(ROOT)}},
                          extra_arguments=['--HistoryManager.hist_file=:memory:'])
    client.execute()
    nbformat.write(notebook,path)
    print(f'Executed {sum(c.cell_type=="code" for c in notebook.cells)} code cells: {path}')


if __name__=='__main__':
    main()
