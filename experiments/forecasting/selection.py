"""Load and validate the actual evaluated BO winner for prediction."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime import file_hash
from model import WINDOW_FILTER


def apply_selection(args):
    directory=getattr(args,'selection_dir',None)
    if not directory: return args
    directory=Path(directory)
    metadata=json.loads((directory/'metadata.json').read_text())
    if metadata.get('status')!='complete': raise ValueError('Selection run is incomplete')
    params=json.loads((directory/'best_params.json').read_text())
    for key in ['num_curves','k','closest']:
        if type(params.get(key)) is not int or params[key]<1: raise ValueError('Invalid selected hyperparameters')
    library=directory/'best_library/curves.csv'
    expected=metadata.get('artifacts_sha256',{})
    for relative in ['best_params.json','best_library/curves.csv']:
        if relative not in expected: raise ValueError('Selection lacks post-audit artifact hashes; rerun optimization')
        if file_hash(directory/relative)!=expected[relative]:
            raise ValueError(f'Modified selection artifact: {relative}')
    protocol=json.loads((directory/'config.json').read_text())
    if protocol.get('window_filter') != WINDOW_FILTER:
        raise ValueError('BO used a different/legacy window filter. Rerun optimization in a fresh directory; '
                         'old winners remain historical results, not selections for nonconstant_v2.')
    selected_smoother=protocol.get('smoothing')
    if selected_smoother and getattr(args,'smoothing',None)!=selected_smoother:
        raise ValueError(f'BO used {selected_smoother}, prediction requests {args.smoothing}. '
                         'Run BO again with the new smoother, or explicitly use the legacy smoother for a legacy comparison.')
    args.synthetic=library;args.k=params['k'];args.closest=params['closest']
    args.selection_protocol=protocol
    return args
