"""Show live progress of a full run without loading the synthetic CSV."""
import argparse
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=ROOT/'outputs/reproduction_full')
    out = p.parse_args().output_dir
    bo_out = out if (out/'config.json').exists() else out/'02_optimization'
    state = read(out/'pipeline.json')
    print(f'Run: {out}')
    lock = out/'run.lock' if (out/'run.lock').exists() else bo_out/'run.lock'
    if lock.exists():
        pid = int(lock.read_text().strip())
        try:
            os.kill(pid, 0); active = True
        except ProcessLookupError:
            active = False
        except PermissionError:
            active = None
        label='PID check restricted' if active is None else 'active' if active else 'exited'
        print(f'Process PID {pid}: {label}')
    print(f"Pipeline: {state.get('status', 'standalone BO' if (bo_out/'config.json').exists() else 'not started')}")
    print('Completed stages:', ', '.join(str(row['stage']) for row in state.get('stages', [])) or 'none')
    full_config=read(out/'run_config.json')
    if full_config.get('selection'):
        print('Hyperparameters:',full_config['selection'],json.dumps(full_config['parameters']))
        print(f"Test: {full_config['states']} states; {len(full_config['origins'])} weekly origins; {full_config['targets_per_model']:,} targets/model")
        progress=read(Path(full_config['archive_dir'])/'progress.json')
        if progress: print('Weekly archive:',json.dumps(progress))
        series_path=out/'04_prediction/series_metrics.csv'
        if series_path.exists():
            with series_path.open() as handle: states=list(csv.DictReader(handle))
            print(f"Prediction: {len(states)}/{full_config['states']} states complete")
    if state.get('status') in ['waiting_for_bo','waiting_for_paper']:
        print('Waiting for completed selection:',state.get('selection_dir'))
        print('Paper baseline run:',state.get('paper_run'))
    synthetic = read(out/'01_synthetic/metadata.json')
    if synthetic:
        print(f"Synthetic: complete; {synthetic['attempted_total']:,} attempts, {synthetic['accepted_total']:,} accepted")
    else:
        checkpoint = read(out/'01_synthetic/checkpoint.json')
        if checkpoint:
            print(f"Synthetic: {checkpoint['next_index']:,}/{checkpoint['total_attempts']:,} attempts; "
                  f"{sum(checkpoint['accepted'].values()):,} accepted")
    config = read(bo_out/'config.json')
    if config:
        history = read(bo_out/'checkpoint.json').get('history', [])
        progress = read(bo_out/'progress.json')
        print(f"BO: {len(history)}/{config['initial_points']+config['iterations']} evaluations complete; "
              f"full_search_budget={config['full_search_budget']}")
        print('BO progress:', json.dumps(progress, ensure_ascii=False))
        if config.get('paper_comparison'):
            comparison = read(bo_out/'paper_comparison/progress.json')
            print('Paired paper comparison:', json.dumps(comparison or {'status':'pending BO selection'},ensure_ascii=False))
            if comparison.get('status')=='complete': print('Comparison report:',bo_out/'paper_comparison/report.md')
    evaluation = read(out/'05_evaluation/summary.json')
    if evaluation:
        import sys
        sys.path.insert(0, str(ROOT/'experiments/forecasting'))
        from evaluation_report import format_metrics
        print(format_metrics(evaluation))
        print('Report:', out/'05_evaluation/report.md')
    if state.get('error'): print('Last error:', state['error'])


if __name__ == '__main__':
    main()
