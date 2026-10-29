"""Show live progress of a full run without loading the synthetic CSV."""
import argparse
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
    state = read(out/'pipeline.json')
    print(f'Run: {out}')
    if (out/'run.lock').exists():
        pid = int((out/'run.lock').read_text().strip())
        try:
            os.kill(pid, 0); active = True
        except ProcessLookupError:
            active = False
        except PermissionError:
            active = None
        label='PID check restricted' if active is None else 'active' if active else 'exited'
        print(f'Process PID {pid}: {label}')
    print(f"Pipeline: {state.get('status', 'not started')}")
    print('Completed stages:', ', '.join(str(row['stage']) for row in state.get('stages', [])) or 'none')
    synthetic = read(out/'01_synthetic/metadata.json')
    if synthetic:
        print(f"Synthetic: complete; {synthetic['attempted_total']:,} attempts, {synthetic['accepted_total']:,} accepted")
    else:
        checkpoint = read(out/'01_synthetic/checkpoint.json')
        if checkpoint:
            print(f"Synthetic: {checkpoint['next_index']:,}/{checkpoint['total_attempts']:,} attempts; "
                  f"{sum(checkpoint['accepted'].values()):,} accepted")
    config = read(out/'02_optimization/config.json')
    if config:
        history = read(out/'02_optimization/checkpoint.json').get('history', [])
        progress = read(out/'02_optimization/progress.json')
        print(f"BO: {len(history)}/{config['initial_points']+config['iterations']} evaluations complete; "
              f"full_search_budget={config['full_search_budget']}")
        print('BO progress:', json.dumps(progress, ensure_ascii=False))
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
