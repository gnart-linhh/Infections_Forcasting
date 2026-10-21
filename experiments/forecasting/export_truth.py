"""Optional one-time conversion of an existing truth_weekly RData using Python."""
import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    import pyreadr  # optional; backtest itself accepts CSV and does not need this
    if a.output.exists():
        p.error('Output already exists')
    truth = pyreadr.read_r(str(a.input))['truth_weekly']
    a.output.parent.mkdir(parents=True, exist_ok=True)
    truth[['target_end_date', 'value']].to_csv(a.output, index=False)


if __name__ == '__main__':
    main()
