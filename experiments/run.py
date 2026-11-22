"""Central entry point for the Robust sMOA cold-start study."""
import argparse
from pathlib import Path

from experiments.robust.pipeline import EXPERIMENTS, load_config, run, regenerate
from experiments.robust.data import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/experiments.json')
    parser.add_argument('--experiment', choices=EXPERIMENTS, default='all')
    parser.add_argument('--smoke', action='store_true', help='Labelled small real and independent-synthetic subset')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--regenerate', type=Path, help='Regenerate tables and comparison figure from saved predictions')
    args = parser.parse_args()
    if args.regenerate:
        regenerate(args.regenerate)
    else:
        run(load_config(args.config), args.experiment, args.smoke, args.output_dir)


if __name__ == '__main__':
    main()
