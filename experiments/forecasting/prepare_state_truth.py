"""One-time Python conversion of state truth archives to independent CSV inputs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    sources = sorted(a.input_dir.glob('*.RData'))
    if not sources:
        p.error('No RData truth archives found')
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        p.error('Output directory must be empty')
    import pyreadr  # conversion only; the experiment itself uses CSV/NumPy/SciPy
    rows, provenance = [], []
    a.output_dir.mkdir(parents=True, exist_ok=True)
    for source in sources:
        truth = pyreadr.read_r(str(source))['truth_weekly']
        names = truth['location_name'].unique()
        if len(names) != 1:
            raise ValueError(f'Expected one state: {source}')
        target = a.output_dir/f'{source.stem}.csv'
        truth[['target_end_date', 'value']].to_csv(target, index=False)
        rows.append(dict(series=names[0], truth=target.name, as_of_truth=''))
        provenance.append(dict(series=names[0], source_name=source.name,
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            csv_sha256=hashlib.sha256(target.read_bytes()).hexdigest(), rows=len(truth),
            negative_value_count=int((truth['value'] < 0).sum()), minimum_value=float(truth['value'].min())))
    with (a.output_dir/'manifest.csv').open('w', newline='') as handle:
        w = csv.DictWriter(handle, fieldnames=['series', 'truth', 'as_of_truth'])
        w.writeheader(); w.writerows(rows)
    (a.output_dir/'provenance.json').write_text(json.dumps(dict(history_mode='final',
        source='https://github.com/lanl/precog/tree/main/smoa/data/state_truths',
        pyreadr=pyreadr.__version__, files=provenance), indent=2)+'\n')
    print(f'Exported {len(rows)} series to {a.output_dir}')


if __name__ == '__main__':
    main()
