"""Prepare genuine historical JHU vintages for the preregistered Experiment 4 calendar."""
from __future__ import annotations
import argparse
from datetime import date, timedelta
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments'))
sys.path.insert(0, str(ROOT / 'experiments/data'))
from runtime import atomic_json, file_hash
from fetch_paper_data import download, weekly_states


def calendar():
    # More frequent observations in the genuinely early epidemic; no error-based selection.
    early = pd.date_range('2020-04-04', '2020-09-05', freq='7D')
    later = pd.date_range('2020-10-03', '2022-10-01', freq='28D')
    return [d.date().isoformat() for d in early.union(later)]


def prepare(out, manifest, cache, offline=False):
    out.mkdir(parents=True, exist_ok=True)
    manifest = manifest.resolve()
    states = pd.read_csv(manifest).series.tolist()
    if len(states) != 50 or len(set(states)) != 50:
        raise ValueError('Experiment 4 requires the 50-state truth manifest')
    # Pin the already acquired official covidData index, rather than following master silently.
    index_path = cache / 'raw/jhu_us_cases_data_links.csv'
    links = pd.read_csv(index_path).sort_values('date')
    records, sources, panel = [], [], []
    for origin in calendar():
        requested = (date.fromisoformat(origin) + timedelta(days=1)).isoformat()
        candidates = links.loc[links.date <= requested]
        if candidates.empty or requested > links.date.max():
            raise ValueError(f'No indexed vintage for {origin}')
        selected = candidates.iloc[-1]
        if (date.fromisoformat(requested) - date.fromisoformat(selected.date)).days > 2:
            raise ValueError(f'Stale source vintage for {origin}')
        name = f'jhu_us_cases_{selected.date}.csv'
        raw = cache / 'raw' / name
        if not raw.exists():
            raw = out / 'raw' / name
        record = dict(forecast_origin=origin, data_cutoff=requested,
                      issuance_date=(date.fromisoformat(origin)+timedelta(days=2)).isoformat(),
                      source_date=selected.date, source_url=selected.file_link)
        if offline and not raw.exists():
            records.append(dict(record, status='not_cached'))
            continue
        download(selected.file_link, raw, sources)
        weekly = weekly_states(raw, states)
        weekly = weekly.loc[weekly.target_end_date <= pd.Timestamp(origin)].copy()
        expected = pd.date_range('2020-01-25', origin, freq='7D')
        for state, group in weekly.groupby('series'):
            if not pd.DatetimeIndex(group.target_end_date).equals(expected):
                raise ValueError(f'Incomplete historical calendar: {state} {origin}')
        weekly['forecast_origin'] = origin
        panel.append(weekly[['forecast_origin', 'series', 'target_end_date', 'value']])
        records.append(dict(record, status='available', raw_sha256=file_hash(raw)))
        pd.DataFrame(records).to_csv(out / 'origins.csv', index=False)
        print(f'Vintage {origin}: {len(states)} states, {len(expected)} observed weeks', flush=True)
    pd.DataFrame(records).to_csv(out / 'origins.csv', index=False)
    if panel:
        combined = pd.concat(panel, ignore_index=True)
        temp = out / 'panel.csv.tmp'
        combined.to_csv(temp, index=False)
        temp.replace(out / 'panel.csv')
    atomic_json(out / 'provenance.json', dict(
        status='complete' if len(panel) == len(calendar()) else 'incomplete',
        planned_origins=calendar(), available_origins=[r['forecast_origin'] for r in records if r['status']=='available'],
        states=states, truth_manifest=str(manifest), truth_manifest_sha256=file_hash(manifest),
        source_index=str(index_path.resolve()), source_index_sha256=file_hash(index_path),
        source='reichlab/covidData -> immutable CSSEGISandData/COVID-19 commits',
        protocol='Raw county cumulative -> daily differences -> Sunday-Saturday state sums; no retrospective correction.',
        timing='Saturday origin; Sunday data cutoff (LANL origin+1); Monday issuance. No data after issuance.',
        script_sha256=file_hash(Path(__file__)), sources=sources,
        panel_sha256=file_hash(out/'panel.csv') if panel else None))
    print(f'{len(panel)}/{len(calendar())} vintages prepared: {out}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs/epidemic_panel')
    parser.add_argument('--manifest', type=Path, default=ROOT/'outputs/truth_states/manifest.csv')
    parser.add_argument('--cache', type=Path, default=ROOT/'outputs/paper_data')
    parser.add_argument('--offline', action='store_true', help='Explicit incomplete panel of cached vintages only')
    args = parser.parse_args()
    prepare(args.output_dir, args.manifest, args.cache, args.offline)
