"""Stream the original tdat_list_tot_weekly.csv into per-state snapshot CSVs."""
import argparse
import csv
from datetime import date
import hashlib
import json
import math
import os
from pathlib import Path


def checksum(path):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''): digest.update(chunk)
    return digest.hexdigest()


def convert(source,truth_manifest,out):
    with truth_manifest.open() as f: entries=list(csv.DictReader(f))
    states={r['series']:r for r in entries}
    if len(states)!=len(entries): raise ValueError('Duplicate manifest series')
    if out.exists() and any(out.iterdir()): raise ValueError('Output directory must be empty')
    out.mkdir(parents=True,exist_ok=True)
    handles={};writers={};counts={state:0 for state in states};future=0;skipped=0
    try:
        with source.open() as f:
            for r in csv.DictReader(f):
                state=r['location_name']
                if state not in states: skipped+=1;continue
                vintage=date.fromisoformat(r['as_of'][:10]);target=date.fromisoformat(r['target_end_date'][:10])
                value=float(r['value'])
                if not math.isfinite(value): raise ValueError('Nonfinite snapshot value')
                if target>vintage: future+=1;continue  # same history truncation as source run_smoa.R
                if state not in writers:
                    path=out/f'{len(writers)+1:02d}_asof.csv';handle=path.open('w',newline='')
                    handles[state]=(handle,path);writers[state]=csv.writer(handle,lineterminator='\n')
                    writers[state].writerow(['as_of','target_end_date','value'])
                writers[state].writerow([vintage.isoformat(),target.isoformat(),value]);counts[state]+=1
    finally:
        for handle,path in handles.values(): handle.close()
    missing=[state for state,n in counts.items() if n==0]
    if missing: raise ValueError(f'No snapshots for states: {missing}')
    with (out/'manifest.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['series','truth','as_of_truth'],lineterminator='\n');w.writeheader()
        for state,entry in states.items():
            truth=(truth_manifest.parent/entry['truth']).resolve()
            w.writerow(dict(series=state,truth=os.path.relpath(truth,out),as_of_truth=handles[state][1].name))
    meta=dict(status='complete',rows_per_series=counts,future_targets_removed=future,other_locations_skipped=skipped,
        source_sha256=checksum(source),
        convention='as_of labels preserved from source; get_data_as_of.R downloaded origin+1 then labelled origin, not an independently verified same-day information cutoff')
    (out/'provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
    return meta


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--truth-manifest',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    print(json.dumps(convert(a.input,a.truth_manifest,a.output_dir),indent=2))


if __name__=='__main__': main()
