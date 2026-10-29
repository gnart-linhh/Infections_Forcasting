"""Normalize the original scores_tot_w_wis_components.csv for paired comparison."""
import argparse
import csv
from datetime import date,timedelta
import hashlib
import json
from pathlib import Path


def checksum(path):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''): digest.update(chunk)
    return digest.hexdigest()


def convert(source,locations,output):
    with locations.open() as f:
        mapping={int(r['STATE']):r['NAME'] for r in csv.DictReader(f) if int(r['STATE'])>0}
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists(): raise ValueError('Output already exists')
    written=0;excluded=0
    with source.open() as f,output.open('w',newline='') as g:
        w=csv.DictWriter(g,fieldnames=['series','forecast_origin','target_end_date','horizon','model','abs_error','wis','submission_date'],lineterminator='\n')
        w.writeheader()
        for r in csv.DictReader(f):
            location=r['location']
            if location=='US' or len(location)>2 or not location.isdigit() or int(location) not in mapping:
                excluded+=1;continue
            h=int(float(r['horizon']))
            if not 1<=h<=4: excluded+=1;continue
            target=date.fromisoformat(r['target_end_date'][:10])
            # The R comparison joins on location,target,horizon, not forecast_date.
            # Canonical origin expresses that exact target/horizon key. Keep the
            # submission date separately; do not claim it is the observed cutoff.
            w.writerow(dict(series=mapping[int(location)],forecast_origin=(target-timedelta(days=7*h)).isoformat(),
                target_end_date=target.isoformat(),horizon=h,model=r['model'],abs_error=r['abs_error'],wis=r['wis'],
                submission_date=r.get('forecast_date','')))
            written+=1
    result=dict(rows=written,excluded=excluded,
        join_convention='Origin inferred from target-horizon to match source location/target/horizon join; original submission date preserved',
        duplicates='Comparator refuses duplicate keys; choose one submission per model/target/horizon explicitly before comparison',
        source_sha256=checksum(source))
    output.with_suffix('.source.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--locations',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps(convert(a.input,a.locations,a.output),indent=2))


if __name__=='__main__': main()
