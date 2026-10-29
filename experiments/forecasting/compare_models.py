"""Paired evaluation against external model scores on exactly matching targets."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

KEYS=('series','forecast_origin','target_end_date','horizon')


def finite(value):
    try: number=float(value)
    except (ValueError,TypeError): return None
    return number if np.isfinite(number) else None


def compare(smoa_path,reference_path,out):
    with smoa_path.open() as f: own=[r for r in csv.DictReader(f) if r['model']=='smoa']
    with reference_path.open() as f: refs=list(csv.DictReader(f))
    index={tuple(r[k] for k in KEYS):r for r in own}
    if len(index)!=len(own): raise ValueError('Duplicate sMOA target')
    groups={};seen=set()
    for r in refs:
        key=tuple(r[k] for k in KEYS)
        if (r['model'],key) in seen: raise ValueError('Duplicate reference model target')
        seen.add((r['model'],key));groups.setdefault(r['model'],[]).append(r)
    report=[]
    for model,rows in sorted(groups.items()):
        paired=[(index[tuple(r[k] for k in KEYS)],r) for r in rows if tuple(r[k] for k in KEYS) in index]
        for metric in ['abs_error','wis']:
            candidates=[(finite(a.get(metric)),finite(b.get(metric))) for a,b in paired]
            valid=[(a,b) for a,b in candidates if a is not None and b is not None]
            if not valid: continue
            own_mean=float(np.mean([v[0] for v in valid]));ref_mean=float(np.mean([v[1] for v in valid]))
            report.append(dict(model=model,metric=metric,paired_targets=len(valid),reference_targets=len(rows),
                smoa_targets=len(own),unmatched_reference_targets=len(rows)-len(paired),
                excluded_nonfinite_or_missing=len(paired)-len(valid),smoa_score=own_mean,reference_score=ref_mean,
                ratio=own_mean/ref_mean if ref_mean>0 else None))
    if not report: raise ValueError('No finite, paired scores')
    out.mkdir(parents=True,exist_ok=True)
    (out/'comparison.json').write_text(json.dumps(dict(results=report,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [smoa_path,reference_path]},
        note='Reference scores must use the documented same quantile grid/scoring convention; no unpaired averaging'),indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--smoa-scores',type=Path,required=True);p.add_argument('--reference-scores',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    print(json.dumps(compare(a.smoa_scores,a.reference_scores,a.output_dir),indent=2))


if __name__=='__main__': main()
