"""Fetch versioned JHU truth and official Hub forecasts without cloning or R."""
from __future__ import annotations
import argparse
from datetime import date, timedelta, datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LINKS = 'https://raw.githubusercontent.com/reichlab/covidData/master/data/jhu_us_cases_data_links.csv'
HUB = 'https://raw.githubusercontent.com/reichlab/covid19-forecast-hub/master/data-processed'
STATES = {'California': '06', 'Florida': '12', 'New York': '36', 'Texas': '48'}
FIRST_TARGETS = ('2020-03-21','2020-04-18','2020-05-23','2020-06-20','2020-07-18','2020-08-15')
FIGURE_VINTAGES = ('2020-08-15','2020-08-22','2020-08-29','2020-09-05')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(url, path, log, optional=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not path.exists():
            with urlopen(Request(url, headers={'User-Agent':'smoa-python-reproduction'}), timeout=120) as response:
                content = response.read()
            tmp = path.with_suffix(path.suffix+'.tmp')
            tmp.write_bytes(content)
            tmp.replace(path)
        log.append(dict(url=url, file=str(path), sha256=sha(path), status='downloaded'))
        return path
    except HTTPError as error:
        if error.code != 404 or not optional:
            raise
        log.append(dict(url=url, file=str(path), status='not_in_archive', http_status=404))
        return None


def weekly_states(path, states=STATES):
    """covidData: sum counties, diff cumulative, sum Sunday-Saturday, drop incomplete week."""
    frame = pd.read_csv(path)
    columns = [name for name in frame if re.fullmatch(r'\d{1,2}/\d{1,2}/\d{2,4}', name)]
    if not columns or frame[columns].isna().any().any():
        raise ValueError('Missing daily cumulative JHU values')
    dates = pd.to_datetime(columns, format='%m/%d/%y')
    if not (dates.to_series().diff().iloc[1:] == pd.Timedelta(days=1)).all():
        raise ValueError('Nonconsecutive JHU daily calendar')
    end = dates + pd.to_timedelta((5-dates.dayofweek) % 7, unit='D')
    rows = []
    for state in states:
        selected = frame.loc[frame.Province_State == state, columns]
        if selected.empty:
            raise ValueError(f'Missing state {state}')
        cumulative = selected.sum(axis=0).to_numpy()
        incident = pd.Series(cumulative, index=dates).diff()
        incident.iloc[0] = cumulative[0]
        weekly = pd.DataFrame({'target_end_date':end, 'value':incident.to_numpy()})
        weekly = weekly.loc[(end <= dates.max()) & (end >= pd.Timestamp('2020-01-25'))]
        weekly = weekly.groupby('target_end_date', as_index=False).value.sum()
        weekly['series'] = state
        rows.append(weekly)
    return pd.concat(rows, ignore_index=True)


def fetch(out, as_of_dates=FIGURE_VINTAGES, states=STATES, hub=True):
    out.mkdir(parents=True, exist_ok=True)
    log = []
    links = pd.read_csv(download(LINKS, out/'raw/jhu_us_cases_data_links.csv', log))
    for label in as_of_dates:
        # LANL get_data_as_of.R requests load_truth(as_of=Saturday+1),
        # then writes the Saturday label into the as_of column.
        requested = (date.fromisoformat(label)+timedelta(days=1)).isoformat()
        if requested > links.date.max():
            raise ValueError('Requested vintage exceeds cached covidData index; update source index explicitly')
        candidates = links.loc[links.date <= requested].sort_values('date')
        if candidates.empty: raise ValueError(f'No vintage for {requested}')
        selected = candidates.iloc[-1]
        raw = download(selected.file_link, out/f'raw/jhu_us_cases_{selected.date}.csv', log)
        weekly = weekly_states(raw, states)
        weekly = weekly.loc[weekly.target_end_date <= pd.Timestamp(label)]
        weekly['as_of'] = label
        for state, group in weekly.groupby('series'):
            group[['as_of','target_end_date','value']].to_csv(out/f'{state}_{label}.csv',index=False)
        log.append(dict(as_of_label=label, requested_as_of=requested, issue_date=selected.date,
                        source_commit=selected.file_link.split('/')[5], source_url=selected.file_link))
    forecasts = []
    if hub:
        for model in ['COVIDhub-baseline','COVIDhub-4_week_ensemble']:
            for target in FIRST_TARGETS:
                submission = (date.fromisoformat(target)-timedelta(days=5)).isoformat()
                filename = f'{submission}-{model}.csv'
                path = download(f'{HUB}/{model}/{filename}',out/'raw/hub'/filename,log,optional=True)
                if path is None: continue
                frame = pd.read_csv(path,dtype={'location':str})
                frame = frame.loc[frame.location.isin(states.values()) & frame.target.str.fullmatch('[1-4] wk ahead inc case')].copy()
                if frame.empty: continue
                frame['model'] = model
                frame['series'] = frame.location.map({v:k for k,v in states.items()})
                frame['horizon'] = frame.target.str[0].astype(int)
                expected = pd.to_datetime(submission)+pd.to_timedelta(5+7*(frame.horizon-1),unit='D')
                if not (pd.to_datetime(frame.target_end_date).to_numpy()==expected.to_numpy()).all():
                    raise ValueError('Hub target/submission calendar mismatch')
                frame['forecast_origin'] = (date.fromisoformat(target)-timedelta(days=7)).isoformat()
                forecasts.append(frame)
        if not forecasts: raise ValueError('No official incident case forecasts downloaded')
        pd.concat(forecasts,ignore_index=True).to_csv(out/'hub_forecasts.csv',index=False)
    provenance = dict(retrieved_utc=datetime.now(timezone.utc).isoformat(),
        protocol='covidData JHU state aggregation; unadjusted incidence, Sunday-Saturday; LANL Saturday+1 retrieval',
        script_sha256=sha(Path(__file__)), states=list(states), sources=log)
    path = out/'provenance.json'
    previous = json.loads(path.read_text()) if path.exists() else {}
    # Reusing unchanged cached inputs must not invalidate experiment resume
    # merely because the preparation command was invoked a second time.
    if {k:v for k,v in previous.items() if k!='retrieved_utc'} != {k:v for k,v in provenance.items() if k!='retrieved_utc'}:
        path.write_text(json.dumps(provenance,indent=2)+'\n')
    print(f'Historical snapshots and official forecasts: {out}',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/paper_data')
    parser.add_argument('--as-of-dates',nargs='+',default=list(FIGURE_VINTAGES))
    parser.add_argument('--skip-hub',action='store_true')
    args = parser.parse_args()
    fetch(args.output_dir,args.as_of_dates,hub=not args.skip_hub)


if __name__ == '__main__': main()
