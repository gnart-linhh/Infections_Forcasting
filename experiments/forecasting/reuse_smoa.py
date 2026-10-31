"""Reuse issued sMOA values while removing an obsolete diagnostic model, with lineage."""
from __future__ import annotations
import csv
import ast
import json
import os
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime import atomic_json,file_hash


def clean_summary(value):
    if isinstance(value,dict):
        return {k:clean_summary(v) for k,v in value.items()
                if k!='baseline' and not k.startswith('baseline_') and k not in
                {'persistence','relative','mae_ratio','wis_ratio'}}
    if isinstance(value,list): return [clean_summary(v) for v in value]
    return value


def project_csv(source,target):
    """Keep source value strings exactly; never parse or recompute forecasts/quantiles."""
    with source.open(newline='') as reader,target.open('w',newline='') as writer:
        rows=csv.DictReader(reader)
        fields=[k for k in rows.fieldnames if k!='baseline' and not k.startswith('baseline_') and k not in {'mae_ratio','wis_ratio'}]
        output=csv.DictWriter(writer,fieldnames=fields);output.writeheader()
        count=0
        for row in rows:
            if 'model' in row and row['model']!='smoa': continue
            output.writerow({key:row[key] for key in fields});count+=1
    return dict(source_sha256=file_hash(source),output_sha256=file_hash(target),rows=count)


def reuse_predictions(previous,destination,config):
    """Project only a completed, matching full run; preserve original generation metadata."""
    previous=previous.resolve();destination=destination.resolve()
    state=json.loads((previous/'pipeline.json').read_text())
    old=json.loads((previous/'run_config.json').read_text())
    if state.get('status')!='complete': raise ValueError('Cannot reuse an incomplete full run')
    for key in ['parameters','library_sha256','origins','truth_sha256','packages']:
        if old[key]!=config[key]: raise ValueError(f'Reuse input differs: {key}')
    for key,value in old['method'].items():
        if key!='models' and config['method'].get(key)!=value:
            raise ValueError(f'Reuse method differs: {key}')
    # The numerical model, smoother, neighbor search and embedding remain identical.
    for name in ['model.py','smoothing.py','neighbors.py','embedding_cache.py','snapshots.py']:
        key='experiments/forecasting/'+name
        if old['source_sha256'][key]!=config['source_sha256'][key]:
            raise ValueError(f'Reuse numerical source differs: {name}')
    def numerical_uq(path):
        functions={}
        for node in ast.parse(path.read_text()).body:
            if isinstance(node,ast.FunctionDef):
                if node.name=='calibration_plan':
                    # Removing other output fields must leave the sMOA calculation unchanged.
                    for part in ast.walk(node):
                        if isinstance(part,ast.For) and isinstance(part.target,ast.Name) and part.target.id=='field':
                            part.iter=ast.List(elts=[ast.Constant('forecast')],ctx=ast.Load())
                functions[node.name]=ast.dump(node,include_attributes=False)
        return functions
    generated_uq=previous/'source_snapshot/experiments/forecasting/uncertainty.py'
    if not generated_uq.exists(): raise ValueError('Reuse requires the frozen UQ generation source')
    if numerical_uq(generated_uq)!=numerical_uq(Path(__file__).with_name('uncertainty.py')):
        raise ValueError('Reuse numerical UQ source differs')
    if destination.exists(): raise ValueError('Reuse destination must be absent')
    destination.mkdir(parents=True)
    provenance=dict(operation='sMOA-only projection of existing issued values; no refit or resampling',
        generation_run=str(previous),generation_run_config_sha256=file_hash(previous/'run_config.json'),
        generation_elapsed_seconds=state['elapsed_seconds'],models=['smoa'],
        processing_source_sha256=config['source_sha256'],csv_artifacts={})
    source=previous/'04_prediction'
    for folder in [source,*sorted(p for p in source.iterdir() if p.is_dir() and p.name.isdigit())]:
        relative=folder.relative_to(source);target=destination/relative;target.mkdir(exist_ok=True)
        for path in folder.glob('*.csv'):
            provenance['csv_artifacts'][str(relative/path.name)]=project_csv(path,target/path.name)
        for path in folder.glob('matches.csv.gz'):
            try: os.link(path,target/path.name)
            except OSError: shutil.copy2(path,target/path.name)
        summary_path=folder/'summary.json'
        if summary_path.exists():
            summary=clean_summary(json.loads(summary_path.read_text()))
            summary['models_generated']=['smoa']
            atomic_json(target/'summary.json',summary)
        metadata_path=folder/'metadata.json'
        if metadata_path.exists():
            original=json.loads(metadata_path.read_text())
            generated_config=clean_summary(original.get('config',{}))
            if 'output_dir' in generated_config: generated_config['output_dir']=str(target)
            metadata=dict(config=generated_config,models=['smoa'],status='complete',
                generation_metadata=str(metadata_path),generation_metadata_sha256=file_hash(metadata_path),
                generation_input_and_source_sha256=original.get('sha256',original.get('source_sha256',{})),
                processing_source_sha256=config['source_sha256'],
                derivation='Existing sMOA point, NB parameters and quantile strings retained exactly; no inference rerun')
            atomic_json(target/'metadata.json',metadata)
    atomic_json(destination/'reuse_provenance.json',provenance)
    print(f'Reused issued sMOA values without inference/resampling: {previous}',flush=True)
    return provenance
