"""Guard model removal, unchanged sMOA RNG and lossless reuse of issued values."""
import csv
from datetime import date
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/forecasting'))
from prediction import predict_origin
from uncertainty import calibration_plan,interval_quantiles,LONG_QUANTILES
from reuse_smoa import reuse_predictions


class SmoaOnlyTests(unittest.TestCase):
    def test_only_smoa_is_generated_with_the_original_rng_key(self):
        origin=date(2020,8,15)
        plan=calibration_plan([],str(origin),4)
        self.assertEqual(set(plan),{('forecast',h) for h in range(1,5)})
        result,_,_,probs=predict_origin([100,101,102],np.array([[1.,1.]]),np.array([[2.,3.,4.,5.]]),origin,plan,
                                      k_closest=1,smoothing='none',seed=13,draws=500)
        self.assertEqual(len(result),4)
        self.assertEqual({r['field'] for r in result},{'forecast'})
        for row in result:
            rng=np.random.default_rng(np.random.SeedSequence([13,origin.toordinal(),row['horizon'],0]))
            expected=interval_quantiles(row['point'],row['size'],rng,500,True,LONG_QUANTILES)
            np.testing.assert_array_equal(row['quantiles'],expected)

    def test_projection_preserves_raw_smoa_strings_and_original_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'previous';folder=source/'04_prediction/01';folder.mkdir(parents=True)
            source_hashes={'experiments/forecasting/'+name:'abc' for name in
                           ['model.py','smoothing.py','neighbors.py','embedding_cache.py','snapshots.py']}
            old=dict(parameters={'N':18387,'k':5,'m':4422,'h':4},library_sha256='lib',origins=['2020-08-15'],
                     truth_sha256={'A':'truth'},packages={'numpy':'v'},method={'seed':13},source_sha256=source_hashes)
            (source/'run_config.json').write_text(json.dumps(old))
            (source/'pipeline.json').write_text(json.dumps({'status':'complete','elapsed_seconds':20}))
            frozen=source/'source_snapshot/experiments/forecasting/uncertainty.py'
            frozen.parent.mkdir(parents=True)
            frozen.write_text((ROOT/'experiments/forecasting/uncertainty.py').read_text().replace("for field in ['forecast']:","for field in ['forecast','baseline']:"))
            raw='series,forecast_origin,horizon,forecast,baseline\nA,2020-08-15,1,123.00000000000000003,999\n'
            (folder/'predictions.csv').write_text(raw)
            (folder/'quantiles.csv').write_text('model,quantile,value\nsmoa,0.5,123.00000000000000003\npersistence,0.5,999\n')
            (folder/'metadata.json').write_text(json.dumps({'config':{'output_dir':str(folder)}}))
            (folder/'summary.json').write_text(json.dumps({'overall':{'mae':3,'baseline_mae':9}}))
            cfg=dict(old,method={'seed':13,'models':['smoa']})
            target=root/'current'
            provenance=reuse_predictions(source,target,cfg)
            with (target/'01/predictions.csv').open() as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(rows,[{'series':'A','forecast_origin':'2020-08-15','horizon':'1','forecast':'123.00000000000000003'}])
            with (target/'01/quantiles.csv').open() as f:
                self.assertEqual(list(csv.DictReader(f)),[{'model':'smoa','quantile':'0.5','value':'123.00000000000000003'}])
            self.assertEqual((folder/'predictions.csv').read_text(),raw)
            self.assertEqual(json.loads((target/'01/summary.json').read_text())['overall'],{'mae':3})
            self.assertEqual(provenance['models'],['smoa'])
            self.assertIn('generation_metadata_sha256',json.loads((target/'01/metadata.json').read_text()))
            bad=dict(cfg,library_sha256='different')
            with self.assertRaisesRegex(ValueError,'library_sha256'):
                reuse_predictions(source,root/'bad',bad)


if __name__=='__main__':unittest.main()
