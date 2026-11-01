"""Scientific invariants: nesting, omitted bridge, exact prefixes and interactions."""
import csv
from itertools import product
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent/'forecasting'))
from design import configurations,REFERENCE,protocol
from library import pack_bank,load_prefix,embed,reuse_banks
from engine import aggregate,run_group
from model import forecast,load_library
from smoothing import thin_plate_gcv
from neighbors import ExactNeighbors
from analysis import factorial_effects,conditional_interactions,paired_deltas,segment_validity_audit
from runtime import atomic_json,file_hash
from extend_n import extend_bank,extension_key


class SensitivityTests(unittest.TestCase):
    def test_design_ofat_and_eight_corners(self):
        df=configurations()
        self.assertEqual(len(df),23)
        self.assertEqual(df.groupby(['N','k']).ngroups,14)
        self.assertFalse(df.duplicated(['N','k','m']).any())
        for row in df.loc[df.sections.str.contains('E2.A')].itertuples():
            self.assertLessEqual(sum(a!=b for a,b in zip((row.N,row.k,row.m),REFERENCE)),1)
        corners=df.loc[df.sections.str.contains('factorial')]
        self.assertEqual(set(map(tuple,corners[['N','k','m']].to_numpy())),set(product([1000,5000],[3,7],[500,2000])))
        self.assertEqual(protocol(),json.loads(json.dumps(protocol())))

    def fixture(self,root):
        path=root/'curves.csv';families=['sir_rollercoaster','sir_rollercoaster_wiggle','seasonal']
        with path.open('w') as handle:
            writer=csv.writer(handle);writer.writerow(['curve_id','seed','curve_type','time_cadence','scale','seasonal','step','value'])
            for curve in range(1,13):
                if curve==9:continue # rejected first seasonal attempt: never replace it.
                for step in range(1,22):writer.writerow([curve,curve,families[(curve-1)//4],'weekly','counts',False,step,float(step*step+curve)])
        atomic_json(root/'metadata.json',dict(status='complete',curves_sha256=file_hash(path),num_curves_per_family=4,accepted_total=11,
            config=dict(seed_offset=0,source_sha256={'generator.py':file_hash(HERE.parent/'synthetic/generator.py')})))
        return path

    def test_nested_prefixes_select_attempts_not_accepted_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=self.fixture(root);packed=root/'packed'
            pack_bank(source,packed,offset=0,max_n=4)
            small,small_ids=load_prefix(packed,2);large,large_ids=load_prefix(packed,4)
            self.assertEqual(small_ids.tolist(),[1,2,5,6,10])
            self.assertEqual(len(small),5)
            self.assertTrue(set(small_ids)<set(large_ids))
            reference=dict(zip(large_ids,large))
            for curve,ident in zip(small,small_ids):np.testing.assert_array_equal(curve.ts,reference[ident].ts)

    def test_reused_banks_link_without_copy_and_reject_modified_data(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=self.fixture(root);old=root/'old';new=root/'new'
            pack_bank(source,old/'packed/seed_0',offset=0,max_n=4)
            result=reuse_banks(old,new,levels=(4,),seeds=(0,))
            self.assertTrue((new/'packed/seed_0').is_symlink())
            self.assertEqual(result['banks'][0]['accepted'],11)
            (old/'packed/seed_0/attempts.npz').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Modified reused bank'):
                reuse_banks(old,new,levels=(4,),seeds=(0,))

    def test_analysis_rejects_legacy_protocol_even_when_complete(self):
        from analysis import analyze
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);atomic_json(root/'protocol.json',dict(design=dict(version=3)))
            with self.assertRaisesRegex(ValueError,'outdated E2 protocol'):
                analyze(root)

    def test_extension_preserves_prefix_ids_values_and_rejections(self):
        from generator import SyntheticCurve
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=self.fixture(root);parent=root/'parent'
            pack_bank(source,parent,offset=0,max_n=4)
            original,ids=load_prefix(parent,4)
            def generate(tasks,workers):
                for f,j,key in tasks:
                    yield None if (f,j)==(2,4) else SyntheticCurve(np.arange(20.)+100*f+j,'sir_rollercoaster' if f==0 else 'sir_rollercoaster_wiggle' if f==1 else 'seasonal','weekly','counts',f==2)
            with patch('extend_n.generate_ordered',side_effect=generate):
                meta=extend_bank(parent,root/'extended',0,target_n=6,base_n=4,workers=1)
            prefix,prefix_ids=load_prefix(root/'extended',4)
            np.testing.assert_array_equal(prefix_ids,ids)
            for before,after in zip(original,prefix):np.testing.assert_array_equal(before.ts,after.ts)
            self.assertEqual(meta['added_attempts'],6)
            self.assertEqual(meta['added_accepted'],5)
            self.assertEqual(meta['attempts'],18)
            self.assertTrue(meta['original_prefix_verified'])
            with patch('extend_n.generate_ordered',side_effect=AssertionError('Must not regenerate')):
                self.assertEqual(extend_bank(parent,root/'extended',0,target_n=6,base_n=4,workers=1),meta)
            with patch('extend_n.generate_ordered',side_effect=generate):
                extend_bank(parent,root/'larger',0,target_n=7,base_n=4,workers=1)
            previous,previous_ids=load_prefix(root/'extended',6)
            same_prefix,same_ids=load_prefix(root/'larger',6)
            np.testing.assert_array_equal(previous_ids,same_ids)
            for before,after in zip(previous,same_prefix):np.testing.assert_array_equal(before.ts,after.ts)
            with patch('extend_n.generate_ordered',side_effect=generate):
                chained=extend_bank(root/'extended',root/'chained',0,target_n=7,base_n=6,workers=1)
            kept,kept_ids=load_prefix(root/'chained',6)
            np.testing.assert_array_equal(kept_ids,previous_ids)
            for before,after in zip(previous,kept):np.testing.assert_array_equal(before.ts,after.ts)
            self.assertEqual(chained['added_attempts'],3)
            self.assertEqual(extension_key(0,2,4),[0,2000202,2,4])
            self.assertNotEqual(extension_key(0,2,4),extension_key(1000000,2,4))

    def test_embedding_matches_full_omit_and_aggregation_core(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=self.fixture(root);packed=root/'packed'
            pack_bank(source,packed,offset=0,max_n=4)
            curves,ids=load_prefix(packed,4)
            x,y,codes=embed(curves,ids,5,root/'scratch')
            ex,ey,_,_=load_library(source,5,5)
            np.testing.assert_array_equal(x,ex);np.testing.assert_array_equal(y,ey[:,1:])
            history=np.array([1,2,3,7,10,20,25,12,30,25.])
            expected,neighbors,_=forecast(history,x,y,closest=7,smoothing='gam')
            fitted=np.maximum(1,thin_plate_gcv(np.maximum(history,1)))
            actual_ids,_=ExactNeighbors(x).query(np.diff(fitted[-6:]),7)
            np.testing.assert_array_equal(actual_ids[0],neighbors)
            np.testing.assert_array_equal(aggregate(history[-1],y[actual_ids[0]]),expected)
            self.assertEqual(len(codes),len(x))

    def test_exact_max_m_prefix_preserves_ties(self):
        x=np.array([[0,0],[1,1],[1,1],[-1,-1],[2,2],[3,3]],float)
        index=ExactNeighbors(x)
        full,dist=index.query([0,0],5)
        for m in [1,2,3,4]:
            subset,sd=index.query([0,0],m)
            np.testing.assert_array_equal(subset,full[:,:m]);np.testing.assert_array_equal(sd,dist[:,:m])

    def test_nested_minimum_distance_cannot_increase(self):
        query=np.array([4.,8.]);small=np.array([[0,0],[1,2]],float);large=np.vstack([small,query])
        _,a=ExactNeighbors(small).query(query,1);_,b=ExactNeighbors(large).query(query,1)
        self.assertLessEqual(b[0,0],a[0,0])

    def test_factorial_effects_have_correct_scaling(self):
        rows=[]
        for n,k,m in product([1000,5000],[3,7],[500,2000]):
            a=-1 if n==1000 else 1;b=-1 if k==3 else 1;c=-1 if m==500 else 1
            rows.append(dict(seed=0,N=n,k=k,m=m,MAE=100+3*a+5*b+7*c+11*a*c+13*a*b*c))
        effects=factorial_effects(pd.DataFrame(rows)).set_index('term')
        self.assertAlmostEqual(effects.loc['N','effect'],6)
        self.assertAlmostEqual(effects.loc['N × m','effect'],22)
        self.assertAlmostEqual(effects.loc['N × m','difference_in_differences'],44)
        self.assertAlmostEqual(effects.loc['N × k × m','effect'],26)
        conditional=conditional_interactions(pd.DataFrame(rows))
        nm=conditional.loc[conditional.interaction.eq('N × m')].sort_values('fixed_value')
        np.testing.assert_allclose(nm.difference_in_differences,[-8,96])
        with self.assertRaises(ValueError):factorial_effects(pd.DataFrame(rows[:-1]))

    def test_paired_deltas_reject_label_or_target_changes(self):
        rows=[]
        for n,error in [(18387,10),(500,5)]:
            for state in ['A','B']:
                rows.append(dict(seed=0,N=n,k=5,m=4422,series=state,forecast_origin='2020-04-04',target_end_date='2020-04-11',horizon=1,actual=20,error=error))
        data=pd.DataFrame(rows);result=paired_deltas(data,draws=20)
        self.assertEqual(result.loc[result.N.eq(500),'delta_MAE'].iloc[0],-5)
        data.loc[0,'actual']=999
        with self.assertRaisesRegex(ValueError,'labels'):paired_deltas(data,draws=20)

    def test_m_exceeding_valid_segments_fails_explicitly(self):
        from argparse import Namespace
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=self.fixture(root);packed=root/'packed'
            pack_bank(source,packed,offset=0,max_n=4)
            with self.assertRaisesRegex(ValueError,'exceeds M'):
                run_group(Namespace(out=root/'job',packed=packed,N=2,k=5,m=[10000]))

    def test_validity_audit_detects_constant_windows_kept_by_roundoff(self):
        from bank import save_packed
        from generator import SyntheticCurve
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);runtime=[]
            for seed in [0,1,2]:
                packed=root/f'packed/seed_{seed}';packed.mkdir(parents=True)
                save_packed([SyntheticCurve(np.full(20,1e-8),'seasonal','weekly','proportion',True)],packed/'curves.npz')
                np.savez(packed/'attempts.npz',index=[0],family=[2],curve_id=[1])
                runtime.append(dict(seed=seed,N=500,k=5,M=10))
            result=segment_validity_audit(root,pd.DataFrame(runtime))
            self.assertTrue(result.constant_history_rows.eq(10).all())
            self.assertTrue(result.nonconstant_rows.eq(0).all())
            updated=pd.DataFrame(runtime).assign(M=0,window_filter='nonconstant_v2')
            corrected=segment_validity_audit(root,updated)
            self.assertTrue(corrected.constant_history_rows.eq(0).all())
            self.assertTrue(corrected.M_core.eq(0).all())


if __name__=='__main__':unittest.main()
