import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from shared_experiment_core import screening_separation


class ScreeningOrderTests(unittest.TestCase):
    def test_one_order_uses_only_first_cache(self):
        d=np.array([[[1.,2.,np.inf]],[[1.,7.,1.]]])
        xi=np.array([0.,32.,4.])
        one=screening_separation(xi,d,[0],.5,1)
        np.testing.assert_array_equal(one,[0.,4.,0.])
        np.testing.assert_array_equal(screening_separation(xi,d,[0],.5,2),xi)
        np.testing.assert_array_equal(xi,[0.,32.,4.])
        with self.assertRaises(ValueError):screening_separation(xi,d,[0],.5,3)

    def test_end_to_end_features_fixed_pools_different(self):
        here=Path(__file__).parent
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);data=root/'data';cache=root/'cache';data.mkdir();cache.mkdir()
            np.save(data/'fixture.npy',np.ones((5,1)))
            first=np.ones((5,5));first[:,0]=[1,4,3,2,1]
            second=np.ones((5,5));second[:,0]=[1,1,5,1,6]
            np.save(cache/'fixture_distance_1.npy',first);np.save(cache/'fixture_distance_2.npy',second)
            records=[];features=[]
            for orders in (1,2):
                cfg=dict(data_root=str(data),cache_root=str(cache),datasets=['fixture'],orders=2,
                    feature_orders=2,screening_orders=orders,alpha=.5,budget=2,rival_count=1,
                    ratios=[.6],modes=['product'],methods=['dqn'],beta=.2,horizon=2,episodes=2,
                    train_simulations=2,eval_simulations=3,repeats=1,learning_rate=.001,gamma=.9,
                    batch_size=1,replay_capacity=20,target_update_steps=1,seed=42)
                config=root/f'config{orders}.json';config.write_text(json.dumps(cfg))
                output=root/f'run{orders}'
                result=subprocess.run([sys.executable,str(here/'shared_experiment_core.py'),'--config',str(config),'--output',str(output)],capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
                records.append(json.loads((output/'results.jsonl').read_text().splitlines()[0]))
                with np.load(output/'fixture/features.npz') as f:features.append({k:f[k].copy() for k in f.files})
            np.testing.assert_array_equal(features[0]['xi'],features[1]['xi'])
            np.testing.assert_array_equal(features[0]['phi'],features[1]['phi'])
            self.assertNotEqual(records[0]['pool'],records[1]['pool'])
            self.assertEqual(records[0]['screening_orders'],1)
            self.assertEqual(records[0]['feature_orders'],2)


if __name__=='__main__':unittest.main()
