import json
from pathlib import Path
import tempfile
import unittest
from fair_comparison import tasks,safe_child
from fair_report import describe

class FairTests(unittest.TestCase):
    def test_exact_expected_cells_and_legacy_identity(self):
        jobs=tasks();self.assertEqual(len(jobs),15);self.assertEqual(len({j['task_id'] for j in jobs}),15)
        first=next(j for j in jobs if j['config']['methods']==['s2v'] and j['config']['repeat_ids']==[0])
        self.assertEqual(first['task_id'],'432feec32039395aa9f2')
        for method in ('dqn','greedy','s2v'):
            self.assertEqual({j['config']['repeat_ids'][0] for j in jobs if j['config']['methods']==[method]},set(range(5)))
    def test_artifact_paths_cannot_escape(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):safe_child(Path(d),'../secret')
            self.assertEqual(safe_child(Path(d),'child.json'),Path(d).resolve()/'child.json')
    def test_missing_or_single_repeat_not_confident_interval(self):
        self.assertIsNone(describe([])['mean']);self.assertIsNone(describe([123])['ci95'])
        self.assertEqual(describe([1,2,3])['n'],3)

if __name__=='__main__':unittest.main()
