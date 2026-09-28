import unittest
import numpy as np
import pandas as pd
from offline_backtest import load_core, Panel, predict, split_embryos


class OfflineTests(unittest.TestCase):
    def test_cached_de_and_mmd_match_pinned_core(self):
        core, _ = load_core()
        rng = np.random.default_rng(3)
        reference = rng.uniform(.3,1.,(80,50)).astype(np.float32)
        truth = rng.uniform(.3,1.,(80,50)).astype(np.float32)
        truth[:,:10] += .6
        truth[:,40:] = np.maximum(truth[:,40:]-.4,0)
        prediction = truth.copy(); prediction[:,:10] -= .1
        panel = Panel(core,truth,reference,0)
        raw = panel.metrics(prediction)
        self.assertAlmostEqual(raw['de_score'],core.de_score(prediction,truth,reference)['score'],places=6)
        self.assertAlmostEqual(raw['de_direction'],core.de_direction(prediction,truth,reference),places=6)
        self.assertAlmostEqual(raw['mmd_u'],core.mmd_unbiased(prediction,truth),places=5)
        floor = panel.metrics(reference)
        self.assertEqual(floor['de_score'],0)
        self.assertEqual(floor['de_direction'],0)

    def test_embryo_splits_never_share_an_embryo(self):
        metadata = pd.DataFrame({'embryo':['a','a','b','c','c','d']})
        a,b,ea,eb = split_embryos(metadata,np.arange(6),9)
        self.assertFalse(set(a)&set(b))
        self.assertFalse(set(ea)&set(eb))
        self.assertEqual(set(a)|set(b),set(range(6)))

    def test_zero_horizon_drift_returns_last_snapshot(self):
        a = np.array([[0.,1.],[1.,0.]],dtype=np.float32)
        b = np.array([[.1,2.],[2.,.1]],dtype=np.float32)
        y = predict(a,b,{'method':'mean_shift','strength':1.},horizon_ratio=0.)
        np.testing.assert_array_equal(y,b)

    def test_worse_than_floor_ceiling_cannot_yield_a_score(self):
        core, _ = load_core()
        panel = Panel.__new__(Panel); panel.core = core
        floor = {'de_score':0.,'de_direction':0.,'mmd_u':.1,'variogram':.01}
        ceiling = {'de_score':.8,'de_direction':.8,'mmd_u':.001,'variogram':.02}
        result = panel.aggregate(floor,floor,ceiling)
        self.assertFalse(result['calibration_valid'])
        self.assertIsNone(result['local_score'])
        self.assertEqual(result['invalid_calibration_metrics'],['variogram'])


if __name__=='__main__': unittest.main()
