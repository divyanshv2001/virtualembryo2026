import unittest
import numpy as np
from threadpoolctl import threadpool_limits
from hurdle_transfer import HurdleTransfer


class HurdleTests(unittest.TestCase):
    def test_frequency_change_does_not_create_positive_abundance_trend(self):
        rng = np.random.default_rng(33)
        stages = np.repeat([8.,8.25,8.5,9.5], 160)
        x = np.log1p(rng.poisson(2,(640,40))).astype(np.float32)
        for j,t in enumerate([8.,8.25,8.5]):
            group = np.flatnonzero(stages == t)
            x[group,0] = 0; x[group[:[32,80,128][j]],0] = np.log1p(3)
        symbols = [f'G{i}' for i in range(40)]
        donors = x[stages == 8.5].copy()
        changed = x.copy(); changed[stages > 8.5] = 100
        with threadpool_limits(limits=2):
            a = HurdleTransfer(x, stages, 8.5, donors, symbols, symbols, states=1)
            b = HurdleTransfer(changed, stages, 8.5, donors, symbols, symbols, states=1)
            self.assertGreater(a.original_state_slope[0,0], 0)
            self.assertAlmostEqual(a.positive_state_slope[0,0], 0)
            np.testing.assert_array_equal(a.positive_state_slope, b.positive_state_slope)
            original = a.model.state_slope.copy()
            a.predict_hurdle(9.5)
            np.testing.assert_array_equal(a.model.state_slope, original)


if __name__ == '__main__': unittest.main()
