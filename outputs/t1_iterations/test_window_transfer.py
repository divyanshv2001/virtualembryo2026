import unittest
import numpy as np
from threadpoolctl import threadpool_limits
from window_transfer import WindowTransfer


class WindowTests(unittest.TestCase):
    def test_population_window_excludes_future_and_restores_incumbent(self):
        rng = np.random.default_rng(5)
        stages = np.repeat([7.5,7.75,8.,8.25,8.5,9.5],60)
        x = np.log1p(rng.poisson(2,(360,40))).astype(np.float32)
        changed = x.copy(); changed[stages > 8.5] = 100
        symbols = [f'G{i}' for i in range(40)]; donors = x[stages == 8.5]
        with threadpool_limits(limits=2):
            a = WindowTransfer(x,stages,8.5,donors,symbols,symbols)
            b = WindowTransfer(changed,stages,8.5,donors,symbols,symbols)
            np.testing.assert_array_equal(a.window_population_slope,b.window_population_slope)
            self.assertEqual(a.window_times[-1],8.5)
            original = a.model.population_slope.copy()
            a.predict_window(9.5,1.)
            np.testing.assert_array_equal(a.model.population_slope,original)


if __name__ == '__main__': unittest.main()
