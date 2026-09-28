import unittest
import numpy as np
from threadpoolctl import threadpool_limits
from constrained_forecast import PopulationForecast
from state_population import StatePopulationForecast


class PopulationTests(unittest.TestCase):
    def test_future_perturbation_cannot_change_forecast(self):
        rng = np.random.default_rng(1)
        x = rng.poisson(2, (160, 30)).astype(float)
        x = np.log1p(x*10000/x.sum(1)[:, None]).astype(np.float32)
        stages = np.repeat([7., 7.25, 7.5, 8.5], 40)
        changed = x.copy(); changed[stages > 7.5] = 100
        with threadpool_limits(limits=2):
            a = PopulationForecast(x, stages, 7.5)
            b = PopulationForecast(changed, stages, 7.5)
            np.testing.assert_allclose(a.predict(8.5, .5, .1)[0], b.predict(8.5, .5, .1)[0])

    def test_support_and_persistence_contract(self):
        rng = np.random.default_rng(4)
        x = rng.poisson(.5, (120, 30)).astype(float)
        x = np.log1p(x*10000/x.sum(1)[:, None]).astype(np.float32)
        stages = np.repeat([7., 7.25, 7.5], 40)
        with threadpool_limits(limits=2):
            model = PopulationForecast(x, stages, 7.5)
            copy, _, _, _ = model.predict(8.5)
            np.testing.assert_array_equal(copy, x[stages == 7.5])
            prediction, donors, _, weights = model.predict(8.5, .5, .25)
            np.testing.assert_array_equal(prediction == 0, donors == 0)
            np.testing.assert_allclose(np.expm1(prediction).sum(1), 10000, rtol=1e-6)
            self.assertLessEqual(weights.max()/weights.min(), 4+1e-9)

    def test_state_forecast_ignores_future_and_preserves_support(self):
        rng = np.random.default_rng(9)
        x = rng.poisson(.8, (240, 35)).astype(float)
        x = np.log1p(x*10000/x.sum(1)[:, None]).astype(np.float32)
        stages = np.repeat([7., 7.25, 7.5, 8.5], 60)
        changed = x.copy(); changed[stages > 7.5] = 99
        with threadpool_limits(limits=2):
            a = StatePopulationForecast(x, stages, 7.5)
            b = StatePopulationForecast(changed, stages, 7.5)
            prediction, donor, _, _ = a.predict(8.5, 1., 1.)
            np.testing.assert_allclose(prediction, b.predict(8.5, 1., 1.)[0])
            np.testing.assert_array_equal(prediction == 0, donor == 0)
            np.testing.assert_allclose(np.expm1(prediction).sum(1), 10000, rtol=1e-6)


if __name__ == '__main__': unittest.main()
