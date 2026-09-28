import unittest
import numpy as np
from threadpoolctl import threadpool_limits
from challenge_transfer import ChallengeTransfer


class TransferForecastTests(unittest.TestCase):
    def test_unit_scaling_remains_past_only(self):
        rng = np.random.default_rng(19)
        stages = np.repeat([8., 8.25, 8.5, 9.5], 60)
        x = np.log1p(rng.poisson(2, (240, 40))).astype(np.float32)
        changed = x.copy(); changed[stages > 8.5] = 100
        symbols = [f'G{i}' for i in range(40)]
        donors = x[stages == 8.5]
        with threadpool_limits(limits=2):
            a = ChallengeTransfer(x, stages, 8.5, donors, symbols, symbols, feature_scaling='unit')
            b = ChallengeTransfer(changed, stages, 8.5, donors, symbols, symbols, feature_scaling='unit')
            np.testing.assert_array_equal(a.model.scale, np.ones(len(a.model.features)))
            np.testing.assert_array_equal(a.predict(9.5, 'state', .5, .25)[0], b.predict(9.5, 'state', .5, .25)[0])

    def test_identity_alignment_projects_measured_anchor_features_directly(self):
        rng = np.random.default_rng(12)
        x = np.log1p(rng.poisson(2, (180, 40))).astype(np.float32)
        stages = np.repeat([8., 8.25, 8.5], 60)
        symbols = [f'G{i}' for i in range(40)]
        donors = x[-60:].copy(); donors[:, :10] *= 1.2
        with threadpool_limits(limits=2):
            model = ChallengeTransfer(x, stages, 8.5, donors, symbols, symbols, alignment='identity')
            fitted = model.model
            expected = fitted.pca.transform(np.clip((donors[:, model.official_features]-fitted.center)/fitted.scale, -10, 10))
            np.testing.assert_allclose(model.z, expected)
            self.assertEqual(model.alignment, 'identity')

    def test_forecast_ignores_future_atlas_and_preserves_protected_genes(self):
        rng = np.random.default_rng(8)
        counts = rng.poisson(.8, (240, 40)).astype(float)
        x = np.log1p(counts*10000/counts.sum(1)[:, None]).astype(np.float32)
        stages = np.repeat([8., 8.25, 8.5, 9.5], 60)
        changed = x.copy(); changed[stages > 8.5] = 99
        official = [f'G{i}' for i in range(40)]+['missing']
        atlas = [f'G{i}' for i in range(40)]; atlas[-1] = atlas[-2]
        donors = np.column_stack([x[stages == 8.5], np.ones(60, dtype=np.float32)])
        with threadpool_limits(limits=2):
            a = ChallengeTransfer(x, stages, 8.5, donors, official, atlas)
            b = ChallengeTransfer(changed, stages, 8.5, donors, official, atlas)
            for method in ['global', 'state']:
                result, indices, audit = a.predict(9.5, method, .5, .25)
                np.testing.assert_allclose(result, b.predict(9.5, method, .5, .25)[0])
                np.testing.assert_array_equal(result[:, 38:], donors[indices, 38:])
                np.testing.assert_array_equal(result == 0, donors[indices] == 0)
                np.testing.assert_allclose(np.expm1(result).sum(1), np.expm1(donors[indices]).sum(1), rtol=1e-6)
                self.assertFalse(any(i >= 38 for i in a.model.features))
                if method == 'state': self.assertLessEqual(audit['covariance_change_vs_reference'], .1)
            np.testing.assert_array_equal(a.predict(9.5)[0], donors)


if __name__ == '__main__': unittest.main()
