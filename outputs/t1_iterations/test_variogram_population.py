import unittest
import numpy as np
from scipy.optimize import check_grad
from threadpoolctl import threadpool_limits
from variogram_population import VariogramPopulation as KernelPopulation
from kernel_population import weight_objective
from detection_transfer import DetectionTransfer
from pathlib import Path
from tempfile import TemporaryDirectory


class VariogramTests(unittest.TestCase):
    def test_future_exclusion_and_whole_cell_resampling(self):
        rng = np.random.default_rng(5); stages = np.repeat([7.5,7.75,8.,8.25,8.5,9.5],60)
        x = np.log1p(rng.poisson(2,(360,40))).astype(np.float32)
        changed = x.copy(); changed[stages > 8.5] = 100
        symbols = [f'G{i}' for i in range(40)]; donors = x[stages == 8.5]
        with threadpool_limits(limits=2),TemporaryDirectory() as directory:
            m = DetectionTransfer(x,stages,8.5,donors,symbols,symbols,alignment='identity',feature_scaling='unit',covariance_limit=.4)
            path = Path(directory)/'model.npz'; m.save(path)
            with np.load(path) as source: encoder = {k:source[k] for k in source.files}
            base,indices,_ = m.predict_detection(9.5,.5,1.,.02)
            a = KernelPopulation(x,stages,8.5,encoder,donors,base,indices,dimensions=50)
            b = KernelPopulation(changed,stages,8.5,encoder,donors,base,indices,dimensions=50)
            np.testing.assert_array_equal(a.stage_means,b.stage_means)
            prediction,origin,audit = a.predict(9.5,.25,.01)
            for row in prediction: self.assertTrue(any(np.array_equal(row,cell) for cell in base))
            np.testing.assert_array_equal(prediction,b.predict(9.5,.25,.01)[0])
            self.assertLessEqual(audit['covariance_change_vs_reference'],.4)
            self.assertGreaterEqual(audit['effective_sample_size'],.6*len(base))


if __name__ == '__main__': unittest.main()
