import unittest
import numpy as np
from threadpoolctl import threadpool_limits
from gene_reliability import calibrated_gain,fit_reliability,predict_reliable
from detection_transfer import DetectionTransfer


class ReliabilityTests(unittest.TestCase):
    def test_gain_does_not_amplify_reversing_or_no_signal_genes(self):
        predicted = np.array([[1.,1.,0.],[1.,1.,0.]])
        observed = np.array([[2.,2.,1.],[2.,-1.,1.]])
        gain = calibrated_gain(predicted,observed,np.zeros_like(predicted))
        np.testing.assert_array_equal(gain,[2.,.5,0.])

    def test_calibration_excludes_future_and_forecast_restores_model(self):
        rng = np.random.default_rng(8); stages = np.repeat([7.5,7.75,8.,8.25,8.5,9.5],60)
        x = np.log1p(rng.poisson(2,(360,40))).astype(np.float32)
        changed = x.copy(); changed[stages > 8.5] = 100
        symbols = [f'G{i}' for i in range(40)]; official = symbols+['protected']
        with threadpool_limits(limits=2):
            gain,audit = fit_reliability(x,stages,official,symbols,states=4,donor_cap=60)
            other,_ = fit_reliability(changed,stages,official,symbols,states=4,donor_cap=60)
            np.testing.assert_array_equal(gain,other)
            self.assertEqual(audit['maximum_calibration_stage'],8.5)
            donors = np.column_stack([x[stages == 8.5],np.ones(60,dtype=np.float32)])
            model = DetectionTransfer(x,stages,8.5,donors,official,symbols,feature_scaling='unit')
            slopes = model.model.state_slope.copy(); detect = model.detection_slope.copy()
            pred,indices,_ = predict_reliable(model,9.5,gain,.5)
            np.testing.assert_array_equal(slopes,model.model.state_slope)
            np.testing.assert_array_equal(detect,model.detection_slope)
            np.testing.assert_array_equal(pred[:,-1],donors[indices,-1])
            np.testing.assert_allclose(np.expm1(pred).sum(1),np.expm1(donors[indices]).sum(1),rtol=1e-6)


if __name__ == '__main__': unittest.main()
