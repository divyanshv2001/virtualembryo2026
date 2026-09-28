import struct
import unittest
import zlib
import numpy as np
from threadpoolctl import threadpool_limits
from prepare_extended_atlas import decode_record
from train_extended_atlas import Dynamics, normalize_log
from pathlib import Path
from tempfile import TemporaryDirectory


class ExtendedAtlasTests(unittest.TestCase):
    def test_decode_rejects_wrong_cell_count_and_invalid_expression(self):
        description = b'TestGene'
        values = np.array([0,2,4],dtype='<f4')
        encoded = zlib.compress(struct.pack('<H',len(description))+description+values.tobytes())
        name,result = decode_record(encoded,3)
        self.assertEqual(name,'TestGene')
        np.testing.assert_array_equal(result,values)
        with self.assertRaises(ValueError): decode_record(encoded,4)
        bad = zlib.compress(struct.pack('<H',len(description))+description+np.array([0,-1,4],dtype='<f4').tobytes())
        with self.assertRaises(ValueError): decode_record(bad,3)

    def test_training_representation_and_decoder_ignore_future_stage(self):
        rng = np.random.default_rng(4)
        x = np.log1p(rng.poisson(3,size=(160,80))).astype(np.float32)
        stages = np.repeat([6.5,6.75,7.,8.5],40)
        altered = x.copy(); altered[stages>7.] = 200
        with TemporaryDirectory() as temp, threadpool_limits(limits=2):
            log = Path(temp)/'events.jsonl'
            a = Dynamics(x,stages,7.,log)
            b = Dynamics(altered,stages,7.,log)
            np.testing.assert_array_equal(a.features,b.features)
            np.testing.assert_allclose(a.center,b.center)
            np.testing.assert_allclose(a.decoder,b.decoder)
            np.testing.assert_allclose(a.velocities,b.velocities)

    def test_clipping_and_normalization_preserve_implied_library(self):
        x = np.array([[-.2,1.,2.],[0.,.5,3.]],dtype=np.float32)
        result = normalize_log(x)
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue((result>=0).all())
        np.testing.assert_allclose(np.expm1(result).sum(1),10000,rtol=1e-6)


if __name__=='__main__': unittest.main()
