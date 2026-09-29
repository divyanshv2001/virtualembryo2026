import hashlib
import numpy as np
from temporary_forecast_cache import TemporaryForecastCache


def test_cache_stream_hash_matches_npy_and_closes_temporary_files(tmp_path):
    forecast = np.arange(24, dtype=np.float32).reshape(4, 6)
    reference = tmp_path / 'reference.npy'
    np.save(reference, forecast, allow_pickle=False)
    cache = TemporaryForecastCache(tmp_path)
    assert cache.put('candidate', forecast) == hashlib.sha256(reference.read_bytes()).hexdigest()
    with cache.read('candidate') as recovered:
        np.testing.assert_array_equal(recovered, forecast)
    assert not cache.handles
    assert sorted(tmp_path.iterdir()) == [reference]


def test_cache_can_read_same_frozen_forecast_across_panels(tmp_path):
    forecast = np.arange(12, dtype=np.float32).reshape(3, 4)
    cache = TemporaryForecastCache(tmp_path)
    cache.put('candidate', forecast)
    for _ in range(3):
        with cache.read('candidate', consume=False) as recovered:
            np.testing.assert_array_equal(recovered, forecast)
    assert list(cache.handles) == ['candidate']
    cache.close()
    assert not cache.handles
    assert list(tmp_path.iterdir()) == []
