"""Short-lived local-disk .npy cache for full-panel forecasts."""
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryFile
import numpy as np


class TemporaryForecastCache:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.handles = {}

    def put(self, name, prediction):
        if name in self.handles:
            raise ValueError('Forecast already frozen')
        handle = TemporaryFile(mode='w+b', dir=self.directory, prefix='t1_forecast_cache_')
        try:
            np.save(handle, prediction, allow_pickle=False)
            handle.flush()
            handle.seek(0)
            digest = sha256()
            while block := handle.read(8 * 1024 * 1024):
                digest.update(block)
            handle.seek(0)
            self.handles[name] = handle
            return digest.hexdigest()
        except BaseException:
            handle.close()
            raise

    @contextmanager
    def read(self, name, consume=True):
        handle = self.handles.pop(name) if consume else self.handles[name]
        try:
            handle.seek(0)
            yield np.load(handle, allow_pickle=False)
        finally:
            if consume:
                handle.close()

    def close(self):
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()

    def __del__(self):
        self.close()
