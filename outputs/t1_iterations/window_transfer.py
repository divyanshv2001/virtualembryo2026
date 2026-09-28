"""Longer past-stage population trends, retaining bounded detection changes."""
import numpy as np
from detection_transfer import DetectionTransfer
from constrained_forecast import slope


class WindowTransfer(DetectionTransfer):
    def __init__(self,x,stages,cutoff,*args,**kwargs):
        super().__init__(x,stages,cutoff,*args,**kwargs)
        rows = np.flatnonzero(stages <= cutoff)
        times = np.unique(stages[rows])[-5:]
        if len(times) < 5: raise ValueError('Five past stages required')
        labels = self.model.clusterer.labels_
        k = len(self.model.state_support)
        probabilities = np.stack([(np.bincount(labels[stages[rows] == t],minlength=k)+4)
            /(int((stages[rows] == t).sum())+4*k) for t in times])
        self.original_population_slope = self.model.population_slope.copy()
        self.window_population_slope = slope(times,np.log(probabilities))
        self.window_times = times; self.window_probabilities = probabilities

    def predict_window(self,target,tilt=.5,expression=1.,cap=.02,window=True):
        # Reuse detection's declared tilting strength of .5 by rescaling slope.
        self.model.population_slope = self.window_population_slope*(tilt/.5) if window else self.original_population_slope
        try: return self.predict_detection(target,.5,expression,cap)
        finally: self.model.population_slope = self.original_population_slope

    def save(self,path):
        super().save(path)
        with np.load(path) as source: fields = {k:source[k] for k in source.files}
        np.savez_compressed(path,**fields,window_population_slope=self.window_population_slope,
            window_times=self.window_times,window_probabilities=self.window_probabilities)
