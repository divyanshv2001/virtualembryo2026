"""Past-only state-conditioned abundance and population trends."""
import numpy as np
from sklearn.cluster import KMeans
from constrained_forecast import PopulationForecast, slope


class StatePopulationForecast(PopulationForecast):
    def __init__(self, x, stages, cutoff):
        super().__init__(x, stages, cutoff)
        rows = np.flatnonzero(stages <= cutoff)
        values = np.asarray(x[np.ix_(rows, self.features)], dtype=float)
        z = self.pca.transform(np.clip((values-self.center)/self.scale, -10, 10))
        self.clusterer = KMeans(n_clusters=24, n_init=5, random_state=20260928).fit(z)
        labels = self.clusterer.labels_
        self.donor_labels = labels[stages[rows] == cutoff]
        recent = np.unique(stages[rows])[-3:]
        proportions = np.stack([(np.bincount(labels[stages[rows] == t], minlength=24)+1)
            /(int((stages[rows] == t).sum())+24) for t in recent])
        self.population_slope = slope(recent, np.log(proportions))
        self.state_slope = np.zeros((24, x.shape[1]))
        self.state_support = np.zeros(24, dtype=int)
        for k in range(24):
            groups = [rows[(stages[rows] == t) & (labels == k)] for t in recent]
            support = min(map(len, groups))
            self.state_support[k] = support
            if support < 8:
                continue  # Unreliable or new states retain donor expression.
            means = np.stack([np.expm1(np.asarray(x[g], dtype=float)).mean(0) for g in groups])
            self.state_slope[k] = slope(recent, np.log(means+.1))*(support/(support+32))

    def predict(self, target, tilt=0., expression=0.):
        if target <= self.cutoff: raise ValueError('Forecast must follow training cutoff')
        horizon = target-self.cutoff
        weights = np.exp(np.clip(horizon*tilt*self.population_slope[self.donor_labels], -np.log(2), np.log(2)))
        weights /= weights.sum()
        indices = np.searchsorted(np.cumsum(weights), (np.arange(len(weights))+.5)/len(weights))
        indices = np.minimum(indices, len(weights)-1)
        donor = self.donor[indices]
        if expression:
            local = self.state_slope[self.donor_labels[indices]]
            factors = np.exp(np.clip(horizon*expression*local, -np.log(1.5), np.log(1.5)))
            abundance = np.expm1(donor.astype(float))*factors
            prediction = np.log1p(abundance*(10000/abundance.sum(1)[:, None])).astype(np.float32)
        else: prediction = donor.copy()
        return prediction, donor, indices, weights
