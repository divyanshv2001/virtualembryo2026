"""Coarse past-only states with uncertainty shrinkage and covariance guards."""
import numpy as np
from sklearn.cluster import KMeans
from constrained_forecast import PopulationForecast, slope


def stable_slope(times, means, errors):
    """Suppress reversing trends and shrink slopes by mean-estimation uncertainty."""
    delta = np.diff(means, axis=0)
    consistent = (delta[0]*delta[1] > 0)
    signal = np.abs(means[-1]-means[0])
    uncertainty = np.sqrt(errors[0]**2+errors[-1]**2)
    return slope(times, means)*consistent*signal/(signal+2*uncertainty+1e-9)


def covariance_change(a, b):
    ca = np.cov(a, rowvar=False); cb = np.cov(b, rowvar=False)
    return float(np.linalg.norm(cb-ca)/max(np.linalg.norm(ca), 1e-12))


class RobustPopulation(PopulationForecast):
    def __init__(self, x, stages, cutoff, states=4, donor_cap=1000, allowed_features=None):
        super().__init__(x, stages, cutoff, allowed_features=allowed_features)
        rows = np.flatnonzero(stages <= cutoff)
        recent = np.unique(stages[rows])[-3:]
        values = np.asarray(x[np.ix_(rows, self.features)], dtype=float)
        z = self.pca.transform(np.clip((values-self.center)/self.scale, -10, 10))
        self.clusterer = KMeans(n_clusters=states, n_init=5, random_state=20260928).fit(z)
        labels = self.clusterer.labels_
        self.state_support = np.zeros(states, dtype=int)
        self.state_slope = np.zeros((states, x.shape[1]))
        proportions = np.stack([(np.bincount(labels[stages[rows] == t], minlength=states)+4)
            /(int((stages[rows] == t).sum())+4*states) for t in recent])
        prop_error = np.sqrt((1-proportions)/(proportions*np.array([(stages[rows] == t).sum() for t in recent])[:, None]))
        self.population_slope = stable_slope(recent, np.log(proportions), prop_error)
        for k in range(states):
            groups = [rows[(stages[rows] == t) & (labels == k)] for t in recent]
            support = min(map(len, groups)); self.state_support[k] = support
            if support < 8: continue
            for start in range(0, x.shape[1], 512):
                means = []; errors = []
                for group in groups:
                    abundance = np.expm1(np.asarray(x[group, start:start+512], dtype=float))
                    mean = abundance.mean(0)+.1
                    means.append(np.log(mean))
                    errors.append(np.sqrt(abundance.var(0)/len(group))/mean)
                self.state_slope[k, start:start+512] = stable_slope(recent, np.stack(means), np.stack(errors))*support/(support+64)
        last_labels = labels[stages[rows] == cutoff]
        self.donor_selection = np.sort(np.random.default_rng(20260928).choice(len(self.donor), min(donor_cap, len(self.donor)), replace=False))
        self.donor = self.donor[self.donor_selection]
        self.z = self.z[self.donor_selection]
        self.donor_labels = last_labels[self.donor_selection]
        self.last_audit = {}

    def predict(self, target, tilt=0., expression=0.):
        if target <= self.cutoff: raise ValueError('Forecast must follow training cutoff')
        horizon = target-self.cutoff
        desired = np.exp(np.clip(horizon*tilt*self.population_slope[self.donor_labels], -np.log(1.5), np.log(1.5)))
        desired /= desired.sum()
        original = self.donor[:, self.features].astype(float)
        accepted = None
        # Gate only against observed donor covariance; target data never enter this search.
        for mix in [1., .5, .25, .125, 0.]:
            weights = mix*desired+(1-mix)/len(desired)
            indices = np.minimum(np.searchsorted(np.cumsum(weights), (np.arange(len(weights))+.5)/len(weights)), len(weights)-1)
            candidate = self.donor[indices]
            if 1/np.sum(weights**2) >= .8*len(weights) and covariance_change(original, candidate[:, self.features]) <= .1:
                accepted = (candidate, indices, weights, mix); break
        donor, indices, weights, mix = accepted
        used_expression = 0.; prediction = donor.copy()
        if expression:
            for multiplier in [1., .5, .25, .125, 0.]:
                strength = expression*multiplier
                local = self.state_slope[self.donor_labels[indices]]
                factors = np.exp(np.clip(horizon*strength*local, -np.log(1.25), np.log(1.25)))
                abundance = np.expm1(donor.astype(float))*factors
                candidate = np.log1p(abundance*(10000/abundance.sum(1)[:, None])).astype(np.float32)
                if covariance_change(original, candidate[:, self.features]) <= .1:
                    prediction = candidate; used_expression = strength; break
        self.last_audit = {'population_mix':mix, 'used_expression_strength':used_expression,
            'covariance_change_vs_reference':covariance_change(original, prediction[:, self.features]),
            'supported_states':int((self.state_support >= 8).sum()), 'state_support':self.state_support.tolist()}
        return prediction, donor, indices, weights
