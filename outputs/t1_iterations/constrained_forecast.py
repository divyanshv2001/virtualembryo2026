"""Past-only population tilting and bounded, support-preserving expression trends."""
import numpy as np
from sklearn.decomposition import PCA


def slope(times, values):
    centered = times-times.mean()
    return centered @ values / np.sum(centered**2)


class PopulationForecast:
    def __init__(self, x, stages, cutoff, allowed_features=None, feature_scaling='standard'):
        if feature_scaling not in ['standard', 'unit']: raise ValueError('Unknown feature scaling')
        self.feature_scaling = feature_scaling
        rows = np.flatnonzero(stages <= cutoff)
        if len(np.unique(stages[rows])) < 3:
            raise ValueError('Three past stages required')
        variance = np.zeros(x.shape[1])
        for start in range(0, x.shape[1], 512):
            variance[start:start+512] = np.asarray(x[rows, start:start+512], dtype=float).var(0)
        eligible = np.arange(x.shape[1]) if allowed_features is None else np.asarray(allowed_features, dtype=int)
        if len(eligible) < 2 or len(np.unique(eligible)) != len(eligible) or (eligible < 0).any() or (eligible >= x.shape[1]).any():
            raise ValueError('Invalid eligible features')
        self.features = np.sort(eligible[np.argsort(-variance[eligible], kind='stable')[:384]])
        values = np.asarray(x[np.ix_(rows, self.features)], dtype=float)
        self.center = values.mean(0)
        self.scale = np.maximum(values.std(0), .1) if feature_scaling == 'standard' else np.ones(values.shape[1])
        self.pca = PCA(n_components=min(16, len(self.features), len(rows)-1), random_state=20260928)
        z = self.pca.fit_transform(np.clip((values-self.center)/self.scale, -10, 10))
        recent = np.unique(stages[rows])[-3:]
        means = np.stack([z[stages[rows] == t].mean(0) for t in recent])
        self.latent_slope = slope(recent, means)
        self.z = z[stages[rows] == cutoff]
        self.donor = np.asarray(x[stages == cutoff], dtype=np.float32)
        # Mean log abundance, fitted exclusively on the latest three past stages.
        abundance_means = np.stack([np.expm1(np.asarray(x[stages == t], dtype=float)).mean(0) for t in recent])
        self.abundance_slope = slope(recent, np.log(abundance_means + .1))
        self.cutoff = cutoff

    def predict(self, target, tilt=0., expression=0.):
        if target <= self.cutoff:
            raise ValueError('Forecast must follow training cutoff')
        horizon = target-self.cutoff
        centered = self.z-self.z.mean(0)
        covariance = centered.T @ centered / len(centered)
        coefficient = np.linalg.solve(covariance + np.eye(centered.shape[1]), horizon*tilt*self.latent_slope)
        weights = np.exp(np.clip(centered @ coefficient, -np.log(2), np.log(2)))
        weights /= weights.sum()
        # Fixed systematic resampling, without seed search or target information.
        indices = np.searchsorted(np.cumsum(weights), (np.arange(len(weights))+.5)/len(weights))
        indices = np.minimum(indices, len(weights)-1)
        donor = self.donor[indices]
        if expression:
            factors = np.exp(np.clip(horizon*expression*self.abundance_slope, -np.log(1.25), np.log(1.25)))
            abundance = np.expm1(donor.astype(float))*factors
            prediction = np.log1p(abundance*(10000/abundance.sum(1)[:, None])).astype(np.float32)
        else:
            prediction = donor.copy()
        return prediction, donor, indices, weights


def distribution_audit(prediction, donor, features):
    abundance = np.expm1(prediction.astype(float)).sum(1)
    a = donor[:, features].astype(float); b = prediction[:, features].astype(float)
    ca = np.cov(a, rowvar=False); cb = np.cov(b, rowvar=False)
    return {'zero_fraction':float((prediction == 0).mean()),
        'new_positive_fraction':float(((donor == 0) & (prediction > 0)).mean()),
        'library_quantiles':np.quantile(abundance, [0, .5, 1]).tolist(),
        'covariance_relative_change':float(np.linalg.norm(cb-ca)/max(np.linalg.norm(ca), 1e-12)),
        'mean_absolute_log_change':float(np.abs(prediction-donor).mean())}
