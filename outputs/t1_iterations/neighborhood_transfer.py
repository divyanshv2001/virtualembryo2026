"""Local past-stage maturation with bounded sparse gene activation."""
from collections import Counter
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from challenge_transfer import ChallengeTransfer
from robust_population import covariance_change


class NeighborhoodTransfer(ChallengeTransfer):
    def __init__(self, x, stages, cutoff, donors, official_symbols, atlas_symbols, **kwargs):
        super().__init__(x, stages, cutoff, donors, official_symbols, atlas_symbols, **kwargs)
        counts = Counter(atlas_symbols)
        lookup = {s:i for i,s in enumerate(atlas_symbols) if s and counts[s] == 1}
        self.mapped = np.array([i for i,s in enumerate(official_symbols) if s in lookup])
        atlas = np.array([lookup[official_symbols[g]] for g in self.mapped])
        times = np.unique(stages[stages <= cutoff])[-3:]
        if len(times) != 3 or not np.allclose(np.diff(times), np.diff(times)[0]):
            raise ValueError('Three equally spaced past stages required')
        means = []
        for t in times:
            rows = np.flatnonzero(stages == t)
            if len(rows) < 8: raise ValueError('Insufficient past-stage support')
            values = np.asarray(x[np.ix_(rows, self.model.features)], dtype=float)
            z = self.model.pca.transform(np.clip((values-self.model.center)/self.model.scale,-10,10))
            distance, indices = NearestNeighbors(n_neighbors=min(32,len(rows))).fit(z).kneighbors(self.z)
            weight = 1/np.maximum(distance, .1); weight /= weight.sum(1)[:,None]
            operator = csr_matrix((weight.ravel(), (np.repeat(np.arange(len(donors)),indices.shape[1]),indices.ravel())),
                shape=(len(donors),len(rows)))
            mean = np.empty((len(donors),len(self.mapped)), dtype=np.float32)
            for start in range(0,len(atlas),512):
                mean[:,start:start+512] = operator@np.asarray(x[np.ix_(rows,atlas[start:start+512])])
            means.append(mean)
        delta0 = means[1]-means[0]; delta1 = means[2]-means[1]
        # Reverse-sign local trajectories are suppressed; future rows never read.
        self.local_slope = (means[2]-means[0])/(times[2]-times[0])*(delta0*delta1 > 0)
        self.local_slope *= self.trusted[:,None]

    def predict_neighborhood(self, target, strength=.25, activations=32):
        if target <= self.model.cutoff or not 0 <= strength <= 1 or not 0 <= activations <= 256:
            raise ValueError('Invalid declared neighborhood forecast')
        delta = np.clip((target-self.model.cutoff)*strength*self.local_slope,-.5,.5)
        original = self.donors[:,self.mapped]
        update = np.maximum(original+delta,0)
        allowed = (original == 0) & (delta > .05)
        if activations == 0: allowed[:] = False
        elif activations < allowed.shape[1]:
            score = np.where(allowed,delta,-np.inf)
            best = np.argpartition(score,-activations,axis=1)[:,-activations:]
            mask = np.zeros_like(allowed); np.put_along_axis(mask,best,True,axis=1)
            allowed &= mask
        update[(original == 0) & ~allowed] = 0
        mass = np.expm1(original.astype(float)).sum(1)
        abundance = np.expm1(update.astype(float)); total = abundance.sum(1)
        empty = (total == 0) & (mass > 0)
        abundance[empty] = np.expm1(original[empty].astype(float)); total = abundance.sum(1)
        ratio = np.divide(mass,total,out=np.ones_like(mass),where=total > 0)
        prediction = self.donors.copy()
        prediction[:,self.mapped] = np.log1p(abundance*ratio[:,None]).astype(np.float32)
        return prediction, np.arange(len(prediction)), {
            'strength':strength, 'maximum_activations_per_cell':activations,
            'actual_additions':int(((self.donors == 0) & (prediction > 0)).sum()),
            'actual_deletions':int(((self.donors > 0) & (prediction == 0)).sum()),
            'covariance_change_vs_reference':covariance_change(self.donors[:,self.official_features],prediction[:,self.official_features]),
            'guard_applied':False, 'delta_cap':.5, 'activation_minimum_delta':.05,
            'trusted_donor_fraction':float(self.trusted.mean()), 'alignment':self.alignment}

    def save(self,path):
        super().save(path)
        # This checkpoint can be large; biological arrays remain private.
        np.save(path.with_name(path.stem+'_local_slope.npy'),self.local_slope)
