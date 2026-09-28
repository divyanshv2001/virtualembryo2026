"""Fit atlas trends and transfer onto past challenge donors without target access."""
from collections import Counter
import numpy as np
from robust_population import RobustPopulation, covariance_change
from transfer_genes import apply_unique_factors


class ChallengeTransfer:
    def __init__(self, x, stages, cutoff, donors, official_symbols, atlas_symbols, states=4, alignment='moments', covariance_limit=.1, feature_scaling='standard', expression_factor_cap=1.25):
        if not .1 <= covariance_limit <= .4: raise ValueError('Covariance limit must be within declared experimental bounds')
        self.covariance_limit = float(covariance_limit)
        if not 1 <= expression_factor_cap <= 4: raise ValueError('Expression factor cap outside experimental bounds')
        self.expression_factor_cap = float(expression_factor_cap)
        if donors.dtype != np.float32 or donors.shape[1] != len(official_symbols):
            raise ValueError('Require full-panel float32 donors')
        counts = Counter(atlas_symbols)
        lookup = {s:i for i, s in enumerate(official_symbols)}
        if len(lookup) != len(official_symbols): raise ValueError('Official panel must be unique')
        allowed = [i for i, s in enumerate(atlas_symbols) if s and counts[s] == 1 and s in lookup]
        self.model = RobustPopulation(x, stages, cutoff, states=states, allowed_features=allowed, feature_scaling=feature_scaling)
        self.donors = donors; self.official_symbols = official_symbols; self.atlas_symbols = atlas_symbols
        model = self.model
        self.official_features = np.array([lookup[atlas_symbols[g]] for g in model.features])
        source = np.asarray(x[np.ix_(np.flatnonzero(stages == cutoff), model.features)], dtype=float)
        anchor = donors[:, self.official_features].astype(float)
        if alignment not in ['moments', 'identity']: raise ValueError('Unknown alignment')
        self.alignment = alignment
        self.anchor_mean = anchor.mean(0); self.source_mean = source.mean(0)
        self.alignment_ratio = np.clip(source.std(0)/np.maximum(anchor.std(0), .1), .5, 2.)
        aligned = (anchor-self.anchor_mean)*self.alignment_ratio+self.source_mean if alignment == 'moments' else anchor
        self.z = model.pca.transform(np.clip((aligned-model.center)/model.scale, -10, 10))
        self.labels = model.clusterer.predict(self.z)
        distance = np.linalg.norm(self.z-model.clusterer.cluster_centers_[self.labels], axis=1)
        source_z = model.pca.transform(np.clip((source-model.center)/model.scale, -10, 10))
        source_labels = model.clusterer.predict(source_z)
        source_distance = np.linalg.norm(source_z-model.clusterer.cluster_centers_[source_labels], axis=1)
        self.distance_cap = float(np.quantile(source_distance, .99))
        self.trusted = distance <= self.distance_cap

    def predict(self, target, method='copy', tilt=0., expression=0.):
        model = self.model
        horizon = target-model.cutoff
        if horizon <= 0: raise ValueError('Target must follow fitted cutoff')
        if method == 'copy':
            return self.donors.copy(), np.arange(len(self.donors)), {'scope':'Measured past-cell persistence', 'guard_applied':False}
        if method not in ['global', 'state']: raise ValueError('Unknown transfer method')
        if method == 'global':
            centered = self.z-self.z.mean(0)
            coefficient = np.linalg.solve(centered.T@centered/len(centered)+np.eye(centered.shape[1]), horizon*tilt*model.latent_slope)
            logits = centered@coefficient
        else:
            logits = horizon*tilt*model.population_slope[self.labels]
        logits = np.where(self.trusted, logits, 0.)
        desired = np.exp(np.clip(logits, -np.log(1.5), np.log(1.5))); desired /= desired.sum()
        original = self.donors[:, self.official_features].astype(float)
        for mix in ([1., .5, .25, .125, 0.] if method == 'state' else [1.]):
            weights = mix*desired+(1-mix)/len(desired)
            indices = np.minimum(np.searchsorted(np.cumsum(weights), (np.arange(len(weights))+.5)/len(weights)), len(weights)-1)
            donor = self.donors[indices]
            cov = covariance_change(original, donor[:, self.official_features])
            if method != 'state' or (1/np.sum(weights**2) >= .8*len(weights) and cov <= self.covariance_limit): break
        prediction = donor.copy(); used_strength = 0.; mapping = None
        for multiplier in ([1., .5, .25, .125, 0.] if method == 'state' else [1.]):
            strength = expression*multiplier
            local = np.broadcast_to(model.abundance_slope, (len(donor), len(model.abundance_slope))) if method == 'global' else model.state_slope[self.labels[indices]]
            local = local*self.trusted[indices, None]
            factors = np.exp(np.clip(horizon*strength*local, -np.log(self.expression_factor_cap), np.log(self.expression_factor_cap)))
            candidate, mapping = apply_unique_factors(donor, self.official_symbols, self.atlas_symbols, factors)
            if method != 'state' or covariance_change(original, candidate[:, self.official_features]) <= self.covariance_limit:
                prediction = candidate; used_strength = strength; break
        audit = {'mapping':mapping, 'guard_applied':method == 'state', 'population_mix':mix,
            'expression_strength_used':used_strength, 'trusted_donor_fraction':float(self.trusted.mean()),
            'effective_sample_size':float(1/np.sum(weights**2)), 'unique_donors':int(len(np.unique(indices))),
            'covariance_change_vs_reference':covariance_change(original, prediction[:, self.official_features]),
            'state_support':model.state_support.tolist(), 'alignment':self.alignment,
            'covariance_limit':self.covariance_limit, 'feature_scaling':model.feature_scaling,
            'expression_factor_cap':self.expression_factor_cap}
        return prediction, indices, audit

    def save(self, path):
        m = self.model
        np.savez_compressed(path, features=m.features, official_features=self.official_features,
            center=m.center, scale=m.scale, pca_components=m.pca.components_, pca_mean=m.pca.mean_,
            latent_slope=m.latent_slope, abundance_slope=m.abundance_slope,
            cluster_centers=m.clusterer.cluster_centers_, state_slope=m.state_slope,
            population_slope=m.population_slope, state_support=m.state_support,
            source_mean=self.source_mean, anchor_mean=self.anchor_mean, alignment_ratio=self.alignment_ratio,
            anchor_latent=self.z, anchor_labels=self.labels, trusted=self.trusted, distance_cap=np.array(self.distance_cap),
            cutoff=np.array(m.cutoff), alignment=np.array(self.alignment), covariance_limit=np.array(self.covariance_limit),
            feature_scaling=np.array(m.feature_scaling), expression_factor_cap=np.array(self.expression_factor_cap))
