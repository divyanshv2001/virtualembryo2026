"""Calibrate gene trends on historical forecasts ending before challenge target."""
from collections import Counter
import numpy as np
from detection_transfer import DetectionTransfer


def calibrated_gain(predicted, observed, errors):
    """Nonnegative ridge gain; errors quantify cell sampling, not embryo replication."""
    predicted = np.asarray(predicted,dtype=float)
    observed = np.asarray(observed,dtype=float); errors = np.asarray(errors,dtype=float)
    if predicted.shape != observed.shape or predicted.shape != errors.shape or predicted.ndim != 2:
        raise ValueError('Require fold-by-gene arrays with matching shapes')
    if not all(np.isfinite(a).all() for a in [predicted,observed,errors]) or (errors < 0).any():
        raise ValueError('Invalid calibration observations')
    numerator = (predicted*observed).sum(0)
    denominator = (predicted**2+4*errors**2).sum(0)
    gain = np.divide(numerator,denominator,out=np.zeros_like(numerator),where=denominator > 0)
    # Only consistent agreement across both past forecasts receives amplification.
    agrees = (predicted*observed > 0).all(0)
    return np.minimum(np.clip(gain,0,2),np.where(agrees,2.,1.))


def fit_reliability(x,stages,official_symbols,atlas_symbols,max_stage=8.5,states=16,donor_cap=1500):
    folds = [(max_stage-.5,max_stage-.25),(max_stage-.25,max_stage)]
    counts = Counter(atlas_symbols)
    lookup = {s:i for i,s in enumerate(atlas_symbols) if s and counts[s] == 1}
    mapped = np.array([i for i,s in enumerate(official_symbols) if s in lookup])
    atlas = np.array([lookup[official_symbols[i]] for i in mapped])
    predicted = []; observed = []; errors = []; audits = []
    for fold,(cutoff,target) in enumerate(folds):
        before = np.flatnonzero(stages == cutoff); after = np.flatnonzero(stages == target)
        if len(before) < 8 or len(after) < 8: raise ValueError('Insufficient historical support')
        selected = np.sort(np.random.default_rng(2026092803+fold).choice(before,min(donor_cap,len(before)),replace=False))
        donors = np.zeros((len(selected),len(official_symbols)),dtype=np.float32)
        donors[:,mapped] = np.asarray(x[np.ix_(selected,atlas)])
        model = DetectionTransfer(x,stages,cutoff,donors,official_symbols,atlas_symbols,
            states=states,alignment='identity',feature_scaling='unit',covariance_limit=.4)
        forecast,_,audit = model.predict_detection(target,.5,1.,.02)
        mean = np.zeros(len(official_symbols)); variance = np.zeros(len(official_symbols))
        for start in range(0,len(mapped),512):
            values = np.asarray(x[np.ix_(after,atlas[start:start+512])],dtype=float)
            mean[mapped[start:start+512]] = values.mean(0)
            variance[mapped[start:start+512]] = values.var(0)
        anchor_mean = donors.mean(0,dtype=float)
        predicted.append((forecast.mean(0,dtype=float)-anchor_mean)/(target-cutoff))
        observed.append((mean-anchor_mean)/(target-cutoff))
        errors.append(np.sqrt(variance/len(after)+donors.var(0,dtype=float)/len(donors))/(target-cutoff))
        audits.append({'cutoff':cutoff,'target':target,'training_cells':int((stages <= cutoff).sum()),
            'anchor_cells':len(donors),'historical_target_cells':len(after),
            'source_anchor_rows':selected.tolist(),'forecast_audit':audit})
        del model,forecast,donors
    gain = calibrated_gain(np.stack(predicted),np.stack(observed),np.stack(errors))
    return gain, {'folds':audits,'maximum_calibration_stage':max_stage,
        'zero_gain_genes':int((gain == 0).sum()),'amplified_genes':int((gain > 1).sum()),
        'scope':'Past-fold cell-mean ridge calibration; no future challenge labels or embryo-level uncertainty estimate.'}


def predict_reliable(model,target,official_gain,blend=1.,detection=True):
    if not 0 <= blend <= 1 or len(official_gain) != len(model.official_symbols): raise ValueError('Invalid calibration blend')
    weight = (1-blend)+blend*np.asarray(official_gain)
    atlas_weight = np.ones(len(model.atlas_symbols))
    atlas_weight[model.atlas_mapped] = weight[model.mapped]
    original = model.model.state_slope; original_detection = model.detection_slope
    model.model.state_slope = original*atlas_weight[None,:]
    if detection: model.detection_slope = original_detection*weight[model.mapped][None,:]
    try:
        prediction,indices,audit = model.predict_detection(target,.5,1.,.02)
        return prediction,indices,{**audit,'gene_reliability_blend':blend,'detection_calibrated':detection}
    finally:
        model.model.state_slope = original; model.detection_slope = original_detection
