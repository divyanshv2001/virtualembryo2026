"""Small SPD covariance controls; these are not benchmark metrics."""
import numpy as np
from sklearn.covariance import LedoitWolf, OAS


def power(matrix, exponent, floor=1e-6):
    matrix=np.asarray(matrix,float)
    if matrix.ndim!=2 or matrix.shape[0]!=matrix.shape[1] or not np.isfinite(matrix).all():
        raise ValueError('Finite square covariance required')
    values,vectors=np.linalg.eigh((matrix+matrix.T)/2)
    if values.min() < -1e-8:raise ValueError('Covariance is not PSD')
    return (vectors*np.maximum(values,floor)**exponent)@vectors.T


def project_psd(matrix):
    values,vectors=np.linalg.eigh((matrix+matrix.T)/2)
    return (vectors*np.maximum(values,1e-6))@vectors.T


def bures_map(source, target):
    half=power(source,.5);inverse=power(source,-.5)
    result=inverse@power(half@target@half,.5,floor=0.)@inverse
    return (result+result.T)/2


def distance_squared(source,target):
    half=power(source,.5)
    value=float(np.trace(source+target-2*power(half@target@half,.5,floor=0.)))
    if value < -1e-7:raise ValueError('Negative Bures squared distance')
    return max(0.,value)


def balanced_covariance(values,captures,min_cells=5):
    records=[];matrices=[]
    for capture in sorted(set(captures)):
        block=values[captures==capture]
        if len(block)<min_cells:
            records.append({'capture':str(capture),'cells':len(block),'included':False});continue
        estimator=LedoitWolf(assume_centered=True).fit(block-block.mean(0))
        matrices.append(estimator.covariance_)
        records.append({'capture':str(capture),'cells':len(block),'included':True,'shrinkage':float(estimator.shrinkage_)})
    if not matrices:raise ValueError('No supported CM capture')
    return project_psd(np.mean(matrices,axis=0)),records


def pooled_covariance(values,labels):
    residual=values.copy()
    for label in sorted(set(labels)):
        mask=labels==label;residual[mask]-=residual[mask].mean(0)
    return project_psd(OAS(assume_centered=True).fit(residual).covariance_)


def forecast(previous,current,previous_time,current_time,target_time,seed=20260928):
    ratio=(target_time-previous_time)/(current_time-previous_time)
    if not (previous_time<current_time<target_time):raise ValueError('Forward times required')
    transport=bures_map(previous,current)
    values,vectors=np.linalg.eigh(transport)
    clipped=(vectors*np.clip(values,.75,4/3))@vectors.T
    extension=np.eye(len(current))+ratio*(clipped-np.eye(len(current)))
    if np.linalg.eigvalsh(extension).min()<=0:raise ValueError('Extrapolated map not positive')
    learned=project_psd(extension@previous@extension.T)
    delta=learned-current
    permutation=np.random.default_rng(seed).permutation(len(current))
    null_delta=delta[np.ix_(permutation,permutation)]
    null=project_psd(current+null_delta)
    a=float(np.linalg.norm(delta));b=float(np.linalg.norm(null-current))
    norm_ratio=a/b if b>0 else None
    return learned,null,{'transport_eigenvalues':values.tolist(),'clipped_eigenvalues':np.clip(values,.75,4/3).tolist(),
        'unclipped_current_reconstruction_error':float(np.max(np.abs(transport@previous@transport.T-current))),
        'single_permutation':permutation.tolist(),'null_norm_ratio':norm_ratio,
        'null_valid':norm_ratio is not None and .9<=norm_ratio<=1.1,
        'learned_displacement_norm':a,'null_displacement_norm':b,'extrapolation_ratio':ratio}
