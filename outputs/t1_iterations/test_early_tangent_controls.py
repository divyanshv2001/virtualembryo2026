import numpy as np
from early_tangent_controls import construct


def main():
    rng=np.random.default_rng(73);reference=rng.uniform(.5,2.,(8,7));reference[0,0]=0
    initial=reference+rng.normal(0,.03,reference.shape)*(reference>0)
    predictions,audit=construct(reference,initial)
    repeated,again=construct(reference,initial)
    assert 0<audit['shared_scale']<=.25 and audit['relative_norm_match_error']<1e-6
    for name in predictions:
        assert np.array_equal(predictions[name],repeated[name])
        assert np.array_equal(predictions[name]>0,reference>0)
    assert not np.array_equal(predictions['learned'],predictions['shuffle'])
    assert audit['learned_linear_audit']['linear_constraint_ratio']<1e-5
    try:construct(reference,reference)
    except ValueError:pass
    else:raise AssertionError('Zero correction accepted')
    # Shared positivity restriction, not clipping, protects small supported entries.
    tiny=reference.copy();tiny[1,1]=1e-5
    bounded,info=construct(tiny,tiny+(initial-reference))
    assert info['shared_scale']<=.25 and all(np.array_equal(v>0,tiny>0) for v in bounded.values())
    print('Early tangent controls passed: deterministic shuffle, matched norm, shared positivity, exact support, null difference, zero rejection.')


if __name__=='__main__':main()
