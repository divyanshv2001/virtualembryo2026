"""Fixed capture-balanced geometric rate decay with matched support controls."""
import numpy as np
from capture_ols_gate_decoder import shifts as baseline_shifts, controls as baseline_controls


def shifts(means, stages, ols, projected, omitted_ols):
    levels = np.unique(stages)
    if len(levels) != 3 or not np.allclose(np.diff(levels), .25, rtol=0, atol=1e-12):
        raise ValueError('Require three equally spaced quarter-day stages')
    eligible = np.ones(means.shape[1], bool)
    for omission in [None, *range(len(means))]:
        keep = np.ones(len(means), bool)
        if omission is not None:
            keep[omission] = False
        if any(np.sum(keep & (stages == t)) < 2 for t in levels):
            raise ValueError('Mandatory omission unsupported')
        avg = [means[keep & (stages == t)].mean(0) for t in levels]
        d1, d2 = avg[1] - avg[0], avg[2] - avg[1]
        eligible &= (d1 != 0) & (d2 != 0) & (np.sign(d1) == np.sign(d2)) & (np.abs(d2) < np.abs(d1))
    avg = [means[stages == t].mean(0) for t in levels]
    d1, recent = avg[1] - avg[0], avg[2] - avg[1]
    ratio = np.divide(recent, d1, out=np.zeros_like(recent), where=eligible)
    result = baseline_shifts(means, stages, ols, projected, omitted_ols)
    result.update(capture_recent=recent,
                  capture_OLS_decay_support=np.where(eligible, ols, 0),
                  capture_recent_decay_support=np.where(eligible, recent, 0),
                  capture_geometric_decay=np.where(eligible, recent * ratio, 0))
    return result


def controls():
    baseline_controls()
    stages = np.repeat([7.5, 7.75, 8.], 3)
    means = np.repeat([[0., 3., 0., 1., 0.], [2., 1., 1., 1., 2.],
                       [3., 0., 3., 1., 1.]], 3, axis=0)
    ols = (means[-1] - means[0]) / 2
    s = shifts(means, stages, ols, ols, np.tile(ols, (9, 1)))
    np.testing.assert_allclose(s['capture_geometric_decay'], [.5, -.5, 0., 0., 0.])
    np.testing.assert_allclose(s['capture_recent_decay_support'], [1., -1., 0., 0., 0.])
    np.testing.assert_allclose(s['capture_OLS_decay_support'], [1.5, -1.5, 0., 0., 0.])
    try:
        shifts(means[:-1], stages[:-1], ols, ols, np.tile(ols, (8, 1)))
    except ValueError as error:
        assert 'unsupported' in str(error)
    else:
        raise AssertionError('Two-capture stage must fail mandatory omission')


if __name__ == '__main__':
    controls()
    print('Positive/negative decay, acceleration/reversal/zero fallback, matched support and omission controls passed')
