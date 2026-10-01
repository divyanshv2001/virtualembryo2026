"""Past-only uniform lineage damping controls matched to geometric shift L2."""
import numpy as np
from capture_geometric_rate import shifts as geometric_shifts, controls as geometric_controls


def shifts(means, stages, ols, projected, omitted_ols):
    result = geometric_shifts(means, stages, ols, projected, omitted_ols)
    desired = np.linalg.norm(result['capture_geometric_decay'])
    for base in ['capture_OLS_decay_support', 'capture_recent_decay_support']:
        source = result[base]
        norm = np.linalg.norm(source)
        result[base + '_amplitude_matched'] = source * (desired / norm if norm else 0.)
        np.testing.assert_allclose(np.linalg.norm(result[base + '_amplitude_matched']), desired, rtol=1e-12, atol=1e-12)
    return result


def controls():
    geometric_controls()
    stages = np.repeat([7.5, 7.75, 8.], 3)
    means = np.repeat([[0., 4., 0.], [4., 2., 0.], [5., 1., 0.]], 3, axis=0)
    ols = (means[-1] - means[0]) / 2
    result = shifts(means, stages, ols, ols, np.tile(ols, (9, 1)))
    for base in ['capture_OLS_decay_support', 'capture_recent_decay_support']:
        matched = result[base + '_amplitude_matched']
        np.testing.assert_allclose(matched @ matched, result['capture_geometric_decay'] @ result['capture_geometric_decay'], rtol=1e-12)
        np.testing.assert_array_equal(np.sign(matched), np.sign(result[base]))
    zeros = np.zeros_like(means)
    result = shifts(zeros, stages, np.zeros(3), np.zeros(3), np.zeros((9, 3)))
    assert all(np.isfinite(x).all() and np.count_nonzero(x) == 0 for x in result.values())


if __name__ == '__main__':
    controls()
    print('Matched requested energy, retained signed support and zero fallback controls passed')
