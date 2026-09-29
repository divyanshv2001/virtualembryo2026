import numpy as np
from growth_prior_audit import fit_proxy_scores


def test_proxy_controls_and_scores_exclude_future_expression():
    rng=np.random.default_rng(711);x=rng.uniform(.1,1.,(128,1000)).astype(np.float32)
    symbols=[f'g{i}' for i in range(1000)];markers={'proliferation':['g1','g2'],'p53_proxy':['g3','g4']}
    rows=np.arange(96);scores,audit=fit_proxy_scores(x,rows,symbols,markers)
    changed=x.copy();changed[96:]=99
    repeated,other_audit=fit_proxy_scores(changed,rows,symbols,markers)
    assert audit==other_audit
    for name in markers:
        np.testing.assert_array_equal(scores[name],repeated[name])
        assert not set(audit[name]['control_gene_indices'])&{1,2,3,4}
        assert audit[name]['unique_overlap']==2
