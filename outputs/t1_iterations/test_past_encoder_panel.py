import numpy as np
from past_encoder_panel import fit_past_encoder


def test_future_rows_and_ambiguous_genes_cannot_change_encoder():
    rng=np.random.default_rng(491);x=rng.uniform(0,2,(80,8)).astype(np.float32)
    stages=np.repeat([7.5,7.75,8.,9.],20);symbols=['a','b','c','d','e','dup','dup','only_atlas'];panel=['a','b','c','d','e','dup','protected']
    first=fit_past_encoder(x,stages,8.,symbols,panel,budget=4,dimensions=2)
    changed=x.copy();changed[stages>8.]=rng.uniform(10,100,(20,8))
    second=fit_past_encoder(changed,stages,8.,symbols,panel,budget=4,dimensions=2)
    for key in first:np.testing.assert_array_equal(first[key],second[key])
    assert set(first['features']).issubset(set(range(5)))
    assert np.all(stages[first['pca_rows']]<=8.)
    np.testing.assert_allclose(first['coordinates'].std(0),np.ones(2),rtol=1e-5)
