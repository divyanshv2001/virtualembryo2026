import numpy as np
from types import SimpleNamespace
from importance_source import IndexedRows,select_source

def test_indexed_view_matches_array_without_duplicates():
    x=np.arange(100).reshape(20,5);rows=np.array([1,7,3,9]);view=IndexedRows(x,rows)
    np.testing.assert_array_equal(view[np.ix_([0,3],[1,4])],x[rows][np.ix_([0,3],[1,4])])
    np.testing.assert_array_equal(view[np.array([True,False,True,False])],x[rows][[0,2]])

def test_sampling_excludes_future_and_retains_unique_cells():
    rng=np.random.default_rng(3);x=rng.normal(size=(300,2));stages=np.repeat([7.5,8.,8.5],100)
    model=SimpleNamespace(features=np.array([0,1]),center=np.zeros(2),scale=np.ones(2),
        pca=SimpleNamespace(transform=lambda values:values),clusterer=SimpleNamespace(cluster_centers_=np.zeros((2,2)),predict=lambda values:(values[:,0]>0).astype(int)))
    teacher=SimpleNamespace(model=model,labels=np.array([0]*80+[1]*20),trusted=np.ones(100,dtype=bool))
    a,audit=select_source(x,stages,teacher,8.,.5);changed=x.copy();changed[stages>8.]=100
    b,_=select_source(changed,stages,teacher,8.,.5)
    np.testing.assert_array_equal(a,b)
    assert len(a)==len(set(a)) and np.all(stages[a]<=8.) and audit['duplicates']==0
