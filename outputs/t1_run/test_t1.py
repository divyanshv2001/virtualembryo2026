"""Exporter boundary tests with synthetic tiny matrices, not embryo performance."""
import tempfile
import unittest
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from run_t1 import validate

class ExportChecks(unittest.TestCase):
    panel=['A','B']
    spec={'min_cells':2,'max_cells':3,'n_genes':2}
    def fixture(self,path,x=None,names=None,coords=False):
        if x is None:
            x=np.array([[0,1],[2,0]],dtype=np.float32)
        a=ad.AnnData(X=sparse.csr_matrix(x),obs=pd.DataFrame({'celltype':['ignored']*len(x)},index=[str(i) for i in range(len(x))]),var=pd.DataFrame(index=names or self.panel))
        if coords:
            a.obsm['spatial_3D']=np.zeros((len(x),3))
        a.write_h5ad(path)
    def test_sparse_valid_and_obs_optional(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.h5ad';self.fixture(p)
            self.assertTrue(validate(p,self.panel,self.spec)['passed'])
    def test_invalid_values_rejected(self):
        for value in (-1,float('nan'),float('inf')):
            with tempfile.TemporaryDirectory() as d:
                p=Path(d)/'x.h5ad';self.fixture(p,np.array([[value,1],[2,0]],dtype=np.float32))
                with self.assertRaises(ValueError):validate(p,self.panel,self.spec)
    def test_reordered_genes_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.h5ad';self.fixture(p,names=['B','A'])
            with self.assertRaises(ValueError):validate(p,self.panel,self.spec)
    def test_coordinate_metadata_rejected_for_t1(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.h5ad';self.fixture(p,coords=True)
            with self.assertRaises(ValueError):validate(p,self.panel,self.spec)
    def test_cell_limits_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.h5ad';self.fixture(p,np.ones((1,2),dtype=np.float32))
            with self.assertRaises(ValueError):validate(p,self.panel,self.spec)

if __name__=='__main__':unittest.main()
