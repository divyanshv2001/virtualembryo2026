"""Boundary and solver checks for the fresh hindcast; no source data or scores."""
import json
import tempfile
from pathlib import Path
import numpy as np
import torch
from scnode_past_fold_training import JointModel, PastOnlyMatrix, past_features
from scnode_past_fold import panel_values
from temporary_forecast_cache import TemporaryForecastCache
from run_t1 import digest

class Constant(torch.nn.Module):
    def forward(self,z):return torch.ones_like(z)*2

class Linear(torch.nn.Module):
    def forward(self,z):return z


def main():
    model=JointModel(6);model.drift=Constant()
    z=torch.ones((2,32),requires_grad=True)
    times=torch.tensor([0.,.25,.5])
    result=model.trajectory(z,times)
    torch.testing.assert_close(result[:,0],z)
    torch.testing.assert_close(result[:,1],z+.5)
    torch.testing.assert_close(result[:,2],z+1.)
    result.sum().backward();assert torch.isfinite(z.grad).all()
    model.drift=Linear()
    torch.testing.assert_close(model.trajectory(z,torch.tensor([0.,.5]))[:,-1],z*(1+.0625)**8)
    torch.testing.assert_close(model.trajectory(z,torch.tensor([0.,.25,.5]))[:,-1],model.trajectory(z,torch.tensor([0.,.5]))[:,-1])
    try:model.trajectory(z,torch.tensor([0.,0.]));raise AssertionError('Duplicate time allowed')
    except ValueError:pass
    stages=np.array([7.5,7.75,8.,8.5]);raw=np.arange(24,dtype=np.float32).reshape(4,6)
    past=np.flatnonzero(stages<=8.)
    matrix=PastOnlyMatrix(raw,stages,8.)
    selected=past_features(matrix,past,np.arange(6),3)
    mutated=raw.copy();mutated[-1]=1e8
    np.testing.assert_array_equal(selected,past_features(PastOnlyMatrix(mutated,stages,8.),past,np.arange(6),3))
    try:matrix[np.ix_([3],[0])];raise AssertionError('Future read allowed')
    except ValueError:pass
    constructed=panel_values(matrix,past,np.array([0,2]),np.array([1,4]),4)
    np.testing.assert_array_equal(constructed[:,[1,3]],np.zeros((3,2)))
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent/'private') as folder:
        cache=TemporaryForecastCache(folder)
        assert cache.put('a',constructed)==cache.put('b',constructed.copy())
        cache.close()
    evidence={'status':'passed','checks':['physical-day constant drift','eight .0625 Euler substeps',
              'identical interval subdivision at training/forecast','finite solver gradient',
              'duplicate times rejected','future expression read rejected','future mutation leaves features unchanged',
              'constructed unmapped panel zero','reference .npy hash replay'],
              'source_sha256':{f:digest(Path(__file__).resolve().parent/f) for f in ['scnode_past_fold.py','scnode_past_fold_training.py']},
              'scope':'Synthetic implementation checks; no benchmark score or biological validation'}
    (Path(__file__).resolve().parent/'SCNODE_PAST_FOLD_PREFLIGHT.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(evidence))

if __name__=='__main__':main()
