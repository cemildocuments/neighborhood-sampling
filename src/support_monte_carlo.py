"""Independently verify the product-inclusion estimator against exact variances."""
from pathlib import Path
import numpy as np,pandas as pd,torch,hashlib,json,time
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(2)
(ROOT/'results/support').mkdir(parents=True,exist_ok=True)
device='mps' if torch.backends.mps.is_available() else 'cpu';rows=[];t0=time.perf_counter();R=10000
exact=pd.read_csv(ROOT/'results/support/exact_variances.csv')
for path in sorted((ROOT/'data/processed').glob('*_support.npz')):
 sample=path.stem.removesuffix('_support');d=np.load(path);ids=d['ids'];y=d['y'];p=d['p'];N=int(d['N']);truth=d['truth'];M=len(p)
 pp=np.r_[p,1.];pi=np.prod(pp[ids],1);w=torch.tensor(y/pi[:,None]/N,dtype=torch.float32,device=device);idx=torch.tensor(ids,dtype=torch.long,device=device)
 rng=np.random.default_rng(int.from_bytes(hashlib.sha256((sample+'support').encode()).digest()[:8],'little'));est=[];precision=0.
 for start in range(0,R,250):
  S=np.column_stack([rng.random((min(250,R-start),M))<p,np.ones(min(250,R-start),bool)]).astype('float32')
  st=torch.from_numpy(S).to(device);inc=st[:,idx[:,0]]
  for k in range(1,ids.shape[1]):inc=inc*st[:,idx[:,k]]
  a=(inc@w).cpu().numpy().astype(float);est.append(a)
  if start==0:precision=np.max(abs(a-(np.prod(S[:,ids],2).astype(float)@(y/pi[:,None]/N))))
 est=np.concatenate(est);var=est.var(0,ddof=1);bias=est.mean(0)-truth
 ref=exact[(exact['sample']==sample)&(exact.design=='targeted')]
 for e,(_,rr) in enumerate(ref.iterrows()):rows.append({'sample':sample,'endpoint':rr.endpoint,'replicates':R,'bias':bias[e],'bias_mcse':np.sqrt(var[e]/R),'variance':var[e],'exact_variance':rr.support_variance,'variance_ratio':var[e]/rr.support_variance,'gpu_float64_audit':precision})
 print(sample,flush=True)
pd.DataFrame(rows).to_csv(ROOT/'results/support/monte_carlo.csv',index=False)
(ROOT/'results/support/runtime.json').write_text(json.dumps({'seconds':time.perf_counter()-t0,'device':device,'replicates':R}))
