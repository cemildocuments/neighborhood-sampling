"""Spatial stress tests with an optional GPU matrix engine and a float64 audit."""
from pathlib import Path
import numpy as np,pandas as pd,json,time,argparse
from scipy.spatial import cKDTree
from neighborhood import nearest_other
from scipy.stats import t as student_t
from evaluate import probabilities,conservative_interval,summary
ROOT=Path(__file__).resolve().parents[1]

def main(replicates=50000,backend='auto'):
 import torch
 use_gpu=backend=='mps' or (backend=='auto' and torch.backends.mps.is_available());device='mps' if use_gpu else 'cpu'
 torch.set_num_threads(2);rows=[];audit=[];t0=time.perf_counter();out=ROOT/'results/simulation';out.mkdir(parents=True,exist_ok=True)
 for pattern in ['independent','clustered']:
  for prevalence in [.005,.05,.2]:
   seed=10000+int(prevalence*10000)+(pattern=='clustered')*17;rng=np.random.default_rng(seed);N=24000
   xy=rng.uniform(0,500,size=(N,2));A=rng.random(N)<.25
   if pattern=='independent':prob=np.full(N,prevalence)
   else:
    hotspot=np.exp(-np.sum((xy-[160,280])**2,axis=1)/(2*65**2));prob=prevalence*(.08+.92*hotspot/hotspot.mean());prob=np.minimum(prob,.95)
   B=rng.random(N)<prob
   spacing=np.median(cKDTree(xy).query(xy,k=2)[0][:,1]);xy/=spacing;L=12.;r=3.
   ij,tile=np.unique(np.floor(xy/L).astype(int),axis=0,return_inverse=True);M=len(ij);counts=np.bincount(tile,minlength=M).astype(float)
   near=nearest_other(xy,np.arange(N),np.flatnonzero(B))<=r
   xyz=np.column_stack([xy,tile*16.]);near_crop=nearest_other(xyz,np.arange(N),np.flatnonzero(B))<=r
   y=np.bincount(tile,weights=A&near,minlength=M)[:,None];yc=np.bincount(tile,weights=A&near_crop,minlength=M)[:,None];truth=y.sum(0)/N
   # An inexpensive spatial pilot defines where acquisition is concentrated, without high-plex outcomes.
   center=(ij+.5)*L;priority=.2+np.exp(-np.sum((center-np.array([160,280])/spacing)**2,axis=1)/(2*(80/spacing)**2));score=counts*priority
   for halo in [False,True]:
    p=probabilities(score,.4/((1+2*r/L)**2 if halo else 1));obs=y if halo else yc
    w=np.column_stack([obs[:,0]/p/N,(1-p)*obs[:,0]**2/p**2/N**2,((1-p)*obs[:,0]**2/p**2)**2/N**4])
    wt=torch.from_numpy(w.astype('float32')).to(device);ests=[];vs=[];cs=[]
    simrng=np.random.default_rng(seed+101)
    for start in range(0,replicates,1000):
     S=(simrng.random((min(1000,replicates-start),M))<p).astype('float32')
     prod=(torch.from_numpy(S).to(device)@wt).cpu().numpy().astype(float)
     if start==0:audit.append({'pattern':pattern,'prevalence':prevalence,'halo':halo,'max_abs_est_error':float(np.max(np.abs(prod[:,0]-(S.astype(float)@w)[:,0])))})
     ests.append(prod[:,0,None]);vs.append(np.maximum(prod[:,1,None],0));cs.append(np.maximum(prod[:,2,None],0))
    est=np.concatenate(ests);var=np.concatenate(vs);four=np.concatenate(cs);eff=var**2/np.maximum(four,1e-300);crit=student_t.ppf(.975,np.maximum(eff-1,1));se=np.sqrt(var)
    lo,hi=conservative_interval(est,var,p,counts,np.zeros_like(y),N)
    for label,low,high,q in [('normal',None,None,1.95996398454),('effective',None,None,crit),('bounded',lo,hi,1.)]:
     z=summary(est,se,truth,q,low,high);rec={'pattern':pattern,'prevalence':prevalence,'method':('halo_' if halo else 'crop_')+label,'truth':float(truth[0]),'replicates':replicates,'n_cells':N,'n_tiles':M,'expected_fields':p.sum()};rec.update({k:float(v[0]) for k,v in z.items()});rows.append(rec)
   np.savez_compressed(out/f'{pattern}_{prevalence}_tissue.npz',xy=xy.astype('float32'),anchor=A,neighbor=B)
   print(pattern,prevalence,'complete',flush=True)
 pd.DataFrame(rows).to_csv(out/'simulation_results.csv',index=False);(out/'gpu_audit.json').write_text(json.dumps({'device':device,'runtime_s':time.perf_counter()-t0,'audit':audit},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--replicates',type=int,default=50000);p.add_argument('--backend',choices=['auto','cpu','mps'],default='auto');a=p.parse_args();main(a.replicates,a.backend)
