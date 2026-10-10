"""Exact complete-neighborhood inclusion baseline at equal expected imaging area.

Variance is computed by the orthogonal expansion of products of independent
Bernoulli selectors. It uses exhaustive reference outcomes for benchmarking.
"""
from pathlib import Path
import numpy as np,pandas as pd,json,time,itertools,hashlib,zipfile
from scipy.spatial import cKDTree
from neighborhood import nearest_other
from concurrent.futures import ProcessPoolExecutor
from evaluate import load,probabilities,PILOT
ROOT=Path(__file__).resolve().parents[1]

def support_one(job):
 sample,cohort=job;t0=time.perf_counter();L=12.;r=3.;d=load(sample,L);N=int(d['N']);cells=np.load(ROOT/'data/processed'/f'{sample}_cells.npz');xy=cells['xy'].astype(float);marks=cells['marks'];
 if cells['xy'].dtype==np.float32:
  info=json.loads((ROOT/'data/processed'/f'{sample}_info.json').read_text())
  if cohort=='CRC':
   from prepare import raw_crc
   for attempt in range(3):
    try:xy=raw_crc(sample)[['Xt','Yt']].to_numpy(float);break
    except (OSError,ValueError,zipfile.BadZipFile) as e:
     if attempt==2:raise
     time.sleep(2)
  else:xy=pd.read_csv(ROOT/'data/raw'/('spatial_'+sample+'.csv'),skiprows=[1])[['X','Y']].to_numpy(float)
  xy=(xy-xy.min(0))/info['coordinate_spacing']
 ij=np.floor(xy/L).astype(int)
 unique,tile=np.unique(ij,axis=0,return_inverse=True)
 # Stored tile sufficient statistics use float64 coordinates. Reconstruct from raw if float32 rounding changes a boundary.
 if not np.array_equal(unique,d['ij']) or not np.array_equal(np.bincount(tile),d['counts']):
  raise ValueError('Coordinate rounding changed tile assignment: '+sample)
 pairs=[(2,0),(0,1),(2,1)] if cohort=='CRC' else [(1,0),(2,0),(1,2)]
 Y=np.zeros((N,3),dtype=bool)
 for e,(a,b) in enumerate(pairs):
  ai=np.flatnonzero(marks[:,a]);bi=np.flatnonzero(marks[:,b]);Y[ai,e]=nearest_other(xy,ai,bi)<=r
 keep=np.flatnonzero(Y.any(1));tree=cKDTree(xy);groups={}
 for start in range(0,len(keep),20000):
  idx=keep[start:start+20000];neigh=tree.query_ball_point(xy[idx],r,workers=1)
  for i,ns in zip(idx,neigh):
   key=tuple(np.unique(tile[ns]).tolist())
   if key not in groups:groups[key]=np.zeros(3)
   groups[key]+=Y[i]
 keys=list(groups);yv=np.array([groups[k] for k in keys]);M=len(d['counts']);maxk=max(map(len,keys));ids=np.full((len(keys),maxk),M,dtype='int32')
 for a,k in enumerate(keys):ids[a,:len(k)]=k
 ids_r3=np.flatnonzero(d['radii']==3)
 # Cell coordinates are stored losslessly for the final run; equality is an audit against preparation.
 assert np.array_equal(yv.sum(0),d['y'][:,ids_r3].sum(0)),(sample,yv.sum(0),d['y'][:,ids_r3].sum(0))
 if cohort=='CRC':score=d['counts']*(.1+np.expm1(d['features'][:,1]))
 else:score=d['counts']**2
 # Form the expansion coefficients once; support size is at most four for r<L/2.
 coeff={}
 for key,val in groups.items():
  for order in range(1,len(key)+1):
   for sub in itertools.combinations(key,order):
    if sub not in coeff:coeff[sub]=np.zeros(3)
    coeff[sub]+=val
 rows=[]
 for design in ['targeted','uniform']:
  p=probabilities(score,.4) if design=='targeted' else np.full(M,.4)
  ph=probabilities(score,.4/(1+2*r/L)**2) if design=='targeted' else np.full(M,.4/(1+2*r/L)**2)
  vorder=np.zeros((maxk,3))
  for key,val in coeff.items():vorder[len(key)-1]+=val**2*np.prod((1-p[list(key)])/p[list(key)])/N**2
  vh=np.sum((1-ph[:,None])*d['y'][:,ids_r3]**2/ph[:,None],axis=0)/N**2
  for e,eidx in enumerate(ids_r3):
   row={'sample':sample,'cohort':cohort,'design':design,'endpoint':str(d['endpoints'][eidx]),'truth':float(yv[:,e].sum()/N),'support_variance':float(vorder[:,e].sum()),'halo_variance':float(vh[e]),'variance_ratio_support_over_halo':float(vorder[:,e].sum()/vh[e]),'support_groups':len(keys),'max_support_size':maxk}
   row.update({f'variance_order_{k+1}':float(vorder[k,e]) for k in range(maxk)});rows.append(row)
  if design=='targeted':np.savez_compressed(ROOT/'data/processed'/f'{sample}_support.npz',ids=ids,y=yv,p=p,N=N,truth=yv.sum(0)/N)
 print(sample,'groups',len(keys),'seconds',round(time.perf_counter()-t0,2),flush=True)
 return rows

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=3);a=p.parse_args();out=ROOT/'results/support';out.mkdir(parents=True,exist_ok=True)
 names=[(j['sample'],j['cohort']) for j in json.load(open(ROOT/'metadata/samples.json')) if j['sample'] not in PILOT[j['cohort']]]
 with ProcessPoolExecutor(max_workers=a.workers) as ex:results=list(ex.map(support_one,names))
 pd.DataFrame([r for rr in results for r in rr]).to_csv(out/'exact_variances.csv',index=False)
