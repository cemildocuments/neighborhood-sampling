"""Recompute exact support variances from compact grouped sufficient statistics."""
from pathlib import Path
import numpy as np,pandas as pd,itertools
from evaluate import load,probabilities
ROOT=Path(__file__).resolve().parents[1]
rows=[]
for f in sorted((ROOT/'data/processed').glob('*_support.npz')):
 sample=f.stem.removesuffix('_support');cohort='CRC' if sample.startswith('CRC') else 'STARmap';d=load(sample,12);ss=np.load(f);N=int(ss['N']);M=len(ss['p']);groups={tuple(x[x<M]):val for x,val in zip(ss['ids'],ss['y'])};coeff={};maxk=max(map(len,groups));eids=np.flatnonzero(d['radii']==3)
 for key,v in groups.items():
  for k in range(1,len(key)+1):
   for sub in itertools.combinations(key,k):coeff[sub]=coeff.get(sub,np.zeros(3))+v
 score=d['counts']*(.1+np.expm1(d['features'][:,1])) if cohort=='CRC' else d['counts']**2
 for design in ['targeted','uniform']:
  p=probabilities(score,.4) if design=='targeted' else np.full(M,.4);ph=probabilities(score,.4/2.25) if design=='targeted' else np.full(M,.4/2.25);vo=np.zeros((maxk,3))
  for k,v in coeff.items():vo[len(k)-1]+=v**2*np.prod((1-p[list(k)])/p[list(k)])/N**2
  vh=((1-ph[:,None])*d['y'][:,eids]**2/ph[:,None]).sum(0)/N**2
  for e,ei in enumerate(eids):
   row={'sample':sample,'cohort':cohort,'design':design,'endpoint':str(d['endpoints'][ei]),'truth':float(ss['truth'][e]),'support_variance':float(vo[:,e].sum()),'halo_variance':float(vh[e]),'variance_ratio_support_over_halo':float(vo[:,e].sum()/vh[e]),'support_groups':len(groups),'max_support_size':maxk};row.update({f'variance_order_{k+1}':float(vo[k,e]) for k in range(maxk)});rows.append(row)
out=ROOT/'results/support';out.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(out/'exact_variances.csv',index=False)
