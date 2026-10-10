"""Exact small-population checks for estimators, variances, and interval bounds."""
import itertools,json
from pathlib import Path
import numpy as np
from evaluate import conservative_interval
ROOT=Path(__file__).resolve().parents[1]
(ROOT/'results').mkdir(parents=True,exist_ok=True)
from neighborhood import nearest_other
xx=np.array([[0.,0.],[2.,0.],[4.,0.]])
assert np.array_equal(nearest_other(xx,np.array([0,1,2]),np.array([0,2])),[4.,2.,4.])
assert np.isinf(nearest_other(xx,np.array([0]),np.array([0]))[0])
rng=np.random.default_rng(609913);checks=[]
for iteration in range(60):
 M=7;p=rng.uniform(.1,.85,M);N=100.;counts=rng.integers(2,20,M).astype(float);y=rng.uniform(0,1,(M,3))*counts[:,None];m=rng.uniform(0,1,(M,3))*counts[:,None]
 S=np.array(list(itertools.product([0.,1.],repeat=M)));mass=np.prod(np.where(S,p,1-p),axis=1);res=y-m
 est=m.sum(0)/N+S@(res/p[:,None])/N;truth=y.sum(0)/N;var=((1-p[:,None])*res**2/p[:,None]).sum(0)/N**2;vh=S@((1-p[:,None])*res**2/p[:,None]**2)/N**2
 assert np.max(np.abs(mass@est-truth))<1e-12
 assert np.max(np.abs(mass@vh-var))<1e-12
 assert np.max(np.abs(mass@((est-truth)**2)-var))<1e-12
 lo,hi=conservative_interval(est,vh,p,counts,m,N);coverage=mass@((lo<=truth)&(hi>=truth));assert coverage.min()>=.95-1e-12
 # Products for overlapping neighborhood supports, including identical and nested supports.
 supports=[(0,),(0,1),(1,2,3),(2,3),(3,4,5,6)];values=rng.uniform(0,3,(len(supports),2));target=values.sum(0)
 zs=np.column_stack([S[:,h].prod(1) for h in supports]);pi=np.array([p[list(h)].prod() for h in supports]);ex=zs@(values/pi[:,None]);vemp=mass@((ex-target)**2)
 co={}
 for h,val in zip(supports,values):
  for k in range(1,len(h)+1):
   for sub in itertools.combinations(h,k):co[sub]=co.get(sub,np.zeros(2))+val
 van=sum(val**2*np.prod((1-p[list(h)])/p[list(h)]) for h,val in co.items())
 assert np.max(np.abs(vemp-van))<1e-10
 checks.append({'iteration':iteration,'bound_min_coverage':float(coverage.min()),'support_variance_max_error':float(np.max(np.abs(vemp-van)))})
for f in (ROOT/'data/processed').glob('*_L*.npz'):
 d=np.load(f);assert np.all(d['crop']<=d['y']);assert np.all(d['y']<=d['counts'][:,None]);assert np.array_equal(d['crop'][:,:3],d['y'][:,:3]);assert d['counts'].sum()==d['N']
from inference import plan_acquisition,infer_prevalence
xy=rng.uniform(0,100,(500,2));plan=plan_acquisition(xy,12,3,seed=51)
yall=np.column_stack([plan['counts']*.2,plan['counts']*.4]);sel=plan['selected'];api=infer_prevalence(plan,yall[sel]);expected=(yall[sel]/plan['probability'][sel,None]).sum(0)/plan['N']
assert np.allclose(api['estimate'],expected)
low=plan['bounds'][:,:2];high=plan['bounds'][:,2:];local=xy-plan['origin'];base=plan['ij']*12+plan['origin'];assert np.allclose(low,base-3) and np.allclose(high,base+15)
out={'exact_populations':len(checks),'selection_patterns_per_population':128,'checks':checks,'status':'passed'}
(ROOT/'results/exact_tests.json').write_text(json.dumps(out,indent=2));print('Passed:',len(checks),'exact populations, each with 128 acquisition patterns; tile integrity checks passed.')
