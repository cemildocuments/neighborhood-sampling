"""Reproduce the frozen acquisition experiments and compare numerical outputs."""
from pathlib import Path
import subprocess,sys,os,argparse,json,time
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
(ROOT/'results').mkdir(parents=True,exist_ok=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['full','smoke','from-source'],default='full');p.add_argument('--workers',type=int,default=6);p.add_argument('--backend',choices=['auto','cpu','mps'],default='auto');a=p.parse_args();env=os.environ.copy()
 for k in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS']:env[k]='1'
 t0=time.perf_counter()
 def run(script,*args):subprocess.run([sys.executable,str(ROOT/'src'/script),*map(str,args)],check=True,env=env,cwd=ROOT)
 if a.mode=='from-source':run('download.py','--workers',min(a.workers,3));run('prepare.py','--workers',a.workers);run('support_benchmark.py','--workers',min(a.workers,3))
 run('test_inference.py')
 if a.mode=='smoke':
  # Two held-out specimens, one scale, with a small acquisition block.
  code="import sys;sys.path.insert(0,'src');from evaluate import run_one;run_one(('CRC03','CRC',12,200,[.4],'smoke',0));run_one(('13months-control-replicate_2','STARmap',12,200,[.4],'smoke',0))"
  subprocess.run([sys.executable,'-c',code],cwd=ROOT,env=env,check=True)
 else:
  run('evaluate.py','--phase','validation','--replicates',2000,'--blocks',5,'--workers',a.workers)
  run('support_cached.py');run('summarize.py');run('simulate.py','--replicates',50000,'--backend',a.backend)
  checks=[]
  for rel,keys,tol in [('summary/validation.csv',['sample','L','radius','budget','method','endpoint'],1e-10),('support/exact_variances.csv',['sample','design','endpoint'],1e-10),('simulation/simulation_results.csv',['pattern','prevalence','method'],5e-5)]:
   actual=pd.read_csv(ROOT/'results'/rel).sort_values(keys).reset_index(drop=True);expected=pd.read_csv(ROOT/'expected'/rel).sort_values(keys).reset_index(drop=True)
   assert actual.shape==expected.shape,(rel,actual.shape,expected.shape)
   assert actual[keys].equals(expected[keys]),rel+' identifiers differ'
   for k in actual.select_dtypes('number').columns:
    assert np.allclose(actual[k],expected[k],atol=tol,rtol=tol,equal_nan=True),(rel,k,float(np.max(abs(actual[k]-expected[k]))))
   checks.append({'file':rel,'rows':len(actual),'absolute_and_relative_tolerance':tol,'status':'passed'})
  (ROOT/'results/reproduction_check.json').write_text(json.dumps({'checks':checks,'elapsed_s':time.perf_counter()-t0},indent=2))
 print('Reproduction completed in',round(time.perf_counter()-t0,1),'seconds',flush=True)
if __name__=='__main__':main()
