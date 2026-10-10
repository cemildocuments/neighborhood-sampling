"""Design-based inference and reproducible Monte Carlo evaluation."""
from pathlib import Path
import argparse,json,time,hashlib,os
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.ensemble import ExtraTreesRegressor
from concurrent.futures import ProcessPoolExecutor
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/processed'
PILOT={'CRC':['CRC02','CRC09'],'STARmap':['8months-control-replicate_1','13months-control-replicate_1']}

def probabilities(score,mean,floor=.25):
    score=np.maximum(np.asarray(score,float),1e-10);score=score/score.mean()
    lo,hi=0.,1./score.min()
    for _ in range(70):
        mid=(lo+hi)/2;p=np.minimum(.99,floor*mean+mid*score)
        if p.mean()>mean:hi=mid
        else:lo=mid
    return np.minimum(.99,floor*mean+(lo+hi)/2*score)

def load(name,L):return dict(np.load(DATA/f'{name}_L{int(L)}.npz',allow_pickle=False))

def prediction(cohort,L,development_exclude=None):
    names=[s for s in PILOT[cohort] if s!=development_exclude]
    d=[load(s,L) for s in names];xx=np.concatenate([x['features'] for x in d]);yy=np.concatenate([x['y']/x['counts'][:,None] for x in d]);ww=np.concatenate([x['counts'] for x in d])
    model=ExtraTreesRegressor(n_estimators=128,min_samples_leaf=8,max_features=1.,random_state=17419,n_jobs=1)
    model.fit(xx,yy,sample_weight=ww);return model

def conservative_interval(est,vhat,p,counts,pred,N,alpha=.05):
    # Two Bernstein bounds and a union bound; outcomes are bounded by core cell count.
    bounds=np.maximum(pred,counts[:,None]-pred)
    c=np.max((1-p[:,None])*bounds**2/p[:,None]**2,axis=0)
    b=np.max(bounds/p[:,None],axis=0)
    lv=np.log(2/alpha);le=np.log(4/alpha)
    raw=vhat*N*N;a=raw+c*lv/3
    upper=a+c*lv+np.sqrt((c*lv)**2+2*a*c*lv)
    hw=(np.sqrt(2*upper*le)+b*le/3)/N
    return np.maximum(0,est-hw),np.minimum(1,est+hw)

def summary(est,se,truth,critical,lo=None,hi=None):
    if lo is None:lo=est-critical*se;hi=est+critical*se
    error=est-truth[None,:]
    return {'bias':error.mean(0),'rmse':np.sqrt((error**2).mean(0)), 'coverage':((lo<=truth)&(hi>=truth)).mean(0),'width':(hi-lo).mean(0),'mean_se':se.mean(0),'empirical_sd':est.std(0,ddof=1),'mcse_bias':est.std(0,ddof=1)/np.sqrt(len(est))}

def run_one(job):
    sample,cohort,L,R,budgets,phase,block=job;t0=time.perf_counter();d=load(sample,L);y=d['y'];cropped=d['crop'];counts=d['counts'];N=int(d['N']);M=len(counts);truth=y.sum(0)/N
    model=prediction(cohort,L, sample if phase=='pilot' else None);pred=np.clip(model.predict(d['features']),0,1)*counts[:,None]
    # Auxiliary information available before high-plex acquisition: nuclei census and optional CD45 preview.
    if cohort=='CRC':score=counts*(.1+np.expm1(d['features'][:,1]))
    else:score=counts**2
    records=[];rawsave={};seed=int.from_bytes(hashlib.sha256(f'{sample}-{L}-{phase}-{block}'.encode()).digest()[:8],'little')
    rng=np.random.default_rng(seed)
    # Common random numbers couple designs. Held-out outcomes never enter probabilities or predictions.
    U=rng.random((R,M)).astype('float32')
    radii=np.unique(d['radii']);radii=radii[radii>0]
    for radius in radii:
      ids=np.r_[np.arange(3),np.flatnonzero(d['radii']==radius)]
      yy=y[:,ids];cc=cropped[:,ids];mm=pred[:,ids];tt=truth[ids]
      for budget in budgets:
       for design in ['targeted','uniform']:
        for halo in [False,True]:
         overhead=(1+2*radius/L)**2 if halo else 1.
         mean=budget/overhead
         p=probabilities(score,mean) if design=='targeted' else np.full(M,mean)
         S=(U<p).astype('float64');n=S.sum(1);crit=student_t.ppf(.975,np.maximum(n-1,1))[:,None]
         observed=yy if halo else cc
         weighted=observed/p[:,None]
         est=S@weighted/N;vhat=S@((1-p[:,None])*observed**2/p[:,None]**2)/(N*N)
         prefix=('halo' if halo else 'crop')+'_'+design
         outputs=[(prefix+'_HT',est,np.sqrt(vhat),None,None)]
         vparts=(1-p[:,None])*observed**2/p[:,None]**2
         neff=(vhat*N*N)**2/np.maximum(S@(vparts**2),1e-300)
         qeff=student_t.ppf(.975,np.maximum(neff-1,1))
         outputs.append((prefix+'_effective',est,np.sqrt(vhat),est-qeff*np.sqrt(vhat),est+qeff*np.sqrt(vhat)))
         denom=S@counts;unw=(S@observed)/np.maximum(denom[:,None],1)
         # Cluster sandwich ratio uncertainty, not an independent-cell error bar.
         A=S@((1-p[:,None])*observed**2); B=S@((1-p[:,None])*observed*counts[:,None]); C=S@((1-p)*counts**2)
         vu=np.maximum(A-2*unw*B+unw**2*C[:,None],0)/np.maximum(denom[:,None],1)**2
         outputs.append((prefix+'_unweighted',unw,np.sqrt(vu),None,None))
         if halo:
          residual=yy-mm;eh=mm.sum(0)/N+(S@(residual/p[:,None]))/N
          vh=S@((1-p[:,None])*residual**2/p[:,None]**2)/(N*N)
          outputs.append((prefix+'_assisted',eh,np.sqrt(vh),None,None))
          rvparts=(1-p[:,None])*residual**2/p[:,None]**2
          reff=(vh*N*N)**2/np.maximum(S@(rvparts**2),1e-300)
          rq=student_t.ppf(.975,np.maximum(reff-1,1))
          outputs.append((prefix+'_assisted_effective',eh,np.sqrt(vh),eh-rq*np.sqrt(vh),eh+rq*np.sqrt(vh)))
          low,high=conservative_interval(eh,vh,p,counts,mm,N)
          outputs.append((prefix+'_bounded',eh,np.sqrt(vh),low,high))
         for label,ee,ss,low,high in outputs:
          met=summary(ee,ss,tt,crit,low,high)
          for k,eidx in enumerate(ids):
           row={'sample':sample,'cohort':cohort,'L':L,'radius':radius,'budget':budget,'design':design,'method':label,'endpoint':str(d['endpoints'][eidx]),'truth':tt[k],'n_cells':N,'n_tiles':M,'expected_fields':p.sum(),'mean_fields':n.mean(),'expected_area_units':p.sum()*L*L*overhead,'replicates':R}
           row.update({name:float(val[k]) for name,val in met.items()});records.append(row)
         if L==12 and radius==3 and budget==.4:
          rawsave[prefix+'_HT_est']=est.astype('float32');rawsave[prefix+'_HT_se']=np.sqrt(vhat).astype('float32')
          if halo:rawsave[prefix+'_assisted_est']=eh.astype('float32');rawsave[prefix+'_assisted_se']=np.sqrt(vh).astype('float32')
    out=ROOT/'results'/phase;out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(records).to_csv(out/f'{sample}_L{int(L)}_b{block}.csv',index=False)
    if rawsave:np.savez_compressed(out/f'{sample}_replicates_b{block}.npz',**rawsave)
    elapsed=time.perf_counter()-t0;print(sample,L,len(records),round(elapsed,2),flush=True);return {'sample':sample,'L':L,'runtime_s':elapsed,'peak_rss_kb':__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['pilot','validation'],default='pilot');p.add_argument('--replicates',type=int,default=2000);p.add_argument('--workers',type=int,default=4);p.add_argument('--blocks',type=int,default=1);p.add_argument('--L',type=float,nargs='+',default=[12,24,48]);a=p.parse_args()
    if a.phase=='pilot':names=[(s,c) for c,ss in PILOT.items() for s in ss]
    else:
     names=[(j['sample'],j['cohort']) for j in json.load(open(ROOT/'metadata/samples.json')) if j['sample'] not in PILOT[j['cohort']]]
    jobs=[(s,c,L,a.replicates,[.2,.4,.6,.8],a.phase,block) for block in range(a.blocks) for s,c in names for L in a.L]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:rows=list(ex.map(run_one,jobs))
    out=ROOT/'results'/a.phase; frames=[pd.read_csv(p) for p in sorted(out.glob('*_L*_b*.csv'))];pd.concat(frames,ignore_index=True).to_csv(out/'all_results.csv',index=False);(out/'runtime.json').write_text(json.dumps(rows,indent=2))
