from pathlib import Path
import pandas as pd,numpy as np,json
ROOT=Path(__file__).resolve().parents[1]
d=pd.read_csv(ROOT/'results/validation/all_results.csv')
keys=['sample','cohort','L','radius','budget','design','method','endpoint']
rows=[]
for kk,g in d.groupby(keys,sort=True):
 n=g.replicates.to_numpy();total=n.sum();bias=np.average(g.bias,weights=n)
 sd=np.sqrt((np.sum((n-1)*g.empirical_sd**2)+np.sum(n*(g.bias-bias)**2))/(total-1))
 row=dict(zip(keys,kk));row.update({k:g[k].iloc[0] for k in ['truth','n_cells','n_tiles','expected_fields','expected_area_units']})
 row.update({k:np.average(g[k],weights=n) for k in ['mean_fields','coverage','width','mean_se']})
 row.update(replicates=int(total),bias=bias,rmse=np.sqrt(np.average(g.rmse**2,weights=n)),empirical_sd=sd,mcse_bias=sd/np.sqrt(total));rows.append(row)
a=pd.DataFrame(rows);a['relative_bias']=a.bias/a.truth;a['relative_rmse']=a.rmse/a.truth
out=ROOT/'results/summary';out.mkdir(exist_ok=True);a.to_csv(out/'validation.csv',index=False)
p=a[(a.L==12)&(a.radius==3)&(a.budget==.4)&a.endpoint.str.contains('_near_')].copy();p.to_csv(out/'primary.csv',index=False)
summary=p.groupby(['cohort','method']).agg(coverage=('coverage','mean'),min_coverage=('coverage','min'),max_coverage=('coverage','max'),rmse=('rmse','mean'),width=('width','mean'),relative_bias=('relative_bias','mean'),maxabsrelativebias=('relative_bias',lambda x:abs(x).max()),absbias=('bias',lambda x:abs(x).mean()),relative_rmse=('relative_rmse','mean'))
summary.to_csv(out/'primary_summary.csv')
print(summary.to_string())
s=pd.read_csv(ROOT/'results/support/exact_variances.csv');print('\nSUPPORT',s.groupby(['cohort','design']).variance_ratio_support_over_halo.agg(['min','median','max','mean']).to_string())
print('\nTARGETED cases ratio >1',s[s.design=='targeted'].groupby('cohort').variance_ratio_support_over_halo.apply(lambda x:(x>1).sum()))
print('\nPrimary bias/MCSE halo max',np.max(abs(p[p.method=='halo_targeted_effective'].bias/p[p.method=='halo_targeted_effective'].mcse_bias)))
print('\nPrimary smallsample',p[p.method=='halo_targeted_effective'].groupby('cohort').expected_fields.agg(['min','median','max']))
print('\nSensitivity coverage',a[(a.method=='halo_targeted_effective')&a.endpoint.str.contains('_near_')].groupby(['cohort','L']).coverage.agg(['min','median','mean','max']))
print('\nAbundance',a[(a.L==12)&(a.radius==3)&(a.budget==.4)&a.endpoint.str.contains('abundance')&(a.method.isin(['crop_targeted_effective','halo_targeted_effective']))].groupby(['cohort','method']).coverage.agg(['mean','min','max']))
