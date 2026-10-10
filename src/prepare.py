"""Create reproducible cell and tile representations from public releases."""
from pathlib import Path
import argparse,json,zipfile,time,hashlib,io
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from neighborhood import nearest_other
from concurrent.futures import ProcessPoolExecutor

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/raw'; OUT=ROOT/'data/processed'; OUT.mkdir(parents=True,exist_ok=True)

def raw_crc(name):
    path=RAW/(name+'-features.zip')
    with zipfile.ZipFile(io.BytesIO(path.read_bytes())) as z:
        return pd.read_csv(z.open(name+'.csv'),usecols=['Xt','Yt','CD3','CD20','Keratin','CD45','AREA','CIRC'])

def gates():
    path=ROOT/'metadata/gates.json'
    if path.exists(): return json.loads(path.read_text())
    ds=[raw_crc(s) for s in ['CRC02','CRC09']]
    g={m:float(np.quantile(np.concatenate([d[m].values for d in ds]),.9)) for m in ['CD3','CD20','Keratin']}
    path.write_text(json.dumps(g,indent=2)); return g

def prep_one(job):
    name,cohort,g=job;t=time.perf_counter()
    if cohort=='CRC':
        d=raw_crc(name);xy=d[['Xt','Yt']].to_numpy(float)
        marks=np.column_stack([d[m].values>g[m] for m in ['CD3','CD20','Keratin']])
        aux=np.column_stack([np.log1p(d.CD45),np.log1p(d.AREA),d.CIRC])
        names=['CD3_high','CD20_high','Keratin_high']
        pairs=[(2,0),(0,1),(2,1)]
        donor=name
    else:
        d=pd.read_csv(RAW/('spatial_'+name+'.csv'),skiprows=[1]);xy=d[['X','Y']].to_numpy(float)
        marks=np.column_stack([d.top_level_cell_type.values==m for m in ['Micro','Astro','Oligo']])
        aux=np.zeros((len(d),3));names=['Micro','Astro','Oligo'];pairs=[(1,0),(2,0),(1,2)]
        meta=pd.read_csv(RAW/'metadata.csv',skiprows=[1],usecols=['label','donor_id']); ids=meta.loc[meta.label==name,'donor_id'].unique();assert len(ids)==1;donor=str(ids[0])
    valid=np.isfinite(xy).all(1)&np.isfinite(aux).all(1);xy=xy[valid];marks=marks[valid];aux=aux[valid]
    assert np.unique(xy,axis=0).shape[0]==len(xy),'Duplicate centroids require explicit resolution'
    spacing=float(np.median(cKDTree(xy).query(xy,k=2,workers=1)[0][:,1]));xy=(xy-xy.min(0))/spacing
    np.savez_compressed(OUT/(name+'_cells.npz'),xy=xy.astype('float64'),marks=marks,aux=aux.astype('float32'))
    settings=[]
    for L in [12.,24.,48.]:
        # Fixed grid origin is label independent. Each cell has exactly one anchor core.
        ij=np.floor(xy/L).astype('int32');ij_unique,tile=np.unique(ij,axis=0,return_inverse=True);M=len(ij_unique);N=len(xy)
        nj=np.bincount(tile,minlength=M).astype(float)
        frac=xy/L-np.floor(xy/L);edge=L*np.minimum(frac,1-frac).min(1)
        sums=np.column_stack([np.bincount(tile,weights=aux[:,k],minlength=M) for k in range(3)])
        means=sums/nj[:,None]
        xx=np.column_stack([np.log1p(nj),means])
        y=[];crop=[];guard=[];namesout=[];r_out=[];counts=[]
        for k,lab in enumerate(names):
            v=marks[:,k].astype(float);y.append(np.bincount(tile,weights=v,minlength=M));crop.append(y[-1]);guard.append(y[-1]);namesout.append(lab+'_abundance');r_out.append(0.)
        for radius in [1.5,3.,6.]:
            for a,b in pairs:
                B=np.flatnonzero(marks[:,b]);A=np.flatnonzero(marks[:,a]);av=np.zeros(N,dtype=bool);cv=np.zeros(N,dtype=bool)
                if len(B):
                    dist=nearest_other(xy,A,B);av[A]=dist<=radius
                    # Crop each acquisition separately; cross-tile neighbors are not observed.
                    for j in np.unique(tile[A]):
                        ai=A[tile[A]==j];bi=B[tile[B]==j]
                        if len(bi):cv[ai]=nearest_other(xy,ai,bi)<=radius
                y.append(np.bincount(tile,weights=av,minlength=M));crop.append(np.bincount(tile,weights=cv,minlength=M));guard.append(np.bincount(tile,weights=av&(edge>=radius),minlength=M));namesout.append(names[a]+'_near_'+names[b]+'_r'+str(radius));r_out.append(radius)
        path=OUT/(name+'_L'+str(int(L))+'.npz')
        np.savez_compressed(path,N=N,counts=nj,features=xx,y=np.array(y).T,crop=np.array(crop).T,guard=np.array(guard).T,ij=ij_unique,radii=r_out,endpoints=namesout)
        settings.append({'L':L,'tiles':M,'truth':(np.array(y).sum(1)/N).tolist()})
    res={'sample':name,'cohort':cohort,'donor':donor,'n_cells':len(xy),'coordinate_spacing':spacing,'runtime_s':time.perf_counter()-t,'settings':settings}
    (OUT/(name+'_info.json')).write_text(json.dumps(res,indent=2));print(name,len(xy),round(res['runtime_s'],1),flush=True);return res

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');p.add_argument('--workers',type=int,default=6);a=p.parse_args();g=gates()
    crc=['CRC02','CRC09'] if a.pilot else [f'CRC{i:02d}' for i in range(2,18)]
    brain=['8months-control-replicate_1','13months-control-replicate_1'] if a.pilot else sorted(f.stem.removeprefix('spatial_') for f in RAW.glob('spatial_*.csv'))
    jobs=[(s,'CRC',g) for s in crc]+[(s,'STARmap',g) for s in brain]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:rows=list(ex.map(prep_one,jobs))
    (ROOT/'metadata'/('pilot_samples.json' if a.pilot else 'samples.json')).write_text(json.dumps(rows,indent=2))
