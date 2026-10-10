"""Plan independent ROI acquisitions and estimate finite-tissue neighborhood prevalence."""
import numpy as np
from scipy.stats import t as student_t
from evaluate import probabilities,conservative_interval

def plan_acquisition(centroids, core_width, radius, area_fraction=.4, priority=None, seed=1):
    """Return core assignments, known inclusion probabilities and acquisition bounds.

    Coordinates and radius must use the same units. The complete centroid census
    defines the target frame. The budget is expected charged square area, with
    overlapping acquisitions charged separately. Priority is one nonnegative
    number per occupied core, in returned lexicographic core order.
    """
    xy=np.asarray(centroids,float);origin=xy.min(0)
    ij,core=np.unique(np.floor((xy-origin)/core_width).astype(int),axis=0,return_inverse=True)
    counts=np.bincount(core);mean=area_fraction/(1+2*radius/core_width)**2
    if not (0<mean<.99):raise ValueError('Expected inclusion rate must be between 0 and .99')
    p=np.full(len(ij),mean) if priority is None else probabilities(priority,mean)
    selected=np.random.default_rng(seed).random(len(p))<p
    lower=origin+ij*core_width-radius;upper=origin+(ij+1)*core_width+radius
    return {'origin':origin,'ij':ij,'core_of_cell':core,'counts':counts,'probability':p,'selected':selected,'bounds':np.column_stack([lower,upper]),'radius':radius,'core_width':core_width,'N':len(xy)}

def infer_prevalence(plan, selected_core_totals, prediction=None, alpha=.05):
    """Estimate prevalence from complete observed neighborhoods of selected cores.

    selected_core_totals is [number of selected cores, number of endpoints], in
    the ascending selected-core order. Count only anchors inside each core;
    border cells provide neighbor labels, never additional anchor contributions.
    prediction, when supplied, is a fixed external prediction for every core.
    """
    S=plan['selected'];p=plan['probability'];N=plan['N'];counts=plan['counts'];y=np.atleast_2d(selected_core_totals)
    if y.shape[0]!=S.sum():raise ValueError('One row is required per selected core')
    if np.any(y<0) or np.any(y>counts[S,None]):raise ValueError('Core totals must lie between zero and census count')
    m=np.zeros((len(p),y.shape[1])) if prediction is None else np.asarray(prediction,float)
    if np.any(m<0) or np.any(m>counts[:,None]):raise ValueError('Predictions must be bounded by core counts')
    residual=y-m[S];estimate=(m.sum(0)+(residual/p[S,None]).sum(0))/N
    vparts=(1-p[S,None])*residual**2/p[S,None]**2;variance=vparts.sum(0)/N**2
    effective=vparts.sum(0)**2/np.maximum((vparts**2).sum(0),1e-300)
    halfwidth=student_t.ppf(1-alpha/2,np.maximum(effective-1,1))*np.sqrt(variance)
    lo,hi=conservative_interval(estimate[None,:],variance[None,:],p,counts,m,N,alpha)
    return {'estimate':estimate,'standard_error':np.sqrt(variance),'effective_contributions':effective,'approximate_interval':np.column_stack([estimate-halfwidth,estimate+halfwidth]),'bounded_interval':np.column_stack([lo[0],hi[0]])}
