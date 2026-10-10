"""Geometry utilities with explicit exclusion of the anchor cell itself."""
import numpy as np
from scipy.spatial import cKDTree

def nearest_other(xy, anchors, neighbors):
    """Distance to the nearest distinct neighbor, even for overlapping marks."""
    anchors=np.asarray(anchors);neighbors=np.asarray(neighbors)
    if not len(neighbors):return np.full(len(anchors),np.inf)
    distance,idx=cKDTree(xy[neighbors]).query(xy[anchors],k=2)
    valid=idx[:,0]<len(neighbors)
    same=np.zeros(len(anchors),dtype=bool)
    same[valid]=neighbors[idx[valid,0]]==anchors[valid]
    return np.where(same,distance[:,1],distance[:,0])
