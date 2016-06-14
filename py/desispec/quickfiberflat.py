"""
desispec.quickfiberflat

Here will be the fiberflat routines specific to quicklook.

G. Dhungana, 2016
"""

import numpy as np

def compute_fiberflat():
    """
    computes fiberflat: A boss like algorithm writing in progress and will fit in here.

        Args:
    """

def apply_fiberflat(frame,fiberflat):
    """
    Args: frame: desispec.frame.Frame object
          fiberflat: desispec.fiberflat.Fiberflat object
    """
    from desispec import frame as fr
    
    #- update ivar (like in offline case)
    
    fivar=(frame.ivar>0)*(fiberflat.ivar>0)*(fiberflat.fiberflat>0)/( 1./((frame.ivar+(frame.ivar==0))*(fiberflat.fiberflat**2+(fiberflat.fiberflat==0))) + frame.flux**2/(fiberflat.ivar*fiberflat.fiberflat**4+(fiberflat.ivar*fiberflat.fiberflat==0)) )

    #- flattened flux
    ok=np.where(fiberflat.fiberflat > 0)
    fflux=frame.flux
    fflux[ok]=frame.flux[ok]/fiberflat.fiberflat[ok]

    #- return a frame object
    fframe=fr.Frame(frame.wave,fflux,fivar,frame.mask,frame.resolution_data,meta=frame.meta,fibermap=frame.fibermap)
    
    return fframe
    
