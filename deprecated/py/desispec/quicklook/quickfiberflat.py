"""
desispec.quicklook.quickfiberflat
=================================

Here will be the fiberflat routines specific to quicklook.

G. Dhungana, 2016
"""

import numpy as np
from desiutil.log import get_logger

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

    # SK. This will not work since the frame object generated here
    # does not have all the parameters used in construction of the
    # input frame. Unfortunately it is not possible to extract all the
    # information from the input either. Possibly correct action would
    # be the directly modify the input frame object

    #- update ivar (like in offline case)

    log = get_logger()

    if frame.flux.shape[0] != fiberflat.fiberflat.shape[0] :
        mess="not same number of fibers: frame.flux.shape[0]={} != fiberflat.fiberflat.shape[0]={}".format(frame.flux.shape[0],fiberflat.fiberflat.shape[0])
        log.error(mess)
        raise RuntimeError(mess)

    if frame.wave.size != fiberflat.wave.size or np.max(np.abs(frame.wave-fiberflat.wave))>0.01 :
        log.warning("interpolating fiber flat")
        flat=np.ones(frame.flux.shape)
        flativar=np.zeros(frame.flux.shape)
        for i in range(frame.flux.shape[0]) :
            flat[i]=np.interp(frame.wave,fiberflat.wave[fiberflat.ivar[i]>0],fiberflat.fiberflat[i,fiberflat.ivar[i]>0])
            flativar[i]=np.interp(frame.wave,fiberflat.wave,fiberflat.ivar[i])
    else :
        flat = fiberflat.fiberflat
        flativar= fiberflat.ivar

    frame.ivar=(frame.ivar>0)*(flativar>0)*(flat>0)/( 1./((frame.ivar+(frame.ivar==0))*(flat**2+(flat==0))) + frame.flux**2/(flativar*flat**4+(flativar*flat==0)) )

    #- flattened flux
    ok=np.where(flat > 0)
    fflux=frame.flux
    fflux[ok]=frame.flux[ok]/flat[ok]

    #- return a frame object

    #fframe=fr.Frame(frame.wave,fflux,fivar,frame.mask,frame.resolution_data,meta=frame.meta,fibermap=frame.fibermap)

    return frame

