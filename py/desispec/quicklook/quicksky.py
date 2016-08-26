"""
desispec.quicksky

Here will be the sky computing and sky subtraction routines for QL
"""
import sys
import numpy as np
from desispec.sky import SkyModel
from desispec import util
from desispec import frame as fr

def compute_sky(fframe,fibermap=None):
    """
       very simple method of sky computation now. This will be replaced by BOSS like algorithm or other much robust one

    Args: fframe: fiberflat fielded frame object
          fibermap: fibermap object
        
    """
    nspec=fframe.nspec
    nwave=fframe.nwave

    #- Check with fibermap. exit if None
    #- use fibermap from frame itself if exists

    if fframe.fibermap is not None:
        fibermap=fframe.fibermap

    if fibermap is None:
        print("Must have fibermap for Sky compute")
        sys.exit(0)

    #- get the sky
    skyfibers = np.where(fibermap['OBJTYPE'] == b'SKY')[0]
    skyfluxes=fframe.flux[skyfibers]
    skyivars=fframe.ivar[skyfibers]
    if skyfibers.shape[0] > 1:

        weights=skyivars
        #- now get weighted meansky and ivar
        meanskyflux=np.average(skyfluxes,axis=0,weights=weights)
        wtot=weights.sum(axis=0)
        werr2=(weights**2*(skyfluxes-meanskyflux)**2).sum(axis=0)
        werr=np.sqrt(werr2)/wtot
        meanskyivar=1./werr**2
    else:
        meanskyflux=skyfluxes
        meanskyivar=skyivar

    #- Create a 2d- sky model replicating this  
    finalskyflux=np.tile(meanskyflux,nspec).reshape(nspec,nwave)
    finalskyivar=np.tile(meanskyivar,nspec).reshape(nspec,nwave)
        
    skymodel=SkyModel(fframe.wave,finalskyflux,finalskyivar,fframe.mask)
    return skymodel    
    
  
def subtract_sky(fframe,skymodel):
    """
    skymodel: skymodel object. 
    fframe: frame object to do the sky subtraction, should be already fiber flat fielded
    need same number of fibers and same wavelength grid
    """
    #- Check number of specs
    assert fframe.nspec == skymodel.nspec
    assert fframe.nwave == skymodel.nwave

    #- check same wavelength grid, die if not
    if not np.allclose(fframe.wave, skymodel.wave):
        message = "frame and sky not on same wavelength grid"
        raise ValueError(message)

    sflux = fframe.flux-skymodel.flux
    sivar = util.combine_ivar(fframe.ivar.clip(0), skymodel.ivar.clip(0))
    smask = fframe.mask | skymodel.mask
    #- create a frame object now
    sframe=fr.Frame(fframe.wave,sflux,sivar,smask,fframe.resolution_data,meta=fframe.meta,fibermap=fframe.fibermap)
    return sframe
    
