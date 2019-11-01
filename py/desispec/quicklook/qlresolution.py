"""
desispec.quicklook.qlresolution
===============================

Quicklook version of resolution object that can 
calculate resolution efficiently from psf information

Author: Sami Kama

"""

import numpy as np
import scipy.sparse
import scipy.special

class QuickResolution(scipy.sparse.dia_matrix):
    """
    Quicklook version of the resolution mimicking desispec.resolution.Resolution 
    with some reduction in dimentionality. Contains code from Resolution implementation
    Note that this is similar to desispec.resolution.Resolution, though faster and differing 
    in implementation details that should be cross checked before merging these 
    or replacing one with the other
    """
    def __init__(self,mu=None,sigma=None,wdict=None,waves=None,ndiag=9):
        self.__ndiag=ndiag
        if ndiag & 0x1 == 0:
            raise ValueError("Need odd numbered diagonals, got %d"%ndiag)
        def _binIntegral(x,mu=None,sigma=None):
            """
            x: bin boundaries vector (self.__ndiag,)
            mu: means vector of shape[nwave,1]
            sigma: sigmas of shape[nwave,1]
            """
            nvecs=1
            if sigma is not None:
                nvecs=sigma.shape[0]
            if mu is None:
                mu=np.zeros((nvecs,1))
            if sigma is None:
                sigma=np.ones(mu.shape)*0.5
            sx=(np.tile(x,(mu.shape[0],1))-mu)/(sigma*np.sqrt(2))
            return 0.5*(np.abs(np.diff(scipy.special.erf(sx))))

        mnone=mu is None
        snone=sigma is None
        dnone=wdict is None
        wnone=waves is None
        if snone:
            if wnone or dnone:
                raise ValueError('Cannot initialize Resolution data need sigma or wdict and waves')
            else:
                from desiutil import funcfits as dufits
                sigma=dufits.func_val(waves,wdict)
        nwave = len(sigma)
        s=sigma.reshape((nwave,1))
        bins=np.arange(ndiag,0,-1)
        bins=bins-(bins[0]+bins[-1])/2.0
        x=np.concatenate([bins+0.5,bins[-1:]-0.5])
        self.offsets=bins
        rdata=_binIntegral(x,mu=mu,sigma=s).T
        
        scipy.sparse.dia_matrix.__init__(self,(rdata,self.offsets),(nwave,nwave))
       
