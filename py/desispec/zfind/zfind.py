import numpy as np

class ZfindBase(object):
    def __init__(self, wave, flux, ivar, R=None, results=None):
        """
        Base class of classification / redshift finders.
        
        Args:
            wave : 1D[nwave] wavelength grid [Angstroms]
            flux : 2D[nspec, nwave] flux [erg/s/cm2/A]
            ivar : 2D[nspec, nwave] inverse variance of flux
            
        Optional:
            R : 1D[nspec] list of resolution objects
            results : ndarray with keys such as z, zerr, zwarn (see below)
                all results.dtype.names are added to this object
            
        Subclasses should perform classification and redshift fitting
        upon initialization and set the following member variables:
            nspec : number of spectra
            nwave : number of wavelegths (may be resampled from input)
            z     : 1D[nspec] best fit redshift
            zerr  : 1D[nspec] redshift uncertainty estimate
            zwarn : 1D[nspec] integer redshift warning bitmask (details TBD)
            type  : 1D[nspec] classification [GALAXY, QSO, STAR, ...]
            subtype : 1D[nspec] sub-classification 
            wave  : 1D[nwave] wavelength grid used; may be resampled from input
            flux  : 2D[nspec, nwave] flux used; may be resampled from input
            ivar  : 2D[nspec, nwave] ivar of flux
            model : 2D[nspec, nwave] best fit model
            
            chi2?
            zbase?
            
        For the purposes of I/O, it is possible to create a ZfindBase
        object that contains only the results, without the input
        wave, flux, ivar, or output model.
        """
        #- Inputs
        if flux is not None:
            nspec, nwave = flux.shape
            self.nspec = nspec
            self.nwave = nwave
            self.wave = wave
            self.flux = flux
            self.ivar = ivar
            self.R = R
        
        #- Outputs to fill
        if results is None:
            self.model = np.zeros((nspec, nwave), dtype=flux.dtype)
            self.z = np.zeros(nspec)
            self.zerr = np.zeros(nspec)
            self.zwarn = np.zeros(nspec, dtype=int)
            self.type = np.zeros(nspec, dtype='S20')
            self.subtype = np.zeros(nspec, dtype='S20')
        else:
            for key in results.dtype.names:
                self.__setattr__(key.lower(), results[key])
    
