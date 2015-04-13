import os

import numpy as np
from redmonster.physics.zfinder import Zfinder
from redmonster.physics.zfitter import Zfitter
from redmonster.physics.zpicker import Zpicker

from desispec.interpolation import resample_flux

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
    

class RedMonsterZfind(ZfindBase):
    def __init__(self, wave, flux, ivar, R=None, dloglam=1e-4):
        """
        Uses Redmonster to classify and find redshifts.  See ZfindBase class
        for inputs/outputs.
        
        TODO: document redmonster specific output variables
        """
        #- RedMonster templates don't quite go far enough into the blue,
        #- so chop off some data
        ii, = np.where(wave>3965)
        wave = wave[ii]
        flux = flux[:, ii]
        ivar = ivar[:, ii]

        #- Resample inputs to a loglam grid
        start = round(np.log10(wave[0]), 4)+dloglam
        stop = round(np.log10(wave[-1]), 4)
        
        nwave = int((stop-start)/dloglam)
        loglam = start + np.arange(nwave)*dloglam

        nspec = flux.shape[0]
        self.flux = np.empty((nspec, nwave))
        self.ivar = np.empty((nspec, nwave))

        for i in range(nspec):
            self.flux[i], self.ivar[i] = resample_flux(10**loglam, wave, flux[i], ivar[i])

        self.dloglam = dloglam
        self.loglam = loglam
        self.wave = 10**loglam
        self.nwave = nwave
        self.nspec = nspec

        #- list of (templatename, zmin, zmax) to fix
        self.template_dir = os.getenv('REDMONSTER_DIR')+'/templates/'
        self.templates = [
            ('ndArch-spEigenStar-55734.fits', -0.005, 0.005),
            ('ndArch-ssp_em_galaxy-v000.fits', 0.6, 1.6),
            # ('ndArch-ssp_em_galaxy_quickdesi-v000.fits', 0.6, 1.6),
            ('ndArch-QSO-V003.fits', 0.0, 3.5),
        ]

        #- Find and refine best redshift per template
        self.zfinders = list()
        self.zfitters = list()
        for template, zmin, zmax in self.templates:
            zfind = Zfinder(self.template_dir+template, npoly=2, zmin=zmin, zmax=zmax)
            zfind.zchi2(self.flux, self.loglam, self.ivar, npixstep=2)
            zfit = Zfitter(zfind.zchi2arr, zfind.zbase)
            zfit.z_refine()     

            self.zfinders.append(zfind)
            self.zfitters.append(zfit)

        #- Create wrapper object needed for zpicker
        specobj = _RedMonsterSpecObj(self.wave, self.flux, self.ivar)
        flags = list()
        for i in range(len(self.zfitters)):
            flags.append(self.zfinders[i].zwarning.astype(int) | \
                         self.zfitters[i].zwarning.astype(int))

        #- Zpicker
        self.zpicker = Zpicker(specobj,
            self.zfinders[0], self.zfitters[0], flags[0],
            self.zfinders[1], self.zfitters[1], flags[1],
            self.zfinders[2], self.zfitters[2], flags[2])
            
        #- Fill in outputs
        self.type = np.asarray(self.zpicker.type, dtype='S20')
        self.subtype = np.asarray(self.zpicker.subtype, dtype='S20')
        self.z = np.array([self.zpicker.z[i,0] for i in range(nspec)])
        self.zerr = np.array([self.zpicker.z_err[i,0] for i in range(nspec)])
        self.zwarn = np.array([self.zpicker.zwarning[i].astype(int) for i in range(nspec)])
        self.model = self.zpicker.models
        
        
#- This is a container class needed by Redmonster zpicker
class _RedMonsterSpecObj(object):
    def __init__(self, wave, flux, ivar, dof=None):
        nspec, nwave = flux.shape
        self.wave = wave
        self.flux = flux
        self.ivar = ivar
        if dof is None:
            self.dof = np.ones(nspec) * nwave
        else:
            self.dof = dof

        #- Leftover BOSS-isms
        self.plate = self.mjd = self.fiberid = self.npix = 0
        self.hdr = None
        self.plugmap = None

               
#-------------------------------------------------------------------------
#- Test code during development
# from desispec import io
# def test_zfind(nspec=None):
#     print "Reading bricks"
#     brick = dict()
#     for channel in ('b', 'r', 'z'):
#         filename = io.findfile('brick', band=channel, brickid='3582m005')
#         brick[channel] = io.Brick(filename)
#     
#     print "Coadding individual channels and exposures"
#     wb = brick['b'].get_wavelength_grid()
#     wr = brick['r'].get_wavelength_grid()
#     wz = brick['z'].get_wavelength_grid()
#     wave = np.concatenate([wb, wr, wz])
#     np.ndarray.sort(wave)
#     nwave = len(wave)
# 
#     if nspec is None:
#         nspec = brick['b'].get_num_targets()
#         
#     flux = np.zeros((nspec, nwave))
#     ivar = np.zeros((nspec, nwave))
# 
#     for i, targetid in enumerate(brick['b'].get_target_ids()):
#         if i>=nspec: break
#         xwave = list()
#         xflux = list()
#         xivar = list()
#         for channel in ('b', 'r', 'z'):
#             exp_flux, exp_ivar, resolution, info = brick[channel].get_target(targetid)
#             weights = np.sum(exp_ivar, axis=0)
#             ii, = np.where(weights > 0)
#             xwave.extend(brick[channel].get_wavelength_grid()[ii])
#             xflux.extend(np.average(exp_flux[:,ii], weights=exp_ivar[:,ii], axis=0))
#             xivar.extend(weights[ii])
#                 
#         xwave = np.array(xwave)
#         xivar = np.array(xivar)
#         xflux = np.array(xflux)
#                 
#         ii = np.argsort(xwave)
#         flux[i], ivar[i] = resample_flux(wave, xwave[ii], xflux[ii], xivar[ii])
#             
#     zf = RedMonsterZfind(wave, flux, ivar)
#     return zf
    
#     return wave, xwave[ii], xflux[ii], xivar[ii]
    
    