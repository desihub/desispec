"""
desispec.zfind.redmonster
=========================

Classes for use with the redmonster package.
"""
from __future__ import division, absolute_import

import os

import numpy as np
import time
import json

from desispec.zfind import ZfindBase
from desispec.interpolation import resample_flux
from desispec.log import get_logger

class RedMonsterZfind(ZfindBase):
    """Class documentation goes here.
    """
    def __init__(self, wave, flux, ivar, R=None, dloglam=1e-4, objtype=None,
                 zrange_galaxy=(0.0, 1.6), zrange_qso=(0.0, 3.5), zrange_star=(-0.005, 0.005),
                 group_galaxy=0, group_qso=1, group_star=2, nproc=1, npoly=2):
        """Uses Redmonster to classify and find redshifts.

        See :class:`desispec.zfind.zfind.ZfindBase` class for inputs/outputs.

        optional:
            objtype : list or string of template object types to try
                [ELG, LRG, QSO, GALAXY, STAR]

        TODO: document redmonster specific output variables
        """
        from redmonster.physics.zfinder import ZFinder
        from redmonster.physics.zfitter import ZFitter
        from redmonster.physics.zpicker2 import ZPicker
        
        log=get_logger()
        

        #- RedMonster templates don't quite go far enough into the blue,
        #- so chop off some data
        ii, = np.where(wave>3965)
        wave = wave[ii]
        flux = flux[:, ii].astype(float)
        ivar = ivar[:, ii].astype(float)

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

        #- Standardize objtype, converting ELG,LRG -> GALAXY, make upper case
        templatetypes = set()
        if objtype is None:
            templatetypes = set(['GALAXY', 'STAR', 'QSO'])
        else:
            if isinstance(objtype, str):
                objtype = [objtype,]
                
            objtype = [x.upper() for x in objtype]
            for x in objtype:
                if x in ['ELG', 'LRG']:
                    templatetypes.add('GALAXY')
                elif x in ['QSO', 'GALAXY', 'STAR']:
                    templatetypes.add(x)
                else:
                    raise ValueError('Unknown objtype '+x)
            
        #- list of (templatename, zmin, zmax) to fix
        self.template_dir = os.getenv('REDMONSTER_TEMPLATES_DIR')
        self.templates = list()
        for x in templatetypes:
            if x == 'GALAXY':
                self.templates.append(('ndArch-ssp_em_galaxy-v000.fits', zrange_galaxy[0], zrange_galaxy[1], group_galaxy))
            elif x == 'STAR':
                self.templates.append(('ndArch-spEigenStar-55734.fits', zrange_star[0], zrange_star[1], group_star))
            elif x == 'QSO':
                self.templates.append(('ndArch-QSO-V003.fits', zrange_qso[0], zrange_qso[1], group_qso))
            else:
                raise ValueError("Bad template type "+x)

        #- Find and refine best redshift per template
        self.zfinders = list()
        self.zfitters = list()
        
        for template, zmin, zmax, group in self.templates:
            start=time.time()
            zfind = ZFinder(os.path.join(self.template_dir, template), npoly=npoly, zmin=zmin, zmax=zmax, nproc=nproc, group=group)
            zfind.zchi2(self.flux, self.loglam, self.ivar, npixstep=2)
            stop=time.time()
            log.debug("Time to find the redshifts of %d fibers for template %s =%f sec"%(self.flux.shape[0],template,stop-start))
            start=time.time()
            zfit = ZFitter(zfind.zchi2arr, zfind.zbase)
            zfit.z_refine2()
            stop=time.time()
            log.debug("Time to refine the redshift fit of %d fibers for template %s =%f sec"%(zfit.z.shape[0],template,stop-start))
            
            for ifiber in range(zfit.z.shape[0]) :
                log.debug("(after z_refine2) fiber #%d %s chi2s=%s zs=%s"%(ifiber,template,zfit.chi2vals[ifiber],zfit.z[ifiber]))
            
            self.zfinders.append(zfind)
            self.zfitters.append(zfit)

        #- Create wrapper object needed for zpicker
        specobj = _RedMonsterSpecObj(self.wave, self.flux, self.ivar)
        flags = list()
        for i in range(len(self.zfitters)):
            flags.append(self.zfinders[i].zwarning.astype(int) | \
                         self.zfitters[i].zwarning.astype(int))

        #- Zpicker
        self.zpicker = ZPicker(specobj, self.zfinders, self.zfitters, flags)

        #- Fill in outputs
        self.spectype = np.asarray([self.zpicker.type[i][0] for i in range(nspec)])
        self.subtype = np.asarray([json.dumps(self.zpicker.subtype[i][0]) for i in range(nspec)])
        self.z = np.array([self.zpicker.z[i][0] for i in range(nspec)])
        self.zerr = np.array([self.zpicker.z_err[i][0] for i in range(nspec)])
        self.zwarn = np.array([int(self.zpicker.zwarning[i]) for i in range(nspec)])
        self.model = self.zpicker.models[:,0]

        for ifiber in range(self.z.size):
            log.debug("(after zpicker) fiber #%d z=%s"%(ifiber,self.z[ifiber]))


#- This is a container class needed by Redmonster zpicker
class _RedMonsterSpecObj(object):
    def __init__(self, wave, flux, ivar, dof=None):
        """
        Create an object with .wave, .flux, .ivar, and .dof attributes;
        these are needed by RedMonster as input
        """
        nspec, nwave = flux.shape
        self.wave = wave
        self.flux = flux
        self.ivar = ivar
        self.npix = flux.shape[-1]
        if dof is None:
            self.dof = np.ones(nspec) * nwave
        else:
            self.dof = dof

        #- Leftover BOSS-isms
        self.plate = self.mjd = self.fiberid = 0
        self.hdr = None
        self.plugmap = None

