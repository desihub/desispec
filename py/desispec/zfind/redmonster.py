"""
desispec.zfind.redmonster
=========================

Classes for use with the redmonster package.
"""
from __future__ import division, absolute_import

import os

import numpy as np

from desispec.zfind import ZfindBase
from desispec.interpolation import resample_flux
from desispec.log import get_logger

class RedMonsterZfind(ZfindBase):
    """Class documentation goes here.
    """
    def __init__(self, wave, flux, ivar, R=None, dloglam=1e-4):
        """Uses Redmonster to classify and find redshifts.

        See :class:`desispec.zfind.zfind.ZfindBase` class for inputs/outputs.

        TODO: document redmonster specific output variables
        """
        try:
            from redmonster.physics.zfinder import Zfinder
            from redmonster.physics.zfitter import Zfitter
            from redmonster.physics.zpicker2 import Zpicker
        except ImportError:
            get_logger().error("You are attempting to use RedMonster, but it is not available for import!")
            raise
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
        self.template_dir = os.getenv('REDMONSTER')+'/templates/'
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
            zfit.z_refine2()

            self.zfinders.append(zfind)
            self.zfitters.append(zfit)

        #- Create wrapper object needed for zpicker
        specobj = _RedMonsterSpecObj(self.wave, self.flux, self.ivar)
        flags = list()
        for i in range(len(self.zfitters)):
            flags.append(self.zfinders[i].zwarning.astype(int) | \
                         self.zfitters[i].zwarning.astype(int))

        #- Zpicker
        self.zpicker = Zpicker(specobj, self.zfinders, self.zfitters, flags)

        #- Fill in outputs
        self.type = np.asarray(self.zpicker.type)
        self.subtype = np.asarray(self.zpicker.subtype)
        self.z = np.array([self.zpicker.z[i][0] for i in range(nspec)])
        self.zerr = np.array([self.zpicker.z_err[i][0] for i in range(nspec)])
        self.zwarn = np.array([self.zpicker.zwarning[i].astype(int) for i in range(nspec)])
        self.model = self.zpicker.models[:,0]


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
        if dof is None:
            self.dof = np.ones(nspec) * nwave
        else:
            self.dof = dof

        #- Leftover BOSS-isms
        self.plate = self.mjd = self.fiberid = self.npix = 0
        self.hdr = None
        self.plugmap = None

