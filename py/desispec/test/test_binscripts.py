
import os, sys
import unittest
from uuid import uuid4

import numpy as np

from astropy.io import fits

from desispec.resolution import Resolution
from desispec.frame import Frame
from desispec.fiberflat import FiberFlat
from desispec.sky import SkyModel
from desispec import io
from desispec.pipeline.core import runcmd

class TestBinScripts(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.nspec = 6
        cls.nwave = 20
        id = uuid4().hex
        cls.calibfile = 'calib-'+id+'.fits'
        cls.framefile = 'frame-'+id+'.fits'
        cls.fiberflatfile = 'fiberflat-'+id+'.fits'
        cls.fibermapfile = 'fibermap-'+id+'.fits'
        cls.skyfile = 'sky-'+id+'.fits'
        cls.stdfile = 'std-'+id+'.fits'
        cls.qafile = 'qa-'+id+'.yaml'
        cls.qafig = 'qa-'+id+'.pdf'
        cls.topDir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        cls.binDir = os.path.join(cls.topDir,'bin')
        try:
            cls.origPath = os.environ['PYTHONPATH']
            os.environ['PYTHONPATH'] = os.path.join(cls.topDir,'py') + ':' + cls.origPath
        except KeyError:
            cls.origPath = None
            os.environ['PYTHONPATH'] = os.path.join(cls.topDir,'py')

    @classmethod
    def tearDownClass(cls):
        """Cleanup in case tests crashed and left files behind"""
        for filename in [cls.framefile, cls.fiberflatfile, cls.fibermapfile, \
            cls.skyfile, cls.calibfile, cls.stdfile, cls.qafile, cls.qafig]:
            if os.path.exists(filename):
                os.remove(filename)
        if cls.origPath is None:
            del os.environ['PYTHONPATH']
        else:
            os.environ['PYTHONPATH'] = cls.origPath

    def _write_frame(self, flavor='none', camera='b'):
        """Write a fake frame"""
        wave = 5000+np.arange(self.nwave)
        flux = np.ones((self.nspec, self.nwave))
        ivar = np.ones((self.nspec, self.nwave))
        mask = np.zeros((self.nspec, self.nwave), dtype=int)
        Rdata = np.ones((self.nspec, 1, self.nwave))
        fibermap = self._get_fibermap()
        frame = Frame(wave, flux, ivar, mask, Rdata, fibermap=fibermap,
                      meta=dict(flavor=flavor, camera=camera))
        io.write_frame(self.framefile, frame)

    def _write_fiberflat(self):
        """Write a fake fiberflat"""
        wave = 5000+np.arange(self.nwave)
        fiberflat = np.ones((self.nspec, self.nwave))
        ivar = np.ones((self.nspec, self.nwave))
        mask = np.zeros((self.nspec, self.nwave), dtype=int)
        meanspec = np.ones(self.nwave)
        ff = FiberFlat(wave, fiberflat, ivar, mask, meanspec)
        io.write_fiberflat(self.fiberflatfile, ff)

    def _get_fibermap(self):
        fibermap = io.empty_fibermap(self.nspec, 1500)
        for i in range(0, self.nspec, 3):
            fibermap['OBJTYPE'][i] = 'SKY'
            fibermap['OBJTYPE'][i+1] = 'STD'
        return fibermap

    def _write_fibermap(self):
        """Write a fake fibermap"""
        fibermap = self._get_fibermap()
        io.write_fibermap(self.fibermapfile, fibermap)

    def _write_skymodel(self):
        """Write a fake SkyModel"""
        wave = 5000+np.arange(self.nwave)
        skyflux = np.ones((self.nspec, self.nwave))*0.1  # Must be less 1
        ivar = np.ones((self.nspec, self.nwave))
        mask = np.zeros((self.nspec, self.nwave), dtype=int)
        sky = SkyModel(wave, skyflux, ivar, mask, nrej=1)
        io.write_sky(self.skyfile, sky)

    def _write_stdstars(self):
        """Write a fake StdStar model file"""
        # First generation is very simple
        wave = 5000+np.arange(self.nwave)
        stdflux = np.ones((self.nspec, self.nwave))
        fibers = np.array([1,4]).astype(int)
        hdu1=fits.PrimaryHDU(stdflux)
        hdu2=fits.ImageHDU(wave)
        hdu3=fits.ImageHDU(fibers)
        hdulist=fits.HDUList([hdu1,hdu2,hdu3])
        hdulist.writeto(self.stdfile,clobber=True)

    def test_compute_fiberflat(self):
        """
        Tests desi_compute_fiberflat --infile frame.fits --outfile fiberflat.fits
        """
        self._write_frame(flavor='flat')
        self._write_fibermap()

        # QA fig requires fibermapfile
        cmd = '{} {}/desi_compute_fiberflat --infile {} --fibermap {} --outfile {} --qafile {} --qafig {}'.format(
                sys.executable, self.binDir, self.framefile, self.fibermapfile,
                self.fiberflatfile, self.qafile, self.qafig)
        err = runcmd(cmd,
                     inputs = [self.framefile, self.fibermapfile],
                     outputs = [self.fiberflatfile,self.qafile,self.qafig], clobber=True)
        self.assertEqual(err, 0)

        #- Confirm that the output file can be read as a fiberflat
        ff = io.read_fiberflat(self.fiberflatfile)

    def test_compute_fluxcalib(self):
        """
        Tests desi_compute_sky --infile frame.fits --fibermap fibermap.fits --fiberflat fiberflat.fits --outfile skymodel.fits
        """
        self._write_frame(flavor='dark', camera='b')
        self._write_fiberflat()
        self._write_fibermap()
        self._write_skymodel()
        self._write_stdstars()

        cmd = "{} {}/desi_compute_fluxcalibration --infile {} --fibermap {} --fiberflat {} --sky {} --models {} --outfile {} --qafile {} --qafig {}".format(
            sys.executable, self.binDir, self.framefile, self.fibermapfile, self.fiberflatfile, self.skyfile, self.stdfile,
                self.calibfile, self.qafile, self.qafig)
        err = runcmd(cmd,
                inputs  = [self.framefile, self.fiberflatfile, self.fibermapfile, self.skyfile, self.stdfile],
                outputs = [self.calibfile,self.qafile,self.qafig,], clobber=True )
        self.assertEqual(err, 0)

    def test_compute_sky(self):
        """
        Tests desi_compute_sky --infile frame.fits --fibermap fibermap.fits --fiberflat fiberflat.fits --outfile skymodel.fits
        """
        self._write_frame(flavor='dark')
        self._write_fiberflat()
        self._write_fibermap()

        cmd = "{} {}/desi_compute_sky --infile {} --fibermap {} --fiberflat {} --outfile {} --qafile {} --qafig {}".format(
            sys.executable, self.binDir, self.framefile, self.fibermapfile, self.fiberflatfile, self.skyfile, self.qafile, self.qafig)
        err = runcmd(cmd,
                inputs  = [self.framefile, self.fiberflatfile, self.fibermapfile],
                outputs = [self.skyfile,self.qafile,self.qafig,], clobber=True )
        self.assertEqual(err, 0)


#- This runs all test* functions in any TestCase class in this file
if __name__ == '__main__':
    unittest.main()
