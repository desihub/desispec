
import os, sys
import unittest
from uuid import uuid4

import numpy as np

from desispec.resolution import Resolution
from desispec.frame import Frame
from desispec.fiberflat import FiberFlat
from desispec import io
from desispec.pipeline.core import runcmd

class TestBinScripts(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.nspec = 5
        cls.nwave = 20
        id = uuid4().hex
        cls.framefile = 'frame-'+id+'.fits'
        cls.fiberflatfile = 'fiberflat-'+id+'.fits'
        cls.fibermapfile = 'fibermap-'+id+'.fits'
        cls.skyfile = 'sky-'+id+'.fits'
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
        for filename in [cls.framefile, cls.fiberflatfile, cls.fibermapfile, cls.skyfile]:
            if os.path.exists(filename):
                os.remove(filename)
        if cls.origPath is None:
            del os.environ['PYTHONPATH']
        else:
            os.environ['PYTHONPATH'] = cls.origPath

    def _write_frame(self):
        """Write a fake frame"""
        wave = 5000+np.arange(self.nwave)
        flux = np.ones((self.nspec, self.nwave))
        ivar = np.ones((self.nspec, self.nwave))
        mask = np.zeros((self.nspec, self.nwave), dtype=int)
        Rdata = np.ones((self.nspec, 1, self.nwave))
        frame = Frame(wave, flux, ivar, mask, Rdata, spectrograph=0)
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

    def _write_fibermap(self):
        """Write a fake fiberflat"""
        fibermap = io.empty_fibermap(self.nspec)
        for i in range(0, self.nspec, 3):
            fibermap['OBJTYPE'][i] = 'SKY'

        io.write_fibermap(self.fibermapfile, fibermap)

    def _write_skymodel(self):
        pass

    def _write_stdstars(self):
        pass

    def test_compute_fiberflat(self):
        """
        Tests desi_compute_fiberflat.py --infile frame.fits --outfile fiberflat.fits
        """
        self._write_frame()
        #- run the command and confirm error code = 0
        cmd = '{} {}/desi_compute_fiberflat.py --infile {} --outfile {}'.format(
            sys.executable, self.binDir, self.framefile, self.fiberflatfile)
        # self.assertTrue(os.path.exists(os.path.join(self.binDir,'desi_compute_fiberflat.py')))
        err = runcmd(cmd, [self.framefile,], [self.fiberflatfile,], clobber=True)
        self.assertEqual(err, 0)

        #- Confirm that the output file can be read as a fiberflat
        ff = io.read_fiberflat(self.fiberflatfile)

    def test_compute_sky(self):
        """
        Tests desi_compute_sky.py --infile frame.fits --fibermap fibermap.fits --fiberflat fiberflat.fits --outfile skymodel.fits
        """
        self._write_frame()
        self._write_fiberflat()
        self._write_fibermap()

        cmd = "{} {}/desi_compute_sky.py --infile {} --fibermap {} --fiberflat {} --outfile {}".format(
            sys.executable, self.binDir, self.framefile, self.fibermapfile, self.fiberflatfile, self.skyfile)
        err = runcmd(cmd,
                inputs  = [self.framefile, self.fiberflatfile, self.fibermapfile],
                outputs = [self.skyfile,], clobber=True )
        self.assertEqual(err, 0)


#- This runs all test* functions in any TestCase class in this file
if __name__ == '__main__':
    unittest.main()
