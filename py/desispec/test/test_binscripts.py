
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
from desispec.util import runcmd
import desispec.scripts

class TestBinScripts(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.nspec = 10
        cls.nwave = 20
        id = uuid4().hex
        cls.calibfile = 'calib-'+id+'.fits'
        cls.framefile = 'frame-'+id+'.fits'
        cls.fiberflatfile = 'fiberflat-'+id+'.fits'
        cls.fibermapfile = 'fibermap-'+id+'.fits'
        cls.skyfile = 'sky-'+id+'.fits'
        cls.stdfile = 'std-'+id+'.fits'
        cls.qa_calib_file = 'qa-calib-'+id+'.yaml'
        cls.qa_data_file = 'qa-data-'+id+'.yaml'
        cls.qafig = 'qa-'+id+'.pdf'

        #- when running "python setup.py test", this file is run from different
        #- locations for python 2.7 vs. 3.5
        #- python 2.7: py/specter/test/test_binscripts.py
        #- python 3.5: build/lib/specter/test/test_binscripts.py

        #- python 2.7 location:
        cls.topDir = os.path.dirname( # top-level
            os.path.dirname( # py/
                os.path.dirname( # desispec/
                    os.path.dirname(os.path.abspath(__file__)) # test/
                    )
                )
            )
        cls.binDir = os.path.join(cls.topDir,'bin')
        if not os.path.isdir(cls.binDir):
            #- python 3.x setup.py test location:
            cls.topDir = os.path.dirname( # top-level
                os.path.dirname( # build/
                    os.path.dirname( # lib/
                        os.path.dirname( # desispec/
                            os.path.dirname(os.path.abspath(__file__)) # test/
                            )
                        )
                    )
                )
            cls.binDir = os.path.join(cls.topDir,'bin')

        #- last attempt
        if not os.path.isdir(cls.binDir):
            cls.topDir = os.getcwd()
            cls.binDir = os.path.join(cls.topDir, 'bin')

        if not os.path.isdir(cls.binDir):
            raise RuntimeError('Unable to auto-locate desispec/bin from {}'.format(__file__))

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
            cls.skyfile, cls.calibfile, cls.stdfile, cls.qa_calib_file,
                         cls.qa_data_file, cls.qafig]:
            if os.path.exists(filename):
                os.remove(filename)
        if cls.origPath is None:
            del os.environ['PYTHONPATH']
        else:
            os.environ['PYTHONPATH'] = cls.origPath

    def _write_frame(self, flavor='none', camera='b', expid=1, night='20160607'):
        """Write a fake frame"""
        wave = 5000+np.arange(self.nwave)
        flux = np.ones((self.nspec, self.nwave))
        ivar = np.ones((self.nspec, self.nwave))
        mask = np.zeros((self.nspec, self.nwave), dtype=int)
        Rdata = np.ones((self.nspec, 1, self.nwave))
        fibermap = self._get_fibermap()
        frame = Frame(wave, flux, ivar, mask, Rdata, fibermap=fibermap,
                      meta=dict(FLAVOR=flavor, CAMERA=camera, EXPID=expid, NIGHT=night))
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
        for i in range(0, self.nspec, 5):
            #- Add at least 2 QSOs, ELGs, LRGs and STDs each (needed for SNR plot)
            fibermap['OBJTYPE'][i] = 'SKY'
            fibermap['OBJTYPE'][i+1] = 'STD'
            fibermap['OBJTYPE'][i+2] = 'ELG'
            fibermap['OBJTYPE'][i+3] = 'LRG'
            fibermap['OBJTYPE'][i+4] = 'QSO'

        #- Add mag and filter needed for skysub qas
        fibermap['MAG']=np.tile(np.random.uniform(18,20,self.nspec),5).reshape(self.nspec,5)
        fibermap['FILTER']=np.tile(['DECAM_R','..','..','..','..'],(self.nspec)).reshape(self.nspec,5)
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
        fibers = np.array([1,6]).astype(int)
        hdu1=fits.PrimaryHDU(stdflux)
        hdu1.header['EXTNAME'] = 'FLUX'
        hdu2=fits.ImageHDU(wave, name='WAVELENGTH')
        hdu3=fits.ImageHDU(fibers, name='FIBERS')
        hdulist=fits.HDUList([hdu1,hdu2,hdu3])
        hdulist.writeto(self.stdfile,clobber=True)

    def _remove_files(self, filenames):
        '''Utility to cleanup output files if they exist'''
        for fx in filenames:
            if os.path.exists(fx):
                os.remove(fx)

    def test_compute_fiberflat(self):
        """
        Tests desi_compute_fiberflat --infile frame.fits --outfile fiberflat.fits
        """
        self._write_frame(flavor='flat')
        self._write_fibermap()

        # QA fig requires fibermapfile
        cmd = '{} {}/desi_compute_fiberflat --infile {} --outfile {} --qafile {} --qafig {}'.format(
                sys.executable, self.binDir, self.framefile,
                self.fiberflatfile, self.qa_calib_file, self.qafig)
        outputs = [self.fiberflatfile,self.qa_calib_file,self.qafig]
        inputs = [self.framefile,]
        err = runcmd(cmd, inputs=inputs, outputs=outputs, clobber=True)
        self.assertEqual(err, 0)

        #- Confirm that the output file can be read as a fiberflat
        ff1 = io.read_fiberflat(self.fiberflatfile)
        
        #- Remove outputs and call again via function instead of system call
        self._remove_files(outputs)
        args = desispec.scripts.fiberflat.parse(cmd.split()[2:])        
        err = runcmd(desispec.scripts.fiberflat.main, args=[args,],
            inputs=inputs, outputs=outputs, clobber=True)

        #- Confirm that the output file can be read as a fiberflat
        ff2 = io.read_fiberflat(self.fiberflatfile)
        
        self.assertTrue(np.all(ff1.fiberflat == ff2.fiberflat))
        self.assertTrue(np.all(ff1.ivar == ff2.ivar))
        self.assertTrue(np.all(ff1.mask == ff2.mask))
        self.assertTrue(np.all(ff1.meanspec == ff2.meanspec))
        self.assertTrue(np.all(ff1.wave == ff2.wave))
        self.assertTrue(np.all(ff1.fibers == ff2.fibers))        

    def test_compute_fluxcalib(self):
        """
        Tests desi_compute_sky --infile frame.fits --fiberflat fiberflat.fits --outfile skymodel.fits
        """
        self._write_frame(flavor='dark', camera='b0')
        self._write_fiberflat()
        self._write_fibermap()
        self._write_skymodel()
        self._write_stdstars()

        cmd = "{} {}/desi_compute_fluxcalibration --infile {} --fiberflat {} --sky {} --models {} --outfile {} --qafile {} --qafig {}".format(
            sys.executable, self.binDir, self.framefile, self.fiberflatfile, self.skyfile, self.stdfile,
                self.calibfile, self.qa_data_file, self.qafig)
        inputs  = [self.framefile, self.fiberflatfile, self.skyfile, self.stdfile]
        outputs = [self.calibfile,self.qa_data_file,self.qafig,]
        err = runcmd(cmd, inputs=inputs, outputs=outputs, clobber=True)
        self.assertEqual(err, 0)

        #- Remove outputs and call again via function instead of system call
        self._remove_files(outputs)
        args = desispec.scripts.fluxcalibration.parse(cmd.split()[2:])        
        err = runcmd(desispec.scripts.fluxcalibration.main, args=[args,],
            inputs=inputs, outputs=outputs, clobber=True)
        self.assertEqual(err, None)

    def test_compute_sky(self):
        """
        Tests desi_compute_sky --infile frame.fits --fiberflat fiberflat.fits --outfile skymodel.fits
        """
        self._write_frame(flavor='dark', camera='b0')  # MUST MATCH FLUXCALIB ABOVE
        self._write_fiberflat()
        self._write_fibermap()

        cmd = "{} {}/desi_compute_sky --infile {} --fiberflat {} --outfile {} --qafile {} --qafig {}".format(
            sys.executable, self.binDir, self.framefile, self.fiberflatfile, self.skyfile, self.qa_data_file, self.qafig)
        inputs  = [self.framefile, self.fiberflatfile]
        outputs = [self.skyfile,self.qa_data_file,self.qafig,]
        err = runcmd(cmd, inputs=inputs, outputs=outputs, clobber=True)
        self.assertEqual(err, 0)

        #- Remove outputs and call again via function instead of system call
        self._remove_files(outputs)
        args = desispec.scripts.sky.parse(cmd.split()[2:])        
        err = runcmd(desispec.scripts.sky.main, args=[args,],
            inputs=inputs, outputs=outputs, clobber=True)
        self.assertEqual(err, None)


#- This runs all test* functions in any TestCase class in this file
if __name__ == '__main__':
    unittest.main()
