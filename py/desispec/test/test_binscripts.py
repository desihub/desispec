
import os, sys
import unittest
from uuid import uuid4

import numpy as np

from astropy.io import fits
from astropy.table import Table

from desitarget.targetmask import desi_mask
from desispec.resolution import Resolution
from desispec.frame import Frame
from desispec.fiberflat import FiberFlat
from desispec.sky import SkyModel
from desispec import io
from desispec.util import runcmd
import desispec.scripts
import desispec.scripts.sky
import desispec.scripts.fiberflat
import desispec.scripts.stdstars
import desispec.scripts.fluxcalibration
from desispec.test.util import get_frame_data, get_models

class TestBinScripts(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.nspec = 6
        cls.nwave = 2000  # Needed for QA
        cls.wave = 4000+np.arange(cls.nwave)
        id = uuid4().hex
        cls.calibfile = 'calib-'+id+'.fits'
        cls.framefile = 'frame-'+id+'.fits'
        cls.fiberflatfile = 'fiberflat-'+id+'.fits'
        cls.fibermapfile = 'fibermap-'+id+'.fits'
        cls.modelfile ='stdstar_templates-'+id+'.fits'
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
                         cls.qa_data_file, cls.modelfile, cls.qafig]:
            if os.path.exists(filename):
                os.remove(filename)
        if cls.origPath is None:
            del os.environ['PYTHONPATH']
        else:
            os.environ['PYTHONPATH'] = cls.origPath

    def _write_frame(self, flavor='none', camera='b', expid=1, night='20160607',gaia_only=False):
        """Write a fake frame"""
        flux = np.ones((self.nspec, self.nwave))
        ivar = np.ones((self.nspec, self.nwave))*100 # S/N=10
        mask = np.zeros((self.nspec, self.nwave), dtype=int)
        Rdata = np.ones((self.nspec, 1, self.nwave))
        fibermap = self._get_fibermap(gaia_only=gaia_only)
        frame = Frame(self.wave, flux, ivar, mask, Rdata, fibermap=fibermap,
                      meta=dict(FLAVOR=flavor, CAMERA=camera, EXPID=expid, NIGHT=night, EXPTIME=1000., DETECTOR='SIM'))
        io.write_frame(self.framefile, frame)

    def _write_models(self):
        wav, mods = get_models(wavemin=2900,wavemax=11000)
        nmod = len(mods)
        tid,logg,feh,teff=[np.arange(nmod) for _ in range(4)]
        tab = Table({'TEMPLATEID':tid,
                          'LOGG':logg,
                          'FEH':feh,
                          'TEFF':teff})
        fits.HDUList([fits.PrimaryHDU(mods),
                      fits.BinTableHDU(tab),
                      fits.ImageHDU(wav)
                      ]).writeto(self.modelfile, overwrite=True)
        
    def _write_fiberflat(self, camera=None):
        """Write a fake fiberflat"""
        fiberflat = np.ones((self.nspec, self.nwave))
        ivar = np.ones((self.nspec, self.nwave))
        mask = np.zeros((self.nspec, self.nwave), dtype=int)
        meanspec = np.ones(self.nwave)
        ff = FiberFlat(self.wave, fiberflat, ivar, mask, meanspec)
        if camera is not None:
            hdr=fits.Header()
            hdr['CAMERA']=camera
        else:
            hdr=None
        io.write_fiberflat(self.fiberflatfile, ff, hdr)

    def _get_fibermap(self, gaia_only=False):
        fibermap = io.empty_fibermap(self.nspec, 1500)
        for i in range(0, self.nspec, 3):
            fibermap['OBJTYPE'][i] = 'SKY'
            fibermap['DESI_TARGET'][i] = desi_mask.SKY
            fibermap['OBJTYPE'][i+1] = 'TGT'
            fibermap['DESI_TARGET'][i+1] = desi_mask.STD_FAINT
            fibermap['GAIA_PHOT_G_MEAN_MAG'][i+1] = 15
            fibermap['GAIA_PHOT_BP_MEAN_MAG'][i+1] = 15
            fibermap['GAIA_PHOT_RP_MEAN_MAG'][i+1] = 15
            if gaia_only:
                fibermap['PHOTSYS'] = 'G'
            else:
                fibermap['FLUX_G'][i+1] = 100
                fibermap['FLUX_R'][i+1] = 100
                fibermap['FLUX_Z'][i+1] = 100
        return fibermap

    def _write_fibermap(self):
        """Write a fake fibermap"""
        fibermap = self._get_fibermap()
        io.write_fibermap(self.fibermapfile, fibermap)

    def _write_skymodel(self, camera=None):
        """Write a fake SkyModel"""
        skyflux = np.ones((self.nspec, self.nwave))*0.1  # Must be less 1
        ivar = np.ones((self.nspec, self.nwave))
        mask = np.zeros((self.nspec, self.nwave), dtype=int)
        sky = SkyModel(self.wave, skyflux, ivar, mask, nrej=1)
        if camera is not None:
            hdr=fits.Header()
            hdr['CAMERA']=camera
        else:
            hdr=None
        io.write_sky(self.skyfile, sky, hdr)

    def _write_stdstars(self):
        """Write a fake StdStar model file"""
        # First generation is very simple
        stdflux = np.ones((2, self.nwave))
        fibers = np.array([1,4]).astype(int)
        data={}
        data['LOGG']=4.*np.ones(fibers.size)
        data['TEFF']=6000.*np.ones(fibers.size)
        data['FEH']=np.zeros(fibers.size)
        data['COEF']=np.zeros((fibers.size,12))
        # cannot be exactly the same values
        data['CHI2DOF']=np.ones(fibers.size)+0.1*(fibers%2)
        data['REDSHIFT']=np.zeros(fibers.size)
        data['DATA_G-R']=0.3*np.ones(fibers.size)
        data['MODEL_G-R']=0.3*np.ones(fibers.size)
        # dummy placeholders, not used downstream
        tmp = np.arange(fibers.size)
        fibermap = Table()
        fibermap['TARGETID'] = tmp
        inframes = Table()
        inframes['NIGHT'] = tmp
        inframes['EXPID'] = tmp
        inframes['CAMERA'] = 'b0'
        io.write_stdstar_models(self.stdfile,stdflux,self.wave,fibers,data,fibermap,inframes)

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
        self.assertEqual(err, 0, 'FAILED: {}'.format(cmd))

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

    def test_fit_stdstars(self):
        """
        Tests desi_fit_stdstars --infile frame.fits --fiberflat fiberflat.fits --outfile skymodel.fits
for legacy standards
        """
        self._write_frame(flavor='science', camera='b0')
        self._write_fiberflat(camera='b0')
        self._write_skymodel(camera='b0')
        self._write_models()
        for opt in ['','--color=R-Z', '--std-targetids 0 1 2 3 4 5']:
            cmd = "{} {}/desi_fit_stdstars {} --delta-color 1000 --frames {} --skymodels {}  --fiberflats {} --starmodels {} --outfile {}".format(
                sys.executable, self.binDir, opt, self.framefile, self.skyfile, self.fiberflatfile, self.modelfile, self.stdfile)
            inputs  = [self.framefile, self.fiberflatfile, self.skyfile, self.modelfile]
            outputs  = [ self.stdfile]
            err = runcmd(cmd, inputs=inputs, outputs=outputs, clobber=True)

            self.assertEqual(err, 0, 'FAILED: {}'.format(cmd))

            #- Remove outputs and call again via function instead of system call
            self._remove_files(outputs) 
            args = desispec.scripts.stdstars.parse(cmd.split()[2:])        
            err = runcmd(desispec.scripts.stdstars.main, args=[args,],
                inputs=inputs, outputs=outputs, clobber=True)
            self.assertEqual(err, None)

    def test_fit_stdstars_gaia(self):
        """
        Tests desi_fit_stdstars --infile frame.fits --fiberflat fiberflat.fits --outfile skymodel.fits
        for gaia standards
        """
        self._write_frame(flavor='science', camera='b0', gaia_only=True)
        self._write_fiberflat(camera='b0')
        self._write_skymodel(camera='b0')
        self._write_models()
        for opt in ['', '--color=GAIA-BP-RP']:
            cmd = "{} {}/desi_fit_stdstars {} --delta-color 1000 --frames {} --skymodels {}  --fiberflats {} --starmodels {} --outfile {}".format(
                sys.executable, self.binDir, opt, self.framefile, self.skyfile, self.fiberflatfile, self.modelfile, self.stdfile)
            inputs  = [self.framefile, self.fiberflatfile, self.skyfile, self.modelfile]
            outputs  = [ self.stdfile]
            err = runcmd(cmd, inputs=inputs, outputs=outputs, clobber=True)

            self.assertEqual(err, 0, 'FAILED: {}'.format(cmd))

            #- Remove outputs and call again via function instead of system call
            self._remove_files(outputs) 
            args = desispec.scripts.stdstars.parse(cmd.split()[2:])        
            err = runcmd(desispec.scripts.stdstars.main, args=[args,],
                inputs=inputs, outputs=outputs, clobber=True)
            self.assertEqual(err, None)

        
    def test_compute_fluxcalib(self):
        """
        Tests desi_compute_fluxcalibration
        """

        if 'DESI_SPECTRO_CALIB' not in os.environ :
            print("do not test desi_compute_fluxcalib without DESI_SPECTRO_CALIB set")
            return

        self._write_frame(flavor='science', camera='b0')
        self._write_fiberflat()
        self._write_fibermap()
        self._write_skymodel()
        self._write_stdstars()

        cmd = "{} {}/desi_compute_fluxcalibration --infile {} --fiberflat {} --sky {} --models {} --outfile {} --qafile {} --qafig {} --min-color 0.".format(
            sys.executable, self.binDir, self.framefile, self.fiberflatfile, self.skyfile, self.stdfile,
                self.calibfile, self.qa_data_file, self.qafig)
        inputs  = [self.framefile, self.fiberflatfile, self.skyfile, self.stdfile]
        outputs = [self.calibfile,self.qa_data_file,self.qafig,]
        err = runcmd(cmd, inputs=inputs, outputs=outputs, clobber=True)

        self.assertEqual(err, 0, 'FAILED: {}'.format(cmd))

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
        self._write_frame(flavor='science', camera='b0')  # MUST MATCH FLUXCALIB ABOVE
        self._write_fiberflat()
        self._write_fibermap()

        cmd = "{} {}/desi_compute_sky --infile {} --fiberflat {} --outfile {} --qafile {} --qafig {}".format(
            sys.executable, self.binDir, self.framefile, self.fiberflatfile, self.skyfile, self.qa_data_file, self.qafig)
        inputs  = [self.framefile, self.fiberflatfile]
        outputs = [self.skyfile,self.qa_data_file,self.qafig,]
        err = runcmd(cmd, inputs=inputs, outputs=outputs, clobber=True)
        self.assertEqual(err, 0, 'FAILED: {}'.format(cmd))

        #- Remove outputs and call again via function instead of system call
        self._remove_files(outputs)
        args = desispec.scripts.sky.parse(cmd.split()[2:])
        err = runcmd(desispec.scripts.sky.main, args=[args,],
            inputs=inputs, outputs=outputs, clobber=True)
        self.assertEqual(err, None)

def test_suite():
    """Allows testing of only this module with the command::

        python setup.py test -m <modulename>
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)


#- This runs all test* functions in any TestCase class in this file
if __name__ == '__main__':
    unittest.main()
