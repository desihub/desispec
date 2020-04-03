import unittest, os, sys, shutil, tempfile
import numpy as np
from astropy.io import fits

if __name__ == '__main__':
    print('Run this instead:')
    print('python setup.py test -m desispec.test.test_pixgroup')
    sys.exit(1)

from ..test.util import get_frame_data
from ..io import findfile, write_frame, read_spectra, specprod_root
from ..scripts import group_spectra

class TestPixGroup(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.testdir = tempfile.mkdtemp()
        cls.outdir = os.path.join(cls.testdir, 'output')
        
        os.environ['DESI_SPECTRO_REDUX'] = cls.testdir
        os.environ['SPECPROD'] = 'grouptest'

        cls.nspec_per_frame = 3
        cls.nframe_per_night = 2
        cls.nights = [20200101, 20200102, 20200103]

        frame = get_frame_data(nspec=cls.nspec_per_frame)
        frame.meta['FLAVOR'] = 'science'
        
        scores = dict()
        for camera in ['b', 'r', 'z']:
            X = camera.upper()
            dtype = [('COUNTS_'+X, int), ('FLUX_'+X, float)]
            scores[camera] = np.zeros(cls.nspec_per_frame, dtype=dtype)
            # scores[camera] = None
        
        expid = 1
        for night in cls.nights:
            frame.meta['NIGHT'] = night
            frame.meta['EXPID'] = expid
            frame.meta['TILEID'] = expid*10
            frame.meta['MJD-OBS'] = 55555.0 + 0.1*expid
            for camera in ('b0', 'r0', 'z0'):
                frame.meta['CAMERA'] = camera
                frame.scores = scores[camera[0]]
                write_frame(findfile('cframe', night, expid, camera), frame)

            expid += 1
            frame.meta['NIGHT'] = night
            frame.meta['EXPID'] = expid
            frame.meta['TILEID'] = expid*10
            frame.meta['MJD-OBS'] = 55555.0 + 0.1*expid
            for camera in ('b0', 'r0', 'z0'):
                frame.meta['CAMERA'] = camera
                frame.scores = scores[camera[0]]
                write_frame(findfile('cframe', night, expid, camera), frame)

        #- Remove one file to test missing data
        os.remove(findfile('cframe', cls.nights[0], 1, 'r0'))

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.testdir):
            shutil.rmtree(cls.testdir)

    def setUp(self):
        os.environ['DESI_SPECTRO_REDUX'] = self.testdir
        os.environ['SPECPROD'] = 'grouptest'
        os.makedirs(self.outdir)

    def tearDown(self):
        if os.path.exists(self.outdir):
            shutil.rmtree(self.outdir)

    def test_regroup_per_night(self):
        #- Run for each night and confirm that spectra file is correct size
        for i, night in enumerate(self.nights):
            cmd = 'desi_group_spectra -o {} --nights {}'.format(self.outdir, night)
            args = group_spectra.parse(cmd.split()[1:])
            group_spectra.main(args)

            specfile = os.path.join(self.outdir, 'spectra-64-19456.fits')
            spectra = read_spectra(specfile)
            num_nights = i+1
            nspec = self.nspec_per_frame * self.nframe_per_night * num_nights
        
            self.assertEqual(len(spectra.fibermap), nspec)
            self.assertEqual(spectra.flux['b'].shape[0], nspec)

    def test_regroup_nights(self):
        #- run on a specific set of nights
        num_nights = 2
        nights = ','.join([str(tmp) for tmp in self.nights[0:num_nights]])
        cmd = 'desi_group_spectra -o {} --nights {}'.format(self.outdir, nights)
        args = group_spectra.parse(cmd.split()[1:])
        group_spectra.main(args)

        specfile = os.path.join(self.outdir, 'spectra-64-19456.fits')
        spectra = read_spectra(specfile)
        nspec = self.nspec_per_frame * self.nframe_per_night * num_nights
    
        self.assertEqual(len(spectra.fibermap), nspec)
        self.assertEqual(spectra.flux['b'].shape[0], nspec)

    def test_regroup(self):
        #- self discover what nights to combine
        cmd = 'desi_group_spectra -o {}'.format(self.outdir)
        args = group_spectra.parse(cmd.split()[1:])
        group_spectra.main(args)

        specfile = os.path.join(self.outdir, 'spectra-64-19456.fits')
        spectra = read_spectra(specfile)
        num_nights = len(self.nights)
        nspec = self.nspec_per_frame * self.nframe_per_night * num_nights
    
        self.assertEqual(len(spectra.fibermap), nspec)
        self.assertEqual(spectra.flux['b'].shape[0], nspec)

        #- confirm that we can read the mask with memmap=True
        with fits.open(specfile, memmap=True) as fx:
            mask = fx['B_MASK'].data

    def test_reduxdir(self):
        #- Test using a non-standard redux directory
        reduxdir = specprod_root()
        cmd = 'desi_group_spectra -o {} --reduxdir {}'.format(
                self.outdir, reduxdir)

        #- Change SPECPROD and confirm that default location changed
        os.environ['SPECPROD'] = 'blatfoo'
        self.assertNotEqual(reduxdir, specprod_root())

        args = group_spectra.parse(cmd.split()[1:])
        group_spectra.main(args)

        specfile = os.path.join(self.outdir, 'spectra-64-19456.fits')
        spectra = read_spectra(specfile)
        num_nights = len(self.nights)
        nspec = self.nspec_per_frame * self.nframe_per_night * num_nights

        self.assertEqual(len(spectra.fibermap), nspec)
        self.assertEqual(spectra.flux['b'].shape[0], nspec)

def test_suite():
    """Allows testing of only this module with the command::

        python setup.py test -m desispec.test.test_pixgroup
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
