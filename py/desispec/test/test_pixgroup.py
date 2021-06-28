import unittest, os, sys, shutil, tempfile
import numpy as np
import numpy.testing as nt
from astropy.io import fits
from copy import deepcopy

if __name__ == '__main__':
    print('Run this instead:')
    print('python setup.py test -m desispec.test.test_pixgroup')
    sys.exit(1)

from ..test.util import get_frame_data
from ..io import findfile, write_frame, read_spectra, write_spectra, empty_fibermap, specprod_root
from ..io.util import add_columns
from ..scripts import group_spectra
from ..pixgroup import SpectraLite
from desispec.maskbits import fibermask
from desiutil.io import encode_table

class TestPixGroup(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.testdir = tempfile.mkdtemp()
        cls.outdir = os.path.join(cls.testdir, 'output')
        
        os.environ['DESI_SPECTRO_REDUX'] = cls.testdir
        os.environ['SPECPROD'] = 'grouptest'

        cls.nspec_per_frame = 8 # needs to be at least 4
        cls.nframe_per_night = 4 # needs to be at least 4
        cls.nights = [20200101, 20200102, 20200103] # needs to be at least 3 nights
        cls.badnight = cls.nights[-1]
        cls.badslice = np.arange(2, int(np.min([6,cls.nspec_per_frame]) ) ).astype(int)
        
        frames = dict()
        meta = {'EXPID': 1.0, 'FLAVOR': 'science'}
        frames['b'] = get_frame_data(nspec=cls.nspec_per_frame, wavemin=4500, wavemax=4600,
                                 nwave=100, meta=meta.copy())
        frames['r'] = get_frame_data(nspec=cls.nspec_per_frame, wavemin=6500, wavemax=6600,
                                 nwave=100, meta=meta.copy())
        frames['z'] = get_frame_data(nspec=cls.nspec_per_frame, wavemin=8500, wavemax=8600,
                                 nwave=100, meta=meta.copy())

        bad_nframs_on_night = {'b':  0 , 'r':  1 , 'z':  2 }
        bad_brz_nfram = 3
        badbits = { 'b': fibermask.BADAMPB,\
                    'r': fibermask.BADAMPR,\
                    'z': fibermask.BADAMPZ    }
        badamp_brz = ( fibermask.BADAMPB & fibermask.BADAMPR & fibermask.BADAMPZ )
        for camera in ['b0', 'r0', 'z0']:
            X = camera[0].upper()
            dtype = [('COUNTS_'+X, int), ('FLUX_'+X, float)]
            bad_nfram_on_night = bad_nframs_on_night[camera[0]]
            badbit = badbits[camera[0]]
            frame = frames[camera[0]]
            frame.meta['CAMERA'] = camera
            frame.scores = np.zeros(cls.nspec_per_frame, dtype=dtype)
            for ii,night in enumerate(cls.nights):
                for nfram in range(cls.nframe_per_night):
                    expid = (ii*cls.nframe_per_night) + (nfram + 1)
                    
                    if night == cls.badnight and \
                       (nfram == bad_nfram_on_night or nfram == bad_brz_nfram):
                        curframe = deepcopy(frame)
                        curframe.fibermap['FIBERSTATUS'][cls.badslice] |= badbit
                    else:
                        curframe = frame
                    
                    curframe.meta['NIGHT'] = night
                    curframe.meta['EXPID'] = expid
                    curframe.meta['TILEID'] = expid*10
                    curframe.meta['MJD-OBS'] = 55555.0+ii + 0.1*nfram
                    write_frame(findfile('cframe', night, expid, camera), curframe)
                    
        #- Remove one file to test missing data
        os.remove(findfile('cframe', cls.nights[0], 1, 'r0'))

        # Setup a dummy SpectraLite for I/O tests
        cls.fileio = 'test_spectralite.fits'

        cls.nwave = 100
        cls.nspec = 5
        cls.ndiag = 3

        fmap = empty_fibermap(cls.nspec)
        fmap = add_columns(fmap,
                           ['NIGHT', 'EXPID', 'TILEID'],
                           [np.int32(0), np.int32(0), np.int32(0)],
                           )

        for s in range(cls.nspec):
            fmap[s]["TARGETID"] = 456 + s
            fmap[s]["FIBER"] = 123 + s
            fmap[s]["NIGHT"] = s
            fmap[s]["EXPID"] = s
        cls.fmap1 = encode_table(fmap)

        fmap = empty_fibermap(cls.nspec)
        fmap = add_columns(fmap,
                           ['NIGHT', 'EXPID', 'TILEID'],
                           [np.int32(0), np.int32(0), np.int32(0)],
                           )

        for s in range(cls.nspec):
            fmap[s]["TARGETID"] = 789 + s
            fmap[s]["FIBER"] = 200 + s
            fmap[s]["NIGHT"] = 1000
            fmap[s]["EXPID"] = 1000+s
        cls.fmap2 = encode_table(fmap)

        for s in range(cls.nspec):
            fmap[s]["TARGETID"] = 1234 + s
            fmap[s]["FIBER"] = 300 + s
            fmap[s]["NIGHT"] = 2000
            fmap[s]["EXPID"] = 2000+s
        cls.fmap3 = encode_table(fmap)

        cls.bands = ["b", "r", "z"]

        cls.wave = {}
        cls.flux = {}
        cls.ivar = {}
        cls.mask = {}
        cls.res = {}

        for s in range(cls.nspec):
            for b in cls.bands:
                cls.wave[b] = np.arange(cls.nwave)
                cls.flux[b] = np.repeat(np.arange(cls.nspec),
                    cls.nwave).reshape( (cls.nspec, cls.nwave) ) + 3.0
                cls.ivar[b] = 1.0 / cls.flux[b]
                cls.mask[b] = np.tile(np.arange(2, dtype=np.uint32),
                    (cls.nwave * cls.nspec) // 2).reshape( (cls.nspec, cls.nwave) )
                cls.res[b] = np.zeros( (cls.nspec, cls.ndiag, cls.nwave),
                    dtype=np.float64)
                cls.res[b][:,1,:] = 1.0

        # In SpectraLite, scores apply to all bands.
        cls.scores = np.arange(cls.nspec)

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
        if os.path.exists(self.fileio):
            os.remove(self.fileio)

    def verify_spectralite(self, spec, fmap):
        nt.assert_array_equal(spec.fibermap, fmap)
        for band in self.bands:
            nt.assert_array_almost_equal(spec.wave[band], self.wave[band])
            nt.assert_array_almost_equal(spec.flux[band], self.flux[band])
            nt.assert_array_almost_equal(spec.ivar[band], self.ivar[band])
            nt.assert_array_equal(spec.mask[band], self.mask[band])
            nt.assert_array_almost_equal(spec.resolution_data[band], self.res[band])
        if spec.scores is not None:
            nt.assert_array_equal(spec.scores, self.scores)

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

    def test_regroup_fiberstatus_propagation(self):
        #- run on a specific set of nights
        cmd = 'desi_group_spectra -o {} --nights {}'.format(self.outdir, self.badnight)
        args = group_spectra.parse(cmd.split()[1:])
        group_spectra.main(args)

        specfile = os.path.join(self.outdir, 'spectra-64-19456.fits')
        spectra = read_spectra(specfile)
        nspec = self.nspec_per_frame * self.nframe_per_night

        fibermap_array = np.array(spectra.fibermap['FIBERSTATUS'])
        self.assertEqual(len(fibermap_array), nspec)
        
        self.assertEqual(np.sum( (fibermap_array & fibermask.BADAMPB) > 0 ), 2*len(self.badslice))
        self.assertEqual(np.sum( (fibermap_array & fibermask.BADAMPR) > 0 ), 2*len(self.badslice))
        self.assertEqual(np.sum( (fibermap_array & fibermask.BADAMPZ) > 0 ), 2*len(self.badslice))

        badamp_brz = ( fibermask.BADAMPB | fibermask.BADAMPR | fibermask.BADAMPZ )
        self.assertEqual(np.sum( (fibermap_array == badamp_brz) ), len(self.badslice))
        
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

    def test_spectraliteio(self):
        # manually create the spectra and write
        spec = SpectraLite(bands=self.bands, wave=self.wave, flux=self.flux,
            ivar=self.ivar, mask=self.mask, resolution_data=self.res,
            fibermap=self.fmap1)

        self.verify_spectralite(spec, self.fmap1)

        path = write_spectra(self.fileio, spec)
        assert(path == os.path.abspath(self.fileio))

        # read back in and verify
        comp = read_spectra(self.fileio)
        self.verify_spectralite(comp, self.fmap1)

def test_suite():
    """Allows testing of only this module with the command::

        python setup.py test -m desispec.test.test_pixgroup
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
