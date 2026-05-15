import unittest, os, sys, shutil, tempfile
from importlib import resources
import numpy as np
import numpy.testing as nt
from astropy.io import fits
from astropy.table import Table
from copy import deepcopy
from ..test.util import get_frame_data
from ..io import findfile, write_frame, read_spectra, write_spectra, empty_fibermap, specprod_root, iterfiles
from ..io.util import add_columns
from ..scripts import group_spectra
from ..pixgroup import SpectraLite, get_exp2uniqpix_map
from desispec.maskbits import fibermask
from desiutil.io import encode_table
import desiutil.healpix

class TestPixGroup(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.testdir = tempfile.mkdtemp()
        cls.outdir = os.path.join(cls.testdir, 'output')

        cls.origenv = dict()
        for key in ['DESI_SPECTRO_REDUX', 'SPECPROD']:
            cls.origenv[key] = os.getenv(key)  #- will be None if not set
        
        os.environ['DESI_SPECTRO_REDUX'] = cls.testdir
        os.environ['SPECPROD'] = 'grouptest'

        cls.nspec_per_frame = 8 # needs to be at least 4
        cls.nframe_per_night = 4 # needs to be at least 4
        cls.nights = [20200101, 20200102, 20200103] # needs to be at least 3 nights
        cls.badnight = cls.nights[-1]
        cls.badslice = np.arange(2, int(np.min([6,cls.nspec_per_frame]) ) ).astype(int)
        
        cls.healpix = 19456
        cls.nside = 64
        cls.uniqpix = desiutil.healpix.hpix2upix(cls.nside, cls.healpix)
        cls.survey = 'main'
        cls.faprogram = 'dark'
        cls.specfile = findfile('spectra', groupname=cls.healpix,
                survey=cls.survey, faprogram=cls.faprogram)
        cls.specbase = os.path.basename(cls.specfile)

        frames = dict()
        meta = {'EXPID': 1.0, 'FLAVOR': 'science', 'SURVEY': cls.survey, 'PROGRAM': cls.faprogram}
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
        exprows = list()
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

                    tileid = expid*10
                    curframe.meta['NIGHT'] = night
                    curframe.meta['EXPID'] = expid
                    curframe.meta['TILEID'] = tileid
                    curframe.meta['MJD-OBS'] = 55555.0+ii + 0.1*nfram
                    curframe.fibermap.meta['SURVEY'] = cls.survey
                    curframe.fibermap.meta['FAPRGRM'] = cls.faprogram
                    write_frame(findfile('cframe', night, expid, camera), curframe)
                    exprows.append((night, expid, tileid,
                        cls.survey, cls.faprogram, 0, cls.healpix))

        #- Remove one file to test missing data
        os.remove(findfile('cframe', cls.nights[0], 1, 'r0'))

        cls.framefiles = list(iterfiles(cls.testdir, prefix='cframe'))
        cls.exptable = Table(rows=exprows,
                names=('NIGHT', 'EXPID', 'TILEID', 'SURVEY', 'PROGRAM', 'SPECTRO', 'HEALPIX'))
        cls.expfile = os.path.join(cls.testdir, 'hpixexp.csv')
        cls.exptable.write(cls.expfile)

        # Setup a dummy SpectraLite for I/O tests
        cls.fileio = os.path.join(cls.testdir, 'test_spectralite.fits')
        cls.fileiogz = os.path.join(cls.testdir, 'test_spectralite.fits.gz')

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

        cls.bands = ["b", "r", "z"]
        cls.wave = {}
        cls.flux = {}
        cls.ivar = {}
        cls.mask = {}
        cls.res = {}

        for s in range(cls.nspec):
            for b in cls.bands:
                cls.wave[b] = np.arange(cls.nwave, dtype=float)
                cls.flux[b] = np.repeat(np.arange(cls.nspec, dtype=float),
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
        #- Remove testdir only if it was created by tempfile.mkdtemp
        if cls.testdir.startswith(tempfile.gettempdir()) and os.path.exists(cls.testdir):
            shutil.rmtree(cls.testdir)

        #- restore environment
        for key, value in cls.origenv.items():
            if value is not None:
                os.environ[key] = value
            elif key in os.environ:
                del os.environ[key]

    def setUp(self):
        os.environ['DESI_SPECTRO_REDUX'] = self.testdir
        os.environ['SPECPROD'] = 'grouptest'
        os.makedirs(self.outdir)

    def tearDown(self):
        if os.path.exists(self.outdir):
            shutil.rmtree(self.outdir)
        if os.path.exists(self.fileio):
            os.remove(self.fileio)
        if os.path.exists(self.fileiogz):
            os.remove(self.fileiogz)

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

        #- don't rely on $DESI_SPECTRO_REDUX/$SPECPROD in case env changed
        tiledir = os.path.join(self.testdir, 'grouptest', 'tiles')
        if os.path.exists(tiledir):
            shutil.rmtree(tiledir)

        hpixdir = os.path.join(self.testdir, 'grouptest', 'healpix')
        if os.path.exists(hpixdir):
            shutil.rmtree(hpixdir)

    @unittest.skipIf(True, 'unsupported regroup without input list')
    def test_regroup_per_night(self):
        #- Run for each night and confirm that spectra file is correct size
        for i, night in enumerate(self.nights):
            cmd = 'desi_group_spectra -o {} --nights {}'.format(
                    self.specfile, night)
            args = group_spectra.parse(cmd.split()[1:])
            group_spectra.main(args)

            spectra = read_spectra(self.specfile)
            num_nights = i+1
            nspec = self.nspec_per_frame * self.nframe_per_night * num_nights
        
            self.assertEqual(len(spectra.fibermap), nspec)
            self.assertEqual(spectra.flux['b'].shape[0], nspec)

    def test_regroup_fiberstatus_propagation(self):
        """Test propagation of fiberstatus bits"""
        cmd = 'desi_group_spectra -o {} --expfile {} --nights {}'.format(
                self.specfile, self.expfile, self.badnight)
        args = group_spectra.parse(cmd.split()[1:])
        group_spectra.main(args)

        spectra = read_spectra(self.specfile)
        nspec = self.nspec_per_frame * self.nframe_per_night

        fibermap_array = np.array(spectra.fibermap['FIBERSTATUS'])
        self.assertEqual(len(fibermap_array), nspec)
        
        self.assertEqual(np.sum( (fibermap_array & fibermask.BADAMPB) > 0 ), 2*len(self.badslice))
        self.assertEqual(np.sum( (fibermap_array & fibermask.BADAMPR) > 0 ), 2*len(self.badslice))
        self.assertEqual(np.sum( (fibermap_array & fibermask.BADAMPZ) > 0 ), 2*len(self.badslice))

        badamp_brz = ( fibermask.BADAMPB | fibermask.BADAMPR | fibermask.BADAMPZ )
        self.assertEqual(np.sum( (fibermap_array == badamp_brz) ), len(self.badslice))
        
    def test_regroup_nights(self):
        """Test filtering to a specific set of nights"""
        num_nights = 2
        nights = ','.join([str(tmp) for tmp in self.nights[0:num_nights]])
        cmd = f'desi_group_spectra -o {self.specfile} --expfile {self.expfile}'
        cmd += f' --nights {nights}'
        args = group_spectra.parse(cmd.split()[1:])
        group_spectra.main(args)

        spectra = read_spectra(self.specfile)
        nspec = self.nspec_per_frame * self.nframe_per_night * num_nights
        
        self.assertEqual(len(spectra.fibermap), nspec)
        self.assertEqual(spectra.flux['b'].shape[0], nspec)
        
    def test_regroup_inframes(self):
        """Test grouping from an input list of frames"""
        cmd = 'desi_group_spectra -o {}'.format(self.specfile)
        cmd += ' --inframes ' + ' '.join(self.framefiles)

        args = group_spectra.parse(cmd.split()[1:])
        group_spectra.main(args)

        spectra = read_spectra(self.specfile)
        num_nights = len(self.nights)
        nspec = self.nspec_per_frame * self.nframe_per_night * num_nights

        self.assertEqual(len(spectra.fibermap), nspec)
        self.assertEqual(spectra.flux['b'].shape[0], nspec)

        self.assertIn('SURVEY', spectra.fibermap.meta)
        self.assertEqual(spectra.fibermap.meta['SURVEY'], self.survey)

        #- confirm that we can read the mask with memmap=True
        with fits.open(self.specfile, memmap=True) as fx:
            mask = fx['B_MASK'].data

    def test_regroup_expfile(self):
        """Test grouping from a table of exposures"""
        cmd = f'desi_group_spectra -o {self.specfile} --expfile {self.expfile}'

        args = group_spectra.parse(cmd.split()[1:])
        group_spectra.main(args)

        spectra = read_spectra(self.specfile)
        num_nights = len(self.nights)
        nspec = self.nspec_per_frame * self.nframe_per_night * num_nights
    
        self.assertEqual(len(spectra.fibermap), nspec)
        self.assertEqual(spectra.flux['b'].shape[0], nspec)

        #- confirm that we can read the mask with memmap=True
        with fits.open(self.specfile, memmap=True) as fx:
            mask = fx['B_MASK'].data

    def test_regroup_healpix(self):
        """Test grouping table of exposures with healpix filter"""
        cmd = f'desi_group_spectra -o {self.specfile} --expfile {self.expfile}'
        cmd += f' --healpix {self.healpix}'
        cmd += f' --header BLAT=True BIM=False BAM=false FOO=bar BIZ=1 BAT=2.3'

        # Note: BLAT=True and BIM=False should be promoted to genuine
        # boolean True/False, but BAM=false remains 'false'

        print(f'RUNNING {cmd}')
        args = group_spectra.parse(cmd.split()[1:])
        group_spectra.main(args)

        spectra = read_spectra(self.specfile)
        num_nights = len(self.nights)
        nspec = self.nspec_per_frame * self.nframe_per_night * num_nights

        self.assertEqual(len(spectra.fibermap), nspec)
        self.assertEqual(spectra.flux['b'].shape[0], nspec)

        #- confirm that we can read the mask with memmap=True
        with fits.open(self.specfile, memmap=True) as fx:
            mask = fx['B_MASK'].data
            hdr = fx[0].header

        print(f"HEADER {type(hdr)} is")
        for key in hdr:
            print(key, hdr[key])

        self.assertEqual(hdr['SPGRP'], 'uniqpix')
        self.assertEqual(hdr['SPGRPVAL'], self.uniqpix)
        self.assertEqual(hdr['UNIQPIX'],  self.uniqpix)
        self.assertEqual(hdr['HPXPIXEL'], self.healpix)
        self.assertEqual(hdr['HPXNSIDE'], self.nside)
        self.assertEqual(hdr['HPXNEST'], True)
        self.assertEqual(hdr['BLAT'], True)
        self.assertEqual(hdr['BIM'], False)
        self.assertEqual(hdr['BAM'], 'false')
        self.assertEqual(hdr['FOO'], 'bar')
        self.assertEqual(hdr['BIZ'], 1)
        self.assertEqual(hdr['BAT'], 2.3)


    def test_reduxdir(self):
        #- Test using a non-standard redux directory
        reduxdir = specprod_root()
        cmd = f'desi_group_spectra -o {self.specfile} --expfile {self.expfile}'
        cmd += f' --reduxdir {reduxdir}'

        #- Change SPECPROD and confirm that default location changed
        os.environ['SPECPROD'] = 'blatfoo'
        self.assertNotEqual(reduxdir, specprod_root())

        args = group_spectra.parse(cmd.split()[1:])
        group_spectra.main(args)

        spectra = read_spectra(self.specfile)
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

        pathgz = write_spectra(self.fileiogz, spec)
        assert(pathgz == os.path.abspath(self.fileiogz))

        # read back in and verify
        comp = read_spectra(self.fileio)
        self.verify_spectralite(comp, self.fmap1)

        compgz = read_spectra(self.fileiogz)
        self.verify_spectralite(compgz, self.fmap1)

    def test_get_exp2uniqpix_map(self):
        """Test desispec.pixgroup.get_exp2uniqpix_map"""
        from ..pixgroup import get_exp2uniqpix_map

        # 3 targets all on the same tile, petal, and at the same sky position
        # so they all land in the same UNIQPIX
        n_targets = 3
        zcat = Table({
            'TARGETID': np.arange(n_targets, dtype=np.int64),
            'TILEID': np.full(n_targets, 1234, dtype=np.int32),
            'PETAL_LOC': np.full(n_targets, 5, dtype=np.int16),
            'TARGET_RA': np.full(n_targets, 150.0, dtype=np.float64),
            'TARGET_DEC': np.full(n_targets, 30.0, dtype=np.float64),
        })

        # 2 exposures: expid 100 has 2 cameras on the same petal and should be
        # deduplicated to one row per exposure-petal combination
        frames = Table({
            'NIGHT': np.array([20201020, 20201020, 20201021], dtype=np.int32),
            'EXPID': np.array([100, 100, 101], dtype=np.int32),
            'TILEID': np.array([1234, 1234, 1234], dtype=np.int32),
            'CAMERA': np.array(['b5', 'r5', 'b5']),
        })

        result = get_exp2uniqpix_map(zcat, frames)

        # Check output columns (PETAL_LOC is renamed to SPECTRO for historical compatibility)
        self.assertEqual(set(result.colnames),
                         {'NIGHT', 'EXPID', 'TILEID', 'SPECTRO', 'UNIQPIX', 'NSIDE', 'HEALPIX', 'NTARGETS'})

        # 2 unique (EXPID, PETAL_LOC) combinations, all targets in same pixel -> 2 rows
        self.assertEqual(len(result), 2)

        # SPECTRO comes from the last character of CAMERA ('b5' -> 5)
        self.assertTrue(np.all(result['SPECTRO'] == 5))

        # TILEID should pass through unchanged
        self.assertTrue(np.all(result['TILEID'] == 1234))

        # Both exposures present with correct NIGHT values
        result_sorted = result[np.argsort(result['EXPID'])]
        self.assertEqual(list(result_sorted['EXPID']), [100, 101])
        self.assertEqual(list(result_sorted['NIGHT']), [20201020, 20201021])

        # All targets at the same position -> same UNIQPIX -> NTARGETS == n_targets
        self.assertTrue(np.all(result['NTARGETS'] == n_targets))

        # UNIQPIX encoding: UNIQPIX == 4 * NSIDE**2 + HEALPIX
        for row in result:
            self.assertEqual(int(row['UNIQPIX']), 4 * int(row['NSIDE'])**2 + int(row['HEALPIX']))

        # NSIDE must be a positive power of 2
        for row in result:
            nside = int(row['NSIDE'])
            self.assertGreater(nside, 0)
            self.assertEqual(nside & (nside - 1), 0)

    def test_get_hpix2upix_map(self):
        """Test desispec.pixgroup.get_hpix2upix_map"""
        from ..pixgroup import get_hpix2upix_map

        # --- Test 1: single nside, partial coverage ---
        # UNIQPIX = 4 * nside**2 + ipix; for nside=2: UNIQPIX = 16 + ipix
        # nside=2 has 12*4 = 48 pixels total
        nside2 = 2
        npix2 = 12 * nside2**2   # 48
        upix_a = 4 * nside2**2 + 0   # 16, covers ipix=0
        upix_b = 4 * nside2**2 + 3   # 19, covers ipix=3

        healpix_map, nside_max = get_hpix2upix_map([upix_a, upix_b])

        self.assertEqual(nside_max, nside2)
        self.assertEqual(len(healpix_map), npix2)
        self.assertEqual(healpix_map[0], upix_a)
        self.assertEqual(healpix_map[3], upix_b)
        # All uncovered pixels should be -1
        covered = np.zeros(npix2, dtype=bool)
        covered[0] = True
        covered[3] = True
        self.assertTrue(np.all(healpix_map[~covered] == -1))

        # --- Test 2: mixed nside, coarse pixel expands to multiple fine slots ---
        # Coarse: nside=2, ipix=0 -> UNIQPIX=16
        #   Expands to 4^(order_max - order_coarse) = 4^(2-1) = 4 fine pixels at nside=4: ipix 0,1,2,3
        # Fine:   nside=4, ipix=5 -> UNIQPIX=4*16+5=69
        # nside_max=4, output shape = 12*16 = 192
        nside4 = 4
        npix4 = 12 * nside4**2   # 192
        upix_coarse = 4 * nside2**2 + 0   # 16, nside=2, ipix=0
        upix_fine   = 4 * nside4**2 + 5   # 69, nside=4, ipix=5

        healpix_map2, nside_max2 = get_hpix2upix_map([upix_coarse, upix_fine])

        self.assertEqual(nside_max2, nside4)
        self.assertEqual(len(healpix_map2), npix4)
        # Coarse pixel (nside=2, ipix=0) maps to fine pixels 0,1,2,3
        self.assertTrue(np.all(healpix_map2[0:4] == upix_coarse))
        # Fine pixel (nside=4, ipix=5) maps to exactly slot 5
        self.assertEqual(healpix_map2[5], upix_fine)
        # Slot 4 is between the two covered regions and should be uncovered
        self.assertEqual(healpix_map2[4], -1)
        # All remaining slots are -1
        covered2 = np.zeros(npix4, dtype=bool)
        covered2[0:4] = True
        covered2[5] = True
        self.assertTrue(np.all(healpix_map2[~covered2] == -1))

    def test_frames2spectra(self):
        """Test frames2spectra"""
        from ..pixgroup import FrameLite, frames2spectra

        #- Read input frames
        frames = list()
        for filename in self.framefiles:
            frames.append( FrameLite.read(filename) )

        #- Set some header keywords for testing
        for i, fr in enumerate(frames):
            fr.fibermap.meta['BLAT'] = i          #- conflicting, never merged
            fr.fibermap.meta['FOO'] = 'biz'       #- consistent, but still not merged
            fr.fibermap.meta['TILEID'] = 1234     #- only merged if onetile=True
            fr.fibermap.meta['SURVEY'] = 'main'   #- always merged
            fr.fibermap.meta['PROGRAM'] = 'dark'  #- always merged

        #- converting list of frames
        spectra = frames2spectra(frames)

        #- converting dict of frames
        framedict = dict()
        for fr in frames:
            night = fr.meta['NIGHT']
            expid = fr.meta['EXPID']
            camera = fr.meta['CAMERA']
            framedict[(night, expid, camera)] = fr

        spectra = frames2spectra(framedict)

        #- Check header keyword propagation
        self.assertNotIn('BLAT', spectra.fibermap.meta)
        self.assertNotIn('FOO', spectra.fibermap.meta)
        self.assertNotIn('TILEID', spectra.fibermap.meta)
        self.assertIn('SURVEY', spectra.fibermap.meta)
        self.assertIn('PROGRAM', spectra.fibermap.meta)

        #- but originals were not changed
        for i, fr in enumerate(frames):
            self.assertEqual(fr.fibermap.meta['BLAT'], i)
            self.assertEqual(fr.fibermap.meta['FOO'], 'biz')
            self.assertEqual(fr.fibermap.meta['TILEID'], 1234)
            self.assertEqual(fr.fibermap.meta['SURVEY'], 'main')
            self.assertEqual(fr.fibermap.meta['PROGRAM'], 'dark')

        #- frames2spectra(..., onetile=True) should propagate TILEID
        spectra = frames2spectra(framedict, onetile=True)
        self.assertNotIn('BLAT', spectra.fibermap.meta)
        self.assertNotIn('FOO', spectra.fibermap.meta)
        self.assertIn('TILEID', spectra.fibermap.meta)  #- different from before
        self.assertIn('SURVEY', spectra.fibermap.meta)
        self.assertIn('PROGRAM', spectra.fibermap.meta)

        #- originals are still not changed
        for i, fr in enumerate(frames):
            self.assertEqual(fr.fibermap.meta['BLAT'], i)
            self.assertEqual(fr.fibermap.meta['FOO'], 'biz')
            self.assertEqual(fr.fibermap.meta['TILEID'], 1234)
            self.assertEqual(fr.fibermap.meta['SURVEY'], 'main')
            self.assertEqual(fr.fibermap.meta['PROGRAM'], 'dark')


