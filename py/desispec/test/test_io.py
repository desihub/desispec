# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""Test desispec.io.
"""
from __future__ import absolute_import, division
# The line above will help with 2to3 support.
import unittest
import os
import sys
import tempfile
from datetime import datetime, timedelta
from shutil import rmtree
from pkg_resources import resource_filename
import numpy as np
from astropy.io import fits
from astropy.table import Table
from ..frame import Frame

skipMock = False
try:
    from unittest.mock import patch
except ImportError:
    # Python 2
    skipMock = True

class TestIO(unittest.TestCase):
    """Test desispec.io.
    """

    @classmethod
    def setUpClass(cls):
        """Create unique test filename in a subdirectory.
        """
        cls.testDir = tempfile.mkdtemp()
        cls.testfile = os.path.join(cls.testDir, 'desispec_test_io.fits')
        cls.testyfile = os.path.join(cls.testDir, 'desispec_test_io.yaml')
        # cls.testbrfile appears to be unused by this class.
        cls.testbrfile = os.path.join(cls.testDir, 'desispec_test_io-br.fits')
        cls.origEnv = {'SPECPROD':None,
            "DESI_SPECTRO_DATA":None,
            "DESI_SPECTRO_REDUX":None}
        cls.testEnv = {'SPECPROD':'dailytest',
            "DESI_SPECTRO_DATA":os.path.join(cls.testDir,'spectro','data'),
            "DESI_SPECTRO_REDUX":os.path.join(cls.testDir,'spectro','redux')}
        cls.datadir = cls.testEnv['DESI_SPECTRO_DATA']
        cls.reduxdir = os.path.join(cls.testEnv['DESI_SPECTRO_REDUX'],
                                    cls.testEnv['SPECPROD'])
        for e in cls.origEnv:
            if e in os.environ:
                cls.origEnv[e] = os.environ[e]
            os.environ[e] = cls.testEnv[e]

    def setUp(self):
        if os.path.isdir(self.datadir):
            rmtree(self.datadir)
        if os.path.isdir(self.reduxdir):
            rmtree(self.reduxdir)

    def tearDown(self):
        if os.path.exists(self.testfile):
            os.remove(self.testfile)

    @classmethod
    def tearDownClass(cls):
        """Cleanup test files if they exist.
        """
        for testfile in [cls.testfile, cls.testyfile, cls.testbrfile]:
            if os.path.exists(testfile):
                os.remove(testfile)

        for e in cls.origEnv:
            if cls.origEnv[e] is None:
                del os.environ[e]
            else:
                os.environ[e] = cls.origEnv[e]

        if os.path.exists(cls.testDir):
            rmtree(cls.testDir)

    def test_write_bintable(self):
        """Test writing binary tables to FITS.
        """
        from ..io.util import write_bintable, fitsheader
        #
        # Input: Table
        #
        hdr = fitsheader(dict(A=1, B=2))
        hdr['C'] = ('BLAT', 'FOO')
        data = Table()
        data['X'] = [1, 2, 3]
        data['Y'] = [3, 4, 5]
        write_bintable(self.testfile, data, header=hdr)
        #
        # Standard suite of table tests.
        #
        result, newhdr = fits.getdata(self.testfile, header=True)
        self.assertEqual(sorted(result.dtype.names), sorted(data.dtype.names))
        for colname in data.dtype.names:
            self.assertTrue(np.all(result[colname] == data[colname]), '{} data mismatch'.format(colname))
        self.assertEqual(newhdr.comments['C'], 'FOO')
        for key in hdr.keys():
            self.assertIn(key, newhdr)
        self.assertIn('DATASUM', newhdr)
        self.assertIn('CHECKSUM', newhdr)
        os.remove(self.testfile)
        #
        # Input: ndarray
        #
        hdr = dict(A=1, B=2)
        data = data.as_array()
        write_bintable(self.testfile, data, header=hdr)
        #
        # Standard suite of table tests.
        #
        result, newhdr = fits.getdata(self.testfile, header=True)
        self.assertEqual(sorted(result.dtype.names), sorted(data.dtype.names))
        for colname in data.dtype.names:
            self.assertTrue(np.all(result[colname] == data[colname]), '{} data mismatch'.format(colname))
        # self.assertEqual(newhdr.comments['C'], 'FOO')
        for key in hdr.keys():
            self.assertIn(key, newhdr)
        self.assertIn('DATASUM', newhdr)
        self.assertIn('CHECKSUM', newhdr)
        os.remove(self.testfile)
        #
        # Input: dictionary
        #
        hdr = dict(A=1, B=2)
        d = dict(X=np.array([1, 2, 3]), Y=np.array([3, 4, 5]))
        write_bintable(self.testfile, d, header=hdr)
        #
        # Standard suite of table tests.
        #
        result, newhdr = fits.getdata(self.testfile, header=True)

        self.assertEqual(sorted(result.dtype.names), sorted(data.dtype.names))

        for colname in data.dtype.names:
            self.assertTrue(np.all(result[colname] == data[colname]), '{} data mismatch'.format(colname))
        # self.assertEqual(newhdr.comments['C'], 'FOO')
        for key in hdr.keys():
            self.assertIn(key, newhdr)
        self.assertIn('DATASUM', newhdr)
        self.assertIn('CHECKSUM', newhdr)
        os.remove(self.testfile)
        #
        # Input: Table with column comments.
        #
        hdr = fitsheader(dict(A=1, B=2))
        hdr['C'] = ('BLAT', 'FOO')
        data = Table()
        data['X'] = [1, 2, 3]
        data['Y'] = [3, 4, 5]
        write_bintable(self.testfile, data, header=hdr,
                       comments={'X': 'This is X', 'Y': 'This is Y'},
                       units={'X': 'mm', 'Y': 'mm'})
        #
        # Standard suite of table tests.
        #
        result, newhdr = fits.getdata(self.testfile, header=True)
        self.assertEqual(sorted(result.dtype.names), sorted(data.dtype.names))
        for colname in data.dtype.names:
            self.assertTrue(np.all(result[colname] == data[colname]), '{} data mismatch'.format(colname))
        # self.assertEqual(newhdr.comments['C'], 'FOO')
        for key in hdr.keys():
            self.assertIn(key, newhdr)
        self.assertIn('DATASUM', newhdr)
        self.assertIn('CHECKSUM', newhdr)
        self.assertEqual(newhdr['TTYPE1'], 'X')
        self.assertEqual(newhdr.comments['TTYPE1'], 'This is X')
        self.assertEqual(newhdr['TTYPE2'], 'Y')
        self.assertEqual(newhdr.comments['TTYPE2'], 'This is Y')
        self.assertEqual(newhdr['TUNIT1'], 'mm')
        self.assertEqual(newhdr.comments['TUNIT1'], 'X units')
        self.assertEqual(newhdr['TUNIT2'], 'mm')
        self.assertEqual(newhdr.comments['TUNIT2'], 'Y units')
        #
        # Input: Table with no EXTNAME, existing file
        #
        write_bintable(self.testfile, data, header=hdr)
        #
        # Input: Table with EXTNAME, existing file
        #
        write_bintable(self.testfile, data, header=hdr, extname='FOOBAR')
        #
        # Standard suite of table tests.
        #
        result, newhdr = fits.getdata(self.testfile, header=True, extname='FOOBAR')
        self.assertEqual(sorted(result.dtype.names), sorted(data.dtype.names))
        for colname in data.dtype.names:
            self.assertTrue(np.all(result[colname] == data[colname]), '{} data mismatch'.format(colname))
        # self.assertEqual(newhdr.comments['C'], 'FOO')
        for key in hdr.keys():
            self.assertIn(key, newhdr)
        self.assertIn('DATASUM', newhdr)
        self.assertIn('CHECKSUM', newhdr)
        #
        # Input: Table with existing EXTNAME, existing file
        #
        write_bintable(self.testfile, data, header=hdr, extname='FOOBAR')
        #
        # Input: Table with EXTNAME, existing file, overwrite
        #
        write_bintable(self.testfile, data, header=hdr, extname='FOOBAR', clobber=True)


    #- Some macs fail `assert_called_with` tests due to equivalent paths
    #- of `/private/var` vs. `/var`, so skip this test on Macs.
    @unittest.skipIf(sys.platform == 'darwin', "Skipping memmap test on Mac.")
    @unittest.skipIf(skipMock, "Skipping test that requires unittest.mock.")
    def test_supports_memmap(self):
        """Test utility to detect when memory-mapping is not possible.
        """
        from ..io.util import _supports_memmap
        foofile = os.path.join(os.path.dirname(self.testfile), 'foo.dat')
        with patch('os.remove') as rm:
            with patch('numpy.memmap') as mm:
                mm.return_value = True
                foo = _supports_memmap(self.testfile)
                self.assertTrue(foo)
                mm.assert_called_with(foofile, dtype='f4', mode='w+', shape=(3, 4))
            rm.assert_called_with(foofile)
        with patch('os.remove') as rm:
            with patch('numpy.memmap') as mm:
                mm.side_effect = OSError(38, 'Function not implemented')
                foo = _supports_memmap(self.testfile)
                self.assertFalse(foo)
                mm.assert_called_with(foofile, dtype='f4', mode='w+', shape=(3, 4))
            rm.assert_called_with(foofile)

    def test_dict2ndarray(self):
        """Test conversion of dictionaries into structured arrays.
        """
        from ..io.util import _dict2ndarray
        x = np.arange(10)
        y = np.arange(10)*2
        z = np.arange(20).reshape((10, 2))
        self.assertEqual(z.shape, (10, 2))
        d = dict(x=x, y=y, z=z)
        nddata = _dict2ndarray(d)
        self.assertTrue((nddata['x'] == x).all())
        self.assertTrue((nddata['y'] == y).all())
        self.assertTrue((nddata['z'] == z).all())
        nddata = _dict2ndarray(d, columns=['x', 'y'])
        self.assertTrue((nddata['x'] == x).all())
        self.assertTrue((nddata['y'] == y).all())
        with self.assertRaises(ValueError) as ex:
            (nddata['z'] == z).all()
            self.assertEqual(ex.exception.args[0], 'no field of name z')

    def test_fitsheader(self):
        """Test desispec.io.util.fitsheader.
        """
        #- None is ok; just returns blank Header
        from ..io.util import fitsheader
        header = fitsheader(None)
        self.assertTrue(isinstance(header, fits.Header))
        self.assertEqual(len(header), 0)

        #- input is dict
        hdr = dict()
        hdr['BLAT'] = 'foo'
        hdr['BAR'] = (1, 'biz bat')
        header = fitsheader(hdr)
        self.assertTrue(isinstance(header, fits.Header))
        self.assertEqual(header['BLAT'], 'foo')
        self.assertEqual(header['BAR'], 1)
        self.assertEqual(header.comments['BAR'], 'biz bat')

        #- input header as a list, get a fits.Header back
        hdr = list()
        hdr.append( ('BLAT', 'foo') )
        hdr.append( ('BAR', (1, 'biz bat')) )
        header = fitsheader(hdr)
        self.assertTrue(isinstance(header, fits.Header))
        self.assertEqual(header['BLAT'], 'foo')
        self.assertEqual(header['BAR'], 1)
        self.assertEqual(header.comments['BAR'], 'biz bat')

        #- fits.Header -> fits.Header
        header = fitsheader(header)
        self.assertTrue(isinstance(header, fits.Header))
        self.assertEqual(header['BLAT'], 'foo')
        self.assertEqual(header['BAR'], 1)
        self.assertEqual(header.comments['BAR'], 'biz bat')

        #- Can't convert and int into a fits Header
        self.assertRaises(ValueError, fitsheader, (1,))

    def _make_frame(self, nspec=5, nwave=10, ndiag=3):
        wave = np.arange(nwave)
        flux = np.random.uniform(size=(nspec, nwave))
        ivar = np.random.uniform(size=(nspec, nwave))
        mask = np.zeros((nspec, nwave), dtype=int)
        R = np.random.uniform( size=(nspec, ndiag, nwave) )
        return Frame(wave, flux, ivar, mask, R)

    def test_frame_rw(self):
        """Test reading and writing Frame objects.
        """
        from ..io.frame import read_frame, write_frame, read_meta_frame
        from ..io.fibermap import empty_fibermap
        nspec, nwave, ndiag = 5, 10, 3
        flux = np.random.uniform(size=(nspec, nwave))
        ivar = np.random.uniform(size=(nspec, nwave))
        meta = dict(BLAT=0, FOO='abc', FIBERMIN=500, FLAVOR='science')
        mask_int = np.zeros((nspec, nwave), dtype=int)
        mask_uint = np.zeros((nspec, nwave), dtype=np.uint32)
        wave = np.arange(nwave)
        R = np.random.uniform( size=(nspec, ndiag, nwave) )

        for mask in (mask_int, mask_uint):
            frx = Frame(wave, flux, ivar, mask, R, meta=meta)
            write_frame(self.testfile, frx)
            frame = read_frame(self.testfile)
            read_meta = read_meta_frame(self.testfile)

            flux2 = flux.astype('f4').astype('f8')
            ivar2 = ivar.astype('f4').astype('f8')
            wave2 = wave.astype('f4').astype('f8')
            R2    = R.astype('f4').astype('f8')

            self.assertTrue(frame.wave.dtype == np.float64)
            self.assertTrue(frame.flux.dtype == np.float64)
            self.assertTrue(frame.ivar.dtype == np.float64)
            self.assertTrue(frame.resolution_data.dtype == np.float64)

            self.assertTrue(np.all(flux2 == frame.flux))
            self.assertTrue(np.all(ivar2 == frame.ivar))
            self.assertTrue(np.all(wave2 == frame.wave))
            self.assertTrue(np.all(mask == frame.mask))
            self.assertTrue(np.all(R2 == frame.resolution_data))
            self.assertTrue(frame.resolution_data.dtype.isnative)
            self.assertEqual(frame.meta['BLAT'], meta['BLAT'])
            self.assertEqual(frame.meta['FOO'], meta['FOO'])
            self.assertEqual(frame.meta['BLAT'], read_meta['BLAT'])
            self.assertEqual(frame.meta['FOO'], read_meta['FOO'])

        #- Test float32 on disk vs. float64 in memory
        for extname in ['FLUX', 'IVAR', 'RESOLUTION']:
            data = fits.getdata(self.testfile, extname)
            self.assertEqual(data.dtype, np.dtype('>f4'), '{} not type >f4'.format(extname))
        for extname in ['WAVELENGTH']:
            data = fits.getdata(self.testfile, extname)
            self.assertEqual(data.dtype, np.dtype('>f8'), '{} not type >f8'.format(extname))

        #- with and without units
        frx = Frame(wave, flux, ivar, mask, R, meta=meta)
        write_frame(self.testfile, frx)
        frame = read_frame(self.testfile)
        self.assertTrue('BUNIT' not in frame.meta)
        write_frame(self.testfile, frx, units='photon/bin')
        frame = read_frame(self.testfile)
        self.assertEqual(frame.meta['BUNIT'], 'photon/bin')
        frx.meta['BUNIT'] = 'blatfoo'
        write_frame(self.testfile, frx)
        frame = read_frame(self.testfile)
        self.assertEqual(frame.meta['BUNIT'], 'blatfoo')
        #- function argument trumps pre-existing BUNIT
        write_frame(self.testfile, frx, units='quat')
        frame = read_frame(self.testfile)
        self.assertEqual(frame.meta['BUNIT'], 'quat')

        #- with and without fibermap
        self.assertEqual(frame.fibermap, None)
        fibermap = empty_fibermap(nspec)
        fibermap['TARGETID'] = np.arange(nspec)*2
        frx = Frame(wave, flux, ivar, mask, R, fibermap=fibermap, meta=dict(FLAVOR='science'))
        write_frame(self.testfile, frx)
        frame = read_frame(self.testfile)
        for name in fibermap.dtype.names:
            match = np.all(fibermap[name] == frame.fibermap[name])
            self.assertTrue(match, 'Fibermap column {} mismatch'.format(name))

    def test_sky_rw(self):
        """Test reading and writing sky files.
        """
        from ..sky import SkyModel
        from ..io.sky import read_sky, write_sky
        nspec, nwave = 5,10
        wave = np.arange(nwave)
        flux = np.random.uniform(size=(nspec, nwave))
        ivar = np.random.uniform(size=(nspec, nwave))
        mask_int = np.zeros(shape=(nspec, nwave), dtype=int)
        mask_uint = np.zeros(shape=(nspec, nwave), dtype=np.uint32)

        for mask in (mask_int, mask_uint):
            # skyflux,skyivar,skymask,cskyflux,cskyivar,wave
            sky = SkyModel(wave, flux, ivar, mask)
            write_sky(self.testfile, sky)
            xsky = read_sky(self.testfile)

            self.assertTrue(np.all(sky.wave.astype('f4').astype('f8')  == xsky.wave))
            self.assertTrue(np.all(sky.flux.astype('f4').astype('f8')  == xsky.flux))
            self.assertTrue(np.all(sky.ivar.astype('f4').astype('f8')  == xsky.ivar))
            self.assertTrue(np.all(sky.mask  == xsky.mask))
            self.assertTrue(xsky.flux.dtype.isnative)
            self.assertEqual(sky.mask.dtype, xsky.mask.dtype)

    # fiberflat,fiberflat_ivar,fiberflat_mask,mean_spectrum,wave
    def test_fiberflat_rw(self):
        """Test reading and writing fiberflat files.
        """
        from ..fiberflat import FiberFlat
        from ..io.fiberflat import read_fiberflat, write_fiberflat
        nspec, nwave, ndiag = 10, 20, 3
        flat = np.random.uniform(size=(nspec, nwave))
        ivar = np.random.uniform(size=(nspec, nwave))
        mask = np.zeros(shape=(nspec, nwave), dtype=int)
        meanspec = np.random.uniform(size=(nwave,))
        wave = np.arange(nwave)

        ff = FiberFlat(wave, flat, ivar, mask, meanspec)

        write_fiberflat(self.testfile, ff)
        xff = read_fiberflat(self.testfile)

        self.assertTrue(np.all(ff.fiberflat.astype('f4').astype('f8') == xff.fiberflat))
        self.assertTrue(np.all(ff.ivar.astype('f4').astype('f8') == xff.ivar))
        self.assertTrue(np.all(ff.mask == xff.mask))
        self.assertTrue(np.all(ff.meanspec.astype('f4').astype('f8') == xff.meanspec))
        self.assertTrue(np.all(ff.wave.astype('f4').astype('f8') == xff.wave))

        self.assertTrue(xff.fiberflat.dtype.isnative)
        self.assertTrue(xff.ivar.dtype.isnative)
        self.assertTrue(xff.mask.dtype.isnative)
        self.assertTrue(xff.meanspec.dtype.isnative)
        self.assertTrue(xff.wave.dtype.isnative)

    def test_empty_fibermap(self):
        """Test creating empty fibermap objects.
        """
        from ..io.fibermap import empty_fibermap
        fm1 = empty_fibermap(20)
        self.assertTrue(np.all(fm1['FIBER'] == np.arange(20)))
        self.assertTrue(np.all(fm1['SPECTROID'] == 0))

        fm2 = empty_fibermap(25, specmin=10)
        self.assertTrue(np.all(fm2['FIBER'] == np.arange(25)+10))
        self.assertTrue(np.all(fm2['SPECTROID'] == 0))
        self.assertTrue(np.all(fm2['LOCATION'][0:10] == fm1['LOCATION'][10:20]))

        fm3 = empty_fibermap(10, specmin=495)
        self.assertTrue(np.all(fm3['FIBER'] == np.arange(10)+495))
        self.assertTrue(np.all(fm3['SPECTROID'] == [0,0,0,0,0,1,1,1,1,1]))

    # See https://github.com/astropy/astropy/issues/5267
    # @unittest.skipIf(PY3, "Skipping due to known problem with round-tripping in Python 3.")
    def test_fibermap_rw(self):
        """Test reading and writing fibermap files.
        """
        from ..io.fibermap import empty_fibermap, read_fibermap, write_fibermap
        fibermap = empty_fibermap(10)
        for key in fibermap.dtype.names:
            column = fibermap[key]
            fibermap[key] = np.random.random(column.shape).astype(column.dtype)

        write_fibermap(self.testfile, fibermap)

        fm = read_fibermap(self.testfile)
        self.assertTrue(isinstance(fm, Table))

        self.assertEqual(set(fibermap.dtype.names), set(fm.dtype.names))
        for key in fibermap.dtype.names:
            c1 = fibermap[key]
            c2 = fm[key]
            #- Endianness may change, but kind, size, shape, and values are same
            self.assertEqual(c1.shape, c2.shape)
            self.assertTrue(np.all(c1 == c2))
            if c1.dtype.kind == 'U':
                self.assertTrue(c2.dtype.kind in ('S', 'U'))
            else:
                self.assertEqual(c1.dtype.kind, c2.dtype.kind)
                self.assertEqual(c1.dtype.itemsize, c2.dtype.itemsize)

    def test_stdstar(self):
        """Test reading and writing standard star files.
        """
        from ..io.fluxcalibration import read_stdstar_models, write_stdstar_models
        nstd = 5
        nwave = 10
        flux = np.random.uniform(size=(nstd, nwave))
        wave = np.arange(nwave)
        fibers = np.arange(nstd)*2
        data = Table()
        data['BESTMODEL'] = np.arange(nstd)
        data['TEMPLATEID'] = np.arange(nstd)
        data['CHI2DOF'] = np.ones(nstd)
        data['REDSHIFT'] = np.zeros(nstd)
        write_stdstar_models(self.testfile, flux, wave, fibers, data)

        fx, wx, fibx, metadata = read_stdstar_models(self.testfile)
        self.assertTrue(np.all(fx == flux.astype('f4').astype('f8')))
        self.assertTrue(np.all(wx == wave.astype('f4').astype('f8')))
        self.assertTrue(np.all(fibx == fibers))

    def test_fluxcalib(self):
        """Test reading and writing flux calibration files.
        """
        from ..fluxcalibration import FluxCalib
        from ..io.fluxcalibration import read_flux_calibration, write_flux_calibration
        nspec = 5
        nwave = 10
        wave = np.arange(nwave)
        calib = np.random.uniform(size=(nspec, nwave))
        ivar = np.random.uniform(size=(nspec, nwave))
        mask = np.random.uniform(0, 2, size=(nspec, nwave)).astype('i4')

        fc = FluxCalib(wave, calib, ivar, mask)
        write_flux_calibration(self.testfile, fc)
        fx = read_flux_calibration(self.testfile)
        self.assertTrue(np.all(fx.wave  == fc.wave.astype('f4').astype('f8')))
        self.assertTrue(np.all(fx.calib == fc.calib.astype('f4').astype('f8')))
        self.assertTrue(np.all(fx.ivar  == fc.ivar.astype('f4').astype('f8')))
        self.assertTrue(np.all(fx.mask == fc.mask))

    def test_image_rw(self):
        """Test reading and writing of Image objects.
        """
        from ..image import Image
        from ..io.image import read_image, write_image
        shape = (5,5)
        pix = np.random.uniform(size=shape)
        ivar = np.random.uniform(size=shape)
        mask = np.random.randint(0, 3, size=shape)
        img1 = Image(pix, ivar, mask, readnoise=1.0, camera='b0')
        write_image(self.testfile, img1)
        img2 = read_image(self.testfile)

        #- Check output datatypes
        self.assertEqual(img2.pix.dtype, np.float64)
        self.assertEqual(img2.ivar.dtype, np.float64)
        self.assertEqual(img2.mask.dtype, np.uint32)

        #- Rounding from keeping np.float32 on disk means they aren't equal
        self.assertFalse(np.all(img1.pix == img2.pix))
        self.assertFalse(np.all(img1.ivar == img2.ivar))

        #- But they should be close, and identical after float64->float32
        self.assertTrue(np.allclose(img1.pix, img2.pix))
        self.assertTrue(np.all(img1.pix.astype(np.float32) == img2.pix))
        self.assertTrue(np.allclose(img1.ivar, img2.ivar))
        self.assertTrue(np.all(img1.ivar.astype(np.float32) == img2.ivar))

        #- masks should agree
        self.assertTrue(np.all(img1.mask == img2.mask))
        self.assertEqual(img1.readnoise, img2.readnoise)
        self.assertEqual(img1.camera, img2.camera)
        self.assertEqual(img2.mask.dtype, np.uint32)

        #- should work with various kinds of metadata header input
        meta = dict(BLAT='foo', BAR='quat', BIZ=1.0)
        img1 = Image(pix, ivar, mask, readnoise=1.0, camera='b0', meta=meta)
        write_image(self.testfile, img1)
        img2 = read_image(self.testfile)
        for key in meta:
            self.assertEqual(meta[key], img2.meta[key], 'meta[{}] not propagated'.format(key))

        #- img2 has meta as a FITS header instead of a dictionary;
        #- confirm that works too
        write_image(self.testfile, img2)
        img3 = read_image(self.testfile)
        for key in meta:
            self.assertEqual(meta[key], img3.meta[key], 'meta[{}] not propagated'.format(key))

    def test_io_qa_frame(self):
        """Test reading and writing QA_Frame.
        """
        from ..qa import QA_Frame
        from ..io.qa import read_qa_frame, write_qa_frame
        nspec = 3
        nwave = 10
        wave = np.arange(nwave)
        flux = np.random.uniform(size=(nspec, nwave))
        ivar = np.ones(flux.shape)
        frame = Frame(wave, flux, ivar, spectrograph=0)
        frame.meta = dict(CAMERA='b0', FLAVOR='science', NIGHT='20160607', EXPID=1)
        #- Init
        qaframe = QA_Frame(frame)
        qaframe.init_skysub()
        # Write
        write_qa_frame(self.testyfile, qaframe)
        # Read
        xqaframe = read_qa_frame(self.testyfile)
        # Check
        self.assertTrue(qaframe.qa_data['SKYSUB']['PARAMS']['PCHI_RESID'] == xqaframe.qa_data['SKYSUB']['PARAMS']['PCHI_RESID'])
        self.assertTrue(qaframe.flavor == xqaframe.flavor)

    def test_native_endian(self):
        """Test desiutil.io.util.native_endian.
        """
        from ..io.util import native_endian
        for dtype in ('>f8', '<f8', '<f4', '>f4', '>i4', '<i4', '>i8', '<i8'):
            data1 = np.arange(100).astype(dtype)
            data2 = native_endian(data1)
            self.assertTrue(data2.dtype.isnative, dtype+' is not native endian')
            self.assertTrue(np.all(data1 == data2))

    def test_findfile(self):
        """Test desispec.io.meta.findfile and desispec.io.download.filepath2url.
        """
        from ..io.meta import findfile
        from ..io.download import filepath2url
        filenames1 = list()
        filenames2 = list()
        kwargs = {
            'night':'20150510',
            'expid':2,
            'spectrograph':3
        }
        for i in ('sky', 'stdstars'):
            # kwargs['i'] = i
            for j in ('b','r','z'):
                kwargs['band'] = j
                if i == 'sky':
                    kwargs['camera'] = '{band}{spectrograph:d}'.format(**kwargs)
                else:
                    kwargs['camera'] = '{spectrograph:d}'.format(**kwargs)
                filenames1.append(findfile(i,**kwargs))
                filenames2.append(os.path.join(os.environ['DESI_SPECTRO_REDUX'],
                    os.environ['SPECPROD'],'exposures',kwargs['night'],
                    '{expid:08d}'.format(**kwargs),
                    '{i}-{camera}-{expid:08d}.fits'.format(i=i,camera=kwargs['camera'],expid=kwargs['expid'])))
        for k,f in enumerate(filenames1):
            self.assertEqual(os.path.basename(filenames1[k]),
                             os.path.basename(filenames2[k]))
            self.assertEqual(filenames1[k],filenames2[k])
            self.assertEqual(filepath2url(filenames1[k]),
                os.path.join('https://portal.nersc.gov/project/desi',
                'collab','spectro','redux',os.environ['SPECPROD'],'exposures',
                kwargs['night'],'{expid:08d}'.format(**kwargs),
                os.path.basename(filenames2[k])))
        #
        # Make sure that all required inputs are set.
        #
        with self.assertRaises(ValueError) as cm:
            foo = findfile('stdstars',expid=2,spectrograph=0)
        the_exception = cm.exception
        self.assertEqual(str(the_exception), "Required input 'night' is not set for type 'stdstars'!")
        with self.assertRaises(ValueError) as cm:
            foo = findfile('spectra')
        the_exception = cm.exception
        self.assertEqual(str(the_exception), "Required input 'groupname' is not set for type 'spectra'!")

        #- Some findfile calls require $DESI_SPECTRO_DATA; others do not
        del os.environ['DESI_SPECTRO_DATA']
        x = findfile('spectra', groupname=123)
        self.assertTrue(x is not None)
        with self.assertRaises(AssertionError):
            x = findfile('fibermap', night='20150101', expid=123)
        os.environ['DESI_SPECTRO_DATA'] = self.testEnv['DESI_SPECTRO_DATA']

        #- Some require $DESI_SPECTRO_REDUX; others to not
        del os.environ['DESI_SPECTRO_REDUX']
        x = findfile('fibermap', night='20150101', expid=123)
        self.assertTrue(x is not None)
        with self.assertRaises(AssertionError):
            x = findfile('spectra', groupname=123)
        os.environ['DESI_SPECTRO_REDUX'] = self.testEnv['DESI_SPECTRO_REDUX']

    def test_findfile_outdir(self):
        """Test using desispec.io.meta.findfile with an output directory.
        """
        from ..io.meta import findfile
        outdir = '/blat/foo/bar'
        x = findfile('fibermap', night='20150101', expid=123, outdir=outdir)
        self.assertEqual(x, os.path.join(outdir, os.path.basename(x)))

    def test_get_nights(self):
        """ Test desispec.io.meta.get_nights
        """
        from ..io.meta import get_nights
        from ..io.meta import findfile
        from ..io.util import makepath
        os.environ['DESI_SPECTRO_REDUX'] = self.testEnv['DESI_SPECTRO_REDUX']
        os.environ['SPECPROD'] = self.testEnv['SPECPROD']
        # Generate dummy path
        for night in ['20150101', '20150102']:
            x1 = findfile('frame', camera='b0', night=night, expid=123)
            makepath(x1)
            x2 = findfile('psfnight', camera='b0', night=night)
            makepath(x2)
        # Add a bad 'night'
        x1 = x1.replace('20150102', 'dummy')
        makepath(x1)
        # Search for nights
        nights = get_nights()
        self.assertEqual(len(nights), 2)
        self.assertTrue(isinstance(nights, list))
        self.assertTrue('20150102' in nights)
        # Keep path
        nights = get_nights(strip_path=False)
        self.assertTrue('/' in nights[0])
        # Calib
        nights = get_nights(sub_folder='calibnight')
        self.assertTrue('20150102' in nights)

    def test_search_framefile(self):
        """ Test desispec.io.frame.search_for_framefile
        """
        from ..io.frame import search_for_framefile
        from ..io.meta import findfile
        from ..io.util import makepath
        # Setup paths
        # os.environ['DESI_SPECTRO_REDUX'] = self.testEnv['DESI_SPECTRO_REDUX']
        # os.environ['SPECPROD'] = self.testEnv['SPECPROD']
        # Generate a dummy frame file
        x = findfile('frame', camera='b0', night='20150101', expid=123)
        makepath(x)
        with open(x,'a') as f:
            pass
        # Find it
        mfile = search_for_framefile('frame-b0-000123.fits')
        self.assertEqual(x, mfile)

    def test_get_reduced_frames(self):
        """ Test desispec.io.get_reduced_frames
        """
        from ..io import get_reduced_frames
        from ..io.meta import findfile
        from ..io.util import makepath
        # Setup paths
        # os.environ['DESI_SPECTRO_REDUX'] = self.testEnv['DESI_SPECTRO_REDUX']
        # os.environ['SPECPROD'] = self.testEnv['SPECPROD']
        # Generate a dummy frame file
        for expid, night in zip((123,150), ['20150101', '20150102']):
            x = findfile('cframe', camera='b0', night=night, expid=expid)
            makepath(x)
            with open(x,'a') as f:
                pass
        # Find it
        mfile = get_reduced_frames()
        self.assertEqual(2, len(mfile))

    def test_find_exposure_night(self):
        """ Test desispec.io.find_exposure_night
        """
        from ..io import find_exposure_night
        from ..io.meta import findfile
        from ..io.util import makepath
        # Setup paths
        # os.environ['DESI_SPECTRO_REDUX'] = self.testEnv['DESI_SPECTRO_REDUX']
        # os.environ['SPECPROD'] = self.testEnv['SPECPROD']
        # Generate a dummy frame file
        for expid, night in zip((123, 150), ['20150101', '20150102']):
            x = findfile('cframe', camera='b0', night=night, expid=expid)
            makepath(x)
            with open(x, 'a') as f:
                pass
        # Find it
        night1 = find_exposure_night(123)
        self.assertEqual(night1, '20150101')
        night1 = find_exposure_night(150)
        self.assertEqual(night1, '20150102')

    @unittest.skipUnless(os.path.exists(os.path.join(os.environ['HOME'],'.netrc')),"No ~/.netrc file detected.")
    @unittest.skipIf(True, "NERSC is out; download will fail")
    def test_download(self):
        """Test desiutil.io.download.
        """
        #
        # Test by downloading a single file.  This sidesteps any issues
        # with running multiprocessing within the unittest environment.
        #
        from ..io.meta import findfile
        from ..io.download import download
        filename = findfile('sky',expid=2,night='20150510',camera='b0',spectrograph=0)
        paths = download(filename)
        self.assertEqual(paths[0],filename)
        self.assertTrue(os.path.exists(paths[0]))
        #
        # Deliberately test a non-existent file.
        #
        filename = findfile('sky',expid=2,night='20150510',camera='b9',spectrograph=9)
        paths = download(filename)
        self.assertIsNone(paths[0])
        # self.assertFalse(os.path.exists(paths[0]))


def test_suite():
    """Allows testing of only this module with the command::

        python setup.py test -m <modulename>
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
