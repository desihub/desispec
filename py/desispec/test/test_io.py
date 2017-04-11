# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""Test desispec.io.
"""
from __future__ import absolute_import, division
# The line above will help with 2to3 support.
import unittest, os
from datetime import datetime, timedelta
from shutil import rmtree
from pkg_resources import resource_filename
import numpy as np
from astropy.io import fits
from astropy.table import Table
from ..frame import Frame

try:
    import sqlalchemy
    sqlalchemy_available = True
except ImportError:
    sqlalchemy_available = False

class TestIO(unittest.TestCase):
    """Test desiutil.io.
    """

    @classmethod
    def setUpClass(cls):
        """Create unique test filename in a subdirectory.
        """
        from uuid import uuid1
        cls.testfile = 'test-{uuid}/test-{uuid}.fits'.format(uuid=uuid1())
        cls.testyfile = 'test-{uuid}/test-{uuid}.yaml'.format(uuid=uuid1())
        cls.testbrfile = 'test-{uuid}/test-br-{uuid}.fits'.format(uuid=uuid1())
        cls.testDir = os.path.join(os.environ['HOME'],'desi_test_io')
        cls.origEnv = {'SPECPROD':None,
            "DESI_SPECTRO_DATA":None,
            "DESI_SPECTRO_REDUX":None}
        cls.testEnv = {'SPECPROD':'dailytest',
            "DESI_SPECTRO_DATA":os.path.join(cls.testDir,'spectro','data'),
            "DESI_SPECTRO_REDUX":os.path.join(cls.testDir,'spectro','redux')}
        for e in cls.origEnv:
            if e in os.environ:
                cls.origEnv[e] = os.environ[e]
            os.environ[e] = cls.testEnv[e]

    @classmethod
    def tearDownClass(cls):
        """Cleanup test files if they exist.
        """
        for testfile in [cls.testfile, cls.testyfile, cls.testbrfile]:
            if os.path.exists(testfile):
                os.remove(testfile)
                testpath = os.path.normpath(os.path.dirname(testfile))
                if testpath != '.':
                    os.removedirs(testpath)

        for e in cls.origEnv:
            if cls.origEnv[e] is None:
                del os.environ[e]
            else:
                os.environ[e] = cls.origEnv[e]

        if os.path.exists(cls.testDir):
            rmtree(cls.testDir)

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
        from ..io.frame import read_frame, write_frame
        from ..io.fibermap import empty_fibermap
        nspec, nwave, ndiag = 5, 10, 3
        flux = np.random.uniform(size=(nspec, nwave))
        ivar = np.random.uniform(size=(nspec, nwave))
        meta = dict(BLAT=0, FOO='abc', FIBERMIN=500)
        mask_int = np.zeros((nspec, nwave), dtype=int)
        mask_uint = np.zeros((nspec, nwave), dtype=np.uint32)
        wave = np.arange(nwave)
        R = np.random.uniform( size=(nspec, ndiag, nwave) )

        for mask in (mask_int, mask_uint):
            frx = Frame(wave, flux, ivar, mask, R, meta=meta)
            write_frame(self.testfile, frx)
            frame = read_frame(self.testfile)

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

        #- Test float32 on disk vs. float64 in memory
        for extname in ['FLUX', 'IVAR', 'WAVELENGTH', 'RESOLUTION']:
            data = fits.getdata(self.testfile, extname)
            self.assertEqual(data.dtype, np.dtype('>f4'), '{} not type >f4'.format(extname))

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
        frx = Frame(wave, flux, ivar, mask, R, fibermap=fibermap)
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
        fibermap = empty_fibermap(10)
        self.assertTrue(np.all(fibermap['FIBER'] == np.arange(10)))
        self.assertTrue(np.all(fibermap['SPECTROID'] == 0))
        fibermap = empty_fibermap(10, specmin=20)
        self.assertTrue(np.all(fibermap['FIBER'] == np.arange(10)+20))
        self.assertTrue(np.all(fibermap['SPECTROID'] == 0))
        fibermap = empty_fibermap(10, specmin=495)
        self.assertTrue(np.all(fibermap['FIBER'] == np.arange(10)+495))
        self.assertTrue(np.all(fibermap['SPECTROID'] == [0,0,0,0,0,1,1,1,1,1]))

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
            self.assertEqual(c1.dtype.kind, c2.dtype.kind)
            self.assertEqual(c1.dtype.itemsize, c2.dtype.itemsize)
            self.assertEqual(c1.shape, c2.shape)
            self.assertTrue(np.all(c1 == c2))

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

    def test_brick(self):
        """Test desispec.io.brick.Brick objects.
        """
        from ..io.brick import Brick
        from ..io.fibermap import empty_fibermap
        from ..io.util import fitsheader
        nspec = 5
        nwave = 10
        wave = np.arange(nwave)
        flux = np.random.uniform(size=(nspec, nwave))
        ivar = np.random.uniform(size=(nspec, nwave))
        resolution = np.random.uniform(size=(nspec, 5, nwave))
        fibermap = empty_fibermap(nspec)
        fibermap['TARGETID'] = 3*np.arange(nspec)
        night = '20101020'
        expid = 2
        header = dict(BRICKNAM = '0002p000', channel='b')
        brick = Brick(self.testfile, mode='update', header=header)
        brick.add_objects(flux, ivar, wave, resolution, fibermap, night, expid)
        brick.add_objects(flux, ivar, wave, resolution, fibermap, night, expid+1)

        #- check dtype consistency for columns in original fibermap
        brick_fibermap = Table(brick.hdu_list['FIBERMAP'].data)
        for colname in fibermap.colnames:
            self.assertEqual(fibermap[colname].dtype, brick_fibermap[colname].dtype)

        #- Check that the two extra columns exist (and only those)
        self.assertIn('NIGHT', brick_fibermap.colnames)
        self.assertIn('EXPID', brick_fibermap.colnames)
        self.assertEqual(len(fibermap.colnames)+2, len(brick_fibermap.colnames))

        brick.close()

        bx = Brick(self.testfile)
        self.assertTrue(np.all(bx.get_wavelength_grid() == wave))
        self.assertEqual(bx.get_num_targets(), nspec)
        self.assertEqual(bx.get_num_spectra(), 2*nspec)
        self.assertEqual(set(bx.get_target_ids()), set(fibermap['TARGETID']))
        flux2, ivar2, resolution2, info2 = bx.get_target(0)
        self.assertEqual(flux2.shape, (2,10))
        self.assertEqual(ivar2.shape, (2,10))
        self.assertEqual(resolution2.shape, (2,5,10))
        self.assertEqual(len(info2), 2)
        self.assertTrue( np.all(flux2[0] == flux[0]) )
        self.assertTrue( np.all(ivar2[0] == ivar[0]) )

        bx.close()

        #- test for incorrect fits file. Use self.testbrfile as this requires many spectra/wavelength.
        #- Using self.testfile breaks above tests for shapes assertion etc. So dealing differently
        nspec2 = 500
        nwave2 = 1000
        wave2 = np.arange(nwave2)
        flux2 = np.random.uniform(size=(nspec2, nwave2))
        ivar2 = np.random.uniform(size=(nspec2, nwave2))
        resolution2 = np.random.uniform(size=(nspec2, 5, nwave2))
        fibermap2 = empty_fibermap(nspec2)
        fibermap2['TARGETID'] = 3*np.arange(nspec2)
        night2 = '20161130'
        expid2 = 5
        header2 = dict(BRICKNAM = '0005p026', channel='r')

        brick2 = Brick(self.testbrfile, mode='update', header=header2)
        brick2.add_objects(flux2, ivar2, wave2, resolution2, fibermap2, night2, expid2)
        brick2.close()

        #- Now open before teardown and add a HDU. A corrupt file will throw IOError while opening
        trueflux=np.ones((nspec2,nwave2))*0.75
        header2 = fitsheader(header2)
        fx = fits.open(self.testbrfile, mode='append')
        self.assertEqual(len(fx),5)
        fx.append(fits.ImageHDU(trueflux, name='_TRUEFLUX', header=header2))
        fx.flush()
        self.assertEqual(len(fx),6)
        self.assertEqual(fx[5].header['EXTNAME'], '_TRUEFLUX')
        fx.close()


    def test_zbest_io(self):
        """Test reading and writing Zfind files.
        """
        from ..zfind import ZfindBase
        from ..io.zfind import read_zbest, write_zbest
        nspec, nflux = 10, 20
        wave = np.arange(nflux)
        flux = np.random.uniform(size=(nspec, nflux))
        ivar = np.random.uniform(size=(nspec, nflux))
        zfind1 = ZfindBase(wave, flux, ivar)

        zfind1.zwarn[:] = np.arange(nspec)
        zfind1.z[:] = np.random.uniform(size=nspec)
        zfind1.zerr[:] = np.random.uniform(size=nspec)
        zfind1.spectype[:] = 'ELG'

        brickname = '1234p567'
        targetids = np.random.randint(0,12345678, size=nspec)

        write_zbest(self.testfile, brickname, targetids, zfind1)
        zfind2 = read_zbest(self.testfile)

        self.assertTrue(np.all(zfind2.z == zfind1.z))
        self.assertTrue(np.all(zfind2.zerr == zfind1.zerr))
        self.assertTrue(np.all(zfind2.zwarn == zfind1.zwarn))
        self.assertTrue(np.all(zfind2.spectype == zfind1.spectype))
        self.assertTrue(np.all(zfind2.subtype == zfind1.subtype))
        self.assertTrue(np.all(zfind2.brickname == brickname))
        self.assertTrue(np.all(zfind2.targetid == targetids))

        write_zbest(self.testfile, brickname, targetids, zfind1, zspec=True)
        zfind3 = read_zbest(self.testfile)

        self.assertTrue(np.all(zfind3.wave == zfind1.wave))
        self.assertTrue(np.all(zfind3.flux == zfind1.flux.astype(np.float32)))
        self.assertTrue(np.all(zfind3.ivar == zfind1.ivar.astype(np.float32)))
        self.assertTrue(np.all(zfind3.model == zfind1.model))

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
        frame.meta = dict(CAMERA='b0', FLAVOR='dark', NIGHT='20160607', EXPID=1)
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
            foo = findfile('brick',brickname='3338p190')
        the_exception = cm.exception
        self.assertEqual(str(the_exception), "Required input 'band' is not set for type 'brick'!")

        #- Some findfile calls require $DESI_SPECTRO_DATA; others do not
        del os.environ['DESI_SPECTRO_DATA']
        x = findfile('brick', brickname='0000p123', band='r1')
        self.assertTrue(x is not None)
        with self.assertRaises(AssertionError):
            x = findfile('fibermap', night='20150101', expid=123)
        os.environ['DESI_SPECTRO_DATA'] = self.testEnv['DESI_SPECTRO_DATA']

        #- Some require $DESI_SPECTRO_REDUX; others to not
        del os.environ['DESI_SPECTRO_REDUX']
        x = findfile('fibermap', night='20150101', expid=123)
        self.assertTrue(x is not None)
        with self.assertRaises(AssertionError):
            x = findfile('brick', brickname='0000p123', band='r1')
        os.environ['DESI_SPECTRO_REDUX'] = self.testEnv['DESI_SPECTRO_REDUX']

    def test_findfile_outdir(self):
        """Test using desispec.io.meta.findfile with an output directory.
        """
        from ..io.meta import findfile
        outdir = '/blat/foo/bar'
        x = findfile('fibermap', night='20150101', expid=123, outdir=outdir)
        self.assertEqual(x, os.path.join(outdir, os.path.basename(x)))

    @unittest.skipUnless(os.path.exists(os.path.join(os.environ['HOME'],'.netrc')),"No ~/.netrc file detected.")
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

    @unittest.skipUnless(sqlalchemy_available, "sqlalchemy not installed; skipping DB tests")
    def test_database(self):
        """Test desispec.io.database.
        """
        from ..io.database import (utc, Base, FrameStatus, BrickStatus, Status,
                                   Night, ExposureFlavor)
        # self.assertIsNotNone(Base.metadata.tables)
        #
        # Simple ForeignKey tables.
        #
        st = Status(status='succeeded')
        self.assertEqual(str(st), "<Status(status='succeeded')>")
        ef = ExposureFlavor(flavor='science')
        self.assertEqual(str(ef), "<ExposureFlavor(flavor='science')>")
        ni = Night(night='20170101')
        self.assertEqual(str(ni), "<Night(night='20170101')>")
        #
        # Status tables.
        #
        fs = FrameStatus(id=1, frame_id=1, status='succeeded',
                         stamp=datetime(2017, 1, 1, 0, 0, 0, tzinfo=utc))
        self.assertEqual(str(fs), "<FrameStatus(id=1, frame_id=1, status='succeeded', stamp='2017-01-01 00:00:00+00:00')>")
        bs = BrickStatus(id=1, brick_id=1, status='succeeded',
                         stamp=datetime(2017, 1, 1, 0, 0, 0, tzinfo=utc))
        self.assertEqual(str(bs), "<BrickStatus(id=1, brick_id=1, status='succeeded', stamp='2017-01-01 00:00:00+00:00')>")

    # def test_quicksurvey(self):
    #     """Test desispec.io.quicksurvey.
    #     """
    #     from ..io.quicksurvey import utc, convert_dateobs
    #     ts = convert_dateobs('2019-01-03T01:11:33.247')
    #     self.assertEqual(ts.year, 2019)
    #     self.assertEqual(ts.month, 1)
    #     self.assertEqual(ts.microsecond, 247000)
    #     self.assertIsNone(ts.tzinfo)
    #     ts = convert_dateobs('2019-01-03T01:11:33.247', tzinfo=utc)
    #     self.assertEqual(ts.year, 2019)
    #     self.assertEqual(ts.month, 1)
    #     self.assertEqual(ts.microsecond, 247000)
    #     self.assertIs(ts.tzinfo, utc)


def test_suite():
    """Allows testing of only this module with the command::

        python setup.py test -m <modulename>
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
