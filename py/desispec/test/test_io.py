import unittest, os
from uuid import uuid1

import numpy as np

from desispec.frame import Frame
from desispec.fiberflat import FiberFlat
from desispec.sky import SkyModel
from desispec.qa import QA_Frame
from desispec.image import Image
import desispec.io
import desispec.io.qa as desio_qa
from astropy.io import fits
from astropy.table import Table
from shutil import rmtree

class TestIO(unittest.TestCase):

    #- Create unique test filename in a subdirectory
    @classmethod
    def setUpClass(cls):
        cls.testfile = 'test-{uuid}/test-{uuid}.fits'.format(uuid=uuid1())
        cls.testyfile = 'test-{uuid}/test-{uuid}.yaml'.format(uuid=uuid1())
        cls.testDir = os.path.join(os.environ['HOME'],'desi_test_io')
        cls.origEnv = {'PRODNAME':None,
            "DESI_SPECTRO_DATA":None,
            "DESI_SPECTRO_REDUX":None}
        cls.testEnv = {'PRODNAME':'dailytest',
            "DESI_SPECTRO_DATA":os.path.join(cls.testDir,'spectro','data'),
            "DESI_SPECTRO_REDUX":os.path.join(cls.testDir,'spectro','redux')}
        for e in cls.origEnv:
            if e in os.environ:
                cls.origEnv[e] = os.environ[e]
            os.environ[e] = cls.testEnv[e]

    #- Cleanup test files if they exist
    @classmethod
    def tearDownClass(cls):
        for testfile in [cls.testfile, cls.testyfile]:
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
        #- None is ok; just returns blank Header
        header = desispec.io.util.fitsheader(None)
        self.assertTrue(isinstance(header, fits.Header))
        self.assertEqual(len(header), 0)

        #- input is dict
        hdr = dict()
        hdr['BLAT'] = 'foo'
        hdr['BAR'] = (1, 'biz bat')
        header = desispec.io.util.fitsheader(hdr)
        self.assertTrue(isinstance(header, fits.Header))
        self.assertEqual(header['BLAT'], 'foo')
        self.assertEqual(header['BAR'], 1)
        self.assertEqual(header.comments['BAR'], 'biz bat')

        #- input header as a list, get a fits.Header back
        hdr = list()
        hdr.append( ('BLAT', 'foo') )
        hdr.append( ('BAR', (1, 'biz bat')) )
        header = desispec.io.util.fitsheader(hdr)
        self.assertTrue(isinstance(header, fits.Header))
        self.assertEqual(header['BLAT'], 'foo')
        self.assertEqual(header['BAR'], 1)
        self.assertEqual(header.comments['BAR'], 'biz bat')

        #- fits.Header -> fits.Header
        header = desispec.io.util.fitsheader(header)
        self.assertTrue(isinstance(header, fits.Header))
        self.assertEqual(header['BLAT'], 'foo')
        self.assertEqual(header['BAR'], 1)
        self.assertEqual(header.comments['BAR'], 'biz bat')

        #- Can't convert and int into a fits Header
        self.assertRaises(ValueError, desispec.io.util.fitsheader, (1,))

    def test_frame_rw(self):
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
            desispec.io.write_frame(self.testfile, frx)
            frame = desispec.io.read_frame(self.testfile)

            self.assertTrue(np.all(flux == frame.flux))
            self.assertTrue(np.all(ivar == frame.ivar))
            self.assertTrue(np.all(wave == frame.wave))
            self.assertTrue(np.all(mask == frame.mask))
            self.assertTrue(np.all(R == frame.resolution_data))
            self.assertTrue(frame.resolution_data.dtype.isnative)
            self.assertEqual(frame.meta['BLAT'], meta['BLAT'])
            self.assertEqual(frame.meta['FOO'], meta['FOO'])
            
        #- with and without fibermap
        self.assertEqual(frame.fibermap, None)
        fibermap = desispec.io.empty_fibermap(nspec)
        fibermap['TARGETID'] = np.arange(nspec)*2
        frx = Frame(wave, flux, ivar, mask, R, fibermap=fibermap)
        desispec.io.write_frame(self.testfile, frx)
        frame = desispec.io.read_frame(self.testfile)
        for name in fibermap.dtype.names:
            match = np.all(fibermap[name] == frame.fibermap[name])
            self.assertTrue(match, 'Fibermap column {} mismatch'.format(name))

    def test_sky_rw(self):
        nspec, nwave = 5,10
        wave = np.arange(nwave)
        flux = np.random.uniform(size=(nspec, nwave))
        ivar = np.random.uniform(size=(nspec, nwave))
        mask_int = np.zeros(shape=(nspec, nwave), dtype=int)
        mask_uint = np.zeros(shape=(nspec, nwave), dtype=np.uint32)

        for mask in (mask_int, mask_uint):
            # skyflux,skyivar,skymask,cskyflux,cskyivar,wave
            sky = SkyModel(wave, flux, ivar, mask)
            desispec.io.write_sky(self.testfile, sky)
            xsky = desispec.io.read_sky(self.testfile)

            self.assertTrue(np.all(sky.wave  == xsky.wave))
            self.assertTrue(np.all(sky.flux  == xsky.flux))
            self.assertTrue(np.all(sky.ivar  == xsky.ivar))
            self.assertTrue(np.all(sky.mask  == xsky.mask))
            self.assertTrue(xsky.flux.dtype.isnative)
            self.assertEqual(sky.mask.dtype, xsky.mask.dtype)

    # fiberflat,fiberflat_ivar,fiberflat_mask,mean_spectrum,wave
    def test_fiberflat_rw(self):
        nspec, nwave, ndiag = 10, 20, 3
        flat = np.random.uniform(size=(nspec, nwave))
        ivar = np.random.uniform(size=(nspec, nwave))
        mask = np.zeros(shape=(nspec, nwave), dtype=int)
        meanspec = np.random.uniform(size=(nwave,))
        wave = np.arange(nwave)

        ff = FiberFlat(wave, flat, ivar, mask, meanspec)

        desispec.io.write_fiberflat(self.testfile, ff)
        xff = desispec.io.read_fiberflat(self.testfile)

        self.assertTrue(np.all(ff.fiberflat == xff.fiberflat))
        self.assertTrue(np.all(ff.ivar == xff.ivar))
        self.assertTrue(np.all(ff.mask == xff.mask))
        self.assertTrue(np.all(ff.meanspec == xff.meanspec))
        self.assertTrue(np.all(ff.wave == xff.wave))

        self.assertTrue(xff.fiberflat.dtype.isnative)
        self.assertTrue(xff.ivar.dtype.isnative)
        self.assertTrue(xff.mask.dtype.isnative)
        self.assertTrue(xff.meanspec.dtype.isnative)
        self.assertTrue(xff.wave.dtype.isnative)

    def test_empty_fibermap(self):
        fibermap = desispec.io.fibermap.empty_fibermap(10)
        self.assertTrue(np.all(fibermap['FIBER'] == np.arange(10)))
        self.assertTrue(np.all(fibermap['SPECTROID'] == 0))
        fibermap = desispec.io.fibermap.empty_fibermap(10, specmin=20)
        self.assertTrue(np.all(fibermap['FIBER'] == np.arange(10)+20))
        self.assertTrue(np.all(fibermap['SPECTROID'] == 0))
        fibermap = desispec.io.fibermap.empty_fibermap(10, specmin=495)
        self.assertTrue(np.all(fibermap['FIBER'] == np.arange(10)+495))
        self.assertTrue(np.all(fibermap['SPECTROID'] == [0,0,0,0,0,1,1,1,1,1]))

    def test_fibermap_rw(self):
        fibermap = desispec.io.fibermap.empty_fibermap(10)
        for key in fibermap.dtype.names:
            column = fibermap[key]
            fibermap[key] = np.random.random(column.shape).astype(column.dtype)

        desispec.io.write_fibermap(self.testfile, fibermap)

        #- Read without and with header
        fm = desispec.io.read_fibermap(self.testfile)
        self.assertTrue(isinstance(fm, np.ndarray))

        fm, hdr = desispec.io.read_fibermap(self.testfile, header=True)
        self.assertTrue(isinstance(fm, np.ndarray))
        self.assertTrue(isinstance(hdr, fits.Header))

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
        desispec.io.write_stdstar_models(self.testfile, flux, wave, fibers, data)
        
        fx, wx, fibx = desispec.io.read_stdstar_models(self.testfile)
        self.assertTrue(np.all(fx == flux))
        self.assertTrue(np.all(wx == wave))
        self.assertTrue(np.all(fibx == fibers))

    def test_fluxcalib(self):
        from desispec.fluxcalibration import FluxCalib
        nspec = 5
        nwave = 10
        wave = np.arange(nwave)
        calib = np.random.uniform(size=(nspec, nwave))
        ivar = np.random.uniform(size=(nspec, nwave))
        mask = np.random.uniform(0, 2, size=(nspec, nwave)).astype('i4')
        
        fc = FluxCalib(wave, calib, ivar, mask)
        desispec.io.write_flux_calibration(self.testfile, fc)
        fx = desispec.io.read_flux_calibration(self.testfile)
        self.assertTrue(np.all(fx.wave == fc.wave))
        self.assertTrue(np.all(fx.calib == fc.calib))
        self.assertTrue(np.all(fx.ivar == fc.ivar))
        self.assertTrue(np.all(fx.mask == fc.mask))

    def test_brick(self):
        from desispec.io.brick import Brick
        nspec = 5
        nwave = 10
        wave = np.arange(nwave)
        flux = np.random.uniform(size=(nspec, nwave))
        ivar = np.random.uniform(size=(nspec, nwave))
        resolution = np.random.uniform(size=(nspec, 5, nwave))
        fibermap = desispec.io.fibermap.empty_fibermap(nspec)
        fibermap['TARGETID'] = 3*np.arange(nspec)
        night = '20101020'
        expid = 2
        header = dict(BRICKNAM = '0002p000', channel='b')
        brick = Brick(self.testfile, mode='update', header=header)
        brick.add_objects(flux, ivar, wave, resolution, fibermap, night, expid)
        brick.add_objects(flux, ivar, wave, resolution, fibermap, night, expid+1)
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

    def test_image_rw(self):
        shape = (5,5)
        pix = np.random.uniform(size=shape)
        ivar = np.random.uniform(size=shape)
        mask = np.random.randint(0, 3, size=shape)
        img1 = Image(pix, ivar, mask, readnoise=1.0, camera='b0')
        desispec.io.write_image(self.testfile, img1)
        img2 = desispec.io.read_image(self.testfile)

        #- Check output datatypes
        self.assertEqual(img2.pix.dtype, np.float64)
        self.assertEqual(img2.ivar.dtype, np.float64)
        self.assertEqual(img2.mask.dtype, np.uint16)

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
        self.assertEqual(img2.mask.dtype, np.uint16)

        #- should work with various kinds of metadata header input
        meta = dict(BLAT='foo', BAR='quat', BIZ=1.0)
        img1 = Image(pix, ivar, mask, readnoise=1.0, camera='b0', meta=meta)
        desispec.io.write_image(self.testfile, img1)
        img2 = desispec.io.read_image(self.testfile)
        for key in meta:
            self.assertEqual(meta[key], img2.meta[key], 'meta[{}] not propagated'.format(key))

        #- img2 has meta as a FITS header instead of a dictionary;
        #- confirm that works too
        desispec.io.write_image(self.testfile, img2)
        img3 = desispec.io.read_image(self.testfile)
        for key in meta:
            self.assertEqual(meta[key], img3.meta[key], 'meta[{}] not propagated'.format(key))

    def test_io_qa_frame(self):        
        #- Init 
        qaframe = QA_Frame(flavor='dark')
        qaframe.init_skysub()
        # Write
        desio_qa.write_qa_frame(self.testyfile, qaframe)
        # Read
        xqaframe = desio_qa.read_qa_frame(self.testyfile)
        # Check
        self.assertTrue(qaframe.data['SKYSUB']['PARAM']['PCHI_RESID'] == xqaframe.data['SKYSUB']['PARAM']['PCHI_RESID'])
        self.assertTrue(qaframe.flavor == xqaframe.flavor)

    def test_native_endian(self):
        for dtype in ('>f8', '<f8', '<f4', '>f4', '>i4', '<i4', '>i8', '<i8'):
            data1 = np.arange(100).astype(dtype)
            data2 = desispec.io.util.native_endian(data1)
            self.assertTrue(data2.dtype.isnative, dtype+' is not native endian')
            self.assertTrue(np.all(data1 == data2))

    def test_findfile(self):
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
                    kwargs['camera'] = 'sp{spectrograph:d}'.format(**kwargs)
                filenames1.append(desispec.io.findfile(i,**kwargs))
                filenames2.append(os.path.join(os.environ['DESI_SPECTRO_REDUX'],
                    os.environ['PRODNAME'],'exposures',kwargs['night'],
                    '{expid:08d}'.format(**kwargs),
                    '{i}-{camera}-{expid:08d}.fits'.format(i=i,camera=kwargs['camera'],expid=kwargs['expid'])))
        for k,f in enumerate(filenames1):
            self.assertEqual(os.path.basename(filenames1[k]),
                os.path.basename(filenames2[k]))
            self.assertEqual(filenames1[k],filenames2[k])
            self.assertEqual(desispec.io.filepath2url(filenames1[k]),
                os.path.join('https://portal.nersc.gov/project/desi',
                'collab','spectro','redux',os.environ['PRODNAME'],'exposures',
                kwargs['night'],'{expid:08d}'.format(**kwargs),
                os.path.basename(filenames2[k])))
        #
        # Make sure that all required inputs are set.
        #
        with self.assertRaises(ValueError) as cm:
            foo = desispec.io.findfile('stdstars',expid=2,spectrograph=0)
        the_exception = cm.exception
        self.assertEqual(the_exception.message, "Required input 'night' is not set for type 'stdstars'!")
        with self.assertRaises(ValueError) as cm:
            foo = desispec.io.findfile('brick',brickname='3338p190')
        the_exception = cm.exception
        self.assertEqual(the_exception.message, "Required input 'band' is not set for type 'brick'!")

        #- Some findfile calls require $DESI_SPECTRO_DATA; others do not
        del os.environ['DESI_SPECTRO_DATA']
        x = desispec.io.findfile('brick', brickname='0000p123', band='r1')
        self.assertTrue(x is not None)
        with self.assertRaises(AssertionError):
            x = desispec.io.findfile('fibermap', night='20150101', expid=123)
        os.environ['DESI_SPECTRO_DATA'] = self.testEnv['DESI_SPECTRO_DATA']

        #- Some require $DESI_SPECTRO_REDUX; others to not
        del os.environ['DESI_SPECTRO_REDUX']
        x = desispec.io.findfile('fibermap', night='20150101', expid=123)
        self.assertTrue(x is not None)
        with self.assertRaises(AssertionError):
            x = desispec.io.findfile('brick', brickname='0000p123', band='r1')
        os.environ['DESI_SPECTRO_REDUX'] = self.testEnv['DESI_SPECTRO_REDUX']
            
    def test_findfile_outdir(self):
        outdir = '/blat/foo/bar'
        x = desispec.io.findfile('fibermap', night='20150101', expid=123, outdir=outdir)
        self.assertEqual(x, os.path.join(outdir, os.path.basename(x)))

    @unittest.skipUnless(os.path.exists(os.path.join(os.environ['HOME'],'.netrc')),"No ~/.netrc file detected.")
    def test_download(self):
        #
        # Test by downloading a single file.  This sidesteps any issues
        # with running multiprocessing within the unittest environment.
        #
        filename = desispec.io.findfile('sky',expid=2,night='20150510',camera='b0',spectrograph=0)
        paths = desispec.io.download(filename)
        self.assertEqual(paths[0],filename)
        self.assertTrue(os.path.exists(paths[0]))
        #
        # Deliberately test a non-existent file.
        #
        filename = desispec.io.findfile('sky',expid=2,night='20150510',camera='b9',spectrograph=9)
        paths = desispec.io.download(filename)
        self.assertIsNone(paths[0])
        # self.assertFalse(os.path.exists(paths[0]))

    def test_memcrc(self):
        test_strings = ('The quick brown fox jumped over the lazy dog.',
            "The sixth sick sheik's sixth sheep's sick.",
            'Jackdaws love my big sphinx of quartz.')
        test_results = (2142034932,2348585565,358631216)
        for k,t in enumerate(test_strings):
            self.assertEqual(desispec.io.memcrc(t),test_results[k])

#- This runs all test* functions in any TestCase class in this file
if __name__ == '__main__':
    unittest.main()
