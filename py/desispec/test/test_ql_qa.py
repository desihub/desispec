"""
tests for Quicklook QA class and functions. It also indludes tests on low level functions on desispec.qa.qalib
"""

import unittest
import shutil
import tempfile
import numpy as np
import os
from desispec.qa import qalib
from desispec.qa import qa_quicklook as QA
from pkg_resources import resource_filename
import desispec.sky
from desispec.preproc import parse_sec_keyword
from specter.psf import load_psf
import astropy.io.fits as fits
from desispec.quicklook import qllogger
import desispec.io
import desispec.image
from desitarget.targetmask import desi_mask

qlog=qllogger.QLLogger("QuickLook",0)
log=qlog.getlog()

def xy2hdr(xyslice):
    '''
    convert 2D slice into IRAF style [a:b,c:d] hdr value
    
    e.g. xyslice2hdr(np.s_[0:10, 5:20]) -> '[6:20,1:10]'
    '''
    yy, xx = xyslice
    value = '[{}:{},{}:{}]'.format(xx.start+1, xx.stop, yy.start+1, yy.stop)
    return value

#- 2D gaussian function to model sky peaks
def gaussian2D(x,y,amp,xmu,ymu,xsigma,ysigma):
    x,y = np.meshgrid(x,y)
    gauss = amp*np.exp(-(x-xmu)**2/(2*xsigma**2)-(y-ymu)**2/(2*ysigma**2))
    return gauss

class TestQL_QA(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Create test filenames in a unique temporary directory
        """
        cls.testDir = tempfile.mkdtemp()
        cls.rawfile = os.path.join(cls.testDir, 'test-raw-abcde.fits')
        cls.pixfile = os.path.join(cls.testDir, 'test-pix-abcde.fits')
        cls.xwfile = os.path.join(cls.testDir, 'test-xw-abcde.fits')
        cls.framefile = os.path.join(cls.testDir, 'test-frame-abcde.fits')
        cls.fibermapfile = os.path.join(cls.testDir, 'test-fibermap-abcde.fits')
        cls.skyfile = os.path.join(cls.testDir, 'test-sky-abcde.fits')
        cls.qafile = os.path.join(cls.testDir, 'test_qa.yaml')
        cls.qajson = os.path.join(cls.testDir, 'test_qa.json')
        cls.qafig = os.path.join(cls.testDir, 'test_qa.png')

    @classmethod
    def tearDownClass(cls):
        """Cleanup temporary directory
        """
        shutil.rmtree(cls.testDir)

    def tearDown(self):
        self.rawimage.close()
        for filename in [self.framefile, self.rawfile, self.pixfile, self.xwfile, self.fibermapfile, self.skyfile, self.qafile, self.qajson, self.qafig]:
            if os.path.exists(filename):
                os.remove(filename)

    #- Create some test data
    def setUp(self):
        #- use specter psf for this test
        self.psffile=resource_filename('specter', 'test/t/psf-monospot.fits') 
        #self.psffile=os.environ['DESIMODEL']+'/data/specpsf/psf-b.fits'
        self.config={"kwargs":{
            "refKey":None,
            "param":{},
            "qso_resid":None
            }}

        #- rawimage

        hdr = dict()
        hdr['CAMERA'] = 'z1'
        hdr['DATE-OBS'] = '2018-09-23T08:17:03.988'
        hdr['PROGRAM'] = 'dark'
        hdr['EXPTIME'] = 100

        #- Dimensions per amp
        ny = self.ny = 500
        nx = self.nx = 400
        noverscan = nover = 50

        hdr['BIASSECA'] = xy2hdr(np.s_[0:ny, nx:nx+nover])
        hdr['DATASECA'] = xy2hdr(np.s_[0:ny, 0:nx])
        hdr['CCDSECA'] = xy2hdr(np.s_[0:ny, 0:nx])
        
        hdr['BIASSECB'] = xy2hdr(np.s_[0:ny, nx+nover:nx+2*nover])
        hdr['DATASECB'] = xy2hdr(np.s_[0:ny, nx+2*nover:nx+2*nover+nx])
        hdr['CCDSECB'] =  xy2hdr(np.s_[0:ny, nx:nx+nx])

        hdr['BIASSECC'] = xy2hdr(np.s_[ny:ny+ny, nx:nx+nover])
        hdr['DATASECC'] = xy2hdr(np.s_[ny:ny+ny, 0:nx])
        hdr['CCDSECC'] = xy2hdr(np.s_[ny:ny+ny, 0:nx])

        hdr['BIASSECD'] = xy2hdr(np.s_[ny:ny+ny, nx+nover:nx+2*nover])
        hdr['DATASECD'] = xy2hdr(np.s_[ny:ny+ny, nx+2*nover:nx+2*nover+nx])
        hdr['CCDSECD'] =  xy2hdr(np.s_[ny:ny+ny, nx:nx+nx])
        
        hdr['NIGHT'] = '20180923'
        hdr['EXPID'] = 1
        hdr['PROGRAM'] = 'dark'
        hdr['FLAVOR'] = 'science'
        hdr['EXPTIME'] = 100.0
        
        rawimage = np.zeros((2*ny, 2*nx+2*noverscan))
        offset = {'A':100.0, 'B':100.5, 'C':50.3, 'D':200.4}
        gain = {'A':1.0, 'B':1.5, 'C':0.8, 'D':1.2}
        rdnoise = {'A':2.0, 'B':2.2, 'C':2.4, 'D':2.6}
        obsrdn = {'A':3.4, 'B':3.3, 'C':3.6, 'D':3.3}

        quad = {
            'A': np.s_[0:ny, 0:nx], 'B': np.s_[0:ny, nx:nx+nx],
            'C': np.s_[ny:ny+ny, 0:nx], 'D': np.s_[ny:ny+ny, nx:nx+nx],
        }
        
        for amp in ('A', 'B', 'C', 'D'):

            hdr['GAIN'+amp] = gain[amp]
            hdr['RDNOISE'+amp] = rdnoise[amp]
            hdr['OBSRDN'+amp] = obsrdn[amp]

            xy = parse_sec_keyword(hdr['BIASSEC'+amp])
            shape = [xy[0].stop-xy[0].start, xy[1].stop-xy[1].start]
            rawimage[xy] += offset[amp]
            rawimage[xy] += np.random.normal(scale=rdnoise[amp], size=shape)/gain[amp]
            xy = parse_sec_keyword(hdr['DATASEC'+amp])
            shape = [xy[0].stop-xy[0].start, xy[1].stop-xy[1].start]
            rawimage[xy] += offset[amp]
            rawimage[xy] += np.random.normal(scale=rdnoise[amp], size=shape)/gain[amp]

        #- set CCD parameters
        self.ccdsec1=hdr["CCDSECA"]
        self.ccdsec2=hdr["CCDSECB"]
        self.ccdsec3=hdr["CCDSECC"]
        self.ccdsec4=hdr["CCDSECD"]

        #- raw data are integers, not floats
        rawimg = rawimage.astype(np.int32)
        self.expid=hdr["EXPID"]
        self.camera=hdr["CAMERA"]
        #- Confirm that all regions were correctly offset
        assert not np.any(rawimage == 0.0)        

        #- write to the rawfile and read it in QA test
        hdr['DOSVER'] = 'SIM'
        hdr['FEEVER'] = 'SIM'
        hdr['DETECTOR'] = 'SIM'

        desispec.io.write_raw(self.rawfile,rawimg,hdr,camera=self.camera)
        self.rawimage=fits.open(self.rawfile)
        
        #- read psf, should use specter.PSF.load_psf instead of desispec.PSF(), otherwise need to create a psfboot somewhere.

        self.psf = load_psf(self.psffile)

        #- make the test pixfile, fibermap file
        img_pix = rawimg
        img_ivar = np.ones_like(img_pix) / 3.0**2
        img_mask = np.zeros(img_pix.shape, dtype=np.uint32)
        img_mask[200] = 1

        self.image = desispec.image.Image(img_pix, img_ivar, img_mask, camera='z1',meta=hdr)
        desispec.io.write_image(self.pixfile, self.image)

        #- Create a fibermap with purposefully overlapping targeting bits
        n = 30
        self.fibermap = desispec.io.empty_fibermap(n)
        self.fibermap['OBJTYPE'][:] = 'TGT'
        self.fibermap['DESI_TARGET'][::2] |= desi_mask.ELG
        self.fibermap['DESI_TARGET'][::5] |= desi_mask.QSO
        self.fibermap['DESI_TARGET'][::7] |= desi_mask.LRG

        #- add some arbitrary fluxes
        for key in ['FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_W2']:
            self.fibermap[key] = 10**((22.5 - np.random.uniform(18, 21, size=n))/2.5)

        #- Make some standards; these still have OBJTYPE = 'TGT'
        ii = [6,18,29]
        self.fibermap['DESI_TARGET'][ii] = desi_mask.STD_FAINT

        #- set some targets to SKY
        ii = self.skyfibers = [5,10,21]
        self.fibermap['OBJTYPE'][ii] = 'SKY'
        self.fibermap['DESI_TARGET'][ii] = desi_mask.SKY
        for key in ['FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_W2']:
            self.fibermap[key][ii] = np.random.normal(scale=100, size=len(ii))

        desispec.io.write_fibermap(self.fibermapfile, self.fibermap)

        #- make a test frame file
        self.night=hdr['NIGHT']
        self.nspec = nspec = 30
        wave=np.arange(7600.0,9800.0,1.0) #- z channel
        nwave = self.nwave = len(wave)
        flux=np.random.uniform(size=(nspec,nwave))+100.
        ivar=np.ones_like(flux)
        resolution_data=np.ones((nspec,13,nwave))
        self.frame=desispec.frame.Frame(wave,flux,ivar,resolution_data=resolution_data,fibermap=self.fibermap)
        self.frame.meta =  hdr
        self.frame.meta['WAVESTEP']=0.5
        desispec.io.write_frame(self.framefile, self.frame)

        #- make a skymodel
        sky=np.ones_like(self.frame.flux)*0.5
        skyivar=np.ones_like(sky)
        self.mask=np.zeros(sky.shape,dtype=np.uint32)
        self.skymodel=desispec.sky.SkyModel(wave,sky,skyivar,self.mask)
        self.skyfile=desispec.io.write_sky(self.skyfile,self.skymodel)
        
        #- Make a dummy boundary map for wavelength-flux in pixel space
        self.map2pix={}
        self.map2pix["LEFT_MAX_FIBER"] = 14
        self.map2pix["RIGHT_MIN_FIBER"] = 17
        self.map2pix["BOTTOM_MAX_WAVE_INDEX"] = 900
        self.map2pix["TOP_MIN_WAVE_INDEX"] = 1100

    #- test some qa utility functions:
    def test_ampregion(self):
        pixboundary=qalib.ampregion(self.image)
        self.assertEqual(pixboundary[0][1],slice(0,self.nx,None))
        self.assertEqual(pixboundary[3][0],slice(self.ny,self.ny+self.ny,None))

    def test_fiducialregion(self):
        leftmax,rightmin,bottommax,topmin=qalib.fiducialregion(self.frame,self.psf)
        self.assertEqual(leftmax,self.nspec-1)  #- as only 30 spectra defined 
        self.assertLess(bottommax,topmin)

    def test_getrms(self):
        img_rms=qalib.getrms(self.image.pix)
        self.assertEqual(img_rms,np.std(self.image.pix))

    def test_countpix(self):
        pix=self.image.pix
        counts1=qalib.countpix(pix,nsig=3) #- counts above 3 sigma
        counts2=qalib.countpix(pix,nsig=4) #- counts above 4 sigma
        self.assertLess(counts2,counts1)

# RS: remove this test because this QA isn't used
#    def test_sky_resid(self):
#        import copy
#        param = dict(
#                     PCHI_RESID=0.05,PER_RESID=95.,BIN_SZ=0.1)
#        qadict=qalib.sky_resid(param,self.frame,self.skymodel,quick_look=True)
#        kk=np.where(self.frame.fibermap['OBJTYPE']=='SKY')[0]
#        self.assertEqual(qadict['NSKY_FIB'],len(kk))
#
#        #- run with different sky flux
#        skym1=desispec.sky.SkyModel(self.frame.wave,self.skymodel.flux,self.skymodel.ivar,self.mask)
#        skym2=desispec.sky.SkyModel(self.frame.wave,self.skymodel.flux*0.5,self.skymodel.ivar,self.mask)
#        frame1=copy.deepcopy(self.frame)
#        frame2=copy.deepcopy(self.frame)
#        desispec.sky.subtract_sky(frame1,skym1)
#        desispec.sky.subtract_sky(frame2,skym2)
#
#        qa1=qalib.sky_resid(param,frame1,skym1)
#        qa2=qalib.sky_resid(param,frame2,skym2)
#        self.assertLess(qa1['RESID'],qa2['RESID']) #- residuals must be smaller for case 1

    def testSignalVsNoise(self):
        import copy
        params=None
        #- first get the sky subtracted frame
        #- copy frame not to override
        thisframe=copy.deepcopy(self.frame)
        desispec.sky.subtract_sky(thisframe,self.skymodel)
        qadict=qalib.SignalVsNoise(thisframe,params)
        #- make sure all the S/N is positive
        self.assertTrue(np.all(qadict['MEDIAN_SNR']) > 0)

        #- Reduce sky
        skym1=desispec.sky.SkyModel(self.frame.wave,self.skymodel.flux,self.skymodel.ivar,self.mask)
        skym2=desispec.sky.SkyModel(self.frame.wave,self.skymodel.flux*0.5,self.skymodel.ivar,self.mask)
        frame1=copy.deepcopy(self.frame)
        frame2=copy.deepcopy(self.frame)
        desispec.sky.subtract_sky(frame1,skym1)
        desispec.sky.subtract_sky(frame2,skym2)
        qa1=qalib.SignalVsNoise(frame1,params)
        qa2=qalib.SignalVsNoise(frame2,params)
        self.assertTrue(np.all(qa2['MEDIAN_SNR'] > qa1['MEDIAN_SNR']))

        #- test for tracer not present
        nullfibermap=desispec.io.empty_fibermap(10)
        qa=qalib.SignalVsNoise(self.frame,params)

        self.assertEqual(len(qa['MEDIAN_SNR']),30)

    #- Test each individual QA:
    def testBiasOverscan(self):
        return
        qa=QA.Bias_From_Overscan('bias',self.config) #- initialize with fake config and name
        inp=self.rawimage
        qargs={}
        qargs["RESULTKEY"] = 'BIAS_AMP'
        qargs["PSFFile"]=self.psf
        qargs["camera"]=self.camera
        qargs["expid"]=self.expid
        qargs["amps"]=True
        qargs["qafile"]=self.qafile
        qargs["qafig"]=self.qafig
        qargs["paname"]="abc"
        qargs["singleqa"]=None
        res1=qa(inp,**qargs)
        self.assertEqual(len(res1['METRICS']['BIAS_AMP']),4)

    def testGetRMS(self):
        return
        qa=QA.Get_RMS('rms',self.config)
        inp=self.image
        qargs={}
        qargs["RESULTKEY"] = 'NOISE_AMP'
        qargs["PSFFile"]=self.psf
        qargs["camera"]=self.camera
        qargs["expid"]=self.expid
        qargs["amps"]=True
        qargs["paname"]="abc"
        qargs["qafile"]=self.qafile
        qargs["qafig"]=self.qafig
        qargs["singleqa"]=None
        qargs["param"]={'PERCENTILES': [68.2,95.4,99.7], 'NOISE_AMP_NORMAL_RANGE': [-1.0, 1.0], 'NOISE_AMP_WARN_RANGE': [-2.0, 2.0]}
        resl=qa(inp,**qargs)
        self.assertTrue("yaml" in qargs["qafile"])
        self.assertTrue("png" in qargs["qafig"])
        self.assertTrue(len(resl['METRICS']['NOISE_AMP'])==4)
        self.assertTrue((np.all(resl['METRICS']['NOISE_AMP'])>0))

    def testCalcXWSigma(self):
        return
        #- Create another pix file for xwsigma test
        xw_hdr = dict()
        xw_hdr['CAMERA'] = self.camera
        xw_hdr['NIGHT'] = self.night
        xw_hdr['EXPID'] = self.expid
        xw_hdr['PROGRAM'] = 'dark'
        xw_hdr['FLAVOR'] = 'science'

        xw_ny = 2000
        xw_nx = 2000
        xw_rawimage = np.zeros((2*xw_ny,2*xw_nx))
        xw_img_pix = xw_rawimage.astype(np.int32)
        xw_img_ivar = np.ones_like(xw_img_pix)/3.0**2
        xw_img_mask = np.zeros(xw_img_pix.shape,dtype=np.uint32)

        #- manually insert gaussian sky peaks
        x = np.arange(7)
        y = np.arange(7)
        a = 10000.
        xmu = np.mean(x)
        ymu = np.mean(y)
        xsigma = 1.0
        ysigma = 1.0
        peak_counts = np.rint(gaussian2D(x,y,a,xmu,ymu,xsigma,ysigma))
        peak_counts = peak_counts.astype(np.int32)
        zpeaks = np.array([8401.5,8432.4,8467.5,9479.4])
        fibers = np.arange(30)
        for i in range(len(zpeaks)):
            pix = np.rint(self.psf.xy(fibers,zpeaks[i]))
            for j in range(len(fibers)):
                for k in range(len(peak_counts)):
                    ypix = int(pix[0][j]-3+k)
                    xpix_start =int(pix[1][j]-3)
                    xpix_stop = int(pix[1][j]+4)
                    xw_img_pix[ypix][xpix_start:xpix_stop] = peak_counts[k]

        #- transpose pixel values to correct place in image
        xw_img_pix=np.ndarray.transpose(xw_img_pix)

        #- write the test pixfile, fibermap file
        xwimage = desispec.image.Image(xw_img_pix, xw_img_ivar, xw_img_mask, camera='z1',meta=xw_hdr)
        desispec.io.write_image(self.xwfile, xwimage)

        qa=QA.Calc_XWSigma('xwsigma',self.config)
        inp=xwimage
        qargs={}
        qargs["RESULTKEY"] = 'XWSIGMA'
        qargs["Flavor"]='science'
        qargs["PSFFile"]=self.psffile
        qargs["FiberMap"]=self.fibermap
        qargs["camera"]=self.camera
        qargs["expid"]=self.expid
        qargs["amps"]=False
        qargs["paname"]="abc"
        qargs["qafile"]=self.qafile
        qargs["qafig"]=self.qafig
        qargs["singleqa"]=None
        
        qargs["param"]={'B_PEAKS': [3914.4, 5199.3, 5578.9],'R_PEAKS': [6301.9, 6365.4, 7318.2, 7342.8, 7371.3],'Z_PEAKS': [8401.5, 8432.4, 8467.5, 9479.4],'PIXEL_RANGE': 7,'XWSIGMA_NORMAL_RANGE': [-2.0, 2.0],'XWSIGMA_WARN_RANGE': [-4.0, 4.0]}
        resl=qa(inp,**qargs)
        self.assertTrue(len(resl["METRICS"]["XWSIGMA"].ravel())==2)
        self.assertTrue("yaml" in qargs["qafile"])
        self.assertTrue("png" in qargs["qafig"])
        self.assertTrue(len(resl['METRICS']['XWSIGMA'])==4)
        self.assertTrue((np.all(resl['METRICS']['XWSIGMA'])>0))
        
    def testCountPixels(self):
        return
        qa=QA.Count_Pixels('countpix',self.config)
        inp=self.image
        qargs={}
        qargs["RESULTKEY"] = 'LITFRAC_AMP'
        qargs["PSFFile"]=self.psf
        qargs["camera"]=self.camera
        qargs["expid"]=self.expid
        qargs["amps"]=False
        qargs["paname"]="abc"
        qargs["singleqa"]=None
        qargs["param"]={'CUTPIX': 5, 'LITFRAC_NORMAL_RANGE': [-0.1, 0.1], 'LITFRAC_WARN_RANGE': [-0.2, 0.2]}
        resl=qa(inp,**qargs)
        #- test if amp QAs exist
        qargs["amps"] = True
        resl2=qa(inp,**qargs)
        self.assertTrue(len(resl2['METRICS']['LITFRAC_AMP'])==4)

    def testCountSpectralBins(self):
        return
        qa=QA.CountSpectralBins('countbins',self.config)
        inp=self.frame
        qargs={}
        qargs["RESULTKEY"] = 'NGOODFIB'
        qargs["PSFFile"]=self.psf
        qargs["FiberMap"]=self.fibermap
        qargs["camera"]=self.camera
        qargs["expid"]=self.expid
        qargs["amps"]=True
        qargs["paname"]="abc"
        qargs["qafile"]=self.qafile
        qargs["qafig"]=None
        qargs["singleqa"]=None
        qargs["param"]={'CUTBINS': 5, 'N_KNOWN_BROKEN_FIBERS': 0, 'NGOODFIB_NORMAL_RANGE': [-5, 5], 'NGOODFIB_WARN_RANGE': [-10, 10]}
        resl=qa(inp,**qargs)
        self.assertTrue(resl["METRICS"]["GOOD_FIBERS"].shape[0]==inp.nspec)
        self.assertTrue((resl["METRICS"]["NGOODFIB"])<=inp.nspec)

    def testSkyCont(self):
        return
        qa=QA.Sky_Continuum('skycont',self.config)
        inp=self.frame
        qargs={}
        qargs["RESULTKEY"] = 'SKYCONT'
        qargs["FiberMap"]=self.fibermap
        qargs["camera"]=self.camera
        qargs["expid"]=self.expid
        qargs["paname"]="abc"
        qargs["singleqa"]=None
        qargs["param"]={'B_CONT': ["4000, 4500", "5250, 5550"],'R_CONT': ["5950, 6200", "6990, 7230"],'Z_CONT': ["8120, 8270", "9110, 9280"]}
        resl=qa(inp,**qargs)
        self.assertTrue(resl["METRICS"]["SKYFIBERID"]==self.skyfibers) #- as defined in the fibermap
        self.assertTrue(resl["METRICS"]["SKYCONT"]>0)
        
    def testSkyPeaks(self):
        return
        qa=QA.Sky_Peaks('skypeaks',self.config)
        inp=self.frame
        qargs={}
        qargs["RESULTKEY"] = 'PEAKCOUNT'
        qargs["FiberMap"]=self.fibermap
        qargs["camera"]=self.camera
        qargs["expid"]=self.expid
        qargs["paname"]="abc"
        qargs["dict_countbins"]=self.map2pix
        qargs["singleqa"]=None
        qargs["param"]={'B_PEAKS': [3914.4, 5199.3, 5201.8],'R_PEAKS': [6301.9, 6365.4, 7318.2, 7342.8, 7371.3],'Z_PEAKS': [8401.5, 8432.4, 8467.5, 9479.4, 9505.6, 9521.8],'PEAKCOUNT_NORMAL_RANGE': [-1.0, 1.0],'PEAKCOUNT_WARN_RANGE': [-2.0, 2.0]}
        resl=qa(inp,**qargs)
        
        #self.assertTrue(np.all(resl['METRICS']['PEAKCOUNT_RMS_AMP'])>=0.)
        self.assertTrue(resl['METRICS']['PEAKCOUNT_NOISE']>0)

    def testIntegrateSpec(self):
        return
        qa=QA.Integrate_Spec('integ',self.config)
        inp=self.frame
        qargs={}
        qargs["RESULTKEY"] = 'DELTAMAG_TGT'
        qargs["PSFFile"]=self.psf
        qargs["FiberMap"]=self.fibermap
        qargs["camera"]=self.camera
        qargs["expid"]=self.expid
        qargs["paname"]="abc"
        qargs["dict_countbins"]=self.map2pix
        qargs["singleqa"]=None
        qargs["param"]={'DELTAMAG_TGT_NORMAL_RANGE': [-2., 2.0], 'DELTAMAG_TGT_WARN_RANGE': [-4., 4.]}
        resl=qa(inp,**qargs)
        self.assertTrue(len(resl["METRICS"]["STD_FIBERID"])>0)

# RS: We are not using this QA anymore, so we don't need this test        
#    def testSkyResidual(self):
#        qa=QA.Sky_Residual('skyresid',self.config)
#        inp=self.frame
#        sky=self.skymodel
#        qargs={}
#        qargs["PSFFile"]=self.psf
#        qargs["FiberMap"]=self.fibermap
#        qargs["camera"]=self.camera
#        qargs["expid"]=self.expid
#        qargs["paname"]="abc"
#        qargs["dict_countbins"]=self.map2pix
#        qargs["singleqa"]=None
#        qargs["param"]={"BIN_SZ":0.2, "PCHI_RESID":0.05, "PER_RESID":95., "SKYRESID_NORMAL_RANGE":[-5.0, 5.0], "SKYRESID_WARN_RANGE":[-10.0, 10.0]}
#
#        resl=qa(inp,sky,**qargs)
#        
#        #self.assertTrue(resl["METRICS"]["NREJ"]==self.skymodel.nrej)
#        #self.assertTrue(len(resl["METRICS"]["MED_RESID_WAVE"]) == self.nwave)
#        #self.assertTrue(len(resl["METRICS"]["MED_RESID_FIBER"]) == 5) #- 5 sky fibers in the input
#        #self.assertTrue(resl["PARAMS"]["BIN_SZ"] == 0.1)
#        ##- test with different parameter set:
#        #resl2=qa(inp,sky,**qargs)
#        #self.assertTrue(len(resl["METRICS"]["DEVS_1D"])>len(resl2["METRICS"]["DEVS_1D"])) #- larger histogram bin size than default 0.1

    def testCalculateSNR(self):
        return
        qa=QA.Calculate_SNR('snr',self.config)
        inp=self.frame
        qargs={}
        qargs["RESULTKEY"] = 'FIDSNR'
        qargs["PSFFile"]=self.psf
        qargs["FiberMap"]=self.fibermap
        qargs["camera"]=self.camera
        qargs["expid"]=self.expid
        qargs["paname"]="abc"
        qargs["qafile"]=self.qafile #- no LRG by construction.
        qargs["dict_countbins"]=self.map2pix
        qargs["singleqa"]=None
        qargs["param"]={'RESIDUAL_CUT': 0.2, 'SIGMA_CUT': 2.0, 'FIDSNR_TGT_NORMAL_RANGE': [-11., 11.], 'FIDSNR_TGT_WARN_RANGE': [-12., 12.], 'FIDMAG': 22.}
        resl=qa(inp,**qargs)
        self.assertTrue("yaml" in qargs["qafile"])
        self.assertTrue(len(resl["METRICS"]["MEDIAN_SNR"])==self.nspec) #- positive definite


def test_suite():
    """Allows testing of only this module with the command::

        python setup.py test -m <modulename>
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)

if __name__ == '__main__':
    unittest.main()

