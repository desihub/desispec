"""
tests for Quicklook QA class and functions. It also indludes tests on low level functions on desispec.qa.qalib
"""

import unittest
import numpy as np
import os
from desispec.qa import qalib
from desispec.qa import qa_quicklook as QA
from pkg_resources import resource_filename
import desispec.sky
from desispec.preproc import _parse_sec_keyword
from specter.psf import load_psf
import astropy.io.fits as fits
from desispec.quicklook import qllogger
import desispec.io
import desispec.image

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

    def tearDown(self):
        self.rawimage.close()
        for filename in [self.framefile, self.rawfile, self.pixfile, self.xwfile, self.fibermapfile, self.skyfile, self.qafile, self.qafig]:
            if os.path.exists(filename):
                os.remove(filename)

    #- Create some test data
    def setUp(self):

        self.rawfile = 'test-raw-abcd.fits'
        self.pixfile = 'test-pix-abcd.fits'
        self.xwfile = 'test-xw-abcd.fits'
        self.framefile = 'test-frame-abcd.fits'
        self.fibermapfile = 'test-fibermap-abcd.fits'
        self.skyfile = 'test-sky-abcd.fits'
        self.qafile = 'test_qa.yaml'
        self.qafig = 'test_qa.png'

        #- use specter psf for this test
        self.psffile=resource_filename('specter', 'test/t/psf-monospot.fits') 
        #self.psffile=os.environ['DESIMODEL']+'/data/specpsf/psf-b.fits'
        self.config={"kwargs":{
            "refKey":{},
            "param":{}
        }
        }

        #- rawimage

        hdr = dict()
        hdr['CAMERA'] = 'z1'
        hdr['DATE-OBS'] = '2018-09-23T08:17:03.988'

        #- Dimensions per amp
        ny = self.ny = 500
        nx = self.nx = 400
        noverscan = nover = 50

        hdr['BIASSEC1'] = xy2hdr(np.s_[0:ny, nx:nx+nover])
        hdr['DATASEC1'] = xy2hdr(np.s_[0:ny, 0:nx])
        hdr['CCDSEC1'] = xy2hdr(np.s_[0:ny, 0:nx])
        
        hdr['BIASSEC2'] = xy2hdr(np.s_[0:ny, nx+nover:nx+2*nover])
        hdr['DATASEC2'] = xy2hdr(np.s_[0:ny, nx+2*nover:nx+2*nover+nx])
        hdr['CCDSEC2'] =  xy2hdr(np.s_[0:ny, nx:nx+nx])

        hdr['BIASSEC3'] = xy2hdr(np.s_[ny:ny+ny, nx:nx+nover])
        hdr['DATASEC3'] = xy2hdr(np.s_[ny:ny+ny, 0:nx])
        hdr['CCDSEC3'] = xy2hdr(np.s_[ny:ny+ny, 0:nx])

        hdr['BIASSEC4'] = xy2hdr(np.s_[ny:ny+ny, nx+nover:nx+2*nover])
        hdr['DATASEC4'] = xy2hdr(np.s_[ny:ny+ny, nx+2*nover:nx+2*nover+nx])
        hdr['CCDSEC4'] =  xy2hdr(np.s_[ny:ny+ny, nx:nx+nx])
        
        hdr['NIGHT'] = '20180923'
        hdr['EXPID'] = 1
        hdr['PROGRAM'] = 'dark'
        hdr['FLAVOR'] = 'science'
        hdr['EXPTIME'] = 100.0
        
        rawimage = np.zeros((2*ny, 2*nx+2*noverscan))
        offset = {'1':100.0, '2':100.5, '3':50.3, '4':200.4}
        gain = {'1':1.0, '2':1.5, '3':0.8, '4':1.2}
        rdnoise = {'1':2.0, '2':2.2, '3':2.4, '4':2.6}
        
        quad = {
            '1': np.s_[0:ny, 0:nx], '2': np.s_[0:ny, nx:nx+nx],
            '3': np.s_[ny:ny+ny, 0:nx], '4': np.s_[ny:ny+ny, nx:nx+nx],
        }
        
        for amp in ('1', '2', '3', '4'):

            hdr['GAIN'+amp] = gain[amp]
            hdr['RDNOISE'+amp] = rdnoise[amp]
            
            xy = _parse_sec_keyword(hdr['BIASSEC'+amp])
            shape = [xy[0].stop-xy[0].start, xy[1].stop-xy[1].start]
            rawimage[xy] += offset[amp]
            rawimage[xy] += np.random.normal(scale=rdnoise[amp], size=shape)/gain[amp]
            xy = _parse_sec_keyword(hdr['DATASEC'+amp])
            shape = [xy[0].stop-xy[0].start, xy[1].stop-xy[1].start]
            rawimage[xy] += offset[amp]
            rawimage[xy] += np.random.normal(scale=rdnoise[amp], size=shape)/gain[amp]

        #- set CCD parameters
        self.ccdsec1=hdr["CCDSEC1"]
        self.ccdsec2=hdr["CCDSEC2"]
        self.ccdsec3=hdr["CCDSEC3"]
        self.ccdsec4=hdr["CCDSEC4"]

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

        psf = self.psf = load_psf(self.psffile)

        #- make the test pixfile, fibermap file
        img_pix = rawimg
        img_ivar = np.ones_like(img_pix) / 3.0**2
        img_mask = np.zeros(img_pix.shape, dtype=np.uint32)
        img_mask[200] = 1

        self.image = desispec.image.Image(img_pix, img_ivar, img_mask, camera='z1',meta=hdr)
        desispec.io.write_image(self.pixfile, self.image)

        self.fibermap = desispec.io.empty_fibermap(30)
        self.fibermap['OBJTYPE'][::2]='ELG'
        self.fibermap['OBJTYPE'][::3]='STD'
        self.fibermap['OBJTYPE'][::5]='QSO'
        self.fibermap['OBJTYPE'][::9]='LRG'
        self.fibermap['OBJTYPE'][::7]='SKY'
        #- add a filter and arbitrary magnitude
        self.fibermap['MAG'][:29]=np.tile(np.random.uniform(18,20,29),5).reshape(29,5) #- Last fiber left
        self.fibermap['FILTER'][:29]=np.tile(['DECAM_R','..','..','..','..'],(29,1)) #- last fiber left 

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
        self.frame.meta = dict(CAMERA=self.camera,PROGRAM='dark',FLAVOR='science',NIGHT=self.night,EXPID=self.expid,EXPTIME=100,CCDSEC1=self.ccdsec1,CCDSEC2=self.ccdsec2,CCDSEC3=self.ccdsec3,CCDSEC4=self.ccdsec4)
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

    def test_sky_resid(self):
        import copy
        param = dict(
                     PCHI_RESID=0.05,PER_RESID=95.,BIN_SZ=0.1)
        qadict=qalib.sky_resid(param,self.frame,self.skymodel,quick_look=True)
        kk=np.where(self.frame.fibermap['OBJTYPE']=='SKY')[0]
        self.assertEqual(qadict['NSKY_FIB'],len(kk))

        #- run with different sky flux
        skym1=desispec.sky.SkyModel(self.frame.wave,self.skymodel.flux,self.skymodel.ivar,self.mask)
        skym2=desispec.sky.SkyModel(self.frame.wave,self.skymodel.flux*0.5,self.skymodel.ivar,self.mask)
        frame1=copy.deepcopy(self.frame)
        frame2=copy.deepcopy(self.frame)
        desispec.sky.subtract_sky(frame1,skym1)
        desispec.sky.subtract_sky(frame2,skym2)

        qa1=qalib.sky_resid(param,frame1,skym1)
        qa2=qalib.sky_resid(param,frame2,skym2)
        self.assertLess(qa1['MED_RESID'],qa2['MED_RESID']) #- residuals must be smaller for case 1

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

        self.assertEqual(self.fibermap['MAG'][29][0],0)   #- No mag for last fiber
        self.assertEqual(self.fibermap['FILTER'][29][0],'') #- No filter for last fiber

        self.assertEqual(len(qa['MEDIAN_SNR']),30)

    #- Test each individual QA:
    def testBiasOverscan(self):
        return
        qa=QA.Bias_From_Overscan('bias',self.config) #- initialize with fake config and name
        inp=self.rawimage
        qargs={}
        qargs["PSFFile"]=self.psf
        qargs["camera"]=self.camera
        qargs["expid"]=self.expid
        qargs["amps"]=True
        qargs["qafile"]=self.qafile
        qargs["qafig"]=self.qafig
        qargs["paname"]="abc"
        res1=qa(inp,**qargs)
        self.assertEqual(len(res1['METRICS']['BIAS_AMP']),4)

    def testGetRMS(self):
        config={"kwargs":{
            "refKey":{},
            "param":{}
        }
        }

        qa=QA.Get_RMS('rms',config)
        inp=self.image
        qargs={}
        qargs["PSFFile"]=self.psf
        qargs["camera"]=self.camera
        qargs["expid"]=self.expid
        qargs["amps"]=True
        qargs["paname"]="abc"
        qargs["qafile"]=self.qafile
        qargs["qafig"]=self.qafig
        resl=qa(inp,**qargs)
        self.assertTrue("yaml" in qargs["qafile"])
        self.assertTrue("png" in qargs["qafig"])
        self.assertTrue(len(resl['METRICS']['NOISE_AMP'])==4)
        self.assertTrue((np.all(resl['METRICS']['NOISE_AMP'])>0))

    def testCalcXWSigma(self):

        #- Create another pix file for xwsigma test
        xw_hdr = dict()
        xw_hdr['CAMERA'] = self.camera
        xw_hdr['NIGHT'] = self.night
        xw_hdr['EXPID'] = self.expid
        xw_hdr['PROGRAM'] = 'dark'
        xw_hdr['FLAVOR'] = 'science'

        psf = self.psf
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
            pix = np.rint(psf.xy(fibers,zpeaks[i]))
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
        qargs["PSFFile"]=self.psf
        qargs["FiberMap"]=self.fibermap
        qargs["camera"]=self.camera
        qargs["expid"]=self.expid
        qargs["amps"]=False
        qargs["paname"]="abc"
        resl=qa(inp,**qargs)
        self.assertTrue(np.all(resl["METRICS"]["XSIGMA"])>0)

    def testCountPixels(self):
        qa=QA.Count_Pixels('countpix',self.config)
        inp=self.image
        qargs={}
        qargs["PSFFile"]=self.psf
        qargs["camera"]=self.camera
        qargs["expid"]=self.expid
        qargs["amps"]=False
        qargs["paname"]="abc"
        resl=qa(inp,**qargs)
        self.assertTrue(resl['METRICS']['NPIX_LOW'] > resl['METRICS']['NPIX_HIGH'])
        #- test if amp QAs exist
        qargs["amps"] = True
        resl2=qa(inp,**qargs)
        self.assertTrue(len(resl2['METRICS']['NPIX_AMP'])==4)

    def testCountSpectralBins(self):
        qa=QA.CountSpectralBins('countbins',self.config)
        inp=self.frame
        qargs={}
        qargs["PSFFile"]=self.psf
        qargs["FiberMap"]=self.fibermap
        qargs["camera"]=self.camera
        qargs["expid"]=self.expid
        qargs["amps"]=True
        qargs["paname"]="abc"
        qargs["qafile"]=self.qafile
        qargs["qafig"]=self.qafig
        resl=qa(inp,**qargs)
        self.assertTrue(np.all(resl["METRICS"]["NBINSMED"]-resl["METRICS"]["NBINSHIGH"])>=0)
        self.assertTrue(np.all(resl["METRICS"]["NBINSLOW"]-resl["METRICS"]["NBINSMED"])>=0)
        self.assertLess(resl["BOTTOM_MAX_WAVE_INDEX"],resl["TOP_MIN_WAVE_INDEX"])

    def testSkyCont(self):
        qa=QA.Sky_Continuum('skycont',self.config)
        inp=self.frame
        qargs={}
        qargs["FiberMap"]=self.fibermap
        qargs["camera"]=self.camera
        qargs["expid"]=self.expid
        qargs["amps"]=False
        qargs["paname"]="abc"
        resl=qa(inp,**qargs)
        self.assertTrue(resl["METRICS"]["SKYFIBERID"]==[0,7,14,21,28]) #- as defined in the fibermap
        self.assertTrue(resl["METRICS"]["SKYCONT"]>0)
        #- Test for amp True Case
        qargs["amps"]=True
        qargs["dict_countbins"]=self.map2pix #- This is not the full dict but contains the map needed here.
        resl2=qa(inp,**qargs)
        self.assertTrue(np.all(resl2["METRICS"]["SKYCONT_AMP"])>0)
        
    def testSkyPeaks(self):
        qa=QA.Sky_Peaks('skypeaks',self.config)
        inp=self.frame
        qargs={}
        qargs["FiberMap"]=self.fibermap
        qargs["camera"]=self.camera
        qargs["expid"]=self.expid
        #qargs["amps"]=True
        qargs["paname"]="abc"
        qargs["dict_countbins"]=self.map2pix
        resl=qa(inp,**qargs)
        #self.assertTrue(np.all(resl['METRICS']['PEAKCOUNT_RMS_AMP'])>=0.)
        self.assertTrue(resl['METRICS']['PEAKCOUNT_RMS']>0)

    def testIntegrateSpec(self):
        qa=QA.Integrate_Spec('integ',self.config)
        inp=self.frame
        qargs={}
        qargs["PSFFile"]=self.psf
        qargs["FiberMap"]=self.fibermap
        qargs["camera"]=self.camera
        qargs["expid"]=self.expid
        qargs["amps"]=False
        qargs["paname"]="abc"
        qargs["dict_countbins"]=self.map2pix
        resl=qa(inp,**qargs)
        self.assertTrue(resl['METRICS']['INTEG_AVG'] >0)
        self.assertTrue(len(resl["METRICS"]["INTEG"])==len(resl["METRICS"]["STD_FIBERID"]))
        #- Test for amps
        qargs["amps"]=True
        qargs["dict_countbins"]=self.map2pix
        resl2=qa(inp,**qargs)
        self.assertTrue(np.all(resl2["METRICS"]["INTEG_AVG_AMP"])>0)
        
    def testSkyResidual(self):
        qa=QA.Sky_Residual('skyresid',self.config)
        inp=self.frame
        sky=self.skymodel
        qargs={}
        qargs["PSFFile"]=self.psf
        qargs["FiberMap"]=self.fibermap
        qargs["camera"]=self.camera
        qargs["expid"]=self.expid
        qargs["amps"]=True
        qargs["paname"]="abc"
        qargs["dict_countbins"]=self.map2pix
        resl=qa(inp,sky,**qargs)
        self.assertTrue(resl["METRICS"]["NREJ"]==self.skymodel.nrej)
        self.assertTrue(len(resl["METRICS"]["MED_RESID_WAVE"]) == self.nwave)
        self.assertTrue(len(resl["METRICS"]["MED_RESID_FIBER"]) == 5) #- 5 sky fibers in the input
        self.assertTrue(resl["PARAMS"]["BIN_SZ"] == 0.1)
        #- test with different parameter set:
        qargs["param"]={"BIN_SZ":0.2, "PCHI_RESID":0.05, "PER_RESID":95., "SKYRESID_NORMAL_RANGE":[-5.0, 5.0], "SKYRESID_WARN_RANGE":[-10.0, 10.0]}
        resl2=qa(inp,sky,**qargs)
        self.assertTrue(len(resl["METRICS"]["DEVS_1D"])>len(resl2["METRICS"]["DEVS_1D"])) #- larger histogram bin size than default 0.1

    def testCalculateSNR(self):
        qa=QA.Calculate_SNR('snr',self.config)
        inp=self.frame
        qargs={}
        qargs["PSFFile"]=self.psf
        qargs["FiberMap"]=self.fibermap
        qargs["camera"]=self.camera
        qargs["expid"]=self.expid
        qargs["amps"]=True
        qargs["paname"]="abc"
        qargs["qafile"]=self.qafile #- no LRG by construction.
        qargs["dict_countbins"]=self.map2pix
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

