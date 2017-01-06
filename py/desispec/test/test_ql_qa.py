"""
tests desispec.ql_qa
"""

import unittest
import numpy as np
import os
from desispec.qa import qa_quicklook as QA
from desispec.quicklook import procalgs as PAs
from desispec.quicklook import qas
from desispec.quicklook import quicklook as ql
from pkg_resources import resource_filename
import desispec
from desispec.preproc import _parse_sec_keyword
from specter.psf import load_psf
import astropy.io.fits as fits
from desispec.quicklook import qllogger


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


class TestQL(unittest.TestCase):

    def tearDown(self):
        for filename in [self.framefile, self.rawfile, self.pixfile, self.fibermapfile, self.skyfile]:
            if os.path.exists(filename):
                os.remove(filename)

    #- Create some test data
    def setUp(self):

        self.rawfile = 'test-raw-abcd.fits'
        self.pixfile = 'test-pix-abcd.fits'
        self.framefile = 'test-frame-abcd.fits'
        self.fibermapfile = 'test-fibermap-abcd.fits'
        self.skyfile = 'test-sky-abcd.fits'

        #- use specter psf for this test
        self.psffile=resource_filename('specter', 'test/t/psf-monospot.fits')
        self.config={}

        #- rawimage

        hdr = dict()
        hdr['CAMERA'] = 'b1'
        hdr['DATE-OBS'] = '2018-09-23T08:17:03.988'

        #- Dimensions per amp, not full 4-quad CCD
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
        hdr['FLAVOR']='dark'
        
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

        #- raw data are integers, not floats
        rawimg = rawimage.astype(np.int32)
        self.expid=hdr["EXPID"]
        self.camera=hdr["CAMERA"]
        #- Confirm that all regions were correctly offset
        assert not np.any(rawimage == 0.0)        

        #- write to the rawfile and read it in QA test
        desispec.io.write_raw(self.rawfile,rawimg,hdr,camera=self.camera)
        self.rawimage=fits.open(self.rawfile)
        
        #- read psf, since using specter test psf, should use specter.PSF.load_psf instead of desispec.PSF(), otherwise need to create a psfboot somewhere.

        self.psf=load_psf(self.psffile)

        #- make the test pixfile, fibermap file
        img_pix = np.random.normal(0, 10.0, size=(400,400))
        img_ivar = np.ones_like(img_pix) / 3.0**2
        img_mask = np.zeros(img_pix.shape, dtype=np.uint32)
        img_mask[200] = 1
        self.image = desispec.image.Image(img_pix, img_ivar, img_mask, camera='r0',meta=hdr)
        desispec.io.write_image(self.pixfile, self.image)
        
        self.fibermap = desispec.io.empty_fibermap(30)
        self.fibermap['OBJTYPE'][::2]='ELG'
        self.fibermap['OBJTYPE'][::3]='LRG'
        self.fibermap['OBJTYPE'][::5]='QSO'
        self.fibermap['OBJTYPE'][::7]='SKY'
        #- add a filter and arbitrary magnitude
        self.fibermap['MAG']=np.tile(np.random.uniform(18,20,30),5).reshape(30,5)
        self.fibermap['FILTER']=np.tile(['DECAM_R','..','..','..','..'],(30,1))
        
        desispec.io.write_fibermap(self.fibermapfile, self.fibermap)        
        
        #- make a test frame file
        self.night=hdr['NIGHT']
        nspec=30
        nwave=200
        wave=np.arange(nwave)
        flux=np.random.uniform(size=(nspec,nwave))
        ivar=np.ones_like(flux)
        resolution_data=np.ones((nspec,13,nwave))
        self.frame=desispec.frame.Frame(wave,flux,ivar,resolution_data=resolution_data,fibermap=self.fibermap)
        #self.frame.meta = dict(CAMERA=self.camera,FLAVOR='dark',NIGHT=self.night, EXPID=self.expid)
        desispec.io.write_frame(self.framefile, self.frame)

        #- make a skymodel
        sky=np.ones_like(self.frame.flux)*0.5
        skyivar=np.ones_like(sky)
        self.mask=np.zeros(sky.shape,dtype=np.uint32)
        self.skymodel=desispec.sky.SkyModel(wave,sky,skyivar,self.mask)
        self.skyfile=desispec.io.write_sky(self.skyfile,self.skymodel)
        

    #- test some qa utillities functions:
    def test_ampregion(self):
        pixboundary=QA.ampregion(self.image)
        self.assertEqual(pixboundary[0][1],slice(0,self.nx,None))
        self.assertEqual(pixboundary[3][0],slice(self.ny,self.ny+self.ny,None))


    def test_getrms(self):
        img_rms=QA.getrms(self.image.pix)
        self.assertEqual(img_rms,np.std(self.image.pix))

    def test_countpix(self):
        pix=self.image.pix
        counts1=QA.countpix(pix,nsig=3) #- counts avove 3 sigma
        counts2=QA.countpix(pix,nsig=4) #- counts above 4 sigma
        self.assertLess(counts2,counts1)
        counts3=QA.countpix(pix,ncounts=15)
        counts4=QA.countpix(pix,ncounts=20)
        self.assertLess(counts4,counts3)

    def test_sky_resid(self):
        import copy
        param = dict(
                     PCHI_RESID=0.05,PER_RESID=95.)
        qadict=QA.sky_resid(param,self.frame,self.skymodel,quick_look=True)
        kk=np.where(self.frame.fibermap['OBJTYPE']=='SKY')[0]
        self.assertEqual(qadict['NSKY_FIB'],len(kk))

        #- run with different sky flux
        skym1=desispec.sky.SkyModel(self.frame.wave,self.skymodel.flux,self.skymodel.ivar,self.mask)
        skym2=desispec.sky.SkyModel(self.frame.wave,self.skymodel.flux*0.5,self.skymodel.ivar,self.mask)
        frame1=copy.deepcopy(self.frame)
        frame2=copy.deepcopy(self.frame)
        desispec.sky.subtract_sky(frame1,skym1)
        desispec.sky.subtract_sky(frame2,skym2)

        qa1=QA.sky_resid(param,frame1,skym1)
        qa2=QA.sky_resid(param,frame2,skym2)
        self.assertLess(qa1['MED_RESID'],qa2['MED_RESID']) #- residuals must be smaller for case 1

    def testSignalVsNoise(self):
        import copy
        params=None
        #- first get the sky subtracted frame
        #- copy frame not to override
        thisframe=copy.deepcopy(self.frame)
        desispec.sky.subtract_sky(thisframe,self.skymodel)
        qadict=QA.SignalVsNoise(thisframe,params)
        #- make sure all the S/N is positive
        self.assertTrue(np.all(qadict['MEDIAN_SNR']) > 0)

        #- Reduce sky
        skym1=desispec.sky.SkyModel(self.frame.wave,self.skymodel.flux,self.skymodel.ivar,self.mask)
        skym2=desispec.sky.SkyModel(self.frame.wave,self.skymodel.flux*0.5,self.skymodel.ivar,self.mask)
        frame1=copy.deepcopy(self.frame)
        frame2=copy.deepcopy(self.frame)
        desispec.sky.subtract_sky(frame1,skym1)
        desispec.sky.subtract_sky(frame2,skym2)
        qa1=QA.SignalVsNoise(frame1,params)
        qa2=QA.SignalVsNoise(frame2,params)
        self.assertTrue(np.all(qa2['MEDIAN_SNR'] > qa1['MEDIAN_SNR']))
   
    #- QA: bias overscan
    def testBiasOverscan(self):
        qa=QA.Bias_From_Overscan('bias',self.config) #- initialize with fake config and name
        inp=self.rawimage
        qargs={}
        qargs["PSFFile"]=self.psf
        qargs["camera"]=self.camera
        qargs["expid"]=self.expid
        qargs["amps"]=True
        qargs["paname"]="abc"
        res1=qa(inp,**qargs)
        self.assertEqual(len(res1['METRICS']['BIAS_AMP']),4)
        

if __name__ == '__main__':
    unittest.main()

