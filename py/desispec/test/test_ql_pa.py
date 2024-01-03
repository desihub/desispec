"""
tests for Quicklook Pipeline steps in desispec.quicklook.procalgs
"""

import unittest
import numpy as np
import os
import shutil
import desispec
from desispec.quicklook import procalgs as PA
from importlib import resources
from desispec.test.test_ql_qa import xy2hdr
from desispec.preproc import parse_sec_keyword
import astropy.io.fits as fits
from desispec.quicklook import qllogger

qlog=qllogger.QLLogger("QuickLook",0)
log=qlog.getlog()

class TestQL_PA(unittest.TestCase):

    def tearDown(self):
        self.rawimage.close()
        for filename in [self.rawfile, self.pixfile]:
            if os.path.exists(filename):
                os.remove(filename)
        if os.path.exists(self.testDir):
            shutil.rmtree(self.testDir)
    
    #- Create some test data
    def setUp(self):
        
        #- Create temporary calib directory
        self.testDir  = os.path.join(os.environ['HOME'], 'ql_test_io')
        calibDir = os.path.join(self.testDir, 'ql_calib')
        if not os.path.exists(calibDir): os.makedirs(calibDir)
        
        #- Generate calib data
        for camera in ['b0', 'r0', 'z0']:
            #- Fiberflat has to exist but can be a dummpy file
            filename = '{}/fiberflat-{}.fits'.format(calibDir, camera)
            fx = open(filename, 'w'); fx.write('fiberflat file'); fx.close()

            #- PSF has to be real file
            psffile = '{}/psf-{}.fits'.format(calibDir, camera)
            example_psf = resources.files('desispec').joinpath(f'test/data/ql/psf-{camera}.fits')
            shutil.copy(example_psf, psffile)
            
        #- Copy test calibration-data.yaml file 
        specdir=calibDir+"spec/sp0"
        if not os.path.isdir(specdir) :
            os.makedirs(specdir)
        for c in "brz" :
            shutil.copy(resources.files('desispec').joinpath(f'test/data/ql/{c}0.yaml'), os.path.join(specdir, f"{c}0.yaml"))
        
        #- Set calibration environment variable
        os.environ['DESI_SPECTRO_CALIB'] = calibDir
                
        self.rawfile =  os.path.join(self.testDir,'test-raw-abcd.fits')
        self.pixfile =  os.path.join(self.testDir,'test-pix-abcd.fits')
        self.config={}
        
        #- rawimage

        hdr = dict()
        hdr['CAMERA'] = 'b0'
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
            
            xy = parse_sec_keyword(hdr['BIASSEC'+amp])
            shape = [xy[0].stop-xy[0].start, xy[1].stop-xy[1].start]
            rawimage[xy] += offset[amp]
            rawimage[xy] += np.random.normal(scale=rdnoise[amp], size=shape)/gain[amp]
            xy = parse_sec_keyword(hdr['DATASEC'+amp])
            shape = [xy[0].stop-xy[0].start, xy[1].stop-xy[1].start]
            rawimage[xy] += offset[amp]
            rawimage[xy] += np.random.normal(scale=rdnoise[amp], size=shape)/gain[amp]
        #- raw data are integers, not floats
        rawimg = rawimage.astype(np.int32)
        self.expid=hdr["EXPID"]
        self.camera=hdr["CAMERA"]
        #- Confirm that all regions were correctly offset
        assert not np.any(rawimage == 0.0)        

        hdr['DOSVER'] = 'SIM'
        hdr['FEEVER'] = 'SIM'
        hdr['DETECTOR'] = 'SIM'
        desispec.io.write_raw(self.rawfile,rawimg,hdr,camera=self.camera)
        self.rawimage=fits.open(self.rawfile)

        

    #- Individual tests already exist in offline tests. So we will mostly test the call etc. here
#    def testPreproc(self):
#        pa=PA.Preproc('Preproc',self.config,logger=log)
#        log.info("Test preproc")
#        inp=self.rawimage
#        rawshape=inp[self.camera.upper()].data.shape
#        bias=np.zeros(rawshape)
#        pixflat=np.ones(rawshape)
#        mask = np.random.randint(0, 2, size=(1000,800))
#        pargs={}
#        pargs["camera"]=self.camera
#        pargs["Bias"]=bias
#        pargs["PixFlat"]=pixflat
#        pargs["Mask"]=mask
#        pargs["DumpIntermediates"]=True
#        pargs["dumpfile"]=self.pixfile
#        img=pa(inp,**pargs) 
#        self.assertTrue(np.all(img.mask == mask))
