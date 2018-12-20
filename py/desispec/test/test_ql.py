"""
Test capabilities of  QuickLook pipeline

python -m desispec.test.test_ql
"""
import os, sys
import shutil
from uuid import uuid4
import unittest
import yaml
import numpy as np
from desispec.util import runcmd
from desispec.io.raw import write_raw
from desispec.io import empty_fibermap
from desispec.io.fibermap import write_fibermap
import datetime
import pytz
from pkg_resources import resource_filename

class TestQL(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.program = program = 'dark'
        cls.flavor = flavor = 'bias'
        cls.night = night = '20150105'
        cls.camera = camera = 'r0'
        cls.expid = expid = 314
        cls.psfExpid = psfExpid = 313
        cls.flatExpid = flatExpid = 312
        cls.templateExpid = templateExpid = 311
        cls.nspec = nspec = 5
        cls.exptime = exptime = 100

        #- Setup environment and override default environment variables

        #- python 2.7 location:
        cls.topDir = os.path.dirname( # top-level
            os.path.dirname( # py/
                os.path.dirname( # desispec/
                    os.path.dirname(os.path.abspath(__file__)) # test/
                    )
                )
            )
        cls.binDir = os.path.join(cls.topDir,'bin')
        if not os.path.isdir(cls.binDir):
            #- python 3.x setup.py test location:
            cls.topDir = os.path.dirname( # top-level
                os.path.dirname( # build/
                    os.path.dirname( # lib/
                        os.path.dirname( # desispec/
                            os.path.dirname(os.path.abspath(__file__)) # test/
                            )
                        )
                    )
                )
            cls.binDir = os.path.join(cls.topDir,'bin')

        #- last attempt
        if not os.path.isdir(cls.binDir):
            cls.topDir = os.getcwd()
            cls.binDir = os.path.join(cls.topDir, 'bin')

        if not os.path.isdir(cls.binDir):
            raise RuntimeError('Unable to auto-locate desispec/bin from {}'.format(__file__))

        id = uuid4().hex
        cls.fibermapfile = 'fibermap-'+id+'.fits'
        cls.framefile = 'frame-'+id+'.fits'

        cls.testDir = testDir = os.path.join(os.environ['HOME'],'ql_test_io')
        datanightDir = os.path.join(testDir,night)
        dataDir = os.path.join(datanightDir,'{:08d}'.format(expid))
        expDir = os.path.join(testDir,'exposures')
        expnightDir = os.path.join(expDir,night)
        reduxDir = os.path.join(expnightDir,'{:08d}'.format(expid))
        calibDir = os.path.join(testDir, 'ql_calib')
        configDir = os.path.join(testDir, 'ql_config')
        os.environ['QL_CALIB_DIR'] = calibDir
        os.environ['QL_CONFIG_DIR'] = configDir
        if not os.path.exists(testDir):
            os.makedirs(testDir)
            os.makedirs(datanightDir)
            os.makedirs(dataDir)
            os.makedirs(expDir)
            os.makedirs(expnightDir)
            os.makedirs(reduxDir)
            os.makedirs(calibDir)
            os.makedirs(configDir)

        #- Write dummy configuration and input files to test merging
        configdict = {'name': 'Test Configuration',
                      'Program': program,
                      'Flavor': flavor,
                      'PSFExpid': psfExpid,
                      'PSFType': 'psf',
                      'FiberflatExpid': flatExpid,
                      'TemplateExpid': templateExpid,
                      'TemplateNight': night,
                      'WritePreprocfile': False,
                      'WriteSkyModelfile': False,
                      'WriteIntermediatefiles': False,
                      'WriteStaticPlots': False,
                      'Debuglevel': 20,
                      'UseResolution': False,
                      'Period': 5.0,
                      'Timeout': 120.0,
                      'Pipeline': ['Initialize','Preproc'],
                      'Algorithms': {'Initialize':{
                                         'QA':{'Check_HDUs':{'PARAMS':{}}
                                             }},
                                     'Preproc':{
                                         'QA':{'Bias_From_Overscan':{'PARAMS':{'BIAS_AMP_NORMAL_RANGE':[-100.0,100.0],'BIAS_AMP_WARN_RANGE':[-200.0,200.0]}},
                                             'Get_RMS':{'PARAMS':{'PERCENTILES':[68.2,95.4,99.7],'NOISE_AMP_NORMAL_RANGE':[-1.0,1.0],'NOISE_AMP_WARN_RANGE':[-2.0,2.0]}},
                                             'Count_Pixels':{'PARAMS':{'CUTPIX':500,'LITFRAC_NORMAL_RANGE':[-0.1,0.1],'LITFRAC_WARN_RANGE':[-0.2,0.2]}}}}}
                      }
        with open('{}/test_config.yaml'.format(testDir),'w') as config:
            yaml.dump(configdict,config)
        cls.configfile = '{}/test_config.yaml'.format(testDir)

        #- Generate raw file
        rawfile = os.path.join(dataDir,'desi-00000314.fits.fz')
        raw_hdr = {}
        raw_hdr['DATE-OBS'] = '2015-01-05T08:17:03.988'
        raw_hdr['NIGHT'] = night
        raw_hdr['PROGRAM'] = program
        raw_hdr['FLAVOR'] = flavor
        raw_hdr['CAMERA'] = camera
        raw_hdr['EXPID'] = expid
        raw_hdr['EXPTIME'] = exptime
        raw_hdr['DOSVER'] = 'SIM'
        raw_hdr['FEEVER'] = 'SIM'
        raw_hdr['DETECTOR'] = 'SIM'
        raw_hdr['PRESEC1'] = '[1:4,1:2048]'
        raw_hdr['DATASEC1'] = '[5:2052,1:2048]'
        raw_hdr['BIASSEC1'] = '[2053:2102,1:2048]'
        raw_hdr['CCDSEC1']  = '[1:2048,1:2048]'
        raw_hdr['PRESEC2']  = '[4201:4204,1:2048]'
        raw_hdr['DATASEC2'] = '[2153:4200,1:2048]'
        raw_hdr['BIASSEC2'] = '[2103:2152,1:2048]'
        raw_hdr['CCDSEC2'] = '[2049:4096,1:2048]'
        raw_hdr['PRESEC3'] = '[1:4,2049:4096]'
        raw_hdr['DATASEC3'] = '[5:2052,2049:4096]'
        raw_hdr['BIASSEC3'] = '[2053:2102,2049:4096]'
        raw_hdr['CCDSEC3'] = '[1:2048,2049:4096]'
        raw_hdr['PRESEC4'] = '[4201:4204,2049:4096]'
        raw_hdr['DATASEC4'] = '[2153:4200,2049:4096]'
        raw_hdr['BIASSEC4'] = '[2103:2152,2049:4096]'
        raw_hdr['CCDSEC4'] = '[2049:4096,2049:4096]'
        raw_hdr['GAIN1'] = 1.0
        raw_hdr['GAIN2'] = 1.0
        raw_hdr['GAIN3'] = 1.0
        raw_hdr['GAIN4'] = 1.0
        raw_hdr['RDNOISE1'] = 3.0
        raw_hdr['RDNOISE2'] = 3.0
        raw_hdr['RDNOISE3'] = 3.0
        raw_hdr['RDNOISE4'] = 3.0

        primary_header={'PROGRAM':program}
        data=np.zeros((4096,4204))+200.
        raw_data=data.astype(int)
        write_raw(rawfile,raw_data,raw_hdr,primary_header=primary_header)

        #- Generate fibermap file
        fibermapfile = os.path.join(dataDir,'fibermap-00000314.fits')
        fibermap = empty_fibermap(nspec)
        write_fibermap(fibermapfile,fibermap)

        #- Generate calib data
        for camera in ['b0', 'r0', 'z0']:
            #- Fiberflat has to exist but can be a dummpy file
            filename = '{}/fiberflat-{}.fits'.format(calibDir, camera)
            fx = open(filename, 'w'); fx.write('fiberflat file'); fx.close()

            #- PSF has to be real file
            psffile = '{}/psf-{}.fits'.format(calibDir, camera)
            example_psf = resource_filename('desispec', 'test/data/ql/psf-{}.fits'.format(camera))
            shutil.copy(example_psf, psffile)
            
        #- Copy test calibration-data.yaml file 
        input_yaml_file = resource_filename('desispec', 'test/data/ql/ccd_calibration.yaml')
        output_yaml_file = os.path.join(calibDir,'ccd_calibration.yaml')
        shutil.copy(input_yaml_file,output_yaml_file)
        
        #- Set calibration environment variable
        os.environ['DESI_CCD_CALIBRATION_DATA'] = calibDir
    

   #- Clean up test files and directories if they exist
    @classmethod
    def tearDown(cls):
        for filename in [cls.fibermapfile,cls.framefile]:
            if os.path.exists(filename):
                os.remove(filename)
        if os.path.exists(cls.testDir):
            shutil.rmtree(cls.testDir)

    #- Test if QuickLook outputs merged QA file
    #def test_mergeQA(self):
        #os.environ['QL_SPEC_REDUX'] = self.testDir
        #cmd = "{} {}/desi_quicklook -i {} -n {} -c {} -e {} --rawdata_dir {} --specprod_dir {} --mergeQA".format(sys.executable,self.binDir,self.configfile,self.night,self.camera,self.expid,self.testDir,self.testDir)
        #pyver = format(sys.executable.split('anaconda')[1])
        #print('NOTE: Test is running on python v'+format(pyver.split('/')[0]))
        
        #if int(format(pyver.split('/')[0])) < 3:
             #pass
        #else:
           #if runcmd(cmd) != 0:
              #raise RuntimeError('quicklook pipeline failed')


    def test_QA(self):
        os.environ['QL_SPEC_REDUX'] = self.testDir
#        cmd = "{} {}/desi_quicklook -i {} -n {} -c {} -e {} --rawdata_dir {} --specprod_dir {} ".format(sys.executable,self.binDir,self.configfile,self.night,self.camera,self.expid,self.testDir,self.testDir)
 
#        if runcmd(cmd) != 0:
#              raise RuntimeError('quicklook pipeline failed')


#- This runs all test* functions in any TestCase class in this file
if __name__ == '__main__':
    unittest.main()
