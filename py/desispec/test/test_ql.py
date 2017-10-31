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

class TestQL(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.program = program = 'dark'
        cls.flavor = flavor = 'bias'
        cls.night = night = '20150105'
        cls.camera = camera = 'r0'
        cls.expid = expid = 314
        cls.flatExpid = flatExpid = 313
        cls.nspec = nspec = 5
        cls.exptime = exptime = 100

        #- Seup environment and override default environment variables

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
        dataDir = os.path.join(testDir,night)
        expDir = os.path.join(testDir,'exposures')
        nightDir = os.path.join(expDir,night)
        reduxDir = os.path.join(nightDir,'{:08d}'.format(expid))
        if not os.path.exists(testDir):
            os.makedirs(testDir)
            os.makedirs(dataDir)
            os.makedirs(expDir)
            os.makedirs(nightDir)
            os.makedirs(reduxDir)
        if 'QL_SPEC_DATA' in os.environ:
            os.environ['QL_SPEC_DATA'] = testDir
        if 'QL_SPEC_REDUX' in os.environ:
            os.environ['QL_SPEC_REDUX'] = testDir

        #- Write dummy configuration and input files to test merging
        configdict = {'name': 'Test Configuration',
                      'Program': program,
                      'Flavor': flavor,
                      'PSFType': 'psfboot',
                      'FiberflatExpid': flatExpid,
                      'WritePixfile': False,
                      'WriteSkyModelfile': False,
                      'WriteIntermediatefiles': False,
                      'WriteStaticPlots': False,
                      'Debuglevel': 20,
                      'UseResolution': False,
                      'Period': 5.0,
                      'Timeout': 120.0,
                      'Pipeline': ['Initialize','Preproc'],
                      'Algorithms': {'Initialize':{
                                         'QA':{
                                             'Bias_From_Overscan':{'PARAMS':{'PERCENTILES':[68.2,95.4,99.7],'DIFF_WARN_RANGE':[-1.0,1.0],'DIFF_ALARM_RANGE':[-2.0,2.0]}}}},
                                     'Preproc':{
                                         'QA':{
                                             'Get_RMS':{'PARAMS':{'RMS_WARN_RANGE':[-1.0,1.0],'RMS_ALARM_RANGE':[-2.0,2.0]}},
                                             'Count_Pixels':{'PARAMS':{'CUTHI':500,'CUTLO':100,'NPIX_WARN_RANGE':[200.0,500.0],'NPIX_ALARM_RANGE':[50.0,650.0]}}}}}
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
        
        data=np.zeros((4096,4204))+200.
        raw_data=data.astype(int)
        write_raw(rawfile,raw_data,raw_hdr)

        #- Generate fibermap file
        fibermapfile = os.path.join(dataDir,'fibermap-00000314.fits')
        fibermap = empty_fibermap(nspec)
        write_fibermap(fibermapfile,fibermap)

   #- Clean up test files and directories if they exist
    @classmethod
    def tearDown(cls):
        for filename in [cls.fibermapfile,cls.framefile]:
            if os.path.exists(filename):
                os.remove(filename)
        if os.path.exists(cls.testDir):
            shutil.rmtree(cls.testDir)

    #- Test if QuickLook outputs merged QA file
    def test_mergeQA(self):
        cmd = "{} {}/desi_quicklook -i {} -n {} -c {} -e {} --mergeQA".format(sys.executable,self.binDir,self.configfile,self.night,self.camera,self.expid)
        if runcmd(cmd) != 0:
            raise RuntimeError('quicklook pipeline failed')


#- This runs all test* functions in any TestCase class in this file
if __name__ == '__main__':
    unittest.main()
