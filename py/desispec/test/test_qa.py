"""
tests desispec.sky
"""

import unittest
import pdb

import numpy as np
import os
from desispec.qa import QA_Frame, QA_Exposure, QA_Brick
from desispec.io import write_qa_frame, write_qa_brick
#from uuid import uuid4
from shutil import rmtree

class TestQA(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.nspec = 6
        cls.nwave = 20
        id = 1
        cls.night = '20160101'
        cls.testDir = os.path.join(os.environ['HOME'],'desi_test_qa')
        cls.qafile_b0 = cls.testDir+'/exposures/'+cls.night+'/{:08d}/qa-b0-{:08d}.yaml'.format(id,id)
        cls.qafile_b1 = cls.testDir+'/exposures/'+cls.night+'/{:08d}/qa-b1-{:08d}.yaml'.format(id,id)
        cls.qafile_brick = cls.testDir+'/brick/3582m005/qa-3582m005.yaml'
        cls.flux_pdf = cls.testDir+'/exposures/'+cls.night+'/{:08d}/qa-flux-{:08d}.pdf'.format(id,id)

    @classmethod
    def tearDownClass(cls):
        """Cleanup in case tests crashed and left files behind"""
        for filename in [cls.qafile_b0, cls.qafile_b1, cls.flux_pdf]:
            if os.path.exists(filename):
                os.remove(filename)
                #testpath = os.path.normpath(os.path.dirname(filename))
                #if testpath != '.':
                #    os.removedirs(testpath)
        if os.path.exists(cls.testDir):
            rmtree(cls.testDir)

    def _write_qaframes(self):
        """Write a QA data frame files"""
        qafrm0 = QA_Frame(camera='b0',flavor='dark')
        qafrm1 = QA_Frame(camera='b1',flavor='dark')
        # SKY
        qafrm0.init_skysub()
        qafrm1.init_skysub()
        qafrm0.data['SKYSUB']['QA'] = {}
        qafrm1.data['SKYSUB']['QA'] = {}
        qafrm0.data['SKYSUB']['QA']['NSKY_FIB'] = 10
        qafrm1.data['SKYSUB']['QA']['NSKY_FIB'] = 30
        # FLUX
        qafrm0.init_fluxcalib()
        qafrm1.init_fluxcalib()
        qafrm0.data['FLUXCALIB']['QA'] = {}
        qafrm0.data['FLUXCALIB']['QA']['ZP'] = 24.
        qafrm0.data['FLUXCALIB']['QA']['RMS_ZP'] = 0.05
        qafrm1.data['FLUXCALIB']['QA'] = {}
        qafrm1.data['FLUXCALIB']['QA']['ZP'] = 24.5
        qafrm1.data['FLUXCALIB']['QA']['RMS_ZP'] = 0.05
        # WRITE
        write_qa_frame(self.qafile_b0, qafrm0)
        write_qa_frame(self.qafile_b1, qafrm1)

    def _write_qabrick(self):
        """Write a QA data brick file"""
        qabrck = QA_Brick()
        # ZBEST
        qabrck.init_zbest()
        qabrck.data['ZBEST']['QA'] = {}
        qabrck.data['ZBEST']['QA']['NFAIL'] = 10
        write_qa_brick(self.qafile_brick, qabrck)

    def test_init_qa_brick(self):
        #- Simple Init calls
        qabrck = QA_Brick(name='tst_brick')
        assert qabrck.brick_name == 'tst_brick'
        #
        qabrck.init_zbest()
        assert qabrck.data['ZBEST']['PARAM']['MAX_NFAIL'] > 0

    def test_init_qa_frame(self):        
        #- Simple Init calls
        qafrm1 = QA_Frame(flavor='arc')
        qafrm2 = QA_Frame(flavor='flat')
        qafrm3 = QA_Frame(flavor='dark')
        assert qafrm3.flavor == 'dark'

    def test_init_qa_fiberflat(self):
        #- Init FiberFlat dict
        qafrm = QA_Frame(flavor='flat')
        qafrm.init_fiberflat()
        assert qafrm.data['FIBERFLAT']['PARAM']['MAX_RMS'] > 0.

        #- ReInit FiberFlat dict
        qafrm.init_fiberflat(re_init=True)
        assert qafrm.data['FIBERFLAT']['PARAM']['MAX_RMS'] > 0.

    def test_init_qa_fluxcalib(self):
        #- Init FluxCalib dict
        qafrm = QA_Frame(camera='b', flavor='dark')
        qafrm.init_fluxcalib()
        assert qafrm.data['FLUXCALIB']['PARAM']['MAX_ZP_OFF'] > 0.

        #- ReInit FluxCalib dict
        qafrm.init_fluxcalib(re_init=True)
        assert qafrm.data['FLUXCALIB']['PARAM']['MAX_ZP_OFF'] > 0.

    def test_init_qa_skysub(self):
        #- Init SkySub dict
        qafrm = QA_Frame(flavor='dark')
        qafrm.init_skysub()
        assert qafrm.data['SKYSUB']['PARAM']['PCHI_RESID'] > 0.

        #- ReInit SkySub dict
        qafrm.init_skysub(re_init=True)
        assert qafrm.data['SKYSUB']['PARAM']['PCHI_RESID'] > 0.

    def test_init_qa_exposure(self):
        # Simple init
        os.environ['PRODNAME'] = './'
        os.environ['DESI_SPECTRO_REDUX'] = './'
        qaexp = QA_Exposure(1, '20150211')
        assert qaexp.expid == 1

    def test_qa_exposure_load_data(self):
        #- Test loading data
        self._write_qaframes()
        qaexp = QA_Exposure(1, self.night, specprod_dir=self.testDir,
                            flavor='dark')
        assert 'b0' in qaexp.data['frames'].keys()
        assert 'b1' in qaexp.data['frames'].keys()

    """
    # This needs to run as a script for the figure generation to pass Travis..
    def test_qa_exposure_fluxcalib(self):
        #- Perform fluxcalib QA on Exposure (including figure)
        self._write_qaframes()
        qaexp = QA_Exposure(1, self.night, specprod_dir=self.testDir,
                            flavor='dark')
        qaexp.fluxcalib(self.flux_pdf)
    """

    def runTest(self):
        pass
                
if __name__ == '__main__':
    unittest.main()
