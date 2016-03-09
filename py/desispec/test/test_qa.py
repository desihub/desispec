"""
tests desispec.sky
"""

import unittest

import numpy as np
import os
from desispec.qa import QA_Frame, QA_Exposure
from desispec.io import write_qa_frame
from uuid import uuid4

class TestQA(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.nspec = 6
        cls.nwave = 20
        id = uuid4().hex
        cls.qafile_b0 = 'qa-b0-'+id+'.yaml'
        cls.qafile_b1 = 'qa-b1-'+id+'.yaml'
        cls.topDir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        cls.binDir = os.path.join(cls.topDir,'bin')
        try:
            cls.origPath = os.environ['PYTHONPATH']
            os.environ['PYTHONPATH'] = os.path.join(cls.topDir,'py') + ':' + cls.origPath
        except KeyError:
            cls.origPath = None
            os.environ['PYTHONPATH'] = os.path.join(cls.topDir,'py')

    @classmethod
    def tearDownClass(cls):
        """Cleanup in case tests crashed and left files behind"""
        for filename in [cls.qafile_b0, cls.qafile_b1]:
            if os.path.exists(filename):
                os.remove(filename)
        if cls.origPath is None:
            del os.environ['PYTHONPATH']
        else:
            os.environ['PYTHONPATH'] = cls.origPath
    
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
        #- Test loading from yaml files
        qafrm = QA_Frame(flavor='dark')
        qafrm.init_skysub()
        qafrm.data['SKYSUB']['NSKY_FIB'] = 10
        write_qa_frame(self.qafile_b0, qafrm)
        qafrm2 = QA_Frame(flavor='dark')
        qafrm2.init_skysub()
        qafrm2.data['SKYSUB']['NSKY_FIB'] = 30
        write_qa_frame(self.qafile_b1, qafrm2)
        #
        os.environ['PRODNAME'] = './'
        os.environ['DESI_SPECTRO_REDUX'] = './'
        qaexp = QA_Exposure(1, '')
        import pdb
        pdb.set_trace()

    def runTest(self):
        pass
                
if __name__ == '__main__':
    unittest.main()
