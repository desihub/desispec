"""
tests desispec.sky
"""

import unittest

import numpy as np
from desispec.qa import QA_Frame
from desispec.io import qa as desio_qa

class TestQA(unittest.TestCase):
    
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


    def runTest(self):
        pass
                
if __name__ == '__main__':
    unittest.main()
