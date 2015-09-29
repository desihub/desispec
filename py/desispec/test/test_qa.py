"""
tests desispec.sky
"""

import unittest

import numpy as np
from desispec.qa.qa_exposure import QA_Frame
from desispec.io import meta as dio_meta
from desispec.io import util as dio_util
from desispec.io import qa as desio_qa

class TestQA(unittest.TestCase):
    
    def test_init_qa_frame(self):        
        #- Simple Init calls
        qafrm1 = QA_Frame('arc')
        qafrm2 = QA_Frame('flat')
        qafrm3 = QA_Frame('science')
        assert qafrm3.flavor == 'science'

    def test_init_qa_skysub(self):        
        #- Init SkySub dict
        qafrm = QA_Frame('science')
        qafrm.init_skysub()
        assert qafrm.data['SKYSUB']['PARAM']['PCHI_RESID'] > 0.

        #- ReInit SkySub dict
        qafrm.init_skysub(re_init=True)
        assert qafrm.data['SKYSUB']['PARAM']['PCHI_RESID'] > 0.


    def runTest(self):
        pass
                
if __name__ == '__main__':
    unittest.main()
