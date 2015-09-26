"""
tests desispec.sky
"""

import unittest

import numpy as np
from desispec.qa.qa_exposure import QA_Exposure
from desispec.io import meta as dio_meta
from desispec.io import util as dio_util
from desispec.io import qa as desio_qa

class TestQA(unittest.TestCase):
    
    def test_init_qa_exposure(self):        
        #- Simple Init calls
        qaexp1 = QA_Exposure('arc',2)
        qaexp2 = QA_Exposure('flat',2)
        qaexp3 = QA_Exposure('science',2)

        #- Init SkySub dict
        qaexp2 = QA_Exposure('science',2)
        qaexp2.init_skysub('b0')
        assert qaexp2._data['b0']['SKYSUB']['NSKY_FIB'] == 0


    def runTest(self):
        pass
                
if __name__ == '__main__':
    unittest.main()
