import unittest

from desispec.maskbits import spmask, ccdmask, fibermask

class TestBitMasks(unittest.TestCase):
    
    #- Which bitmasks to test
    def setUp(self):
        self.masks = [spmask, ccdmask, fibermask]
            
    def test_names(self):
        """
        Test consistency for names to bits to masks
        """
        for m in self.masks:
            for name in m.names():
                self.assertEqual(m.mask(name), 2**m.bitnum(name), 'Failed matching mask to bitnum for '+name)
                self.assertEqual(m.mask(name), m.mask(m.bitnum(name)), 'Failed matching mask to name for '+name)
                self.assertEqual(m.bitname(m.bitnum(name)), name, 'Failed bit name->num->name roundtrip for '+name)
                c = m.comment(name)
                
if __name__ == '__main__':
    unittest.main()