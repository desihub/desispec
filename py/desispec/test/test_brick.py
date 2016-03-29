import unittest

import desispec.brick
import numpy as np

class TestBrick(unittest.TestCase):
    
    def setUp(self):
        n = 10
        self.ra = np.linspace(0, 3, n) - 1.5
        self.dec = np.linspace(0, 3, n) - 1.5
        self.names = np.array(
            ['3587m015', '3592m010', '3597m010', '3597m005', '0002p000',
            '0002p000', '0007p005', '0007p010', '0012p010', '0017p015'])
            
    def test1(self):
        brickname = desispec.brick.brickname(0, 0)
        self.assertEqual(brickname, '0002p000')

    def test2(self):
        brickname = desispec.brick.brickname(self.ra, self.dec)
        self.assertEqual(len(brickname), len(self.ra))
        self.assertTrue(np.all(brickname == self.names))

    def test3(self):
        brickname = desispec.brick.brickname(np.array(self.ra), np.array(self.dec))
        self.assertEqual(len(brickname), len(self.ra))
        self.assertTrue(np.all(brickname == self.names))
                
if __name__ == '__main__':
    unittest.main()
