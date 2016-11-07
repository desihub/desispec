"""
Test desispec.util.*
"""

import os
import unittest
from uuid import uuid4
import imp

import numpy as np

from desispec import util
import desispec.parallel as dpl

class TestNight(unittest.TestCase):
    
    def test_night(self):
        self.assertEqual(util.ymd2night(2015, 1, 2), '20150102')
        self.assertEqual(util.night2ymd('20150102'), (2015, 1, 2))
        self.assertRaises(ValueError, util.night2ymd, '20150002')
        self.assertRaises(ValueError, util.night2ymd, '20150100')
        self.assertRaises(ValueError, util.night2ymd, '20150132')
        self.assertRaises(ValueError, util.night2ymd, '20151302')

    def test_mask32(self):
        for dtype in (
            int, 'int64', 'uint64', 'i8', 'u8',
            'int32', 'uint32', 'i4', 'u4',
            'int16', 'uint16', 'i2', 'u2',
            'int8', 'uint8', 'i1', 'u1',
            ):
            x = np.ones(10, dtype=np.dtype(dtype))
            m32 = util.mask32(x)                
            self.assertTrue(np.all(m32 == 1))
            
        x = util.mask32( np.array([-1,0,1], dtype='i4') )
        self.assertEqual(x[0], 2**32-1)
        self.assertEqual(x[1], 0)
        self.assertEqual(x[2], 1)
        
        with self.assertRaises(ValueError):
            util.mask32(np.arange(2**35, 2**35+5))

        with self.assertRaises(ValueError):
            util.mask32(np.arange(-2**35, -2**35+5))

    def test_combine_ivar(self):
        #- input inverse variances with some zeros (1D)
        ivar1 = np.random.uniform(-1, 10, size=200).clip(0)
        ivar2 = np.random.uniform(-1, 10, size=200).clip(0)
        ivar = util.combine_ivar(ivar1, ivar2)
        izero = np.where(ivar1 == 0)
        self.assertTrue(np.all(ivar[izero] == 0))
        izero = np.where(ivar2 == 0)
        self.assertTrue(np.all(ivar[izero] == 0))
        self.assertTrue(ivar.dtype == np.float64)
        
        #- input inverse variances with some zeros (2D)
        np.random.seed(0)
        ivar1 = np.random.uniform(-1, 10, size=(10,20)).clip(0)
        ivar2 = np.random.uniform(-1, 10, size=(10,20)).clip(0)
        ivar = util.combine_ivar(ivar1, ivar2)
        izero = np.where(ivar1 == 0)
        self.assertTrue(np.all(ivar[izero] == 0))
        izero = np.where(ivar2 == 0)
        self.assertTrue(np.all(ivar[izero] == 0))
        self.assertTrue(ivar.dtype == np.float64)

        #- Dimensionality
        self.assertRaises(AssertionError, util.combine_ivar, ivar1, ivar2[0])

        #- ivar must be positive
        self.assertRaises(AssertionError, util.combine_ivar, -ivar1, ivar2)
        self.assertRaises(AssertionError, util.combine_ivar, ivar1, -ivar2)
        
        #- does it actually combine them correctly?
        ivar = util.combine_ivar(1, 2)
        self.assertEqual(ivar, 1.0/(1.0 + 0.5))
        
        #- float -> float, int -> float, 0-dim ndarray -> 0-dim ndarray
        ivar = util.combine_ivar(1, 2)
        self.assertTrue(isinstance(ivar, float))
        ivar = util.combine_ivar(1.0, 2.0)
        self.assertTrue(isinstance(ivar, float))
        ivar = util.combine_ivar(np.asarray(1.0), np.asarray(2.0))
        self.assertTrue(isinstance(ivar, np.ndarray))
        self.assertEqual(ivar.ndim, 0)


#- TODO: override log level to quiet down error messages that are supposed
#- to be there from these tests
class TestRunCmd(unittest.TestCase):
    
    def test_runcmd(self):
        self.assertEqual(0, util.runcmd('echo hello > /dev/null'))

    def test_missing_inputs(self):
        cmd = 'echo hello > /dev/null'
        self.assertNotEqual(0, util.runcmd(cmd, inputs=[uuid4().hex]))

    def test_existing_inputs(self):
        cmd = 'echo hello > /dev/null'
        self.assertEqual(0, util.runcmd(cmd, inputs=[self.infile]))

    def test_missing_outputs(self):
        cmd = 'echo hello > /dev/null'
        self.assertNotEqual(0, util.runcmd(cmd, outputs=[uuid4().hex]))

    def test_existing_outputs(self):
        token = uuid4().hex
        cmd = 'echo {} > {}'.format(token, self.testfile)
        self.assertEqual(0, util.runcmd(cmd, outputs=[self.outfile]))
        fx = open(self.testfile)
        line = fx.readline().strip()
        #- command should not have run, so tokens should not be equal
        self.assertNotEqual(token, line)

    def test_clobber(self):
        token = uuid4().hex
        cmd = 'echo {} > {}'.format(token, self.testfile)
        self.assertEqual(0, util.runcmd(cmd, outputs=[self.outfile], clobber=True))
        fx = open(self.testfile)
        line = fx.readline().strip()
        #- command should have run, so tokens should be equal
        self.assertEqual(token, line)

    def test_function(self):
        def blat(*args):
            return list(args)
        
        self.assertEqual(util.runcmd(blat, args=[1,2,3]), [1,2,3])
        self.assertEqual(util.runcmd(blat), [])

    def test_zz(self):
        """
        Even if clobber=False and outputs exist, run cmd if inputs are
        newer than outputs.  Run this test last since it alters timestamps.
        """
        #- update input timestamp to be newer than output
        fx = open(self.infile, 'w')
        fx.write('This file is leftover from a test; you can remove it\n')
        fx.close()
        
        #- run a command
        token = uuid4().hex
        cmd = 'echo {} > {}'.format(token, self.testfile)
        self.assertEqual(0, util.runcmd(cmd, outputs=[self.outfile], clobber=False))

        #- command should have run even though outputs exist,
        #- so tokens should be equal
        fx = open(self.testfile)
        line = fx.readline().strip()        
        self.assertNotEqual(token, line)
        
    def test_utils_default_nproc(self):
        n = 4
        tmp = os.getenv('SLURM_CPUS_PER_TASK')
        os.environ['SLURM_CPUS_PER_TASK'] = str(n)
        imp.reload(dpl)
        self.assertEqual(dpl.default_nproc, n)
        os.environ['SLURM_CPUS_PER_TASK'] = str(2*n)
        imp.reload(dpl)
        self.assertEqual(dpl.default_nproc, 2*n)
        del os.environ['SLURM_CPUS_PER_TASK']
        imp.reload(dpl)
        import multiprocessing
        self.assertEqual(dpl.default_nproc, multiprocessing.cpu_count()//2)

    @classmethod
    def setUpClass(cls):
        cls.infile = 'test-'+uuid4().hex
        cls.outfile = 'test-'+uuid4().hex
        cls.testfile = 'test-'+uuid4().hex
        for filename in [cls.infile, cls.outfile]:
            fx = open(filename, 'w')
            fx.write('This file is leftover from a test; you can remove it\n')
            fx.close()

    @classmethod
    def tearDownClass(cls):
        for filename in [cls.infile, cls.outfile, cls.testfile]:
            if os.path.exists(filename):
                os.remove(filename)

        
if __name__ == '__main__':
    unittest.main()
        