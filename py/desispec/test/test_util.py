"""
Test desispec.util.*
"""

import os
import time
import unittest
from uuid import uuid4
import importlib

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

    def test_parse_fibers(self):
        """
        test the util func parse_fibers
        """
        str1 = '0:10'
        str2 = '1,2,3,4:8'
        str3 = '1..5,6,7,8:10,11-14'
        arr1 = np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9])
        arr2  = np.array([1, 2, 3, 4, 5, 6, 7])
        arr3 = np.array([ 1,  2,  3,  4,  6,  7,  8,  9, 11, 12, 13])
        for instr,arr in zip([str1,str2,str3],
                             [arr1,arr2,arr3]):
            returned_arr = util.parse_fibers(instr)
            self.assertEqual(len(returned_arr),len(arr))
            for v1,v2 in zip(returned_arr,arr):
                self.assertEqual(int(v1),int(v2))

        arr1 = util.parse_fibers('0-3', include_end=True)
        self.assertTrue(np.all(arr1 == np.array([0,1,2,3])))
        arr2 = util.parse_fibers('0-3,6-8', include_end=True)
        self.assertTrue(np.all(arr2 == np.array([0,1,2,3, 6,7,8])))

#- TODO: override log level to quiet down error messages that are supposed
#- to be there from these tests
class TestRunCmd(unittest.TestCase):
    
    def test_runcmd(self):
        """Test calling a script"""
        result, success = util.runcmd('echo hello > /dev/null')
        self.assertEqual(result, 0)
        self.assertTrue(success)

    def test_command_args(self):
        """Test calling a script with args"""
        result, success = util.runcmd('echo', ['hello', 'biz', 'bat'])
        self.assertEqual(result, 0)
        self.assertTrue(success)

    def test_failed_script(self):
        """Test a script call that should fail"""
        result, success = util.runcmd('blargbitbatfoo')
        self.assertNotEqual(result, 0)
        self.assertFalse(success)

    def test_failed_function(self):
        """Test a function call returning an exception"""
        def blat():
            raise ValueError

        result, success = util.runcmd(blat)
        self.assertTrue(isinstance(result, ValueError))
        self.assertFalse(success)

    def test_missing_inputs(self):
        """test failure from missing inputs"""
        token = uuid4().hex
        cmd = f'echo {token} > {self.testfile}'
        result, success = util.runcmd(cmd, inputs=[uuid4().hex,])
        self.assertNotEqual(result, 0)
        self.assertFalse(success)

        # command should not have run, so token should not be in file
        with open(self.testfile) as fx:
            line = fx.readline().strip()

        self.assertNotEqual(token, line)

    def test_existing_inputs(self):
        """test success with existing inputs"""
        token = uuid4().hex
        cmd = f'echo {token} > {self.testfile}'
        result, success = util.runcmd(cmd, inputs=[self.infile])
        self.assertEqual(result, 0)
        self.assertTrue(success)

        # command should have run, so token should be in file
        with open(self.testfile) as fx:
            line = fx.readline().strip()

        self.assertEqual(token, line)

    def test_missing_outputs(self):
        """Test (purposefully) missing outputs"""
        token = uuid4().hex
        cmd = f'echo {token} > {self.testfile}'
        result, success = util.runcmd(cmd, outputs=[uuid4().hex])
        #- command ran (result=0) but fake outputs don't exist (success=False)
        self.assertEqual(result, 0)
        self.assertFalse(success)

        #- since command ran, token shoudl be in file
        with open(self.testfile) as fx:
            line = fx.readline().strip()

        self.assertEqual(token, line)

    def test_existing_outputs(self):
        """Test skipping if outputs alredy exist"""
        token = uuid4().hex
        cmd = 'echo {} > {}'.format(token, self.testfile)
        result, success = util.runcmd(cmd, outputs=[self.outfile,])
        #- outputs exist = skipped test = result=None, success=True
        self.assertEqual(result, None)
        self.assertTrue(success)

        #- command should not have run, so tokens should not be equal
        with open(self.testfile) as fx:
            line = fx.readline().strip()
        self.assertNotEqual(token, line)

    def test_clobber(self):
        """Test overwriting output if clobber=True"""
        token = uuid4().hex
        cmd = 'echo {} > {}'.format(token, self.testfile)
        result, success = util.runcmd(cmd, outputs=[self.outfile], clobber=True)
        self.assertEqual(result, 0)
        self.assertTrue(success)

        #- command should have run, so tokens should be equal
        with open(self.testfile) as fx:
            line = fx.readline().strip()
        self.assertEqual(token, line)

    def test_function(self):
        """Test calling a function instead of spawning a script"""
        def blat(args='hello'):
            return args

        def foo(a, b, c):
            return a + b + c

        result, success = util.runcmd(blat, args=[1,2,3])
        self.assertEqual(result, [1,2,3])
        self.assertTrue(success)

        result, success = util.runcmd(foo, args=[1,2,3], expandargs=True)
        self.assertEqual(result, 6)
        self.assertTrue(success)

        self.assertEqual(util.runcmd(blat)[0], 'hello')

    def test_newer_input(self):
        """
        Even if clobber=False and outputs exist, run cmd if inputs are
        newer than outputs.
        """
        #- update input timestamp to be newer than output
        fx = open(self.infile, 'w')
        fx.write('This file is leftover from a test; you can remove it\n')
        fx.close()
        
        #- run a command
        token = uuid4().hex
        cmd = 'echo {} > {}'.format(token, self.testfile)
        result, success = util.runcmd(cmd, inputs=[self.infile,], outputs=[self.testfile,], clobber=False)
        self.assertEqual(result, 0)
        self.assertTrue(success)

        #- command should have run even though outputs exist,
        #- so updated token should be equal
        fx = open(self.testfile)
        line = fx.readline().strip()        
        self.assertEqual(token, line)
        
    @classmethod
    def setUpClass(cls):
        cls.infile = 'test-'+uuid4().hex
        cls.outfile = 'test-'+uuid4().hex
        cls.testfile = 'test-'+uuid4().hex

    def setUp(self):
        # refresh timestamps so that outfile is older than infile
        for filename in [self.infile, self.outfile]:
            with open(filename, 'w') as fx:
                fx.write('This file is leftover from a test; you can remove it\n')
            time.sleep(0.1)

    @classmethod
    def tearDownClass(cls):
        for filename in [cls.infile, cls.outfile, cls.testfile]:
            if os.path.exists(filename):
                os.remove(filename)

class TestUtil(unittest.TestCase):

    def test_utils_default_nproc(self):
        n = 4
        tmp = os.getenv('SLURM_CPUS_PER_TASK')
        os.environ['SLURM_CPUS_PER_TASK'] = str(n)
        importlib.reload(dpl)
        self.assertEqual(dpl.default_nproc, n)
        os.environ['SLURM_CPUS_PER_TASK'] = str(2*n)
        importlib.reload(dpl)
        self.assertEqual(dpl.default_nproc, 2*n)
        del os.environ['SLURM_CPUS_PER_TASK']
        importlib.reload(dpl)
        import multiprocessing
        
        self.assertEqual(dpl.default_nproc, max(multiprocessing.cpu_count()//2, 1))

    def test_header2night(self):
        from astropy.time import Time
        night = 20210105
        dateobs = '2021-01-06T04:33:55.704316928'
        mjd = Time(dateobs).mjd
        hdr = dict()

        #- Missing NIGHT and DATE-OBS falls back to MJD-OBS
        hdr['MJD-OBS'] = mjd
        self.assertEqual(util.header2night(hdr), night)

        #- Missing NIGHT and MJD-OBS falls back to DATE-OBS
        del hdr['MJD-OBS']
        hdr['DATE-OBS'] = dateobs
        self.assertEqual(util.header2night(hdr), night)

        #- NIGHT is NIGHT
        del hdr['DATE-OBS']
        hdr['NIGHT'] = night
        self.assertEqual(util.header2night(hdr), night)
        hdr['NIGHT'] = str(night)
        self.assertEqual(util.header2night(hdr), night)

        #- NIGHT trumps DATE-OBS
        hdr['NIGHT'] = night+1
        hdr['DATE-OBS'] = dateobs
        self.assertEqual(util.header2night(hdr), night+1)

        #- Bogus NIGHT falls back to DATE-OBS
        hdr['NIGHT'] = None
        self.assertEqual(util.header2night(hdr), night)
        hdr['NIGHT'] = '        '
        self.assertEqual(util.header2night(hdr), night)
        hdr['NIGHT'] = 'Sunday'
        self.assertEqual(util.header2night(hdr), night)

        #- Check rollover at noon KPNO (MST) = UTC 19:00
        hdr = dict()
        hdr['DATE-OBS'] = '2021-01-05T18:59:00'
        self.assertEqual(util.header2night(hdr), 20210104)
        hdr['DATE-OBS'] = '2021-01-05T19:00:01'
        self.assertEqual(util.header2night(hdr), 20210105)
        hdr['DATE-OBS'] = '2021-01-06T01:00:01'
        self.assertEqual(util.header2night(hdr), 20210105)
        hdr['DATE-OBS'] = '2021-01-06T18:59:59'
        self.assertEqual(util.header2night(hdr), 20210105)

    def test_ordered_unique(self):
        a = util.ordered_unique([1,2,3])
        self.assertEqual(list(a), [1,2,3])

        a = util.ordered_unique([2,3,1])
        self.assertEqual(list(a), [2,3,1])

        a = util.ordered_unique([2,3,2,1])
        self.assertEqual(list(a), [2,3,1])

        a = util.ordered_unique([3,2,3,2,1])
        self.assertEqual(list(a), [3,2,1])

        a, idx = util.ordered_unique([3,2,3,2,1], return_index=True)
        self.assertEqual(list(a), [3,2,1])
        self.assertEqual(list(idx), [0,1,4])

        a, idx = util.ordered_unique([1,1,2,3,0], return_index=True)
        self.assertEqual(list(a), [1,2,3,0])
        self.assertEqual(list(idx), [0,2,3,4])

    def test_itemindices(self):
        r = util.itemindices([10,30,20,30])
        self.assertEqual(r, {10: [0], 30: [1,3], 20: [2]})

        r = util.itemindices([10,30,20,30,20])
        self.assertEqual(r, {10: [0], 30: [1,3], 20: [2,4]})

        r = util.itemindices([20,10,30,20,30,20])
        self.assertEqual(r, {20: [0,3,5], 10: [1], 30: [2,4]})

    def test_parse_keyval(self):
        key, value = util.parse_keyval("BLAT=0")
        self.assertEqual(key, 'BLAT')
        self.assertEqual(value, 0)

        key, value = util.parse_keyval("BLAT=1")
        self.assertEqual(value, 1)

        key, value = util.parse_keyval("BLAT=1.0")
        self.assertEqual(type(value), float)
        self.assertEqual(value, 1.0)

        key, value = util.parse_keyval("BLAT=True")
        self.assertEqual(type(value), bool)
        self.assertEqual(value, True)

        key, value = util.parse_keyval("BLAT=False")
        self.assertEqual(type(value), bool)
        self.assertEqual(value, False)

        key, value = util.parse_keyval("BLAT=true")
        self.assertEqual(type(value), str)
        self.assertEqual(value, 'true')

        key, value = util.parse_keyval("BLAT=false")
        self.assertEqual(type(value), str)
        self.assertEqual(value, 'false')

        # trailing space preserved
        key, value = util.parse_keyval("biz=bat ")
        self.assertEqual(key, 'biz')
        self.assertEqual(value, 'bat ')

        # trailing space ignored for bool
        key, value = util.parse_keyval("biz=True ")
        self.assertEqual(type(value), bool)
        self.assertEqual(value, True)
        key, value = util.parse_keyval("biz=False  ")
        self.assertEqual(type(value), bool)
        self.assertEqual(value, False)

    def test_argmatch(self):
        #- basic argmatch
        a = np.array([1,3,2,4])
        b = np.array([3,2,1,4])
        ii = util.argmatch(a, b)
        self.assertTrue(np.all(a[ii] == b), f'{a=}, {ii=}, {a[ii]=} != {b=}')

        #- b with duplicates
        a = np.array([1,3,2,4])
        b = np.array([3,2,1,4,2,3])
        ii = util.argmatch(a, b)
        self.assertTrue(np.all(a[ii] == b), f'{a=}, {ii=}, {a[ii]=} != {b=}')
        
        #- special case already matching
        a = np.array([1,3,2,4])
        b = a.copy()
        ii = util.argmatch(a, b)
        self.assertTrue(np.all(a[ii] == b), f'{a=}, {ii=}, {a[ii]=} != {b=}')

        #- special case already sorted
        a = np.array([1,2,3,4])
        b = a.copy()
        ii = util.argmatch(a, b)
        self.assertTrue(np.all(a[ii] == b), f'{a=}, {ii=}, {a[ii]=} != {b=}')
        
        #- a with extras (before, in middle, and after range of b values)
        a = np.array([1,3,2,4,0,5])
        b = np.array([3,1,4])
        ii = util.argmatch(a, b)
        self.assertTrue(np.all(a[ii] == b), f'{a=}, {ii=}, {a[ii]=} != {b=}')

        #- a has duplicates
        a = np.array([1,3,3,2,4])
        b = np.array([3,2,1,4])
        ii = util.argmatch(a, b)
        self.assertTrue(np.all(a[ii] == b), f'{a=}, {ii=}, {a[ii]=} != {b=}')

        #- equal length arrays, not not equal values
        a = np.array([1,3,2,4])
        b = np.array([3,1,1,2])
        ii = util.argmatch(a, b)
        self.assertTrue(np.all(a[ii] == b), f'{a=}, {ii=}, {a[ii]=} != {b=}')

        a = np.array([1,3,2,4,2])
        b = np.array([3,1,1,2,4])
        ii = util.argmatch(a, b)
        self.assertTrue(np.all(a[ii] == b), f'{a=}, {ii=}, {a[ii]=} != {b=}')
        
        #- a can have extras, but not b
        a = np.array([1,3,2,4])
        b = np.array([3,2,5,4])
        with self.assertRaises(ValueError):
            ii = util.argmatch(a, b)
        
        #- Brute force random testing with shuffles
        a = np.arange(10)
        b = a.copy()
        for test in range(100):
            np.random.shuffle(a)
            np.random.shuffle(b)
            ii = util.argmatch(a,b)
            self.assertTrue(np.all(a[ii] == b), f'test number {test}\n{a=}\n{ii=}\n{a[ii]=} !=\n{b=}')

        #- Brute force random testing with repeats and extras
        for test in range(100):
            a = np.random.randint(0,20, size=50)
            b = np.random.randint(5,15, size=51)

            #- all values in b must be in a, so remove extras in b
            #- Note: extras in a is ok, just not in b
            keep = np.isin(b, a)
            b = b[keep]
            
            ii = util.argmatch(a,b)
            self.assertTrue(np.all(a[ii] == b), f'test number {test}\n{a=}\n{ii=}\n{a[ii]=} !=\n{b=}')

