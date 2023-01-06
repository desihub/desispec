"""
tests desispec.gpu
"""

import os
import unittest
import desispec.gpu

class TestGPU(unittest.TestCase):
    """
    Unit tests for interpolation.resample_flux
    """

    def test_gpu_context(self):
        before = desispec.gpu.is_gpu_available()
        with desispec.gpu.NoGPU():
            use_gpu = desispec.gpu.is_gpu_available()

        after = desispec.gpu.is_gpu_available()

        #- test system may or may not have a GPU,
        #- but these should agree either way
        self.assertEqual(before, after)

        #- during context manager, there shouldn't be a GPU
        self.assertFalse(use_gpu)

    def test_gpu_envvar(self):
        """Test if setting $DESI_NO_GPU disables GPU usage

        This is a real test only on a system that has a GPU to disable"""
        desi_no_gpu = os.getenv('DESI_NO_GPU')
        os.environ['DESI_NO_GPU'] = '1'
        self.assertFalse(desispec.gpu.is_gpu_available())
        if desi_no_gpu is not None:
            os.environ['DESI_NO_GPU'] = desi_no_gpu

    def test_doesnt_crash(self):
        """At minimum, these shouldn't crash with or without a GPU"""
        desispec.gpu.free_gpu_memory()
        desispec.gpu.redistribute_gpu_ranks(comm=None)

        class FakeComm:
            def __init__(self, rank, size):
                self.rank = rank
                self.size = size

            def gather(self, value, root):
                return [value,]*self.size

        desispec.gpu.redistribute_gpu_ranks(comm=FakeComm(0,16))
        desispec.gpu.redistribute_gpu_ranks(comm=FakeComm(1,16))




#- This runs all test* functions in any TestCase class in this file
if __name__ == '__main__':
    unittest.main()           
