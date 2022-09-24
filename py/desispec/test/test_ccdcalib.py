import unittest
import numpy as np
import os


class TestCcdCalib(unittest.TestCase):

    def test_select_zero_expids(self):
        """Test select_zero_expids"""
        original_log_level = os.getenv('DESI_LOGLEVEL')
        os.environ['DESI_LOGLEVEL'] = 'CRITICAL'

        from ..ccdcalib import select_zero_expids
        night = 20000101
        cam = 'b0'

        ## Test standard case of 25 calibs with first two removed
        calib_exps = np.arange(3,26)
        noncalib_exps = np.arange(28, 36)
        expids = select_zero_expids(calib_exps, noncalib_exps, night, cam,
                            nzeros=25, minzeros=15, nskip=2, anyzeros=False)
        self.assertListEqual(calib_exps.tolist(),expids.tolist())

        ## Test case of 25 calibs
        calib_exps = np.arange(1,26)
        noncalib_exps = np.arange(30, 40)
        expids = select_zero_expids(calib_exps, noncalib_exps, night, cam,
                            nzeros=25, minzeros=15, nskip=2, anyzeros=False)
        self.assertListEqual(calib_exps.tolist(),expids.tolist())

        ## Test case of 28 calibs
        calib_exps = np.arange(1,29)
        noncalib_exps = np.arange(30, 40)
        expids = select_zero_expids(calib_exps, noncalib_exps, night, cam,
                            nzeros=25, minzeros=15, nskip=2, anyzeros=False)
        self.assertNotEqual(len(calib_exps),len(expids))
        self.assertEqual(len(expids), 25)

        ## Test case of 15 calibs and plenty of noncals
        calib_exps = np.arange(1,16)
        noncalib_exps = np.arange(30, 40)
        expids = select_zero_expids(calib_exps, noncalib_exps, night, cam,
                            nzeros=25, minzeros=15, nskip=2, anyzeros=False)
        self.assertEqual(len(expids),25-2)
        self.assertListEqual(calib_exps.tolist(), expids[:15].tolist())

        ## Test case of 15 calibs and fewer than enough noncals
        calib_exps = np.arange(1,16)
        noncalib_exps = np.arange(30, 36)
        expids = select_zero_expids(calib_exps, noncalib_exps, night, cam,
                            nzeros=25, minzeros=15, nskip=2, anyzeros=False)
        self.assertEqual(len(expids),21)
        self.assertListEqual(calib_exps.tolist(), expids[:15].tolist())

        ## Test case of 12 calibs and just enough noncalibs
        calib_exps = np.arange(1,13)
        noncalib_exps = np.arange(30, 52)
        expids = select_zero_expids(calib_exps, noncalib_exps, night, cam,
                            nzeros=25, minzeros=15, nskip=2, anyzeros=False)
        self.assertEqual(len(expids),25-2)
        self.assertListEqual(calib_exps.tolist(), expids[:12].tolist())

        ## Test case of 12 calibs and just enough noncalibs
        calib_exps = np.arange(1,13)
        noncalib_exps = np.arange(30, 42)
        expids = select_zero_expids(calib_exps, noncalib_exps, night, cam,
                            nzeros=25, minzeros=15, nskip=2, anyzeros=False)
        self.assertEqual(len(expids),25-2)
        self.assertListEqual(calib_exps.tolist(), expids[:12].tolist())

        ## Test case of 12 calibs and less than max noncalibs
        calib_exps = np.arange(1,13)
        noncalib_exps = np.arange(30, 40)
        expids = select_zero_expids(calib_exps, noncalib_exps, night, cam,
                            nzeros=25, minzeros=15, nskip=2, anyzeros=False)
        self.assertEqual(len(expids),22)
        self.assertListEqual(calib_exps.tolist(), expids[:12].tolist())

        ## Test case of 15 calibs and no noncalibs
        calib_exps = np.arange(1,16)
        noncalib_exps = np.array([])
        expids = select_zero_expids(calib_exps, noncalib_exps, night, cam,
                            nzeros=25, minzeros=15, nskip=2, anyzeros=False)
        self.assertEqual(len(expids), 15)
        self.assertListEqual(calib_exps.tolist(), expids.tolist())

        ## Test case of 0 calibs and 15 noncalibs
        calib_exps = np.array([])
        noncalib_exps = np.arange(1,16)
        expids = select_zero_expids(calib_exps, noncalib_exps, night, cam,
                            nzeros=25, minzeros=15, nskip=2, anyzeros=False)
        self.assertEqual(len(expids), 15)
        self.assertListEqual(noncalib_exps.tolist(), expids.tolist())

        ## Test case of less than minzeros total
        calib_exps = np.arange(1,9)
        noncalib_exps = np.arange(30, 32)
        expids = select_zero_expids(calib_exps, noncalib_exps, night, cam,
                            nzeros=25, minzeros=15, nskip=2, anyzeros=False)
        self.assertIsNone(expids)

        ## Test case of 12 calibs and no noncalibs
        calib_exps = np.arange(1,13)
        noncalib_exps = np.array([])
        expids = select_zero_expids(calib_exps, noncalib_exps, night, cam,
                            nzeros=25, minzeros=15, nskip=2, anyzeros=False)
        self.assertIsNone(expids)

        if original_log_level is None:
            del os.environ['DESI_LOGLEVEL']
        else:
            os.environ['DESI_LOGLEVEL'] = original_log_level


def test_suite():
    """Allows testing of only this module with the command::

        python setup.py test -m desispec.test.test_ccdcalib
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)


if __name__ == '__main__':
    unittest.main()           
