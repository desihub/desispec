# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""Test desispec.tilecompleteness.
"""
import os
import unittest
from unittest.mock import patch, call
from ..tilecompleteness import read_gfa_data


class TestTileCompleteness(unittest.TestCase):
    """Test desispec.tilecompleteness.
    """

    @patch('desispec.tilecompleteness.vstack')
    @patch('desispec.tilecompleteness.read_table')
    @patch('desispec.tilecompleteness.glob')
    @patch('desispec.tilecompleteness.get_logger')
    def test_read_gfa_data(self, mock_log, mock_glob, mock_table, mock_vstack):
        """Test identification of the most recent GFA file.
        """
        gfa_proc_dir = '/global/cfs/cdirs/desi/survey/GFA'
        mock_glob.glob.side_effect = [[os.path.join(gfa_proc_dir, 'offline_matched_coadd_ccds_SV1-thru_20201231.fits'),],
                                      [os.path.join(gfa_proc_dir, 'offline_matched_coadd_ccds_SV1-thru_20201231.fits'),
                                       os.path.join(gfa_proc_dir, 'offline_matched_coadd_ccds_SV2-thru_20210101.fits'),
                                       os.path.join(gfa_proc_dir, 'offline_matched_coadd_ccds_SV3-thru_20210102.fits'),
                                       os.path.join(gfa_proc_dir, 'offline_matched_coadd_ccds_main-thru_20210103.fits'),
                                       os.path.join(gfa_proc_dir, 'offline_matched_coadd_ccds_foobar-thru_20210104.fits'),]]
        mock_table.side_effect = [[(1, 2, 3, 4),], [(1, 2, 3, 4), (1, 2, 3, 4), (1, 2, 3, 4),]]
        mock_vstack.return_value = [(1, 2, 3, 4), (1, 2, 3, 4), (1, 2, 3, 4), (1, 2, 3, 4)]
        table = read_gfa_data(gfa_proc_dir)
        self.assertListEqual(table, [(1, 2, 3, 4), (1, 2, 3, 4), (1, 2, 3, 4), (1, 2, 3, 4)])
        mock_table.assert_has_calls([call(os.path.join(gfa_proc_dir, 'offline_matched_coadd_ccds_SV1-thru_20201231.fits'), 2),
                                     call(os.path.join(gfa_proc_dir, 'offline_matched_coadd_ccds_foobar-thru_20210104.fits'), 2),])
        mock_log().info.assert_has_calls([call("Reading SV1 file: %s", os.path.join(gfa_proc_dir, 'offline_matched_coadd_ccds_SV1-thru_20201231.fits')),
                                          call('%d GFA SV1 table entries', 1),
                                          call("Reading most recent file: %s", os.path.join(gfa_proc_dir, 'offline_matched_coadd_ccds_foobar-thru_20210104.fits')),
                                          call('%d GFA table entries', 3),
                                          call('%d GFA merged table entries', 4)])

    @patch('desispec.tilecompleteness.glob')
    @patch('desispec.tilecompleteness.get_logger')
    def test_read_gfa_data_no_files(self, mock_log, mock_glob):
        """Test identification of the most recent GFA file.
        """
        gfa_proc_dir = '/global/cfs/cdirs/desi/survey/GFA'
        mock_glob.glob.return_value = []
        with self.assertRaises(RuntimeError) as e:
            table = read_gfa_data(gfa_proc_dir)
        self.assertEqual(str(e.exception), "did not find any file offline_matched_coadd_ccds_*-thru_????????.fits in %s" % gfa_proc_dir)
        mock_log().warning.assert_called_once_with("Could not find a SV1 file, skipping!")
        mock_log().critical.assert_called_once_with("did not find any file offline_matched_coadd_ccds_*-thru_????????.fits in %s", gfa_proc_dir)
