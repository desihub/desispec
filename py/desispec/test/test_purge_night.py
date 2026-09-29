"""
test desispec.scripts.purge_night
"""

import os
import unittest
import tempfile
from shutil import rmtree
from unittest.mock import patch

import numpy as np
from astropy.table import Table

from desispec.io.meta import findfile
from desispec.scripts.purge_night import purge_night


class TestPurgeNight(unittest.TestCase):
    """Test desispec.scripts.purge_night"""

    @classmethod
    def setUpClass(cls):
        #- cache environment variables so that we can reset originals when done
        cls.cache_env = dict()
        for name in ['DESI_SPECTRO_REDUX', 'SPECPROD']:
            cls.cache_env[name] = os.getenv(name)
        cls.origdir = os.getcwd()

    @classmethod
    def tearDownClass(cls):
        for name, value in cls.cache_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        os.chdir(cls.origdir)

    def setUp(self):
        self.reduxdir = tempfile.mkdtemp()
        os.environ['DESI_SPECTRO_REDUX'] = self.reduxdir
        os.environ['SPECPROD'] = 'test'
        self.night = 20250101
        #- purge_night only checks that the exposure table exists; its
        #- contents come from the mocked load_table
        epathname = findfile('exposure_table', night=self.night)
        os.makedirs(os.path.dirname(epathname))
        open(epathname, 'w').close()
        self.etable = Table()
        self.etable['TILEID'] = [2000, 1000, 1000, 1000, 3000, 4000]
        self.etable['EXPID'] = [1, 2, 3, 4, 5, 6]
        self.etable['OBSTYPE'] = ['science', 'science', 'science', 'science', 'science', 'arc']
        self.etable['LASTSTEP'] = ['all', 'all', 'all', 'all', 'skysub', 'all']

    def tearDown(self):
        os.chdir(self.origdir)
        if os.path.isdir(self.reduxdir):
            rmtree(self.reduxdir)

    def _run_purge_night(self, **kwargs):
        """Run purge_night with a mocked purge_tilenight and return the mock"""
        with patch('desispec.scripts.purge_night.load_table', return_value=self.etable), \
                patch('desispec.scripts.purge_night.purge_tilenight') as mock_purge:
            purge_night(self.night, **kwargs)
        mock_purge.assert_called_once()
        return mock_purge

    def test_no_attic_forwarded(self):
        """no_attic and dry_run are passed through to purge_tilenight"""
        for dry_run in [True, False]:
            for no_attic in [True, False]:
                mock_purge = self._run_purge_night(dry_run=dry_run, no_attic=no_attic)
                self.assertEqual(mock_purge.call_args.kwargs['dry_run'], dry_run)
                self.assertEqual(mock_purge.call_args.kwargs['no_attic'], no_attic)

    def test_unique_tiles(self):
        """Each selected tile is passed to purge_tilenight once"""
        mock_purge = self._run_purge_night(dry_run=True)
        tiles, night = mock_purge.call_args.args
        self.assertEqual(night, self.night)
        self.assertEqual(list(tiles), [1000, 2000])
