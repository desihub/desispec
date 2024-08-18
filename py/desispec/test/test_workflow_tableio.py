"""
Test desispec.workflow.tableio and tables readers in exptable and proctable
"""

import os, glob, tempfile, shutil
import unittest
from importlib import resources

import numpy as np
from astropy.table import Table, vstack

from desispec.workflow import tableio

class TestWorkflowTableIO(unittest.TestCase):
    """Test desispec.workflow.tableio and friends
    """

    @classmethod
    def setUpClass(cls):
        cls.testdir = tempfile.mkdtemp()
        cls.specprod = 'daily'  #- because that's where the example processing tables are from

        proddir = os.path.join(cls.testdir, cls.specprod)
        os.makedirs(proddir)

        exampledir = str(resources.files('desispec').joinpath('test/data'))
        shutil.copytree(f'{exampledir}/exposure_tables', f'{proddir}/exposure_tables')
        shutil.copytree(f'{exampledir}/processing_tables', f'{proddir}/processing_tables')

        cls.origenv = dict()
        cls.origenv['DESI_SPECTRO_REDUX'] = os.getenv('DESI_SPECTRO_REDUX')
        cls.origenv['SPECPROD'] = os.getenv('SPECPROD')

        os.environ['DESI_SPECTRO_REDUX'] = cls.testdir
        os.environ['SPECPROD'] = cls.specprod

        cls.exptabfiles = sorted(glob.glob(f'{proddir}/exposure_tables/??????/exposure_table*.csv'))
        cls.proctabfiles = sorted(glob.glob(f'{proddir}/processing_tables/processing*.csv'))

    def setUp(self):
        pass

    def tearDown(self):
        pass

    @classmethod
    def tearDownClass(cls):
        #- Remove testdir if it was created with tempfile.mkdtemp
        if cls.testdir.startswith(tempfile.gettempdir()) and os.path.isdir(cls.testdir):
            shutil.rmtree(cls.testdir)

        #- Restore environment variables
        for key, value in cls.origenv.items():
            if value is not None:
                os.environ[key] = value
            elif key in os.environ:
                del os.environ[key]


    def test_exptable_io(self):
        """Test tableio on exposure tables"""
        for filename in self.exptabfiles:
            etab = tableio.load_table(filename)

    def test_proctable_io(self):
        """Test tableio on processing tables"""
        for filename in self.proctabfiles:
            ptab = tableio.load_table(filename)

    def test_read_minimal_exptables(self):
        """test workflow.exptable.read_minimal_science_exptab_cols"""
        from desispec.workflow.exptable import read_minimal_science_exptab_cols
        allexp = read_minimal_science_exptab_cols()

        #- pre-trimmed to OBSTYPE=science LASTSTEP=all
        self.assertTrue(np.all(allexp['TILEID']>0))

        #- test night and tileid filtering
        night = np.unique(allexp['NIGHT'])[2]
        tileid = np.unique(allexp['TILEID'])[2]

        exps2 = read_minimal_science_exptab_cols(nights=[night,])
        self.assertTrue(np.all(exps2['NIGHT'] == night))

        exps3 = read_minimal_science_exptab_cols(tileids=[tileid,])
        self.assertTrue(np.all(exps3['TILEID'] == tileid))


    def test_read_minimal_proctables(self):
        """test workflow.exptable.read_minimal_science_exptab_cols"""
        from desispec.workflow.proctable import read_minimal_full_proctab_cols, read_minimal_tilenight_proctab_cols
        allproc = read_minimal_full_proctab_cols()
        self.assertGreater(len(set(allproc['JOBDESC'])), 1)

        allproc = read_minimal_tilenight_proctab_cols()
        print('BLAT', allproc)
        self.assertEqual(set(allproc['JOBDESC']), set(['tilenight',]))

        #- test filtering by night and tileid
        self.assertGreater(len(set(allproc['NIGHT'])), 1)
        self.assertGreater(len(set(allproc['TILEID'])), 1)

        night = np.unique(allproc['NIGHT'])[1]
        tileid = np.unique(allproc['TILEID'])[1]

        allproc = read_minimal_tilenight_proctab_cols(nights=[night,])
        self.assertTrue(np.all(allproc['NIGHT'] == night))

        allproc = read_minimal_tilenight_proctab_cols(tileids=[tileid,])
        self.assertTrue(np.all(allproc['TILEID'] == tileid))


