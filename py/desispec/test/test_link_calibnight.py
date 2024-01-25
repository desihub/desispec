"""
test desispec.scripts.link_calibnight
"""

import os, sys, unittest, tempfile
from shutil import rmtree

from desispec.io import findfile
from desispec.io.util import decode_camword
from desispec.scripts import link_calibnight

class TestLinkCalibNight(unittest.TestCase):
    """Test desispec.scripts.link_calibnight"""

    @classmethod
    def setUpClass(cls):
        #- cache environment variables so that we can reset originals when done
        cls.cache_env = dict()
        for name in ['DESI_SPECTRO_REDUX', 'SPECPROD']:
            cls.cache_env[name] = os.getenv(name)

        #- Setup a test night of calib files
        cls.testdir = tempfile.mkdtemp()
        os.environ['DESI_SPECTRO_REDUX'] = cls.testdir
        os.environ['SPECPROD'] = 'testlinks'
        cls.reduxdir = f'{cls.testdir}/testlinks'
        cls.refnight = 20201010
        cls.altrefnight = 20201020
        cls.newnight = 20201011
        os.makedirs(f'{cls.testdir}/testlinks/calibnight/{cls.refnight}')
        os.makedirs(f'{cls.testdir}/testlinks/calibnight/{cls.altrefnight}')
        cls.prefixes = ['badcolumns', 'biasnight', 'fiberflatnight', 'psfnight']
        for night in (cls.refnight, cls.altrefnight):
            for prefix in cls.prefixes:
                for camera in decode_camword('a0123456789'):
                    filename = findfile(prefix, night=night, camera=camera)
                    with open(filename, 'w') as fx:
                        fx.write(os.path.basename(filename))

    def tearDown(self):
        newdir = f'{self.reduxdir}/calibnight/{self.newnight}'
        if os.path.isdir(newdir):
            rmtree(newdir)

    @classmethod
    def tearDownClass(cls):
        #- remove files
        rmtree(cls.testdir)

        #- reset environment variables
        for name in cls.cache_env:
            if cls.cache_env[name] is None:
                del os.environ[name]
            else:
                os.environ[name] = cls.cache_env[name]

    def test_basic_links(self):
        options = f'--refnight {self.refnight} --newnight {self.newnight}'.split()
        link_calibnight.main(options)
        for prefix in self.prefixes:
            for camera in decode_camword('a0123456789'):
                reffile = findfile(prefix, night=self.refnight, camera=camera)
                newfile = findfile(prefix, night=self.newnight, camera=camera)

                self.assertNotEqual(os.path.basename(reffile), os.path.basename(newfile))
                self.assertTrue(os.path.islink(newfile))
                with open(newfile) as fp:
                    contents = fp.readline().strip()
                    self.assertEqual(contents, os.path.basename(reffile))

    def test_link_cameras(self):
        """test -c/--cameras option to link a subset of cameras"""

        #- should link cameras a12 but not a03456789
        options = f'--refnight {self.refnight} --newnight {self.newnight} --c a12'.split()
        link_calibnight.main(options)
        for prefix in self.prefixes:
            for camera in decode_camword('a12'):
                newfile = findfile(prefix, night=self.newnight, camera=camera)
                self.assertTrue(os.path.islink(newfile))
            for camera in decode_camword('a03456789'):
                newfile = findfile(prefix, night=self.newnight, camera=camera)
                self.assertFalse(os.path.islink(newfile))

    def test_link_include(self):
        """test --include option to link a subset of prefixes"""

        #- link biasnight and fiberflatnight, but not the other prefixes
        options = f'--refnight {self.refnight} --newnight {self.newnight} --include biasnight,fiberflatnight'.split()
        link_calibnight.main(options)
        for prefix in self.prefixes:
            for camera in decode_camword('a0123456789'):
                newfile = findfile(prefix, night=self.newnight, camera=camera)
                if prefix in ('biasnight', 'fiberflatnight'):
                    self.assertTrue(os.path.islink(newfile))
                else:
                    self.assertFalse(os.path.islink(newfile))


    def test_link_exclude(self):
        """test --include option to link a subset of prefixes"""

        #- link all prefixes except biasnight and fiberflatnight
        options = f'--refnight {self.refnight} --newnight {self.newnight} --exclude biasnight,fiberflatnight'.split()
        link_calibnight.main(options)
        for prefix in self.prefixes:
            for camera in decode_camword('a0123456789'):
                newfile = findfile(prefix, night=self.newnight, camera=camera)
                if prefix in ('biasnight', 'fiberflatnight'):
                    self.assertFalse(os.path.islink(newfile))
                else:
                    self.assertTrue(os.path.islink(newfile))

    def test_dryrun(self):
        """test --dryrun doesn't make any links"""

        options = f'--refnight {self.refnight} --newnight {self.newnight} --dryrun'.split()
        link_calibnight.main(options)
        for prefix in self.prefixes:
            for camera in decode_camword('a0123456789'):
                newfile = findfile(prefix, night=self.newnight, camera=camera)
                self.assertFalse(os.path.islink(newfile))

    def test_missing(self):
        """test missing file behavior"""

        #- linking to a night with missing files should fail
        options = f'--refnight {self.refnight-1} --newnight {self.newnight}'.split()
        with self.assertRaises(SystemExit):
            link_calibnight.main(options)

    def test_preexisting(self):
        """test pre-existing file behavior"""

        #- Harmless to run twice making the same links
        options = f'--refnight {self.refnight} --newnight {self.newnight}'.split()
        link_calibnight.main(options)
        link_calibnight.main(options)

        #- by trying to change links should fail
        #- ... first link to refnight
        options = f'--refnight {self.refnight} --newnight {self.newnight}'.split()
        link_calibnight.main(options)
        #- ... then try to change to link to altrefnight
        options = f'--refnight {self.altrefnight} --newnight {self.newnight}'.split()
        with self.assertRaises(SystemExit):
            link_calibnight.main(options)

        #- and trying to replace files with links should also fail
        options = f'--refnight {self.refnight} --newnight {self.altrefnight}'.split()
        with self.assertRaises(SystemExit):
            link_calibnight.main(options)

