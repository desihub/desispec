"""
test desispec.scripts.purge_tilenight
"""

import os
import stat
import unittest
import tempfile
from shutil import rmtree
from unittest.mock import patch

from astropy.table import Table

from desispec.scripts.purge_tilenight import move_to_attic, remove_directory, purge_tilenight


def _write(filename, content):
    """Write content to filename, creating parent directories as needed"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as fx:
        fx.write(content)


def _read(filename):
    """Return the contents of filename"""
    with open(filename) as fx:
        return fx.read()


class TestPurgeTileNight(unittest.TestCase):
    """Test desispec.scripts.purge_tilenight"""

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
        self.prodroot = os.path.join(self.reduxdir, 'test')
        self.nightdir = os.path.join(self.prodroot, 'preproc', '20250101')
        self.atticdir = os.path.join(self.prodroot, 'attic', 'preproc', '20250101')
        _write(os.path.join(self.nightdir, '00001234', 'a.fits'), 'new_a')
        _write(os.path.join(self.nightdir, '00001234', 'b.fits'), 'new_b')
        _write(os.path.join(self.nightdir, '00001235', 'c.fits'), 'new_c')
        os.symlink('../00001234/a.fits', os.path.join(self.nightdir, '00001235', 'link.fits'))

        #- archived cumulative tile dir and link left by desi_archive_tilenight
        self.archivedir = os.path.join(self.prodroot, 'tiles', 'archive', '1000', '20250105')
        self.archivefile = os.path.join(self.archivedir, 'redrock-0-1000-thru20250101.fits')
        _write(self.archivefile, 'archived')
        self.linkdir = os.path.join(self.prodroot, 'tiles', 'cumulative', '1000', '20250101')
        os.makedirs(os.path.dirname(self.linkdir))
        os.symlink('../../archive/1000/20250105', self.linkdir)
        self.atticlink = os.path.join(self.prodroot, 'attic', 'tiles', 'cumulative', '1000', '20250101')

    def tearDown(self):
        os.chdir(self.origdir)
        if os.path.isdir(self.reduxdir):
            #- restore write access removed by any freeze
            for dirpath, dirnames, filenames in os.walk(self.reduxdir):
                os.chmod(dirpath, stat.S_IRWXU)
            rmtree(self.reduxdir)

    def _check_archive_intact(self):
        """Check that the archive target of the cumulative link is unchanged"""
        self.assertTrue(os.path.isdir(self.archivedir))
        self.assertFalse(os.path.islink(self.archivedir))
        self.assertEqual(os.listdir(self.archivedir), [os.path.basename(self.archivefile)])
        self.assertEqual(_read(self.archivefile), 'archived')

    def _check_link_moved_to_attic(self):
        """Check that the cumulative link was moved to a working attic link"""
        self.assertFalse(os.path.lexists(self.linkdir))
        self.assertTrue(os.path.islink(self.atticlink))
        self.assertFalse(os.path.isabs(os.readlink(self.atticlink)))
        self.assertEqual(os.path.realpath(self.atticlink), os.path.realpath(self.archivedir))
        self.assertEqual(_read(os.path.join(self.atticlink, os.path.basename(self.archivefile))),
                         'archived')
        self._check_archive_intact()

    def _check_moved_to_attic(self):
        """Check that the standard test night ended up in the attic"""
        self.assertFalse(os.path.lexists(self.nightdir))
        self.assertEqual(_read(os.path.join(self.atticdir, '00001234', 'a.fits')), 'new_a')
        self.assertEqual(_read(os.path.join(self.atticdir, '00001234', 'b.fits')), 'new_b')
        self.assertEqual(_read(os.path.join(self.atticdir, '00001235', 'c.fits')), 'new_c')
        link = os.path.join(self.atticdir, '00001235', 'link.fits')
        self.assertTrue(os.path.islink(link))
        self.assertEqual(os.readlink(link), '../00001234/a.fits')
        self.assertEqual(_read(link), 'new_a')

    def test_dry_run(self):
        """dry_run should not change anything"""
        remove_directory(self.nightdir, dry_run=True)
        self.assertTrue(os.path.isfile(os.path.join(self.nightdir, '00001234', 'a.fits')))
        self.assertFalse(os.path.exists(os.path.join(self.prodroot, 'attic')))

    def test_no_attic(self):
        """no_attic should delete without saving to attic"""
        remove_directory(self.nightdir, dry_run=False, no_attic=True)
        self.assertFalse(os.path.exists(self.nightdir))
        self.assertFalse(os.path.exists(os.path.join(self.prodroot, 'attic')))

    def test_missing_dir(self):
        """A non-existent directory is a no-op"""
        missing = os.path.join(self.prodroot, 'preproc', '20250102')
        remove_directory(missing, dry_run=False)
        self.assertFalse(os.path.exists(os.path.join(self.prodroot, 'attic')))

    def test_new_attic_absolute(self):
        """Move an absolute path into an attic that doesn't exist yet"""
        remove_directory(self.nightdir, dry_run=False)
        self._check_moved_to_attic()

    def test_new_attic_relative(self):
        """Move a path relative to the prod root, as purge_night does"""
        os.chdir(self.prodroot)
        remove_directory('preproc/20250101', dry_run=False)
        self._check_moved_to_attic()

    def test_merge_existing_attic(self):
        """Merge into an attic dir left over from an earlier purge"""
        #- older versions of a.fits and link.fits, plus files not in the new purge
        _write(os.path.join(self.atticdir, '00001234', 'a.fits'), 'old_a')
        _write(os.path.join(self.atticdir, '00001234', 'old.fits'), 'old')
        _write(os.path.join(self.atticdir, '00001236', 'd.fits'), 'old_d')
        _write(os.path.join(self.atticdir, '00001235', 'link.fits'), 'old_link')

        remove_directory(self.nightdir, dry_run=False)

        self._check_moved_to_attic()
        #- previously atticed files not in the new purge are preserved
        self.assertEqual(_read(os.path.join(self.atticdir, '00001234', 'old.fits')), 'old')
        self.assertEqual(_read(os.path.join(self.atticdir, '00001236', 'd.fits')), 'old_d')

    def test_merge_file_dir_conflict(self):
        """A new entry replaces an old attic entry of a different type"""
        _write(os.path.join(self.atticdir, '00001234'), 'old file where dir now is')
        _write(os.path.join(self.atticdir, '00001235', 'c.fits', 'x'), 'old dir where file now is')
        remove_directory(self.nightdir, dry_run=False)
        self._check_moved_to_attic()

    def test_merge_link_keeps_attic_dir(self):
        """A nested link doesn't replace an existing real attic directory"""
        os.symlink('00001234', os.path.join(self.nightdir, '00001236'))
        _write(os.path.join(self.atticdir, '00001236', 'd.fits'), 'old_d')
        remove_directory(self.nightdir, dry_run=False)
        self._check_moved_to_attic()
        self.assertFalse(os.path.islink(os.path.join(self.atticdir, '00001236')))
        self.assertEqual(_read(os.path.join(self.atticdir, '00001236', 'd.fits')), 'old_d')

    def test_repeated_purge(self):
        """Purging, recreating, and purging again keeps the latest version"""
        remove_directory(self.nightdir, dry_run=False)
        _write(os.path.join(self.nightdir, '00001234', 'a.fits'), 'newer_a')
        remove_directory(self.nightdir, dry_run=False)
        self.assertFalse(os.path.exists(self.nightdir))
        self.assertEqual(_read(os.path.join(self.atticdir, '00001234', 'a.fits')), 'newer_a')
        self.assertEqual(_read(os.path.join(self.atticdir, '00001234', 'b.fits')), 'new_b')

    def test_not_under_specprod(self):
        """A directory outside the prod raises instead of losing data"""
        otherdir = tempfile.mkdtemp()
        try:
            _write(os.path.join(otherdir, 'x', 'a.fits'), 'a')
            with self.assertRaises(ValueError):
                remove_directory(os.path.join(otherdir, 'x'), dry_run=False)
            self.assertEqual(_read(os.path.join(otherdir, 'x', 'a.fits')), 'a')
        finally:
            rmtree(otherdir)

    def test_move_to_attic_new(self):
        """move_to_attic creates missing parent directories"""
        dest = os.path.join(self.reduxdir, 'a', 'b', 'c')
        move_to_attic(self.nightdir, dest)
        self.assertFalse(os.path.exists(self.nightdir))
        self.assertEqual(_read(os.path.join(dest, '00001234', 'a.fits')), 'new_a')

    def test_link_new_attic(self):
        """A relative link is moved to the attic with a target that still resolves"""
        remove_directory(self.linkdir, dry_run=False)
        self._check_link_moved_to_attic()

    def test_link_new_attic_relative(self):
        """Same as above, but with a relative path and trailing slash"""
        os.chdir(self.prodroot)
        remove_directory('tiles/cumulative/1000/20250101/', dry_run=False)
        self._check_link_moved_to_attic()

    def test_link_frozen_archive(self):
        """Purging a link to a read-only archive only touches the link"""
        for dirpath in [self.archivedir, os.path.dirname(self.archivedir)]:
            os.chmod(dirpath, stat.S_IRUSR | stat.S_IXUSR)
        remove_directory(self.linkdir, dry_run=False)
        self._check_link_moved_to_attic()

    def test_link_existing_attic_dir(self):
        """An existing attic dir is kept and nothing is moved out of the link target"""
        _write(os.path.join(self.atticlink, 'old.fits'), 'old')
        remove_directory(self.linkdir, dry_run=False)
        self.assertFalse(os.path.lexists(self.linkdir))
        self.assertFalse(os.path.islink(self.atticlink))
        self.assertEqual(os.listdir(self.atticlink), ['old.fits'])
        self.assertEqual(_read(os.path.join(self.atticlink, 'old.fits')), 'old')
        self._check_archive_intact()

    def test_link_existing_attic_link(self):
        """An existing attic link is replaced"""
        os.makedirs(os.path.dirname(self.atticlink))
        os.symlink('/does/not/exist', self.atticlink)
        remove_directory(self.linkdir, dry_run=False)
        self._check_link_moved_to_attic()

    def test_link_absolute(self):
        """An absolute link stays absolute in the attic"""
        os.remove(self.linkdir)
        os.symlink(self.archivedir, self.linkdir)
        remove_directory(self.linkdir, dry_run=False)
        self.assertFalse(os.path.lexists(self.linkdir))
        self.assertEqual(os.readlink(self.atticlink), self.archivedir)
        self._check_archive_intact()

    def test_link_symlinked_parent(self):
        """A relative link reached through a symlinked parent dir still resolves in the attic"""
        #- tiles/cumulative is a link to external storage holding the archive link and archive
        externaldir = os.path.join(self.reduxdir, 'external', 'tiles')
        os.makedirs(externaldir)
        cumulativedir = os.path.join(self.prodroot, 'tiles', 'cumulative')
        os.rename(cumulativedir, os.path.join(externaldir, 'cumulative'))
        os.rename(os.path.join(self.prodroot, 'tiles', 'archive'), os.path.join(externaldir, 'archive'))
        os.symlink(os.path.join(externaldir, 'cumulative'), cumulativedir)
        realtarget = os.path.realpath(self.linkdir)
        self.assertEqual(realtarget, os.path.realpath(os.path.join(externaldir, 'archive', '1000', '20250105')))

        remove_directory(self.linkdir, dry_run=False)
        self.assertFalse(os.path.lexists(self.linkdir))
        self.assertTrue(os.path.islink(self.atticlink))
        self.assertEqual(os.path.realpath(self.atticlink), realtarget)
        self.assertEqual(_read(os.path.join(self.atticlink, os.path.basename(self.archivefile))),
                         'archived')

    def test_link_absolute_with_symlink(self):
        """An absolute target through a symlink followed by '..' is copied unchanged"""
        currentdir = os.path.join(self.prodroot, 'tiles', 'archive', 'current')
        os.makedirs(currentdir)
        aliasdir = os.path.join(self.reduxdir, 'alias')
        os.symlink(currentdir, aliasdir)
        linktarget = os.path.join(aliasdir, '..', '1000', '20250105')
        os.remove(self.linkdir)
        os.symlink(linktarget, self.linkdir)
        realtarget = os.path.realpath(self.linkdir)
        self.assertEqual(realtarget, os.path.realpath(self.archivedir))

        remove_directory(self.linkdir, dry_run=False)
        self.assertFalse(os.path.lexists(self.linkdir))
        self.assertEqual(os.readlink(self.atticlink), linktarget)
        self.assertEqual(os.path.realpath(self.atticlink), realtarget)
        self.assertEqual(_read(self.archivefile), 'archived')

    def test_link_no_attic(self):
        """no_attic removes just the link"""
        remove_directory(self.linkdir, dry_run=False, no_attic=True)
        self.assertFalse(os.path.lexists(self.linkdir))
        self.assertFalse(os.path.exists(os.path.join(self.prodroot, 'attic')))
        self._check_archive_intact()

    def test_link_dry_run(self):
        """dry_run leaves the link in place"""
        remove_directory(self.linkdir, dry_run=True)
        self.assertTrue(os.path.islink(self.linkdir))
        self.assertFalse(os.path.exists(os.path.join(self.prodroot, 'attic')))
        self._check_archive_intact()

    def test_dangling_link(self):
        """A dangling link is still removed"""
        os.remove(self.linkdir)
        os.symlink('../../archive/1000/20991231', self.linkdir)
        remove_directory(self.linkdir, dry_run=False)
        self.assertFalse(os.path.lexists(self.linkdir))
        self.assertTrue(os.path.islink(self.atticlink))
        self._check_archive_intact()

    def test_dir_replaces_attic_link(self):
        """A real dir purged where the attic has a link doesn't move files into the link target"""
        os.makedirs(os.path.dirname(self.atticdir))
        os.symlink(self.archivedir, self.atticdir)
        remove_directory(self.nightdir, dry_run=False)
        self._check_moved_to_attic()
        self.assertFalse(os.path.islink(self.atticdir))
        self._check_archive_intact()

    def test_purge_tilenight_unique_tiles(self):
        """Repeated tiles are only purged once, including all of their exposures"""
        night = 20250101
        etable = Table()
        etable['TILEID'] = [1000, 1000, 1000, 2000]
        etable['EXPID'] = [1, 2, 3, 4]
        with patch('desispec.scripts.purge_tilenight.load_table', return_value=etable), \
                patch('desispec.scripts.purge_tilenight.remove_directory') as mock_remove:
            purge_tilenight([1000, 2000, 1000, 1000], night, dry_run=True)

        dirnames = [call.args[0] for call in mock_remove.call_args_list]
        #- 3 per exposure dirs (preproc, exposures, perexp) per exposure, plus pernight per
        #- tile, plus the cumulative link for tile 1000 from setUp
        self.assertEqual(len(dirnames), 3*4 + 2 + 1)
        self.assertEqual(dirnames.count(self.linkdir), 1)
        self.assertEqual(len(dirnames), len(set(dirnames)))
        preprocdirs = [d for d in dirnames if f'/preproc/{night}/' in d]
        self.assertEqual(sorted(int(os.path.basename(d)) for d in preprocdirs), [1, 2, 3, 4])
