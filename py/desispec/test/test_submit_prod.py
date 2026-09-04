"""
Test reference-night discovery in desispec.scripts.submit_prod.

An override file normally links calibrations from an *earlier* night, which the
chronological submission order in submit_production() handles for free.
Occasionally an override points forward in time, e.g. when a night's flats are
bad but the following night's are good. Those reference nights have to be
calibrated ahead of the normal order, which is what these functions identify.
"""

import os
import shutil
import tempfile
import unittest

from desispec.io.meta import findfile
from desispec.scripts.submit_prod import (
    get_linkcal_refnight,
    get_refnights_needing_early_calibration,
)


class TestEarlyRefnightDiscovery(unittest.TestCase):
    """Tests for get_linkcal_refnight and get_refnights_needing_early_calibration."""

    @classmethod
    def setUpClass(cls):
        cls.reduxdir = tempfile.mkdtemp()
        cls.specprod = 'test'
        cls.proddir = os.path.join(cls.reduxdir, cls.specprod)
        os.makedirs(cls.proddir)

        cls.origenv = os.environ.copy()
        os.environ['DESI_SPECTRO_REDUX'] = cls.reduxdir
        os.environ['SPECPROD'] = cls.specprod

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.reduxdir)
        for key in ('DESI_SPECTRO_REDUX', 'SPECPROD'):
            if key in cls.origenv:
                os.environ[key] = cls.origenv[key]
            elif key in os.environ:
                del os.environ[key]

    def tearDown(self):
        exptabdir = os.path.join(self.proddir, 'exposure_tables')
        if os.path.exists(exptabdir):
            shutil.rmtree(exptabdir)

    def _write_override(self, night, refnight, include=None, exclude=None):
        """Write an override file linking calibrations from refnight"""
        linkcal = {'refnight': refnight}
        if include is not None:
            linkcal['include'] = include
        if exclude is not None:
            linkcal['exclude'] = exclude
        pathname = findfile('override', night=night)
        os.makedirs(os.path.dirname(pathname), exist_ok=True)
        with open(pathname, 'w') as fil:
            fil.write('calibration:\n    linkcal:\n')
            for key, val in linkcal.items():
                fil.write(f'        {key}: {val}\n')

    def _write_exptable(self, night):
        """Touch an exposure table so the night looks processable"""
        pathname = findfile('exposure_table', night=night)
        os.makedirs(os.path.dirname(pathname), exist_ok=True)
        open(pathname, 'w').close()

    # ==================================================================
    # get_linkcal_refnight
    # ==================================================================

    def test_no_override_file(self):
        """A night without an override file links nothing"""
        self.assertEqual(get_linkcal_refnight(20230914), (None, set()))

    def test_override_without_linkcal(self):
        """An override file that doesn't link calibrations returns None"""
        pathname = findfile('override', night=20230914)
        os.makedirs(os.path.dirname(pathname), exist_ok=True)
        with open(pathname, 'w') as fil:
            fil.write('calibration:\n  nightlyflat:\n'
                      + '    extra_cmd_args: [--autocal-ff-solve-grad]\n')
        self.assertEqual(get_linkcal_refnight(20230914), (None, set()))

    def test_malformed_include_raises(self):
        """A yaml list include is not supported and must not pass silently

        derive_include_exclude() expects a comma separated string, so a yaml
        list would fail later at submission time. Since every night's override
        file is inspected up front, it needs to fail here rather than be
        quietly treated as linking nothing.
        """
        pathname = findfile('override', night=20230914)
        os.makedirs(os.path.dirname(pathname), exist_ok=True)
        with open(pathname, 'w') as fil:
            fil.write('calibration:\n    linkcal:\n        refnight: 20230915\n'
                      + '        include: [biasnight, badcolumns]\n')
        with self.assertRaises(AttributeError):
            get_linkcal_refnight(20230914)

    def test_include_and_exclude_together_raises(self):
        """include and exclude are mutually exclusive"""
        self._write_override(20230914, 20230915, include='fiberflatnight',
                             exclude='biasnight')
        with self.assertRaises(ValueError):
            get_linkcal_refnight(20230914)

    def test_include_is_resolved(self):
        """An explicit include list is returned as the set of linked files"""
        self._write_override(20230914, 20230913, include='biasnight')
        refnight, files_to_link = get_linkcal_refnight(20230914)
        self.assertEqual(refnight, 20230913)
        self.assertEqual(files_to_link, {'biasnight'})

    def test_exclude_is_resolved(self):
        """An exclude list resolves to everything else, including biasnight"""
        self._write_override(20230914, 20230913, exclude='fiberflatnight')
        refnight, files_to_link = get_linkcal_refnight(20230914)
        self.assertEqual(refnight, 20230913)
        self.assertIn('biasnight', files_to_link)
        self.assertNotIn('fiberflatnight', files_to_link)

    # ==================================================================
    # get_refnights_needing_early_calibration
    # ==================================================================

    def test_earlier_refnight_needs_nothing(self):
        """Linking from an earlier night is handled by chronological order"""
        self._write_exptable(20230913)
        self._write_exptable(20230914)
        self._write_override(20230914, 20230913, include='biasnight')
        self.assertEqual(
            get_refnights_needing_early_calibration([20230913, 20230914]), [])

    def test_later_refnight_with_biasnight(self):
        """Linking biasnight forward requires the bias be submitted first"""
        self._write_exptable(20230914)
        self._write_exptable(20230915)
        self._write_override(20230914, 20230915, include='biasnight')
        self.assertEqual(
            get_refnights_needing_early_calibration([20230914, 20230915]),
            [(20230915, True)])

    def test_later_refnight_without_biasnight(self):
        """Linking something other than biasnight doesn't need a bias-only pass"""
        self._write_exptable(20230914)
        self._write_exptable(20230915)
        self._write_override(20230914, 20230915, include='fiberflatnight')
        self.assertEqual(
            get_refnights_needing_early_calibration([20230914, 20230915]),
            [(20230915, False)])

    def test_refnight_without_exposure_table_is_skipped(self):
        """A refnight with no exposure table can't be processed, so skip it"""
        self._write_exptable(20230914)
        self._write_override(20230914, 20230915, include='biasnight')
        self.assertEqual(
            get_refnights_needing_early_calibration([20230914]), [])

    def test_chain_puts_dependencies_first(self):
        """A refnight that itself links from another night comes after it"""
        for night in (20230910, 20230914, 20230915):
            self._write_exptable(night)
        ## 20230914 links forward to 20230915, which itself links back to 20230910
        self._write_override(20230914, 20230915, include='biasnight')
        self._write_override(20230915, 20230910, include='fiberflatnight')
        self.assertEqual(
            get_refnights_needing_early_calibration([20230914, 20230915]),
            [(20230910, False), (20230915, True)])

    def test_circular_reference_terminates(self):
        """A circular set of overrides is reported rather than looping forever"""
        self._write_exptable(20230914)
        self._write_exptable(20230915)
        self._write_override(20230914, 20230915, include='fiberflatnight')
        self._write_override(20230915, 20230914, include='fiberflatnight')
        self.assertEqual(
            get_refnights_needing_early_calibration([20230914, 20230915]),
            [(20230914, False), (20230915, False)])

    def test_multiple_nights_share_one_refnight(self):
        """Two nights linking from the same refnight yield a single entry"""
        for night in (20230913, 20230914, 20230915):
            self._write_exptable(night)
        self._write_override(20230913, 20230915, include='fiberflatnight')
        self._write_override(20230914, 20230915, include='biasnight')
        self.assertEqual(
            get_refnights_needing_early_calibration([20230913, 20230914]),
            [(20230915, True)])


if __name__ == '__main__':
    unittest.main()
