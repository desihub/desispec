"""
test desispec.validredshifts
"""

import os
import shutil
import tempfile
import unittest

import numpy as np
from astropy.table import Table
import fitsio

from desispec.maskbits import fibermask
from desispec.validredshifts import (get_good_fiberstatus, actually_validate,
                                     validate)

try:
    from desitarget.targetmask import desi_mask, bgs_mask, scnd_mask
    desitarget_available = True
except ImportError:
    desitarget_available = False

QN_COLS = ['C_LYA', 'C_CIV', 'C_CIII', 'C_MgII', 'C_Hbeta', 'C_Halpha']

#- Bits hard-coded in the GOOD_Z_LYA branch of actually_validate;
#- test_hardcoded_bits checks that these still agree with desitarget
ELG_BIT = 2**1
QSO_BIT = 2**2
BGS_BIT = 2**60
WISE_VAR_QSO_BIT = 2**35


def make_cat(n):
    """Build an n-row catalog with all columns needed by actually_validate.

    Defaults correspond to a well-measured, unremarkable galaxy at z=0.5:
    ZWARN=0, DELTACHI2=100, good fiberstatus, bright [OII], no QSO evidence.
    Individual tests override single entries to exercise single criteria.
    """
    cat = Table()
    cat['TARGETID'] = np.arange(n, dtype=np.int64)
    cat['CHI2'] = np.full(n, 1000.0)
    cat['Z'] = np.full(n, 0.5)
    cat['ZERR'] = np.full(n, 1.0e-4)
    cat['ZWARN'] = np.zeros(n, dtype=np.int64)
    cat['SPECTYPE'] = np.array(['GALAXY'] * n, dtype='U10')
    cat['DELTACHI2'] = np.full(n, 100.0)
    cat['COADD_FIBERSTATUS'] = np.zeros(n, dtype=np.int32)
    cat['TARGET_RA'] = np.linspace(10.0, 11.0, n)
    cat['TARGET_DEC'] = np.linspace(20.0, 21.0, n)
    cat['OBJTYPE'] = np.array(['TGT'] * n, dtype='U3')
    cat['OII_FLUX'] = np.full(n, 10.0)
    cat['OII_FLUX_IVAR'] = np.full(n, 1.0)
    cat['IS_QSO_MGII'] = np.zeros(n, dtype=bool)
    cat['Z_NEW'] = np.full(n, 2.0)
    cat['ZERR_NEW'] = np.full(n, 1.0e-3)
    cat['DELTACHI2_NEW'] = np.full(n, 200.0)
    cat['IS_QSO_QN_NEW_RR'] = np.zeros(n, dtype=bool)
    for col in QN_COLS:
        cat[col] = np.zeros(n, dtype=np.float32)
    cat['DESI_TARGET'] = np.zeros(n, dtype=np.int64)
    cat['BGS_TARGET'] = np.zeros(n, dtype=np.int64)
    cat['MWS_TARGET'] = np.zeros(n, dtype=np.int64)
    cat['SCND_TARGET'] = np.zeros(n, dtype=np.int64)
    return cat


def write_mock_files(dirname, cat, survey='main', include_emline=True,
                     include_qso=True):
    """Write mock redrock + afterburner files; return the redrock path."""
    n = len(cat)
    redrock_path = os.path.join(dirname, 'redrock-0-101151-thru20251019.fits')

    redshifts = np.empty(n, dtype=[('TARGETID', 'i8'), ('CHI2', 'f8'),
                                   ('Z', 'f8'), ('ZERR', 'f8'), ('ZWARN', 'i8'),
                                   ('SPECTYPE', 'S10'), ('DELTACHI2', 'f8')])
    for col in ('TARGETID', 'CHI2', 'Z', 'ZERR', 'ZWARN', 'DELTACHI2'):
        redshifts[col] = cat[col]
    redshifts['SPECTYPE'] = np.char.encode(np.asarray(cat['SPECTYPE'], dtype='U10'))

    fm_dtype = [('TARGETID', 'i8'), ('COADD_FIBERSTATUS', 'i4'),
                ('TARGET_RA', 'f8'), ('TARGET_DEC', 'f8'), ('OBJTYPE', 'S3')]
    if survey == 'cmx':
        fm_dtype.append(('CMX_TARGET', 'i8'))
    else:
        prefix = 'SV1_' if survey == 'sv1' else ''
        for col in ('DESI_TARGET', 'BGS_TARGET', 'MWS_TARGET', 'SCND_TARGET'):
            fm_dtype.append((prefix + col, 'i8'))

    fibermap = np.zeros(n, dtype=fm_dtype)
    for col in ('TARGETID', 'COADD_FIBERSTATUS', 'TARGET_RA', 'TARGET_DEC'):
        fibermap[col] = cat[col]
    fibermap['OBJTYPE'] = np.char.encode(np.asarray(cat['OBJTYPE'], dtype='U3'))
    if survey != 'cmx':
        prefix = 'SV1_' if survey == 'sv1' else ''
        for col in ('DESI_TARGET', 'BGS_TARGET', 'MWS_TARGET', 'SCND_TARGET'):
            fibermap[prefix + col] = cat[col]

    fitsio.write(redrock_path, redshifts, extname='REDSHIFTS', clobber=True)
    fitsio.write(redrock_path, fibermap, extname='FIBERMAP')

    if include_emline:
        emline = np.empty(n, dtype=[('TARGETID', 'i8'), ('OII_FLUX', 'f4'),
                                    ('OII_FLUX_IVAR', 'f4')])
        emline['TARGETID'] = cat['TARGETID']
        emline['OII_FLUX'] = cat['OII_FLUX']
        emline['OII_FLUX_IVAR'] = cat['OII_FLUX_IVAR']
        fitsio.write(os.path.join(dirname, 'emline-0-101151-thru20251019.fits'),
                     emline, extname='EMLINEFIT', clobber=True)

    if include_qso:
        mgii = np.empty(n, dtype=[('TARGETID', 'i8'), ('IS_QSO_MGII', 'bool')])
        mgii['TARGETID'] = cat['TARGETID']
        mgii['IS_QSO_MGII'] = cat['IS_QSO_MGII']
        fitsio.write(os.path.join(dirname, 'qso_mgii-0-101151-thru20251019.fits'),
                     mgii, extname='MGII', clobber=True)

        qn_dtype = [('TARGETID', 'i8'), ('Z_NEW', 'f8'), ('ZERR_NEW', 'f8'),
                    ('DELTACHI2_NEW', 'f8'), ('IS_QSO_QN_NEW_RR', 'bool')]
        qn_dtype += [(col, 'f4') for col in QN_COLS]
        qn = np.empty(n, dtype=qn_dtype)
        for name, _ in qn_dtype:
            qn[name] = cat[name]
        fitsio.write(os.path.join(dirname, 'qso_qn-0-101151-thru20251019.fits'),
                     qn, extname='QN_RR', clobber=True)

    return redrock_path


class TestValidRedshifts(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.testdir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        if os.path.isdir(cls.testdir):
            shutil.rmtree(cls.testdir)

    def setUp(self):
        #- each test writes its mock files into its own subdirectory
        self.dirname = tempfile.mkdtemp(dir=self.testdir)

    #-------------------------------------------------------------------
    #- get_good_fiberstatus

    def test_get_good_fiberstatus(self):
        restricted = fibermask.mask('RESTRICTED')
        variable = fibermask.mask('VARIABLE')
        broken = fibermask.mask('BROKENFIBER')

        cat = Table()
        cat['COADD_FIBERSTATUS'] = np.array(
            [0, restricted, variable, restricted | variable,
             broken, broken | restricted], dtype=np.int32)
        expected = np.array([True, True, True, True, False, False])
        self.assertTrue(np.all(get_good_fiberstatus(cat) == expected))

    @unittest.skipUnless(desitarget_available, 'desitarget not available')
    def test_hardcoded_bits(self):
        """The bits hard-coded in actually_validate must match desitarget."""
        self.assertEqual(ELG_BIT, desi_mask['ELG'])
        self.assertEqual(QSO_BIT, desi_mask['QSO'])
        self.assertEqual(BGS_BIT, desi_mask['BGS_ANY'])
        self.assertEqual(WISE_VAR_QSO_BIT, scnd_mask['WISE_VAR_QSO'])

    #-------------------------------------------------------------------
    #- individual tracer criteria

    def test_good_z_bgs(self):
        cat = make_cat(6)
        cat['ZWARN'][1] = 4          # ZWARN != 0
        cat['DELTACHI2'][2] = 40.0   # DELTACHI2 must be > 40, not >=
        cat['DELTACHI2'][3] = 41.0
        cat['Z'][4] = 0.8            # Z must be < 0.8
        cat['Z'][5] = 0.79

        res = actually_validate(cat, ignore_qso=True, ignore_lya=True)
        expected = np.array([True, False, False, True, False, True])
        self.assertTrue(np.all(res['GOOD_Z_BGS'] == expected))

    def test_good_z_lrg(self):
        cat = make_cat(6)
        cat['ZWARN'][1] = 4
        cat['DELTACHI2'][2] = 15.0   # must be > 15
        cat['DELTACHI2'][3] = 16.0
        cat['Z'][4] = 1.5            # must be < 1.5
        cat['Z'][5] = 1.49

        res = actually_validate(cat, ignore_qso=True, ignore_lya=True)
        expected = np.array([True, False, False, True, False, True])
        self.assertTrue(np.all(res['GOOD_Z_LRG'] == expected))

        #- LRG cut is looser in redshift than BGS
        self.assertTrue(res['GOOD_Z_LRG'][5])
        self.assertFalse(res['GOOD_Z_BGS'][5])

    def test_good_z_elg(self):
        #- cut is log10(OII_FLUX*sqrt(IVAR)) > 0.9 - 0.2*log10(DELTACHI2)
        #- with DELTACHI2=100 the [OII] SNR threshold is 10**0.5 = 3.162
        cat = make_cat(6)
        cat['OII_FLUX'] = [10.0, 3.0, -5.0, 100.0, 1.0, 1.0]
        cat['OII_FLUX_IVAR'] = [1.0, 1.0, 1.0, 0.0, 100.0, 1.0]
        cat['DELTACHI2'][5] = 1.0e6  # low SNR rescued by huge DELTACHI2
        cat['ZWARN'][0] = 4          # ELG cut does not use ZWARN

        res = actually_validate(cat, ignore_qso=True, ignore_lya=True)
        expected = np.array([True, False, False, False, True, True])
        self.assertTrue(np.all(res['GOOD_Z_ELG'] == expected))

    def test_ignore_emline(self):
        cat = make_cat(3)
        res = actually_validate(cat, ignore_emline=True, ignore_qso=True,
                                ignore_lya=True)
        self.assertNotIn('GOOD_Z_ELG', res.colnames)
        self.assertIn('GOOD_Z_BGS', res.colnames)

    def test_star_rejection(self):
        cat = make_cat(4)
        cat['SPECTYPE'][1] = 'STAR'
        cat['Z'][2] = 0.001          # Z must be > 0.001
        cat['SPECTYPE'][3] = 'STAR'  # star that the QSO afterburners like
        cat['Z'][3] = 0.0005
        cat['DELTACHI2'][3] = 1.0e4
        cat['IS_QSO_MGII'][3] = True

        res = actually_validate(cat, ignore_lya=True)
        for col in ['GOOD_Z_BGS', 'GOOD_Z_LRG', 'GOOD_Z_ELG']:
            self.assertTrue(np.all(res[col] == [True, False, False, False]),
                            '{} failed star rejection'.format(col))

    #-------------------------------------------------------------------
    #- QSO criteria

    def test_good_z_qso(self):
        cat = make_cat(12)
        cat['SPECTYPE'][1] = 'QSO'              # redrock QSO
        cat['IS_QSO_MGII'][2] = True            # MgII afterburner
        cat['C_CIV'][3] = 0.995                 # QuasarNet, above 0.99
        cat['C_CIV'][4] = 0.995                 # QuasarNet with new redshift
        cat['IS_QSO_QN_NEW_RR'][4] = True
        cat['Z'][4] = 1.0
        cat['Z_NEW'][4] = 2.5
        cat['ZERR_NEW'][4] = 0.02
        cat['C_CIV'][5] = 0.95                  # below the 0.99 threshold
        cat['IS_QSO_QN_NEW_RR'][5] = True
        cat['SPECTYPE'][6] = 'QSO'              # not a science target
        cat['OBJTYPE'][6] = 'SKY'
        cat['SPECTYPE'][7] = 'QSO'              # z > 5 fits are all bad
        cat['Z'][7] = 5.5
        cat['SPECTYPE'][8] = 'QSO'              # low-z, low DELTACHI2 failure
        cat['Z'][8] = 0.2
        cat['DELTACHI2'][8] = 10.0
        cat['SPECTYPE'][9] = 'QSO'              # low-z, high DELTACHI2 is ok
        cat['Z'][9] = 0.2
        cat['DELTACHI2'][9] = 1000.0
        cat['SPECTYPE'][10] = 'QSO'             # QSO cuts ignore ZWARN
        cat['ZWARN'][10] = 4
        cat['SPECTYPE'][11] = 'QSO'
        cat['COADD_FIBERSTATUS'][11] = fibermask.mask('BROKENFIBER')

        res = actually_validate(cat, ignore_lya=True)
        expected = np.array([False, True, True, True, True, False,
                             False, False, False, True, True, False])
        self.assertTrue(np.all(res['GOOD_Z_QSO'] == expected))

        #- Z_QSO/ZERR_QSO are taken from QuasarNet only for IS_QSO_QN_NEW_RR
        self.assertAlmostEqual(res['Z_QSO'][4], 2.5)
        self.assertAlmostEqual(res['ZERR_QSO'][4], 0.02)
        notnew = np.ones(len(cat), dtype=bool)
        notnew[4] = False
        self.assertTrue(np.allclose(res['Z_QSO'][notnew], cat['Z'][notnew]))
        self.assertTrue(np.allclose(res['ZERR_QSO'][notnew], cat['ZERR'][notnew]))

    def test_deltachi2_new(self):
        """DELTACHI2_NEW is used for the low-z QSO failure mode cut if Z_QSO=Z_NEW."""
        cat = make_cat(2)
        cat['SPECTYPE'] = 'QSO'
        cat['C_CIV'] = 0.999
        cat['IS_QSO_QN_NEW_RR'] = True
        cat['Z'] = 2.0
        cat['Z_NEW'] = 0.2
        #- log10(DELTACHI2_QSO) < 3 - 3.5*0.2 = 2.3 is a bad low-z QSO
        cat['DELTACHI2'] = 1.0e6         # redrock DELTACHI2 must not be used
        cat['DELTACHI2_NEW'] = [10.0, 1.0e4]

        res = actually_validate(cat, ignore_lya=True)
        self.assertTrue(np.all(res['Z_QSO'] == 0.2))
        self.assertTrue(np.all(res['GOOD_Z_QSO'] == [False, True]))

    def test_ignore_qso(self):
        cat = make_cat(3)
        res = actually_validate(cat, ignore_qso=True, ignore_lya=True)
        for col in ['GOOD_Z_QSO', 'GOOD_Z_LYA', 'Z_QSO', 'ZERR_QSO']:
            self.assertNotIn(col, res.colnames)

        #- LyA quality requires the QSO quality
        with self.assertRaises(ValueError):
            actually_validate(cat, ignore_qso=True, ignore_lya=False)

    #-------------------------------------------------------------------
    #- LyA criteria

    def test_good_z_lya(self):
        cat = make_cat(12)
        #- QSO target with a redrock QSO classification
        cat['DESI_TARGET'][0] = QSO_BIT
        cat['SPECTYPE'][0] = 'QSO'
        cat['Z'][0] = 2.3
        #- ELG target, spectype QSO, relaxed (>0.6) QuasarNet confidence,
        #- no rerun disagreement -> Z_QSO stays at the original redrock Z
        cat['DESI_TARGET'][1] = ELG_BIT
        cat['SPECTYPE'][1] = 'QSO'
        cat['C_LYA'][1] = 0.7
        cat['Z'][1] = 2.4
        #- same, but QuasarNet confidence below 0.6
        cat['DESI_TARGET'][2] = ELG_BIT
        cat['SPECTYPE'][2] = 'QSO'
        cat['C_LYA'][2] = 0.5
        cat['Z'][2] = 2.4
        #- WISE_VAR_QSO secondary passing the main QSO cuts
        cat['SCND_TARGET'][3] = WISE_VAR_QSO_BIT
        cat['SPECTYPE'][3] = 'QSO'
        cat['Z'][3] = 2.1
        #- WISE_VAR_QSO that is also a BGS target is excluded
        cat['SCND_TARGET'][4] = WISE_VAR_QSO_BIT
        cat['DESI_TARGET'][4] = BGS_BIT
        cat['SPECTYPE'][4] = 'QSO'
        cat['Z'][4] = 2.1
        #- QSO target with no QSO evidence at all
        cat['DESI_TARGET'][5] = QSO_BIT
        #- QSO target at z>5 is rejected for GOOD_Z_QSO, but allowed for
        #- GOOD_Z_LYA (z>5 Lya QSOs are real; only GOOD_Z_QSO treats z>5 as
        #- evidence of a bad fit)
        cat['DESI_TARGET'][6] = QSO_BIT
        cat['SPECTYPE'][6] = 'QSO'
        cat['Z'][6] = 6.0
        #- QSO target identified purely by the MgII afterburner
        #- (not a redrock QSO, no QuasarNet evidence)
        cat['DESI_TARGET'][7] = QSO_BIT
        cat['IS_QSO_MGII'][7] = True
        cat['Z'][7] = 2.5
        #- QSO target identified purely by QuasarNet (>0.99), not redrock or
        #- MgII, with a rerun that disagrees with redrock -> use Z_NEW
        cat['DESI_TARGET'][8] = QSO_BIT
        cat['C_CIV'][8] = 0.995
        cat['IS_QSO_QN_NEW_RR'][8] = True
        cat['Z'][8] = 1.0
        cat['Z_NEW'][8] = 2.5
        #- WISE_VAR_QSO secondary that is also an ELG target is excluded from
        #- the WISE_VAR_QSO branch (QuasarNet confidence left below the 0.6
        #- relaxed threshold, so it does not qualify via the ELG branch either)
        cat['SCND_TARGET'][9] = WISE_VAR_QSO_BIT
        cat['DESI_TARGET'][9] = ELG_BIT
        cat['SPECTYPE'][9] = 'QSO'
        cat['Z'][9] = 2.1
        #- ELG target, spectype QSO, relaxed (>0.6 but <0.99) QuasarNet
        #- confidence, WITH a rerun that disagrees with redrock -> use Z_NEW.
        #- Regression check: on a buggy version of this code, this case
        #- incorrectly kept the original (bad) redrock redshift.
        cat['DESI_TARGET'][10] = ELG_BIT
        cat['SPECTYPE'][10] = 'QSO'
        cat['C_LYA'][10] = 0.75
        cat['IS_QSO_QN_NEW_RR'][10] = True
        cat['Z'][10] = 1.0
        cat['Z_NEW'][10] = 2.5
        #- QSO target with the low-z misclassification failure mode
        #- (see DESI-doc-9981) is rejected for both GOOD_Z_QSO and GOOD_Z_LYA
        cat['DESI_TARGET'][11] = QSO_BIT
        cat['SPECTYPE'][11] = 'QSO'
        cat['Z'][11] = 0.2
        cat['DELTACHI2'][11] = 10.0

        res = actually_validate(cat)
        expected = np.array([True, True, False, True, False, False, True,
                             True, True, False, True, False])
        self.assertTrue(np.all(res['GOOD_Z_LYA'] == expected))
        self.assertFalse(res['GOOD_Z_QSO'][6])
        self.assertFalse(res['GOOD_Z_QSO'][11])

        #- ELG relaxed branch without a rerun keeps the original redrock Z
        self.assertAlmostEqual(res['Z_QSO'][1], cat['Z'][1])
        #- QN>0.99-only branch with a rerun uses Z_NEW
        self.assertAlmostEqual(res['Z_QSO'][8], cat['Z_NEW'][8])
        #- ELG relaxed branch with a rerun uses Z_NEW (the regression case)
        self.assertAlmostEqual(res['Z_QSO'][10], cat['Z_NEW'][10])

    def test_correct_target_column(self):
        """The WISE_VAR_QSO LyA branch reads SCND_TARGET, not DESI_ or BGS_TARGET."""
        cat = make_cat(3)
        cat['SPECTYPE'] = 'QSO'
        cat['Z'] = 2.1
        cat['SCND_TARGET'][0] = WISE_VAR_QSO_BIT   # genuine WISE_VAR_QSO
        cat['DESI_TARGET'][1] = WISE_VAR_QSO_BIT    # same bit, wrong column
        cat['BGS_TARGET'][2] = WISE_VAR_QSO_BIT    # same bit, wrong column

        res = actually_validate(cat)
        self.assertTrue(np.all(res['GOOD_Z_LYA'] == [True, False, False]))

    #-------------------------------------------------------------------
    #- options and output structure

    def test_fiberstatus_cut(self):
        cat = make_cat(3)
        cat['DESI_TARGET'] = QSO_BIT
        cat['SPECTYPE'] = 'QSO'
        cat['Z'] = 2.0
        cat['COADD_FIBERSTATUS'][1] = fibermask.mask('BROKENFIBER')
        cat['COADD_FIBERSTATUS'][2] = fibermask.mask('RESTRICTED')

        res = actually_validate(cat, fiberstatus_cut=True)
        for col in ['GOOD_Z_BGS', 'GOOD_Z_LRG', 'GOOD_Z_ELG', 'GOOD_Z_QSO',
                    'GOOD_Z_LYA']:
            self.assertFalse(res[col][1], '{} ignored fiberstatus'.format(col))
        self.assertTrue(res['GOOD_Z_QSO'][2])

        res = actually_validate(cat, fiberstatus_cut=False)
        self.assertTrue(res['GOOD_Z_QSO'][1])

    def test_populate_missing_columns(self):
        cat = make_cat(3)
        res = actually_validate(cat, ignore_emline=True, ignore_qso=True,
                                ignore_lya=True, populate_missing_columns=True)
        for col in ['GOOD_Z_BGS', 'GOOD_Z_LRG', 'GOOD_Z_ELG', 'GOOD_Z_QSO',
                    'GOOD_Z_LYA']:
            self.assertIn(col, res.colnames)
        for col in ['GOOD_Z_ELG', 'GOOD_Z_QSO', 'GOOD_Z_LYA']:
            self.assertFalse(np.any(res[col]))

    def test_output_columns(self):
        cat = make_cat(3)
        res = actually_validate(cat)
        self.assertEqual(res.colnames,
                         ['GOOD_Z_BGS', 'GOOD_Z_LRG', 'GOOD_Z_ELG',
                          'GOOD_Z_QSO', 'GOOD_Z_LYA', 'Z_QSO', 'ZERR_QSO'])
        self.assertEqual(len(res), len(cat))
        #- intermediate columns are not leaked into the output
        for col in ['IS_QSO_QN', 'QSO_MASKBITS', 'DELTACHI2_QSO']:
            self.assertNotIn(col, res.colnames)

    def test_input_not_modified(self):
        cat = make_cat(3)
        colnames = list(cat.colnames)
        z = cat['Z'].copy()
        actually_validate(cat)
        self.assertEqual(cat.colnames, colnames)
        self.assertTrue(np.all(cat['Z'] == z))

    #-------------------------------------------------------------------
    #- validate() with files on disk

    @unittest.skipUnless(desitarget_available, 'desitarget not available')
    def test_validate_no_afterburners(self):
        cat = make_cat(4)
        redrock_path = write_mock_files(self.dirname, cat, include_emline=False,
                                        include_qso=False)
        out = validate(redrock_path)
        self.assertEqual(out.colnames, ['TARGETID', 'Z', 'ZWARN',
                                        'COADD_FIBERSTATUS', 'GOOD_Z_BGS',
                                        'GOOD_Z_LRG'])
        self.assertTrue(np.all(out['TARGETID'] == cat['TARGETID']))

    @unittest.skipUnless(desitarget_available, 'desitarget not available')
    def test_validate_emline_only(self):
        cat = make_cat(4)
        cat['OII_FLUX'][1] = 1.0e-3
        redrock_path = write_mock_files(self.dirname, cat, include_qso=False)
        out = validate(redrock_path)
        self.assertIn('GOOD_Z_ELG', out.colnames)
        self.assertNotIn('GOOD_Z_QSO', out.colnames)
        self.assertNotIn('GOOD_Z_LYA', out.colnames)
        self.assertTrue(out['GOOD_Z_ELG'][0])
        self.assertFalse(out['GOOD_Z_ELG'][1])

    @unittest.skipUnless(desitarget_available, 'desitarget not available')
    def test_validate_main(self):
        """Full main-survey validation with all afterburners present."""
        cat = make_cat(4)
        cat['DESI_TARGET'][0] = desi_mask['LRG']
        cat['DESI_TARGET'][1] = desi_mask['ELG']
        cat['DESI_TARGET'][2] = desi_mask['QSO']
        cat['DESI_TARGET'][3] = desi_mask['BGS_ANY']
        cat['BGS_TARGET'][3] = bgs_mask['BGS_BRIGHT']
        cat['SPECTYPE'][2] = 'QSO'
        cat['Z'][2] = 2.2

        redrock_path = write_mock_files(self.dirname, cat)
        out = validate(redrock_path, return_target_columns=True)

        expected_columns = ['TARGETID', 'Z', 'ZWARN', 'COADD_FIBERSTATUS',
                            'LRG', 'ELG', 'QSO', 'LGE', 'ELG_LOP', 'ELG_HIP',
                            'ELG_VLO', 'BGS_ANY', 'BGS_FAINT', 'BGS_BRIGHT',
                            'WISE_VAR_QSO', 'GOOD_Z_BGS', 'GOOD_Z_LRG',
                            'GOOD_Z_ELG', 'GOOD_Z_QSO', 'GOOD_Z_LYA']
        self.assertEqual(out.colnames, expected_columns)

        self.assertTrue(np.all(out['LRG'] == [True, False, False, False]))
        self.assertTrue(np.all(out['ELG'] == [False, True, False, False]))
        self.assertTrue(np.all(out['QSO'] == [False, False, True, False]))
        self.assertTrue(np.all(out['BGS_ANY'] == [False, False, False, True]))
        self.assertTrue(np.all(out['BGS_BRIGHT'] == [False, False, False, True]))
        self.assertFalse(np.any(out['WISE_VAR_QSO']))
        self.assertTrue(out['GOOD_Z_QSO'][2])
        self.assertTrue(out['GOOD_Z_LYA'][2])
        self.assertFalse(out['GOOD_Z_QSO'][0])

    @unittest.skipUnless(desitarget_available, 'desitarget not available')
    def test_validate_wise_var_qso_column(self):
        """WISE_VAR_QSO membership must come from the SCND_TARGET column."""
        cat = make_cat(4)
        wise_bit = scnd_mask['WISE_VAR_QSO']
        #- row 0: WISE_VAR_QSO secondary, no primary targeting bits at all
        cat['SCND_TARGET'][0] = wise_bit
        #- row 1: WISE_VAR_QSO secondary that is also a BGS_BRIGHT primary
        cat['SCND_TARGET'][1] = wise_bit
        cat['DESI_TARGET'][1] = desi_mask['BGS_ANY']
        cat['BGS_TARGET'][1] = bgs_mask['BGS_BRIGHT']
        #- row 2: a different secondary target, not WISE_VAR_QSO
        cat['SCND_TARGET'][2] = wise_bit << 1
        #- row 3: no targeting bits

        redrock_path = write_mock_files(self.dirname, cat)
        out = validate(redrock_path, return_target_columns=True)

        self.assertIn('WISE_VAR_QSO', out.colnames)
        self.assertEqual(out['WISE_VAR_QSO'].dtype, bool)
        expected = np.array([True, True, False, False])
        self.assertTrue(np.all(out['WISE_VAR_QSO'] == expected),
                        'WISE_VAR_QSO={} != expected {}'.format(
                            list(out['WISE_VAR_QSO']), list(expected)))
        #- and it must not be a copy of any BGS column
        self.assertFalse(np.all(out['WISE_VAR_QSO'] == out['BGS_BRIGHT']))

    @unittest.skipUnless(desitarget_available, 'desitarget not available')
    def test_validate_wise_var_qso_not_from_bgs_target(self):
        """BGS_TARGET bits must not leak into the WISE_VAR_QSO column.

        The WISE_VAR_QSO secondary bit and the BGS_TARGET bits are different
        bits of different columns; a target with that bit set in BGS_TARGET
        but not in SCND_TARGET is not a WISE_VAR_QSO target.
        """
        cat = make_cat(3)
        wise_bit = scnd_mask['WISE_VAR_QSO']
        cat['SCND_TARGET'][0] = wise_bit    # true WISE_VAR_QSO
        cat['BGS_TARGET'][1] = wise_bit     # same bit in the wrong column
        cat['DESI_TARGET'][1] = desi_mask['BGS_ANY']
        cat['BGS_TARGET'][2] = bgs_mask['BGS_FAINT']
        cat['DESI_TARGET'][2] = desi_mask['BGS_ANY']

        redrock_path = write_mock_files(self.dirname, cat)
        out = validate(redrock_path, return_target_columns=True)

        self.assertTrue(out['WISE_VAR_QSO'][0],
                        'SCND_TARGET WISE_VAR_QSO bit was not picked up')
        self.assertFalse(out['WISE_VAR_QSO'][1],
                         'WISE_VAR_QSO was read from BGS_TARGET')
        self.assertFalse(out['WISE_VAR_QSO'][2])

    @unittest.skipUnless(desitarget_available, 'desitarget not available')
    def test_validate_lya_wise_consistency(self):
        """A LyA QSO from the WISE_VAR_QSO branch must be a WISE_VAR_QSO."""
        cat = make_cat(2)
        cat['SCND_TARGET'][0] = scnd_mask['WISE_VAR_QSO']
        cat['SPECTYPE'][0] = 'QSO'
        cat['Z'][0] = 2.1
        cat['DESI_TARGET'][1] = desi_mask['QSO']
        cat['SPECTYPE'][1] = 'QSO'
        cat['Z'][1] = 2.1

        redrock_path = write_mock_files(self.dirname, cat)
        out = validate(redrock_path, return_target_columns=True)

        #- row 0 can only be a GOOD_Z_LYA through the WISE_VAR_QSO branch,
        #- so the membership column must agree with the quality flag
        self.assertTrue(out['GOOD_Z_LYA'][0])
        self.assertTrue(out['WISE_VAR_QSO'][0])
        lya_from_wise = out['GOOD_Z_LYA'] & ~out['QSO'] & ~out['ELG']
        self.assertTrue(np.all(out['WISE_VAR_QSO'][lya_from_wise]),
                        'GOOD_Z_LYA set for a non-QSO, non-ELG target that is '
                        'not flagged as WISE_VAR_QSO')

    @unittest.skipUnless(desitarget_available, 'desitarget not available')
    def test_validate_extra_columns(self):
        cat = make_cat(4)
        redrock_path = write_mock_files(self.dirname, cat, include_qso=False)
        out = validate(redrock_path, extra_columns=['TARGETID', 'TARGET_RA',
                                                    'TARGET_DEC'])
        self.assertEqual(out.colnames, ['TARGETID', 'TARGET_RA', 'TARGET_DEC',
                                        'GOOD_Z_BGS', 'GOOD_Z_LRG',
                                        'GOOD_Z_ELG'])
        #- columns requested twice are not duplicated
        out = validate(redrock_path, extra_columns=['TARGETID', 'GOOD_Z_BGS'])
        self.assertEqual(out.colnames, ['TARGETID', 'GOOD_Z_BGS',
                                        'GOOD_Z_LRG', 'GOOD_Z_ELG'])

    @unittest.skipUnless(desitarget_available, 'desitarget not available')
    def test_validate_matches_actually_validate(self):
        cat = make_cat(5)
        cat['OII_FLUX'] = [10.0, 2.0, 10.0, -1.0, 10.0]
        cat['ZWARN'][2] = 4
        cat['Z'][3] = 1.2
        redrock_path = write_mock_files(self.dirname, cat, include_qso=False)
        out = validate(redrock_path)
        res = actually_validate(cat, ignore_qso=True, ignore_lya=True)
        for col in ['GOOD_Z_BGS', 'GOOD_Z_LRG', 'GOOD_Z_ELG']:
            self.assertTrue(np.all(out[col] == res[col]), col)

    @unittest.skipUnless(desitarget_available, 'desitarget not available')
    def test_validate_sv(self):
        """SV data are supported, but the LyA cuts are skipped."""
        cat = make_cat(4)
        redrock_path = write_mock_files(self.dirname, cat, survey='sv1')
        out = validate(redrock_path)
        self.assertNotIn('GOOD_Z_LYA', out.colnames)
        self.assertIn('GOOD_Z_QSO', out.colnames)

    @unittest.skipUnless(desitarget_available, 'desitarget not available')
    def test_validate_sv_target_columns(self):
        from desitarget.sv1.sv1_targetmask import scnd_mask as sv1_scnd_mask
        if 'WISE_VAR_QSO' not in sv1_scnd_mask.names():
            self.skipTest('WISE_VAR_QSO is not an SV1 secondary bit')
        cat = make_cat(4)
        cat['SCND_TARGET'][0] = sv1_scnd_mask['WISE_VAR_QSO']
        redrock_path = write_mock_files(self.dirname, cat, survey='sv1')
        out = validate(redrock_path, return_target_columns=True)
        for name in ['LRG', 'ELG', 'QSO', 'LGE', 'ELG_LOP', 'ELG_HIP',
                     'ELG_VLO', 'BGS_ANY', 'BGS_FAINT', 'BGS_BRIGHT',
                     'WISE_VAR_QSO']:
            self.assertIn(name, out.colnames)
            self.assertEqual(out[name].dtype, bool)
        self.assertTrue(np.all(out['WISE_VAR_QSO'] == [True, False, False, False]))

    @unittest.skipUnless(desitarget_available, 'desitarget not available')
    def test_validate_cmx(self):
        cat = make_cat(4)
        redrock_path = write_mock_files(self.dirname, cat, survey='cmx')
        #- NotImplementedError from validate, or ValueError from
        #- main_cmx_or_sv(..., scnd=True), which has no cmx secondaries
        with self.assertRaises((NotImplementedError, ValueError)):
            validate(redrock_path)

    @unittest.skipUnless(desitarget_available, 'desitarget not available')
    def test_validate_targetid_mismatch(self):
        """Afterburner rows must be aligned with the redrock rows."""
        cat = make_cat(4)
        redrock_path = write_mock_files(self.dirname, cat, include_qso=False)
        emline_path = os.path.join(self.dirname,
                                   'emline-0-101151-thru20251019.fits')
        emline = Table(fitsio.read(emline_path))
        emline['TARGETID'] = emline['TARGETID'][::-1]
        emline.write(emline_path, overwrite=True)
        with self.assertRaises(AssertionError):
            validate(redrock_path)


if __name__ == '__main__':
    unittest.main()
