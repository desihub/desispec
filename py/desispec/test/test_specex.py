"""
Test desispec.scripts.specex
"""

import os
import shutil
import tempfile
import unittest

import numpy as np
from astropy.io import fits

from desispec.scripts.specex import merge_psf

#- Fiber status codes used by specex PSF files:
#-   -1 : fiber not part of this bundle (not applicable)
#-    0 : fiber successfully fit
#-   >0 : fiber fit failed with this error code
NOT_APPLICABLE = -1
FIT_OK = 0
FIT_FAILED = 2


def _write_psf(filename, status, bundle, legcoeff, xtrace, ytrace,
               fit_bundles=()):
    """
    Write a minimal specex-like per-bundle PSF fits file for testing merge_psf.

    Args:
        filename: output path
        status: per-fiber STATUS values, 1D array of length nfibers
        bundle: per-fiber BUNDLE values, 1D array of length nfibers
        legcoeff: per-fiber "trace" legendre coefficients, 2D (nfibers, ncoeff)
        xtrace, ytrace: 2D (nfibers, ncoeff) arrays
        fit_bundles: bundle ids that were actually fit in this file; merge_psf
            requires B{bundle:02d}RCHI2/NDATA/NPAR header keys for these
    """
    nfibers = len(status)
    ncoeff = legcoeff.shape[1]

    param = np.array(['STATUS', 'BUNDLE', 'LEGCOEFF'])
    coeff = np.zeros((len(param), nfibers, ncoeff))
    coeff[0, :, 0] = status
    coeff[1, :, 0] = bundle
    coeff[2] = legcoeff

    col_param = fits.Column(name='PARAM', format='15A', array=param)
    col_coeff = fits.Column(name='COEFF', format='{}D'.format(nfibers * ncoeff),
                             dim='({},{})'.format(ncoeff, nfibers), array=coeff)
    psf_hdu = fits.BinTableHDU.from_columns([col_param, col_coeff], name='PSF')
    psf_hdu.header['FIBERMIN'] = 0
    psf_hdu.header['FIBERMAX'] = nfibers - 1
    for b in fit_bundles:
        psf_hdu.header['B{:02d}RCHI2'.format(b)] = 1.0
        psf_hdu.header['B{:02d}NDATA'.format(b)] = 100
        psf_hdu.header['B{:02d}NPAR'.format(b)] = 10

    xtrace_hdu = fits.ImageHDU(xtrace, name='XTRACE')
    ytrace_hdu = fits.ImageHDU(ytrace, name='YTRACE')

    hdulist = fits.HDUList([fits.PrimaryHDU(), psf_hdu, xtrace_hdu, ytrace_hdu])
    hdulist.writeto(filename, overwrite=True)


class TestMergePSF(unittest.TestCase):

    def setUp(self):
        self.testdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.testdir, ignore_errors=True)

    def test_failed_fibers_status_written_before_continue(self):
        """
        A bundle where every fiber fails to fit has no "selected" fibers,
        so merge_psf hits its `continue` for that bundle without copying
        xtrace/ytrace/other parameters. Regardless, the STATUS for those
        failed fibers must still be recorded in the merged output instead
        of being skipped over by that `continue`.
        """
        nfibers = 4
        ncoeff = 2

        #- reference psf covering all fibers; STATUS gets reset to -1
        #- internally by merge_psf for every fiber before merging in inputs
        ref_status = np.zeros(nfibers)
        ref_bundle = np.array([0, 0, 1, 1])
        ref_legcoeff = np.zeros((nfibers, ncoeff))
        ref_xtrace = np.zeros((nfibers, ncoeff))
        ref_ytrace = np.zeros((nfibers, ncoeff))
        reffile = os.path.join(self.testdir, 'psf-ref.fits')
        _write_psf(reffile, ref_status, ref_bundle, ref_legcoeff,
                   ref_xtrace, ref_ytrace)

        #- bundle 0 (fibers 0,1): fit succeeded
        b0_status = np.array([FIT_OK, FIT_OK, NOT_APPLICABLE, NOT_APPLICABLE])
        b0_bundle = np.array([0, 0, -1, -1])
        b0_legcoeff = np.full((nfibers, ncoeff), 1.0)
        b0_xtrace = np.full((nfibers, ncoeff), 11.0)
        b0_ytrace = np.full((nfibers, ncoeff), 12.0)
        b0file = os.path.join(self.testdir, 'psf-bundle0.fits')
        _write_psf(b0file, b0_status, b0_bundle, b0_legcoeff,
                   b0_xtrace, b0_ytrace, fit_bundles=[0])

        #- bundle 1 (fibers 2,3): every fiber in the bundle failed to fit,
        #- so there are no "selected" fibers for this input file
        b1_status = np.array([NOT_APPLICABLE, NOT_APPLICABLE,
                               FIT_FAILED, FIT_FAILED])
        b1_bundle = np.array([-1, -1, 1, 1])
        b1_legcoeff = np.full((nfibers, ncoeff), 99.0)
        b1_xtrace = np.full((nfibers, ncoeff), 99.0)
        b1_ytrace = np.full((nfibers, ncoeff), 99.0)
        b1file = os.path.join(self.testdir, 'psf-bundle1.fits')
        _write_psf(b1file, b1_status, b1_bundle, b1_legcoeff,
                   b1_xtrace, b1_ytrace)

        outfile = os.path.join(self.testdir, 'psf-merged.fits')
        merge_psf(reffile, [b0file, b1file], outfile)

        with fits.open(outfile) as merged:
            data = merged['PSF'].data
            i_status = np.where(data['PARAM'] == 'STATUS')[0][0]
            merged_status = data['COEFF'][i_status][:, 0]

            #- fibers 0,1 fit successfully
            np.testing.assert_array_equal(merged_status[[0, 1]],
                                           [FIT_OK, FIT_OK])

            #- fibers 2,3 failed to fit; their failure status must still be
            #- recorded, not silently skipped by the `continue` that fires
            #- because bundle 1 has zero selected (status==0) fibers
            np.testing.assert_array_equal(merged_status[[2, 3]],
                                           [FIT_FAILED, FIT_FAILED])

            #- since bundle 1 had no selected fibers, its xtrace/ytrace and
            #- other parameters should NOT have been copied into the output
            i_leg = np.where(data['PARAM'] == 'LEGCOEFF')[0][0]
            merged_legcoeff = data['COEFF'][i_leg]
            np.testing.assert_array_equal(merged_legcoeff[[2, 3]],
                                           ref_legcoeff[[2, 3]])
            np.testing.assert_array_equal(merged['XTRACE'].data[[2, 3]],
                                           ref_xtrace[[2, 3]])
            np.testing.assert_array_equal(merged['YTRACE'].data[[2, 3]],
                                           ref_ytrace[[2, 3]])

            #- bundle 0's successfully fit fibers should be copied over
            np.testing.assert_array_equal(merged_legcoeff[[0, 1]],
                                           b0_legcoeff[[0, 1]])
            np.testing.assert_array_equal(merged['XTRACE'].data[[0, 1]],
                                           b0_xtrace[[0, 1]])
            np.testing.assert_array_equal(merged['YTRACE'].data[[0, 1]],
                                           b0_ytrace[[0, 1]])


if __name__ == '__main__':
    unittest.main()
