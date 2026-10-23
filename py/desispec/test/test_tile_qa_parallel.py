"""
Test that the parallel I/O options of the tile QA plot give the same
results as the serial code path.
"""

import os
import shutil
import tempfile
import unittest

import numpy as np
from astropy.table import Table
from astropy.io import fits

from desispec.tile_qa_plot import get_expids_efftimes, get_expid_vccdsec
from desispec.tile_qa_spectra import read_spectra_tsnr2


def _write_spectra(filename, petal, expids, nights, nfiber=20, tsnr2_key="TSNR2_LRG"):
    """Write a minimal gzipped spectra file with the HDUs read by tile QA"""
    rng = np.random.default_rng(petal)
    n = nfiber * len(expids)
    fibermap = Table()
    fibermap["TARGETID"] = np.arange(n) + 1000 * petal
    fibermap["EXPID"] = np.repeat(expids, nfiber).astype(np.int32)
    fibermap["NIGHT"] = np.repeat(nights, nfiber).astype(np.int32)
    fibermap["PETAL_LOC"] = np.full(n, petal, dtype=np.int16)
    scores = Table()
    for camera in "BRZ":
        scores[f"{tsnr2_key}_{camera}"] = rng.uniform(1, 100, n)
    hdus = fits.HDUList([
        fits.PrimaryHDU(),
        fits.table_to_hdu(fibermap),
        fits.table_to_hdu(scores),
    ])
    hdus[1].name = "FIBERMAP"
    hdus[2].name = "SCORES"
    hdus.writeto(filename, overwrite=True)


class TestTileQAParallel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.testdir = tempfile.mkdtemp()
        cls.tileid, cls.night = 4242, 20230520
        cls.expids, cls.nights = [181534, 181535], [cls.night, cls.night]

        cls.tiledir = os.path.join(cls.testdir, "tiles", "cumulative", str(cls.tileid), str(cls.night))
        os.makedirs(cls.tiledir)
        for petal in range(4):
            fn = os.path.join(cls.tiledir, f"spectra-{petal}-{cls.tileid}-thru{cls.night}.fits")
            _write_spectra(fn, petal, cls.expids, cls.nights)
            # tile QA usually sees the gzipped files
            os.system(f"gzip -f {fn}")

        # tile-qa file with just the header keywords that get_expids_efftimes needs
        cls.tileqafits = os.path.join(cls.tiledir, f"tile-qa-{cls.tileid}-thru{cls.night}.fits")
        hdr = fits.Header()
        hdr["TILEID"], hdr["LASTNITE"], hdr["GOALTYPE"] = cls.tileid, cls.night, "dark"
        hdus = fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU(Table({"X": [0]}), header=hdr, name="FIBERQA")])
        hdus.writeto(cls.tileqafits)

        # frame files with CAMERA and, for all but one, VCCDSEC in the primary header
        cls.nframes = 0
        for expid in cls.expids:
            expdir = os.path.join(cls.testdir, "exposures", str(cls.night), f"{expid:08d}")
            os.makedirs(expdir)
            for petal in range(3):
                for camera in "brz":
                    hdr = fits.Header()
                    hdr["CAMERA"] = f"{camera}{petal}"
                    if not (expid == cls.expids[0] and petal == 1 and camera == "z"):
                        hdr["VCCDSEC"] = 1000.0 + petal * 10 + ord(camera)
                    fits.PrimaryHDU(header=hdr).writeto(os.path.join(expdir, f"frame-{camera}{petal}-{expid:08d}.fits"))
                    cls.nframes += 1

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.testdir)

    def test_read_spectra_tsnr2(self):
        fn = os.path.join(self.tiledir, f"spectra-2-{self.tileid}-thru{self.night}.fits.gz")
        d = read_spectra_tsnr2(fn, "TSNR2_LRG")
        self.assertEqual(len(d), 40)
        for col in ["TARGETID", "EXPID", "NIGHT", "PETAL_LOC", "TSNR2_B", "TSNR2_R", "TSNR2_Z"]:
            self.assertIn(col, d.colnames)
        self.assertTrue(np.all(d["PETAL_LOC"] == 2))
        self.assertTrue(np.all(d["TSNR2_B"] > 0))

        # missing key -> zeros
        d = read_spectra_tsnr2(fn, "TSNR2_BGS")
        self.assertTrue(np.all(d["TSNR2_B"] == 0))

    def test_get_expids_efftimes_parallel(self):
        serial = get_expids_efftimes(self.tileqafits, self.testdir)
        self.assertEqual(len(serial), 2)
        self.assertTrue(np.all(serial["EFFTIME_SPEC"] > 0))
        for nworkers in (2, 8):
            parallel = get_expids_efftimes(self.tileqafits, self.testdir, nworkers=nworkers)
            self.assertEqual(serial.colnames, parallel.colnames)
            for col in serial.colnames:
                self.assertTrue(np.array_equal(serial[col], parallel[col]), f"{col} differs for {nworkers=}")

    def test_get_expid_vccdsec_parallel(self):
        for expid in self.expids:
            serial = get_expid_vccdsec(expid, self.night, self.testdir)
            self.assertEqual(len(serial[0]), 9)
            for nworkers in (2, 16):
                parallel = get_expid_vccdsec(expid, self.night, self.testdir, nworkers=nworkers)
                self.assertEqual(serial, parallel)

        # the frame without VCCDSEC is flagged with -99 and listed as problematic
        vccdsec, pb_campets = get_expid_vccdsec(self.expids[0], self.night, self.testdir, nworkers=4)
        self.assertEqual(vccdsec["z1"], -99.0)
        self.assertIn("z1", pb_campets)


if __name__ == '__main__':
    unittest.main()
