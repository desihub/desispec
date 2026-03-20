import os
import tempfile
import numpy as np
import fitsio
import unittest
from importlib import resources
from astropy.table import Table

from desispec.scripts import inspect_dark

class TestInspectDark(unittest.TestCase):
    def setUp(self):
        # Create temporary directory for test files using TemporaryDirectory
        self.temp_dir = tempfile.TemporaryDirectory()
        # Create fake dark image FITS with a bright column
        ny, nx = 40, 50
        rnd = np.random.RandomState(0)
        image = rnd.normal(scale=1.0, size=(ny, nx))
        self.badcol = 10
        image[:, self.badcol] += 20.0  # bright column
        ivar = np.ones_like(image)
        mask = np.zeros_like(image, dtype=int)
        fibermap = {"FIBER": np.arange(1, 5, dtype=int)}
        dark_path = os.path.join(self.temp_dir.name, "fake_dark.fits")
        with fitsio.FITS(dark_path, "rw") as f:
            f.write(image, extname="IMAGE")
            f.write(ivar, extname="IVAR")
            f.write(mask, extname="MASK")
            f.write_table(fibermap, extname="FIBERMAP")
            f["IMAGE"].write_key("EXPTIME", 300.0)
            f["IMAGE"].write_key("camera", "B0")
        self.fake_dark_file = dark_path
        # Locate PSF file using importlib.resources
        self.psf_path = str(resources.files('desispec.test.data.ql').joinpath('psf-r0.fits'))

    def tearDown(self):
        # Cleanup the temporary directory; files within it will be removed automatically
        self.temp_dir.cleanup()

    def test_inspect_dark_generates_bad_fiber_table(self):
        """Running inspect_dark should complete without error and generate a bad fiber table."""
        badfiber_file = os.path.join(self.temp_dir.name, "bad_fibers.csv")
        badcol_file = os.path.join(self.temp_dir.name, "bad_columns.csv")
        args = [
            "--infile", self.fake_dark_file,
            "--psf", self.psf_path,
            "--badfiber-table", badfiber_file,
            "--badcol-table", badcol_file,
        ]
        # Run inspect_dark; should not raise
        inspect_dark.main(args)

        # Verify that the output files were created
        self.assertTrue(os.path.exists(badfiber_file), "Bad fiber table was not created")
        self.assertTrue(os.path.exists(badcol_file), "Bad column table was not created")

        t = Table.read(badcol_file)
        self.assertIn(self.badcol, t['COLUMN'], f"Expected bad column {self.badcol} not found in output table {t}")

if __name__ == '__main__':
    unittest.main()
