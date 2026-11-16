"""
desispec.tile_qa_spectra
========================

Lightweight reader of the per-fiber quantities that the tile QA plot needs
from ``spectra-*.fits[.gz]`` files.

This is kept in its own module with few imports so that worker processes
started by :func:`desispec.tile_qa_plot.get_expids_efftimes` do not have to
import all of :mod:`desispec.tile_qa_plot` (several seconds) just to unpickle
the function they run.
"""

import fitsio
from astropy.table import Table
from desiutil.log import get_logger


def read_spectra_tsnr2(spectra_fn, tsnr2_key):
    """
    Read the per-fiber columns needed for EFFTIME_SPEC from one spectra file.

    Args:
        spectra_fn: path to a ``spectra-*.fits[.gz]`` file
        tsnr2_key: TSNR2 key without camera, e.g. "TSNR2_LRG"

    Returns:
        astropy Table with TARGETID, EXPID, NIGHT, PETAL_LOC, TSNR2_{B,R,Z}

    Notes:
        The spectra files are typically gzipped and SCORES is the last HDU,
        so the whole file must be decompressed (CPU bound, ~3 s per petal).
    """
    log = get_logger()
    f = fitsio.FITS(spectra_fn)
    myd = Table()
    myd["TARGETID"] = f["FIBERMAP"].read_column("TARGETID")
    myd["EXPID"] = f["FIBERMAP"].read_column("EXPID")
    myd["NIGHT"] = f["FIBERMAP"].read_column("NIGHT")
    myd["PETAL_LOC"] = f["FIBERMAP"].read_column("PETAL_LOC")
    scores_keys = f["SCORES"].get_colnames()
    for camera in ["B", "R", "Z"]:
        key = "{}_{}".format(tsnr2_key, camera)
        if key in scores_keys:
            myd["TSNR2_{}".format(camera)] = f["SCORES"].read_column(key)
        else:
            myd["TSNR2_{}".format(camera)] = 0
            log.warning(f"no {key} column in {spectra_fn} -> set TSNR2_{camera}=0")
    f.close()
    return myd
