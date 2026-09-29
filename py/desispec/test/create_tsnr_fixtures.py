#!/usr/bin/env python
"""Create synthetic on-disk inputs for afterburner integration tests.

Run ``python py/desispec/test/create_tsnr_fixtures.py OUTPUT_DIRECTORY`` to
write a small production with exposure tables, QA, and cframe SCORES. These
are test inputs, not numerical reference outputs from the old afterburner.
Calculation tests mock calibration readers and calc_tsnr2; the synthetic
cframes are not intended for scientific recomputation.
"""

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table

from desispec.io.meta import findfile
from desispec.workflow.exptable import get_exposure_table_column_defaults, instantiate_exposure_table
from desispec.workflow.tableio import load_table, write_table

_TRACERS = ('ELG', 'QSO', 'LRG', 'LYA', 'BGS', 'GPBDARK', 'GPBBRIGHT', 'GPBBACKUP')


def exposure_header(night, expid, tileid=1234):
    """Return a complete synthetic science-exposure FITS header."""
    return fits.Header({
        'NIGHT': night, 'EXPID': expid, 'TILEID': tileid,
        'TILERA': 180., 'TILEDEC': 30., 'MJD-OBS': 59488., 'EXPTIME': 900.,
        'AIRMASS': 1.1, 'ACQFWHM': 1.2, 'ETCTEFF': 800., 'FLAVOR': 'science',
        'SURVEY': 'main', 'FAPRGRM': 'dark', 'FAFLAVOR': 'maindark',
        'GOALTYPE': 'dark', 'GOALTIME': 1000., 'MINTFRAC': .9,
        'HIERARCH SKY_MAG_G_SPEC': 22., 'HIERARCH SKY_MAG_R_SPEC': 21., 'HIERARCH SKY_MAG_Z_SPEC': 20.,
    })


def write_exptable_fixture(prod, night, expid, tileid=1234, laststep='all', camword='a0'):
    """Upsert one synthetic workflow exposure-table row and return its filename."""
    filename = findfile('exposure_table', night=night, specprod_dir=str(prod))
    row = get_exposure_table_column_defaults()
    row.update(NIGHT=night, EXPID=expid, TILEID=tileid, LASTSTEP=laststep,
               CAMWORD=camword, BADCAMWORD='', OBSTYPE='science', EXPTIME=900.,
               SURVEY='main', FAPRGRM='dark', GOALTYPE='dark', GOALTIME=1000.)
    table = instantiate_exposure_table()
    if Path(filename).exists():
        table = load_table(filename, tabletype='exptable')
        table = table[table['EXPID'] != expid]
    table.add_row({col: row[col] for col in table.colnames})
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    write_table(table, filename, tabletype='exptable')
    return filename


def write_qa_fixture(prod, night, expid, tileid=1234, value=10., missing=(), petal_hdu=True):
    """Write QA with one petal; missing names remove selected TSNR2 columns."""
    filename = findfile('exposureqa', night=night, expid=expid, specprod_dir=str(prod))
    header = exposure_header(night, expid, tileid)
    fiber = fits.BinTableHDU(Table({'EBV': [.05, .05]}), header=header, name='FIBERQA')
    petal = Table({'PETAL_LOC': [0]})
    for tracer in _TRACERS:
        for band in ('B', 'R', 'Z'):
            col = 'TSNR2_{}_{}'.format(tracer, band)
            if col not in missing:
                petal[col] = np.array([value], dtype=np.float32)
    hdus = [fits.PrimaryHDU(), fiber]
    if petal_hdu:
        hdus.append(fits.BinTableHDU(petal, name='PETALQA'))
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList(hdus).writeto(filename, overwrite=True)
    return filename


def write_cframe_fixture(prod, night, expid, camera, value=20., missing=(), scores=True):
    """Write a cframe with metadata, fibermap, and optional per-fiber SCORES."""
    filename = findfile('cframe', night=night, expid=expid, camera=camera, specprod_dir=str(prod))
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    flat = Path(filename).parent / 'test-flat.fits'
    flat.touch()
    header = exposure_header(night, expid)
    header['FIBERFLT'] = str(flat)
    fibermap = fits.BinTableHDU(Table({'EBV': [.05, .05]}), header=header, name='FIBERMAP')
    hdus = [fits.PrimaryHDU(header=header), fits.ImageHDU(), fibermap]
    if scores:
        table = Table()
        for tracer in _TRACERS:
            col = 'TSNR2_{}_{}'.format(tracer, camera[0].upper())
            if col not in missing:
                table[col] = np.array([value, value], dtype=np.float32)
        table['TSNR2_ALPHA_' + camera[0].upper()] = np.array([1., 1.], dtype=np.float32)
        hdus.append(fits.BinTableHDU(table, name='SCORES'))
    fits.HDUList(hdus).writeto(filename, overwrite=True)
    return filename


def main(options=None):
    """Write a small synthetic production to the requested directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    args = parser.parse_args(options)
    prod = Path(args.directory).resolve()
    write_exptable_fixture(prod, 20211001, 100)
    write_qa_fixture(prod, 20211001, 100)
    for camera in ('b0', 'r0', 'z0'):
        write_cframe_fixture(prod, 20211001, 100, camera)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
