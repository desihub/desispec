"""
desispec.scripts.tsnr_afterburner
====================================

Compute TSNR-based effective exposure times for all science exposures in a
production. Reads TSNR2 from exposureqa PETALQA HDUs, falling back to cframe
SCORES and then calc_tsnr2() for missing values. Use --recompute to force
calculation, or --alpha-only to recalculate alpha while retaining stored TSNR2.

Standard invocation::

    desi_tsnr_afterburner -o ${DESI_SPECTRO_REDUX}/${SPECPROD}/exposures-${SPECPROD}.fits \
                           --prod ${SPECPROD} \
                           --tile-completeness ${DESI_SPECTRO_REDUX}/${SPECPROD}/tiles-${SPECPROD}.fits \
                           --aux /global/cfs/cdirs/desi/survey/observations/SV1/sv1-tiles.fits \
                           --gfa-proc-dir /global/cfs/cdirs/desi/survey/GFA/ \
                           --add-badexp --nights $NIGHT
"""

import os
import glob
import json
import argparse
import multiprocessing
from pathlib import Path

import fitsio
import numpy as np
import astropy.io.fits as fits
from astropy.table import Table, vstack

from desiutil.log import get_logger

from desispec.calibfinder import CalibFinder
from desispec.efftime import compute_efftime
from desispec.io import read_frame, read_fiberflat, read_sky
from desispec.io.fluxcalibration import read_flux_calibration
from desispec.io import read_table
from desispec.io.meta import findfile, specprod_root, faflavor2program
from desispec.io.util import get_tempfilename, decode_camword, difference_camwords, parse_cameras
from desispec.skymag import compute_skymag
from desispec.tilecompleteness import (read_gfa_data, compute_tile_completeness_table,
                                       merge_tile_completeness_table)
from desispec.tsnr import calc_tsnr2, tsnr2_to_efftime
from desispec.util import parse_int_args
from desispec.workflow.tableio import load_table
from desiutil.depend import getdep

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Column ordering for the EXPOSURES output table.
_EXP_SUMMARY_COLUMN_ORDER = [
    'NIGHT', 'EXPID', 'TILEID', 'TILERA', 'TILEDEC', 'MJD',
    'SURVEY', 'PROGRAM', 'FAPRGRM', 'FAFLAVOR', 'EXPTIME', 'EFFTIME_SPEC', 'GOALTIME', 'GOALTYPE',
    'MINTFRAC', 'AIRMASS', 'EBV', 'SEEING_ETC', 'EFFTIME_ETC', 'TSNR2_ELG', 'TSNR2_QSO', 'TSNR2_LRG',
    'TSNR2_LYA', 'TSNR2_BGS', 'TSNR2_GPBDARK', 'TSNR2_GPBBRIGHT', 'TSNR2_GPBBACKUP', 'LRG_EFFTIME_DARK',
    'ELG_EFFTIME_DARK', 'BGS_EFFTIME_BRIGHT', 'LYA_EFFTIME_DARK', 'GPB_EFFTIME_DARK', 'GPB_EFFTIME_BRIGHT',
    'GPB_EFFTIME_BACKUP', 'TRANSPARENCY_GFA', 'SEEING_GFA', 'FIBER_FRACFLUX_GFA', 'FIBER_FRACFLUX_ELG_GFA',
    'FIBER_FRACFLUX_BGS_GFA', 'FIBERFAC_GFA', 'FIBERFAC_ELG_GFA', 'FIBERFAC_BGS_GFA', 'AIRMASS_GFA',
    'SKY_MAG_AB_GFA', 'SKY_MAG_G_SPEC', 'SKY_MAG_R_SPEC', 'SKY_MAG_Z_SPEC', 'EFFTIME_GFA',
    'EFFTIME_DARK_GFA', 'EFFTIME_BRIGHT_GFA', 'EFFTIME_BACKUP_GFA',
]

#: Night cutoff dividing ELG-based from LRG-based EFFTIME_SPEC for dark program.
_SEPT_2021_CUTOFF = 20210901

_CAMERA_BANDS = ('b', 'r', 'z')
_PETALS = list(range(10))
_TSNR2_TRACERS = ('ELG', 'QSO', 'LRG', 'LYA', 'BGS', 'GPBDARK', 'GPBBRIGHT', 'GPBBACKUP')

#: Default values for targeting metadata fields.
_TARG_DEFAULTS = {
    'SURVEY': 'unknown',
    'GOALTYPE': 'unknown',
    'FAPRGRM': 'unknown',
    'FAFLAVOR': 'unknown',
    'MINTFRAC': 0.9,
    'GOALTIME': 0.0,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse(options=None):
    """Parse command-line arguments for desi_tsnr_afterburner.

    Args:
        options: list of str, optional. If None, reads sys.argv.

    Returns:
        argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description='Compute TSNR-based effective exposure times for all science exposures in a production.')
    parser.add_argument('-o', '--outfile', type=str, default=None, required=False,
                        help='Output summary FITS file.')
    parser.add_argument('--prod', type=str, default=None, required=False,
                        help='Production name or full path. Defaults to $DESI_SPECTRO_REDUX/$SPECPROD.')
    parser.add_argument('-n', '--nights', type=str, default=None, required=False,
                        help='Comma- or colon-separated list of nights to process, e.g. 20210501 or 20210501:20210531.')
    parser.add_argument('-e', '--expids', type=str, default=None, required=False,
                        help='Comma-separated list of EXPIDs to process.')
    parser.add_argument('-c', '--cameras', type=str, default=None, required=False,
                        help='Comma-separated list of cameras to include, e.g. b0,r0,z0.')
    parser.add_argument('-t', '--tile-completeness', type=str, default=None, required=False,
                        help='Output tile completeness file (FITS/CSV base path).')
    parser.add_argument('--aux', type=str, default=None, required=False, nargs='*',
                        help='Auxiliary tile table files (e.g. SV1 tiles).')
    parser.add_argument('--gfa-proc-dir', type=str, default=None, required=False,
                        help='Directory containing GFA offline processing files.')
    parser.add_argument('--recompute-skymags', action='store_true',
                        help='Recompute sky magnitudes from sky files even when the exposure QA already '
                             'stores them. Missing sky magnitudes are always computed.')
    parser.add_argument('--skymags', type=str, default=None,
                        help='Table of NIGHT, EXPID, SKY_MAG_G/R/Z values to use for sky magnitudes.')
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument('--overwrite', action='store_true',
                        help='Overwrite any existing output file (default is to merge/upsert).')
    output_group.add_argument('--update', action=argparse.BooleanOptionalAction, default=True,
                              help='Use --no-update to add new rows to existing output, but exit with an '
                                   'error if any requested exposures (or nights, if no --expids) are '
                                   'already present. --update is the default (replace or add rows) and '
                                   'is retained only for compatibility.')
    parser.add_argument('--add-badexp', action='store_true',
                        help='Add zero-filled rows for known bad/unprocessed exposures. Without it, '
                             'rows for such exposures are removed from existing output.')
    parser.add_argument('--details-dir', type=str, default=None, required=False,
                        help='Directory for per-camera TSNR2 detail files (--recompute path only).')
    parser.add_argument('--recompute', action='store_true',
                        help='Recompute TSNR2 values by calling calc_tsnr2() on cframes.')
    parser.add_argument('--alpha-only', '--alpha_only', action='store_true',
                        help='Recompute alpha, preserving stored TSNR2 and filling missing TSNR2 if needed.')

    parser.add_argument('--nproc', type=int, default=1,
                        help='Number of parallel worker processes (per MPI rank with --mpi).')
    parser.add_argument('--mpi', action='store_true',
                        help='Use MPI to distribute nights across nodes.')

    args = parser.parse_args(options)
    return args


# ---------------------------------------------------------------------------
# Targeting metadata normalization
# ---------------------------------------------------------------------------

def derive_targ_info(entry):
    """Normalize survey/targeting metadata for legacy observations.

    Fills SURVEY and FAPRGRM from FAFLAVOR when those fields are 'unknown' or
    absent.  When FAFLAVOR is known, GOALTYPE is always set from
    faflavor2program(FAFLAVOR), matching the original afterburner; otherwise
    GOALTYPE is kept, or guessed from FAPRGRM if unknown.  Handles
    SV1/SV2/CMX/main survey conventions.

    Args:
        entry: dict with at least FAFLAVOR, SURVEY, GOALTYPE, FAPRGRM keys.

    Returns:
        entry: dict with normalized values.
    """
    for key, default in _TARG_DEFAULTS.items():
        if key not in entry:
            entry[key] = default

    faflavor = str(entry.get('FAFLAVOR', 'unknown')).strip().lower()
    entry['FAFLAVOR'] = faflavor

    if entry['FAPRGRM'] == 'unknown':
        entry['FAPRGRM'] = faflavor.replace('sv1', '').replace('sv2', '').replace('cmx', '')

    if entry['SURVEY'] == 'unknown':
        if faflavor.find('sv1') >= 0 or faflavor in ('cmxlrgqso', 'cmxelg'):
            entry['SURVEY'] = 'sv1'
        elif faflavor.find('sv2') >= 0:
            entry['SURVEY'] = 'sv2'
        elif faflavor.find('cmx') >= 0:
            entry['SURVEY'] = 'cmx'

    if faflavor != 'unknown':
        entry['GOALTYPE'] = faflavor2program(faflavor)

    if entry['GOALTYPE'] == 'unknown':
        faprgrm = entry['FAPRGRM']
        if any(x in faprgrm for x in ('qso', 'lrg', 'elg', 'dark')):
            entry['GOALTYPE'] = 'dark'
        elif any(x in faprgrm for x in ('mws', 'bgs', 'bright')):
            entry['GOALTYPE'] = 'bright'

    if entry['FAPRGRM'].startswith('dith') and entry['SURVEY'] == 'unknown':
        entry['SURVEY'] = 'cmx'

    if entry['GOALTYPE'] in ('dark1b', 'bright1b'):
        entry['GOALTYPE'] = entry['GOALTYPE'].replace('1b', '')

    return entry


def _fill_targ_from_header(entry, hdr):
    """Fill targeting metadata that is absent or still at its default value.

    Values already set to something other than the _TARG_DEFAULTS default are
    kept, so earlier sources take precedence.  GOALTIME <= 0 (including the
    exposure-table sentinel -99) is treated as missing.  FA_SURV is used for SURVEY when
    SURVEY is absent from the header (fiberassign files).

    Args:
        entry: dict of exposure metadata.
        hdr: FITS header (fitsio or astropy) or dict, e.g. a cframe FIBERMAP
            header or a fiberassign primary header.

    Returns:
        entry: the same dict with missing targeting values filled in.
    """
    for key, default in _TARG_DEFAULTS.items():
        hdrkey = key
        if key == 'SURVEY' and 'SURVEY' not in hdr:
            hdrkey = 'FA_SURV'
        if hdrkey not in hdr:
            continue
        # exposure tables use GOALTIME=-99 for unknown, so treat <= 0 as missing
        missing = entry.get(key, default) <= 0 if key == 'GOALTIME' else entry.get(key, default) == default
        if not missing:
            continue
        if isinstance(default, str):
            entry[key] = str(hdr[hdrkey]).strip().lower()
        else:
            entry[key] = hdr[hdrkey]
    return entry


# ---------------------------------------------------------------------------
# Sky magnitude computation
# ---------------------------------------------------------------------------

def get_skymag_values(night, expid):
    """Compute per-exposure sky magnitudes from sky model files.

    Called from read_one_exposure() when any SKY_MAG_G/R/Z_SPEC keywords are
    absent from the FIBERQA header, or when recompute_skymags=True is passed
    to read_one_exposure().  Wraps compute_skymag() from desispec.skymag.

    Args:
        night: int, YYYYMMDD.
        expid: int, exposure ID.

    Returns:
        dict with keys SKY_MAG_G_SPEC, SKY_MAG_R_SPEC, SKY_MAG_Z_SPEC
        (float, AB mag/arcsec2).  Returns 99.0 for all three if
        compute_skymag() finds no valid petals.
    """
    log = get_logger()
    try:
        gmag, rmag, zmag = compute_skymag(night, expid)
    except Exception as e:
        log.warning('compute_skymag failed for night={} expid={}: {}'.format(night, expid, e))
        gmag, rmag, zmag = 99.0, 99.0, 99.0
    return {'SKY_MAG_G_SPEC': gmag, 'SKY_MAG_R_SPEC': rmag, 'SKY_MAG_Z_SPEC': zmag}


def _read_etc_values(night, expid, hdr):
    """Return (seeing_etc, efftime_etc) from header keywords or ETC JSON file.

    Reads the ETC JSON at most once even when both header keywords are absent.

    Args:
        night: int, YYYYMMDD.
        expid: int, exposure ID.
        hdr: fitsio header object to check for ACQFWHM and ETCTEFF.

    Returns:
        (seeing_etc, efftime_etc): tuple of np.float32.
    """
    seeing = np.float32(0.0)
    efftime = np.float32(0.0)

    need_json = ('ACQFWHM' not in hdr) or ('ETCTEFF' not in hdr)
    etcdata = None
    if need_json:
        etc_filename = findfile('etc', night=night, expid=expid, readonly=True)
        if os.path.exists(etc_filename):
            with open(etc_filename) as f:
                etcdata = json.load(f)

    if 'ACQFWHM' in hdr:
        seeing = np.float32(hdr['ACQFWHM'])
    elif etcdata is not None:
        try:
            seeing = np.float32(etcdata['expinfo']['acq_fwhm'])
        except (KeyError, TypeError):
            pass

    if 'ETCTEFF' in hdr:
        efftime = np.float32(hdr['ETCTEFF'])
    elif etcdata is not None:
        try:
            efftime = np.float32(etcdata['expinfo']['efftime'])
        except (KeyError, TypeError):
            pass

    if np.isnan(efftime):
        efftime = np.float32(0.0)

    return seeing, efftime


# ---------------------------------------------------------------------------
# Recompute path: per-camera TSNR2 via calc_tsnr2()
# ---------------------------------------------------------------------------

def _median_tsnr_values(table, band):
    """Return finite per-camera TSNR2 medians from a per-fiber table.

    Args:
        table: astropy Table containing TSNR2 columns.
        band: Camera band, in either case.

    Returns:
        dict: Available tracer values; absent or entirely invalid columns
        are omitted so the caller can compute them. All-zero columns are valid.
    """
    values = {}
    for tracer in _TSNR2_TRACERS:
        col = 'TSNR2_{}_{}'.format(tracer, band.upper())
        if col in table.colnames:
            vals = np.asarray(table[col])
            finite = np.isfinite(vals)
            positive = finite & (vals > 0)
            if positive.any():
                values['TSNR2_' + tracer] = np.float32(np.median(vals[positive]))
            elif finite.any():
                values['TSNR2_' + tracer] = np.float32(0.0)
    return values


def read_one_camera(night, expid, camera, alpha_only=False, details_dir=None,
                    recompute=True, tsnr_values=None, read_skymags=True):
    """Read or calculate TSNR2 for one camera.

    Args:
        night: Integer observing night.
        expid: Exposure ID.
        camera: Camera name, e.g. 'b5'.
        alpha_only: Recalculate alpha while retaining available TSNR2 values.
        details_dir: Optional directory for cached per-fiber calculation results.
        recompute: Force TSNR2 calculation instead of using SCORES.
        tsnr_values: Optional TSNR2 values from QA, preferred over SCORES when
            filling missing values or calculating alpha only.
        read_skymags: Include sky magnitudes for standalone camera reads. The
            exposure loader disables this to compute them at most once.

    Returns:
        dict: Camera metadata and TSNR2 values, including TSNR2_ALPHA when
        available. None if the cframe is missing or is not a science exposure.
        Raises if missing TSNR2 cannot be calculated from the available files.
    """
    log = get_logger()
    cframe_filename = findfile('cframe', night=night, expid=expid, camera=camera, readonly=True)
    if not os.path.isfile(cframe_filename):
        log.warning('Missing cframe: {}'.format(cframe_filename))
        return None

    cframe_fits = fitsio.FITS(cframe_filename)
    try:
        hdr0 = cframe_fits[0].read_header()
        if hdr0.get('FLAVOR', '') != 'science':
            return None
        fibermap = cframe_fits['FIBERMAP'].read()
        fibermap_hdr = cframe_fits['FIBERMAP'].read_header()
        values = {}
        alpha = None
        if not recompute or alpha_only:
            if 'SCORES' in cframe_fits:
                scores = Table(cframe_fits['SCORES'].read())
                values.update(_median_tsnr_values(scores, camera[0]))
                alpha_col = 'TSNR2_ALPHA_' + camera[0].upper()
                if alpha_col in scores.colnames:
                    finite = np.isfinite(scores[alpha_col])
                    if finite.any():
                        alpha = float(np.median(scores[alpha_col][finite]))
            if tsnr_values is not None:
                values.update(tsnr_values)

        missing = any('TSNR2_' + tracer not in values for tracer in _TSNR2_TRACERS)
        if recompute or alpha_only or missing:
            log.info('Calculating TSNR2/alpha for {}'.format(cframe_filename))
            tsnr_table = None
            table_output_filename = None
            if details_dir is not None and not alpha_only:
                table_output_filename = '{}/{}/{:08d}/tsnr-{}-{:08d}.fits'.format(
                    details_dir, night, expid, camera, expid)
                # --recompute still writes the cache but never reads it
                if not recompute and os.path.isfile(table_output_filename):
                    tsnr_table = Table.read(table_output_filename)
                    cached = _median_tsnr_values(tsnr_table, camera[0])
                    if len(cached) != len(_TSNR2_TRACERS):
                        tsnr_table = None

            if tsnr_table is None:
                hdr1 = cframe_fits[1].read_header()
                flat = hdr0.get('FIBERFLT')
                if flat is None:
                    flat = CalibFinder([hdr0, hdr1]).findfile('FIBERFLAT')
                if 'SPECPROD' in flat:
                    flat = flat.replace('SPECPROD', specprod_root())
                if 'SPCALIB' in flat:
                    flat = flat.replace('SPCALIB', getdep(hdr0, 'DESI_SPECTRO_CALIB'))
                if not os.path.exists(flat):
                    raise FileNotFoundError('Flat not found for {} on {}: {}'.format(camera, night, flat))
                frame_filename = findfile('frame', night=night, expid=expid, camera=camera, readonly=True)
                sky_filename = findfile('sky', night=night, expid=expid, camera=camera, readonly=True)
                calib_filename = findfile('fluxcalib', night=night, expid=expid, camera=camera, readonly=True)
                cframe_obj = read_frame(cframe_filename, skip_resolution=True)
                frame_obj = read_frame(frame_filename, skip_resolution=True)
                results, alpha = calc_tsnr2(
                    cframe_obj, frame_obj, fiberflat=read_fiberflat(flat),
                    skymodel=read_sky(sky_filename), fluxcalib=read_flux_calibration(calib_filename),
                    alpha_only=alpha_only and not missing)
                tsnr_table = Table({k: np.asarray(v, dtype=np.float32) for k, v in results.items()})
                # alpha-only returns no tracer arrays; use the fibermap length.
                tsnr_table['TSNR2_ALPHA_' + camera[0].upper()] = np.full(len(fibermap), alpha, dtype=np.float32)
                if table_output_filename is not None:
                    Path(os.path.dirname(table_output_filename)).mkdir(parents=True, exist_ok=True)
                    tmpfile = get_tempfilename(table_output_filename)
                    tsnr_table.write(tmpfile, format='fits', overwrite=True)
                    os.rename(tmpfile, table_output_filename)

            computed = _median_tsnr_values(tsnr_table, camera[0])
            if recompute and not alpha_only:
                values = computed
            else:
                for key, value in computed.items():
                    values.setdefault(key, value)
            alpha_col = 'TSNR2_ALPHA_' + camera[0].upper()
            if alpha_col in tsnr_table.colnames and len(tsnr_table):
                alpha = float(np.median(tsnr_table[alpha_col]))
            missing_cols = ['TSNR2_' + t for t in _TSNR2_TRACERS if 'TSNR2_' + t not in values]
            if missing_cols:
                raise ValueError('Cannot obtain {} for {}'.format(missing_cols, cframe_filename))
    finally:
        cframe_fits.close()

    entry = {}
    entry['NIGHT'] = np.int32(night)
    entry['EXPID'] = np.int32(expid)
    entry['TILEID'] = np.int32(hdr0.get('TILEID', 0))
    entry['TILERA'] = np.float32(hdr0.get('TILERA', 0.0))
    entry['TILEDEC'] = np.float32(hdr0.get('TILEDEC', 0.0))
    entry['MJD'] = np.float64(hdr0.get('MJD-OBS', 0.0))
    entry['EXPTIME'] = np.float32(hdr0.get('EXPTIME', 0.0))
    entry['AIRMASS'] = np.float32(hdr0.get('AIRMASS', 0.0))
    entry['CAMERA'] = camera

    # -- EBV from fibermap --------------------------------------------------
    entry['EBV'] = 0.0
    if 'EBV' in fibermap.dtype.names:
        sel = fibermap['EBV'] > 0
        if sel.sum() > 0:
            entry['EBV'] = np.float32(np.median(fibermap['EBV'][sel]))

    # -- SEEING_ETC, EFFTIME_ETC from header with ETC JSON fallback ---------
    entry['SEEING_ETC'], entry['EFFTIME_ETC'] = _read_etc_values(night, expid, hdr0)

    entry.update(values)
    if alpha is not None:
        entry['TSNR2_ALPHA'] = np.float32(alpha)

    # -- targeting metadata from fibermap header ----------------------------
    targ_dict = {k: fibermap_hdr[k] for k in fibermap_hdr.keys()}
    for key, default in _TARG_DEFAULTS.items():
        if key in targ_dict:
            val = targ_dict[key]
            if isinstance(default, str):
                entry[key] = str(val).strip().lower()
            else:
                entry[key] = val
        else:
            entry.setdefault(key, default)

    entry = derive_targ_info(entry)

    # -- sky magnitudes (per-exposure; same value for all cameras) ----------
    if read_skymags:
        entry.update(get_skymag_values(night, expid))

    return entry


# ---------------------------------------------------------------------------
# Exposure collection
# ---------------------------------------------------------------------------

def find_processed_nights():
    """Return the nights that have an exposures directory in the production.

    Exposure tables can list nights that were never processed (e.g. partial
    productions), so the processed nights are taken from
    $DESI_SPECTRO_REDUX/$SPECPROD/exposures/{NIGHT} instead.

    Returns:
        list of int: sorted YYYYMMDD nights.
    """
    log = get_logger()
    exposures_dir = os.path.join(specprod_root(), 'exposures')
    nights = []
    for dirname in sorted(glob.glob(os.path.join(exposures_dir, '*'))):
        basename = os.path.basename(dirname)
        if basename.isdigit() and len(basename) == 8:
            nights.append(int(basename))
    if not nights:
        log.warning('No night directories found in {}'.format(exposures_dir))
    return nights


def collect_science_expids(nights=None, expids=None):
    """Return processed science exposures from the exposure tables.

    Reads the workflow exposure_table for each night and returns all science
    exposures that have LASTSTEP='all'.  Also returns a separate list of
    bad-exposure entries (LASTSTEP != 'all', TILEID > 0) for zero-filling.

    Args:
        nights: list of int, optional. If None, use find_processed_nights().
        expids: list of int, optional. Filter to these EXPIDs if given.

    Returns:
        good_expids: list of dict, each with keys
            NIGHT, EXPID, TILEID, CAMWORD, BADCAMWORD, EXPTIME, MJD-OBS,
            EFFTIME_ETC, AIRMASS, SURVEY, FAPRGRM, GOALTYPE, GOALTIME, EBVFAC,
            LASTSTEP.
        bad_expids: list of dict, same keys, for exposures not fully processed.
    """
    log = get_logger()

    if nights is None:
        nights = find_processed_nights()

    good_expids = []
    bad_expids = []
    expids_set = set(expids) if expids is not None else None

    for night in nights:
        exptab_filename = findfile('exposure_table', night=int(night), readonly=True)
        if not os.path.isfile(exptab_filename):
            log.warning('Exposure table not found: {}'.format(exptab_filename))
            continue

        try:
            exptab = load_table(tablename=exptab_filename, tabletype='exptable', suppress_logging=True)
        except Exception as e:
            log.error('Failed to read {}: {}'.format(exptab_filename, e))
            continue

        for i in range(len(exptab)):
            row = exptab[i]
            expid = int(row['EXPID'])

            if expids_set is not None and expid not in expids_set:
                continue

            tileid = int(row['TILEID']) if 'TILEID' in exptab.colnames else 0
            if tileid <= 0:
                continue

            entry = {
                'NIGHT': int(night),
                'EXPID': expid,
                'TILEID': tileid,
                'CAMWORD': str(row['CAMWORD']) if 'CAMWORD' in exptab.colnames else 'a0123456789',
                'BADCAMWORD': str(row['BADCAMWORD']) if 'BADCAMWORD' in exptab.colnames else '',
                'EXPTIME': float(row['EXPTIME']) if 'EXPTIME' in exptab.colnames else 0.0,
                'MJD-OBS': float(row['MJD-OBS']) if 'MJD-OBS' in exptab.colnames else 0.0,
                'EFFTIME_ETC': float(row['EFFTIME_ETC']) if 'EFFTIME_ETC' in exptab.colnames else 0.0,
                'AIRMASS': float(row['AIRMASS']) if 'AIRMASS' in exptab.colnames else 0.0,
                'SURVEY': str(row['SURVEY']).strip().lower() if 'SURVEY' in exptab.colnames else 'unknown',
                'FAPRGRM': str(row['FAPRGRM']).strip().lower() if 'FAPRGRM' in exptab.colnames else 'unknown',
                'GOALTYPE': str(row['GOALTYPE']).strip().lower() if 'GOALTYPE' in exptab.colnames else 'unknown',
                'GOALTIME': float(row['GOALTIME']) if 'GOALTIME' in exptab.colnames else 0.0,
                'EBVFAC': float(row['EBVFAC']) if 'EBVFAC' in exptab.colnames else 1.0,
                'LASTSTEP': str(row['LASTSTEP']).strip() if 'LASTSTEP' in exptab.colnames else 'all',
                'FAFLAVOR': str(row['FAFLAVOR']).strip().lower() if 'FAFLAVOR' in exptab.colnames else 'unknown',
                'MINTFRAC': float(row['MINTFRAC']) if 'MINTFRAC' in exptab.colnames else 0.9,
            }

            # An exposure with every camera marked bad (CAMWORD == BADCAMWORD)
            # has no usable data, so treat it like an unprocessed exposure.
            if entry['LASTSTEP'] == 'all' and entry['CAMWORD'] != entry['BADCAMWORD']:
                good_expids.append(entry)
            else:
                bad_expids.append(entry)

    log.info('Found {} good exposures, {} bad exposures over {} nights'.format(
        len(good_expids), len(bad_expids), len(nights)))
    return good_expids, bad_expids


# ---------------------------------------------------------------------------
# Default path: read per-exposure data from exposureqa
# ---------------------------------------------------------------------------

def _read_cframe_fibermap_header(night, expid, cameras=None):
    """Return the FIBERMAP header of the first existing cframe of an exposure.

    Args:
        night: int, YYYYMMDD.
        expid: int, exposure ID.
        cameras: Optional list of cameras to try, in order; defaults to all.

    Returns:
        fitsio header, or None if no cframe is found.
    """
    if cameras is None:
        cameras = [band + str(petal) for petal in _PETALS for band in _CAMERA_BANDS]
    for camera in cameras:
        filename = findfile('cframe', night=night, expid=expid, camera=camera, readonly=True)
        if os.path.isfile(filename):
            return fitsio.read_header(filename, 'FIBERMAP')
    return None


def _read_exposureqa(night, expid, recompute_skymags=False, exptab_entry=None, cameras=None):
    """Read all per-exposure data from the exposureqa file.

    Opens a single file (exposure-qa-{expid:08d}.fits) and extracts all
    information needed to build the FRAMES and EXPOSURES output rows.  This is
    the default (non-recompute) I/O path.

    Sky magnitudes are read from FIBERQA.meta header keywords
    (SKY_MAG_G_SPEC, SKY_MAG_R_SPEC, SKY_MAG_Z_SPEC) when present.  If any
    are absent, get_skymag_values() is called to compute them from the sky
    model files.  If recompute_skymags=True, get_skymag_values() is called
    even when all three keywords are present.

    Older exposureqa files lack some header keywords.  MJD-OBS, EXPTIME and
    AIRMASS then come from the exposure table entry, and missing targeting
    keywords (e.g. FAFLAVOR, absent before 20260601) from the FIBERMAP header
    of a cframe.

    Args:
        night: int, YYYYMMDD.
        expid: int, exposure ID.
        recompute_skymags: bool, if True always call get_skymag_values() even
            when the FIBERQA.meta keywords are present (equivalent to the
            --recompute-skymags CLI flag).
        exptab_entry: Optional dict for this exposure from
            collect_science_expids(), used for missing MJD-OBS/EXPTIME/AIRMASS.
        cameras: Optional cameras whose cframes may supply missing targeting
            keywords.

    Returns:
        dict with keys:
            NIGHT, EXPID, TILEID, TILERA, TILEDEC, MJD, EXPTIME, AIRMASS,
            SEEING_ETC, EFFTIME_ETC, EBV,
            SKY_MAG_G_SPEC, SKY_MAG_R_SPEC, SKY_MAG_Z_SPEC,
            SURVEY, FAPRGRM, FAFLAVOR, GOALTYPE, GOALTIME, MINTFRAC,
            PETALQA: numpy structured array (as returned by fitsio.read),
                the raw PETALQA HDU data containing PETAL_LOC and
                TSNR2_{tracer}_{band} columns for all tracers and bands.
        Returns None if the exposureqa file does not exist.
    """
    log = get_logger()

    filename = findfile('exposureqa', night=night, expid=expid, readonly=True)
    if not os.path.isfile(filename):
        log.warning('exposureqa not found: {}'.format(filename))
        return None

    log.debug('Reading {}'.format(filename))

    # -- metadata from FIBERQA header ---------------------------------------
    try:
        fiberqa_hdr = fitsio.read_header(filename, 'FIBERQA')
    except (OSError, KeyError) as error:
        log.warning('Cannot read FIBERQA in {}: {}; using cframes'.format(filename, error))
        return None

    entry = {}
    entry['NIGHT'] = np.int32(night)
    entry['EXPID'] = np.int32(expid)
    entry['TILEID'] = np.int32(fiberqa_hdr.get('TILEID', 0))
    entry['TILERA'] = np.float32(fiberqa_hdr.get('TILERA', 0.0))
    entry['TILEDEC'] = np.float32(fiberqa_hdr.get('TILEDEC', 0.0))
    if exptab_entry is None:
        exptab_entry = {}
    for key, hdrkey, dtype in (('MJD', 'MJD-OBS', np.float64), ('EXPTIME', 'EXPTIME', np.float32),
                               ('AIRMASS', 'AIRMASS', np.float32)):
        if hdrkey in fiberqa_hdr:
            entry[key] = dtype(fiberqa_hdr[hdrkey])
        else:
            log.debug('{} not in {}; using exposure table'.format(hdrkey, filename))
            entry[key] = dtype(exptab_entry.get(hdrkey, 0.0))

    entry['SEEING_ETC'], entry['EFFTIME_ETC'] = _read_etc_values(night, expid, fiberqa_hdr)

    # -- targeting metadata from header -------------------------------------
    for key in ('SURVEY', 'FAPRGRM', 'FAFLAVOR', 'GOALTYPE', 'GOALTIME', 'MINTFRAC'):
        default = _TARG_DEFAULTS.get(key, 'unknown')
        if key in fiberqa_hdr:
            val = fiberqa_hdr[key]
            if isinstance(default, str):
                entry[key] = str(val).strip().lower()
            else:
                entry[key] = val
        else:
            entry[key] = default

    if any(key not in fiberqa_hdr for key in _TARG_DEFAULTS):
        fibermap_hdr = _read_cframe_fibermap_header(night, expid, cameras)
        if fibermap_hdr is not None:
            entry = _fill_targ_from_header(entry, fibermap_hdr)
        else:
            log.warning('No cframe FIBERMAP header for missing targeting keywords of expid={}'.format(expid))

    entry = derive_targ_info(entry)

    # -- EBV from FIBERQA data column ---------------------------------------
    entry['EBV'] = np.float32(0.0)
    try:
        fiberqa_data = fitsio.read(filename, 'FIBERQA', columns=['EBV'])
        ebv_vals = fiberqa_data['EBV']
        sel = ebv_vals > 0
        if sel.sum() > 0:
            entry['EBV'] = np.float32(np.median(ebv_vals[sel]))
        else:
            log.warning('No positive EBV values in FIBERQA for expid={}, using 0.0'.format(expid))
    except Exception as e:
        log.warning('Could not read EBV from FIBERQA for expid={}: {}. Using 0.0'.format(expid, e))

    # -- sky magnitudes -----------------------------------------------------
    sky_keys = ('SKY_MAG_G_SPEC', 'SKY_MAG_R_SPEC', 'SKY_MAG_Z_SPEC')
    have_all = all(k in fiberqa_hdr for k in sky_keys)
    if not recompute_skymags and have_all:
        for k in sky_keys:
            entry[k] = np.float32(fiberqa_hdr[k])
    else:
        skymags = get_skymag_values(night, expid)
        entry.update(skymags)

    # -- PETALQA data -------------------------------------------------------
    try:
        petalqa = fitsio.read(filename, 'PETALQA')
        entry['PETALQA'] = petalqa
    except Exception as e:
        log.warning('Could not read PETALQA for expid={}: {}'.format(expid, e))
        entry['PETALQA'] = None

    return entry


def _petalqa_camera_values(petalqa, camera):
    """Return the TSNR2 values that PETALQA recorded for one camera.

    exposure_qa initializes every TSNR2_{tracer}_{band} to 0 and only fills a
    petal/band when it could read that camera (and the petal's r-band cframe).
    Some tracer/band pairs are legitimately 0 (e.g. TSNR2_LYA_Z), so a band is
    treated as not recorded only when all of its tracers are 0.

    Args:
        petalqa: numpy structured array from the PETALQA HDU, or None.
        camera: Camera name, e.g. 'b5'.

    Returns:
        dict: TSNR2_{tracer} -> np.float32 for finite values; empty if the
        camera was not recorded (the caller then falls back to the cframe).
    """
    values = {}
    if petalqa is None:
        return values
    matches = petalqa['PETAL_LOC'] == int(camera[1])
    if not matches.any():
        return values
    for tracer in _TSNR2_TRACERS:
        col = 'TSNR2_{}_{}'.format(tracer, camera[0].upper())
        if col in petalqa.dtype.names:
            value = petalqa[col][matches][0]
            if np.isfinite(value):
                values['TSNR2_' + tracer] = np.float32(value)
    if values and all(value == 0 for value in values.values()):
        return {}
    return values


def read_one_exposure(night, expid, recompute_skymags=False, cameras=None,
                      recompute=False, alpha_only=False, details_dir=None, exptab_entry=None):
    """Load an exposure, using QA, SCORES, then calculation for missing TSNR2.

    Args:
        night: Integer observing night.
        expid: Exposure ID.
        recompute_skymags: Ignore stored sky magnitudes when True.
        cameras: Selected cameras; defaults to QA petals or all cameras if QA
            is unavailable.
        recompute: Force calculation of TSNR2 for selected cameras.
        alpha_only: Recalculate alpha while retaining stored TSNR2 values.
        details_dir: Optional per-camera calculation cache directory.
        exptab_entry: Optional exposure table entry from
            collect_science_expids(), used for metadata missing from the QA.

    Returns:
        dict: Exposure metadata and CAMERA_ROWS, keyed by camera. Cameras with
        neither QA values nor a cframe are skipped with a warning, as in the
        original afterburner. None if no camera could be read; main() then
        writes a zeroed row for the exposure. Calculation failures when the
        inputs exist still propagate.
    """
    log = get_logger()
    entry = _read_exposureqa(night, expid, recompute_skymags, exptab_entry=exptab_entry, cameras=cameras)
    petalqa = entry.get('PETALQA') if entry is not None else None
    if cameras is None:
        petals = petalqa['PETAL_LOC'] if petalqa is not None else _PETALS
        cameras = [band + str(petal) for petal in petals for band in _CAMERA_BANDS]
    camera_rows = {}
    for camera in cameras:
        values = _petalqa_camera_values(petalqa, camera)
        if recompute or alpha_only or len(values) != len(_TSNR2_TRACERS):
            camera_row = read_one_camera(
                night, expid, camera, alpha_only=alpha_only, details_dir=details_dir,
                recompute=recompute, tsnr_values=values, read_skymags=False)
            if camera_row is None:
                log.warning('Skipping camera {} of night={} expid={}: no QA TSNR2 and no science '
                            'cframe'.format(camera, night, expid))
                continue
            if entry is None:
                entry = dict(camera_row)
                entry['PETALQA'] = None
                entry.update(get_skymag_values(night, expid))
            camera_rows[camera] = camera_row
        else:
            camera_rows[camera] = values
    if not camera_rows:
        log.error('No cameras could be read for night={} expid={}; writing a zeroed row'.format(
            night, expid))
        return None
    entry['CAMERA_ROWS'] = camera_rows
    return entry


def _read_one_exposure_wrapper(args_tuple):
    """Wrapper for multiprocessing.Pool.map."""
    return read_one_exposure(*args_tuple)


# ---------------------------------------------------------------------------
# Table construction (single-pass)
# ---------------------------------------------------------------------------

def build_tables(exposure_rows, camword_map):
    """Build the FRAMES and EXPOSURES tables in a single pass over exposure_rows.

    exposure_rows is a list of dicts from read_one_exposure(), each with
    exposure metadata and CAMERA_ROWS, a dict mapping camera -> dict of
    TSNR2_{tracer} values (and TSNR2_ALPHA when available).  For each
    exposure, one FRAMES row is emitted per camera that is both active in
    camword_map[expid] and present in CAMERA_ROWS, while per-petal TSNR2 sums
    are accumulated so the EXPOSURES row can be appended immediately.
    Per-exposure TSNR2 is the sum over bands per petal, mean over petals.

    In addition:
    - tsnr2_to_efftime() is called once per EXPID for the EXPOSURES row.
    - faflavor2program() is called once per EXPID for the PROGRAM column.
    - _EXP_SUMMARY_COLUMN_ORDER is enforced on the EXPOSURES table.

    Args:
        exposure_rows: list of dict from read_one_exposure(); None entries
            are skipped.
        camword_map: dict mapping EXPID (int) -> list of active camera strings.

    Returns:
        (frames_table, exposures_table): tuple of astropy.table.Table,
            with EXTNAME='FRAMES' and EXTNAME='EXPOSURES' respectively.
    """
    log = get_logger()

    frames_rows = []
    exposures_rows = []

    for row in exposure_rows:
        if row is None:
            continue
        expid = int(row['EXPID'])
        camera_rows = row.get('CAMERA_ROWS')
        if not camera_rows:
            log.warning('No CAMERA_ROWS for expid={}, skipping'.format(expid))
            continue

        active_cameras = camword_map.get(expid, [])
        if not active_cameras:
            log.warning('No active cameras for expid={}'.format(expid))
            continue

        # accumulate per-petal TSNR2 sums for EXPOSURES aggregation
        petal_tsnr2 = {}  # petal -> {tracer -> sum_over_active_bands}

        for camera in active_cameras:
            if camera not in camera_rows:
                continue
            petal = int(camera[1])

            frame_row = _make_metadata_row(row)
            frame_row['CAMERA'] = camera

            if petal not in petal_tsnr2:
                petal_tsnr2[petal] = {t: 0.0 for t in _TSNR2_TRACERS}
            for tracer in _TSNR2_TRACERS:
                val = float(camera_rows[camera]['TSNR2_' + tracer])
                frame_row['TSNR2_{}'.format(tracer)] = np.float32(val)
                petal_tsnr2[petal][tracer] += val

            # QA summaries do not store alpha; keep the column with NaN.
            frame_row['TSNR2_ALPHA'] = np.float32(camera_rows[camera].get('TSNR2_ALPHA', np.nan))
            frames_rows.append(frame_row)

        if not petal_tsnr2:
            continue

        # build EXPOSURES row from aggregated per-petal TSNR2
        exp_row = _make_metadata_row(row)
        exp_row['PROGRAM'] = faflavor2program(exp_row['FAFLAVOR'])
        exp_row['SKY_MAG_G_SPEC'] = np.float32(row.get('SKY_MAG_G_SPEC', 99.0))
        exp_row['SKY_MAG_R_SPEC'] = np.float32(row.get('SKY_MAG_R_SPEC', 99.0))
        exp_row['SKY_MAG_Z_SPEC'] = np.float32(row.get('SKY_MAG_Z_SPEC', 99.0))
        for tracer in _TSNR2_TRACERS:
            exp_row['TSNR2_{}'.format(tracer)] = np.float32(
                np.mean([petal_tsnr2[p][tracer] for p in petal_tsnr2]))

        exp_row = _add_efftimes(exp_row)
        exp_row = _add_gfa_zero_cols(exp_row)
        exposures_rows.append(exp_row)

    if not frames_rows:
        log.warning('No FRAMES rows produced by build_tables')

    if not exposures_rows:
        log.warning('No EXPOSURES rows produced by build_tables')
        return _empty_tables()

    frames_table = Table(rows=frames_rows)
    frames_table.meta['EXTNAME'] = 'FRAMES'

    # -- enforce column order on EXPOSURES ----------------------------------
    exposures_raw = Table(rows=exposures_rows)
    exposures_raw.meta['EXTNAME'] = 'EXPOSURES'
    exposures_table = _reorder_exposures(exposures_raw)

    # -- sort by EXPID -------------------------------------------------------
    if len(exposures_table) > 0:
        ii = np.argsort(exposures_table['EXPID'])
        exposures_table = exposures_table[ii]
    if len(frames_table) > 0:
        sort_keys = ['{:08d}-{}'.format(int(e), c) for e, c in
                     zip(frames_table['EXPID'], frames_table['CAMERA'])]
        ii = np.argsort(sort_keys)
        frames_table = frames_table[ii]

    return frames_table, exposures_table


def _make_metadata_row(src):
    """Extract scalar metadata from a row dict.

    Args:
        src: dict with NIGHT, EXPID, TILEID, etc.

    Returns:
        dict with scalar metadata keys only.
    """
    keys = ('NIGHT', 'EXPID', 'TILEID', 'TILERA', 'TILEDEC', 'MJD', 'EXPTIME',
            'AIRMASS', 'EBV', 'SEEING_ETC', 'EFFTIME_ETC',
            'SURVEY', 'FAPRGRM', 'FAFLAVOR', 'GOALTYPE', 'GOALTIME', 'MINTFRAC')
    row = {}
    for k in keys:
        if k in src:
            row[k] = src[k]
    return row


def _empty_tables():
    """Return empty (FRAMES, EXPOSURES) tables with schemas for bad-only runs."""
    string_cols = {'SURVEY', 'PROGRAM', 'FAPRGRM', 'FAFLAVOR', 'GOALTYPE'}
    int_cols = {'NIGHT', 'EXPID', 'TILEID'}
    exposures = Table()
    for col in _EXP_SUMMARY_COLUMN_ORDER:
        if col in string_cols:
            dtype = 'U32'
        elif col in int_cols:
            dtype = 'i4'
        elif col == 'MJD':
            dtype = 'f8'
        else:
            dtype = 'f4'
        exposures[col] = np.array([], dtype=dtype)
    frames = exposures[list(_make_metadata_row({col: None for col in exposures.colnames}))].copy()
    frames['CAMERA'] = np.array([], dtype='U2')
    for tracer in _TSNR2_TRACERS:
        frames['TSNR2_' + tracer] = np.array([], dtype='f4')
    frames['TSNR2_ALPHA'] = np.array([], dtype='f4')
    frames.meta['EXTNAME'] = 'FRAMES'
    exposures.meta['EXTNAME'] = 'EXPOSURES'
    return frames, exposures


def _add_efftimes(row):
    """Return a copy of row with EFFTIME_* columns added.

    Args:
        row: dict with TSNR2_{tracer} keys and NIGHT, GOALTYPE keys.

    Returns:
        row: the same dict with EFFTIME_* keys added.
    """
    row['LRG_EFFTIME_DARK'] = np.float32(tsnr2_to_efftime(row.get('TSNR2_LRG', 0.0), 'LRG'))
    row['ELG_EFFTIME_DARK'] = np.float32(tsnr2_to_efftime(row.get('TSNR2_ELG', 0.0), 'ELG'))
    row['BGS_EFFTIME_BRIGHT'] = np.float32(tsnr2_to_efftime(row.get('TSNR2_BGS', 0.0), 'BGS'))
    row['LYA_EFFTIME_DARK'] = np.float32(tsnr2_to_efftime(row.get('TSNR2_LYA', 0.0), 'LYA'))
    row['GPB_EFFTIME_DARK'] = np.float32(tsnr2_to_efftime(row.get('TSNR2_GPBDARK', 0.0), 'GPBDARK'))
    row['GPB_EFFTIME_BRIGHT'] = np.float32(tsnr2_to_efftime(row.get('TSNR2_GPBBRIGHT', 0.0), 'GPBBRIGHT'))
    row['GPB_EFFTIME_BACKUP'] = np.float32(tsnr2_to_efftime(row.get('TSNR2_GPBBACKUP', 0.0), 'GPBBACKUP'))

    night = int(row.get('NIGHT', 0))
    goaltype = str(row.get('GOALTYPE', 'dark')).lower()

    if goaltype == 'bright':
        efftime_spec = row['BGS_EFFTIME_BRIGHT']
    elif goaltype == 'backup':
        gpb_backup = row['GPB_EFFTIME_BACKUP']
        efftime_spec = gpb_backup if gpb_backup > 0.0 else row['BGS_EFFTIME_BRIGHT']
    else:
        # dark: ELG before Sept 2021, LRG after
        if night < _SEPT_2021_CUTOFF:
            efftime_spec = row['ELG_EFFTIME_DARK']
        else:
            efftime_spec = row['LRG_EFFTIME_DARK']

    row['EFFTIME_SPEC'] = np.float32(efftime_spec)
    return row


def _add_gfa_zero_cols(row):
    """Return row with GFA and EFFTIME_GFA columns initialised to zero if absent.

    Args:
        row: dict.

    Returns:
        row: the same dict with GFA zero-fill keys added.
    """
    gfa_cols = ('TRANSPARENCY_GFA', 'SEEING_GFA', 'FIBER_FRACFLUX_GFA',
                'FIBER_FRACFLUX_ELG_GFA', 'FIBER_FRACFLUX_BGS_GFA',
                'FIBERFAC_GFA', 'FIBERFAC_ELG_GFA', 'FIBERFAC_BGS_GFA',
                'AIRMASS_GFA', 'SKY_MAG_AB_GFA',
                'EFFTIME_GFA', 'EFFTIME_DARK_GFA', 'EFFTIME_BRIGHT_GFA', 'EFFTIME_BACKUP_GFA')
    # float64, as in the stored files and the GFA tables, so comparisons of
    # GFA values across runs are not affected by float32 rounding.
    for col in gfa_cols:
        row.setdefault(col, np.float64(0.0))
    return row


def _reorder_exposures(table):
    """Reorder EXPOSURES table columns to match _EXP_SUMMARY_COLUMN_ORDER.

    Missing columns are zero-filled.  Extra columns not in the order list
    are appended after the standard columns with a warning.

    Args:
        table: astropy.table.Table with EXTNAME='EXPOSURES'.

    Returns:
        astropy.table.Table with columns in _EXP_SUMMARY_COLUMN_ORDER order
        followed by any extra columns.
    """
    log = get_logger()

    extra_cols = [c for c in table.colnames if c not in _EXP_SUMMARY_COLUMN_ORDER]
    if extra_cols:
        log.warning('Columns not in _EXP_SUMMARY_COLUMN_ORDER (appending at end): {}'.format(extra_cols))

    # String columns in the standard order need string fill values.
    _STR_COLS = frozenset({'SURVEY', 'PROGRAM', 'FAPRGRM', 'FAFLAVOR', 'GOALTYPE'})

    new_table = Table()
    new_table.meta = dict(table.meta)
    for col in _EXP_SUMMARY_COLUMN_ORDER:
        if col in table.colnames:
            new_table[col] = table[col]
        elif col in _STR_COLS:
            new_table[col] = np.full(len(table), '', dtype='U32')
        else:
            new_table[col] = np.zeros(len(table), dtype=np.float32)
    for col in extra_cols:
        new_table[col] = table[col]

    return new_table


def _get_default_for_col(table, col):
    """Return a type-appropriate default value for a missing column entry.

    Args:
        table: astropy.table.Table containing col.
        col: str, column name.

    Returns:
        '' for string/bytes/object dtypes, 0.0 otherwise.
    """
    if table[col].dtype.kind in ('U', 'S', 'O'):
        return ''
    return 0.0


# ---------------------------------------------------------------------------
# Bad exposure injection
# ---------------------------------------------------------------------------

def _bad_exposure_skymags(args_tuple):
    """Return sky magnitudes for a bad exposure, given (night, expid).

    Computed whenever the exposure directory exists, as for good exposures
    without stored values (compute_skymag gives 99 if there are no sky
    files); NaN if nothing was processed at all.  Takes one tuple so that it
    can be used with multiprocessing.Pool.map.
    """
    night, expid = args_tuple
    expdir = os.path.join(specprod_root(), 'exposures', str(int(night)), '{:08d}'.format(int(expid)))
    if os.path.isdir(expdir):
        return get_skymag_values(int(night), int(expid))
    return {'SKY_MAG_G_SPEC': np.float32(np.nan), 'SKY_MAG_R_SPEC': np.float32(np.nan),
            'SKY_MAG_Z_SPEC': np.float32(np.nan)}


def inject_bad_exposures(exposures_table, frames_table, bad_expids, cameras=None, skymags=None):
    """Add zero-filled rows for EXPIDs that were not fully processed, or
    whose processed outputs could not be read.

    For each entry in bad_expids that is not already in exposures_table, adds
    one row to exposures_table and one row per camera in frames_table.  TSNR2,
    EFFTIME_SPEC, and all computed columns are set to zero.  Targeting metadata
    is populated from the exposure_table entry, with values it lacks (e.g.
    FAFLAVOR, MINTFRAC) and TILERA/TILEDEC taken from the fiberassign header.

    Args:
        exposures_table: astropy.table.Table.
        frames_table: astropy.table.Table.
        bad_expids: list of dict, as returned by collect_science_expids().
        cameras: Optional selected camera names, also applied to bad exposures.
        skymags: Optional dict mapping EXPID -> sky magnitude dict, e.g.
            precomputed in parallel with _bad_exposure_skymags(); exposures
            not in it are computed here.

    Returns:
        (exposures_table, frames_table): tuple of astropy.table.Table with
            bad-exposure rows appended.
    """
    log = get_logger()

    existing_expids = set(exposures_table['EXPID'].tolist()) if len(exposures_table) > 0 else set()

    for be in bad_expids:
        expid = int(be['EXPID'])
        if expid in existing_expids:
            continue

        entry = {
            'NIGHT': int(be['NIGHT']),
            'EXPID': expid,
            'TILEID': int(be['TILEID']),
            'MJD': float(be.get('MJD-OBS', 0.0)),
            'EXPTIME': float(be.get('EXPTIME', 0.0)),
            'AIRMASS': np.float32(0.0),
            'EBV': np.float32(0.0),
            'SEEING_ETC': np.float32(0.0),
            'EFFTIME_ETC': np.float32(float(be.get('EFFTIME_ETC', 0.0))),
            'TILERA': np.float32(-999.0),
            'TILEDEC': np.float32(-999.0),
        }

        # -- EBV from EBVFAC ------------------------------------------------
        ebvfac = float(be.get('EBVFAC', 1.0))
        if ebvfac > 0:
            entry['EBV'] = np.float32(2.5 * np.log10(ebvfac) / 2.165)

        # -- targeting metadata: exposure table, then fiberassign header ----
        for key in ('SURVEY', 'FAPRGRM', 'FAFLAVOR', 'GOALTYPE', 'GOALTIME', 'MINTFRAC'):
            entry[key] = be.get(key, _TARG_DEFAULTS.get(key, 'unknown'))

        # TILERA/TILEDEC and targeting from fiberassignsvn, else raw data dir
        fa_filename, exists = findfile('fiberassignsvn', tile=int(be['TILEID']), return_exists=True, readonly=True)
        if not exists:
            fa_filename, exists = findfile('fiberassign', night=int(be['NIGHT']), expid=expid,
                                           tile=int(be['TILEID']), return_exists=True, readonly=True)
        if exists:
            fa_hdr = fitsio.read_header(fa_filename, 0)
            entry = _fill_targ_from_header(entry, fa_hdr)
            entry['TILERA'] = np.float32(fa_hdr['TILERA'])
            entry['TILEDEC'] = np.float32(fa_hdr['TILEDEC'])
        else:
            log.error('No fiberassign for TILEID={} expid={}'.format(be['TILEID'], expid))

        entry = derive_targ_info(entry)
        entry['PROGRAM'] = faflavor2program(entry['FAFLAVOR'])

        # -- sky mags (see _bad_exposure_skymags) -----------------------------
        if skymags is not None and expid in skymags:
            entry.update(skymags[expid])
        else:
            entry.update(_bad_exposure_skymags((be['NIGHT'], expid)))

        for tracer in _TSNR2_TRACERS:
            entry['TSNR2_{}'.format(tracer)] = np.float32(0.0)
        entry['TSNR2_ALPHA'] = np.float32(np.nan)
        entry = _add_efftimes(entry)
        entry = _add_gfa_zero_cols(entry)

        # -- add EXPOSURES row ----------------------------------------------
        exp_vals = [entry.get(col, _get_default_for_col(exposures_table, col))
                    for col in exposures_table.colnames]
        log.warning('Adding bad exposure EXPID={}'.format(expid))
        exposures_table.add_row(exp_vals)
        existing_expids.add(expid)

        # -- add FRAMES rows ------------------------------------------------
        camword = be.get('CAMWORD', 'a0123456789')
        badcamword = be.get('BADCAMWORD', '')
        active_cameras = decode_camword(difference_camwords(camword, badcamword))
        if cameras is not None:
            active_cameras = [cam for cam in active_cameras if cam in cameras]

        for cam in active_cameras:
            entry['CAMERA'] = cam
            cam_vals = [entry.get(col, _get_default_for_col(frames_table, col))
                        for col in frames_table.colnames]
            log.debug('Adding bad exposure FRAMES row EXPID={} CAMERA={}'.format(expid, cam))
            frames_table.add_row(cam_vals)

    return exposures_table, frames_table


# ---------------------------------------------------------------------------
# GFA enrichment
# ---------------------------------------------------------------------------

def add_gfa_columns(exposures_table, gfa_proc_dir):
    """Join GFA summary data onto the exposures table.

    Reads the GFA offline summary file via read_gfa_data() and joins on EXPID.
    Renames FWHM_ASEC -> SEEING_GFA.  NaN values are replaced with zero.

    Args:
        exposures_table: astropy.table.Table.
        gfa_proc_dir: str, path to the GFA processing directory.

    Returns:
        (exposures_table, changed_nights): tuple of the updated
            astropy.table.Table and a list of int nights where GFA values
            changed.
    """
    log = get_logger()

    try:
        gfa_table = read_gfa_data(gfa_proc_dir)
    except Exception as e:
        log.error('Could not read GFA data from {}: {}'.format(gfa_proc_dir, e))
        return exposures_table, []

    # build EXPID -> index map for the GFA table
    e2i = {int(e): i for i, e in enumerate(gfa_table['EXPID'])}

    jj = []  # indices into exposures_table
    ii = []  # corresponding indices into gfa_table
    for j, e in enumerate(exposures_table['EXPID']):
        if int(e) in e2i:
            jj.append(j)
            ii.append(e2i[int(e)])

    if not jj:
        log.warning('No EXPIDs matched between exposures_table and GFA data')
        return exposures_table, []

    jj = np.array(jj)
    ii = np.array(ii)
    matched_nights = exposures_table['NIGHT'][jj]
    changed_nights = []

    gfa_col_map = {
        'TRANSPARENCY': 'TRANSPARENCY_GFA',
        'FWHM_ASEC': 'SEEING_GFA',
        'FIBER_FRACFLUX': 'FIBER_FRACFLUX_GFA',
        'FIBER_FRACFLUX_ELG': 'FIBER_FRACFLUX_ELG_GFA',
        'FIBER_FRACFLUX_BGS': 'FIBER_FRACFLUX_BGS_GFA',
        'FIBERFAC': 'FIBERFAC_GFA',
        'FIBERFAC_ELG': 'FIBERFAC_ELG_GFA',
        'FIBERFAC_BGS': 'FIBERFAC_BGS_GFA',
        'SKY_MAG_AB': 'SKY_MAG_AB_GFA',
        'AIRMASS': 'AIRMASS_GFA',
    }

    for gfa_col, exp_col in gfa_col_map.items():
        if gfa_col not in gfa_table.colnames:
            continue
        if exp_col not in exposures_table.colnames:
            exposures_table[exp_col] = np.zeros(len(exposures_table), dtype=np.float64)

        gfa_vals = np.asarray(gfa_table[gfa_col][ii], dtype=float)
        exp_vals = np.asarray(exposures_table[exp_col][jj], dtype=float)

        # replace NaN with zero
        nan_mask = ~np.isfinite(gfa_vals)
        if nan_mask.any():
            log.warning('{} NaN values in GFA column {}; replacing with zero'.format(
                nan_mask.sum(), gfa_col))
            gfa_vals[nan_mask] = 0.0

        changed = ~np.isclose(exp_vals, gfa_vals, rtol=1e-6, atol=0)
        if changed.any():
            exposures_table[exp_col][jj] = gfa_vals
            changed_nights.extend(matched_nights[changed].tolist())

    return exposures_table, sorted(set(changed_nights))


def add_gfa_efftimes(exposures_table):
    """Compute GFA-based effective times and return the updated exposures table.

    Requires GFA columns (TRANSPARENCY_GFA, FIBERFAC_GFA, etc.) and sky
    magnitude columns (SKY_MAG_R_SPEC) to be present.  Calls
    compute_efftime() from desispec.efftime.

    Adds EFFTIME_DARK_GFA, EFFTIME_BRIGHT_GFA, EFFTIME_BACKUP_GFA, EFFTIME_GFA.

    Args:
        exposures_table: astropy.table.Table.

    Returns:
        exposures_table: astropy.table.Table with GFA efftime columns added.
    """
    log = get_logger()

    required_cols = ('TRANSPARENCY_GFA', 'FIBERFAC_GFA', 'FIBERFAC_ELG_GFA', 'FIBERFAC_BGS_GFA',
                     'FIBER_FRACFLUX_GFA', 'FIBER_FRACFLUX_ELG_GFA', 'FIBER_FRACFLUX_BGS_GFA',
                     'SKY_MAG_R_SPEC', 'EBV', 'EXPTIME', 'AIRMASS')
    missing = [c for c in required_cols if c not in exposures_table.colnames]
    if missing:
        log.warning('add_gfa_efftimes: missing columns {}; skipping'.format(missing))
        return exposures_table

    # A later GFA update can invalidate an earlier measurement. Clear stale
    # effective times before selecting the rows that can be recalculated.
    # float64 matches the stored columns, so unchanged values round-trip exactly.
    for col in ('EFFTIME_DARK_GFA', 'EFFTIME_BRIGHT_GFA', 'EFFTIME_BACKUP_GFA', 'EFFTIME_GFA'):
        exposures_table[col] = np.zeros(len(exposures_table), dtype=np.float64)

    # only rows with valid GFA data (transparency > 0) and a measured sky
    # (compute_skymag returns 99 when no sky data is available)
    skymag = np.asarray(exposures_table['SKY_MAG_R_SPEC'], dtype=float)
    valid = (exposures_table['TRANSPARENCY_GFA'] > 0) & np.isfinite(skymag) & (skymag < 90)
    if not valid.any():
        log.warning('No rows with valid GFA transparency and sky magnitude; skipping GFA efftimes')
        return exposures_table

    efftime_dark, efftime_bright, efftime_backup = compute_efftime(exposures_table[valid])

    exposures_table['EFFTIME_DARK_GFA'][valid] = efftime_dark
    exposures_table['EFFTIME_BRIGHT_GFA'][valid] = efftime_bright
    exposures_table['EFFTIME_BACKUP_GFA'][valid] = efftime_backup

    # EFFTIME_GFA: dark by default, bright for GOALTYPE=bright, backup for GOALTYPE=backup
    efftime_gfa = exposures_table['EFFTIME_DARK_GFA'].copy()
    goaltype = np.array([str(g).lower() for g in exposures_table['GOALTYPE']])
    efftime_gfa[goaltype == 'bright'] = exposures_table['EFFTIME_BRIGHT_GFA'][goaltype == 'bright']
    efftime_gfa[goaltype == 'backup'] = exposures_table['EFFTIME_BACKUP_GFA'][goaltype == 'backup']
    exposures_table['EFFTIME_GFA'] = efftime_gfa

    log.info('Added GFA efftimes for {} rows'.format(valid.sum()))
    return exposures_table


# ---------------------------------------------------------------------------
# Merge with pre-existing output
# ---------------------------------------------------------------------------

def merge_exposures(existing_table, new_table):
    """Upsert new rows into an existing EXPOSURES table by EXPID.

    Rows in new_table replace matching rows in existing_table.  Rows present
    only in existing_table are preserved. GFA enrichment is retained for
    matching rows, since new rows contain placeholders until the GFA join.

    Args:
        existing_table: astropy.table.Table.
        new_table: astropy.table.Table.

    Returns:
        astropy.table.Table: merged table.
    """
    log = get_logger()

    existing_table = existing_table.copy()
    new_table = new_table.copy()
    existing_indices = {int(expid): i for i, expid in enumerate(existing_table['EXPID'])}
    for col in existing_table.colnames:
        if col.endswith('_GFA'):
            if col not in new_table.colnames:
                new_table[col] = np.zeros(len(new_table), dtype=existing_table[col].dtype)
            for j, expid in enumerate(new_table['EXPID']):
                if int(expid) in existing_indices:
                    new_table[col][j] = existing_table[col][existing_indices[int(expid)]]

    existing_expids = set(existing_table['EXPID'].tolist())
    new_expids = set(new_table['EXPID'].tolist())
    replace_count = len(existing_expids & new_expids)
    keep_count = len(existing_expids - new_expids)
    add_count = len(new_expids - existing_expids)

    log.info('EXPOSURES merge: keep={}, replace={}, add={}'.format(
        keep_count, replace_count, add_count))

    keep_mask = ~np.isin(existing_table['EXPID'], list(new_expids))

    # add any columns present in new_table but absent from existing
    for col in new_table.colnames:
        if col not in existing_table.colnames:
            log.info('Adding new column {} to existing EXPOSURES table'.format(col))
            existing_table[col] = np.zeros(len(existing_table), dtype=new_table[col].dtype)

    if keep_mask.any():
        merged = vstack([existing_table[keep_mask], new_table])
    else:
        merged = new_table.copy()

    merged.meta['EXTNAME'] = 'EXPOSURES'
    ii = np.argsort(merged['EXPID'])
    return merged[ii]


def merge_frames(existing_table, new_table, replace_expids=None, cameras=None):
    """Upsert new rows into an existing FRAMES table by (EXPID, CAMERA).

    Args:
        existing_table: astropy.table.Table.
        new_table: astropy.table.Table.
        replace_expids: Optional exposure IDs whose selected camera rows are
            replaced as a group, including removal of cameras no longer usable.
        cameras: Camera selection for group replacement; None means all cameras.

    Returns:
        astropy.table.Table: merged table.
    """
    log = get_logger()
    existing_table = existing_table.copy()
    new_table = new_table.copy()

    def _joint_key(table):
        return ['{:08d}-{}'.format(int(e), c) for e, c in zip(table['EXPID'], table['CAMERA'])]

    existing_keys = set(_joint_key(existing_table))
    new_keys = set(_joint_key(new_table))
    replace_count = len(existing_keys & new_keys)
    keep_count = len(existing_keys - new_keys)
    add_count = len(new_keys - existing_keys)

    log.info('FRAMES merge: keep={}, replace={}, add={}'.format(keep_count, replace_count, add_count))

    new_key_arr = np.array(list(new_keys))
    existing_key_arr = np.array(_joint_key(existing_table))
    keep_mask = ~np.isin(existing_key_arr, new_key_arr)
    if replace_expids is not None:
        replace = np.isin(existing_table['EXPID'], list(replace_expids))
        if cameras is not None:
            replace &= np.isin(existing_table['CAMERA'], list(cameras))
        keep_mask &= ~replace

    # QA summaries need not contain alpha. Preserve measured alpha for any
    # updated camera that did not recalculate it, including masked entries.
    if 'TSNR2_ALPHA' in existing_table.colnames:
        if 'TSNR2_ALPHA' not in new_table.colnames:
            new_table['TSNR2_ALPHA'] = np.full(len(new_table), np.nan, dtype=np.float32)
        indices = {key: i for i, key in enumerate(_joint_key(existing_table))}
        for j, key in enumerate(_joint_key(new_table)):
            value = new_table['TSNR2_ALPHA'][j]
            if key in indices and (np.ma.is_masked(value) or not np.isfinite(value)):
                new_table['TSNR2_ALPHA'][j] = existing_table['TSNR2_ALPHA'][indices[key]]

    for col in new_table.colnames:
        if col not in existing_table.colnames:
            log.info('Adding new column {} to existing FRAMES table'.format(col))
            existing_table[col] = np.zeros(len(existing_table), dtype=new_table[col].dtype)

    if keep_mask.any():
        merged = vstack([existing_table[keep_mask], new_table])
    else:
        merged = new_table.copy()

    merged.meta['EXTNAME'] = 'FRAMES'
    sort_keys = np.array(_joint_key(merged))
    ii = np.argsort(sort_keys)
    return merged[ii]


def update_exposure_tsnr(exposures_table, frames_table, expids):
    """Return exposure summaries with TSNR2 recomputed from merged camera rows.

    Args:
        exposures_table: Exposure metadata and current summary values.
        frames_table: Merged camera rows, including retained unselected cameras.
        expids: Exposure IDs to update.

    Returns:
        Table: Copy of exposures_table with consistent TSNR2 and effective times.
    """
    result = exposures_table.copy()
    indices = {int(expid): i for i, expid in enumerate(result['EXPID'])}
    groups = {}
    selected = set(expids)
    for frame in frames_table[np.isin(frames_table['EXPID'], list(selected))]:
        expid = int(frame['EXPID'])
        groups.setdefault(expid, {}).setdefault(int(frame['CAMERA'][1]), []).append(frame)
    for expid in selected:
        i = indices[expid]
        row = dict(result[i])
        petals = groups.get(expid, {})
        for tracer in _TSNR2_TRACERS:
            col = 'TSNR2_' + tracer
            sums = [sum(float(frame[col]) for frame in frames) for frames in petals.values()]
            row[col] = np.float32(np.mean(sums) if sums else 0.0)
        row = _add_efftimes(row)
        for col in result.colnames:
            if col.startswith('TSNR2_') or ('EFFTIME' in col and not col.endswith(('_GFA', '_ETC'))):
                result[col][i] = row[col]
    return result


def add_skymag_columns(exposures_table, sky_table):
    """Return a copy with sky-table values joined by NIGHT and EXPID.

    Args:
        exposures_table: Exposure summary table.
        sky_table: Table containing EXPID, optionally NIGHT, and SKY_MAG_G/R/Z
            (or SKY_MAG_G/R/Z_SPEC) columns.

    Returns:
        Table: Updated exposure summary; unmatched rows keep their values.
    """
    result = exposures_table.copy()
    keys = ['NIGHT', 'EXPID'] if 'NIGHT' in sky_table.colnames else ['EXPID']
    indices = {tuple(row[key] for key in keys): i for i, row in enumerate(sky_table)}
    for i, row in enumerate(result):
        j = indices.get(tuple(row[key] for key in keys))
        if j is None:
            continue
        for band in ('G', 'R', 'Z'):
            col = 'SKY_MAG_' + band + '_SPEC'
            source = col if col in sky_table.colnames else 'SKY_MAG_' + band
            if source in sky_table.colnames:
                result[col][i] = sky_table[source][j]
    return result


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------

def write_output(exposures_table, frames_table, outfile):
    """Write the FITS output file with EXPOSURES and FRAMES extensions.

    Writes atomically via a temporary file and os.rename.  Also writes the
    EXPOSURES table as a CSV sidecar.  Numeric columns are rounded before
    writing the CSV to match v1 precision.

    Uses astropy.io.fits for the FITS write, preserving the table column dtypes.

    Args:
        exposures_table: astropy.table.Table with EXTNAME='EXPOSURES'.
        frames_table: astropy.table.Table with EXTNAME='FRAMES'.
        outfile: str, path to the output FITS file (must end in '.fits').

    Returns:
        None
    """
    log = get_logger()

    hdus = fits.HDUList()
    hdus.append(fits.convenience.table_to_hdu(exposures_table))
    hdus.append(fits.convenience.table_to_hdu(frames_table))

    tmpfile = get_tempfilename(outfile)
    hdus.writeto(tmpfile, overwrite=True)
    os.rename(tmpfile, outfile)
    log.info('Wrote {}'.format(outfile))

    # -- CSV sidecar --------------------------------------------------------
    csv_file = os.path.splitext(outfile)[0] + '.csv'
    csv_table = exposures_table.copy(copy_data=True)
    for col in csv_table.colnames:
        if any(x in col for x in ('TIME', 'SNR2')):
            try:
                csv_table[col] = np.around(csv_table[col].astype(float), 1)
            except (ValueError, TypeError) as e:
                log.warning('Cannot round column {}: {}'.format(col, e))
        elif any(x in col for x in ('SEEING', 'FWHM', 'FIBER_FRACFLUX', 'FIBERFAC',
                                    'TRANS', 'MAG', 'AIRMASS', 'EBV')):
            try:
                csv_table[col] = np.around(csv_table[col].astype(float), 3)
            except (ValueError, TypeError) as e:
                log.warning('Cannot round column {}: {}'.format(col, e))

    csv_table.write(csv_file, overwrite=True)
    log.info('Wrote {}'.format(csv_file))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(options=None):
    """Main entry point for desi_tsnr_afterburner.

    Returns:
        int: 0 on success, 1 on error.
    """
    log = get_logger()

    args = parse(options)

    # -- resolve production directory --------------------------------------
    if args.prod is not None:
        prod = specprod_root(args.prod)
        # Override environment so all subsequent findfile calls use it
        prod_clean = prod.rstrip('/')
        os.environ['DESI_SPECTRO_REDUX'] = os.path.dirname(prod_clean)
        os.environ['SPECPROD'] = os.path.basename(prod_clean)
    else:
        prod = specprod_root()

    if args.outfile is None:
        specprod_name = os.path.basename(prod.rstrip('/'))
        args.outfile = os.path.join(prod, 'exposures-{}.fits'.format(specprod_name))
        log.info('outfile not specified; using {}'.format(args.outfile))

    if args.outfile.endswith('.csv'):
        log.info(f"Output filename '{args.outfile}' ends with '.csv'; changing to '.fits' but both versions will be saved.")
        args.outfile = os.path.splitext(args.outfile)[0] + '.fits'

    if not args.outfile.endswith('.fits'):
        log.critical("Output filename '{}' must end with '.fits'".format(args.outfile))
        return 1

    log.info('prod = {}'.format(prod))
    log.info('outfile = {}'.format(args.outfile))

    # -- MPI setup ----------------------------------------------------------
    comm = None
    rank = 0
    size = 1
    if args.mpi:
        try:
            from desispec.parallel import use_mpi as _use_mpi
            if _use_mpi():
                from mpi4py import MPI
                comm = MPI.COMM_WORLD
                rank = comm.Get_rank()
                size = comm.Get_size()
        except ImportError:
            log.warning('mpi4py not available; running without MPI')

    # -- determine nights ---------------------------------------------------
    if args.nights is None:
        all_nights = find_processed_nights() if rank == 0 else []
    else:
        all_nights = parse_int_args(args.nights, include_end=True) if rank == 0 else []

    if comm is not None:
        all_nights = comm.bcast(all_nights, root=0)

    # each rank processes a slice of nights
    ranknights = all_nights[rank::size]

    if args.expids is not None:
        expids_filter = set(parse_int_args(args.expids, include_end=True))
    else:
        expids_filter = None

    if args.cameras is not None:
        cameras_filter = set(decode_camword(parse_cameras(args.cameras)))
    else:
        cameras_filter = None

    # -- load pre-existing output for merge (default behavior) -------------
    preexisting_exposures = None
    preexisting_frames = None
    if rank == 0 and not args.overwrite and os.path.isfile(args.outfile):
        log.info('Will merge with pre-existing {}'.format(args.outfile))
        try:
            preexisting_exposures = read_table(args.outfile, 'EXPOSURES')
        except (KeyError, OSError):
            log.warning('EXPOSURES HDU not found; trying TSNR2_EXPID')
            try:
                preexisting_exposures = read_table(args.outfile, 'TSNR2_EXPID')
            except (KeyError, OSError):
                pass
        try:
            preexisting_frames = read_table(args.outfile, 'FRAMES')
        except (KeyError, OSError):
            log.warning('FRAMES HDU not found; trying TSNR2_FRAME')
            try:
                preexisting_frames = read_table(args.outfile, 'TSNR2_FRAME')
            except (KeyError, OSError):
                pass

    # -- refuse to merge into an unreadable or inconsistent summary ---------
    conflict = False
    if rank == 0 and not args.overwrite and os.path.isfile(args.outfile):
        if preexisting_exposures is None or preexisting_frames is None:
            conflict = True
            log.critical('Cannot read EXPOSURES and FRAMES from existing {}; not merging. '
                         'Use --overwrite to replace it'.format(args.outfile))
        else:
            orphans = set(preexisting_frames['EXPID'].tolist()) - set(preexisting_exposures['EXPID'].tolist())
            if orphans:
                conflict = True
                log.critical('FRAMES in existing {} has {} EXPIDs not in EXPOSURES; not merging. '
                             'Use --overwrite to replace it'.format(args.outfile, len(orphans)))

    # -- --no-update: refuse to replace rows already in the output ----------
    if rank == 0 and not conflict and not args.update and preexisting_exposures is not None:
        present = np.isin(preexisting_exposures['NIGHT'], all_nights)
        if expids_filter is not None:
            present &= np.isin(preexisting_exposures['EXPID'], list(expids_filter))
        if present.any():
            conflict = True
            log.critical('--no-update: {} exposures on nights {} are already in {}; '
                         'rerun without --no-update to replace them'.format(
                             present.sum(), sorted(set(preexisting_exposures['NIGHT'][present].tolist())),
                             args.outfile))
    if comm is not None:
        conflict = comm.bcast(conflict, root=0)
    if conflict:
        return 1

    # -- collect exposures --------------------------------------------------
    good_expids, bad_expids = collect_science_expids(ranknights, expids_filter)

    if not good_expids:
        log.warning('No good science exposures found for rank {}'.format(rank))

    # build camword_map: expid -> list of active cameras
    camword_map = {}
    for entry in good_expids:
        active = decode_camword(difference_camwords(entry['CAMWORD'], entry['BADCAMWORD']))
        if cameras_filter is not None:
            active = [c for c in active if c in cameras_filter]
        camword_map[int(entry['EXPID'])] = active
        if cameras_filter is not None and not active:
            log.warning('No --cameras selected for expid={}; it will be written as a zeroed row'.format(
                entry['EXPID']))

    read_error = None
    try:
        # One loading path supplies QA values, disk-score fallbacks, or recomputation.
        args_list = [(entry['NIGHT'], int(entry['EXPID']), args.recompute_skymags,
                      camword_map[int(entry['EXPID'])], args.recompute, args.alpha_only, args.details_dir,
                      entry)
                     for entry in good_expids]
        # One pool serves both the exposure loading and the bad-exposure sky mags.
        pool = multiprocessing.Pool(args.nproc) if args.nproc > 1 else None
        try:
            if pool is not None:
                mapper = pool.map
            else:
                mapper = lambda func, items: [func(item) for item in items]
            results = mapper(_read_one_exposure_wrapper, args_list)
            exposure_rows = [row for row in results if row is not None]
            frames_table, exposures_table = build_tables(exposure_rows, camword_map)

            # Processed exposures with nothing readable on disk always get zeroed
            # rows, replacing any earlier values; bad exposures only with --add-badexp.
            unreadable = [entry for entry, row in zip(good_expids, results) if row is None]
            zero_fill = (bad_expids if args.add_badexp else []) + unreadable
            if cameras_filter is not None and zero_fill:
                log.warning('With --cameras, existing rows for cameras outside the selection are kept for '
                            'zero-filled exposures {}'.format([int(be['EXPID']) for be in zero_fill]))
            if zero_fill:
                sky_args = [(be['NIGHT'], int(be['EXPID'])) for be in zero_fill]
                skymags = dict(zip([expid for night, expid in sky_args],
                                   mapper(_bad_exposure_skymags, sky_args)))
                exposures_table, frames_table = inject_bad_exposures(
                    exposures_table, frames_table, zero_fill, cameras=cameras_filter, skymags=skymags)
        finally:
            if pool is not None:
                pool.close()
                pool.join()

    except Exception as error:
        if comm is None:
            raise
        # All ranks must reach gather even when one fails to load an exposure.
        read_error = 'Rank {}: {}: {}'.format(rank, type(error).__name__, error)
        frames_table, exposures_table = _empty_tables()

    # Gather only new data. Both serial and MPI then use the same upsert path.
    if comm is not None:
        bad_ids = [int(be['EXPID']) for be in bad_expids]
        gathered = comm.gather((frames_table, exposures_table, read_error, bad_ids), root=0)
        if rank != 0:
            return int(read_error is not None)
        errors = [error for frm, exp, error, bad in gathered if error is not None]
        if errors:
            for error in errors:
                log.error(error)
            return 1
        bad_expids = [{'EXPID': expid} for frm, exp, error, bad in gathered for expid in bad]
        exposure_parts = [exp for frm, exp, error, bad in gathered if len(exp)]
        frame_parts = [frm for frm, exp, error, bad in gathered if len(frm)]
        frames_table, exposures_table = _empty_tables()
        if exposure_parts:
            exposures_table = vstack(exposure_parts, metadata_conflicts='silent')
        if frame_parts:
            frames_table = vstack(frame_parts, metadata_conflicts='silent')

    updated_expids = exposures_table['EXPID'].tolist()
    if preexisting_exposures is not None:
        exposures_table = merge_exposures(preexisting_exposures, exposures_table)
    if preexisting_frames is not None:
        frames_table = merge_frames(preexisting_frames, frames_table,
                                    replace_expids=updated_expids, cameras=cameras_filter)
    exposures_table = update_exposure_tsnr(exposures_table, frames_table, updated_expids)

    # Without --add-badexp, bad exposures (LASTSTEP != 'all', or every camera
    # marked bad) are removed entirely, as if they had been ignored from the
    # start. Their tiles are recomputed, or dropped if no exposures remain.
    removed_tiles = set()
    removed_any = False
    if not args.add_badexp and bad_expids:
        remove = np.isin(exposures_table['EXPID'], [int(be['EXPID']) for be in bad_expids])
        if remove.any():
            log.warning('Removing {} bad exposures from output: {}'.format(
                remove.sum(), exposures_table['EXPID'][remove].tolist()))
            removed_tiles = set(exposures_table['TILEID'][remove].tolist())
            removed_any = True
            frames_table = frames_table[~np.isin(frames_table['EXPID'], exposures_table['EXPID'][remove])]
            exposures_table = exposures_table[~remove]

    exposures_table.meta['EXTNAME'] = 'EXPOSURES'
    frames_table.meta['EXTNAME'] = 'FRAMES'
    if len(exposures_table):
        exposures_table.sort('EXPID')
    if len(frames_table):
        frames_table.sort(['EXPID', 'CAMERA'])

    # An empty result is only written when rows were intentionally removed.
    if len(exposures_table) == 0 and not removed_any:
        log.error('No valid exposures; nothing to write')
        return 1

    # An explicit sky table overrides stored values, unless recomputation was
    # requested with --recompute-skymags.
    if args.skymags is not None and not args.recompute_skymags:
        exposures_table = add_skymag_columns(exposures_table, Table.read(args.skymags))

    # -- GFA enrichment (rank 0 or non-MPI) --------------------------------
    gfa_nights = []
    if args.gfa_proc_dir is not None:
        exposures_table, gfa_nights = add_gfa_columns(exposures_table, args.gfa_proc_dir)
    # Always recompute: sky or GFA inputs may have changed for merged rows.
    if 'TRANSPARENCY_GFA' in exposures_table.colnames:
        eff_cols = ('EFFTIME_GFA', 'EFFTIME_DARK_GFA', 'EFFTIME_BRIGHT_GFA', 'EFFTIME_BACKUP_GFA')
        before = {col: np.asarray(exposures_table[col]).copy() for col in eff_cols}
        exposures_table = add_gfa_efftimes(exposures_table)
        # Ignore float rounding, but not NaN <-> finite transitions: tile
        # completeness treats NaN (skipped) and 0 (missing data) differently.
        changed = np.zeros(len(exposures_table), dtype=bool)
        for col in eff_cols:
            after = np.asarray(exposures_table[col])
            changed |= np.isfinite(before[col]) != np.isfinite(after)
            changed |= ~np.isclose(before[col], after, rtol=1e-5, atol=1e-3, equal_nan=True)
        gfa_nights = sorted(set(gfa_nights) | set(exposures_table['NIGHT'][changed].tolist()))
        if gfa_nights:
            log.info('GFA values changed for {} nights'.format(len(gfa_nights)))

    # -- tile completeness --------------------------------------------------
    if args.tile_completeness is not None:
        selection = np.ones(len(exposures_table), dtype=bool)
        if expids_filter is not None:
            selection &= np.isin(exposures_table['EXPID'], list(expids_filter))
        if args.nights is not None:
            selection &= np.isin(exposures_table['NIGHT'], all_nights)
        if gfa_nights:
            selection |= np.isin(exposures_table['NIGHT'], gfa_nights)

        if removed_tiles:
            selection |= np.isin(exposures_table['TILEID'], list(removed_tiles))

        tiles = np.unique(exposures_table['TILEID'][selection])
        selection = np.isin(exposures_table['TILEID'], tiles)
        log.info('Updating tile completeness for {} tiles'.format(len(tiles)))

        new_tile_table = compute_tile_completeness_table(
            exposures_table[selection], prod, auxiliary_table_filenames=args.aux)

        if os.path.isfile(args.tile_completeness):
            previous = Table.read(args.tile_completeness)
            # An empty CSV reads back with integer text columns, so only merge
            # when there are previous rows.
            if len(previous) > 0:
                new_tile_table = merge_tile_completeness_table(previous, new_tile_table)

        # drop tiles whose only exposures were removed above
        orphan = np.isin(new_tile_table['TILEID'], list(removed_tiles - set(exposures_table['TILEID'].tolist())))
        if orphan.any():
            log.warning('Removing {} tiles with no remaining exposures: {}'.format(
                orphan.sum(), new_tile_table['TILEID'][orphan].tolist()))
            new_tile_table = new_tile_table[~orphan]

        head = os.path.splitext(args.tile_completeness)[0]
        new_tile_table.write(head + '.fits', overwrite=True)
        new_tile_table.write(head + '.csv', format='ascii.csv', overwrite=True)
        log.info('Wrote tile completeness to {}'.format(head + '.fits'))

        # propagate any GOALTIME updates from tile table back to exposure tables
        for tileid, goaltime in zip(new_tile_table['TILEID'], new_tile_table['GOALTIME']):
            if goaltime > 0.0:
                ii = exposures_table['TILEID'] == tileid
                if np.any(exposures_table['GOALTIME'][ii] == 0.0):
                    exposures_table['GOALTIME'][ii] = goaltime
                jj = frames_table['TILEID'] == tileid
                if np.any(frames_table['GOALTIME'][jj] == 0.0):
                    frames_table['GOALTIME'][jj] = goaltime

    # -- write final output -------------------------------------------------
    write_output(exposures_table, frames_table, args.outfile)

    return 0
