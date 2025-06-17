# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
desispec.io.meta
================

IO metadata functions.
"""
from __future__ import absolute_import, division, print_function

import os
import datetime
import glob
import re
import numpy as np

from desiutil.log import get_logger
from .util import healpix_subdirectory, checkgzip, get_log_pathname

_desi_root_readonly=None
def get_desi_root_readonly():
    """
    Returns $DESI_ROOT_READONLY if set and path exists, otherwise $DESI_ROOT.

    Caches answer upon first call, i.e. setting $DESI_ROOT_READONLY to a
    different value part way through running will use previously cached value.
    This prevents it from re-checking a non-existent path N>>1 times.
    """
    global _desi_root_readonly
    log = get_logger()
    if _desi_root_readonly is not None:
        log.debug('Using cached _desi_root_readonly=%s', _desi_root_readonly)
        return _desi_root_readonly
    elif 'DESI_ROOT_READONLY' in os.environ:
        if os.path.exists(os.environ['DESI_ROOT_READONLY']):
            _desi_root_readonly = os.environ['DESI_ROOT_READONLY']
            log.debug("Using $DESI_ROOT_READONLY=%s", _desi_root_readonly)
        else:
            log.debug("$DESI_ROOT_READONLY=%s set but doesn't exist; using $DESI_ROOT=%s",
                    os.environ['DESI_ROOT_READONLY'], os.environ['DESI_ROOT'])
            _desi_root_readonly = os.environ['DESI_ROOT']
    else:
        log.debug('$DESI_ROOT_READONLY not set; using $DESI_ROOT=%s',
                os.environ['DESI_ROOT'])
        _desi_root_readonly = os.environ['DESI_ROOT']

    return _desi_root_readonly

def get_readonly_filepath(filepath):
    """
    Generate optimized path for read-only usage of filepath

    Args:
        filepath (str): full path to input file

    Returns: readonly_filepath using $DESI_ROOT_READONLY

    If a readonly filepath can't be derived, return original filepath
    """
    if 'DESI_ROOT' not in os.environ:
        return filepath
    else:
        desi_root = os.environ['DESI_ROOT']

    desi_root_readonly = get_desi_root_readonly()
    if filepath.startswith(desi_root) and desi_root != desi_root_readonly:
        filepath = filepath.replace(desi_root, desi_root_readonly, 1)

    return filepath

def get_findfile_argparser():
    """
    Defines an argparser for a command line tool to call findfile
    """
    import argparse
    parser = argparse.ArgumentParser(description="Returns the location where a file should be.")

    ## Required argument
    parser.add_argument("filetype", type=str, help="File type, typically the prefix, e.g., 'frame' or 'psf'.")

    ## Optional arguments depending on filetype
    parser.add_argument("-n", "--night", type=str,
                        help="YEARMMDD string or int.")
    parser.add_argument("-e", "--expid", type=int,
                        help="Integer exposure ID.")
    parser.add_argument("-c", "--camera", type=str,
                        help="'b0', 'r1', ..., 'z9'.")
    parser.add_argument("-t", "--tile", type=int,
                        help="Integer tile (pointing) number.")
    parser.add_argument("-g", "--groupname", type=str,
                        help="Spectral grouping name (e.g., 'healpix', 'cumulative', 'pernight').")
    parser.add_argument("--subgroup", type=str,
                        help="Subgrouping name for non-standard group names.")
    parser.add_argument("--healpix", type=int,
                        help="Healpix pixel number.")
    parser.add_argument("--nside", type=int, default=64,
                        help="Healpix nside (default: 64).")
    parser.add_argument("--band", type=str, choices=['b', 'r', 'z'],
                        help="Camera band ('b', 'r', or 'z').")
    parser.add_argument("--spectrograph", type=int,
                        help="Integer spectrograph number (0-9).")
    parser.add_argument("--survey", type=str,
                        help="Survey, e.g., sv1, sv3, main, special.")
    parser.add_argument("--faprogram", type=str,
                        help="Fiberassign program, e.g., dark, bright.")

    ## General options
    parser.add_argument("--rawdata_dir", type=str,
                        help="Overrides $DESI_SPECTRO_DATA.")
    parser.add_argument("--specprod_dir", type=str,
                        help="Overrides $DESI_SPECTRO_REDUX/$SPECPROD/.")
    parser.add_argument("--specprod", type=str,
                        help="Production name or full path to production.")
    parser.add_argument("--qaprod_dir", type=str,
                        help="Defaults to $DESI_SPECTRO_REDUX/$SPECPROD/QA/ if not provided.")
    parser.add_argument("--tiles_dir", type=str,
                        help="Defaults to $FIBER_ASSIGN_DIR if not provided.")
    parser.add_argument("--outdir", type=str,
                        help="Use this directory for output instead of canonical location.")
    parser.add_argument("--download", action="store_true",
                        help="If not found locally, try to fetch remotely.")
    parser.add_argument("--return_exists", action="store_true",
                        help="Also return whether the file exists.")
    parser.add_argument("--readonly", action="store_true",
                        help="Return read-only version of path if possible.")
    parser.add_argument("--logfile", action="store_true",
                        help="Return the pathname of the log instead of the data product itself.")

    return parser


def get_fits_compression_suffix() :
    """Return the prefered suffix for the compression of fits images.
       Can be set with the environment variable DESI_COMPRESSION with values NONE or GZ.
    """

    if not "DESI_COMPRESSION" in os.environ :
        return ".gz" # the default compression level

    compression_type = os.environ["DESI_COMPRESSION"].upper()
    valid_compression_types=["NONE","GZ","GZIP"]
    if compression_type not in valid_compression_types :
        raise KeyError(f"Invalid compression type '{compression_type}' from environment variable DESI_COMPRESSION. It must be among {valid_compression_types}")
    if compression_type in ["GZ","GZIP"] :
        return ".gz"
    elif compression_type=="NONE"  :
        return ""
    else :
        raise KeyError(f"unknown compression type '{comptype}'")

def findfile(filetype, night=None, expid=None, camera=None,
        tile=None, groupname=None, subgroup=None,
        healpix=None, nside=64, band=None, spectrograph=None,
        survey=None, faprogram=None, version=None,
        rawdata_dir=None, specprod_dir=None, specprod=None,
        tiles_dir=None, outdir=None,
        download=False, return_exists=False,
        readonly=False, logfile=False):
    """Returns location where file should be

    Args:
        filetype : file type, typically the prefix, e.g. "frame" or "psf"

    Args depending upon filetype:
        night : YEARMMDD string or int
        expid : integer exposure id
        camera : 'b0' 'r1' .. 'z9'
        tile : integer tile (pointing) number
        groupname : spectral grouping name (e.g. "healpix", "cumulative", "pernight")
        subgroup : (str) subgrouping name for non-standard groupnames
        healpix : healpix pixel number
        nside : healpix nside
        band : one of 'b','r','z' identifying the camera band
        spectrograph : integer spectrograph number, 0-9
        survey : e.g. sv1, sv3, main, special
        faprogram : fiberassign program, e.g. dark, bright
        version : (str) version of the zcatalog

    Options:
        rawdata_dir : overrides $DESI_SPECTRO_DATA
        specprod_dir : overrides $DESI_SPECTRO_REDUX/$SPECPROD/
        specprod : production name, or full path to production
        tiles_dir : defaults to $FIBER_ASSIGN_DIR if not provided
        download : if not found locally, try to fetch remotely
        outdir : use this directory for output instead of canonical location
        return_exists: if True, also return whether the file exists
        readonly: if True, return read-only version of path if possible
        logfile: if True, returns the pathname of the log instead of the data product itself

    Returns filename, or (filename, exists) if return_exists=True

    Raises:
        ValueError: for invalid file types, and other invalid input
        KeyError: for missing environment variables

    Notes:
        The readonly option uses $DESI_ROOT_READONLY if it is set and
        exists; otherwise it returns the normal read/write path.
        Also, desispec allows to compress or not fits files (controlled with $DESI_COMPRESSION).
        If findfile would return a (un)compressed file (like FILE.fits.gz) but the alternate already exists
        (FILE.fits), it will return the alternate filename if the option readonly is True but
        will raise an IOError otherwise (in order to avoid having both FILE.fits and FILE.fits.gz on disk).

    """
    log = get_logger()

    compsuffix=get_fits_compression_suffix()
    log.debug("compression suffix = '%s'", compsuffix)

    #- NOTE: specprod_dir is the directory $DESI_SPECTRO_REDUX/$SPECPROD,
    #-       specprod is just the environment variable $SPECPROD
    location = dict(
        #
        # Raw data.
        #
        raw = '{rawdata_dir}/{night}/{expid:08d}/desi-{expid:08d}.fits.fz',
        coordinates = '{rawdata_dir}/{night}/{expid:08d}/coordinates-{expid:08d}.fits',
        fiberassign = '{rawdata_dir}/{night}/{expid:08d}/fiberassign-{tile:06d}.fits.gz',
        etc = '{rawdata_dir}/{night}/{expid:08d}/etc-{expid:08d}.json',
        #
        # SVN products
        #
        fiberassignsvn = '{tiles_dir}/{tile3:03d}/fiberassign-{tile:06d}.fits.gz',
        #
        # Top level
        #
        exposure_table = '{specprod_dir}/exposure_tables/{month}/exposure_table_{night}.csv',
        override='{specprod_dir}/exposure_tables/{month}/override_{night}.yaml',
        processing_table = '{specprod_dir}/processing_tables/processing_table_{specprod}-{night}.csv',
        unprocessed_table = '{specprod_dir}/processing_tables/unprocessed_table_{specprod}-{night}.csv',
        exposures = '{specprod_dir}/exposures-{specprod}.fits',
        tiles = '{specprod_dir}/tiles-{specprod}.fits',
        exposures_csv = '{specprod_dir}/exposures-{specprod}.csv',
        tiles_csv = '{specprod_dir}/tiles-{specprod}.csv',
        #
        # preproc/
        # Note: fibermap files will eventually move to preproc.
        #
        fibermap = '{specprod_dir}/preproc/{night}/{expid:08d}/fibermap-{expid:08d}.fits',
        preproc = '{specprod_dir}/preproc/{night}/{expid:08d}/preproc-{camera}-{expid:08d}.fits{compsuffix}',
        preproc_for_cte = '{specprod_dir}/preproc/{night}/{expid:08d}/ctepreproc-{camera}-{expid:08d}.fits{compsuffix}',
        tilepix = '{specprod_dir}/preproc/{night}/{expid:08d}/tilepix-{tile}.json',
        #
        # exposures/
        # Note: calib has been renamed to fluxcalib, but that has not propagated fully through the pipeline.
        # Note: psfboot has been deprecated, but not ready to be removed yet.
        #
        calib = '{specprod_dir}/exposures/{night}/{expid:08d}/calib-{camera}-{expid:08d}.fits{compsuffix}',
        cframe = '{specprod_dir}/exposures/{night}/{expid:08d}/cframe-{camera}-{expid:08d}.fits{compsuffix}',
        fframe = '{specprod_dir}/exposures/{night}/{expid:08d}/fframe-{camera}-{expid:08d}.fits{compsuffix}',
        fluxcalib = '{specprod_dir}/exposures/{night}/{expid:08d}/fluxcalib-{camera}-{expid:08d}.fits{compsuffix}',
        frame = '{specprod_dir}/exposures/{night}/{expid:08d}/frame-{camera}-{expid:08d}.fits{compsuffix}',
        psf = '{specprod_dir}/exposures/{night}/{expid:08d}/psf-{camera}-{expid:08d}.fits',
        fitpsf='{specprod_dir}/exposures/{night}/{expid:08d}/fit-psf-{camera}-{expid:08d}.fits',
        qframe = '{specprod_dir}/exposures/{night}/{expid:08d}/qframe-{camera}-{expid:08d}.fits',
        sframe = '{specprod_dir}/exposures/{night}/{expid:08d}/sframe-{camera}-{expid:08d}.fits{compsuffix}',
        sky = '{specprod_dir}/exposures/{night}/{expid:08d}/sky-{camera}-{expid:08d}.fits{compsuffix}',
        skycorr = '{specprod_dir}/exposures/{night}/{expid:08d}/skycorr-{camera}-{expid:08d}.fits',
        fiberflat = '{specprod_dir}/exposures/{night}/{expid:08d}/fiberflat-{camera}-{expid:08d}.fits{compsuffix}',
        fiberflatexp = '{specprod_dir}/exposures/{night}/{expid:08d}/fiberflatexp-{camera}-{expid:08d}.fits{compsuffix}',
        stdstars = '{specprod_dir}/exposures/{night}/{expid:08d}/stdstars-{spectrograph:d}-{expid:08d}.fits{compsuffix}',
        calibstars = '{specprod_dir}/exposures/{night}/{expid:08d}/calibstars-{expid:08d}.csv',
        psfboot = '{specprod_dir}/exposures/{night}/{expid:08d}/psfboot-{camera}-{expid:08d}.fits',
        #
        #  qa
        #
        exposureqa = '{specprod_dir}/exposures/{night}/{expid:08d}/exposure-qa-{expid:08d}.fits',
        tileqa     = '{specprod_dir}/tiles/{groupname}/{tile:d}/{night}/tile-qa-{tile:d}-{nightprefix}{night}.fits',
        tileqapng  = '{specprod_dir}/tiles/{groupname}/{tile:d}/{night}/tile-qa-{tile:d}-{nightprefix}{night}.png',
        zmtl  = '{specprod_dir}/tiles/{groupname}/{tile:d}/{night}/zmtl-{spectrograph:d}-{tile:d}-{nightprefix}{night}.fits',
        #
        # calibnight/
        #
        fiberflatnight = '{specprod_dir}/calibnight/{night}/fiberflatnight-{camera}-{night}.fits',
        psfnight = '{specprod_dir}/calibnight/{night}/psfnight-{camera}-{night}.fits',
        biasnight = '{specprod_dir}/calibnight/{night}/biasnight-{camera}-{night}.fits{compsuffix}',
        badfibers =  '{specprod_dir}/calibnight/{night}/badfibers-{night}.csv',
        badcolumns = '{specprod_dir}/calibnight/{night}/badcolumns-{camera}-{night}.csv',
        ctecorrnight = '{specprod_dir}/calibnight/{night}/ctecorr-{night}.yaml',
        ctecorr      = '{specprod_dir}/calibnight/{night}/ctecorr-{night}.yaml', #- alias, same file
        #
        # spectra- healpix based
        #
        coadd_hp   = '{specprod_dir}/healpix/{survey}/{faprogram}/{hpixdir}/coadd-{survey}-{faprogram}-{healpix}.fits',
        rrdetails_hp = '{specprod_dir}/healpix/{survey}/{faprogram}/{hpixdir}/rrdetails-{survey}-{faprogram}-{healpix}.h5',
        rrmodel_hp = '{specprod_dir}/healpix/{survey}/{faprogram}/{hpixdir}/rrmodel-{survey}-{faprogram}-{healpix}.fits',
        spectra_hp = '{specprod_dir}/healpix/{survey}/{faprogram}/{hpixdir}/spectra-{survey}-{faprogram}-{healpix}.fits{compsuffix}',
        redrock_hp   = '{specprod_dir}/healpix/{survey}/{faprogram}/{hpixdir}/redrock-{survey}-{faprogram}-{healpix}.fits',
        qso_mgii_hp='{specprod_dir}/healpix/{survey}/{faprogram}/{hpixdir}/qso_mgii-{survey}-{faprogram}-{healpix}.fits',
        qso_qn_hp='{specprod_dir}/healpix/{survey}/{faprogram}/{hpixdir}/qso_qn-{survey}-{faprogram}-{healpix}.fits',
        emline_hp='{specprod_dir}/healpix/{survey}/{faprogram}/{hpixdir}/emline-{survey}-{faprogram}-{healpix}.fits',
        hpixexp='{specprod_dir}/healpix/{survey}/{faprogram}/{hpixdir}/hpixexp-{survey}-{faprogram}-{healpix}.csv',
        #
        # spectra- tile based
        #
        coadd_tile='{specprod_dir}/tiles/{groupname}/{tile:d}/{subgroup}/coadd-{spectrograph:d}-{tile:d}-{nightprefix}{subgroup}.fits',
        rrdetails_tile='{specprod_dir}/tiles/{groupname}/{tile:d}/{subgroup}/rrdetails-{spectrograph:d}-{tile:d}-{nightprefix}{subgroup}.h5',
        rrmodel_tile='{specprod_dir}/tiles/{groupname}/{tile:d}/{subgroup}/rrmodel-{spectrograph:d}-{tile:d}-{nightprefix}{subgroup}.fits',
        spectra_tile='{specprod_dir}/tiles/{groupname}/{tile:d}/{subgroup}/spectra-{spectrograph:d}-{tile:d}-{nightprefix}{subgroup}.fits{compsuffix}',
        redrock_tile='{specprod_dir}/tiles/{groupname}/{tile:d}/{subgroup}/redrock-{spectrograph:d}-{tile:d}-{nightprefix}{subgroup}.fits',
        qso_mgii_tile='{specprod_dir}/tiles/{groupname}/{tile:d}/{subgroup}/qso_mgii-{spectrograph:d}-{tile:d}-{nightprefix}{subgroup}.fits',
        qso_qn_tile='{specprod_dir}/tiles/{groupname}/{tile:d}/{subgroup}/qso_qn-{spectrograph:d}-{tile:d}-{nightprefix}{subgroup}.fits',
        emline_tile='{specprod_dir}/tiles/{groupname}/{tile:d}/{subgroup}/emline-{spectrograph:d}-{tile:d}-{nightprefix}{subgroup}.fits',
        #
        # spectra- single exp tile based requires cusom formatting for expid:08d
        #
        coadd_single='{specprod_dir}/tiles/perexp/{tile:d}/{expid:08d}/coadd-{spectrograph:d}-{tile:d}-exp{expid:08d}.fits',
        rrdetails_single='{specprod_dir}/tiles/perexp/{tile:d}/{expid:08d}/rrdetails-{spectrograph:d}-{tile:d}-exp{expid:08d}.h5',
        rrmodel_single='{specprod_dir}/tiles/perexp/{tile:d}/{expid:08d}/rrmodel-{spectrograph:d}-{tile:d}-exp{expid:08d}.fits',
        spectra_single='{specprod_dir}/tiles/perexp/{tile:d}/{expid:08d}/spectra-{spectrograph:d}-{tile:d}-exp{expid:08d}.fits{compsuffix}',
        redrock_single='{specprod_dir}/tiles/perexp/{tile:d}/{expid:08d}/redrock-{spectrograph:d}-{tile:d}-exp{expid:08d}.fits',
        qso_mgii_single='{specprod_dir}/tiles/perexp/{tile:d}/{expid:08d}/qso_mgii-{spectrograph:d}-{tile:d}-exp{expid:08d}.fits',
        qso_qn_single='{specprod_dir}/tiles/perexp/{tile:d}/{expid:08d}/qso_qn-{spectrograph:d}-{tile:d}-exp{expid:08d}.fits',
        emline_single='{specprod_dir}/tiles/perexp/{tile:d}/{expid:08d}/emline-{spectrograph:d}-{tile:d}-exp{expid:08d}.fits',
        tileqa_single  = '{specprod_dir}/tiles/perexp/{tile:d}/{expid:08d}/tile-qa-{tile:d}-exp{expid:08d}.fits',
        tileqapng_single = '{specprod_dir}/tiles/perexp/{tile:d}/{expid:08d}/tile-qa-{tile:d}-exp{expid:08d}.png',
        #
        # z catalogs
        #
        zcatalog='{specprod_dir}/zcatalog-{specprod}.fits',  # deprecated
        zcat_hp = '{specprod_dir}/zcatalog/{version}/zpix-{survey}-{faprogram}.fits',
        zcat_tile = '{specprod_dir}/zcatalog/{version}/ztile-{survey}-{faprogram}-{groupname}.fits',
        zall_hp = '{specprod_dir}/zcatalog/{version}/zall-pix-{specprod}.fits',
        zall_tile='{specprod_dir}/zcatalog/{version}/zall-tile{groupname}-{specprod}.fits',
        #
        # Dashboard files
        #
        expinfo = '{specprod_dir}/run/dashboard/expjsons/expinfo_{specprod}_{night}.json',
        zinfo = '{specprod_dir}/run/dashboard/zjsons/zinfo_{specprod}_{night}.json',
    )
    ## aliases
    location['desi'] = location['raw']
    location['exptable'] = location['exposure_table']
    location['proctable'] = location['processing_table']
    location['unproctable'] = location['unprocessed_table']

    ## Define the month if night is specified
    if night is not None:
        month = str(night)[:-2]
    else:
        month = None

    #- default group is "cumulative" for tile-based files
    if groupname is None and tile is not None and filetype in (
            'spectra', 'coadd', 'redrock', 'rrdetails', 'rrmodel', 'tileqa', 'tileqapng', 'zmtl',
            'spectra_tile', 'coadd_tile', 'redrock_tile', 'rrdetails_tile', 'rrmodel_tile',
            'zcat_tile', 'zall_tile'
            ):
        groupname = 'cumulative'

    if groupname == "cumulative":
        nightprefix = "thru"
        subgroup = str(night)
    elif groupname == "pernight":
        nightprefix = ""
        subgroup = str(night)
    elif groupname == "perexp":
        nightprefix = "exp"
    elif groupname == "healpix":
        nightprefix = ""
    else:
        nightprefix = str(groupname)+'-'

    #- backwards compatibility: try interpreting groupname as a healpix number
    if healpix is None and tile is None and groupname is not None:
        try:
            healpix = int(groupname)
            groupname = 'healpix'
        except (TypeError, ValueError):
            pass

    #- tile or healpix but not both
    if tile is not None and healpix is not None:
        raise ValueError(f'Set healpix or tile but not both ({healpix}, {tile})')

    #- Setting healpix implies groupname='healpix'
    if healpix is not None and groupname is None:
        groupname = 'healpix'

    #- be robust to str vs. int
    if isinstance(healpix, str): healpix = int(healpix)
    if isinstance(night, str):   night = int(night)
    if isinstance(expid, str):   expid = int(expid)
    if isinstance(tile, str):    tile = int(tile)
    if isinstance(spectrograph, str): spectrograph = int(spectrograph)

    #- Determine if this is healpix-based or tile-based objects, and update
    #- location dict for which flavor of coadd/spectra/redrock/etc is needed,
    #- removing the _hp, _single, _tile suffixes from the keys
    loc_copy = location.copy()
    if tile is not None:
        log.debug("Tile-based files selected; healpix-based files and input will be ignored.")
        if groupname == 'perexp':
            ## If perexp, then use the _single naming scheme
            ## Do loop to improve scaling with additional file types
            for key, val in loc_copy.items():
                if key.endswith('_single'):
                    root_key = key.removesuffix('_single')
                    location[root_key] = val
        else:
            ## If cumulative/pernight, then use the tile naming scheme
            ## Do loop to improve scaling with additional file types
            for key, val in loc_copy.items():
                if key.endswith('_tile'):
                    root_key = key.removesuffix('_tile')
                    location[root_key] = val

        ## cumulative and pernight use night as subgroup
        if groupname in ('cumulative', 'pernight'):
            subgroup = night

    else:
        ## If not tile based then use the hp naming scheme
        ## Do loop to improve scaling with additional file types
        for key, val in loc_copy.items():
            if key.endswith('_hp'):
                root_key = key.removesuffix('_hp')
                location[root_key] = val
    del loc_copy

    if groupname is not None and tile is None and healpix is not None:
        hpixdir = healpix_subdirectory(nside, healpix)
    else:
        #- set to anything so later logic will trip on groupname not hpixdir
        hpixdir = 'hpix'
    log.debug("hpixdir = '%s'", hpixdir)

    #- Do we know about this kind of file?
    if filetype not in location:
        raise ValueError("Unknown filetype {}; known types are {}".format(filetype, list(location.keys())))

    #- Check for missing inputs, deduplicate via frozenset()
    required_inputs = frozenset([i[0] for i in re.findall(r'\{([a-z_]+)(|[:0-9d]+)\}', location[filetype])])

    if rawdata_dir is None and 'rawdata_dir' in required_inputs:
        rawdata_dir = rawdata_root()
        log.debug("rawdata_dir = '%s'", rawdata_dir)

    if specprod_dir is None and 'specprod_dir' in required_inputs and outdir is None :
        specprod_dir = specprod_root(specprod)
        log.debug("specprod_dir = '%s', specprod = '%s'", specprod_dir, specprod)
    elif outdir is not None :
        # if outdir is set, we will replace specprod_dir anyway
        # but we may need the variable to be set in the meantime
        specprod_dir = "dummy"

    if tiles_dir is None and 'tiles_dir' in required_inputs:
        tiles_dir = os.environ['FIBER_ASSIGN_DIR']

    if 'specprod' in required_inputs and specprod is None and outdir is None :
        specprod = os.path.basename(specprod_dir)
        log.debug("Setting specprod = '%s'", specprod)

    if camera is not None:
        camera = camera.lower()

        #- Check camera b0, r1, .. z9
        if spectrograph is not None and len(camera) == 1 \
           and camera in ['b', 'r', 'z']:
            raise ValueError('Specify camera=b0,r1..z9, not camera=b/r/z + spectrograph')

        if camera != '*' and re.match(r'[brz\*\?][0-9\*\?]', camera) is None:
            raise ValueError('Camera {} should be b0,r1..z9, or with ?* wildcards'.format(camera))

    actual_inputs = {
        'specprod_dir':specprod_dir, 'specprod':specprod, 'tiles_dir':tiles_dir,
        'night':night, 'expid':expid, 'tile':tile, 'camera':camera,
        'groupname':groupname, 'subgroup':subgroup, 'version':version,
        'healpix':healpix, 'nside':nside, 'hpixdir':hpixdir, 'band':band,
        'spectrograph':spectrograph, 'nightprefix':nightprefix, 'month':month,
        'compsuffix':compsuffix
        }

    #- survey and faprogram should be lower, but don't trip on None
    actual_inputs['survey'] = None if survey is None else survey.lower()
    actual_inputs['faprogram'] = None if faprogram is None else faprogram.lower()
    actual_inputs['tile3'] = None if tile is None else tile // 1000

    if 'rawdata_dir' in required_inputs:
        actual_inputs['rawdata_dir'] = rawdata_dir

    #- If any inputs missing, print all missing inputs, then raise single ValueError
    missing_inputs = False
    for i in required_inputs:
        if actual_inputs[i] is None:
            log.error("Required input '{0}' is not set for type '{1}'!".format(i,filetype))
            missing_inputs = True

    if missing_inputs:
        msg = f"Missing inputs for {location[filetype]}"
        log.critical(msg)
        raise ValueError(msg)

    #- normpath to remove extraneous double slashes /a/b//c/d
    filepath = os.path.normpath(location[filetype].format(**actual_inputs))

    if outdir:
        filepath = os.path.join(outdir, os.path.basename(filepath))

    if logfile:
        logtypes = ['spectra', 'coadd', 'redrock', 'rrdetails', 'tileqa',
                    'tileqapng', 'zmtl', 'qso_qn', 'qso_mgii', 'emline']
        if not np.any([filetype.startswith(term) for term in logtypes]):
            msg = "Requested log pathname, but we don't produce logs " \
                  + f"specific to filetype={filetype}. " \
                  + f"Supported types are: {logtypes}."
            log.error(msg)
            raise NotImplementedError(msg)

        filepath = get_log_pathname(filepath)

    if download:
        from .download import download
        log.debug("download('%s', single_thread=True)", filepath)
        filepath = download(filepath, single_thread=True)[0]

    if readonly:
        filepath = get_readonly_filepath(filepath)

    try:
        filepath = checkgzip(filepath,readonly)
        exists = True
    except FileNotFoundError:
        exists = False

    if return_exists:
        return filepath, exists
    else:
        return filepath

def get_raw_files(filetype, night, expid, rawdata_dir=None):
    """Get files for a specified exposure.

    Uses :func:`findfile` to determine the valid file names for the specified
    type.  Any camera identifiers not matching the regular expression
    [brz][0-9] will be silently ignored.

    Args:
        filetype(str): Type of files to get. Valid choices are 'raw', 'preproc',
            'fibermap'.
        night(str): Date string for the requested night in the format
            YYYYMMDD.
        expid(int): Exposure number to get files for.
        rawdata_dir(str): [optional] overrides $DESI_SPECTRO_DATA

    Returns:
        dict: Dictionary of found file names using camera id strings as keys,
            which are guaranteed to match the regular expression [brz][0-9].
    """
    glob_pattern = findfile(filetype, night, expid, camera='*', rawdata_dir=rawdata_dir)
    literals = [re.escape(tmp) for tmp in glob_pattern.split('*')]
    re_pattern = re.compile('([brz][0-9])'.join(literals))
    listing = glob.glob(glob_pattern)
    if len(listing) == 1:
        return listing[0]
    files = {}
    for entry in listing:
        found = re_pattern.match(entry)
        files[found.group(1)] = entry
    return files


def get_files(filetype, night, expid, specprod_dir=None, **kwargs):
    """Get files for a specified exposure.

    Uses :func:`findfile` to determine the valid file names for the specified
    type.  Any camera identifiers not matching the regular expression
    [brz][0-9] will be silently ignored.

    Args:
        filetype(str): Type of files to get. Valid choices are 'frame',
            'cframe', 'psf', etc.
        night(str): Date string for the requested night in the format YYYYMMDD.
        expid(int): Exposure number to get files for.
        specprod_dir(str): Path containing the exposures/ directory to use. If
            the value is None, then the value of :func:`specprod_root` is used
            instead. Ignored when raw is True.

    Returns:
        dict: Dictionary of found file names using camera id strings as keys,
            which are guaranteed to match the regular expression [brz][0-9].
    """
    glob_pattern = findfile(filetype, night, expid, camera='*', specprod_dir=specprod_dir)
    literals = [re.escape(tmp) for tmp in glob_pattern.split('*')]
    re_pattern = re.compile('([brz][0-9])'.join(literals))
    files = { }
    for entry in glob.glob(glob_pattern):
        found = re_pattern.match(entry)
        files[found.group(1)] = entry
    return files


def validate_night(night):
    """Validates a night string and converts to a date.

    Args:
        night(str): Date string for the requested night in the format YYYYMMDD.

    Returns:
        datetime.date: Date object representing this night.

    Raises:
        ValueError: Badly formatted night string.
    """
    try:
        return datetime.datetime.strptime(night,'%Y%m%d').date()
    except ValueError:
        raise ValueError('Badly formatted night %s' % night)


def find_exposure_night(expid, specprod_dir=None):
    """ Find the night that has the exposure
    Args:
        expid: int
        specprod_dir: str, optional

    Returns:
        night: str

    """
    # Search for the exposure folder
    nights = get_nights(specprod_dir=specprod_dir)
    for night in nights:
        for exposure in get_exposures(night, specprod_dir=specprod_dir):
            if exposure == expid:
                return night


def get_exposures(night, raw=False, rawdata_dir=None, specprod_dir=None):
    """Get a list of available exposures for the specified night.

    Exposures are identified as correctly formatted subdirectory names within the
    night directory, but no checks for valid contents of these exposure subdirectories
    are performed.

    Args:
        night(str): Date string for the requested night in the format YYYYMMDD.
        raw(bool): Returns raw exposures if set, otherwise returns processed exposures.
        rawdata_dir(str): [optional] overrides $DESI_SPECTRO_DATA
        specprod_dir(str): Path containing the exposures/ directory to use. If the value
            is None, then the value of :func:`specprod_root` is used instead. Ignored
            when raw is True.

    Returns:
        list: List of integer exposure numbers available for the specified night. The
            list will be empty if no the night directory exists but does not contain
            any exposures.

    Raises:
        ValueError: Badly formatted night date string
        IOError: non-existent night.
    """
    date = validate_night(night)

    if raw:
        if rawdata_dir is None:
            rawdata_dir = rawdata_root()
        night_path = os.path.join(rawdata_dir, night)
    else:
        if specprod_dir is None:
            specprod_dir = specprod_root()
        night_path = os.path.join(specprod_dir, 'exposures', night)

    if not os.path.exists(night_path):
        raise IOError('Non-existent night {0}'.format(night))

    exposures = []

    for entry in glob.glob(os.path.join(night_path, '*')):
        e = os.path.basename(entry)
        try:
            exposure = int(e)
            assert e == "{0:08d}".format(exposure)
            exposures.append(exposure)
        except (ValueError, AssertionError):
            # Silently ignore entries that are not exposure subdirectories.
            pass

    return sorted(exposures)


def get_reduced_frames(channels=['b','r','z'], nights=None, ftype='cframe', **kwargs):
    """ Loops through a production to find all reduced frames (default is cframes)
    One can choose a subset of reduced frames by argument
    Args:
        channels: list, optional
        nights: list, optional
        ftype: str, optional
        kwargs: passed to get_files()

    Returns:
        all_frames: list for frame filenames

    """
    all_frames = []
    # Nights
    if nights is None:
        nights = get_nights()
    # Loop on night
    for night in nights:
        exposures = get_exposures(night)
        for exposure in exposures:
            frames_dict = get_files(filetype=ftype, night=night, expid=exposure, **kwargs)
            # Restrict on channel
            for key in frames_dict.keys():
                for channel in channels:
                    if channel in key:
                        all_frames.append(frames_dict[key])
    # Return
    return all_frames

def get_nights_up_to_date(date, specprod_dir=None):
    """
    Returns nights with an existing exposure_table in the current specprod

    Args:
        date (int): return all dates up to and including this date
        specprod_dir (str): the specprod directory where the exposure_tables reside

    Returns:
        nights (list of ints): list of nights up to and including date that
                               have an exposure table
    """
    if specprod_dir is None:
        specprod_dir = specprod_root()

    nights = []
    exptabdir = os.path.join(specprod_dir, 'exposure_tables')
    for yearmonth in os.listdir(exptabdir):
        if int(yearmonth) > date // 100:
            continue
        exptabmonthdir = os.path.join(exptabdir, yearmonth)
        for exptabname in os.listdir(exptabmonthdir):
            try:
                night = int(re.findall(r'[0-9]{8}', exptabname)[0])
                if night <= date:
                    nights.append(night)
            except IndexError:
                continue

def get_nights(strip_path=True, specprod_dir=None, sub_folder='exposures'):
    """ Generate a list of nights in a given folder (default is exposures/)
    Demands an 8 digit name beginning with 20

    Args:
        strip_path:  bool, optional; Strip the path to the nights folders
        rawdata_dir:
        specprod_dir:
        sub_root: str, optional;  'exposures', 'calib2d'

    Returns:
        nights: list of nights (without or with paths)
    """
    # Init
    if specprod_dir is None:
        specprod_dir = specprod_root()
    # Glob for nights
    sub_path = os.path.join(specprod_dir, sub_folder)
    nights_with_path = glob.glob(sub_path+'/*')
    # Strip off path
    stripped = [os.path.basename(inight_path) for inight_path in nights_with_path]
    # Vet and generate
    nights = []
    for ii,istrip in enumerate(stripped):
        if (istrip[0:2] == '20') and len(istrip) == 8:
            if strip_path:
                nights.append(istrip)
            else:
                nights.append(nights_with_path[ii])
    # Return
    return sorted(nights)

def shorten_filename(filename):
    """Attempt to shorten filename to fit in FITS header without CONTINUE

    Args:
        filename (str): input filename

    Returns potentially shortened filename

    Replaces prefixes from environment variables:
      * $DESI_SPECTRO_CALIB -> SPCALIB
      * $DESI_SPECTRO_REDUX/$SPECPROD -> SPECPROD
    """

    if filename is None : return "None"

    spcalib = os.getenv('DESI_SPECTRO_CALIB')
    if spcalib is not None and filename.startswith(spcalib):
        return filename.replace(spcalib, 'SPCALIB', 1)

    try:
        specprod = specprod_root()
        specprod_ro = specprod_root(readonly=True)
    except KeyError:
        specprod = None
        specprod_ro = None

    if specprod is not None and filename.startswith(specprod):
        return filename.replace(specprod, 'SPECPROD', 1)

    if specprod_ro is not None and filename.startswith(specprod_ro):
        return filename.replace(specprod_ro, 'SPECPROD', 1)

    #- no substitutions
    return filename


def rawdata_root():
    """Returns directory root for raw data, i.e. ``$DESI_SPECTRO_DATA``

    Raises:
        KeyError: if these environment variables aren't set.
    """
    if 'DESI_SPECTRO_DATA' in os.environ:
        return os.environ['DESI_SPECTRO_DATA']
    else:
        return os.path.join(os.environ['DESI_ROOT'], 'spectro', 'data')


def specprod_root(specprod=None, readonly=False):
    """Return directory root for spectro production, i.e.
    ``$DESI_SPECTRO_REDUX/$SPECPROD``.

    Options:
        specprod (str): production name or full path to prodution
        readonly (bool): replace $DESI_ROOT -> $DESI_ROOT_READONLY if defined

    Raises:
        KeyError: if these environment variables aren't set.

    If specprod contains '/', treat as path and return that.
    If specprod is None, return $DESI_SPECTRO_REDUX/$SPECPROD.
    Otherwise, treat specprod as production name to override $SPECPROD
    and return $DESI_SPECTRO_REDUX/$SPECPROD

    If readonly=True, also replace $DESI_ROOT -> $DESI_ROOT_READONLY
    if those variables are defined
    """
    if specprod is None:
        specprod = os.environ['SPECPROD']

    if '/' not in specprod:
        if 'DESI_SPECTRO_REDUX' in os.environ:
            specprod = os.path.join(os.environ['DESI_SPECTRO_REDUX'], specprod)
        else:
            specprod = os.path.join(os.environ['DESI_ROOT'], 'spectro', 'redux', specprod)

    if readonly:
        specprod = get_readonly_filepath(specprod)

    return specprod

def faflavor2program(faflavor):
    """
    Map FAFLAVOR keywords to what we wish we had set for FAPRGRM

    Args:
        faflavor (str or array of str): FAFLAVOR keywords from fiberassign

    Returns:
        faprgm (str or array of str): what FAPRGM would be if we had set it
        (dark, bright, backup, other)

    Note: this was standardized by sv3 and main, but evolved during sv1 and sv2.
        for the survey=special tiles (m31, odin, and tertiary), the info
        is/can be retrieve from the GOALTYPE keyword in the zero-th extension
        of the fiberassign file.
    """
    #- Handle scalar or array input, upcasting bytes to str as needed
    scalar_input = np.isscalar(faflavor)
    faflavor = np.atleast_1d(faflavor).astype(str)

    #- Default FAPRGRM is "other"
    faprogram = np.tile('other', len(faflavor)).astype('U6')

    #- FAFLAVOR options that map to FAPRGM='dark'
    #- Note: some sv1 tiles like 80605 had "cmx" in the faflavor name
    dark  = faflavor == 'cmxelg'
    dark |= faflavor == 'cmxlrgqso'
    dark |= faflavor == 'sv1elg'
    dark |= faflavor == 'sv1elgqso'
    dark |= faflavor == 'sv1lrgqso'
    dark |= faflavor == 'sv1lrgqso2'
    dark |= np.isin(
        faflavor,
        np.char.add(
            "special",
            [
                'm31', 'odin', 'tertiary1', 'tertiary2', 'tertiary4', 'tertiary5',
                'tertiary7', 'tertiary9', 'tertiary11', 'tertiary14', 'tertiary15',
                'tertiary16', 'tertiary17', 'tertiary18', 'tertiary21', 'tertiary23',
                'tertiary25', 'tertiary26', 'tertiary27', 'tertiary31', 'tertiary35',
                'tertiary37', 'tertiary38', 'tertiary40', 'tertiary41',
            ]
        )
    )
    dark |= np.char.endswith(faflavor, 'dark')

    #- SV1 FAFLAVOR options that map to FAPRGRM='bright'
    bright  = faflavor == 'sv1bgsmws'
    bright |= (faflavor != 'sv1unwisebluebright') & np.char.endswith(faflavor, 'bright')
    bright |= np.isin(
        faflavor,
        np.char.add(
            "special",
            [
                'tertiary3', 'tertiary6', 'tertiary8', 'tertiary10', 'tertiary12',
                'tertiary13', 'tertiary19', 'tertiary20', 'tertiary22', 'tertiary24',
                'tertiary28', 'tertiary29', 'tertiary30', 'tertiary32', 'tertiary33',
                'tertiary34', 'tertiary36', 'tertiary39',
            ]
        )
    )

    #- SV1 FAFLAVOR options that map to FAPRGRM='backup'
    backup  = faflavor == 'sv1backup1'
    backup |= np.char.endswith(faflavor, 'backup')

    faprogram[dark] = 'dark'
    faprogram[bright] = 'bright'
    faprogram[backup] = 'backup'

    if scalar_input:
        return str(faprogram[0])
    else:
        return faprogram


def get_pipe_database():
    """Get the production database location based on the environment.

    """
    if "DESI_SPECTRO_DB" in os.environ:
        # Use an alternate location for the DB
        dbpath = os.environ["DESI_SPECTRO_DB"]
    else:
        proddir = specprod_root()
        dbpath = os.path.join(proddir, "desi.db")
        os.environ["DESI_SPECTRO_DB"] = dbpath
    return dbpath


def get_pipe_rundir(specprod_dir=None):
    """
    Return the directory path for pipeline runtime files.

    Args:
        specprod_dir (str): Optional path to production directory.  If None,
            the this is obtained from :func:`specprod_root`.

    Returns (str):
        the directory path for pipeline runtime files.
    """
    if specprod_dir is None:
        specprod_dir = specprod_root()
    return os.path.join(os.path.abspath(specprod_dir), "run")


def get_pipe_scriptdir():
    """
    Return the name of the subdirectory containing pipeline scripts.

    Returns (str):
        The name of the subdirectory.
    """
    return "scripts"


def get_pipe_logdir():
    """
    Return the name of the subdirectory containing pipeline logs.

    Returns (str):
        The name of the subdirectory.
    """
    return "logs"


def get_pipe_nightdir():
    """
    Return the name of the subdirectory containing per-night files.

    Returns (str):
        The name of the subdirectory.
    """
    return "night"


def get_pipe_pixeldir():
    """
    Return the name of the subdirectory containing per-pixel files.

    Returns (str):
        The name of the subdirectory.
    """
    return "healpix"
