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

from .util import healpix_subdirectory


def findfile(filetype, night=None, expid=None, camera=None, groupname=None,
    nside=64, band=None, spectrograph=None, rawdata_dir=None, specprod_dir=None,
    download=False, outdir=None, qaprod_dir=None):
    """Returns location where file should be

    Args:
        filetype : file type, typically the prefix, e.g. "frame" or "psf"

    Args depending upon filetype:
        night : YEARMMDD string
        expid : integer exposure id
        camera : 'b0' 'r1' .. 'z9'
        groupname : spectral grouping name (brick name or healpix pixel)
        nside : healpix nside
        band : one of 'b','r','z' identifying the camera band
        spectrograph : spectrograph number, 0-9

    Options:
        rawdata_dir : overrides $DESI_SPECTRO_DATA
        specprod_dir : overrides $DESI_SPECTRO_REDUX/$SPECPROD/
        qaprod_dir : defaults to $DESI_SPECTRO_REDUX/$SPECPROD/QA/ if not provided
        download : if not found locally, try to fetch remotely
        outdir : use this directory for output instead of canonical location
    """

    #- NOTE: specprod_dir is the directory $DESI_SPECTRO_REDUX/$SPECPROD,
    #-       specprod is just the environment variable $SPECPROD
    location = dict(
        raw = '{rawdata_dir}/{night}/desi-{expid:08d}.fits.fz',
        preproc = '{specprod_dir}/preproc/{night}/{expid:08d}/preproc-{camera}-{expid:08d}.fits',
        fiberflat = '{specprod_dir}/exposures/{night}/{expid:08d}/fiberflat-{camera}-{expid:08d}.fits',
        fiberflatnight = '{specprod_dir}/calibnight/{night}/fiberflatnight-{camera}-{night}.fits',
        frame = '{specprod_dir}/exposures/{night}/{expid:08d}/frame-{camera}-{expid:08d}.fits',
        cframe = '{specprod_dir}/exposures/{night}/{expid:08d}/cframe-{camera}-{expid:08d}.fits',
        fframe = '{specprod_dir}/exposures/{night}/{expid:08d}/fframe-{camera}-{expid:08d}.fits',
        sframe = '{specprod_dir}/exposures/{night}/{expid:08d}/sframe-{camera}-{expid:08d}.fits',
        sky = '{specprod_dir}/exposures/{night}/{expid:08d}/sky-{camera}-{expid:08d}.fits',
        stdstars = '{specprod_dir}/exposures/{night}/{expid:08d}/stdstars-{spectrograph:d}-{expid:08d}.fits',
        calib = '{specprod_dir}/exposures/{night}/{expid:08d}/calib-{camera}-{expid:08d}.fits',
        qa_data = '{qaprod_dir}/exposures/{night}/{expid:08d}/qa-{camera}-{expid:08d}.yaml',
        qa_data_exp = '{qaprod_dir}/exposures/{night}/{expid:08d}/qa-{expid:08d}.yaml',
        qa_bootcalib = '{qaprod_dir}/calib2d/psf/{night}/qa-psfboot-{camera}.pdf',
        qa_sky_fig = '{qaprod_dir}/exposures/{night}/{expid:08d}/qa-sky-{camera}-{expid:08d}.png',
        qa_skychi_fig = '{qaprod_dir}/exposures/{night}/{expid:08d}/qa-skychi-{camera}-{expid:08d}.png',
        qa_flux_fig = '{qaprod_dir}/exposures/{night}/{expid:08d}/qa-flux-{camera}-{expid:08d}.png',
        qa_toplevel_html = '{qaprod_dir}/qa-toplevel.html',
        qa_calib = '{qaprod_dir}/calib2d/{night}/qa-{camera}-{expid:08d}.yaml',
        qa_calib_html = '{qaprod_dir}/calib2d/qa-calib2d.html',
        qa_calib_exp = '{qaprod_dir}/calib2d/{night}/qa-{expid:08d}.yaml',
        qa_calib_exp_html = '{qaprod_dir}/calib2d/{night}/qa-{expid:08d}.html',
        qa_exposures_html = '{qaprod_dir}/exposures/qa-exposures.html',
        qa_exposure_html = '{qaprod_dir}/exposures/{night}/{expid:08d}/qa-{expid:08d}.html',
        qa_flat_fig = '{qaprod_dir}/calib2d/{night}/qa-flat-{camera}-{expid:08d}.png',
        qa_ztruth = '{qaprod_dir}/exposures/{night}/qa-ztruth-{night}.yaml',
        qa_ztruth_fig = '{qaprod_dir}/exposures/{night}/qa-ztruth-{night}.png',
        ql_bootcalib_fig = '{specprod_dir}/exposures/{night}/{expid:08d}/ql-bootcalib-{camera}-{expid:08d}.png',
        ql_bootcalib_file = '{specprod_dir}/exposures/{night}/{expid:08d}/ql-bootcalib-{camera}-{expid:08d}.yaml',
        ql_boxextract_fig = '{specprod_dir}/exposures/{night}/{expid:08d}/ql-boxextract-{camera}-{expid:08d}.png',
        ql_boxextract_file = '{specprod_dir}/exposures/{night}/{expid:08d}/ql-boxextract-{camera}-{expid:08d}.yaml',
        ql_boxextract_arc_fig = '{specprod_dir}/calib2d/{night}/ql-boxextractarc-{camera}-{expid:08d}.png',
        ql_boxextract_arc_file = '{specprod_dir}/calib2d/{night}/ql-boxextractarc-{camera}-{expid:08d}.yaml',
        ql_computeflat_fig = '{specprod_dir}/calib2d/{night}/ql-computeflat-{camera}-{expid:08d}.png',
        ql_computeflat_file = '{specprod_dir}/calib2d/{night}/ql-computeflat-{camera}-{expid:08d}.yaml',
        ql_countbins_fig = '{specprod_dir}/exposures/{night}/{expid:08d}/ql-countbins-{camera}-{expid:08d}.png',
        ql_countbins_file = '{specprod_dir}/exposures/{night}/{expid:08d}/ql-countbins-{camera}-{expid:08d}.yaml',
        ql_countbins_arc_fig = '{specprod_dir}/calib2d/{night}/ql-countbinsarc-{camera}-{expid:08d}.png',
        ql_countbins_arc_file = '{specprod_dir}/calib2d/{night}/ql-countbinsarc-{camera}-{expid:08d}.yaml',
        ql_countpix_fig = '{specprod_dir}/exposures/{night}/{expid:08d}/ql-countpix-{camera}-{expid:08d}.png',
        ql_countpix_file = '{specprod_dir}/exposures/{night}/{expid:08d}/ql-countpix-{camera}-{expid:08d}.yaml',
        ql_countpix_arc_fig = '{specprod_dir}/calib2d/{night}/ql-countpixarc-{camera}-{expid:08d}.png',
        ql_countpix_arc_file = '{specprod_dir}/calib2d/{night}/ql-countpixarc-{camera}-{expid:08d}.yaml',
        ql_countpix_bias_fig = '{specprod_dir}/calib2d/{night}/ql-countpixbias-{camera}-{expid:08d}.png',
        ql_countpix_bias_file = '{specprod_dir}/calib2d/{night}/ql-countpixbias-{camera}-{expid:08d}.yaml',
        ql_countpix_dark_fig = '{specprod_dir}/calib2d/{night}/ql-countpixdark-{camera}-{expid:08d}.png',
        ql_countpix_dark_file = '{specprod_dir}/calib2d/{night}/ql-countpixdark-{camera}-{expid:08d}.yaml',
        ql_fiberflat_fig = '{specprod_dir}/exposures/{night}/{expid:08d}/ql-fiberflat-{camera}-{expid:08d}.png',
        ql_fiberflat_file = '{specprod_dir}/exposures/{night}/{expid:08d}/ql-fiberflat-{camera}-{expid:08d}.yaml',
        ql_getbias_fig = '{specprod_dir}/exposures/{night}/{expid:08d}/ql-getbias-{camera}-{expid:08d}.png',
        ql_getbias_file = '{specprod_dir}/exposures/{night}/{expid:08d}/ql-getbias-{camera}-{expid:08d}.yaml',
        ql_getbias_arc_fig = '{specprod_dir}/calib2d/{night}/ql-getbiasarc-{camera}-{expid:08d}.png',
        ql_getbias_arc_file = '{specprod_dir}/calib2d/{night}/ql-getbiasarc-{camera}-{expid:08d}.yaml',
        ql_getbias_bias_fig = '{specprod_dir}/calib2d/{night}/ql-getbiasbias-{camera}-{expid:08d}.png',
        ql_getbias_bias_file = '{specprod_dir}/calib2d/{night}/ql-getbiasbias-{camera}-{expid:08d}.yaml',
        ql_getbias_dark_fig = '{specprod_dir}/calib2d/{night}/ql-getbiasdark-{camera}-{expid:08d}.png',
        ql_getbias_dark_file = '{specprod_dir}/calib2d/{night}/ql-getbiasdark-{camera}-{expid:08d}.yaml',
        ql_getrms_fig = '{specprod_dir}/exposures/{night}/{expid:08d}/ql-getrms-{camera}-{expid:08d}.png',
        ql_getrms_file = '{specprod_dir}/exposures/{night}/{expid:08d}/ql-getrms-{camera}-{expid:08d}.yaml',
        ql_getrms_arc_fig = '{specprod_dir}/calib2d/{night}/ql-getrmsarc-{camera}-{expid:08d}.png',
        ql_getrms_arc_file = '{specprod_dir}/calib2d/{night}/ql-getrmsarc-{camera}-{expid:08d}.yaml',
        ql_getrms_bias_fig = '{specprod_dir}/calib2d/{night}/ql-getrmsbias-{camera}-{expid:08d}.png',
        ql_getrms_bias_file = '{specprod_dir}/calib2d/{night}/ql-getrmsbias-{camera}-{expid:08d}.yaml',
        ql_getrms_dark_fig = '{specprod_dir}/calib2d/{night}/ql-getrmsdark-{camera}-{expid:08d}.png',
        ql_getrms_dark_file = '{specprod_dir}/calib2d/{night}/ql-getrmsdark-{camera}-{expid:08d}.yaml',
        ql_initial_fig = '{specprod_dir}/exposures/{night}/{expid:08d}/ql-initial-{camera}-{expid:08d}.png',
        ql_initial_file = '{specprod_dir}/exposures/{night}/{expid:08d}/ql-initial-{camera}-{expid:08d}.yaml',
        ql_initial_arc_fig = '{specprod_dir}/calib2d/{night}/ql-initialarc-{camera}-{expid:08d}.png',
        ql_initial_arc_file = '{specprod_dir}/calib2d/{night}/ql-initialarc-{camera}-{expid:08d}.yaml',
        ql_initial_bias_fig = '{specprod_dir}/calib2d/{night}/ql-initialbias-{camera}-{expid:08d}.png',
        ql_initial_bias_file = '{specprod_dir}/calib2d/{night}/ql-initialbias-{camera}-{expid:08d}.yaml',
        ql_initial_dark_fig = '{specprod_dir}/calib2d/{night}/ql-initialdark-{camera}-{expid:08d}.png',
        ql_initial_dark_file = '{specprod_dir}/calib2d/{night}/ql-initialdark-{camera}-{expid:08d}.yaml',
        ql_integ_fig = '{specprod_dir}/exposures/{night}/{expid:08d}/ql-integ-{camera}-{expid:08d}.png',
        ql_integ_file = '{specprod_dir}/exposures/{night}/{expid:08d}/ql-integ-{camera}-{expid:08d}.yaml',
        ql_mergedQA_file = '{specprod_dir}/exposures/{night}/{expid:08d}/ql-mergedQA-{camera}-{expid:08d}.yaml',
        ql_mergedQAarc_file = '{specprod_dir}/calib2d/{night}/ql-mergedQAarc-{camera}-{expid:08d}.yaml',
        ql_preproc_fig = '{specprod_dir}/exposures/{night}/{expid:08d}/ql-preproc-{camera}-{expid:08d}.png',
        ql_preproc_file = '{specprod_dir}/exposures/{night}/{expid:08d}/ql-preproc-{camera}-{expid:08d}.yaml',
        ql_preproc_arc_fig = '{specprod_dir}/calib2d/{night}/ql-preprocarc-{camera}-{expid:08d}.png',
        ql_preproc_arc_file = '{specprod_dir}/calib2d/{night}/ql-preprocarc-{camera}-{expid:08d}.yaml',
        ql_preproc_bias_fig = '{specprod_dir}/calib2d/{night}/ql-preprocbias-{camera}-{expid:08d}.png',
        ql_preproc_bias_file = '{specprod_dir}/calib2d/{night}/ql-preprocbias-{camera}-{expid:08d}.yaml',
        ql_preproc_dark_fig = '{specprod_dir}/calib2d/{night}/ql-preprocdark-{camera}-{expid:08d}.png',
        ql_preproc_dark_file = '{specprod_dir}/calib2d/{night}/ql-preprocdark-{camera}-{expid:08d}.yaml',
        ql_resfit_fig = '{specprod_dir}/exposures/{night}/{expid:08d}/ql-resfit-{camera}-{expid:08d}.png',
        ql_resfit_file = '{specprod_dir}/exposures/{night}/{expid:08d}/ql-resfit-{camera}-{expid:08d}.yaml',
        ql_resfit_arc_fig = '{specprod_dir}/calib2d/{night}/ql-resfitarc-{camera}-{expid:08d}.png',
        ql_resfit_arc_file = '{specprod_dir}/calib2d/{night}/ql-resfitarc-{camera}-{expid:08d}.yaml',
        ql_skycont_fig = '{specprod_dir}/exposures/{night}/{expid:08d}/ql-skycont-{camera}-{expid:08d}.png',
        ql_skycont_file = '{specprod_dir}/exposures/{night}/{expid:08d}/ql-skycont-{camera}-{expid:08d}.yaml',
        ql_skypeak_fig = '{specprod_dir}/exposures/{night}/{expid:08d}/ql-skypeak-{camera}-{expid:08d}.png',
        ql_skypeak_file = '{specprod_dir}/exposures/{night}/{expid:08d}/ql-skypeak-{camera}-{expid:08d}.yaml',
        ql_skyresid_fig = '{specprod_dir}/exposures/{night}/{expid:08d}/ql-skyresid-{camera}-{expid:08d}.png',
        ql_skyresid_file = '{specprod_dir}/exposures/{night}/{expid:08d}/ql-skyresid-{camera}-{expid:08d}.yaml',
        ql_skysub_fig = '{specprod_dir}/exposures/{night}/{expid:08d}/ql-skysub-{camera}-{expid:08d}.png',
        ql_skysub_file = '{specprod_dir}/exposures/{night}/{expid:08d}/ql-skysub-{camera}-{expid:08d}.yaml',
        ql_snr_fig = '{specprod_dir}/exposures/{night}/{expid:08d}/ql-snr-{camera}-{expid:08d}.png',
        ql_snr_file = '{specprod_dir}/exposures/{night}/{expid:08d}/ql-snr-{camera}-{expid:08d}.yaml',
        ql_xwsigma_fig = '{specprod_dir}/exposures/{night}/{expid:08d}/ql-xwsigma-{camera}-{expid:08d}.png',
        ql_xwsigma_file = '{specprod_dir}/exposures/{night}/{expid:08d}/ql-xwsigma-{camera}-{expid:08d}.yaml',
        ql_xwsigma_arc_fig = '{specprod_dir}/calib2d/{night}/ql-xwsigmaarc-{camera}-{expid:08d}.png',
        ql_xwsigma_arc_file = '{specprod_dir}/calib2d/{night}/ql-xwsigmaarc-{camera}-{expid:08d}.yaml',
        psf = '{specprod_dir}/exposures/{night}/{expid:08d}/psf-{camera}-{expid:08d}.fits',
        psfnight = '{specprod_dir}/calibnight/{night}/psfnight-{camera}-{night}.fits',
        psfboot = '{specprod_dir}/exposures/{night}/{expid:08d}/psfboot-{camera}-{expid:08d}.fits',
        fibermap = '{rawdata_dir}/{night}/fibermap-{expid:08d}.fits',
        zcatalog = '{specprod_dir}/zcatalog-{specprod}.fits',
        spectra = '{specprod_dir}/spectra-{nside}/{hpixdir}/spectra-{nside}-{groupname}.fits',
        redrock = '{specprod_dir}/spectra-{nside}/{hpixdir}/redrock-{nside}-{groupname}.h5',
        coadd = '{specprod_dir}/spectra-{nside}/{hpixdir}/coadd-{nside}-{groupname}.fits',
        zbest = '{specprod_dir}/spectra-{nside}/{hpixdir}/zbest-{nside}-{groupname}.fits',
    )
    location['desi'] = location['raw']

    if groupname is not None:
        hpix = int(groupname)
        hpixdir = healpix_subdirectory(nside, hpix)
    else:
        #- set to anything so later logic will trip on groupname not hpixdir
        hpixdir = 'hpix'

    #- Do we know about this kind of file?
    if filetype not in location:
        raise IOError("Unknown filetype {}; known types are {}".format(filetype, list(location.keys())))

    #- Check for missing inputs
    required_inputs = [i[0] for i in re.findall(r'\{([a-z_]+)(|[:0-9d]+)\}',location[filetype])]

    if rawdata_dir is None and 'rawdata_dir' in required_inputs:
        rawdata_dir = rawdata_root()

    if specprod_dir is None and 'specprod_dir' in required_inputs:
        specprod_dir = specprod_root()

    if qaprod_dir is None and 'qaprod_dir' in required_inputs:
        qaprod_dir = qaprod_root()

    if 'specprod' in required_inputs:
        #- Replace / with _ in $SPECPROD so we can use it in a filename
        specprod = os.getenv('SPECPROD').replace('/', '_')
    else:
        specprod = None

    actual_inputs = {
        'specprod_dir':specprod_dir, 'specprod':specprod, 'qaprod_dir':qaprod_dir,
        'night':night, 'expid':expid, 'camera':camera, 'groupname':groupname,
        'nside':nside, 'hpixdir':hpixdir, 'band':band,
        'spectrograph':spectrograph,
        'hpixdir':hpixdir,
        }

    if 'rawdata_dir' in required_inputs:
        actual_inputs['rawdata_dir'] = rawdata_dir

    for i in required_inputs:
        if actual_inputs[i] is None:
            raise ValueError("Required input '{0}' is not set for type '{1}'!".format(i,filetype))

    #- normpath to remove extraneous double slashes /a/b//c/d
    filepath = os.path.normpath(location[filetype].format(**actual_inputs))

    if outdir:
        filepath = os.path.join(outdir, os.path.basename(filepath))

    if download:
        from .download import download
        filepath = download(filepath,single_thread=True)[0]

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
        RuntimeError: Badly formatted night string.
    """
    try:
        return datetime.datetime.strptime(night,'%Y%m%d').date()
    except ValueError:
        raise RuntimeError('Badly formatted night %s' % night)


def find_exposure_night(expid):
    """ Find the night that has the exposure
    Args:
        expid: int

    Returns:
        night: str

    """
    # Search for the exposure folder
    nights = get_nights()
    for night in nights:
        for exposure in get_exposures(night):
            if exposure == expid:
                return night


def get_exposures(night, raw=False, rawdata_dir=None, specprod_dir=None, ):
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
        RuntimeError: Badly formatted night date string or non-existent night.
    """
    date = validate_night(night)

    if raw:
        if rawdata_dir is None:
            rawdata_dir = rawdata_root()
        night_path = os.path.join(rawdata_dir, night)
    else:
        if specprod_dir is None:
            specprod_dir = specprod_root()
        night_path = os.path.join(specprod_dir,'exposures',night)

    if not os.path.exists(night_path):
        raise RuntimeError('Non-existent night %s' % night)

    exposures = []

    if raw:
        fpat = re.compile(r'.*fibermap-(.*).fits')
        for entry in glob.glob(os.path.join(night_path,'fibermap-*.fits')):
            mat = fpat.match(entry)
            if mat is not None:
                iexp = int(mat.group(1))
                assert mat.group(1) == "{:08d}".format(iexp)
                exposures.append(iexp)
    else:
        for entry in glob.glob(os.path.join(night_path,'*')):
            head,tail = os.path.split(entry)
            try:
                exposure = int(tail)
                assert tail == "{:08d}".format(exposure)
                exposures.append(exposure)
            except (ValueError,AssertionError):
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



def rawdata_root():
    """Returns directory root for raw data, i.e. ``$DESI_SPECTRO_DATA``

    Raises:
        AssertionError: if these environment variables aren't set.
    """
    assert 'DESI_SPECTRO_DATA' in os.environ, 'Missing $DESI_SPECTRO_DATA environment variable'
    return os.environ['DESI_SPECTRO_DATA']


def specprod_root():
    """Return directory root for spectro production, i.e.
    ``$DESI_SPECTRO_REDUX/$SPECPROD``.

    Raises:
        AssertionError: if these environment variables aren't set.
    """
    assert 'SPECPROD' in os.environ, 'Missing $SPECPROD environment variable'
    assert 'DESI_SPECTRO_REDUX' in os.environ, 'Missing $DESI_SPECTRO_REDUX environment variable'
    return os.path.join(os.getenv('DESI_SPECTRO_REDUX'), os.getenv('SPECPROD'))


def qaprod_root():
    """Return directory root for spectro production QA, i.e.
    ``$DESI_SPECTRO_REDUX/$SPECPROD/QA``.

    Raises:
        AssertionError: if these environment variables aren't set.
    """
    assert 'SPECPROD' in os.environ, 'Missing $SPECPROD environment variable'
    assert 'DESI_SPECTRO_REDUX' in os.environ, 'Missing $DESI_SPECTRO_REDUX environment variable'
    return os.path.join(os.getenv('DESI_SPECTRO_REDUX'), os.getenv('SPECPROD'), 'QA')


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
