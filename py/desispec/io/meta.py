# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
desispec.io.meta
================

IO metadata functions.
"""

import os
import datetime
import glob
import re


def findfile(filetype, night=None, expid=None, camera=None, brickname=None,
    band=None, spectrograph=None, rawdata_dir=None, specprod_dir=None,
    download=False, outdir=None):
    """Returns location where file should be

    Args:
        filetype : file type, typically the prefix, e.g. "frame" or "psf"

    Args depending upon filetype:
        night : YEARMMDD string
        expid : integer exposure id
        camera : 'b0' 'r1' .. 'z9'
        brickname : brick name string
        band : one of 'b','r','z' identifying the camera band
        spectrograph : spectrograph number, 0-9

    Options:
        rawdata_dir : overrides $DESI_SPECTRO_DATA
        specprod_dir : overrides $DESI_SPECTRO_REDUX/$SPECPROD/
        download : if not found locally, try to fetch remotely
        outdir : use this directory for output instead of canonical location
    """

    #- NOTE: specprod_dir is the directory $DESI_SPECTRO_REDUX/$SPECPROD,
    #-       specprod is just the environment variable $SPECPROD

    location = dict(
        raw = '{rawdata_dir}/{night}/desi-{expid:08d}.fits.fz',
        pix = '{rawdata_dir}/{night}/pix-{camera}-{expid:08d}.fits',
        fiberflat = '{specprod_dir}/calib2d/{night}/fiberflat-{camera}-{expid:08d}.fits',
        frame = '{specprod_dir}/exposures/{night}/{expid:08d}/frame-{camera}-{expid:08d}.fits',
        cframe = '{specprod_dir}/exposures/{night}/{expid:08d}/cframe-{camera}-{expid:08d}.fits',
        sky = '{specprod_dir}/exposures/{night}/{expid:08d}/sky-{camera}-{expid:08d}.fits',
        stdstars = '{specprod_dir}/exposures/{night}/{expid:08d}/stdstars-{spectrograph:d}-{expid:08d}.fits',
        calib = '{specprod_dir}/exposures/{night}/{expid:08d}/calib-{camera}-{expid:08d}.fits',
        qa_data = '{specprod_dir}/exposures/{night}/{expid:08d}/qa-{camera}-{expid:08d}.yaml',
        qa_data_exp = '{specprod_dir}/exposures/{night}/{expid:08d}/qa-{expid:08d}.yaml',
        qa_bootcalib = '{specprod_dir}/calib2d/psf/{night}/qa-psfboot-{camera}.pdf',
        qa_sky_fig = '{specprod_dir}/exposures/{night}/{expid:08d}/qa-sky-{camera}-{expid:08d}.png',
        qa_flux_fig = '{specprod_dir}/exposures/{night}/{expid:08d}/qa-flux-{camera}-{expid:08d}.png',
        qa_calib = '{specprod_dir}/calib2d/{night}/qa-{camera}-{expid:08d}.yaml',
        qa_calib_exp = '{specprod_dir}/calib2d/{night}/qa-{expid:08d}.yaml',
        qa_flat_fig = '{specprod_dir}/calib2d/{night}/qa-flat-{camera}-{expid:08d}.png',
        qa_ztruth = '{specprod_dir}/exposures/{night}/qa-ztruth-{night}.yaml',
        qa_ztruth_fig = '{specprod_dir}/exposures/{night}/qa-ztruth-{night}.png',
        psf = '{specprod_dir}/exposures/{night}/{expid:08d}/psf-{camera}-{expid:08d}.fits',
        psfnight = '{specprod_dir}/calib2d/psf/{night}/psfnight-{camera}.fits',
        psfboot = '{specprod_dir}/calib2d/psf/{night}/psfboot-{camera}.fits',
        fibermap = '{rawdata_dir}/{night}/fibermap-{expid:08d}.fits',
        brick = '{specprod_dir}/bricks/{brickname}/brick-{band}-{brickname}.fits',
        coadd = '{specprod_dir}/bricks/{brickname}/coadd-{band}-{brickname}.fits',
        coadd_all = '{specprod_dir}/bricks/{brickname}/coadd-{brickname}.fits',
        zbest = '{specprod_dir}/bricks/{brickname}/zbest-{brickname}.fits',
        zspec = '{specprod_dir}/bricks/{brickname}/zspec-{brickname}.fits',
        zcatalog = '{specprod_dir}/zcatalog-{specprod}.fits',
    )
    location['desi'] = location['raw']

    #- Do we know about this kind of file?
    if filetype not in location:
        raise IOError("Unknown filetype {}; known types are {}".format(filetype, list(location.keys())))

    #- Check for missing inputs
    required_inputs = [i[0] for i in re.findall(r'\{([a-z_]+)(|[:0-9d]+)\}',location[filetype])]

    if rawdata_dir is None and 'rawdata_dir' in required_inputs:
        rawdata_dir = rawdata_root()

    if specprod_dir is None and 'specprod_dir' in required_inputs:
        specprod_dir = specprod_root()

    if 'specprod' in required_inputs:
        #- Replace / with _ in $SPECPROD so we can use it in a filename
        specprod = os.getenv('SPECPROD').replace('/', '_')
    else:
        specprod = None

    actual_inputs = {
        'specprod_dir':specprod_dir, 'specprod':specprod,
        'night':night, 'expid':expid, 'camera':camera, 'brickname':brickname,
        'band':band, 'spectrograph':spectrograph
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

    Uses :func:`findfile` to determine the valid file names for the specified type.
    Any camera identifiers not matching the regular expression [brz][0-9] will be
    silently ignored.

    Args:
        filetype(str): Type of files to get. Valid choices are 'frame','cframe','psf'.
        night(str): Date string for the requested night in the format YYYYMMDD.
        expid(int): Exposure number to get files for.
        rawdata_dir(str): [optional] overrides $DESI_SPECTRO_DATA

    Returns:
        dict: Dictionary of found file names using camera id strings as keys, which are
            guaranteed to match the regular expression [brz][0-9].
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


def get_files(filetype, night, expid, specprod_dir=None):
    """Get files for a specified exposure.

    Uses :func:`findfile` to determine the valid file names for the specified type.
    Any camera identifiers not matching the regular expression [brz][0-9] will be
    silently ignored.

    Args:
        filetype(str): Type of files to get. Valid choices are 'frame','cframe','psf'.
        night(str): Date string for the requested night in the format YYYYMMDD.
        expid(int): Exposure number to get files for.
        specprod_dir(str): Path containing the exposures/ directory to use. If the value
            is None, then the value of :func:`specprod_root` is used instead. Ignored
            when raw is True.

    Returns:
        dict: Dictionary of found file names using camera id strings as keys, which are
            guaranteed to match the regular expression [brz][0-9].
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


def get_pipe_plandir(specprod_dir=None):
    """
    Return the directory path for pipeline planning files.

    Args:
        specprod_dir (str): Optional path to production directory.  If None,
            the this is obtained from :func:`specprod_root`.

    Returns (str):
        the directory path for pipeline planning files.
    """
    if specprod_dir is None:
        specprod_dir = specprod_root()
    return os.path.join(os.path.abspath(specprod_dir), "plan")


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


def get_pipe_faildir():
    """
    Return the name of the subdirectory containing pipeline failures.

    Returns (str):
        The name of the subdirectory.
    """
    return "failed"
