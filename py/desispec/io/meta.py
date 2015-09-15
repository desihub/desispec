"""
desispec.io.meta
================

IO metadata functions.
"""

import os
import os.path
import datetime
import glob
import re

def findfile(filetype, night=None, expid=None, camera=None, brickid=None,
    band=None, spectrograph=None, specprod=None, download=False):
    """Returns location where file should be

    Args:
        filetype : file type, typically the prefix, e.g. "frame" or "psf"
        night : [optional] YEARMMDD string
        expid : [optional] integer exposure id
        camera : [optional] 'b0' 'r1' .. 'z9'
        brickid : [optional] brick ID string
        band : [optional] one of 'b','r','z' identifying the camera band
        spectrograph : [optional] spectrograph number, 0-9
        specprod : [optional] overrides $DESI_SPECTRO_REDUX/$PRODNAME/
        download : [optional, not yet implemented]
            if not found locally, try to fetch remotely
    """
    location = dict(
        raw = '{rawdatadir}/{night}/desi-{expid:08d}.fits',
        pix = '{rawdatadir}/{night}/pix-{camera}-{expid:08d}.fits',
        ### fiberflat = '{specprod}/exposures/{night}/{expid:08d}/fiberflat-{camera}-{expid:08d}.fits',
        fiberflat = '{specprod}/calib2d/{night}/fiberflat-{camera}-{expid:08d}.fits',
        frame = '{specprod}/exposures/{night}/{expid:08d}/frame-{camera}-{expid:08d}.fits',
        cframe = '{specprod}/exposures/{night}/{expid:08d}/cframe-{camera}-{expid:08d}.fits',
        sky = '{specprod}/exposures/{night}/{expid:08d}/sky-{camera}-{expid:08d}.fits',
        stdstars = '{specprod}/exposures/{night}/{expid:08d}/stdstars-sp{spectrograph:d}-{expid:08d}.fits',
        calib = '{specprod}/exposures/{night}/{expid:08d}/fluxcalib-{camera}-{expid:08d}.fits',
        ### psf = '{specprod}/exposures/{night}/{expid:08d}/psf-{camera}-{expid:08d}.fits',
        psf = '{specprod}/calib2d/{night}/psf-{camera}-{expid:08d}.fits',
        fibermap = '{rawdatadir}/{night}/fibermap-{expid:08d}.fits',
        brick = '{specprod}/bricks/{brickid}/brick-{band}-{brickid}.fits',
        coadd = '{specprod}/bricks/{brickid}/coadd-{band}-{brickid}.fits',
        coadd_all = '{specprod}/bricks/{brickid}/coadd-{brickid}.fits',
        zbest = '{specprod}/bricks/{brickid}/zbest-{brickid}.fits',
        zspec = '{specprod}/bricks/{brickid}/zspec-{brickid}.fits',
    )
    location['desi'] = location['raw']

    #- Do we know about this kind of file?
    if filetype not in location:
        raise IOError("Unknown filetype {}; known types are {}".format(filetype, location.keys()))

    if specprod is None:
        specprod = specprod_root()

    #- Check for missing inputs
    required_inputs = [i[0] for i in re.findall(r'\{([a-z]+)(|[:0-9d]+)\}',location[filetype])]
    actual_inputs = {
        'rawdatadir':rawdata_root(), 'specprod':specprod,
        'night':night, 'expid':expid, 'camera':camera, 'brickid':brickid,
        'band':band, 'spectrograph':spectrograph
        }
    for i in required_inputs:
        if actual_inputs[i] is None:
            raise ValueError("Required input '{0}' is not set for type '{1}'!".format(i,filetype))
    #- normpath to remove extraneous double slashes /a/b//c/d
    filepath = os.path.normpath(location[filetype].format(**actual_inputs))

    if download:
        from .download import download
        filepath = download(filepath,single_thread=True)[0]
    return filepath

def get_files(filetype,night,expid,specprod = None):
    """Get files for a specified exposure.

    Uses :func:`findfile` to determine the valid file names for the specified type.
    Any camera identifiers not matching the regular expression [brz][0-9] will be
    silently ignored.

    Args:
        filetype(str): Type of files to get. Valid choices are 'frame','cframe','psf'.
        night(str): Date string for the requested night in the format YYYYMMDD.
        expid(int): Exposure number to get files for.
        specprod(str): Path containing the exposures/ directory to use. If the value
            is None, then the value of :func:`specprod_root` is used instead. Ignored
            when raw is True.

    Returns:
        dict: Dictionary of found file names using camera id strings as keys, which are
            guaranteed to match the regular expression [brz][0-9].
    """
    glob_pattern = findfile(filetype,night,expid,camera = '*',specprod = specprod)
    literals = map(re.escape,glob_pattern.split('*'))
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

def get_exposures(night,raw = False,specprod = None):
    """Get a list of available exposures for the specified night.

    Exposures are identified as correctly formatted subdirectory names within the
    night directory, but no checks for valid contents of these exposure subdirectories
    are performed.

    Args:
        night(str): Date string for the requested night in the format YYYYMMDD.
        raw(bool): Returns raw exposures if set, otherwise returns processed exposures.
        specprod(str): Path containing the exposures/ directory to use. If the value
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
        night_path = os.path.join(rawdata_root(),'exposures',night)
    else:
        if specprod is None:
            specprod = specprod_root()
        night_path = os.path.join(specprod,'exposures',night)

    if not os.path.exists(night_path):
        raise RuntimeError('Non-existent night %s' % night)

    exposures = [ ]
    for entry in glob.glob(os.path.join(night_path,'*')):
        head,tail = os.path.split(entry)
        try:
            exposure = int(tail)
            assert tail == ('%08d' % exposure)
            exposures.append(exposure)
        except (ValueError,AssertionError):
            # Silently ignore entries that are not exposure subdirectories.
            pass

    return exposures

def rawdata_root():
    """Returns directory root for raw data, i.e. ``$DESI_SPECTRO_DATA``

    Raises:
        AssertionError: if these environment variables aren't set.
    """
    assert 'DESI_SPECTRO_DATA' in os.environ, 'Missing $DESI_SPECTRO_DATA environment variable'
    return os.environ['DESI_SPECTRO_DATA']

def specprod_root():
    """Return directory root for spectro production, i.e.
    ``$DESI_SPECTRO_REDUX/$PRODNAME``.

    Raises:
        AssertionError: if these environment variables aren't set.
    """
    assert 'PRODNAME' in os.environ, 'Missing $PRODNAME environment variable'
    assert 'DESI_SPECTRO_REDUX' in os.environ, 'Missing $DESI_SPECTRO_REDUX environment variable'
    return os.path.join(os.getenv('DESI_SPECTRO_REDUX'), os.getenv('PRODNAME'))
