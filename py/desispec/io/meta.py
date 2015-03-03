#
# See top-level LICENSE file for Copyright information
#
# -*- coding: utf-8 -*-


import os
import os.path
import datetime
import glob

def findfile(filetype, night, expid, camera=None, specprod=None):
    """
    Returns location where file should be
    
    Args:
        filetype : file type, typically the prefix, e.g. "frame" or "psf"
        night : YEARMMDD string
        expid : integer exposure id
        camera : [optional] 'b0' 'r1' .. 'z9'
        specprod : [optional] overrides $DESI_SPECTRO_REDUX/$PRODNAME/
        fetch : [optional, not yet implemented]
            if not found locally, try to fetch remotely
    """
    location = dict(
        raw = '{data}/exposures/{night}/{expid:08d}/desi-{expid:08d}.fits',
        fiberflat = '{specprod}/exposures/{night}/{expid:08d}/fiberflat-{expid:08d}.fits',
        frame = '{specprod}/exposures/{night}/{expid:08d}/frame-{camera}-{expid:08d}.fits',
        cframe = '{specprod}/exposures/{night}/{expid:08d}/cframe-{camera}-{expid:08d}.fits',
        psf = '{specprod}/calib2d/{night}/{expid:08d}/psf-{camera}-{expid:08d}.fits',
        fibermap = '{specprod}/exposures/{night}/{expid:08d}/fibermap-{expid:08d}.fits',
    )
    location['desi'] = location['raw']
    
    if specprod is None:
        specprod = specprod_root()
        
    filepath = location[filetype].format(data=data_root(), specprod=specprod,
        night=night, expid=expid, camera=camera)

    #- normpath to remove extraneous double slashes /a/b//c/d
    return os.path.normpath(filepath)

def validate_night(night):
    """
    Validates a night string and converts to a date.

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
    """
    Get a list of available exposures for the specified night.

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
        night_path = os.path.join(data_root(),'exposures',night)
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

def data_root():
    dir = os.environ[ 'DESI_SPECTRO_DATA' ]
    if dir == None:
        raise RuntimeError('DESI_SPECTRO_DATA environment variable not set')
    return dir

def specprod_root():
    """
    Return $DESI_SPECTRO_REDUX/$PRODNAME

    raises AssertionError if these environment variables aren't set
    """
    assert 'PRODNAME' in os.environ, 'Missing $PRODNAME environment variable'
    assert 'DESI_SPECTRO_REDUX' in os.environ, 'Missing $DESI_SPECTRO_REDUX environment variable'
    return os.path.join(os.getenv('DESI_SPECTRO_REDUX'), os.getenv('PRODNAME'))


