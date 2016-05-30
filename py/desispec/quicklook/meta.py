"""
desispec.quicklook.meta
================
This mostly imports from desispec.io.meta etc but also modifies as needed for quicklook.

IO metadata functions.
"""

import os
import os.path
import datetime
import glob
import re


def findfile(filetype, night=None, expid=None, camera=None,
    band=None, spectrograph=None, rawdata_dir=None, specprod_dir=None, outdir=None):
    """Returns location where file should be

    Args:
        filetype : file type, typically the prefix, e.g. "frame" or "psf"

    Args depending upon filetype:
        night : YEARMMDD string
        expid : integer exposure id
        camera : 'b0' 'r1' .. 'z9'
        band : one of 'b','r','z' identifying the camera band
        spectrograph : spectrograph number, 0-9

    Options:
        rawdata_dir : overrides $QUICKLOOK_SPECTRO_DATA
        specprod_dir : overrides $QUICKLOOK_SPECTRO_REDUX/$PRODNAME/
        outdir : use this directory for output instead of canonical location
    """
    #- NOTE: specprod_dir is the directory $DESI_SPECTRO_REDUX/$PRODNAME,
    #-       specprod is just the environment variable $PRODNAME

    location = dict(
        raw = '{rawdata_dir}/{night}/desi-{expid:08d}.fits.fz',
        bias= '{rawdata_dir}/{night}/bias-{camera}.fits', # putting some generic name for bias, dark and pixflat for now
        dark= '{rawdata_dir}/{night}/dark-{camera}.fits', #TODO Preprocess does not seem to have a dark. Check with Stephen.
        pixelflat='{rawdata_dir}/{night}/pixflat-{camera}.fits',
        pix = '{rawdata_dir}/{night}/pix-{camera}-{expid:08d}.fits',
        fiberflat = '{specprod_dir}/calib2d/{night}/fiberflat-{camera}-{expid:08d}.fits',
        frame = '{specprod_dir}/exposures/{night}/{expid:08d}/frame-{camera}-{expid:08d}.fits',
        sky = '{specprod_dir}/exposures/{night}/{expid:08d}/sky-{camera}-{expid:08d}.fits',
        qa_data = '{specprod_dir}/exposures/{night}/{expid:08d}/qa-{camera}-{expid:08d}.yaml',
        psf = '{specprod_dir}/calib2d/{night}/psf-{camera}.fits', #Note: TODO desispec.io.findfile has expid here too. What expid should be here?
        fibermap = '{rawdata_dir}/{night}/fibermap-{expid:08d}.fits'
    )
    location['desi'] = location['raw']
    #- Do we know about this kind of file?
    if filetype not in location:
        raise IOError("Unknown filetype {}; known types are {}".format(filetype, location.keys()))

    #- Check for missing inputs
    required_inputs = [i[0] for i in re.findall(r'\{([a-z_]+)(|[:0-9d]+)\}',location[filetype])]

    if rawdata_dir is None and 'rawdata_dir' in required_inputs:
        rawdata_dir = rawdata_root()

    if specprod_dir is None and 'specprod_dir' in required_inputs:
        specprod_dir = specprod_root()

    if 'specprod' in required_inputs:
        #- Replace / with _ in $PRODNAME so we can use it in a filename
        specprod = os.getenv('PRODNAME').replace('/', '_')
    else:
        specprod = None

    actual_inputs = {
        'specprod_dir':specprod_dir, 'specprod':specprod,
        'night':night, 'expid':expid, 'camera':camera,
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

    return filepath

def rawdata_root():
    """Returns directory root for raw data, i.e. ``$QUICKLOOK_SPECTRO_DATA``

    Raises:
        AssertionError: if these environment variables aren't set.
    """
    assert 'QUICKLOOK_SPECTRO_DATA' in os.environ, 'Missing $QUICKLOOK_SPECTRO_DATA environment variable'
    return os.environ['QUICKLOOK_SPECTRO_DATA']


def specprod_root():
    """Return directory root for spectro production, i.e.
    ``$QUICKLOOK_SPECTRO_REDUX/$PRODNAME``.

    Raises:
        AssertionError: if these environment variables aren't set.
    """
    assert 'PRODNAME' in os.environ, 'Missing $PRODNAME environment variable'
    assert 'QUICKLOOK_SPECTRO_REDUX' in os.environ, 'Missing $QUICKLOOK_SPECTRO_REDUX environment variable'
    return os.path.join(os.getenv('QUICKLOOK_SPECTRO_REDUX'), os.getenv('PRODNAME'))

