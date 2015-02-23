#
# See top-level LICENSE file for Copyright information
#
# -*- coding: utf-8 -*-


import os

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
        fibermap = '{data}/{night}/fibermap-{expid:08d}.fits',
    )
    location['desi'] = location['raw']
    
    if specprod is None:
        specprod = specprod_root()
        
    filepath = location[filetype].format(data=data_root(), specprod=specprod,
        night=night, expid=expid, camera=camera)
    
    #- normpath to remove extraneous double slashes /a/b//c/d
    return os.path.normpath(filepath)

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


