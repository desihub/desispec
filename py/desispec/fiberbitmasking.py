"""
desispec.fiberbitmasking
==============

Functions to properly take FIBERSTATUS into account in the variances for data reduction
"""

from __future__ import absolute_import, division
import numpy as np
from astropy.table import Table

from desiutil.log import get_logger
from desispec.maskbits import fibermask as fmsk
from desispec.maskbits import specmask


def get_fiberbitmasked_frame(frame,bitmask=None,ivar_framemask=True):
    """
    Wrapper script of get_fiberbitmasked_frame_arrays that will
    return a modified version of the frame instead of just the
    flux and ivar
    NOTE: The input "frame" variable itself is modified and returned,
          not a copy.
    """
    ivar,mask = get_fiberbitmasked_frame_arrays(frame,bitmask,ivar_framemask,return_mask=True)
    frame.mask = mask
    frame.ivar = ivar
    return frame

def get_fiberbitmasked_frame_arrays(frame,bitmask=None,ivar_framemask=True,return_mask=False):
    """
    Function that takes a frame object and a bitmask and
    returns ivar (and optionally mask) array(s) that have fibers with
    offending bits in fibermap['FIBERSTATUS'] set to
    0 in ivar and optionally flips a bit in mask.

    input:
        frame: frame object
        bitmask: int32 or list/array of int32's derived from desispec.maskbits.fibermask
                 OR string indicating a keyword for get_fiberbitmask_comparison_value()
        ivar_framemask: bool (default=True), tells code whether to multiply the output
                 variance by (frame.mask==0)
        return_mask: bool, (default=False). Returns the frame.mask with the logic of
                 FIBERSTATUS applied.

    output:
        ivar: frame.ivar where the fibers with FIBERSTATUS & bitmask > 0
              set to zero ivar
        mask: (optional) frame.mask logically OR'ed with BADFIBER bit in cases with
              a bad FIBERSTATUS

    example bitmask list:
        bitmask = [fmsk.BROKENFIBER,fmsk.UNASSIGNED,fmsk.BADFIBER,\
                    fmsk.BADTRACE,fmsk.MANYBADCOL, fmsk.MANYREJECTED]
        bitmask = get_fiberbitmask_comparison_value(kind='fluxcalib', band='brz')
        bitmask = 'fluxcalib'
        bitmask = 4128780
    """
    ivar = frame.ivar.copy()
    mask = frame.mask.copy()
    if ivar_framemask and frame.mask is not None:
        ivar *= (frame.mask==0)

    fmap = Table(frame.fibermap)

    if frame.fibermap is None:
        log = get_logger()
        log.warning("No fibermap was given, so no FIBERSTATUS check applied.")

    if bitmask is None or frame.fibermap is None:
        if return_mask:
            return ivar, mask
        else:
            return ivar

    if type(bitmask) in [int,np.int32]:
        bad = bitmask
    elif type(bitmask) == str:
        if bitmask.isnumeric():
            bad = np.int32(bitmask)
        else:
            band = 'brz' # all by default
            if frame.meta is not None :
                if "CAMERA" in frame.meta.keys() :
                    camera = frame.meta["CAMERA"].lower()
                    band   = camera[0]
            bad    = get_fiberbitmask_comparison_value(kind=bitmask,band=band)
    else:
        bad = bitmask[0]
        for bit in bitmask[1:]:
            bad |= bit

    # find if any fibers have an intersection with the bad bits
    badfibers = fmap['FIBER'][ (fmap['FIBERSTATUS'] & bad) > 0 ].data
    badfibers = badfibers % 500
    # For the bad fibers, loop through and nullify them
    for fiber in badfibers:
        mask[fiber] |= specmask.BADFIBER
        if ivar_framemask :
            ivar[fiber] = 0.

    if return_mask:
        return ivar,mask
    else:
        return ivar


def get_fiberbitmask_comparison_value(kind,band):
    """
        Takes a string argument and returns a 32-bit integer representing the logical OR of all
        relevant fibermask bits for that given reduction step

        input:
             kind: str : string designating which combination of bits to use based on the operation.
                         Possible values are "all", "sky" (or "skysub"), "flat", 
                         "flux" (or "fluxcalib"), "star" (or "stdstars")
             band: str : BADAMP band bits to set. Values include 'b', 'r', 'z', or 
                         combinations thereof such as 'brz'
                         
        output:
             bitmask : 32 bit bitmask corresponding to the fiberbitmask of the desired kind
                       in the desired cameras (bands).
        
    """
    if kind.lower() == 'all':
        return get_all_fiberbitmask_with_amp(band)
    elif kind.lower()[:3] == 'sky':
        return get_skysub_fiberbitmask_val(band)
    elif kind.lower() == 'flat':
        return get_flat_fiberbitmask_val(band)
    elif 'star' in kind.lower():
        return get_stdstars_fiberbitmask_val(band)
    elif 'flux' in kind.lower():
        return get_fluxcalib_fiberbitmask_val(band)
    else:
        log = get_logger()
        log.warning("Keyword {} given to get_fiberbitmask_comparison_value() is invalid.".format(kind)+\
                    " Using 'fluxcalib' fiberbitmask.")
        return get_fluxcalib_fiberbitmask_val(band)


def get_skysub_fiberbitmask_val(band):
    return get_all_fiberbitmask_with_amp(band)

def get_flat_fiberbitmask_val(band):
    return (fmsk.BROKENFIBER | fmsk.BADFIBER | fmsk.BADTRACE | fmsk.BADARC | \
            fmsk.MANYBADCOL | fmsk.MANYREJECTED )

def get_fluxcalib_fiberbitmask_val(band):
    return get_all_fiberbitmask_with_amp(band)

def get_stdstars_fiberbitmask_val(band):
    return get_all_fiberbitmask_with_amp(band) | fmsk.POORPOSITION

def get_all_nonamp_fiberbitmask_val():
    """Return a mask for all fatally bad FIBERSTATUS bits except BADAMPB/R/Z

    Note: does not include STUCKPOSITIONER or RESTRICTED, which could still
    be on a valid sky location, or even a target for RESTRICTED.
    Also does not include POORPOSITION which is bad for stdstars
    but not necessarily fatal for otherwise processing a normal fiber.
    """
    return (fmsk.UNASSIGNED | fmsk.BROKENFIBER | fmsk.MISSINGPOSITION | \
            fmsk.BADPOSITION | \
            fmsk.BADFIBER | fmsk.BADTRACE | fmsk.BADARC | fmsk.BADFLAT | \
            fmsk.MANYBADCOL | fmsk.MANYREJECTED )

def get_justamps_fiberbitmask():
    return ( fmsk.BADAMPB | fmsk.BADAMPR | fmsk.BADAMPZ )

def get_all_fiberbitmask_with_amp(band):
    amp_mask = get_all_nonamp_fiberbitmask_val()
    if band.lower().find('b')>=0:
        amp_mask |= fmsk.BADAMPB
    if band.lower().find('r')>=0:
        amp_mask |= fmsk.BADAMPR
    if band.lower().find('z')>=0:
        amp_mask |= fmsk.BADAMPZ
    return amp_mask

def get_all_fiberbitmask_val():
    return ( get_all_nonamp_fiberbitmask_val() | get_justamps_fiberbitmask() )
