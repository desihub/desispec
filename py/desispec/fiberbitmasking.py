"""
desispec.frame
==============

Lightweight wrapper class for spectra, to be returned by io.read_frame
"""

from __future__ import absolute_import, division
import numpy as np

from desiutil.log import get_logger
from desispec.maskbits import fibermask as fmsk
from astropy.table import Table

def get_fiberbitmasked_frame(frame,bitmask=None,ivar_framemask=True):
    """
        Wrapper script of get_fiberbitmasked_frame_arrays that will
        return a modified version of the cframe instead of just the 
        flux and ivar
    """
    flux,ivar = get_fiberbitmasked_frame_arrays(frame,bitmask,ivar_framemask)
    outframe = frame
    outframe.flux = flux
    outframe.ivar = ivar
    return outframe

def get_fiberbitmasked_frame_arrays(frame,bitmask=None,ivar_framemask=True):
    """
       Function that takes a frame object and a bitmask and
       returns flux and ivar arrays that have fibers with 
       offending bits in fibermap['FIBERSTATUS'] set to
       0 flux and ivar

       input:
            frame: frame object
            bitmask: int32 or list/array of int32's derived from desispec.maskbits.fibermask
                     OR string indicating a keyword for get_fiberbitmask_comparison_value()
            ivar_framemask: bool (default=True), tells code whether to multiply the output
                     variance by (frame.mask==0)

       output:
            flux: frame.flux where the fibers with FIBERSTATUS & bitmask > 0
                  set to zero fluxes
            ivar: frame.ivar where the fibers with FIBERSTATUS & bitmask > 0                               
                  set to zero ivar
    
       example bitmask list:
                  bad_bits =  [fmsk.BROKENFIBER,fmsk.BADTARGET,fmsk.BADFIBER,\
                               fmsk.BADTRACE,fmsk.MANYBADCOL, fmsk.MANYREJECTED]
    """
    flux = frame.flux.copy()
    ivar = frame.ivar.copy()
    if ivar_framemask and frame.mask is not None:
        ivar *= (frame.mask==0)

    fmap = Table(frame.fibermap)
    
    if frame.fibermap is None:
        log = get_logger()
        log.warning("No fibermap was given, so no FIBERSTATUS check applied.")
        return flux, ivar
    if bitmask is None:
        return flux, ivar

    if type(bitmask) in [int,np.int32]:
        bad = bitmask
    elif type(bitmask) == str:
        if bitmask.isnumeric():
            bad = np.int32(bitmask)
        else:
            bad = get_fiberbitmask_comparison_value(kind=bitmask)
    else:
        bad = bitmask[0]
        for bit in bitmask[1:]:
            bad |= bit
            
    # find if any fibers have an intersection with the bad bits                                           
    badfibers = fmap['FIBER'][ (fmap['FIBERSTATUS'] & bad) > 0 ].data

    # For the bad fibers, loop through and nullify them                                                   
    for fiber in badfibers:                                                
        flux[fiber] = 0.
        ivar[fiber] = 0.
        
    return flux, ivar


def get_fiberbitmask_comparison_value(kind='fluxcalib'):
    """
        Takes a string argument and returns a 32-bit integer representing the logical OR of all
        relevant fibermask bits for that given reduction step

        input:
             kind: str : string designating which combination of bits to use based on the operation 
    
        possible values are:
              "all", "sky" (or "skysub"), "flat", "flux" (or "fluxcalib"), "star" (or "stdstars")
    """
    if kind.lower() == 'all':
        return get_all_fiberbitmask_val()
    elif kind.lower()[:3] == 'sky':
        return get_skysub_fiberbitmask_val()
    elif kind.lower() == 'flat':
        return get_flat_fiberbitmask_val()
    elif 'star' in kind.lower():
        return get_stdstars_fiberbitmask_val()
    elif 'flux' in kind.lower():
        return get_fluxcalib_fiberbitmask_val()
    else:
        log = get_logger()
        log.warning("Keyword {} given to get_fiberbitmask_comparison_value() is invalid.".format(kind)+\
                    " Using 'fluxcalib' fiberbitmask.")
        return get_fluxcalib_fiberbitmask_val()

    
def get_skysub_fiberbitmask_val():
    return (fmsk.BROKENFIBER | fmsk.BADTARGET | fmsk.BADFIBER | fmsk.BADTRACE | \
            fmsk.MANYBADCOL | fmsk.MANYREJECTED)

def get_flat_fiberbitmask_val():
    return (fmsk.BROKENFIBER | fmsk.BADTARGET | fmsk.BADFIBER | fmsk.BADTRACE | \
            fmsk.MANYBADCOL | fmsk.MANYREJECTED | fmsk.BADARC)

def get_fluxcalib_fiberbitmask_val():
    return (fmsk.BROKENFIBER | fmsk.BADTARGET | fmsk.BADFIBER | fmsk.BADTRACE | \
            fmsk.MANYBADCOL | fmsk.MANYREJECTED | fmsk.BADARC | fmsk.BADFLAT)

def get_stdstars_fiberbitmask_val():
    return (fmsk.BROKENFIBER | fmsk.BADTARGET | fmsk.BADFIBER | fmsk.BADTRACE | \
            fmsk.MANYBADCOL | fmsk.MANYREJECTED | fmsk.BADARC | fmsk.BADFLAT)
    
def get_all_fiberbitmask_val():
    return (fmsk.STUCKPOSITIONER | fmsk.BROKENFIBER | fmsk.BADTARGET | fmsk.BADFIBER | fmsk.BADTRACE | \
            fmsk.MANYBADCOL | fmsk.MANYREJECTED | fmsk.BADARC | fmsk.BADFLAT)


