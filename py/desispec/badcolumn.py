"""
desispec.badcolumn
========================

Utility functions to treat bad CCD columns
"""

import numpy as np

from desiutil.log import get_logger
from desispec.maskbits import specmask,fibermask

def flux_bias_function(delta_x) :
    """Multiply the excess counts in a bad column by this function to determine the bias on the extracted flux of a spectral trace situated at a distance dx (in pixels) from the bad CCD column

    Args:

     delta_x: float or numpy array

    Returns:

     flux bias, same dimension as input delta_x
    """
    scalar=np.isscalar(delta_x)
    delta_x = np.atleast_1d(delta_x)
    val = np.zeros(delta_x.shape,dtype=float)
    nonnull  = (np.abs(delta_x)<4.5)
    val[nonnull] = 1.1/(1+np.abs(delta_x[nonnull]/2.)**5)
    if scalar :
        return float(val)
    else :
        return val

def compute_badcolumn_specmask(frame,xyset,badcolumns_table,threshold_value=2) :
    """
    returns mask numpy array of same shape as frame.flux if some spectral values are masked
    or None if nothing is masked
    """

    if len(badcolumns_table)==0 : return None

    log = get_logger()

    if frame.mask is None :
        mask=np.zeros(frame.flux.shape,dtype='uint32')
    else :
        mask=np.zeros_like(frame.mask)

    dx_threshold=4

    fiber_x=np.zeros(frame.flux.shape,dtype=float)
    for fiber in range(frame.flux.shape[0]) :
        fiber_x[fiber] = xyset.x_vs_wave(fiber,frame.wave)

    for column_x,column_val in zip(badcolumns_table["COLUMN"],badcolumns_table["VALUE"]) :
        dx = fiber_x - column_x
        log.info("Processing col at x={} val={}".format(column_x,column_val))
        bias = column_val*flux_bias_function(dx)
        mask[np.abs(bias)>=threshold_value] |= specmask.BADCOLUMN

    nvalsperfiber=np.sum(mask>0,axis=1)
    nvals=np.sum(nvalsperfiber)
    nfibers=np.sum(nvalsperfiber>0)
    log.info("Masked {} flux values from {} fibers".format(nvals,nfibers))

    return mask

def compute_badcolumn_fibermask(frame_mask,threshold_specfrac=0.4) :

    fiber_mask = np.zeros(frame_mask.shape[0],dtype='uint32')
    badfibers  = np.sum((frame_mask & specmask.BADCOLUMN)>0,axis=1) >= (threshold_specfrac*frame_mask.shape[1])
    fiber_mask[badfibers] |= fibermask.BADCOLUMN
    return fiber_mask
