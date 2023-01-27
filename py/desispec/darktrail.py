"""
desispec.darktrail
==================

Utility functions to correct dark trails found for some LBNL CCD
amplifiers due to an imperfect CDS (correlated double sampling) suppression of the reset level.
The cause of the effect is unknown (as this is written), and we have
not been able to correcting it with adjustment of the readout electronic parameters.

The function here are used to do a software correction of the effect.
"""

import numpy as np
import scipy.signal
from desiutil.log import get_logger

def correct_dark_trail(image,xyslice,left,width,amplitude) :
    """
    remove the dark trails from a preprocessed image with a processing of the form::

        image[j,i] += sum_{k<i} image[j,k] * a * |i-k|/width * exp( -|i-k|/width)

    Args:
        image : desispec.Image class instance
        xyslice : tuple of python slices (yy,xx) where yy is indexing CCD rows (firt index in 2D numpy array) and xx is the other
        left : if true , the trails are to the left of the image spots, as is the case for amplifiers B and D.
        width : width parameter in unit of pixels
        amplitude : amplitude of the dark trail

    """
    hwidth=int(5*width+1)
    kernel = np.zeros((3,2*hwidth+1))
    dx=np.arange(-hwidth,hwidth+1)
    adx=np.abs(dx)
    if left :
        kernel[1]=amplitude*(adx/width)*np.exp(-adx/width)*(dx<0)
    else :
        kernel[1]=amplitude*(adx/width)*np.exp(-adx/width)*(dx>0)
    trail = scipy.signal.fftconvolve(image[xyslice],kernel,"same")
    image[xyslice] += trail
