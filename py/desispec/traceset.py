"""
desispec.traceset
==============

Lightweight wrapper class for trace coordinates and wavelength solution, to be returned by io.read_traceset
"""


from __future__ import absolute_import, division

import numbers
import numpy as np

from desispec import util
from desiutil.log import get_logger

from specter.util import legval_numba
  

class TraceSet(object):
    def __init__(self, xcoef, ycoef, wavemin,wavemax) :
        """
        Lightweight wrapper for trace coordinates and wavelength solution
        

        Args:
            xcoef: 2D[ntrace, ncoef] Legendre coefficient of x as a function of wavelength
            ycoef: 2D[ntrace, ncoef] Legendre coefficient of y as a function of wavelength
            wavemin : float 
            wavemax : float. wavemin and wavemax are used to define a reduced variable legx(wave,wavemin,wavemax)=2*(wave-wavemin)/(wavemax-wavemin)-1
        used to compute the traces, xccd=legval(legx(wave,wavemin,wavemax),xtrace[fiber])
        """

        self.xcoef=xcoef
        self.ycoef=ycoef
        self.wavemin=wavemin
        self.wavemax=wavemax
    
    def x(fiber,wavelength) :
        
