"""
desispec.linalg
===============

Some linear algebra functions.
"""
import numpy as np
import scipy,scipy.linalg,scipy.interpolate
from desispec.log import get_logger

def cholesky_solve(A,B,overwrite=False,lower=False):
    """Returns the solution X of the linear system A.X=B
    assuming A is a positive definite matrix

    Args :
         A : 2D (real symmetric) (nxn) positive definite matrix (numpy.ndarray)
         B : 1D vector, must have dimension n  (numpy.ndarray)

    Options :
        overwrite: replace A data by cholesky decomposition (faster)
        lower: cholesky decomposition triangular matrix is lower instead of upper

    Returns :
         X : 1D vector, same dimension as B  (numpy.ndarray)

    """
    UorL,lower = scipy.linalg.cho_factor(A, lower=lower, overwrite_a=overwrite)
    X = scipy.linalg.cho_solve((UorL,lower),B)
    return X

def cholesky_solve_and_invert(A,B,overwrite=False,lower=False) :
    """
    returns the solution X of the linear system A.X=B
    assuming A is a positive definite matrix

    Args :
         A : 2D (real symmetric) (nxn) positive definite matrix (numpy.ndarray)
         B : 1D vector, must have dimension n  (numpy.ndarray)

    Options :
        overwrite: replace A data by cholesky decomposition (faster)
        lower: cholesky decomposition triangular matrix is lower instead of upper

    Returns:
         X,cov, where
         X : 1D vector, same dimension n as B  (numpy.ndarray)
         cov : 2D positive definite matrix, inverse of A (numpy.ndarray)
    """
    UorL,lower = scipy.linalg.cho_factor(A, overwrite_a=overwrite)
    X   = scipy.linalg.cho_solve((UorL,lower),B)
    inv = scipy.linalg.cho_solve((UorL,lower),scipy.eye(A.shape[0]))
    return X,inv

def cholesky_invert(A) :
    """
    returns the inverse of a positive definite matrix

    Args :
         A : 2D (real symmetric) (nxn) positive definite matrix (numpy.ndarray)

    Returns:
         cov : 2D positive definite matrix, inverse of A (numpy.ndarray)
    """
    UorL,lower = scipy.linalg.cho_factor(A,overwrite_a=False)
    inv = scipy.linalg.cho_solve((UorL,lower),scipy.eye(A.shape[0]))
    return inv


def spline_fit(output_wave,input_wave,input_flux,required_resolution,input_ivar=None,order=3):
    """Performs spline fit of input_flux vs. input_wave and resamples at output_wave
    
    Args:
        output_wave : 1D array of output wavelength samples
        input_wave : 1D array of input wavelengths
        input_flux : 1D array of input flux density
        required_resolution (float) : resolution for spline knot placement
    
    Options:
        input_ivar : 1D array of weights for input_flux
        order (int) : spline order
        
    Returns:
        output_flux : 1D array of flux sampled at output_wave    
    """
    if input_ivar is not None :
        selection=np.where(input_ivar>0)[0]
        if selection.size < 2 :
            log=get_logger()
            log.error("cannot do spline fit because only {0:d} values with ivar>0".format(selection.size))
            raise Error
        w1=input_wave[selection[0]]
        w2=input_wave[selection[-1]]
    else :
        w1=input_wave[0]
        w2=input_wave[-1]

    res=required_resolution
    n=int((w2-w1)/res)
    res=(w2-w1)/(n+1)
    knots=w1+res*(0.5+np.arange(n))
    toto=scipy.interpolate.splrep(input_wave,input_flux,w=input_ivar,k=order,task=-1,t=knots)
    output_flux = scipy.interpolate.splev(output_wave,toto)
    return output_flux
