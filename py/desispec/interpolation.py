"""
desispec.interpolation
======================

Utility functions for interpolation of spectra over different wavelength grids.
"""

import numpy as np
import sys
from desiutil.log import get_logger



"""
This is a comment, not a code documentation

A rigorous resampling requires using the resolution matrix, solving
a linear system, and redetermining the new resolution matrix.
This is implemented in the co-addition but the following routine
is intended to be fast.

Another approach, which is also time consuming is the following :

    - input and output are interpreted as with a basis of functions,
    where xin(out) are the barycenter of each function and play the role
    of indices, and yin(out) the amplitudes
    - y(x) = sum_i y_i * f_i(x)
    for a simple binning :
    - f_i(x) =  (x_{i-1}+x_{i})/2<x<=(x_{i}+x_{i+1})/2
    for higher orders, their exists an equivalent f_i(x)
    - the uncertainties on the y_i are given by ivar
    - it is not a spline fit because the input is not necessarily
    dense with respect to the output, so we need this approach

    calling f_i(x) the input basis, a_i, the input node values,
    and g_j(x) the output basis and b_j the output node values,
    a change of b, db, results in a change of a continuous function y(x) :
    dy(x) = g.db = sum_i db_i g_i(x)
    defining the scalar product f.g_{ij} = integral_x f_i(x)*g_j(x) dx
    the induced change of a is
    da = (f.f)^{-1} (f.g) db
    calling H the matrix (f.f)^{-1} (f.g)
    we can write a chi2 on the 'a' coefficents,
    chi2 = sum_i ivar_i ( a_i - sum_j H_ij b_j )**2
    the best fit b is given by the linear system
    A b = B
    where A_kl = sum_i ivar_i H_ik H_il
    and   B_k  = sum_i ivar_i a_i H_ik

    This is the general solution. It is more or less complex depending
    on the basis f and g.

    In many case, a minimal regularization would be needed.
    For instance if we rebin to a finer grid than the
    original.

    We consider a practical approach in the following.

"""



def resample_flux(xout, x, flux, ivar=None, extrapolate=False):
    """Returns a flux conserving resampling of an input flux density.
    The total integrated flux is conserved.

    Args:
        - xout: output SORTED vector, not necessarily linearly spaced
        - x: input SORTED vector, not necessarily linearly spaced
        - flux: input flux density dflux/dx sampled at x

    both x and xout must represent the same quantity with the same unit

    Options:
        - ivar: weights for flux; default is unweighted resampling
        - extrapolate: extrapolate using edge values of input array, default is False,
          in which case values outside of input array are set to zero.
    
    Setting both ivar and extrapolate raises a ValueError because one cannot
    assign an ivar outside of the input data range. 

    Returns:
        if ivar is None, returns outflux
        if ivar is not None, returns outflux, outivar

    This interpolation conserves flux such that, on average,
    output_flux_density = input_flux_density

    The input flux density outside of the range defined by the edges of the first
    and last bins is considered null. The bin size of bin 'i' is given by (x[i+1]-x[i-1])/2
    except for the first and last bin where it is (x[1]-x[0]) and (x[-1]-x[-2])
    so flux density is zero for x<x[0]-(x[1]-x[0])/2 and x>x[-1]-(x[-1]-x[-2])/2

    The input is interpreted as the nodes positions and node values of
    a piece-wise linear function::

        y(x) = sum_i y_i * f_i(x)

    with::

        f_i(x) =    (x_{i-1}<x<=x_{i})*(x-x_{i-1})/(x_{i}-x_{i-1})
                  + (x_{i}<x<=x_{i+1})*(x-x_{i+1})/(x_{i}-x_{i+1})

    the output value is the average flux density in a bin::

        flux_out(j) = int_{x>(x_{j-1}+x_j)/2}^{x<(x_j+x_{j+1})/2} y(x) dx /  0.5*(x_{j+1}+x_{j-1})

    """

    
    
    if ivar is None:
        return _unweighted_resample(xout, x, flux, extrapolate=extrapolate)
    else:
        if extrapolate :
            raise ValueError("Cannot extrapolate ivar. Either set ivar=None and extrapolate=True or the opposite")
        a = _unweighted_resample(xout, x, flux*ivar, extrapolate=False)
        b = _unweighted_resample(xout, x, ivar, extrapolate=False)
        mask = (b>0)
        outflux = np.zeros(a.shape)
        outflux[mask] = a[mask] / b[mask]
        dx = np.gradient(x)
        dxout = np.gradient(xout)
        outivar = _unweighted_resample(xout, x, ivar/dx)*dxout
        
        return outflux, outivar

def _unweighted_resample(output_x,input_x,input_flux_density, extrapolate=False) :
    """Returns a flux conserving resampling of an input flux density.
    The total integrated flux is conserved.

    Args:
        output_x: SORTED vector, not necessarily linearly spaced
        input_x: SORTED vector, not necessarily linearly spaced
        input_flux_density: input flux density dflux/dx sampled at x

    both must represent the same quantity with the same unit
    input_flux_density =  dflux/dx sampled at input_x
    
    Options:
        extrapolate: extrapolate using edge values of input array, default is False,
                     in which case values outside of input array are set to zero

    Returns:
        returns output_flux

    This interpolation conserves flux such that, on average,
    output_flux_density = input_flux_density

    The input flux density outside of the range defined by the edges of the first
    and last bins is considered null. The bin size of bin 'i' is given by (x[i+1]-x[i-1])/2
    except for the first and last bin where it is (x[1]-x[0]) and (x[-1]-x[-2])
    so flux density is zero for x<x[0]-(x[1]-x[0])/2 and x>x[-1]+(x[-1]-x[-2])/2

    The input is interpreted as the nodes positions and node values of
    a piece-wise linear function::

        y(x) = sum_i y_i * f_i(x)

    with::
        f_i(x) =    (x_{i-1}<x<=x_{i})*(x-x_{i-1})/(x_{i}-x_{i-1})
                + (x_{i}<x<=x_{i+1})*(x-x_{i+1})/(x_{i}-x_{i+1})

    the output value is the average flux density in a bin
    flux_out(j) = int_{x>(x_{j-1}+x_j)/2}^{x<(x_j+x_{j+1})/2} y(x) dx /  0.5*(x_{j+1}+x_{j-1})

    """

    # shorter names
    ix=input_x
    iy=input_flux_density
    ox=output_x

    # boundary of output bins
    bins=np.zeros(ox.size+1)
    bins[1:-1]=(ox[:-1]+ox[1:])/2.
    bins[0]=1.5*ox[0]-0.5*ox[1]     # = ox[0]-(ox[1]-ox[0])/2
    bins[-1]=1.5*ox[-1]-0.5*ox[-2]  # = ox[-1]+(ox[-1]-ox[-2])/2
    
    # make a temporary node array including input nodes and output bin bounds
    # first the boundaries of output bins
    tx=bins.copy()
    ty=np.interp(tx,ix,iy) # this sets values left and right of input range to first and/or last input values
    
    #  add input nodes which are inside the node array
    k=np.where((ix>=tx[0])&(ix<=tx[-1]))[0]
    if k.size :
        tx=np.append(tx,ix[k])
        ty=np.append(ty,iy[k])
    
    # add the edges of the first and last input bins
    # to the temporary node array if inside array
    ixmin=1.5*ix[0]-0.5*ix[1]  # = ix[0]-(ix[1]-ix[0])/2
    if ixmin>np.min(tx) :
        y_ixmin=1.5*iy[0]-0.5*iy[1] # extrapolation using slope (and not constant) for this point to preserve integral in first input bin
        tx=np.append(tx,ixmin)
        ty=np.append(ty,y_ixmin)
        
    ixmax=1.5*ix[-1]-0.5*ix[-2]
    if ixmax<np.max(tx) :
        y_ixmax=1.5*iy[-1]-0.5*iy[-2] # extrapolation using slope (and not constant) for this point to preserve integral in last input bin
        tx=np.append(tx,ixmax)
        ty=np.append(ty,y_ixmax)
    
    
    # sort this node array
    p = tx.argsort()
    tx=tx[p]
    ty=ty[p]
    
    # now we do a simple integration in each bin of the piece-wise
    # linear function of the temporary nodes

    # integral of individual trapezes
    trapeze_integrals=(ty[1:]+ty[:-1])*(tx[1:]-tx[:-1])/2.
    
    # output flux
    # for each bin, we sum the trapeze_integrals that belong to that bin
    # and divide by the bin size

    trapeze_centers=(tx[1:]+tx[:-1])/2.
    binsize = bins[1:]-bins[:-1]
    
    if np.min(binsize)<=0  :
        raise ValueError("Minimum bin size = {}".format(np.min(binsize)))
    
    if extrapolate :
        of = np.histogram(trapeze_centers, bins=bins, weights=trapeze_integrals)[0] / binsize
    else : # only keep trapezes inside input array (including assumed input bin edges) to preserve flux and ivar       
        ii=(tx[1:]<=ixmax)&(tx[:-1]>=ixmin)
        of = np.histogram(trapeze_centers[ii], bins=bins, weights=trapeze_integrals[ii])[0] / binsize
    
    return of
