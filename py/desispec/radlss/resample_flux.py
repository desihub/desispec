# import jax
import time
import numba
import numpy as np

from   numba import jit

@numba.jit(nopython=True)
def compute_bin(x, bin_edges):
    # assuming uniform bins for now
    n     = bin_edges.shape[0] - 1

    a_min = bin_edges[0]
    a_max = bin_edges[-1]

    # special case to mirror NumPy behavior for last bin
    if x == a_max:
        return n - 1 # a_max always in last bin

    bin = int(n * (x - a_min) / (a_max - a_min))

    if bin < 0 or bin >= n:
        return None

    else:
        return bin

@numba.jit(nopython=True)
def numba_histogram(a, bin_edges, weights):
    hist      = np.zeros(len(bin_edges) -1)
    
    for i, x in enumerate(a.flat):
        bin   = compute_bin(x, bin_edges)

        if bin is not None:
            hist[int(bin)] += weights[i]
            
    return  hist, bin_edges

@jit(nopython=True)
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
    
    See:  https://github.com/desihub/desispec/blob/d43caaac473c4586b2beeb1308919925920d64ef/py/desispec/interpolation.py#L63
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

@jit(nopython=True)
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

    # bins=jax.ops.index_update(bins, jax.ops.index[1:-1],  (ox[:-1]+ox[1:])/2.)
    # bins=jax.ops.index_update(bins, 0,  1.5*ox[0]-0.5*ox[1])
    # bins=jax.ops.index_update(bins, -1, 1.5*ox[-1]-0.5*ox[-2])

    bins[1:-1]=(ox[:-1]+ox[1:])/2.
    bins[0]=1.5*ox[0]-0.5*ox[1]     # = ox[0]-(ox[1]-ox[0])/2
    bins[-1]=1.5*ox[-1]-0.5*ox[-2]  # = ox[-1]+(ox[-1]-ox[-2])/2
    
    # make a temporary node array including input nodes and output bin bounds
    # first the boundaries of output bins
    tx=bins.copy()

    # if we do not extrapolate,
    # because the input is a considered a piece-wise linear function, i.e. the sum of triangles f_i(x),
    # we add two points at ixmin = ix[0]-(ix[1]-ix[0]) and  ixmax = ix[-1]+(ix[-1]-ix[-2])
    # with zero flux densities, corresponding to the edges of the first and last triangles.
    # this solves naturally the edge problem.
    if not extrapolate :
        # note we have to keep the array sorted here because we are going to use it for interpolation
        ix = np.append( 2*ix[0]-ix[1] , ix)
        iy = np.append(0.,iy)
        ix = np.append(ix, 2*ix[-1]-ix[-2])
        iy = np.append(iy, 0.)

    # this sets values left and right of input range to first and/or last input values
    # first and last values are=0 if we are not extrapolating
    ty=np.interp(tx,ix,iy)
    
    #  add input nodes which are inside the node array
    k=np.where((ix>=tx[0])&(ix<=tx[-1]))[0]
    if k.size :
        tx=np.append(tx,ix[k])
        ty=np.append(ty,iy[k])
        
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
    
    if np.any(binsize<=0):
        raise ValueError("Zero or negative bin size")
    
    return  numba_histogram(trapeze_centers, bins, trapeze_integrals)[0] / binsize


if __name__ == '__main__':
    import pylab as pl

    '''
    rands = np.random.uniform(0.0, 1.0, 10)
    bins  = np.arange(0., 1., 0.1) 
    wghts = np.ones_like(rands)
    
    numba_histogram(rands, bins, wghts)
    '''
    
    a = np.arange(0, 1.e3, 0.8)
    b = 10. * a
    c = np.arange(0, 1.e3,  5.)

    start = time.time()

    d = resample_flux(c, a, b, ivar=None, extrapolate=False)
      
    end = time.time()
    
    print("Elapsed (with compilation) = %s" % (end - start))
      
    start = time.time()

    d = resample_flux(c, a, b, ivar=None, extrapolate=False)
    
    end = time.time()

    print("Elapsed (without compilation) = %s" % (end - start))
    
    pl.plot(a, b, lw=0.1)
    pl.plot(c, d, lw=0.1)

    pl.show()
    
    print('Done.')

