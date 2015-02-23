"""
Utility functions for interpolation of spectra over different wavelength grid
"""

import numpy as np
import sys
from desispec.log import get_logger

#import time # for debugging

def bin_bounds(x) :
    if x.size<2 :
        get_logger().error("bin_bounds, x.size=%d"%x.size)
        exit(12)
    tx=np.sort(x)
    x_minus=np.roll(tx,1)
    x_minus[0]=x_minus[1]+tx[0]-tx[1]
    x_plus=np.roll(tx,-1)
    x_plus[-1]=x_plus[-2]+tx[-1]-tx[-2]
    x_minus=0.5*(x+x_minus)
    x_plus=0.5*(x+x_plus)
    
    del tx
    return x_minus,x_plus

def resample_flux(output_x,input_x,input_flux_density,left=0.,right=0.) :
    """
    Returns a flux conserving resampling of an input flux density.
     
    Inputs:   
    input_x is a SORTED vector, not necessarily linearly spaced
    output_x is a SORTED vector, not necessarily linearly spaced
    both must represent the same quantity with the same unit
    input_flux_density =  dflux/dx sampled at input_x
    
    This interpolation conserves flux such that, on average,
    output_flux_density = input_flux_density 
    
    This interpolation scheme is a simple average in an x interval
    for which boundaries are placed at the mid-distance between
    consecutive x points.  The advantage with respect to other
    methods is that the weights are all positive or null, such
    that there is no anti-correlation in the output
    (only positive correlation)
    
    Options: 
    left=value for expolation to the left, if None, use input_flux_density[0], default=0
    right=value for expolation to the right, if None, use input_flux_density[-1], default=0
    """
    
    # shorter names
    ow=output_x
    iw=input_x
    iflux=input_flux_density
    
    # boundary of output bins
    owm,owp=bin_bounds(ow)
    # interpolated fluxes at boundaries
    ofm=np.interp(owm,iw,iflux,left=left,right=right)
    ofp=np.interp(owp,iw,iflux,left=left,right=right)
        
    # make arrays of x and flux that contain all x points.
    # bounds appear twice as a trick to compute easily the weights at edge of output bins
    eps=0.0000001*(owp-owm) # anything better ?
    k=np.where((iw>owm[0])&(iw<owp[-1]))[0]
    tw=iw[k]
    tw=np.append(tw,owm+eps)
    tw=np.append(tw,owp-eps)
    tf=iflux[k]
    tf=np.append(tf,ofm)
    tf=np.append(tf,ofp)
    
    # sort this array
    p = tw.argsort()
    tw=tw[p]
    tf=tf[p]
    del p
    
    # compute bounds to associate a weight to each flux density
    twm,twp=bin_bounds(tw)
    weight=(twp-twm) # because of the fact we have duplicate the output bin bounds location, they are given the correct weight
    
    # set weight to zero ouside input if no extrapolation
    if not left==None :
        # no interpolation
        weight=weight*(twm>=iw[0])
    if not right==None :
        # no interpolation
        weight=weight*(twp<=iw[-1])

    del twm
    del twp
    
    # compute output flux as weighted mean
    bins=np.append(owm,owp[-1])
    output_flux,bin_edges=np.histogram(tw,bins,weights=weight*tf)
    sw,bin_edges=np.histogram(tw,bins,weights=weight)
    output_flux=output_flux/(sw+(sw==0))
            
    del tw
    del tf
    del ofm
    del ofp
    del weight
    
    # set flux to given value ouside input if no extrapolation
    if not left==None :
        output_flux=output_flux*(ow>=iw[0])+left*(ow<iw[0])
    if not right==None :
        output_flux=output_flux*(ow<=iw[-1])+right*(ow>iw[-1])
        
    return output_flux
