"""
Coadd spectra
"""

from __future__ import absolute_import, division, print_function
import os, sys, time

import numpy as np

import scipy.sparse
import scipy.linalg
import scipy.sparse.linalg

from astropy.table import Column

# for debugging
import astropy.io.fits as pyfits

from desiutil.log import get_logger

from desispec.interpolation import resample_flux
from desispec.spectra import Spectra
from desispec.resolution import Resolution



def coadd(spectra, cosmics_nsig=0.) :
    log = get_logger()
    targets = np.unique(spectra.fibermap["TARGETID"])
    ntarget=targets.size
    log.debug("number of targets= {}".format(ntarget))
    for b in spectra._bands :
        log.debug("coadding band '{}'".format(b))
        nwave=spectra.wave[b].size
        tflux=np.zeros((ntarget,nwave),dtype=spectra.flux[b].dtype)
        tivar=np.zeros((ntarget,nwave),dtype=spectra.ivar[b].dtype)
        tmask=np.zeros((ntarget,nwave),dtype=spectra.mask[b].dtype)
        trdata=np.zeros((ntarget,spectra.resolution_data[b].shape[1],nwave),dtype=spectra.resolution_data[b].dtype)
        
        for i,tid in enumerate(targets) :
            jj=np.where(spectra.fibermap["TARGETID"]==tid)[0]


            if cosmics_nsig is not None and cosmics_nsig > 0 :
                # interpolate over bad measurements
                # to be able to compute gradient next
                # to a bad pixel and identify oulier
                # many cosmics residuals are on edge
                # of cosmic ray trace, and so can be
                # next to a masked flux bin
                grad=[]
                gradvar=[]
                for j in jj :
                    ttivar = spectra.ivar[b][j]*(spectra.mask[b][j]==0)
                    good = (ttivar>0)
                    bad  = (ttivar<=0)
                    ttflux = spectra.flux[b][j]
                    ttflux[bad] = np.interp(spectra.wave[b][bad],spectra.wave[b][good],ttflux[good])
                    ttivar = spectra.ivar[b][j]
                    ttivar[bad] = np.interp(spectra.wave[b][bad],spectra.wave[b][good],ttivar[good])
                    ttvar = 1./ttivar
                    ttflux[1:] = ttflux[1:]-ttflux[:-1]
                    ttvar[1:]  = ttvar[1:]+ttvar[:-1]
                    ttflux[0]  = 0
                    grad.append(ttflux)
                    gradvar.append(ttvar)

            tivar_unmasked= np.sum(spectra.ivar[b][jj],axis=0)
            ivarjj=spectra.ivar[b][jj]*(spectra.mask[b][jj]==0)
            if cosmics_nsig is not None and cosmics_nsig > 0 and len(grad)>0  :
                grad=np.array(grad)
                gradivar=1/np.array(gradvar)
                nspec=grad.shape[0]
                meangrad=np.sum(gradivar*grad,axis=0)/np.sum(gradivar)
                deltagrad=grad-meangrad
                chi2=np.sum(gradivar*deltagrad**2,axis=0)/(nspec-1)
                
                for l in np.where(chi2>cosmics_nsig**2)[0]  :
                    k=np.argmax(gradivar[:,l]*deltagrad[:,l]**2)
                    #k=np.argmax(flux[:,j])
                    log.debug("masking spec {} wave={}".format(k,spectra.wave[b][l]))
                    ivarjj[k][l]=0.
            
            tivar[i]=np.sum(ivarjj,axis=0)
            tflux[i]=np.sum(ivarjj*spectra.flux[b][jj],axis=0)
            for r in range(spectra.resolution_data[b].shape[1]) :
                trdata[i,r]=np.sum((spectra.ivar[b][jj]*spectra.resolution_data[b][jj,r]),axis=0) # not sure applying mask is wise here
            bad=(tivar[i]==0)
            if np.sum(bad)>0 :
                tivar[i][bad] = np.sum(spectra.ivar[b][jj][bad],axis=0) # if all masked, keep original ivar
                tflux[i][bad] = np.sum(spectra.ivar[b][jj][bad]*spectra.tflux[b][jj][bad],axis=0)
            ok=(tivar[i]>0)
            if np.sum(ok)>0 :
                tflux[i][ok] /= tivar[i][ok]
            ok=(tivar_unmasked>0)
            if np.sum(ok)>0 :
                trdata[i][:,ok] /= tivar_unmasked[ok]
            tmask[i]      = np.bitwise_and.reduce(spectra.mask[b][jj],axis=0)
        spectra.flux[b] = tflux
        spectra.ivar[b] = tivar
        spectra.mask[b] = tmask
        spectra.resolution_data[b] = trdata


    log.debug("merging fibermap")
    jj=np.zeros(ntarget,dtype=int)
    for i,tid in enumerate(targets) :
        jj[i]=np.where(spectra.fibermap["TARGETID"]==tid)[0][0]
    tfmap=spectra.fibermap[jj]
    # smarter values for some columns
    for k in ['DELTA_X','DELTA_Y'] :
        tfmap.rename_column(k,'MEAN_'+k)
        xx = Column(np.arange(ntarget))
        tfmap.add_column(xx,name='RMS_'+k)
    for k in ['NIGHT','EXPID','TILEID','SPECTROID','FIBER'] :
        tfmap.rename_column(k,'FIRST_'+k)
        xx = Column(np.arange(ntarget))
        tfmap.add_column(xx,name='LAST_'+k)
        xx = Column(np.arange(ntarget))
        tfmap.add_column(xx,name='NUM_'+k)

    for i,tid in enumerate(targets) :
        jj = spectra.fibermap["TARGETID"]==tid
        for k in ['DELTA_X','DELTA_Y'] :
            vals=spectra.fibermap[k][jj]
            tfmap['MEAN_'+k][i] = np.mean(vals)
            tfmap['RMS_'+k][i] = np.sqrt(np.mean(vals**2)) # inc. mean offset, not same as std
        for k in ['NIGHT','EXPID','TILEID','SPECTROID','FIBER'] :
            vals=spectra.fibermap[k][jj]
            tfmap['FIRST_'+k][i] = np.min(vals)
            tfmap['LAST_'+k][i] = np.max(vals)
            tfmap['NUM_'+k][i] = np.unique(vals).size
        for k in ['DESIGN_X', 'DESIGN_Y','FIBER_RA', 'FIBER_DEC'] :
            tfmap[k][i]=np.mean(spectra.fibermap[k][jj])
        for k in ['FIBER_RA_IVAR', 'FIBER_DEC_IVAR','DELTA_X_IVAR', 'DELTA_Y_IVAR'] :
            tfmap[k][i]=np.sum(spectra.fibermap[k][jj])

    spectra.fibermap=tfmap
    spectra.scores=None

def get_resampling_matrix(global_grid,local_grid,sparse=False):
    """Build the rectangular matrix that linearly resamples from the global grid to a local grid.

    The local grid range must be contained within the global grid range.

    Args:
        global_grid(numpy.ndarray): Sorted array of n global grid wavelengths.
        local_grid(numpy.ndarray): Sorted array of m local grid wavelengths.

    Returns:
        numpy.ndarray: Array of (m,n) matrix elements that perform the linear resampling.
    """
    assert np.all(np.diff(global_grid) > 0),'Global grid is not strictly increasing.'
    assert np.all(np.diff(local_grid) > 0),'Local grid is not strictly increasing.'
    # Locate each local wavelength in the global grid.
    global_index = np.searchsorted(global_grid,local_grid)

    assert local_grid[0] >= global_grid[0],'Local grid extends below global grid.'
    assert local_grid[-1] <= global_grid[-1],'Local grid extends above global grid.'
    
    # Lookup the global-grid bracketing interval (xlo,xhi) for each local grid point.
    # Note that this gives xlo = global_grid[-1] if local_grid[0] == global_grid[0]
    # but this is fine since the coefficient of xlo will be zero.
    global_xhi = global_grid[global_index]
    global_xlo = global_grid[global_index-1]
    # Create the rectangular interpolation matrix to return.
    alpha = (local_grid - global_xlo)/(global_xhi - global_xlo)
    local_index = np.arange(len(local_grid),dtype=int)
    matrix = np.zeros((len(local_grid),len(global_grid)))
    matrix[local_index,global_index] = alpha
    matrix[local_index,global_index-1] = 1-alpha

    # turn into a sparse matrix
    return scipy.sparse.csc_matrix(matrix)

def decorrelate_divide_and_conquer(Cinv,Cinvf,wavebin) :
    """Decorrelate an inverse covariance using the matrix square root.

    Implements the decorrelation part of the spectroperfectionism algorithm described in
    Bolton & Schlegel 2009 (BS) http://arxiv.org/abs/0911.2689, w uses the matrix square root of
    Cinv to form a diagonal basis. This is generally a better choice than the eigenvector or
    Cholesky bases since it leads to more localized basis vectors, as described in
    Hamilton & Tegmark 2000 http://arxiv.org/abs/astro-ph/9905192.

    Args:
        Cinv(numpy.ndarray): Square array of inverse covariance matrix elements. The input can
            either be a scipy.sparse format or else a regular (dense) numpy array, but a
            sparse format will be internally converted to a dense matrix so there is no
            performance advantage.

    Returns:
        tuple: Tuple ivar,R of uncorrelated flux inverse variances and the corresponding
            resolution matrix. These have shapes (nflux,) and (nflux,nflux) respectively.
            The rows of R give the resolution-convolved responses to unit flux for each
            wavelength bin. Note that R is returned as a regular (dense) numpy array but
            will normally have non-zero values concentrated near the diagonal.
    """
    #Cinv = 0.5*(Cinv + Cinv.T)
    #if scipy.sparse.issparse(Cinv): Cinv = Cinv.todense()
    chw=max(10,int(50/wavebin)) #core is 2*50+1 A
    skin=max(2,int(10/wavebin)) #skin is 10A
    nn=Cinv.shape[0]
    flux=np.zeros(Cinv.shape[0])
    ivar=np.zeros(Cinv.shape[0])
    R=np.zeros_like(Cinv)
    nstep=nn//(2*chw+1)+1
    log = get_logger()
    t0=time.time()
    Lmin=1e-15/np.mean(np.diag(Cinv)) # Lmin is scaled with Cinv values
    for c in range(chw,nn+(2*chw+1),(2*chw+1)) :
        b=max(0,c-chw-skin)
        e=min(nn,c+chw+skin+1)
        b1=max(0,c-chw)
        e1=min(nn,c+chw+1)
        bb=max(0,b1-b)
        ee=min(e-b,e1-b)
        if e<=b : continue
        L,X = scipy.linalg.eigh(Cinv[b:e,b:e],overwrite_a=False,turbo=True)
        nbad = np.count_nonzero(L < Lmin)
        if nbad > 0:
            log.warning('zeroing {0:d} negative eigenvalue(s).'.format(nbad))
            L[L < Lmin] = Lmin
        Q = X.dot(np.diag(np.sqrt(L)).dot(X.T))
        s = np.sum(Q,axis=1)

        b1x=max(0,c-chw-3)
        e1x=min(nn,c+chw+1+3)

        tR = (Q/s[:,np.newaxis])
        tR_it = scipy.linalg.inv(tR.T)
        tivar = s**2
        
        flux[b1:e1] = (tR_it.dot(Cinvf[b:e])/tivar)[bb:ee]
        ivar[b1:e1] = (s[bb:ee])**2
        R[b:e,b1:e1] = tR[:,bb:ee]
        
    t1=time.time()
    log.debug("Time for decorrelate_divide_and_conquer (block={},skin={}) = {} sec".format(2*chw+1,skin,t1-t0))
    return flux,ivar,R
    


def spectroperf_resample_spectra(spectra, wave) :

    # largely inspired by the coaddition developped by N. Busca
    
    log = get_logger()
    log.debug("Resampling to wave grid if size {}: {}".format(wave.size,wave))

    b=spectra._bands[0]
    ntarget=spectra.flux[b].shape[0]
    nwave=wave.size
    flux = np.zeros((ntarget,nwave),dtype=spectra.flux[b].dtype)
    ivar = np.zeros((ntarget,nwave),dtype=spectra.ivar[b].dtype)
    mask = np.zeros((ntarget,nwave),dtype=spectra.mask[b].dtype)
    ndiag = 5
    rdata = np.ones((ntarget,ndiag,nwave),dtype=spectra.resolution_data[b].dtype) # pointless for this resampling
    dw=np.gradient(wave)
    wavebin=np.min(dw[dw>0.]) # min wavelength bin size
    log.debug("Min wavelength bin= {:2.1f} A".format(wavebin))
    log.debug("Compute resampling matrices ...")
    RS=dict()
    for b in spectra._bands :
        twave=spectra.wave[b]
        jj=np.where((twave>=wave[0])&(twave<=wave[-1]))[0]
        twave=spectra.wave[b][jj]
        RS[b] = get_resampling_matrix(wave,twave)
    
    for i in range(ntarget) :
        log.debug("resampling {}/{}".format(i+1,ntarget))
        cinv = None
        for b in spectra._bands :
            twave=spectra.wave[b]
            jj=np.where((twave>=wave[0])&(twave<=wave[-1]))[0]
            twave=twave[jj]
            tivar=spectra.ivar[b][i][jj]
            diag_ivar = scipy.sparse.dia_matrix((tivar,[0]),(twave.size,twave.size))
            RR = Resolution(spectra.resolution_data[b][i][:,jj]).dot(RS[b])
            tcinv  = RR.T.dot(diag_ivar.dot(RR))
            tcinvf = RR.T.dot(tivar*spectra.flux[b][i][jj])
            if cinv is None :
                cinv  = tcinv
                cinvf = tcinvf
            else :
                cinv  += tcinv
                cinvf += tcinvf
        if scipy.sparse.issparse(cinv): cinv = cinv.todense()
        ii=np.arange(cinv.shape[0],dtype=int)
        keep = np.where(np.diag(cinv) > 0)[0]
        keep_t = keep[:,np.newaxis]
        R    = np.zeros_like(cinv)
        flux[i,keep],ivar[i,keep],R[keep_t,keep] = decorrelate_divide_and_conquer(cinv[keep_t,keep],cinvf[keep],wavebin=wavebin)
        
        dd=np.arange(ndiag,dtype=int)-ndiag//2
        for j in keep :
            k=(dd>=-j)&(dd<nwave-j)
            rdata[i,k,j] = R[j+dd[k],j].ravel()
            

    bands=""
    for b in spectra._bands : bands += b
    
    res=Spectra(bands=[bands,],wave={bands:wave,},flux={bands:flux,},ivar={bands:ivar,},mask={bands:mask,},resolution_data={bands:rdata,},
                fibermap=spectra.fibermap,meta=spectra.meta,extra=spectra.extra,scores=spectra.scores)
    return res


def fast_resample_spectra(spectra, wave) :

    log = get_logger()
    log.debug("Resampling to wave grid: {}".format(wave))
    
    
    nwave=wave.size
    b=spectra._bands[0]
    ntarget=spectra.flux[b].shape[0]
    nres=spectra.resolution_data[b].shape[1]
    ivar=np.zeros((ntarget,nwave),dtype=spectra.flux[b].dtype)
    flux=np.zeros((ntarget,nwave),dtype=spectra.ivar[b].dtype)
    mask=np.zeros(flux.shape,dtype=spectra.mask[b].dtype)
    rdata=np.ones((ntarget,1,nwave),dtype=spectra.resolution_data[b].dtype) # pointless for this resampling
    bands=""
    for b in spectra._bands :
        tivar=spectra.ivar[b]*(spectra.mask[b]==0)
        for i in range(ntarget) :
            ivar[i]  += resample_flux(wave,spectra.wave[b],tivar[i])
            flux[i]  += resample_flux(wave,spectra.wave[b],tivar[i]*spectra.flux[b][i])
        bands += b
    for i in range(ntarget) :
        ok=(ivar[i]>0)
        flux[i,ok]/=ivar[i,ok]    
    res=Spectra(bands=[bands,],wave={bands:wave,},flux={bands:flux,},ivar={bands:ivar,},mask={bands:mask,},resolution_data={bands:rdata,},
                fibermap=spectra.fibermap,meta=spectra.meta,extra=spectra.extra,scores=spectra.scores)
    return res
    
def resample_spectra_lin_or_log(spectra, linear_step=0, log10_step=0, fast=False, wave_min=None, wave_max=None) :

    wmin=None
    wmax=None
    for b in spectra._bands :
        if wmin is None :
            wmin=spectra.wave[b][0]
            wmax=spectra.wave[b][-1]
        else :
            wmin=min(wmin,spectra.wave[b][0])
            wmax=max(wmax,spectra.wave[b][-1])

    if wave_min is not None :
        wmin = wave_min
    if wave_max is not None :
        wmax = wave_max

    if linear_step>0 :
        nsteps=int((wmax-wmin)/linear_step) + 1
        wave=wmin+np.arange(nsteps)*linear_step
    elif log10_step>0 :
        lwmin=np.log10(wmin)
        lwmax=np.log10(wmax)
        nsteps=int((lwmax-lwmin)/log10_step) + 1
        wave=10**(lwmin+np.arange(nsteps)*log10_step)
    if fast :
        return fast_resample_spectra(spectra=spectra,wave=wave)
    else :
        return spectroperf_resample_spectra(spectra=spectra,wave=wave)
    
def main(args=None):

    log = get_logger()

    if args is None:
        args = parse()

    if args.resample_linear_step is not None and args.resample_log10_step is not None :
        print("cannot have both linear and logarthmic bins :-), choose either --resample-linear-step or --resample-log10-step")
        return 12
    
    spectra = read_spectra(args.infile)

    coadd(spectra,cosmics_nsig=args.nsig)

    if args.resample_linear_step is not None :
        spectra = resample_spectra_lin_or_log(spectra, linear_step=args.resample_linear_step, wave_min =args.wave_min, spectro_perf = args.spectro_perf)
    if args.resample_log10_step is not None :
        spectra = resample_spectra_lin_or_log(spectra, log10_step=args.resample_log10_step, wave_min =args.wave_min, spectro_perf = args.spectro_perf)
    
    log.debug("writing {} ...".format(args.outfile))
    write_spectra(args.outfile,spectra)
