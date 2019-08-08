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
        if spectra.mask is not None :
            tmask=np.zeros((ntarget,nwave),dtype=spectra.mask[b].dtype)
        else :
            tmask=None
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
                    if spectra.mask is not None :
                        ttivar = spectra.ivar[b][j]*(spectra.mask[b][j]==0)
                    else :
                        ttivar = spectra.ivar[b][j]
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
            if spectra.mask is not None :
                ivarjj=spectra.ivar[b][jj]*(spectra.mask[b][jj]==0)
            else :
                ivarjj=spectra.ivar[b][jj]
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
            if spectra.mask is not None :
                tmask[i]      = np.bitwise_and.reduce(spectra.mask[b][jj],axis=0)
        spectra.flux[b] = tflux
        spectra.ivar[b] = tivar
        if spectra.mask is not None :
            spectra.mask[b] = tmask
        spectra.resolution_data[b] = trdata


    log.debug("merging fibermap")
    jj=np.zeros(ntarget,dtype=int)
    for i,tid in enumerate(targets) :
        jj[i]=np.where(spectra.fibermap["TARGETID"]==tid)[0][0]
    tfmap=spectra.fibermap[jj]
    # smarter values for some columns
    for k in ['DELTA_X','DELTA_Y'] :
        if k in spectra.fibermap.colnames :
            tfmap.rename_column(k,'MEAN_'+k)
            xx = Column(np.arange(ntarget))
            tfmap.add_column(xx,name='RMS_'+k)
    for k in ['NIGHT','EXPID','TILEID','SPECTROID','FIBER'] :
        if k in spectra.fibermap.colnames :
            xx = Column(np.arange(ntarget))
            tfmap.add_column(xx,name='FIRST_'+k)
            xx = Column(np.arange(ntarget))
            tfmap.add_column(xx,name='LAST_'+k)
            xx = Column(np.arange(ntarget))
            tfmap.add_column(xx,name='NUM_'+k)

    for i,tid in enumerate(targets) :
        jj = spectra.fibermap["TARGETID"]==tid
        for k in ['DELTA_X','DELTA_Y'] :
            if k in spectra.fibermap.colnames :
                vals=spectra.fibermap[k][jj]
                tfmap['MEAN_'+k][i] = np.mean(vals)
                tfmap['RMS_'+k][i] = np.sqrt(np.mean(vals**2)) # inc. mean offset, not same as std
        for k in ['NIGHT','EXPID','TILEID','SPECTROID','FIBER'] :
            if k in spectra.fibermap.colnames :
                vals=spectra.fibermap[k][jj]
                tfmap['FIRST_'+k][i] = np.min(vals)
                tfmap['LAST_'+k][i] = np.max(vals)
                tfmap['NUM_'+k][i] = np.unique(vals).size
        for k in ['DESIGN_X', 'DESIGN_Y','FIBER_RA', 'FIBER_DEC'] :
            if k in spectra.fibermap.colnames :
                tfmap[k][i]=np.mean(spectra.fibermap[k][jj])
        for k in ['FIBER_RA_IVAR', 'FIBER_DEC_IVAR','DELTA_X_IVAR', 'DELTA_Y_IVAR'] :
            if k in spectra.fibermap.colnames :
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


    
def decorrelate_divide_and_conquer(Cinv,Cinvf,wavebin,flux,ivar,R) :
    """Decorrelate an inverse covariance using the matrix square root.

    Implements the decorrelation part of the spectroperfectionism algorithm described in
    Bolton & Schlegel 2009 (BS) http://arxiv.org/abs/0911.2689.

    Args:
        Cinv: Square 2D array: input inverse covariance matrix
        Cinvf: 1D array: input
        wavebin: minimal size of wavelength bin in A
        flux: 1D array: output flux (has to be allocated)
        ivar: 1D array: output flux inverse variance (has to be allocated)
        R: Square 2D array: output resolution matrix (has to be allocated)
    """
    
    chw=max(10,int(50/wavebin)) #core is 2*50+1 A
    skin=max(2,int(10/wavebin)) #skin is 10A
    nn=Cinv.shape[0]
    nstep=nn//(2*chw+1)+1
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
            #log.warning('zeroing {0:d} negative eigenvalue(s).'.format(nbad))
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
    
    
    


def spectroperf_resample_spectra(spectra, wave) :
    """
    docstring
    """

    log = get_logger()
    log.debug("Resampling to wave grid if size {}: {}".format(wave.size,wave))

    b=spectra._bands[0]
    ntarget=spectra.flux[b].shape[0]
    nwave=wave.size
    flux = np.zeros((ntarget,nwave),dtype=spectra.flux[b].dtype)
    ivar = np.zeros((ntarget,nwave),dtype=spectra.ivar[b].dtype)
    if spectra.mask is not None :
        mask = np.zeros((ntarget,nwave),dtype=spectra.mask[b].dtype)
    else :
        mask = None
    ndiag = 5
    rdata = np.ones((ntarget,ndiag,nwave),dtype=spectra.resolution_data[b].dtype) # pointless for this resampling
    dw=np.gradient(wave)
    wavebin=np.min(dw[dw>0.]) # min wavelength bin size
    log.debug("Min wavelength bin= {:2.1f} A".format(wavebin))
    log.debug("compute resampling matrices ...")
    RS=dict()
    for b in spectra._bands :
        twave=spectra.wave[b]
        jj=np.where((twave>=wave[0])&(twave<=wave[-1]))[0]
        twave=spectra.wave[b][jj]
        RS[b] = get_resampling_matrix(wave,twave)

    R = np.zeros((nwave,nwave))
    for i in range(ntarget) :
        
        log.debug("resampling {}/{}".format(i+1,ntarget))

        t0=time.time()
        
        cinv = None
        for b in spectra._bands :
            twave=spectra.wave[b]
            jj=(twave>=wave[0])&(twave<=wave[-1])
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

        #t1=time.time()
        #log.debug("done filling matrix in {} sec".format(t1-t0))
        
        cinv = cinv.todense()

        #t2=time.time()
        #log.debug("done densify matrix in {} sec".format(t2-t1))

        decorrelate_divide_and_conquer(cinv,cinvf,wavebin,flux[i],ivar[i],R)

        #t3=time.time()
        #log.debug("done decorrelate in {} sec".format(t3-t2))
       
        dd=np.arange(ndiag,dtype=int)-ndiag//2
        for j in range(nwave) :
            k=(dd>=-j)&(dd<nwave-j)
            rdata[i,k,j] = R[j+dd[k],j].ravel()

        t4=time.time()
        #log.debug("done resolution data in {} sec".format(t4-t3))
        log.debug("done one spectrum in {} sec".format(t4-t0))

    bands=""
    for b in spectra._bands : bands += b

    if spectra.mask is not None :
        dmask={bands:mask,}
    else :
        dmask=None
    res=Spectra(bands=[bands,],wave={bands:wave,},flux={bands:flux,},ivar={bands:ivar,},mask=dmask,resolution_data={bands:rdata,},
                fibermap=spectra.fibermap,meta=spectra.meta,extra=spectra.extra,scores=spectra.scores)
    return res


def fast_resample_spectra(spectra, wave) :
    """
    docstring
    """

    log = get_logger()
    log.debug("Resampling to wave grid: {}".format(wave))
    
    
    nwave=wave.size
    b=spectra._bands[0]
    ntarget=spectra.flux[b].shape[0]
    nres=spectra.resolution_data[b].shape[1]
    ivar=np.zeros((ntarget,nwave),dtype=spectra.flux[b].dtype)
    flux=np.zeros((ntarget,nwave),dtype=spectra.ivar[b].dtype)
    if spectra.mask is not None :
        mask = np.zeros((ntarget,nwave),dtype=spectra.mask[b].dtype)
    else :
        mask = None
    rdata=np.ones((ntarget,1,nwave),dtype=spectra.resolution_data[b].dtype) # pointless for this resampling
    bands=""
    for b in spectra._bands :
        if spectra.mask is not None :
            tivar=spectra.ivar[b]*(spectra.mask[b]==0)
        else :
            tivar=spectra.ivar[b]
        for i in range(ntarget) :
            ivar[i]  += resample_flux(wave,spectra.wave[b],tivar[i])
            flux[i]  += resample_flux(wave,spectra.wave[b],tivar[i]*spectra.flux[b][i])
        bands += b
    for i in range(ntarget) :
        ok=(ivar[i]>0)
        flux[i,ok]/=ivar[i,ok]    
    if spectra.mask is not None :
        dmask={bands:mask,}
    else :
        dmask=None
    res=Spectra(bands=[bands,],wave={bands:wave,},flux={bands:flux,},ivar={bands:ivar,},mask=dmask,resolution_data={bands:rdata,},
                fibermap=spectra.fibermap,meta=spectra.meta,extra=spectra.extra,scores=spectra.scores)
    return res
    
def resample_spectra_lin_or_log(spectra, linear_step=0, log10_step=0, fast=False, wave_min=None, wave_max=None) :
    """
    docstring
    """

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
