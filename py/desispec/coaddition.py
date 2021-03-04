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

import multiprocessing

from desiutil.log import get_logger

from desispec.interpolation import resample_flux
from desispec.spectra import Spectra
from desispec.resolution import Resolution
from desispec.fiberbitmasking import get_all_fiberbitmask_with_amp, get_all_nonamp_fiberbitmask_val, get_justamps_fiberbitmask
from desispec.specscore import compute_coadd_scores

def coadd_fibermap(fibermap) :

    log = get_logger()
    log.debug("'coadding' fibermap")

    targets = np.unique(fibermap["TARGETID"])
    ntarget = targets.size


    jj=np.zeros(ntarget,dtype=int)
    for i,tid in enumerate(targets) :
        jj[i]=np.where(fibermap["TARGETID"]==tid)[0][0]
    tfmap=fibermap[jj]

    #- initialize NUMEXP=-1 to check that they all got filled later
    tfmap['COADD_NUMEXP'] = np.zeros(len(tfmap), dtype=np.int16) - 1
    tfmap['COADD_EXPTIME'] = np.zeros(len(tfmap), dtype=np.float32) - 1

    # smarter values for some columns
    for k in ['DELTA_X','DELTA_Y'] :
        if k in fibermap.colnames :
            tfmap.rename_column(k,'MEAN_'+k)
            xx = Column(np.zeros(ntarget))
            tfmap.add_column(xx,name='RMS_'+k)
    for k in ['NIGHT','EXPID','TILEID','SPECTROID','FIBER'] :
        if k in fibermap.colnames :
            if not 'FIRST_'+k in tfmap.dtype.names :
                xx = Column(np.arange(ntarget))
                tfmap.add_column(xx,name='FIRST_'+k)
            if not 'LAST_'+k in tfmap.dtype.names :
                xx = Column(np.arange(ntarget))
                tfmap.add_column(xx,name='LAST_'+k)
            if not 'NUM_'+k in tfmap.dtype.names :
                xx = Column(np.arange(ntarget))
                tfmap.add_column(xx,name='NUM_'+k)

    for i,tid in enumerate(targets) :
        jj = fibermap["TARGETID"]==tid

        #- coadded FIBERSTATUS = bitwise AND of input FIBERSTATUS
        tfmap['FIBERSTATUS'][i] = np.bitwise_and.reduce(fibermap['FIBERSTATUS'][jj])

        #- Only FIBERSTATUS=0 were included in the coadd
        fiberstatus_nonamp_bits = get_all_nonamp_fiberbitmask_val()
        fiberstatus_amp_bits = get_justamps_fiberbitmask()
        targ_fibstatuses = fibermap['FIBERSTATUS'][jj]
        nonamp_fiberstatus_flagged = ( (targ_fibstatuses & fiberstatus_nonamp_bits) > 0 )
        allamps_flagged = ( (targ_fibstatuses & fiberstatus_amp_bits) == fiberstatus_amp_bits )
        good_coadds = np.bitwise_not( nonamp_fiberstatus_flagged | allamps_flagged )
        tfmap['COADD_NUMEXP'][i] = np.count_nonzero(good_coadds)
        if 'EXPTIME' in fibermap.colnames :
            tfmap['COADD_EXPTIME'][i] = np.sum(fibermap['EXPTIME'][jj][good_coadds])
        for k in ['DELTA_X','DELTA_Y'] :
            if k in fibermap.colnames :
                vals=fibermap[k][jj]
                tfmap['MEAN_'+k][i] = np.mean(vals)
                tfmap['RMS_'+k][i] = np.sqrt(np.mean(vals**2)) # inc. mean offset, not same as std

        for k in ['NIGHT','EXPID','TILEID','SPECTROID','FIBER'] :
            if k in fibermap.colnames :
                vals=fibermap[k][jj]
                tfmap['FIRST_'+k][i] = np.min(vals)
                tfmap['LAST_'+k][i] = np.max(vals)
                tfmap['NUM_'+k][i] = np.unique(vals).size
        for k in ['FIBERASSIGN_X', 'FIBERASSIGN_Y','FIBER_RA', 'FIBER_DEC'] :
            if k in fibermap.colnames :
                tfmap[k][i]=np.mean(fibermap[k][jj])
        for k in ['FIBER_RA_IVAR', 'FIBER_DEC_IVAR','DELTA_X_IVAR', 'DELTA_Y_IVAR'] :
            if k in fibermap.colnames :
                tfmap[k][i]=np.sum(fibermap[k][jj])

    return tfmap

def coadd(spectra, cosmics_nsig=0.) :
    """
    Coaddition the spectra for each target and each camera. The input spectra is modified.

    Args:
       spectra: desispec.spectra.Spectra object

    Options:
       cosmics_nsig: float, nsigma clipping threshold for cosmics rays
    """
    log = get_logger()
    targets = np.unique(spectra.fibermap["TARGETID"])
    ntarget=targets.size
    log.debug("number of targets= {}".format(ntarget))
    for b in spectra.bands :
        log.debug("coadding band '{}'".format(b))
        nwave=spectra.wave[b].size
        tflux=np.zeros((ntarget,nwave),dtype=spectra.flux[b].dtype)
        tivar=np.zeros((ntarget,nwave),dtype=spectra.ivar[b].dtype)
        if spectra.mask is not None :
            tmask=np.zeros((ntarget,nwave),dtype=spectra.mask[b].dtype)
        else :
            tmask=None
        trdata=np.zeros((ntarget,spectra.resolution_data[b].shape[1],nwave),dtype=spectra.resolution_data[b].dtype)

        fiberstatus_bits = get_all_fiberbitmask_with_amp(b)
        good_fiberstatus = ( (spectra.fibermap["FIBERSTATUS"] & fiberstatus_bits) == 0 )
        for i,tid in enumerate(targets) :
            jj=np.where( (spectra.fibermap["TARGETID"]==tid) & good_fiberstatus )[0]

            #- if all spectra were flagged as bad (FIBERSTATUS != 0), contine
            #- to next target, leaving tflux and tivar=0 for this target
            if len(jj) == 0:
                continue

            if cosmics_nsig is not None and cosmics_nsig > 0  and len(jj)>2 :
                # interpolate over bad measurements
                # to be able to compute gradient next
                # to a bad pixel and identify outlier
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
                    bad  = ~good
                    if np.sum(good)==0 :
                        continue
                    nbad = np.sum(bad)
                    ttflux = spectra.flux[b][j].copy()
                    if nbad>0 :
                        ttflux[bad] = np.interp(spectra.wave[b][bad],spectra.wave[b][good],ttflux[good])
                    ttivar = spectra.ivar[b][j].copy()
                    if nbad>0 :
                        ttivar[bad] = np.interp(spectra.wave[b][bad],spectra.wave[b][good],ttivar[good])
                    ttvar = 1./(ttivar+(ttivar==0))
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
            if cosmics_nsig is not None and cosmics_nsig > 0 and len(jj)>2  :
                grad=np.array(grad)
                gradvar=np.array(gradvar)
                gradivar=(gradvar>0)/np.array(gradvar+(gradvar==0))
                nspec=grad.shape[0]
                sgradivar=np.sum(gradivar)
                if sgradivar>0 :
                    meangrad=np.sum(gradivar*grad,axis=0)/sgradivar
                    deltagrad=grad-meangrad
                    chi2=np.sum(gradivar*deltagrad**2,axis=0)/(nspec-1)

                    bad  = (chi2>cosmics_nsig**2)
                    nbad = np.sum(bad)
                    if nbad>0 :
                        log.info("masking {} values for targetid={}".format(nbad,tid))
                        badindex=np.where(bad)[0]
                        for bi in badindex  :
                            k=np.argmax(gradivar[:,bi]*deltagrad[:,bi]**2)
                            ivarjj[k,bi]=0.
                            log.debug("masking spec {} wave={}".format(k,spectra.wave[b][bi]))

            tivar[i]=np.sum(ivarjj,axis=0)
            tflux[i]=np.sum(ivarjj*spectra.flux[b][jj],axis=0)
            for r in range(spectra.resolution_data[b].shape[1]) :
                trdata[i,r]=np.sum((spectra.ivar[b][jj]*spectra.resolution_data[b][jj,r]),axis=0) # not sure applying mask is wise here
            bad=(tivar[i]==0)
            if np.sum(bad)>0 :
                tivar[i][bad] = np.sum(spectra.ivar[b][jj][:,bad],axis=0) # if all masked, keep original ivar
                tflux[i][bad] = np.sum(spectra.ivar[b][jj][:,bad]*spectra.flux[b][jj][:,bad],axis=0)
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

    if spectra.scores is not None:
        orig_scores = spectra.scores.copy()
        orig_scores['TARGETID'] = spectra.fibermap['TARGETID']
    else:
        orig_scores = None

    spectra.fibermap=coadd_fibermap(spectra.fibermap)
    spectra.scores=None
    compute_coadd_scores(spectra, orig_scores, update_coadd=True)

def coadd_cameras(spectra,cosmics_nsig=0.) :

    #check_alignement_of_camera_wavelength(spectra)

    log = get_logger()

    # ordering
    mwave=[np.mean(spectra.wave[b]) for b in spectra.bands]
    sbands=np.array(spectra.bands)[np.argsort(mwave)] # bands sorted by inc. wavelength
    log.debug("wavelength sorted cameras= {}".format(sbands))

    # create wavelength array
    wave=None
    tolerance=0.0001 #A , tolerance
    for b in sbands :
        if wave is None :
            wave=spectra.wave[b]
        else :
            wave=np.append(wave,spectra.wave[b][spectra.wave[b]>wave[-1]+tolerance])
    nwave=wave.size

    # check alignment
    number_of_overlapping_cameras=np.zeros(nwave)
    for b in spectra.bands :
        windices=np.argmin((np.tile(wave,(spectra.wave[b].size,1))-np.tile(spectra.wave[b],(wave.size,1)).T)**2,axis=1)
        dist=np.sqrt(np.max(spectra.wave[b] - wave[windices]))
        log.debug("camera {} max dist= {}A".format(b,dist))
        if dist > tolerance :
            log.error("Cannot directly coadd the camera spectra because wavelength are not aligned, use --lin-step or --log10-step to resample to a common grid")
            sys.exit(12)
        number_of_overlapping_cameras[windices] += 1

    # targets
    targets = np.unique(spectra.fibermap["TARGETID"])
    ntarget=targets.size
    log.debug("number of targets= {}".format(ntarget))


    # ndiag = max of all cameras
    ndiag=0
    for b in sbands :
        ndiag=max(ndiag,spectra.resolution_data[b].shape[1])
    log.debug("ndiag= {}".format(ndiag))


    b = sbands[0]
    flux=np.zeros((ntarget,nwave),dtype=spectra.flux[b].dtype)
    ivar=np.zeros((ntarget,nwave),dtype=spectra.ivar[b].dtype)
    if spectra.mask is not None :
        ivar_unmasked=np.zeros((ntarget,nwave),dtype=spectra.ivar[b].dtype)
        mask=np.zeros((ntarget,nwave),dtype=spectra.mask[b].dtype)
    else :
        ivar_unmasked=ivar
        mask=None

    rdata=np.zeros((ntarget,ndiag,nwave),dtype=spectra.resolution_data[b].dtype)

    for b in spectra.bands :
        log.debug("coadding band '{}'".format(b))

        # indices
        windices=np.argmin((np.tile(wave,(spectra.wave[b].size,1))-np.tile(spectra.wave[b],(wave.size,1)).T)**2,axis=1)

        band_ndiag = spectra.resolution_data[b].shape[1]

        fiberstatus_bits = get_all_fiberbitmask_with_amp(b)
        good_fiberstatus = ( (spectra.fibermap["FIBERSTATUS"] & fiberstatus_bits) == 0 )
        for i,tid in enumerate(targets) :
            jj=np.where( (spectra.fibermap["TARGETID"]==tid) & good_fiberstatus )[0]

            #- if all spectra were flagged as bad (FIBERSTATUS != 0), contine
            #- to next target, leaving tflux and tivar=0 for this target
            if len(jj) == 0:
                continue

            if cosmics_nsig is not None and cosmics_nsig > 0 and len(jj)>2 :
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
                    bad  = ~good
                    ttflux = spectra.flux[b][j].copy()
                    ttflux[bad] = np.interp(spectra.wave[b][bad],spectra.wave[b][good],ttflux[good])
                    ttivar = spectra.ivar[b][j].copy()
                    ttivar[bad] = np.interp(spectra.wave[b][bad],spectra.wave[b][good],ttivar[good])
                    ttvar = 1./(ttivar+(ttivar==0))
                    ttflux[1:] = ttflux[1:]-ttflux[:-1]
                    ttvar[1:]  = ttvar[1:]+ttvar[:-1]
                    ttflux[0]  = 0
                    grad.append(ttflux)
                    gradvar.append(ttvar)

            ivar_unmasked[i,windices] += np.sum(spectra.ivar[b][jj],axis=0)

            if spectra.mask is not None :
                ivarjj=spectra.ivar[b][jj]*(spectra.mask[b][jj]==0)
            else :
                ivarjj=spectra.ivar[b][jj]

            if cosmics_nsig is not None and cosmics_nsig > 0 and len(jj)>2  :
                grad=np.array(grad)
                gradivar=1/np.array(gradvar)
                nspec=grad.shape[0]
                meangrad=np.sum(gradivar*grad,axis=0)/np.sum(gradivar)
                deltagrad=grad-meangrad
                chi2=np.sum(gradivar*deltagrad**2,axis=0)/(nspec-1)
                bad  = (chi2>cosmics_nsig**2)
                nbad = np.sum(bad)
                if nbad>0 :
                    log.info("masking {} values for targetid={}".format(nbad,tid))
                    badindex=np.where(bad)[0]
                    for bi in badindex  :
                        k=np.argmax(gradivar[:,bi]*deltagrad[:,bi]**2)
                        ivarjj[k,bi]=0.
                        log.debug("masking spec {} wave={}".format(k,spectra.wave[b][bi]))

            ivar[i,windices] += np.sum(ivarjj,axis=0)
            flux[i,windices] += np.sum(ivarjj*spectra.flux[b][jj],axis=0)
            for r in range(band_ndiag) :
                rdata[i,r+(ndiag-band_ndiag)//2,windices] += np.sum((spectra.ivar[b][jj]*spectra.resolution_data[b][jj,r]),axis=0)
            if spectra.mask is not None :
                # this deserves some attention ...

                tmpmask=np.bitwise_and.reduce(spectra.mask[b][jj],axis=0)

                # directly copy mask where no overlap
                jj=(number_of_overlapping_cameras[windices]==1)
                mask[i,windices[jj]] = tmpmask[jj]

                # 'and' in overlapping regions
                jj=(number_of_overlapping_cameras[windices]>1)
                mask[i,windices[jj]] = mask[i,windices[jj]] & tmpmask[jj]


    for i,tid in enumerate(targets) :
        ok=(ivar[i]>0)
        if np.sum(ok)>0 :
            flux[i][ok] /= ivar[i][ok]
        ok=(ivar_unmasked[i]>0)
        if np.sum(ok)>0 :
            rdata[i][:,ok] /= ivar_unmasked[i][ok]

    if 'COADD_NUMEXP' in spectra.fibermap.colnames:
        fibermap = spectra.fibermap
    else:
        fibermap = coadd_fibermap(spectra.fibermap)

    bands=""
    for b in sbands :
        bands+=b

    if spectra.mask is not None :
        dmask={bands:mask,}
    else :
        dmask=None

    res=Spectra(bands=[bands,],wave={bands:wave,},flux={bands:flux,},ivar={bands:ivar,},mask=dmask,resolution_data={bands:rdata,},
                fibermap=fibermap,meta=spectra.meta,extra=spectra.extra,scores=None)

    if spectra.scores is not None:
        orig_scores = spectra.scores.copy()
        orig_scores['TARGETID'] = spectra.fibermap['TARGETID']
    else:
        orig_scores = None

    compute_coadd_scores(res, orig_scores, update_coadd=True)

    return res

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



def decorrelate_divide_and_conquer(Cinv,Cinvf,wavebin,flux,ivar,rdata) :
    """Decorrelate an inverse covariance using the matrix square root.

    Implements the decorrelation part of the spectroperfectionism algorithm described in
    Bolton & Schlegel 2009 (BS) http://arxiv.org/abs/0911.2689.

    with the divide and conquer approach, i.e. per diagonal block of the matrix, with an
    overlapping 'skin' from one block to another.

    Args:
        Cinv: Square 2D array: input inverse covariance matrix
        Cinvf: 1D array: input
        wavebin: minimal size of wavelength bin in A, used to define the core and skin size
        flux: 1D array: output flux (has to be allocated)
        ivar: 1D array: output flux inverse variance (has to be allocated)
        rdata: 2D array: output resolution matrix per diagonal (has to be allocated)
    """

    chw=max(10,int(50/wavebin)) #core is 2*50+1 A
    skin=max(2,int(10/wavebin)) #skin is 10A
    nn=Cinv.shape[0]
    nstep=nn//(2*chw+1)+1
    Lmin=1e-15/np.mean(np.diag(Cinv)) # Lmin is scaled with Cinv values
    ndiag=rdata.shape[0]
    dd=np.arange(ndiag,dtype=int)-ndiag//2

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
        for j in range(b1,e1) :
            k=(dd>=-j)&(dd<nn-j)
            # k is the diagonal index
            # j is the wavelength index
            # it could be the transposed, I am following what it is specter.ex2d, L209
            rdata[k,j] = tR[j-b+dd[k],j-b]

def spectroperf_resample_spectrum_singleproc(spectra,target_index,wave,wavebin,resampling_matrix,ndiag,flux,ivar,rdata) :
    cinv = None
    for b in spectra.bands :
        twave=spectra.wave[b]
        jj=(twave>=wave[0])&(twave<=wave[-1])
        twave=twave[jj]
        tivar=spectra.ivar[b][target_index][jj]
        diag_ivar = scipy.sparse.dia_matrix((tivar,[0]),(twave.size,twave.size))
        RR = Resolution(spectra.resolution_data[b][target_index][:,jj]).dot(resampling_matrix[b])
        tcinv  = RR.T.dot(diag_ivar.dot(RR))
        tcinvf = RR.T.dot(tivar*spectra.flux[b][target_index][jj])
        if cinv is None :
            cinv  = tcinv
            cinvf = tcinvf
        else :
            cinv  += tcinv
            cinvf += tcinvf
    cinv = cinv.todense()
    decorrelate_divide_and_conquer(cinv,cinvf,wavebin,flux[target_index],ivar[target_index],rdata[target_index])

# for multiprocessing, with shared memory buffers
def spectroperf_resample_spectrum_multiproc(shm_in_wave,shm_in_flux,shm_in_ivar,shm_in_rdata,in_nwave,in_ndiag,in_bands,target_indices,wave,wavebin,resampling_matrix,ndiag,ntarget,shm_flux,shm_ivar,shm_rdata) :

    nwave = wave.size

    # manipulate shared memory as np arrays

    # input shared memory
    in_wave = list()
    in_flux = list()
    in_ivar  = list()
    in_rdata  = list()

    nbands = len(shm_in_wave)
    for b in range(nbands) :
        in_wave.append( np.array(shm_in_wave[b],copy=False).reshape(in_nwave[b]) )
        in_flux.append( np.array(shm_in_flux[b],copy=False).reshape((ntarget,in_nwave[b])) )
        in_ivar.append( np.array(shm_in_ivar[b],copy=False).reshape((ntarget,in_nwave[b])) )
        in_rdata.append( np.array(shm_in_rdata[b],copy=False).reshape((ntarget,in_ndiag[b],in_nwave[b])) )


    # output shared memory

    flux  = np.array(shm_flux,copy=False).reshape(ntarget,nwave)
    ivar  = np.array(shm_ivar,copy=False).reshape(ntarget,nwave)
    rdata = np.array(shm_rdata,copy=False).reshape(ntarget,ndiag,nwave)

    for target_index in target_indices :

        cinv = None
        for b in range(nbands) :
            twave=in_wave[b]
            jj=(twave>=wave[0])&(twave<=wave[-1])
            twave=twave[jj]
            tivar=in_ivar[b][target_index][jj]
            diag_ivar = scipy.sparse.dia_matrix((tivar,[0]),(twave.size,twave.size))
            RR = Resolution(in_rdata[b][target_index][:,jj]).dot(resampling_matrix[in_bands[b]])
            tcinv  = RR.T.dot(diag_ivar.dot(RR))
            tcinvf = RR.T.dot(tivar*in_flux[b][target_index][jj])
            if cinv is None :
                cinv  = tcinv
                cinvf = tcinvf
            else :
                cinv  += tcinv
                cinvf += tcinvf
        cinv = cinv.todense()
        decorrelate_divide_and_conquer(cinv,cinvf,wavebin,flux[target_index],ivar[target_index],rdata[target_index])


def spectroperf_resample_spectra(spectra, wave, nproc=1) :
    """
    Resampling of spectra file using the spectrophotometic approach

    Args:
       spectra: desispec.spectra.Spectra object
       wave: 1D numy array with new wavelenght grid

    Returns:
       desispec.spectra.Spectra object
    """

    log = get_logger()
    log.debug("resampling to wave grid of size {}: {}".format(wave.size,wave))

    b=spectra.bands[0]
    ntarget=spectra.flux[b].shape[0]
    nwave=wave.size

    if spectra.mask is not None :
        mask = np.zeros((ntarget,nwave),dtype=spectra.mask[b].dtype)
    else :
        mask = None
    # number of diagonals is the max of the number of diagonals in the
    # input spectra cameras
    ndiag = 0
    for b in spectra.bands :
        ndiag = max(ndiag,spectra.resolution_data[b].shape[1])


    dw=np.gradient(wave)
    wavebin=np.min(dw[dw>0.]) # min wavelength bin size
    log.debug("min wavelength bin= {:2.1f} A; ndiag= {:d}".format(wavebin,ndiag))
    log.debug("compute resampling matrices")
    resampling_matrix=dict()
    for b in spectra.bands :
        twave=spectra.wave[b]
        jj=np.where((twave>=wave[0])&(twave<=wave[-1]))[0]
        twave=spectra.wave[b][jj]
        resampling_matrix[b] = get_resampling_matrix(wave,twave)


    if nproc==1 :

        # allocate array
        flux  = np.zeros((ntarget,nwave),dtype=float)
        ivar  = np.zeros((ntarget,nwave),dtype=float)
        rdata = np.zeros((ntarget,ndiag,nwave),dtype=float)

        # simply loop on targets
        for target_index in range(ntarget) :
            log.debug("resampling {}/{}".format(target_index+1,ntarget))
            t0=time.time()
            spectroperf_resample_spectrum_singleproc(spectra,target_index,wave,wavebin,resampling_matrix,ndiag,flux,ivar,rdata)
            t1=time.time()
            log.debug("done one spectrum in {} sec".format(t1-t0))
    else :

        log.debug("allocate shared memory")

        # input
        shm_in_wave = list()
        shm_in_flux = list()
        shm_in_ivar  = list()
        shm_in_rdata  = list()
        in_nwave = list()
        in_ndiag = list()
        for b in spectra.bands :
            shm_in_wave.append( multiprocessing.Array('d',spectra.wave[b],lock=False) )
            shm_in_flux.append( multiprocessing.Array('d',spectra.flux[b].ravel(),lock=False) )
            shm_in_ivar.append( multiprocessing.Array('d',spectra.ivar[b].ravel(),lock=False) )
            shm_in_rdata.append( multiprocessing.Array('d',spectra.resolution_data[b].ravel(),lock=False) )
            in_nwave.append(spectra.wave[b].size)
            in_ndiag.append(spectra.resolution_data[b].shape[1])

        # output
        shm_flux=multiprocessing.Array('d',ntarget*nwave,lock=False)
        shm_ivar=multiprocessing.Array('d',ntarget*nwave,lock=False)
        shm_rdata=multiprocessing.Array('d',ntarget*ndiag*nwave,lock=False)

        # manipulate shared memory as np arrays
        flux  = np.array(shm_flux,copy=False).reshape(ntarget,nwave)
        ivar  = np.array(shm_ivar,copy=False).reshape(ntarget,nwave)
        rdata = np.array(shm_rdata,copy=False).reshape(ntarget,ndiag,nwave)

        # split targets per process
        target_indices = np.array_split(np.arange(ntarget),nproc)

        # loop on processes
        procs=list()
        for proc_index in range(nproc) :
            log.debug("starting process #{}".format(proc_index+1))
            proc = multiprocessing.Process(target=spectroperf_resample_spectrum_multiproc,
                                           args=(shm_in_wave,shm_in_flux,shm_in_ivar,shm_in_rdata,
                                                 in_nwave,in_ndiag,spectra.bands,
                                                 target_indices[proc_index],wave,wavebin,
                                                 resampling_matrix,ndiag,ntarget,
                                                 shm_flux,shm_ivar,shm_rdata))
            proc.start()
            procs.append(proc)

        # wait for the processes to finish
        log.info("waiting for the {} processes to finish ...".format(nproc))
        for proc in procs :
            proc.join()
        log.info("all done!")

    bands=""
    for b in spectra.bands : bands += b

    if spectra.mask is not None :
        dmask={bands:mask,}
    else :
        dmask=None
    res=Spectra(bands=[bands,],wave={bands:wave,},flux={bands:flux,},ivar={bands:ivar,},mask=dmask,resolution_data={bands:rdata,},
                fibermap=spectra.fibermap,meta=spectra.meta,extra=spectra.extra,scores=spectra.scores)
    return res


def fast_resample_spectra(spectra, wave) :
    """
    Fast resampling of spectra file.
    The output resolution = Id. The neighboring
    flux bins are correlated.

    Args:
       spectra: desispec.spectra.Spectra object
       wave: 1D numy array with new wavelenght grid

    Returns:
       desispec.spectra.Spectra object, resolution data=Id
    """

    log = get_logger()
    log.debug("Resampling to wave grid: {}".format(wave))


    nwave=wave.size
    b=spectra.bands[0]
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
    for b in spectra.bands :
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

def resample_spectra_lin_or_log(spectra, linear_step=0, log10_step=0, fast=False, wave_min=None, wave_max=None, nproc=1) :
    """
    Resampling of spectra file.


    Args:
       spectra: desispec.spectra.Spectra object
       linear_step: if not null the ouput wavelenght grid will be linear with this step
       log10_step: if not null the ouput wavelenght grid will be logarthmic with this step

    Options:
       fast: simple resampling. fast but at the price of correlated output flux bins and no information on resolution
       wave_min: if set, use this min wavelength
       wave_max: if set, use this max wavelength

    Returns:
       desispec.spectra.Spectra object
    """

    wmin=None
    wmax=None
    for b in spectra.bands :
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
        return spectroperf_resample_spectra(spectra=spectra,wave=wave,nproc=nproc)
