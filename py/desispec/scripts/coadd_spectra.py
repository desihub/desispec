"""
Coadd spectra
"""

from __future__ import absolute_import, division, print_function
import os, sys, time

import numpy as np

from desiutil.log import get_logger

from ..io import read_spectra,write_spectra

from astropy.table import Column

from desispec.interpolation import resample_flux
from desispec.spectra import Spectra

def parse(options=None):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--infile", type=str,  help="input spectra file")
    parser.add_argument("-o","--outfile", type=str,  help="output spectra file")
    parser.add_argument("--nsig", type=float, default=None, help="nsigma rejection threshold for cosmic rays")
    parser.add_argument("--resample-linear-step", type=float, default=None, help="resampling to single linear wave array of given step in A")
    parser.add_argument("--resample-log10-step", type=float, default=None, help="resampling to single log10 wave array of given step in units of log10")
    parser.add_argument("--wave-min", type=float, default=None, help="specify the min wavelength in A (default is the min wavelength in the input spectra)")
    
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)

    return args

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

def resample_spectra(spectra, wave, spectro_perf=False) :

    log = get_logger()
    log.debug("Resampling to wave grid: {}".format(wave))
    
    if spectro_perf :
        raise RuntimeError("spectroperf resampling not implemented yet")

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
    
def resample_spectra_lin_or_log(spectra, linear_step=0, log10_step=0, spectro_perf=False, wave_min=None) :

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
    
    if linear_step>0 :
        nsteps=int((wmax-wmin)/linear_step) + 1
        wave=wmin+np.arange(nsteps)*linear_step
    elif log10_step>0 :
        lwmin=np.log10(wmin)
        lwmax=np.log10(wmax)
        nsteps=int((lwmax-lwmin)/log10_step) + 1
        wave=10**(lwmin+np.arange(nsteps)*log10_step)
        
    return resample_spectra(spectra,wave,spectro_perf)
    
    
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
        spectra = resample_spectra_lin_or_log(spectra, linear_step=args.resample_linear_step, wave_min =args.wave_min)
    if args.resample_log10_step is not None :
        spectra = resample_spectra_lin_or_log(spectra, log10_step=args.resample_log10_step, wave_min =args.wave_min)
    
    log.debug("writing {} ...".format(args.outfile))
    write_spectra(args.outfile,spectra)
