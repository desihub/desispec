

"""
Get the normalized best template to do flux calibration.

desi_fit_stdstars.py
    --indir INDIR
    --fiberflat FILENAME
    --models STDSTAR_MODELS
    --fibermapdir FMDIR
    --spectrograph N
    --outfile X
"""

#- TODO: refactor algorithmic code into a separate module/function

import argparse
import os
import sys

import numpy as np
from astropy.io import fits

from desispec import io
from desispec.fluxcalibration import match_templates,normalize_templates
from desispec.interpolation import resample_flux
from desispec.log import get_logger
from desispec.pipeline.utils import default_nproc


def parse(options=None):
    parser = argparse.ArgumentParser(description="Extract spectra from pre-processed raw data.")
    parser.add_argument('--frames', type = str, default = None, required=True, nargs='*', 
                        help = 'list of path to DESI frame fits files (needs to be same exposure, spectro)')
    parser.add_argument('--skymodels', type = str, default = None, required=True, nargs='*', 
                        help = 'list of path to DESI sky model fits files (needs to be same exposure, spectro)')
    parser.add_argument('--fiberflats', type = str, default = None, required=True, nargs='*', 
                        help = 'list of path to DESI fiberflats fits files (needs to be same exposure, spectro)')
    parser.add_argument('--starmodels', type = str, help = 'path of spectro-photometric stellar spectra fits')
    parser.add_argument('-o','--outfile', type = str, help = 'output file for normalized stdstar model flux')
    parser.add_argument('--ncpu', type = int, default = default_nproc, required = False, help = 'use ncpu for multiprocessing')
    
    args = None
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args

def safe_read_key(header,key) :
    value = None
    try :
        value=header[key]
    except KeyError :
        value = None
        pass
    if value is None : # second try
        value=header[key.ljust(8).upper()]
    return value
        


def main(args) :
    """ finds the best models of all standard stars in the frame
    and normlize the model flux. Output is written to a file and will be called for calibration.
    """

    log = get_logger()
    
    
    frames={}
    flats={}
    skies={}
    
    spectrograph=None
    starfibers=None
    starindices=None
    fibermap=None
    

    # READ DATA
    ############################################
    
    for filename in args.frames :
        
        log.info("reading %s"%filename)
        frame=io.read_frame(filename)
        header=fits.getheader(filename, 0)
        frame_fibermap,junk=io.read_fibermap(filename, header=True)
        frame_starindices=np.where(frame_fibermap["OBJTYPE"]=="STD")[0] 
        camera=safe_read_key(header,"CAMERA").strip().lower()
        
        if spectrograph is None :
            spectrograph = frame.spectrograph
            fibermap = frame_fibermap            
            starindices=frame_starindices
            starfibers=fibermap["FIBER"][starindices]
            
        elif spectrograph != frame.spectrograph :
            log.error("incompatible spectrographs %d != %d"%(spectrograph,frame.spectrograph))
            raise ValueError("incompatible spectrographs %d != %d"%(spectrograph,frame.spectrograph))
        elif starindices.size != frame_starindices.size or np.sum(starindices!=frame_starindices)>0 :
            log.error("incompatible fibermap")
            raise ValueError("incompatible fibermap")
        
        if frames.has_key(camera) :
            log.error("cannot handle for now several frame of same camera (%s)"%camera)
            raise ValueError("cannot handle for now several frame of same camera (%s)"%camera)
            
        frames[camera]=frame
        
    for filename in args.skymodels :
        log.info("reading %s"%filename)
        sky=io.read_sky(filename)
        header=fits.getheader(filename, 0)
        camera=safe_read_key(header,"CAMERA").strip().lower()
                
        # NEED TO ADD MORE CHECKS
        if skies.has_key(camera) :
            log.error("cannot handle several skymodels of same camera (%s)"%camera)
            raise ValueError("cannot handle several skymodels of same camera (%s)"%camera)
            
        
        skies[camera]=sky
    
    for filename in args.fiberflats :
        log.info("reading %s"%filename)
        header=fits.getheader(filename, 0)
        flat=io.read_fiberflat(filename)        
        camera=safe_read_key(header,"CAMERA").strip().lower()
        
        # NEED TO ADD MORE CHECKS
        if flats.has_key(camera) :
            log.error("cannot handle several flats of same camera (%s)"%camera)
            raise ValueError("cannot handle several flats of same camera (%s)"%camera)
        flats[camera]=flat
    

    if starindices.size == 0 :
        log.error("no STD star found in fibermap")
        raise ValueError("no STD star found in fibermap")
    
    log.info("found %d STD stars"%starindices.size)
    
    imaging_filters=fibermap["FILTER"][starindices]
    imaging_mags=fibermap["MAG"][starindices]
    
    # DIVIDE FLAT AND SUBTRACT SKY , TRIM DATA
    ############################################     
    for cam in frames :
        
        if not skies.has_key(cam) :
            log.warning("Missing sky for %s"%cam)
            frames.pop(cam)
            continue
        if not flats.has_key(cam) :
            log.warning("Missing flat for %s"%cam)
            frames.pop(cam)
            continue
        
        frames[cam].flux = frames[cam].flux[starindices]
        frames[cam].ivar = frames[cam].ivar[starindices]
        

        frames[cam].ivar *= (frames[cam].mask[starindices] == 0)
        frames[cam].ivar *= (skies[cam].ivar[starindices] != 0)
        frames[cam].ivar *= (skies[cam].mask[starindices] == 0)
        frames[cam].ivar *= (flats[cam].ivar[starindices] != 0)
        frames[cam].ivar *= (flats[cam].mask[starindices] == 0)
        frames[cam].flux *= ( frames[cam].ivar > 0) # just for clean plots
        for star in range(frames[cam].flux.shape[0]) :
            ok=np.where((frames[cam].ivar[star]>0)&(flats[cam].fiberflat[star]!=0))[0]
            if ok.size > 0 :
                frames[cam].flux[star] = frames[cam].flux[star]/flats[cam].fiberflat[star] - skies[cam].flux[star]
    nstars = starindices.size
    starindices=None # we don't need this anymore
    
    # READ MODELS
    ############################################   
    log.info("reading star models in %s"%args.starmodels)
    stdwave,stdflux,templateid,teff,logg,feh=io.read_stdstar_templates(args.starmodels)
    
    if 0 :
        # we don't need an infinite resolution even for z scanning here
        # nor the full wavelength range
        # but we cannot at this stage map directly the model on the extraction grid because 
        # of the redshifts
        minwave  = 100000.
        maxwave  = 0.
        mindwave = 1000. # wave step
        for cam in frames :
            minwave=min(minwave,np.min(frames[cam].wave))
            maxwave=max(maxwave,np.max(frames[cam].wave))
            mindwave=min(mindwave,np.min(np.gradient(frames[cam].wave)))
        z_max = 0.01 # that's a huge velocity of 3000 km/s
        minwave/=(1+z_max)
        maxwave*=(1+z_max)
        dwave=mindwave/2. # that's good enough
        resampled_stdwave=minwave+dwave*np.arange(int((maxwave-minwave)/dwave))
        log.info("first resampling of the standard star models (to go faster later) ...")
        resampled_stdflux=np.zeros((stdflux.shape[0],resampled_stdwave.size))
        for i in range(stdflux.shape[0]) :
            resampled_stdflux[i]=resample_flux(resampled_stdwave,stdwave,stdflux[i])

    # LOOP ON STARS TO FIND BEST MODEL
    ############################################
    bestModelIndex=np.arange(nstars)
    templateID=np.arange(nstars)
    chi2dof=np.zeros((nstars))
    redshift=np.zeros((nstars))
    normflux=[]
    
    for star in range(nstars) :
        
        log.info("finding best model for observed star #%d"%star)
        
        # np.array of wave,flux,ivar,resol
        wave = {}
        flux = {}
        ivar = {}
        resolution_data = {}
        for camera in frames :
            band=camera[0]
            wave[band]=frames[camera].wave
            flux[band]=frames[camera].flux[star]
            ivar[band]=frames[camera].ivar[star]
            resolution_data[band]=frames[camera].resolution_data[star]
        
        
        bestModelIndex[star],redshift[star],chi2dof[star]=match_templates(wave,flux,ivar,resolution_data,stdwave,stdflux, teff, logg, feh,ncpu=args.ncpu)
        
        log.info('Star Fiber: {0}; TemplateID: {1}; Redshift: {2}; Chisq/dof: {3}'.format(starfibers[star],bestModelIndex[star],redshift[star],chi2dof[star]))
        # Apply redshift to original spectrum at full resolution
        tmp=np.interp(stdwave,stdwave/(1+redshift[star]),stdflux[bestModelIndex[star]])
        # Normalize the best model using reported magnitude
        normalizedflux=normalize_templates(stdwave,tmp,imaging_mags[star],imaging_filters[star])
        normflux.append(normalizedflux)
    
    
    # Now write the normalized flux for all best models to a file
    normflux=np.array(normflux)
    data={}
    data['BESTMODEL']=bestModelIndex
    data['TEMPLATEID']=bestModelIndex # IS THAT IT?
    data['CHI2DOF']=chi2dof
    data['REDSHIFT']=redshift
    norm_model_file=args.outfile
    io.write_stdstar_models(args.outfile,normflux,stdwave,starfibers,data)


