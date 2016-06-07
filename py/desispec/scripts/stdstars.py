

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
<<<<<<< HEAD
from desispec.pipeline.utils import default_nproc
=======
from desispec.io.filters import load_filter
import argparse
import numpy as np
import os
import sys
from astropy.io import fits
from astropy import units
>>>>>>> 8b147a301c80a983fab782a43715c9ff59a33ed4


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
<<<<<<< HEAD
    parser.add_argument('--ncpu', type = int, default = default_nproc, required = False, help = 'use ncpu for multiprocessing')
=======
    parser.add_argument('--ncpu', type = int, default = 1, required = False, help = 'use ncpu')
    parser.add_argument('--delta-color', type = float, default = 0.1, required = False, help = 'max delta-color for the selection of standard stars (on top of meas. errors)')
    parser.add_argument('--color', type = str, default = "G-R", required = False, help = 'color for selection of standard stars')
    parser.add_argument('--z-max', type = float, default = 0.005, required = False, help = 'max peculiar velocity (blue/red)shift range')
    parser.add_argument('--z-res', type = float, default = 0.00005, required = False, help = 'dz grid resolution')
    
>>>>>>> 8b147a301c80a983fab782a43715c9ff59a33ed4
    
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
        
def get_color_filter_indices(filters,color_name) :
    bands=color_name.strip().split("-")
    if len(bands) != 2 :
        log.error("cannot split color name '%s' as 'mag1-mag2'"%color_name)
        raise ValueError("cannot split color name '%s' as 'mag1-mag2'"%color_name)
    
    index1 = -1
    for i,fname in enumerate(filters):
        if fname[-1]==bands[0] :
            index1=i
            break
    index2 = -1
    for i,fname in enumerate(filters):
        if fname[-1]==bands[1] :
            index2=i
            break
    return index1,index2

def get_color(mags,filters,color_name) :
    index1,index2=get_color_filter_indices(filters,color_name)
    if index1<0 or index2<0 :
        log.warning("cannot compute '%s' color from %s"%(color_name,filters))
        return 0.
    return mags[index1]-mags[index2]

def main(args) :
    """ finds the best models of all standard stars in the frame
    and normlize the model flux. Output is written to a file and will be called for calibration.
    """

    log = get_logger()
    
    log.info("mag delta %s = %f (for the pre-selection of stellar models)"%(args.color,args.delta_color))
    
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
    
    log.warning("NO MAG ERRORS IN FIBERMAP, I AM IGNORING MEASUREMENT ERRORS !!")
    log.warning("NO EXTINCTION VALUES IN FIBERMAP, I AM IGNORING THIS FOR NOW !!")
    
    
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
    
    # COMPUTE MAGS OF MODELS FOR EACH STD STAR MAG
    ############################################
    model_filters = []
    for tmp in np.unique(imaging_filters) :
        if len(tmp)>0 : # can be one empty entry
            model_filters.append(tmp)
    
    log.info("computing model mags %s"%model_filters)
    model_mags = np.zeros((stdflux.shape[0],len(model_filters)))
    fluxunits = 1e-17 * units.erg / units.s / units.cm**2 / units.Angstrom
    for index in range(len(model_filters)) :
        filter_response=load_filter(model_filters[index])
        for m in range(stdflux.shape[0]) :
            model_mags[m,index]=filter_response.get_ab_magnitude(stdflux[m]*fluxunits,stdwave)
    log.info("done computing model mags")
    
    
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
        
        # preselec models based on magnitudes
        
        # compute star color
        index1,index2=get_color_filter_indices(imaging_filters[star],args.color)
        if index1<0 or index2<0 :
            log.error("cannot compute '%s' color from %s"%(color_name,filters))
        filter1=imaging_filters[star][index1]
        filter2=imaging_filters[star][index2]        
        star_color=imaging_mags[star][index1]-imaging_mags[star][index2]
        
        # compute models color
        model_index1=-1
        model_index2=-1        
        for i,fname in enumerate(model_filters) :
            if fname==filter1 :
                model_index1=i
            elif fname==filter2 :
                model_index2=i
        
        if model_index1<0 or model_index2<0 :
            log.error("cannot compute '%s' model color from %s"%(color_name,filters))
        model_colors = model_mags[:,model_index1]-model_mags[:,model_index2]

        # selection
        selection = np.where(np.abs(model_colors-star_color)<args.delta_color)[0]
        
        log.info("star#%d fiber #%d, %s = %s-%s = %f, number of pre-selected models = %d/%d"%(star,starfibers[star],args.color,filter1,filter2,star_color,selection.size,stdflux.shape[0]))
        
        index_in_selection,redshift[star],chi2dof[star]=match_templates(wave,flux,ivar,resolution_data,stdwave,stdflux[selection], teff[selection], logg[selection], feh[selection], ncpu=args.ncpu,z_max=args.z_max,z_res=args.z_res)
        
        bestModelIndex[star] = selection[index_in_selection]

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


