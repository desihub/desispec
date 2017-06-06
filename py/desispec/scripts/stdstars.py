

"""
Get the normalized best template to do flux calibration.
"""

#- TODO: refactor algorithmic code into a separate module/function

import argparse

import numpy as np
from astropy.io import fits
from astropy import units

from desispec import io
from desispec.fluxcalibration import match_templates,normalize_templates
from desispec.interpolation import resample_flux
from desiutil.log import get_logger
from desispec.parallel import default_nproc
from desispec.io.filters import load_filter
from desispec.dustextinction import ext_odonnell

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
    parser.add_argument('--delta-color', type = float, default = 0.2, required = False, help = 'max delta-color for the selection of standard stars (on top of meas. errors)')
    parser.add_argument('--color', type = str, default = "G-R", required = False, help = 'color for selection of standard stars')
    parser.add_argument('--z-max', type = float, default = 0.008, required = False, help = 'max peculiar velocity (blue/red)shift range')
    parser.add_argument('--z-res', type = float, default = 0.00002, required = False, help = 'dz grid resolution')
    parser.add_argument('--template-error', type = float, default = 0.1, required = False, help = 'fractional template error used in chi2 computation (about 10% for BOSS b1)')
    
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

def dust_transmission(wave,ebv) :
    Rv = 3.1
    extinction = ext_odonnell(wave,Rv=Rv)
    return 10**(-Rv*extinction*ebv/2.5)

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
        frame_fibermap = frame.fibermap
        frame_starindices=np.where(frame_fibermap["OBJTYPE"]=="STD")[0]
        
        # check magnitude are well defined or discard stars
        tmp=[]
        for i in frame_starindices :
            mags=frame_fibermap["MAG"][i]
            ok=np.sum((mags>0)&(mags<30))
            if np.sum((mags>0)&(mags<30)) == mags.size :
                tmp.append(i)
        frame_starindices=np.array(tmp).astype(int)
        
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

        if not camera in frames :
            frames[camera]=[]
        frames[camera].append(frame)
 
    for filename in args.skymodels :
        log.info("reading %s"%filename)
        sky=io.read_sky(filename)
        header=fits.getheader(filename, 0)
        camera=safe_read_key(header,"CAMERA").strip().lower()
        if not camera in skies :
            skies[camera]=[]
        skies[camera].append(sky)
        
    for filename in args.fiberflats :
        log.info("reading %s"%filename)
        header=fits.getheader(filename, 0)
        flat=io.read_fiberflat(filename)
        camera=safe_read_key(header,"CAMERA").strip().lower()

        # NEED TO ADD MORE CHECKS
        if camera in flats:

            log.warning("cannot handle several flats of same camera (%s), will use only the first one"%camera)
            #raise ValueError("cannot handle several flats of same camera (%s)"%camera)
        else :
            flats[camera]=flat
    

    if starindices.size == 0 :
        log.error("no STD star found in fibermap")
        raise ValueError("no STD star found in fibermap")

    log.info("found %d STD stars"%starindices.size)


    imaging_filters=fibermap["FILTER"][starindices]
    imaging_mags=fibermap["MAG"][starindices]

    log.warning("NO MAG ERRORS IN FIBERMAP, I AM IGNORING MEASUREMENT ERRORS !!")

    ebv=np.zeros(starindices.size)
    if "SFD_EBV" in fibermap.columns.names  :
        log.info("Using 'SFD_EBV' from fibermap")
        ebv=fibermap["SFD_EBV"][starindices] 
    else : 
        log.warning("NO EXTINCTION VALUES IN FIBERMAP!!")
    
    
    # DIVIDE FLAT AND SUBTRACT SKY , TRIM DATA
    ############################################
    for cam in frames :

        if not cam in skies:
            log.warning("Missing sky for %s"%cam)
            frames.pop(cam)
            continue
        if not cam in flats:
            log.warning("Missing flat for %s"%cam)
            frames.pop(cam)
            continue
        

        flat=flats[cam]
        for frame,sky in zip(frames[cam],skies[cam]) :
            frame.flux = frame.flux[starindices]
            frame.ivar = frame.ivar[starindices]
            frame.ivar *= (frame.mask[starindices] == 0)
            frame.ivar *= (sky.ivar[starindices] != 0)
            frame.ivar *= (sky.mask[starindices] == 0)
            frame.ivar *= (flat.ivar[starindices] != 0)
            frame.ivar *= (flat.mask[starindices] == 0)
            frame.flux *= ( frame.ivar > 0) # just for clean plots
            for star in range(frame.flux.shape[0]) :
                ok=np.where((frame.ivar[star]>0)&(flat.fiberflat[star]!=0))[0]
                if ok.size > 0 :
                    frame.flux[star] = frame.flux[star]/flat.fiberflat[star] - sky.flux[star]
            frame.resolution_data = frame.resolution_data[starindices]
        
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
        if model_filters[index].startswith('WISE'):
            log.warning('not computing stdstar {} mags'.format(model_filters[index]))
            continue

        filter_response=load_filter(model_filters[index])
        for m in range(stdflux.shape[0]) :
            model_mags[m,index]=filter_response.get_ab_magnitude(stdflux[m]*fluxunits,stdwave)
    log.info("done computing model mags")
    
    mean_extinction_delta_mags = None
    mean_ebv = np.mean(ebv)
    if mean_ebv > 0 :
        log.info("Compute a mean delta_color from average E(B-V) = %3.2f based on canonial model star"%mean_ebv)
        # compute a mean delta_color from mean_ebv based on canonial model star
        #######################################################################
        # will then use this color offset in the model pre-selection
        # find canonical f-type model: Teff=6000, logg=4, Fe/H=-1.5
        canonical_model = np.argmin((teff-6000.0)**2+(logg-4.0)**2+(feh+1.5)**2)
        canonical_model_mags_without_extinction = model_mags[canonical_model]
        canonical_model_mags_with_extinction    = np.zeros(canonical_model_mags_without_extinction.shape)

        canonical_model_reddened_flux = stdflux[canonical_model]*dust_transmission(stdwave,mean_ebv)                
        for index in range(len(model_filters)) :
            if model_filters[index].startswith('WISE'):
                log.warning('not computing stdstar {} mags'.format(model_filters[index]))
                continue
            filter_response=load_filter(model_filters[index])
            canonical_model_mags_with_extinction[index]=filter_response.get_ab_magnitude(canonical_model_reddened_flux*fluxunits,stdwave)

        mean_extinction_delta_mags = canonical_model_mags_with_extinction - canonical_model_mags_without_extinction
         
    

    # LOOP ON STARS TO FIND BEST MODEL
    ############################################
    linear_coefficients=np.zeros((nstars,stdflux.shape[0]))
    chi2dof=np.zeros((nstars))
    redshift=np.zeros((nstars))
    normflux=[]


    star_colors_array=np.zeros((nstars))
    model_colors_array=np.zeros((nstars))
    
    for star in range(nstars) :

        log.info("finding best model for observed star #%d"%star)

        # np.array of wave,flux,ivar,resol
        wave = {}
        flux = {}
        ivar = {}
        resolution_data = {}
        for camera in frames :

            for i,frame in enumerate(frames[camera]) :
                identifier="%s-%d"%(camera,i)
                wave[identifier]=frame.wave
                flux[identifier]=frame.flux[star]
                ivar[identifier]=frame.ivar[star]
                resolution_data[identifier]=frame.resolution_data[star]
        
        
        # preselec models based on magnitudes

        # compute star color
        index1,index2=get_color_filter_indices(imaging_filters[star],args.color)
        if index1<0 or index2<0 :
            log.error("cannot compute '%s' color from %s"%(color_name,filters))
        filter1=imaging_filters[star][index1]
        filter2=imaging_filters[star][index2]
        star_color=imaging_mags[star][index1]-imaging_mags[star][index2]
        star_colors_array[star]=star_color
        

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

        # apply extinction here
        # use the colors derived from the cannonical model with the mean ebv of the stars
        # and simply apply a scaling factor based on the ebv of this star
        # this is sufficiently precise for the broad model pre-selection we are doing here
        # the exact reddening of the star to each pre-selected model is 
        # apply afterwards
        if mean_extinction_delta_mags is not None  and mean_ebv != 0 :
            delta_color = ( mean_extinction_delta_mags[model_index1] - mean_extinction_delta_mags[model_index2] ) * ebv[star]/mean_ebv 
            model_colors += delta_color
            log.info("Apply a %s-%s color offset = %4.3f to the models for star with E(B-V)=%4.3f"%(model_filters[model_index1],model_filters[model_index2],delta_color,ebv[star]))
        # selection
        
        selection = np.abs(model_colors-star_color)<args.delta_color
        # smallest cube in parameter space including this selection (needed for interpolation)
        new_selection = (teff>=np.min(teff[selection]))&(teff<=np.max(teff[selection]))
        new_selection &= (logg>=np.min(logg[selection]))&(logg<=np.max(logg[selection]))
        new_selection &= (feh>=np.min(feh[selection]))&(feh<=np.max(feh[selection]))
        selection = np.where(new_selection)[0]
        
        
        log.info("star#%d fiber #%d, %s = %s-%s = %f, number of pre-selected models = %d/%d"%(star,starfibers[star],args.color,filter1,filter2,star_color,selection.size,stdflux.shape[0]))
        
        # apply extinction to selected_models
        dust_transmission_of_this_star = dust_transmission(stdwave,ebv[star])
        selected_reddened_stdflux = stdflux[selection]*dust_transmission_of_this_star
        
        coefficients,redshift[star],chi2dof[star]=match_templates(wave,flux,ivar,resolution_data,stdwave,selected_reddened_stdflux, teff[selection], logg[selection], feh[selection], ncpu=args.ncpu,z_max=args.z_max,z_res=args.z_res,template_error=args.template_error)
        
        linear_coefficients[star,selection] = coefficients
        
        log.info('Star Fiber: {0}; TEFF: {1}; LOGG: {2}; FEH: {3}; Redshift: {4}; Chisq/dof: {5}'.format(starfibers[star],np.inner(teff,linear_coefficients[star]),np.inner(logg,linear_coefficients[star]),np.inner(feh,linear_coefficients[star]),redshift[star],chi2dof[star]))
        

        # Apply redshift to original spectrum at full resolution
        model=np.zeros(stdwave.size)
        for i,c in enumerate(linear_coefficients[star]) :
            if c != 0 :
                model += c*np.interp(stdwave,stdwave*(1+redshift[star]),stdflux[i])

        # Apply dust extinction
        model *= dust_transmission_of_this_star
        
        # Compute final model color
        mag1=load_filter(model_filters[model_index1]).get_ab_magnitude(model*fluxunits,stdwave)
        mag2=load_filter(model_filters[model_index2]).get_ab_magnitude(model*fluxunits,stdwave)
        model_colors_array[star] = mag1-mag2        

        # Normalize the best model using reported magnitude
        normalizedflux=normalize_templates(stdwave,model,imaging_mags[star],imaging_filters[star])
        normflux.append(normalizedflux)



    # Now write the normalized flux for all best models to a file
    normflux=np.array(normflux)
    data={}
    data['LOGG']=linear_coefficients.dot(logg)
    data['TEFF']= linear_coefficients.dot(teff)
    data['FEH']= linear_coefficients.dot(feh)
    data['CHI2DOF']=chi2dof
    data['REDSHIFT']=redshift
    data['COEFF']=linear_coefficients
    data['DATA_%s'%args.color]=star_colors_array
    data['MODEL_%s'%args.color]=model_colors_array
    norm_model_file=args.outfile
    io.write_stdstar_models(args.outfile,normflux,stdwave,starfibers,data)

