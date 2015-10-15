#!/usr/bin/env python

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
#- TODO: move filters out of desisim and remove $DESISIM dependency

from desispec import io
from desispec.fluxcalibration import match_templates,normalize_templates
from desispec.log import get_logger
import argparse
import numpy as np
import os,sys

def main() :
    """ finds the best models of all standard stars in the frame
    and normlize the model flux. Output is written to a file and will be called for calibration.
    """

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--fiberflatexpid', type = int, help = 'fiberflat exposure ID')
    parser.add_argument('--fibermap', type = str, help = 'path of fibermap file')
    parser.add_argument('--models', type = str, help = 'path of spectro-photometric stellar spectra fits')
    parser.add_argument('--spectrograph', type = int, default = 0, help = 'spectrograph number, can go 0-9')
    parser.add_argument('--outfile', type = str, help = 'output file for normalized stdstar model flux')

    args = parser.parse_args()
    log = get_logger()
    # Call necessary environment variables. No need if add argument to give full file path.
    if 'DESI_SPECTRO_REDUX' not in os.environ:
        raise RuntimeError('Set environment DESI_SPECTRO_REDUX. It is needed to read the needed datafiles')

    DESI_SPECTRO_REDUX=os.environ['DESI_SPECTRO_REDUX']
    PRODNAME=os.environ['PRODNAME']
    if 'DESISIM' not in os.environ:
        raise RuntimeError('Set environment DESISIM. It will be neede to read the filter transmission files for calibration')

    DESISIM=os.environ['DESISIM']   # to read the filter transmission files

    if args.fibermap is None or args.models is None or \
       args.spectrograph is None or args.outfile is None or \
       args.fiberflatexpid is None:
        log.critical('Missing a required argument')
        parser.print_help()
        sys.exit(12)

    # read Standard Stars from the fibermap file
    # returns the Fiber id, filter names and mags for the standard stars

    fiber_tbdata,fiber_header=io.read_fibermap(args.fibermap, header=True)

    #- Trim to just fibers on this spectrograph
    ii =  (500*args.spectrograph <= fiber_tbdata["FIBER"])
    ii &= (fiber_tbdata["FIBER"] < 500*(args.spectrograph+1))
    fiber_tbdata = fiber_tbdata[ii]

    #- Get info for the standard stars
    refStarIdx=np.where(fiber_tbdata["OBJTYPE"]=="STD")
    refFibers=fiber_tbdata["FIBER"][refStarIdx]
    refFilters=fiber_tbdata["FILTER"][refStarIdx]
    refMags=fiber_tbdata["MAG"]

    fibers={"FIBER":refFibers,"FILTER":refFilters,"MAG":refMags}

    NIGHT=fiber_header['NIGHT']
    EXPID=fiber_header['EXPID']
    filters=fibers["FILTER"]
    if 'DESISIM' not in os.environ:
        raise RuntimeError('Set environment DESISIM. Can not find filter response files')
    basepath=DESISIM+"/data/"

    #now load all the skyfiles, framefiles, fiberflatfiles etc
    # all three channels files are simultaneously treated for model fitting
    skyfile={}
    framefile={}
    fiberflatfile={}
    for i in ["b","r","z"]:
        camera = i+str(args.spectrograph)
        skyfile[i] = io.findfile('sky', NIGHT, EXPID, camera)
        framefile[i] = io.findfile('frame', NIGHT, EXPID, camera)
        fiberflatfile[i] = io.findfile('fiberflat', NIGHT, args.fiberflatexpid, camera)

    #Read Frames, Flats and Sky files
    frameFlux={}
    frameIvar={}
    frameWave={}
    frameResolution={}
    framehdr={}
    fiberFlat={}
    ivarFlat={}
    maskFlat={}
    meanspecFlat={}
    waveFlat={}
    headerFlat={}
    sky={}
    skyivar={}
    skymask={}
    skywave={}
    skyhdr={}

    for i in ["b","r","z"]:
       #arg=(night,expid,'%s%s'%(i,spectrograph))
       #- minimal code change for refactored I/O, while not taking advantage of simplified structure
       frame = io.read_frame(framefile[i])
       frameFlux[i] = frame.flux
       frameIvar[i] = frame.ivar
       frameWave[i] = frame.wave
       frameResolution[i] = frame.resolution_data
       framehdr[i] = frame.meta

       ff = io.read_fiberflat(fiberflatfile[i])
       fiberFlat[i] = ff.fiberflat
       ivarFlat[i] = ff.ivar
       maskFlat[i] = ff.mask
       meanspecFlat[i] = ff.meanspec
       waveFlat[i] = ff.wave
       headerFlat[i] = ff.header

       skymodel = io.read_sky(skyfile[i])
       sky[i] = skymodel.flux
       skyivar[i] = skymodel.ivar
       skymask[i] = skymodel.mask
       skywave[i] = skymodel.wave
       skyhdr[i] = skymodel.header

    # Convolve Sky with Detector Resolution, so as to subtract from data. Convolve for all 500 specs. Subtracting sky this way should be equivalent to sky_subtract

    convolvedsky={"b":sky["b"], "r":sky["r"], "z":sky["z"]}

    # Read the standard Star data and divide by flat and subtract sky

    stars=[]
    ivars=[]
    for i in fibers["FIBER"]:
        #flat and sky should have same wavelength binning as data, otherwise should be rebinned.

        stars.append((i,{"b":[frameFlux["b"][i]/fiberFlat["b"][i]-convolvedsky["b"][i],frameWave["b"]],
                         "r":[frameFlux["r"][i]/fiberFlat["r"][i]-convolvedsky["r"][i],frameWave["r"]],
                         "z":[frameFlux["z"][i]/fiberFlat["z"][i]-convolvedsky["z"][i],frameWave]},fibers["MAG"][i]))
        ivars.append((i,{"b":[frameIvar["b"][i]],"r":[frameIvar["r"][i,:]],"z":[frameIvar["z"][i,:]]}))


    stdwave,stdflux,templateid=io.read_stdstar_templates(args.models)

    #- Trim standard star wavelengths to just the range we need
    minwave = min([min(w) for w in frameWave.values()])
    maxwave = max([max(w) for w in frameWave.values()])
    ii = (minwave-10 < stdwave) & (stdwave < maxwave+10)
    stdwave = stdwave[ii]
    stdflux = stdflux[:, ii]

    log.info('Number of Standard Stars in this frame: {0:d}'.format(len(stars)))
    if len(stars) == 0:
        log.critical("No standard stars!  Exiting")
        sys.exit(1)

    # Now for each star, find the best model and normalize.

    normflux=[]
    bestModelIndex=np.arange(len(stars))
    templateID=np.arange(len(stars))
    chi2dof=np.zeros(len(stars))

    #- TODO: don't use 'l' as a variable name.  Can look like a '1'
    for k,l in enumerate(stars):
        log.info("checking best model for star {0}".format(l[0]))

        starindex=l[0]
        mags=l[2]
        filters=fibers["FILTER"][k]
        rflux=stars[k][1]["r"][0]
        bflux=stars[k][1]["b"][0]
        zflux=stars[k][1]["z"][0]
        flux={"b":bflux,"r":rflux,"z":zflux}

        #print ivars
        rivar=ivars[k][1]["r"][0]
        bivar=ivars[k][1]["b"][0]
        zivar=ivars[k][1]["z"][0]
        ivar={"b":bivar,"r":rivar,"z":zivar}

        resol_star={"r":frameResolution["r"][l[0]],"b":frameResolution["b"][l[0]],"z":frameResolution["z"][l[0]]}

        # Now find the best Model

        bestModelIndex[k],bestmodelWave,bestModelFlux,chi2dof[k]=match_templates(frameWave,flux,ivar,resol_star,stdwave,stdflux)

        log.info('Star Fiber: {0}; Best Model Fiber: {1}; TemplateID: {2}; Chisq/dof: {3}'.format(l[0],bestModelIndex[k],templateid[bestModelIndex[k]],chi2dof[k]))
        # Normalize the best model using reported magnitude
        modelwave,normalizedflux=normalize_templates(stdwave,stdflux[bestModelIndex[k]],mags,filters,basepath)
        normflux.append(normalizedflux)

    # Now write the normalized flux for all best models to a file
    normflux=np.array(normflux)
    stdfibers=fibers["FIBER"]
    data={}
    data['BESTMODEL']=bestModelIndex
    data['CHI2DOF']=chi2dof
    data['TEMPLATEID']=templateid[bestModelIndex]
    norm_model_file=args.outfile
    io.write_stdstar_model(norm_model_file,normflux,stdwave,stdfibers,data)

if "__main__" == __name__:
    main()
