#!/usr/bin/env python

"""
Get the normalized best template to do flux calibration.

"""

from desispec.io.fibermap import read_fibermap
from desispec.io.frame import read_frame
from desispec.io.sky import read_sky
from desispec.io.fiberflat import read_fiberflat
from desispec.io.fluxcalibration import read_filter_response,loadStellarModels,write_normalized_model
from desispec.fluxcalibration import match_templates,normalize_templates,convolveFlux,rebinSpectra
import argparse
import numpy as np
import os,sys

def main() :
    """ finds the best models of all standard stars in the frame
    and normlize the model flux. Output is written to a file and will be called for calibration.
    """

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--fibermap', type = str, default = None,help = 'path of DESI exposure frame fits file')
    parser.add_argument('--models', type = str, default = None,help = 'path of spectro-photometric stellar spectra fits') 
    parser.add_argument('--spectrograph', type = str, default = 0, help = 'spectrograph number, can go 0-9') 
    parser.add_argument('--outfile', type = str, default = None,help = 'path of output file. This is file for normalized model output') 
    
    args = parser.parse_args()

    # Call necessary environment variables. No need if add argument to give full file path.
    if 'DESI_SPECTRO_REDUX' not in os.environ:
        raise RuntimeError('Set environment DESI_SPECTRO_REDUX. It is needed to read the needed datafiles')

    DESI_SPECTRO_REDUX=os.environ['DESI_SPECTRO_REDUX']
    PRODNAME=os.environ['PRODNAME']
    if 'DESISIM' not in os.environ:
        raise RuntimeError('Set environment DESISIM. It will be neede to read the filter transmission files for calibration')

    DESISIM=os.environ['DESISIM']   # to read the filter transmission files

    if args.fibermap is None or args.models is None or args.spectrograph is None or args.outfile is None:
        print('Missing something')
        parser.print_help()
        sys.exit(12)

    # read Standard Stars from the fibermap file
    # returns the Fiber id, filter names and mags for the standard stars

    fiber_tbdata,fiber_header=read_fibermap(args.fibermap)
    refStarIdx=np.where(fiber_tbdata["OBJTYPE"]=="STD")
    refFibers=fiber_tbdata["FIBER"][refStarIdx]
    refFilters=fiber_tbdata["FILTER"][refStarIdx]
    refMags=fiber_tbdata["MAG"]
    FIBER=refFibers
    FILTERS=refFilters
    MAGS=refMags

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
        skyfile[i]=DESI_SPECTRO_REDUX+'/'+PRODNAME+'/exposures/%s/%08d/'%(NIGHT,EXPID)+"sky-%s%s-%08d.fits"%(i,args.spectrograph,EXPID) # or give full absolute path in the arguments ???
        framefile[i]=DESI_SPECTRO_REDUX+'/'+PRODNAME+'/exposures/%s/%08d/'%(NIGHT,EXPID)+"frame-%s%s-%08d.fits"%(i,args.spectrograph,EXPID)
        fiberflatfile[i]=DESI_SPECTRO_REDUX+'/'+PRODNAME+'/calib2d/%s/'%(NIGHT)+"fiberflat-%s%s-%08d.fits"%(i,args.spectrograph,1)

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
    cskyflux={}
    civar={}
    skywave={}
    skyhdr={}

    for i in ["b","r","z"]:
       #arg=(night,expid,'%s%s'%(i,spectrograph))
       
       frameFlux[i],frameIvar[i],frameWave[i],frameResolution[i],framehdr[i]=read_frame(framefile[i])
       
       fiberFlat[i],ivarFlat[i],maskFlat[i],meanspecFlat[i],waveFlat[i],headerFlat[i]=read_fiberflat(fiberflatfile[i])

       sky[i],skyivar[i],skymask[i],cskyflux[i],civar[i],skywave[i],skyhdr[i]=read_sky(skyfile[i])


    # Convolve Sky with Detector Resolution, so as to subtract from data. Convolve for all 500 specs. Subtracting sky this way should be equivalent to sky_subtract

    convolvedsky={"b":convolveFlux(frameWave["b"],frameResolution["b"],sky["b"]),"r":convolveFlux(frameWave["r"],frameResolution["r"],sky["r"]),"z":convolveFlux(frameWave["z"],frameResolution["z"],sky["z"])} # wave and sky are one-dimensional
    
    # Read the standard Star data and divide by flat and subtract sky

    stars=[]
    ivars=[]
    #- Should this be "for i in fibers["FIBER"]%500:" instead?
    for i in [ x for x in fibers["FIBER"] if x < 500]:
        #flat and sky should have same wavelength binning as data, otherwise should be rebinned.

        stars.append((i,{"b":[frameFlux["b"][i]/fiberFlat["b"][i]-convolvedsky["b"][i],frameWave["b"]],
                         "r":[frameFlux["r"][i]/fiberFlat["r"][i]-convolvedsky["r"][i],frameWave["r"]],
                         "z":[frameFlux["z"][i]/fiberFlat["z"][i]-convolvedsky["z"][i],frameWave]},fibers["MAG"][i]))
        ivars.append((i,{"b":[frameIvar["b"][i]],"r":[frameIvar["r"][i,:]],"z":[frameIvar["z"][i,:]]}))


    stdwave,stdflux,templateid=loadStellarModels(args.models)

    #- Trim standard star wavelengths to just the range we need
    minwave = min([min(w) for w in frameWave.values()])
    maxwave = max([max(w) for w in frameWave.values()])
    ii = (minwave-10 < stdwave) & (stdwave < maxwave+10)
    stdwave = stdwave[ii]
    stdflux = stdflux[:, ii]

    print 'No. of Standard Stars in this frame:',len(stars)

    # Now for each star, find the best model and normalize.
    
    normflux=[]
    bestModelIndex=np.arange(len(stars))
    templateID=np.arange(len(stars))
    chi2dof=np.zeros(len(stars))

    for k,l in enumerate(stars):
        print "checking best model for star", l[0]
        
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

        print 'Star Fiber:',l[0],';','Best Model Fiber:',bestModelIndex[k],';','TemplateID:',templateid[bestModelIndex[k]],';','Chisq/dof:',chi2dof[k]

        # Normalize the best model using reported magnitude
        modelwave,normalizedflux=normalize_templates(stdwave,stdflux[bestModelIndex[k]],mags,filters,basepath)   
        normflux.append(normalizedflux)

    # Now write the normalized flux for all best models to a file
    normflux=np.array(normflux)
    p=np.where(fibers["FIBER"]<500)   #- TODO: Fix
    stdfibers=fibers["FIBER"][p]
    data={}
    data['BESTMODEL']=bestModelIndex
    data['CHI2DOF']=chi2dof
    data['TEMPLATEID']=templateid[bestModelIndex]
    norm_model_file=args.outfile
    write_normalized_model(norm_model_file,normflux,stdwave,stdfibers,data)
 
if "__main__" == __name__:
    main()
    
