#!/usr/bin/env python

"""
Flux Calibraion for Desi Spectra.

"""

from desispec.io.fibermap import read_fibermap
from desispec.io.frame import read_frame
from desispec.io.sky import read_sky
from desispec.io.fiberflat import read_fiberflat
from desispec.io.fluxcalibration import read_filter_response
from desispec.io.fluxcalibration import loadStellarModels
from desispec.fluxcalibration import match_templates
from desispec.fluxcalibration import normalize_templates
from desispec.fluxcalibration import convolveFlux
from desispec.fluxcalibration import get_calibVector
from desispec.fluxcalibration import rebinSpectra

import argparse
import astropy.io.fits as pyfits
import json,pylab,string,numpy,os,scipy,scipy.sparse,scipy.linalg
from scipy.sparse.linalg import spsolve
import scipy.interpolate
import scipy.integrate as sc_int
import scipy.constants as const
import scipy.ndimage
from pylab import *
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os,sys
from scipy.sparse import spdiags
from astropy.io import fits

def main() :

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--fibermap', type = str, default = None,help = 'path of DESI exposure frame fits file')
    parser.add_argument('--models', type = str, default = None,help = 'path of spetro-photometric stellar spectra fits') 
    parser.add_argument('--spectrograph', type = str, default = None,help = 'spectrograph number, can go 0-9') 

    args = parser.parse_args()

# Call necessary environment variables. No need if add argument to give full file path.
    if 'DESI_SPECTRO_REDUX' not in os.environ:
        raise RuntimeError('Set environment DESI_SPECTRO_REDUX. It is needed to read the needed datafiles')

    DESI_SPECTRO_REDUX=os.environ['DESI_SPECTRO_REDUX']
    PRODNAME=os.environ['PRODNAME']
    if 'DESISIM' not in os.environ:
        raise RuntimeError('Set environment DESISIM. It will be neede to read the filter transmission files for calibration')

    DESISIM=os.environ['DESISIM'] # to read the filter transmission files



# read Standard Stars from the fibermap file
# returns the Fiber id, filter names and mags for the standard stars

    fiber_tbdata,fiber_header=read_fibermap(args.fibermap)
    refStarIdx=numpy.where(fiber_tbdata["OBJTYPE"]=="STD")
    refFibers=fiber_tbdata["FIBER"][refStarIdx]
    refFilters=fiber_tbdata["FILTER"][refStarIdx]
    refMags=fiber_tbdata["MAG"]
    FIBER=refFibers
    FILTERS=refFilters
    MAGS=refMags

    fibers={"FIBER":refFibers,"FILTER":refFilters,"MAG":refMags}
    
    fiber_hdulist=pyfits.open(args.fibermap)
    NIGHT=fiber_hdulist[1].header['NIGHT']
    EXPID=fiber_hdulist[1].header['EXPID']
    filters=fibers["FILTER"]
    if 'DESISIM' not in os.environ:
        raise RuntimeError('Set environment DESISIM. Can not find filter response files')
    basepath=DESISIM+"data/"

    print basepath
    #now load all the skyfiles, framefiles, fiberflatfiles etc
    # all three channels files are simultaneously treated
    skyfile={}
    framefile={}
    fiberflatfile={}
    for i in ["b","r","z"]:
        skyfile[i]=DESI_SPECTRO_REDUX+'/'+PRODNAME+'/exposures/%s/%08d/'%(NIGHT,EXPID)+"sky-%s%s-%08d.fits"%(i,args.spectrograph,EXPID) # or give full absolute path in the arguments ???
        framefile[i]=DESI_SPECTRO_REDUX+'/'+PRODNAME+'/exposures/%s/%08d/'%(NIGHT,EXPID)+"frame-%s%s-%08d.fits"%(i,args.spectrograph,EXPID)
        fiberflatfile[i]=DESI_SPECTRO_REDUX+'/'+PRODNAME+'/exposures/%s/'%(NIGHT)+"fiberflat-%s%s-%08d.fits"%(i,args.spectrograph,1)


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


    # Convolve Sky with Detector Resolution, so as to subtract from data. Convolve for all 500 specs

    convolvedsky={"b":convolveFlux(frameWave["b"],frameResolution["b"],sky["b"]),"r":convolveFlux(frameWave["r"],frameResolution["r"],sky["r"]),"z":convolveFlux(frameWave["z"],frameResolution["z"],sky["z"])} # wave and sky are one-dimensional
    
    # Read the standard Star data and divide by flat and subtract sky

    stars=[]
    ivars=[]
    for i in [ x for x in fibers["FIBER"] if x < 500]:
        #flat and sky should have same wavelength binning as data, otherwise should be rebinned.

        stars.append((i,{"b":[frameFlux["b"][i,:]/fiberFlat["b"][i,:]-convolvedsky["b"][i],frameWave["b"]],
                         "r":[frameFlux["r"][i,:]/fiberFlat["r"][i,:]-convolvedsky["r"][i],frameWave["r"]],
                         "z":[frameFlux["z"][i,:]/fiberFlat["z"][i,:]-convolvedsky["z"][i],frameWave]},fibers["MAG"][i]))
        ivars.append((i,{"b":[frameIvar["b"][i,:]],"r":[frameIvar["r"][i,:]],"z":[frameIvar["z"][i,:]]}))


    stdwave,stdflux,templateid=loadStellarModels(args.models)
    calibVector=[]
    calibration={}
    print frameWave["b"].shape,frameWave["r"].shape,frameWave["z"].shape
    print 'No. of Standard Stars in this frame:',len(stars)

    # Now do the calibraion
    # For each star, several filters, several mags, several normalization scales, so several calibration vector. What is the strategy of writing the clibraion?

    for k,l in enumerate(stars):
        print "checking best model for star", l[0]
        
        starindex=l[0]
        mags=l[2]
        filters=fibers["FILTER"][k]
        print filters
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
        
        bestModel,bestmodelWave,bestModelFlux,chisq=match_templates(frameWave,flux,ivar,resol_star,stdwave,stdflux)
        print 'Best model for star',l[0],'is',bestModel
        print 'Chi sq/dof=',chisq
        print "calibrating for star",l[0],'and Best model',bestModel
        print "Magnitudes", l[2]
        print 'filters',filters
        modelwave,normflux=normalize_templates(stdwave,stdflux[bestModel],mags,filters,basepath)
        for i in filters:
            # this is convolved calibration
            calibration["b"]=get_calibVector(frameWave["b"],flux["b"],ivar["b"],resol_star["b"],stdwave,normflux[i])
            calibration["r"]=get_calibVector(frameWave["r"],flux["r"],ivar["r"],resol_star["r"],stdwave,normflux[i])   
            calibration["z"]=get_calibVector(frameWave["z"],flux["z"],ivar["z"],resol_star["z"],stdwave,normflux[i])
            calibVector.append((l[0],bestModel,i,calibration))
        #print calibVector   
    return calibVector
    # Now apply calibration to the all spectra and write to files.
    
    
if "__main__" in __name__:
    main()
    
