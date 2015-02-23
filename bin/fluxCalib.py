#!/usr/bin/env python

"""
Flux Calibraion for Desi Spectra.
From each frame file, with the info from corresponding fiber file,
picks the standard stars, the spectrum is compared to the best model scaled to 
the star's magnitudes.
Photon Counts = calibration Vector * flux(ergs/cm^2/s/A)
This code computes such calibration vectors for each filter and 
for each Desi arm.

Returns the list of Calibration Vectors in the following format:
(233, 320, 'SDSS_I', 16.169659, {'z': array([  1.17371119e+19,   9.19475805e+18,   9.33595519e+18, ...,
         1.43126422e+19,   1.38841832e+19,   1.40583344e+19])})

233- star index in the frame
320- model index in the stellarmodelfile
SDSS-I - filter
16.17 - mag in given filter
'z' - z arm
array - Calibration Vector(counts/Flux(ergs/cm^2/s/A)

version:0.1

Sami Kama/Govinda Dhungana, Jan 2015
gdhungana_at_smu_dot_edu
"""


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


# Call necessary environment variables
if 'DESIMODEL' not in os.environ:
    raise RuntimeError('The environment variable DESIMODEL must be set.')
DESIMODEL_DIR=os.environ['DESIMODEL']
if 'DESI_SPECTRO_REDUX' not in os.environ:
    raise RuntimeError('Set environment DESI_SPECTRO_REDUX. It is needed to read the needed datafiles')

DESI_SPECTRO_REDUX=os.environ['DESI_SPECTRO_REDUX']
PRODNAME=os.environ['PRODNAME']
if 'DESISIM' not in os.environ:
    raise RuntimeError('Set environment DESISIM. It will be neede to read the filter transmission files for calibration')

DESISIM=os.environ['DESISIM'] # to read the filter transmission files

#Some global constants from scipy.constants

h=const.h
pi=const.pi
e=const.e
c=const.c
erg=const.erg
hc= h/erg*c*1.e10 #(in units of ergsA)

pdf= PdfPages('Std_stars-20150130-00000002.pdf')


def ergs2photons(flux,wave):
    return flux*wave/hc


# Rebin the spectra from old Wavelength Bins to new Wavelength Bins
# by a 5th order spline interpolation
# Valid if the function is well behaving between measurement points and 
# new points are close to old ones

def rebinSpectra(spectra,oldWaveBins,newWaveBins):
    tck=scipy.interpolate.splrep(oldWaveBins,spectra,s=0,k=5)
    specnew=scipy.interpolate.splev(newWaveBins,tck,der=0)
    return specnew


def loadStellarModels(stellarmodelfile):

    phdu=pyfits.open(stellarmodelfile)
    hdr0=phdu[0].header
    crpix1=hdr0['CRPIX1']
    crval1=hdr0['CRVAL1']
    cdelt1=hdr0['CDELT1']
    if hdr0["LOGLAM"]==1: #log bins
        wavebins=10**(crval1+cdelt1*numpy.arange(len(phdu[0].data[0])))
    else: #lin bins
        model_wave_step   = cdelt1
        model_wave_offset = (crval1-cdelt1*(crpix1-1))
        wavebins=model_wave_step*numpy.arange(n_model_wave) + model_wave_offset
    paramData=phdu[1].data
    fluxData=phdu[0].data
    #modelFlux=ergs2photons(fluxData,wavebins)# Not required, will be normalized anyway while comparison
    modelData=[]
    for i in paramData["TEMPLATEID"]:
        modelData.append((i,fluxData[i]))
    models={"WAVEBINS":wavebins,"FLUX":modelData}
    phdu.close()
    return models
    # returns a dict of Model wavelenths, and Flux, which is a list of model templateid and flux

#integrate flux over given wavelength region

def integrate(flux,wave):
    deltaWave=numpy.zeros(wave.shape)
    deltaWave[:-1]=wave[1:]-wave[:-1]
    deltaWave[-1]=deltaWave[-2]
    deltaWave=numpy.abs(deltaWave)
    return numpy.dot(flux,deltaWave)
    # returns integrated flux

def integratedFlux(flux,wave):
    return integrate(flux,wave)

def applySmoothingFilter(flux):
    return scipy.ndimage.filters.median_filter(flux,50)

# read Standard Stars from the fibermap file

def readRefStarFibers(fibermapfile):
    phdr=pyfits.open(fibermapfile)
    table=phdr["FIBERMAP"].data
    hdr=phdr["FIBERMAP"].header
    refStarIdx=numpy.where(table["OBJTYPE"]=="STD")
    refFibers=table["FIBER"][refStarIdx]
    refFilters=table["FILTER"][refStarIdx]
    refMags=table["MAG"]
    return {"FIBER":refFibers,"FILTER":refFilters,"MAG":refMags}
    # returns the Fiber id, filter names and mags for the standard stars

def readFrame(framefile):

    frameFlux=fits.getdata(framefile,'FLUX').astype(np.float64)
    frameWave=fits.getdata(framefile,'WAVELENGTH').astype(np.float64)
    frameIvar=fits.getdata(framefile,'IVAR').astype(np.float64)
    frameResolution=fits.getdata(framefile,'RESOLUTION').astype(np.float64)

    return frameFlux,frameWave,frameIvar,frameResolution


def readSky(skyfile):
    sky=fits.getdata(skyfile,'SKY').astype(np.float64)
    return sky

def readfiberFlat(fiberFlatfile):
    fiberFlat=fits.getdata(fiberFlatfile,'FIBERFLAT').astype(np.float64)
    
    return fiberFlat

# convolve flux with detector resolution
# returns the convolved flux
def convolveFlux(wave,resolution,flux):
    
    diags=np.arange(10,-11,-1)
    nwave=len(wave)
    nspec=500
    convolved=np.zeros((nspec,nwave))
    print 'resolution',resolution[1].shape
    for i in range(nspec):
       R=spdiags(resolution[i],diags,nwave,nwave)
       convolved[i]=R.dot(flux)
       
    return convolved

# apply DESI throughput efficiency to flux. should have same bins
def applyThroughput(flux,filt):
    return flux*filt


# find best model for the given Standard Star data

def compare2Models(Models,flux,wave,ivar,resolution,starindex):
   
    # Models will be normalized here. 
    # flux should be flat and sky subtracted.Also flux has to be already normalized by median filtering. Ivar should
    # have to be correctly propagated while normalizing the flux 
    # For each standard star data, this step does a Chi square minimisation and picks the best model
    # returns starindex, best model index( in the model file) and reduced Chi square for that model.
 
    maxDelta=1e100
    bestId=-1
    red_Chisq=0.0
    rchisq=0
    bchisq=0
    zchisq=0
    #print len(data["b"]),len(data["r"]),len(data["z"])
    bwave=wave["b"]
    rwave=wave["r"]
    zwave=wave["z"]
    
    bivar=ivar["b"]
    rivar=ivar["r"]
    zivar=ivar["z"]


    # For refstars no need to convolve through all 500 spectral resulution???
    # So convlove with only needed resolution at refstar spectra position.
    def convolveModel(wave,resolution,flux):   

            diags=np.arange(10,-11,-1)
            nwave=len(wave)
            convolved=np.zeros(nwave)
            #print 'resolution',resolution[1].shape
            R=spdiags(resolution,diags,nwave,nwave)
            convolved=R.dot(flux)
       
            return convolved

    
    for i in xrange(len(Models["b"])):
           
        bconvolveFlux=convolveModel(bwave,resolution["b"],Models["b"][i])
        rconvolveFlux=convolveModel(rwave,resolution["r"],Models["r"][i])
        zconvolveFlux=convolveModel(zwave,resolution["z"],Models["z"][i])

        b_models=bconvolveFlux/applySmoothingFilter(bconvolveFlux)
        r_models=rconvolveFlux/applySmoothingFilter(rconvolveFlux)
        z_models=zconvolveFlux/applySmoothingFilter(zconvolveFlux)
        
        rdelta=numpy.sum(((r_models-flux["r"])**2)*rivar)
        bdelta=numpy.sum(((b_models-flux["b"])**2)*bivar)
        zdelta=numpy.sum(((z_models-flux["z"])**2)*zivar)
        #print i, (rdelta+bdelta+zdelta)/(len(bwave)+len(rwave)+len(zwave))
        if (rdelta+bdelta+zdelta)<maxDelta:
                bestmodel={"r":r_models,"b":b_models,"z":z_models}
                bestId=i
                maxDelta=(rdelta+bdelta+zdelta)
                bchisq=bdelta 
                rchisq=rdelta 
                zchisq=zdelta 
                dof=len(bwave)+len(rwave)+len(zwave)
                red_Chisq=maxDelta/dof
                
    #Plot the data and best model matching
    #fig=plt.figure()
    #frame1=fig.add_axes((.1,.3,.8,.6))
    #plt.plot(bwave,flux["b"],label="db")
    #plt.plot(rwave,flux["r"],label="dr")
    #plt.plot(zwave,flux["z"],label="dz")
    #plt.plot(bwave,bestmodel["b"],label="mb")
    #plt.plot(rwave,bestmodel["r"],label="mr")
    #plt.plot(zwave,bestmodel["z"],label="mz")
    #plt.legend(loc='upper left',ncol=2,fancybox=True,shadow=True)
    #plt.title("StarIndex: %s  Best Model ID: %s"%(starindex,bestId))
    #plt.show()
    # Residual:
    #frame2=fig.add_axes((.1,.1,.8,.2))
    #plt.plot(bwave,flux["b"]-bestmodel["b"])
    #plt.plot(rwave,flux["r"]-bestmodel["r"])
    #plt.plot(zwave,flux["z"]-bestmodel["z"])
    #plt.show()
    #fig.savefig('fig-%s-04_res.png'%starindex)
    #plt.close()  
    #print starindex,bestId,dof,bchisq,rchisq,zchisq,red_Chisq

    return starindex,bestId,red_Chisq

# Read filter transmission. Needed for flux calibration after model fitting.
# Currently Reading from DESISIM/data
# return a dict of tuple composed of wavelengths, efficiency and spline parameters

dbg=5
def readModelFilters(fileNames):
    if dbg>3 : print "running readModelFilters"
    filters={}
    for k,f in fileNames.items():
        fileName=os.path.basename(f)
        filt=numpy.loadtxt(f,unpack=True)
        tck=scipy.interpolate.splrep(filt[0],filt[1],s=0)
        filters[k]=(filt[0],filt[1],tck)
    return filters

# read desithroughput from DESIMODEL for each arm
# returns a dict that contains the efficiency and wavelength corresponding to throughput files

def readDesithroughput(thrunames):
    throughput={}
    for k,v in thrunames.items():
        hdr=pyfits.open(v)
        thru=hdr["THROUGHPUT"].data
        throughput[k]=[thru["throughput"].copy(),thru["wavelength"].copy()]
        hdr.close()
    return throughput

def findappMag(flux,wave,filt):

    # flux :(ergs/cm^2/s/A)
    # wave :(A)
    # filt : filter quantum efficiency( electrons/photons)


    """
    This is something similar to spflux_v5.pro
    
    # flux in ergs/cm^2/s/A, model flux are already in ergs/cm^2/s/A
    # wave in A, vac/air?
    # Fukugita 1996 :
    # Mag_AB = -2.5 log10( f_nu (ergs/cm2/s/Hz) ) -48.60

    #flux_filt=flux*filt
    #flux_nu,wave_nu=Angstrom2Hz(flux_filt,wave)
    #total_flux_nu=integratedFlux(flux_nu,wave)
    #print total_flux_nu
    #appMag=-2.5*numpy.log10(total_flux_nu)-(48.6) #-2.5*17) # formula for AB_mag?    
    #Check this about flux/flux density? 
    """  
    # convert flux to photons/cm^2/s/A and integrate with filter response

    flux_in_photons=ergs2photons(flux,wave)
    flux_filt_integrated=numpy.dot(flux_in_photons,filt)

    ab_spectrum = 2.99792458 * 10**(18-48.6/2.5)/hc/wave #in photons/cm^2/s/A, taken from specex_flux_calibration.py)
    ab_spectrum_filt_integrated=numpy.dot(ab_spectrum,filt)

    if flux_filt_integrated <=0:
       appMag=99.
    else:
       appMag=-2.5*numpy.log10(flux_filt_integrated/ab_spectrum_filt_integrated)
    return appMag
    #returns the apparent magnitude corresponding to that filter

def doCalib(fibermapfile,stellarmodelfile,spectrograph):
    
    fiber_hdulist=pyfits.open(fibermapfile)
    NIGHT=fiber_hdulist[1].header['NIGHT']
    EXPID=fiber_hdulist[1].header['EXPID']

    fibers=readRefStarFibers(fibermapfile)

    # pick the inputs. This should go in I/O.

    filterNameMap={}
    for i in xrange(fibers["FILTER"].shape[0]):
        filt=fibers["FILTER"][i]

        filttype=str.split(filt[0],'_')
        if filttype[0]=='SDSS':
            for f in filt:
                filterNameMap[f]=f.lower()+"0.txt"
        else: #if breakfilt[0]=='DECAM':
            for f in filt:
                filterNameMap[f]=f.lower()+".txt"

    if 'DESISIM' not in os.environ:
        raise RuntimeError('Set environment DESISIM. Can not find filter response files')
    basepath=DESISIM+"/data/"
    for i in filterNameMap:
        filterNameMap[i]=basepath+filterNameMap[i]
    print "My_filter:",filterNameMap
    filters=readModelFilters(filterNameMap)

    #now load all the skyfiles, framefiles, fiberflatfiles etc
 
    skyfile={}
    framefile={}
    fiberflatfile={}
    for i in ["b","r","z"]:
        skyfile[i]=DESI_SPECTRO_REDUX+'/'+PRODNAME+'/exposures/%s/%08d/'%(NIGHT,EXPID)+"sky-%s%s-%08d.fits"%(i,spectrograph,EXPID)
        framefile[i]=DESI_SPECTRO_REDUX+'/'+PRODNAME+'/exposures/%s/%08d/'%(NIGHT,EXPID)+"frame-%s%s-%08d.fits"%(i,spectrograph,EXPID)
        fiberflatfile[i]=DESI_SPECTRO_REDUX+'/'+PRODNAME+'/exposures/%s/'%(NIGHT)+"fiberflat-%s%s-%08d.fits"%(i,spectrograph,1)


    #Read Frames

    b_flux,b_wave,b_ivar,b_resolution=readFrame(framefile["b"])
    r_flux,r_wave,r_ivar,r_resolution=readFrame(framefile["r"])
    z_flux,z_wave,z_ivar,z_resolution=readFrame(framefile["z"])
    flux={"b":b_flux,"r":r_flux,"z":z_flux}
    wave={"b":b_wave,"r":r_wave,"z":z_wave}
    ivar={"b":b_ivar,"r":r_ivar,"z":z_ivar}
    resolution={"b":b_resolution,"r":r_resolution,"z":z_resolution}  

    # Read Flats

    b_flat=readfiberFlat(fiberflatfile["b"])
    r_flat=readfiberFlat(fiberflatfile["r"])
    z_flat=readfiberFlat(fiberflatfile["z"])

    # Read Sky

    b_sky=readSky(skyfile["b"])
    r_sky=readSky(skyfile["r"])
    z_sky=readSky(skyfile["z"])

    # Convolve Sky with Detector Resolution, so as to subtract from data. Convolve for all 500 specs

    convolvedsky={"b":convolveFlux(b_wave,b_resolution,b_sky),"r":convolveFlux(r_wave,r_resolution,r_sky),"z":convolveFlux(z_wave,z_resolution,z_sky)} # wave and sky are one-dimensional
    
    # Read the standard Star data and divide by flat and subtract sky

    stars=[]
    ivars=[]
    for i in [ x for x in fibers["FIBER"] if x < 500]:
        #flat and sky should have same wavelength binning as data, otherwise should be rebinned.

        stars.append((i,{"r":[r_flux[i,:]/r_flat[i,:]-convolvedsky["r"][i],r_wave],
                         "b":[b_flux[i,:]/b_flat[i,:]-convolvedsky["b"][i],b_wave],
                         "z":[z_flux[i,:]/z_flat[i,:]-convolvedsky["z"][i],z_wave]},fibers["MAG"][i]))
        ivars.append((i,{"r":[r_ivar[i,:]],"b":[b_ivar[i,:]],"z":[z_ivar[i,:]]}))
 
    #read Desi throughput

    dfNames={}
    for i in ["b","r","z"]:
        dfNames[i]=DESIMODEL_DIR+"/data/throughput/thru-%s.fits"%i

    desiThroughput=readDesithroughput(dfNames)
    for k,v in desiThroughput.items():
        desiThroughput[k]=rebinSpectra(desiThroughput[k][0],desiThroughput[k][1],stars[0][1][k][1])
    
    # Load model file and start the procedure to get the best model

    models=loadStellarModels(stellarmodelfile)

    # Rebin each Models first and create a dictionary of all arms
    rmodels={}
    bmodels={}
    zmodels={}
    modelWave=models["WAVEBINS"]
    

    for i,v in enumerate(models["FLUX"]):
        
        print "Rebining model",i
        rmodels[i]=rebinSpectra(v[1],modelWave,r_wave)
        zmodels[i]=rebinSpectra(v[1],modelWave,z_wave)
        bmodels[i]=rebinSpectra(v[1],modelWave,b_wave)
 
    Models={"r":rmodels,"b":bmodels,"z":zmodels}
    
    bestModels={}
    calibVector=[]
    apMag={}
    refmag={}
    scalefac={}
    for k,l in enumerate(stars):
        print "checking best model for star", l[0]
        
        rdata=stars[k][1]["r"][0]
        bdata=stars[k][1]["b"][0]
        zdata=stars[k][1]["z"][0]
        rnorm=rdata/applySmoothingFilter(rdata)
        bnorm=bdata/applySmoothingFilter(bdata)
        znorm=zdata/applySmoothingFilter(zdata)
        normData={"r":rnorm,"b":bnorm,"z":znorm}
        #print ivars
        rivar=ivars[k][1]["r"][0]
        bivar=ivars[k][1]["b"][0]
        zivar=ivars[k][1]["z"][0]
        
        starindex=l[0]
        flux={"r":rdata,"b":bdata,"z":zdata}
        ivar={"r":rivar,"b":bivar,"z":zivar}

        # Smoothing has to be propagated in the error for computing Chi square.
        ivar["b"]=ivar["b"]*(applySmoothingFilter(bdata))**2
        ivar["r"]=ivar["r"]*(applySmoothingFilter(rdata))**2
        ivar["z"]=ivar["z"]*(applySmoothingFilter(zdata))**2
        resol_star={"r":resolution["r"][l[0]],"b":resolution["b"][l[0]],"z":resolution["z"][l[0]]}

        bestModels[l[0]]=compare2Models(Models,normData,wave,ivar,resol_star,l[0])

        # Now do Calibration using this star and best model
        print "calibrating using star",l[0],'and Best model',bestModels[l[0]][1]
        print "Magnitudes", l[2]
        
        #Before computing apparent magnitude and scale factor,
        # apply filter responses to the model
       
        for f,v in enumerate(filterNameMap):
            refmag=l[2]
            print "************************************************"
            print "Current Star:", l[0],' ', "Filter:", v,' ', "Refmag:", refmag[f]
            rebinned_model_flux=rebinSpectra(models["FLUX"][bestModels[l[0]][1]][1],models["WAVEBINS"],filters[v][0])

            apMag[f]=findappMag(rebinned_model_flux,filters[v][0],filters[v][1])
            # Scaling should be dependent on filter response and not on desi throughput. Final calibration will need desi throughput??

            print "scaling Mag:",apMag[f],"to Refmag:",refmag[f]
            print "************************************************"
            scalefac[f]=10**((apMag[f]-refmag[f])/2.5)
            for i in ["b","r","z"]:
                calibration_model_flux=rebinSpectra(models["FLUX"][bestModels[l[0]][1]][1],models["WAVEBINS"],l[1][i][1])
        
                # Apply desi throughput before computing final calibration vector. ???
                calibration_model_flux=calibration_model_flux*desiThroughput[i] # Desi throughput should be same binning
                # Apply Scaling and compute the calibration vector
    	        calibVector.append((l[0],bestModels[l[0]][1],v,refmag[f],{i:l[1][i][0]/(scalefac[f]*calibration_model_flux)}))
        #print 'calib vector:',calibVector
    """
    #starindex=[]
    #bestModelIndex=[]
    #Chisq=[]
    #for k,v in bestModels.iteritems():
     
    #   starindex+=[int(v[0])]
    #   bestModelIndex+=[int(v[1])]
    #   Chisq+=[v[2]]

    #return (starindex,bestModelIndex,Chisq)#,models["WAVEBINS"],models["FLUX"][bestModelIndex][1]))
    """   
    return calibVector


if "__main__" in __name__:
    doCalib()
    
