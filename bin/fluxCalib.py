#!/usr/bin/env python

#based on specex_flux_calibration.py
#copied some functions directly

"""
Flux Calibraion for Desi Spectra.
From each frame file, with the info from corresponding fiber file,
picks the standard stars, the spectrum is compared to the best model scaled to 
the star's magnitudes.
Photon Counts = calibration Vector * flux(ergs/cm^2/s/A)
This code computes such calibration vectors for each filter band and 
for each Desi arm.
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
import os,sys


#dust extinction not implemented for now!

# Call necessary environment variables
if 'DESIMODEL' not in os.environ:
    raise RuntimeError('The environment variable DESIMODEL must be set.')
DESIMODEL_DIR=os.environ['DESIMODEL']

DESI_SPECTRO_REDUX=os.environ['DESI_SPECTRO_REDUX']
PRODNAME=os.environ['PRODNAME']
DESISIM=os.environ['DESISIM'] # to read the filter transmission files

#Some global constants from scipy.constants

h=const.h
pi=const.pi
e=const.e
c=const.c
erg=const.erg
hc= h/erg*c*1.e10 #(in units of ergsA)
#print hc

dbg=5

#def rebin(wave,spec,width) :
#    n1=wave.shape[0]
#    n2=int((wave[-1]-wave[0])/width)
#    n2=n1/(n1/n2)
#    owave    = wave[0:n1-n1%n2].reshape((n2,n1/n2)).mean(-1)
#    ospec    = spec[:,0:n1-n1%n2].reshape((spec.shape[0],n2,n1/n2)).mean(-1)
#    return owave,ospec


#load Stellar Model file
#Returns a dict of models with wavelengths(linear in A) and flux

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
 
#    if hdr0["AIRORVAC"] != "vac":
#        wavebins=convert_air_to_vacuum(wavebins)
    
    paramData=phdu[1].data
    fluxData=phdu[0].data
    modelData=[]
    for i in paramData["TEMPLATEID"]:
        modelData.append((i,fluxData[i]))
    models={"WAVEBINS":wavebins,"FLUX":modelData}
    phdu.close()
    return models

# Rebin the spectra from old Wavelength Bins to new Wavelength Bins
# by a 5th order spline interpolation
# Valid if the function is well behaving between measurement points and 
# new points are close to old ones

def rebinSpectra(spectra,oldWaveBins,newWaveBins):
    tck=scipy.interpolate.splrep(oldWaveBins,spectra,s=0,k=5)
    specnew=scipy.interpolate.splev(newWaveBins,tck,der=0)
    return specnew

# read model filters from ascii files
# return a dict of tuple composed of wavelengths, efficiency and spline parameters
def readModelFilters(fileNames):
    if dbg>3 : print "running readModelFilters"
    filters={}
    for k,f in fileNames.items():
        fileName=os.path.basename(f)
        filt=numpy.loadtxt(f,unpack=True)
        tck=scipy.interpolate.splrep(filt[0],filt[1],s=0)
        filters[k]=(filt[0],filt[1],tck)
    return filters

# apply filter efficiency to flux given at waves
def applyFilter(flux,filt):
    return flux*filt

# ergs2photons from specex
# convert erg/s/cm2/A to photons/s/cm2/A 
# number of photons = energy/(h*nu) = energy * wl/(2*pi*hbar*c)
# 2*pi* hbar*c = 2* pi * 197 eV nm = 6.28318*197.326*1.60218e-12*10 = 1.986438e-8 = 1/5.034135e7 ergs.A    
# (but we don't care of the norm. anyway here)

#convert flux(ergs) to photons
def ergs2photons(flux,wave):
    return flux*wave/hc

# convert flux(ergs/cm^2/s/A) to flux(ergs/cm^2/s/Hz)
# waves(A) to waves(Hz)
#c=c*1.0e10 ( c in Angstrom/s)
def Angstrom2Hz(flux,waves):
    return (flux*waves**2/(c*1.0e10),c*1.0e10/waves) 


#apply redshift
def applyRedshift(wave,z=0.):
    return (1+z)*wave

#integrate flux over given wavelength region
# returns integrated flux
def integrate(flux,wave):
    deltaWave=numpy.zeros(wave.shape)
    deltaWave[:-1]=wave[1:]-wave[:-1]
    deltaWave[-1]=deltaWave[-2]
    deltaWave=numpy.abs(deltaWave)
    return numpy.dot(flux,deltaWave)

def integratedFlux(flux,wave):
    return integrate(flux,wave) 


#find apparent AB magnitude by comparing the model spectrum to standard AB star spectrum
#Relations adopted from spflux_flux_calibration.py

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
        
def applySmoothingFilter(flux):
    return scipy.ndimage.filters.median_filter(flux,size(100))
    

# Read standard stars from fibermapfile, keep track of fiber,filter and magnitude

def readRefStarFibers(fibermapfile):
    phdr=pyfits.open(fibermapfile)
    table=phdr["FIBERMAP"].data
    hdr=phdr["FIBERMAP"].header
    refStarIdx=numpy.where(table["OBJTYPE"]=="STD")
    refFibers=table["FIBER"][refStarIdx]
    refFilters=table["FILTER"][refStarIdx]
    refMags=table["MAG"]
    #NIGHT=phdr[1].header['NIGHT']
    #EXPID=phdr[1].header['EXPID']
    #print refFibers,refFilters,refMags
    return {"FIBER":refFibers,"FILTER":refFilters,"MAG":refMags}
    
def readFrame(bnam,rnam,znam):
    bhdr=pyfits.open(bnam)
    bwaves=bhdr["WAVELENGTH"].data.copy()
    bflux=bhdr["PRIMARY"].data.copy()
    bhdr.close()
    rhdr=pyfits.open(rnam)
    rwaves=rhdr["WAVELENGTH"].data.copy()
    rflux=rhdr["PRIMARY"].data.copy()
    rhdr.close()
    zhdr=pyfits.open(znam)
    zwaves=zhdr["WAVELENGTH"].data.copy()
    zflux=zhdr["PRIMARY"].data.copy()
    zhdr.close()
    return {"b":(bflux,bwaves),"r":(rflux,rwaves),"z":(zflux,zwaves)}

# compare data with model 
# returns best stellar model index and error 
def compare2Models(model,data,waves,starindex):
    maxDelta=1e100
    bestId=-1
    dm=(integratedFlux(data["r"],waves["r"]),
        integratedFlux(data["b"],waves["b"]),
        integratedFlux(data["z"],waves["z"]))

    for i in xrange(len(model["r"])):
        rdelta=numpy.sum(((model["r"][i]-data["r"])**2)/data["r"])
        bdelta=numpy.sum(((model["b"][i]-data["b"])**2)/data["b"])
        zdelta=numpy.sum(((model["z"][i]-data["z"])**2)/data["z"])
        if (rdelta+bdelta+zdelta)<maxDelta:
            #print i,"Magnitudes (r,b,z) m(%s,%s,%s), d(%s,%s,%s)"%(
            #    integratedFlux(model["r"][i],waves["r"]),
            #    integratedFlux(model["b"][i],waves["b"]),
            #    integratedFlux(model["z"][i],waves["z"]),
            #    dm[0],dm[1],dm[2])
            bestId=i
            maxDelta=(rdelta+bdelta+zdelta)

    #Plot the data and best model matching
    #plt.plot(waves["r"],model["r"][bestId],label="mr")
    #plt.plot(waves["b"],model["b"][bestId],label="mb")
    #plt.plot(waves["z"],model["z"][bestId],label="mz")
    #plt.plot(waves["r"],data["r"],label="dr")
    #plt.plot(waves["b"],data["b"],label="db")
    #plt.plot(waves["z"],data["z"],label="dz")
    #plt.legend()
    #plt.title("StarIndex: %s  Best Model ID: %s"%(starindex,bestId))
    print "STD star %s best modelId="%starindex,bestId," maxDelta=",maxDelta
    #plt.show()
    #time.sleep(2)
    clf()
    return (bestId,maxDelta)

#Find The best model for the data. Compare normalised data with Normalised model


def findModel(fluxes,models,camFilters,desiThroughput):
    rmodels={}
    bmodels={}
    zmodels={}
    modelWave=models["WAVEBINS"]
    rwave=fluxes[0][1]["r"][1]
    bwave=fluxes[0][1]["b"][1]
    zwave=fluxes[0][1]["z"][1]
    rmags={}
    bmags={}
    zmags={}
    # this is not really efficient no need to calculate
    # spline parameters for every fit
    # apply Desi throughput to models as don't have Desi throughput corrected inputs
    model_IntegratedFlux={"r":[],"b":[],"z":[]}
    for i,v in enumerate(models["FLUX"]):
        #print i,v
        rmodels[i]=rebinSpectra(v[1],modelWave,rwave)
        zmodels[i]=rebinSpectra(v[1],modelWave,zwave)
        bmodels[i]=rebinSpectra(v[1],modelWave,bwave)
        # conversion of flux to photons not required as will be absorbed in overall normalisation
        rmodels[i]=applyFilter(rmodels[i],desiThroughput["r"][0])
        bmodels[i]=applyFilter(bmodels[i],desiThroughput["b"][0])
        zmodels[i]=applyFilter(zmodels[i],desiThroughput["z"][0])
        #normalize flux to 1
        model_IntegratedFlux["r"].append(integrate(rmodels[i],rwave))
        model_IntegratedFlux["b"].append(integrate(bmodels[i],bwave))
        model_IntegratedFlux["z"].append(integrate(zmodels[i],zwave))
        print "scaling model %s with"%i,model_IntegratedFlux["r"][-1],model_IntegratedFlux["b"][-1],model_IntegratedFlux["z"][-1]
        rmodels[i]/=np.abs(model_IntegratedFlux["r"][-1])
        bmodels[i]/=np.abs(model_IntegratedFlux["b"][-1])
        zmodels[i]/=np.abs(model_IntegratedFlux["z"][-1])
        #apply smoothing filters
        rmodels[i]=applySmoothingFilter(rmodels[i])
        bmodels[i]=applySmoothingFilter(bmodels[i])
        zmodels[i]=applySmoothingFilter(zmodels[i])
    normModels={"r":rmodels,"b":bmodels,"z":zmodels}
    waves={"r":rwave,"b":bwave,"z":zwave}
    bestModels={}
    data_IntegratedFlux={}
    for i in fluxes:
        data_IntegratedFlux["r"]=integrate(i[1]["r"][0],i[1]["r"][1])
        data_IntegratedFlux["b"]=integrate(i[1]["b"][0],i[1]["b"][1])
        data_IntegratedFlux["z"]=integrate(i[1]["z"][0],i[1]["z"][1])
        rdata=i[1]["r"][0]/np.abs(data_IntegratedFlux["r"])
        bdata=i[1]["b"][0]/np.abs(data_IntegratedFlux["b"])
        zdata=i[1]["z"][0]/np.abs(data_IntegratedFlux["z"])
        rdata=applySmoothingFilter(rdata)
        bdata=applySmoothingFilter(bdata)
        zdata=applySmoothingFilter(zdata)
        normData={"r":rdata,"b":bdata,"z":zdata}
        bestModels[i[0]]=compare2Models(normModels,normData,waves,i[0])
    	#print (bestModels[i[0]])[0]
    print bestModels
    return bestModels

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

def doCalib(fibermapfile,stellarmodelfile,spectrograph):
    #spectrograph can run from 0 to 9
    fiber_hdulist=pyfits.open(fibermapfile)
    NIGHT=fiber_hdulist[1].header['NIGHT']
    EXPID=fiber_hdulist[1].header['EXPID']
    
    fibers=readRefStarFibers(fibermapfile)
    #print "fibers= ",fibers
    filterNameMap={}
    for i in xrange(fibers["FILTER"].shape[0]):
        filt=fibers["FILTER"][i]

        #if isinstance(filt,str):
        #    filts=re.split('\W+',filt)
        #    for f in filts:
        #        filterNameMap[f]=f.lower()+"0.txt"
        #else:
        filttype=str.split(filt[0],'_')
        if filttype[0]=='SDSS':
            for f in filt:
                filterNameMap[f]=f.lower()+"0.txt"
        else: #if breakfilt[0]=='DECAM':
            for f in filt:
                filterNameMap[f]=f.lower()+".txt"
        #else:
        #    for f in filt:
        #        filterNameMap[f]=f.lower()+"1.txt"
    basepath=DESISIM+"/data/"
    for i in filterNameMap:
        filterNameMap[i]=basepath+filterNameMap[i]
    print "My_filter:",filterNameMap
    filters=readModelFilters(filterNameMap)

    print "Looking into frame files" 
    print "----------------*******************************-----------------"
    framefile={}
    for i in ["b","r","z"]:
        framefile[i]=DESI_SPECTRO_REDUX+'/'+PRODNAME+'/exposures/%s/%08d/'%(NIGHT,EXPID)+"frame-%s%s-%08d.fits"%(i,spectrograph,EXPID)

        print "framefile[%s]:"%i,framefile[i]
    b_frame=framefile["b"]
    r_frame=framefile["r"]
    z_frame=framefile["z"]

    #frames contains flux in terms of photons
    frame=readFrame(b_frame,r_frame,z_frame)
    stars=[]
    for i in [ x for x in fibers["FIBER"] if x < 500]:
	#print i #fiber["FIBER"]
        stars.append((i,{"r":[frame["r"][0][i,:],frame["r"][1]],
                         "b":[frame["b"][0][i,:],frame["b"][1]],
                         "z":[frame["z"][0][i,:],frame["z"][1]]},
                      fibers["MAG"][i]))
    print (stars)[0][0]
    #these steps won't be necessary if I get filter corrected flux
    dfNames={}
    for i in ["b","r","z"]:
        dfNames[i]=DESIMODEL_DIR+"/data/throughput/thru-%s.fits"%i
	#print dfNames[i]
    desiThroughput=readDesithroughput(dfNames)
    print type(desiThroughput)
    for k,v in desiThroughput.items():
        desiThroughput[k]=[rebinSpectra(desiThroughput[k][0],desiThroughput[k][1],stars[0][1][k][1]), stars[0][1][k][1]]
    models=loadStellarModels(stellarmodelfile)
    
    print "Finding Best Model for the STD stars. Scaling model First"
    bestModels=findModel(stars,models,filters,desiThroughput)
    #filter_frame={"b":0,"r":1,"z":2}
    apMag={}
    refmag={}
    scalefac={}
    #Define Calibration vector:
    # photons/cm^2/s/A=calibVector* flux(ergs/cm^2/s/A)
    calibVector=[]
    for p,l in enumerate(stars):
        refmag=l[2]

        for k,v in enumerate(filterNameMap):
            print "************************************************"
            print "Current Star:", l[0],' ', "Filter:", v,' ', "Refmag:", refmag[k]
           
            #print filters[v][0]
            #print filters[v][1]
            rebinned_model_flux=rebinSpectra(models["FLUX"][bestModels[l[0]][0]][1],models["WAVEBINS"],filters[v][0])

            apMag[k]=findappMag(rebinned_model_flux,filters[v][0],filters[v][1])

            # Scaling should be dependent on filter response and not on desi throughput. Final calibration will need desi throughput??

            print "scaling Mag:",apMag[k],"to Refmag:",refmag[k]
            print "************************************************"
            scalefac[k]=10**((apMag[k]-refmag[k])/2.5)
            for i in ["b","r","z"]:
                calibration_model_flux=rebinSpectra(models["FLUX"][bestModels[l[0]][0]][1],models["WAVEBINS"],l[1][i][1])
        
                # Apply desi throughput before computing final calibration vector
                calibration_model_flux=applyFilter(calibration_model_flux,desiThroughput[i])
                # Apply Scaling and compute the calibration vector
    	        calibVector.append((p,k,{i:[l[1][i][0]/(scalefac[k]*calibration_model_flux)]}))


    #print calibVector
    return calibVector
    #written CalibVector to file ..............
if "__main__" in __name__:
    #- Parse arguments
    #parser=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #      usage = "%prog [arguments]"
    #      )


    #parser.add_argument("--stellarmodel",type=str, help="stellar model file")
    #parser.add_argument("--fiberfile",type=str, help='fiber map file')
    #parser.add_argument("--spectrograph", type=int, help="spectrograph no.(0-9)",default=0)

    #args = parser.parse_args()

    #stellarmodelfile=args.stellarmodel
    #fibermapfile=args.fiberfile
    #spectrograph=args.spectrograph
    doCalib()

