""" 
Monitoring algorithms for Quicklook pipeline

"""

import numpy as np
import scipy.ndimage
import yaml
from desispec.quicklook.qas import MonitoringAlg
from desispec.quicklook import qlexceptions
from desispec.quicklook import qllogger
import os,sys
import datetime
from astropy.time import Time
from desispec import util
from desiutil import stats as dustat

qlog=qllogger.QLLogger("QuickLook",0)
log=qlog.getlog()


#- Few utility functions that a corresponding method of a QA class may call

def ampregion(image):
    """
    Get the pixel boundary regions for amps
       
    Args:
        image: desispec.image.Image object
    """
    from desispec.preproc import _parse_sec_keyword

    pixboundary=[]
    for kk in ['1','2','3','4']: #- 4 amps
        #- get the amp region in pix
        ampboundary=_parse_sec_keyword(image.meta["CCDSEC"+kk])
        pixboundary.append(ampboundary)  
    return pixboundary        

def fiducialregion(frame,psf):
    """ 
    Get the fiducial amplifier regions on the CCD pixel to fiber by wavelength space
       
    Args:
        frame: desispec.frame.Frame object
        psf: desispec.psf.PSF like object
    """
    from desispec.preproc import _parse_sec_keyword

    startspec=0 #- will be None if don't have fibers on the right of the CCD.
    endspec=499 #- will be None if don't have fibers on the right of the CCD
    startwave0=0 #- lower index for the starting fiber
    startwave1=0 #- lower index for the last fiber for the amp region
    endwave0=frame.wave.shape[0] #- upper index for the starting fiber
    endwave1=frame.wave.shape[0] #- upper index for the last fiber for that amp
    pixboundary=[]
    fidboundary=[]
    
    #- Adding the min, max boundary individually for the benefit of dumping to yaml.
    leftmax=499 #- for amp 1 and 3
    rightmin=0 #- for amp 2 and 4
    bottommax=frame.wave.shape[0] #- for amp 1 and 2
    topmin=0 #- for amp 3 and 4

    for kk in ['1','2','3','4']: #- 4 amps
        #- get the amp region in pix
        ampboundary=_parse_sec_keyword(frame.meta["CCDSEC"+kk])
        pixboundary.append(ampboundary)
        for ispec in range(frame.flux.shape[0]):
            if np.all(psf.x(ispec) > ampboundary[1].start):
                startspec=ispec
                #-cutting off wavelenth boundaries from startspec 
                yy=psf.y(ispec,frame.wave)
                k=np.where(yy > ampboundary[0].start)[0]
                startwave0=k[0]
                yy=psf.y(ispec,frame.wave)
                k=np.where(yy < ampboundary[0].stop)[0]
                endwave0=k[-1]                
                break
            else:
                startspec=None
                startwave0=None
                endwave0=None
        if startspec is not None:
            for ispec in range(frame.flux.shape[0])[::-1]:
                if np.all(psf.x(ispec) < ampboundary[1].stop):
                    endspec=ispec 
                    #-cutting off wavelenth boundaries from startspec 
                    yy=psf.y(ispec,frame.wave)
                    k=np.where(yy > ampboundary[0].start)[0]
                    startwave1=k[0]
                    yy=psf.y(ispec,frame.wave)
                    k=np.where(yy < ampboundary[0].stop)[0]
                    endwave1=k[-1]  
                    break
        else:
            endspec=None
            startwave1=None
            endwave1=None

        startwave=max(startwave0,startwave1) 
        endwave=min(endwave0,endwave1)
        if endspec is not None:
            #endspec+=1 #- last entry exclusive in slice, so add 1
            #endwave+=1

            if endspec < leftmax:
                leftmax=endspec
            if startspec > rightmin:
                rightmin=startspec
            if endwave < bottommax:
                bottommax=endwave
            if startwave > topmin:
                topmin=startwave
        else:
            rightmin=0 #- Only if no spec in right side of CCD. passing 0 to encertain valid data type. Nontype throws a type error in yaml.dump. 

        #fiducialb=(slice(startspec,endspec,None),slice(startwave,endwave,None))  #- Note: y,x --> spec, wavelength 
        #fidboundary.append(fiducialb)

    #- return pixboundary,fidboundary
    return leftmax,rightmin,bottommax,topmin

def slice_fidboundary(frame,leftmax,rightmin,bottommax,topmin):
    """
    Runs fiducialregion function and makes the boundary slice for the amps:
    
    Returns (list):
        list of tuples of slices for spec- wavelength boundary for the amps.
    """
    leftmax+=1 #- last entry not counted in slice
    bottommax+=1
    if rightmin ==0:
        return [(slice(0,leftmax,None),slice(0,bottommax,None)), (slice(None,None,None),slice(None,None,None)),
                (slice(0,leftmax,None),slice(topmin,frame.wave.shape[0],None)),(slice(None,None,None),slice(None,None,None))]
    else:
        return [(slice(0,leftmax,None),slice(0,bottommax,None)), (slice(rightmin,frame.nspec,None),slice(0,bottommax,None)),
                (slice(0,leftmax,None),slice(topmin,frame.wave.shape[0],None)),(slice(rightmin,frame.nspec,None),slice(topmin,frame.wave.shape[0]-1,None))]


def getrms(image):
    """
    Calculate the rms of the pixel values)
    
    Args:
        image: 2d array
    """
    pixdata=image.ravel()
    rms=np.std(pixdata)
    return rms


def countpix(image,nsig=None,ncounts=None):
    """
    Count the pixels above a given threshold.
    
    Threshold can be in n times sigma or counts. 
    
    Args:
        image: 2d image array 
        nsig: threshold in units of sigma, e.g 2 for 2 sigma
        ncounts: threshold in units of count, e.g 100
    """
    if nsig is not None:
        sig=np.std(image.ravel())
        counts_nsig=np.where(image.ravel() > nsig*sig)[0].shape[0]
        return counts_nsig
    if ncounts is not None:
        counts_thresh=np.where(image.ravel() > ncounts)[0].shape[0]
        return counts_thresh

def countbins(flux,threshold=0):
    """
    Count the number of bins above a given threshold on each fiber
    
    Args:
        flux: 2d (nspec,nwave)
        threshold: threshold counts 
    """
    counts=np.zeros(flux.shape[0])
    for ii in range(flux.shape[0]):
        ok=np.where(flux[ii]> threshold)[0]
        counts[ii]=ok.shape[0]
    return counts

def continuum(wave,flux,wmin=None,wmax=None):
    """
    Find the continuum of the spectrum inside a wavelength region.
    
    Args:
        wave: 1d wavelength array
        flux: 1d counts/flux array
        wmin and wmax: region to consider for the continuum
    """
    if wmin is None:
        wmin=min(wave)
    if wmax is None:
        wmax=max(wave)

    kk=np.where((wave>wmin) & (wave < wmax))
    newwave=wave[kk]
    newflux=flux[kk]
    #- find the median continuum 
    medcont=np.median(newflux)
    return medcont

def integrate_spec(wave,flux):
    """
    Calculate the integral of the spectrum in the given range using trapezoidal integration

    Note: limits of integration are min and max values of wavelength
    
    Args:
        wave: 1d wavelength array
        flux: 1d flux array 
    """   
    integral=np.trapz(flux,wave)
    return integral

def sky_resid(param, frame, skymodel, quick_look=False):
    """
    Algorithm for sky residual
    To be called from desispec.sky.qa_skysub and desispec.qa.qa_quicklook.Sky_residual.run_qa
    Args: 
        param : dict of QA parameters
        frame : desispec.Frame object after sky subtraction
        skymodel : desispec.SkyModel object #- no need of this as frame is sky subtracted.
    Returns a qa dictionary for sky resid
    """
    # Output dict
    qadict = {}
    qadict['NREJ'] = int(skymodel.nrej)

    # Grab sky fibers on this frame
    skyfibers = np.where(frame.fibermap['OBJTYPE'] == 'SKY')[0]
    assert np.max(skyfibers) < 500  #- indices, not fiber numbers
    nfibers=len(skyfibers)
    qadict['NSKY_FIB'] = int(nfibers)

    #current_ivar=frame.ivar[skyfibers].copy()
    #flux = frame.flux[skyfibers]

    # Subtract
    #res = flux - skymodel.flux[skyfibers] # Residuals
    #res_ivar = util.combine_ivar(current_ivar, skymodel.ivar[skyfibers])

    #- Residuals
    res=frame.flux[skyfibers] #- as this frame is already sky subtracted
    res_ivar=frame.ivar[skyfibers]

    # Chi^2 and Probability
    chi2_fiber = np.sum(res_ivar*(res**2),1)
    chi2_prob = np.zeros(nfibers)
    for ii in range(nfibers):
        # Stats
        dof = np.sum(res_ivar[ii,:] > 0.)
        chi2_prob[ii] = scipy.stats.chisqprob(chi2_fiber[ii], dof)
    # Bad models
    qadict['NBAD_PCHI'] = int(np.sum(chi2_prob < param['PCHI_RESID']))
    if qadict['NBAD_PCHI'] > 0:
        log.warning("Bad Sky Subtraction in {:d} fibers".format(
                qadict['NBAD_PCHI']))
    # Median residual
    qadict['MED_RESID'] = float(np.median(res)) # Median residual (counts)
    log.info("Median residual for sky fibers = {:g}".format(
        qadict['MED_RESID']))

    # Residual percentiles
    perc = dustat.perc(res, per=param['PER_RESID'])
    qadict['RESID_PER'] = [float(iperc) for iperc in perc]

    #- Residuals in wave and fiber axes
    qadict["MED_RESID_FIBER"]=np.median(res,axis=1)        
    qadict["MED_RESID_WAVE"]=np.median(res,axis=0)

    #- Weighted average for each bin on all fibers
    qadict["WT_AVG_WAVE"]= np.sum(res*res_ivar,0) / np.sum(res_ivar,0)        

    if quick_look:
        qadict["WAVELENGTH"]=frame.wave
        qadict["SKY_FIBERID"]=skyfibers.tolist()
    # Return
    return qadict

def SN_ratio(flux,ivar):
    """
    SN Ratio

    At current QL setting, can't use offline QA for S/N calculation for sky 
    subtraction, as that requires frame before sky subtration as QA itself 
    does the sky subtration. QL should take frame after sky subtration. 
    Also a S/N calculation there needs skymodel object (as it is specific 
    to Sky subtraction), that is not needed for S/N calculation itself.

    Args:
        flux (array): 2d [nspec,nwave] the signal (typically for spectra, 
            this comes from frame object
        ivar (array): 2d [nspec,nwave] corresponding inverse variance
    """

    #- we calculate median and total S/N assuming no correlation bin by bin
    medsnr=np.zeros(flux.shape[0])
    #totsnr=np.zeros(flux.shape[0])
    for ii in range(flux.shape[0]):
        signalmask=flux[ii,:]>0 #- mask negative values
        snr=flux[ii,signalmask]*np.sqrt(ivar[ii,signalmask])
        medsnr[ii]=np.median(snr)
        # totsnr[ii]=np.sqrt(np.sum(snr**2))
    return medsnr #, totsnr

def SignalVsNoise(frame,params,fidboundary=None):
    """
    Signal vs. Noise

    Take flux and inverse variance arrays and calculate S/N for individual
    targets (ELG, LRG, QSO, STD) and for each amplifier of the camera.

    Args:
        flux (array): 2d [nspec,nwave] the signal (typically for spectra, 
            this comes from frame object
        ivar (array): 2d [nspec,nwave] corresponding inverse variance
        fidboundary : list of slices indicating where to select in fiber
            and wavelength directions for each amp (output of fiducialregion)
    """

    #- we calculate median and total S/N assuming no correlation bin by bin

    medsnr=SN_ratio(frame.flux,frame.ivar)
    elgfibers=np.where(frame.fibermap['OBJTYPE']=='ELG')[0]
    elg_medsnr=medsnr[elgfibers]
    elg_mag=np.zeros(len(elgfibers))
    for ii,fib in enumerate(elgfibers):
        elg_mag[ii]=frame.fibermap['MAG'][fib][frame.fibermap['FILTER'][fib]=="DECAM_R"]
    elg_snr_mag=np.array((elg_medsnr,elg_mag)) #- not storing fiber number

    lrgfibers=np.where(frame.fibermap['OBJTYPE']=='LRG')[0]
    lrg_medsnr=medsnr[lrgfibers]
    lrg_mag=np.zeros(len(lrgfibers))
    for ii,fib in enumerate(lrgfibers):
        lrg_mag[ii]=frame.fibermap['MAG'][fib][frame.fibermap['FILTER'][fib]=="DECAM_R"]
    lrg_snr_mag=np.array((lrg_medsnr,lrg_mag))

    qsofibers=np.where(frame.fibermap['OBJTYPE']=='QSO')[0]
    qso_medsnr=medsnr[qsofibers]
    qso_mag=np.zeros(len(qsofibers))
    for ii,fib in enumerate(qsofibers):
        qso_mag[ii]=frame.fibermap['MAG'][fib][frame.fibermap['FILTER'][fib]=="DECAM_R"]
    qso_snr_mag=np.array((qso_medsnr,qso_mag))

    stdfibers=np.where(frame.fibermap['OBJTYPE']=='STD')[0]
    std_medsnr=medsnr[stdfibers]
    std_mag=np.zeros(len(stdfibers))
    for ii,fib in enumerate(stdfibers):
        std_mag[ii]=frame.fibermap['MAG'][fib][frame.fibermap['FILTER'][fib]=="DECAM_R"] 
    std_snr_mag=np.array((std_medsnr,std_mag))

    average_amp = None
    if fidboundary is not None:        
        average1=average2=average3=average4=0.0
        medsnr1=SN_ratio(frame.flux[fidboundary[0]],frame.ivar[fidboundary[0]])
        average1=np.mean(medsnr1)

        medsnr3=SN_ratio(frame.flux[fidboundary[2]],frame.ivar[fidboundary[2]])
        average3=np.mean(medsnr3)

        if fidboundary[1][0].start is not None: #- to the right bottom of the CCD
            medsnr2=SN_ratio(frame.flux[fidboundary[1]],frame.ivar[fidboundary[1]])
            average2=np.mean(medsnr2)

        if fidboundary[3][0].start is not None : #- to the right top of the CCD
            medsnr4=SN_ratio(frame.flux[fidboundary[3]],frame.ivar[fidboundary[3]])
            average4=np.mean(medsnr4)

        average_amp=np.array([average1,average2,average3,average4])

    qadict={"MEDIAN_SNR":medsnr,"MEDIAN_AMP_SNR":average_amp, "ELG_FIBERID":elgfibers.tolist(), "ELG_SNR_MAG": elg_snr_mag, "LRG_FIBERID":lrgfibers.tolist(), "LRG_SNR_MAG": lrg_snr_mag, "QSO_FIBERID": qsofibers.tolist(), "QSO_SNR_MAG": qso_snr_mag, "STAR_FIBERID": stdfibers.tolist(), "STAR_SNR_MAG":std_snr_mag}

    return qadict

def gauss(x,a,mu,sigma):
    """
    Gaussian fit of input data
    """
    return a*np.exp(-(x-mu)**2/(2*sigma**2))

def qlf_post(qadict):
    """
    A general function to HTTP post the QA output dictionary, intended for QLF
    requires environmental variables: QLF_API_URL, QLF_USER, QLF_PASSWD
    
    Args: 
        qadict: returned dictionary from a QA
    """
    #- Check for environment variables and set them here
    if "QLF_API_URL" in os.environ:
        qlf_url=os.environ.get("QLF_API_URL")
        if "QLF_USER" not in os.environ or "QLF_PASSWD" not in os.environ: 
            log.warning("Environment variables are not set for QLF. Set QLF_USER and QLF_PASSWD.")
        else: 
            qlf_user=os.environ.get("QLF_USER")
            qlf_passwd=os.environ.get("QLF_PASSWD")
            log.info("Environment variables are set for QLF. Now trying HTTP post.")
            #- All set. Now try to HTTP post
            try: 
                import requests
                response=requests.get(qlf_url)
                #- Check if the api has json
                api=response.json()
                #- proceed with post
                job={"name":"QL","status":0,"dictionary":qadict} #- QLF should disintegrate dictionary
                response=requests.post(api['job'],json=job,auth=(qlf_user,qlf_passwd))
            except:
                log.info("Skipping HTTP post...")    

    else:   
        log.warning("Skipping QLF. QLF_API_URL must be set as environment variable")

class Get_RMS(MonitoringAlg):
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="RMS"
        from desispec.image import Image as im
        MonitoringAlg.__init__(self,name,im,config,logger)
    def run(self,*args,**kwargs):
        if len(args) == 0 :
            raise qlexceptions.ParameterException("Missing input parameter")
        if not self.is_compatible(type(args[0])):
            raise qlexceptions.ParameterException("Incompatible parameter type. Was expecting desispec.image.Image got {}".format(type(args[0])))

        input_image=args[0]

        if "paname" not in kwargs:
            paname=None
        else:
            paname=kwargs["paname"]

        amps=False
        if "amps" in kwargs:
            amps=kwargs["amps"]

        if "qlf" in kwargs:
             qlf=kwargs["qlf"]
        else: qlf=False

        if "qafile" in kwargs: qafile = kwargs["qafile"]
        else: qafile = None

        if "qafig" in kwargs: qafig=kwargs["qafig"]
        else: qafig = None

        return self.run_qa(input_image,paname=paname,amps=amps,qafile=qafile,qafig=qafig, qlf=qlf)

    def run_qa(self,image,paname=None,amps=False,qafile=None, qafig=None,qlf=False):
        retval={}
        retval["EXPID"]= '{0:08d}'.format(image.meta["EXPID"])
        retval["PANAME"]=paname
        retval["QATIME"]=datetime.datetime.now().isoformat()
        retval["CAMERA"] = image.meta["CAMERA"]
        retval["FLAVOR"] = image.meta["FLAVOR"]
        retval["NIGHT"] = image.meta["NIGHT"]

        rmsccd=getrms(image.pix) #- should we add dark current and/or readnoise to this as well?
        if amps:
            rms_amps=[]
            rms_over_amps=[]
            overscan_values=[]
            #- get amp/overcan boundary in pixels
            from desispec.preproc import _parse_sec_keyword
            for kk in ['1','2','3','4']:
                thisampboundary=_parse_sec_keyword(image.meta["CCDSEC"+kk])
                thisoverscanboundary=_parse_sec_keyword(image.meta["BIASSEC"+kk])
                rms_thisover_thisamp=getrms(image.pix[thisoverscanboundary])
                thisoverscan_values=np.ravel(image.pix[thisoverscanboundary])
                rms_thisamp=getrms(image.pix[thisampboundary])
                rms_amps.append(rms_thisamp)
                rms_over_amps.append(rms_thisover_thisamp)
                overscan_values+=thisoverscan_values.tolist()
            rmsover=np.std(overscan_values)
            retval["METRICS"]={"RMS":rmsccd,"RMS_OVER":rmsover,"RMS_AMP":np.array(rms_amps),"RMS_OVER_AMP":np.array(rms_over_amps)}
        else:
            retval["METRICS"]={"RMS":rmsccd}     

        if qlf:
            qlf_post(retval)  

        if qafile is not None:
            yaml.dump(retval,open(qafile,"wb"))
            log.info("Output QA data is in {}".format(qafile))

        if qafig is not None:
            from desispec.qa.qa_plots_ql import plot_RMS
            plot_RMS(retval,qafig)            
            log.info("Output QA fig {}".format(qafig))      

        return retval    

    def get_default_config(self):
        return {}

class Count_Pixels(MonitoringAlg):
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="COUNTPIX"
        from desispec.image import Image as im
        MonitoringAlg.__init__(self,name,im,config,logger)
    def run(self,*args,**kwargs):
        if len(args) == 0 :
            raise qlexceptions.ParameterException("Missing input parameter")
        if not self.is_compatible(type(args[0])):
            raise qlexceptions.ParameterException("Incompatible input. Was expecting {} got {}".format(type(self.__inpType__),type(args[0])))

        input_image=args[0]

        if "paname" not in kwargs:
            paname=None
        else:
            paname=kwargs["paname"]

        amps=False
        if "amps" in kwargs:
            amps=kwargs["amps"]

        if "param" in kwargs: param=kwargs["param"]
        else: param=None

        if "qlf" in kwargs:
             qlf=kwargs["qlf"]
        else: qlf=False

        if "qafile" in kwargs: qafile = kwargs["qafile"]
        else: qafile = None

        if "qafig" in kwargs: qafig=kwargs["qafig"]
        else: qafig = None

        return self.run_qa(input_image,paname=paname,amps=amps,qafile=qafile,qafig=qafig, param=param, qlf=qlf)

    def run_qa(self,image,paname=None,amps=False,qafile=None,qafig=None, param=None, qlf=False):
        retval={}
        retval["PANAME"]=paname
        retval["QATIME"]=datetime.datetime.now().isoformat()
        retval["EXPID"]= '{0:08d}'.format(image.meta["EXPID"])
        retval["CAMERA"] = image.meta["CAMERA"]
        retval["FLAVOR"] = image.meta["FLAVOR"]
        retval["NIGHT"] = image.meta["NIGHT"]

        if param is None:
            log.info("Param is None. Using default param instead")
            param = dict(
                 CUTLO = 100,   # low threshold for number of counts
                 CUTHI = 500
                 )

        retval["PARAMS"] = param

        #- get the counts over entire CCD
        npix3sig=countpix(image.pix,nsig=3) #- above 3 sigma
        npixlo=countpix(image.pix,ncounts=param['CUTLO']) #- above 100 pixel count
        npixhi=countpix(image.pix,ncounts=param['CUTHI']) #- above 500 pixel count
        #- get the counts for each amp
        if amps:
            npix3sig_amps=[]
            npixlo_amps=[]
            npixhi_amps=[]
            #- get amp boundary in pixels
            from desispec.preproc import _parse_sec_keyword
            for kk in ['1','2','3','4']:
                ampboundary=_parse_sec_keyword(image.meta["CCDSEC"+kk])
                npix3sig_thisamp=countpix(image.pix[ampboundary],nsig=3)
                npix3sig_amps.append(npix3sig_thisamp)
                npixlo_thisamp=countpix(image.pix[ampboundary],ncounts=param['CUTLO'])
                npixlo_amps.append(npixlo_thisamp)
                npixhi_thisamp=countpix(image.pix[ampboundary],ncounts=param['CUTHI'])
                npixhi_amps.append(npixhi_thisamp)
            retval["METRICS"]={"NPIX3SIG":npix3sig,"NPIX_LOW":npixlo,"NPIX_HIGH":npixhi, "NPIX3SIG_AMP": npix3sig_amps, "NPIX_LOW_AMP": npixlo_amps,"NPIX_HIGH_AMP": npixhi_amps}
        else:
            retval["METRICS"]={"NPIX3SIG":npix3sig,"NPIX_LOW":npixlo,"NPIX_HIGH":npixhi}     

        if qlf:
            qlf_post(retval)      

        if qafile is not None:
            yaml.dump(retval,open(qafile,"wb"))
            log.info("Output QA data is in {}".format(qafile))

        if qafig is not None:
            from desispec.qa.qa_plots_ql import plot_countpix
            plot_countpix(retval,qafig)
            
            log.info("Output QA fig {}".format(qafig))      

        return retval    

    def get_default_config(self):
        return {}

class Integrate_Spec(MonitoringAlg):
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="INTEG"
        from desispec.frame import Frame as fr
        MonitoringAlg.__init__(self,name,fr,config,logger)
    def run(self,*args,**kwargs):
        if len(args) == 0 :
            raise qlexceptions.ParameterException("Missing input parameter")
        if not self.is_compatible(type(args[0])):
            raise qlexceptions.ParameterException("Incompatible input. Was expecting {}, got {}".format(type(self.__inpType__),type(args[0])))
        input_frame=args[0]

        if "paname" not in kwargs:
            paname=None
        else:
            paname=kwargs["paname"]

        amps=False
        if "amps" in kwargs:
            amps=kwargs["amps"]

        dict_countbins=None
        if "dict_countbins" in kwargs:
            dict_countbins=kwargs["dict_countbins"] 

        if "qlf" in kwargs:
             qlf=kwargs["qlf"]
        else: qlf=False

        if "qafile" in kwargs: qafile = kwargs["qafile"]
        else: qafile = None

        if "qafig" in kwargs: qafig=kwargs["qafig"]
        else: qafig = None
        return self.run_qa(input_frame,paname=paname,amps=amps, dict_countbins=dict_countbins, qafile=qafile,qafig=qafig, qlf=qlf)

    def run_qa(self,frame,paname=None,amps=False,dict_countbins=None, qafile=None,qafig=None, qlf=False):
        retval={}
        retval["PANAME"]=paname
        retval["QATIME"]=datetime.datetime.now().isoformat()
        retval["EXPID"]= '{0:08d}'.format(frame.meta["EXPID"])
        retval["CAMERA"] = frame.meta["CAMERA"]
        retval["FLAVOR"] = frame.meta["FLAVOR"]
        retval["NIGHT"] = frame.meta["NIGHT"]

        #- get the integrals for all fibers
        flux=frame.flux
        wave=frame.wave
        integrals=np.zeros(flux.shape[0])

        for ii in range(len(integrals)):
            integrals[ii]=integrate_spec(wave,flux[ii])
        
        #- average integrals over star fibers
        starfibers=np.where(frame.fibermap['OBJTYPE']=='STD')[0]
        if len(starfibers) < 1:
            log.info("WARNING: no STD fibers found.")
        int_stars=integrals[starfibers]
        int_average=np.mean(int_stars)

        #- get the counts for each amp
        if amps:

            #- get the fiducial boundary
            leftmax = dict_countbins["LEFT_MAX_FIBER"]
            rightmin = dict_countbins["RIGHT_MIN_FIBER"]
            bottommax = dict_countbins["BOTTOM_MAX_WAVE_INDEX"]
            topmin = dict_countbins["TOP_MIN_WAVE_INDEX"]

            fidboundary = slice_fidboundary(frame,leftmax,rightmin,bottommax,topmin)

            int_avg_amps=np.zeros(4)
           
            for amp in range(4):
                wave=frame.wave[fidboundary[amp][1]]
                select_thisamp=starfibers[(starfibers >= fidboundary[amp][0].start) & (starfibers < fidboundary[amp][0].stop)]
                stdflux_thisamp=frame.flux[select_thisamp,fidboundary[amp][1]]

                if len(stdflux_thisamp)==0:
                    continue
                else:
                    integ_thisamp=np.zeros(stdflux_thisamp.shape[0])

                    for ii in range(stdflux_thisamp.shape[0]):
                        integ_thisamp[ii]=integrate_spec(wave,stdflux_thisamp[ii])
                    int_avg_amps[amp]=np.mean(integ_thisamp)

            retval["METRICS"]={"INTEG":int_stars, "INTEG_AVG":int_average,"INTEG_AVG_AMP":int_avg_amps, "STD_FIBERID": starfibers.tolist()}
        else:
            retval["METRICS"]={"INTEG":int_stars,"INTEG_AVG":int_average,"STD_FIBERID":starfibers.tolist()}     

        if qlf:
            qlf_post(retval)    

        if qafile is not None:
            yaml.dump(retval,open(qafile,"wb"))
            log.info("Output QA data is in {}".format(qafile))

        if qafig is not None:
            from desispec.qa.qa_plots_ql import plot_integral
            plot_integral(retval,qafig)
            
            log.info("Output QA fig {}".format(qafig))      

        return retval    

    def get_default_config(self):
        return {}
 
 
class Sky_Continuum(MonitoringAlg):
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="SKYCONT"
        from  desispec.frame import Frame as fr
        MonitoringAlg.__init__(self,name,fr,config,logger)
    def run(self,*args,**kwargs):
        if len(args) == 0 :
            raise qlexceptions.ParameterException("Missing input parameter")
        if not self.is_compatible(type(args[0])):
            raise qlexceptions.ParameterException("Incompatible input. Was expecting {}, got {}".format(type(self.__inpType__),type(args[0])))

        input_frame=args[0]
        camera=input_frame.meta["CAMERA"]
        
        wrange1=None
        wrange2=None
        if "wrange1" in kwargs:
            wrange1=kwargs["wrange1"]
        if "wrange2" in kwargs:
            wrange2=kwargs["wrange2"]

        if wrange1==None:
            if camera[0]=="b": wrange1= "4000,4500"
            if camera[0]=="r": wrange1= "5950,6200"
            if camera[0]=="z": wrange1= "8120,8270"

        if wrange2==None:
            if camera[0]=="b": wrange2= "5250,5550"
            if camera[0]=="r": wrange2= "6990,7230"
            if camera[0]=="z": wrange2= "9110,9280"
        paname=None
        if "paname" in kwargs:
            paname=kwargs["paname"]

        amps=False
        if "amps" in kwargs:
            amps=kwargs["amps"]

        dict_countbins=None
        if "dict_countbins" in kwargs:
            dict_countbins=kwargs["dict_countbins"]

        if "qlf" in kwargs:
             qlf=kwargs["qlf"]
        else: qlf=False

        if "qafile" in kwargs: qafile = kwargs["qafile"]
        else: qafile = None

        if "qafig" in kwargs: qafig=kwargs["qafig"]
        else: qafig=None
        return self.run_qa(input_frame,wrange1=wrange1,wrange2=wrange2,paname=paname,amps=amps, dict_countbins=dict_countbins,qafile=qafile,qafig=qafig, qlf=qlf)

    def run_qa(self,frame,wrange1=None,wrange2=None,paname=None,amps=False,
dict_countbins=None,qafile=None,qafig=None, qlf=False):

        #- qa dictionary 
        retval={}
        retval["PANAME"]=paname
        retval["QATIME"]=datetime.datetime.now().isoformat()
        retval["EXPID"]= '{0:08d}'.format(frame.meta["EXPID"])
        retval["CAMERA"] = frame.meta["CAMERA"]
        retval["FLAVOR"] = frame.meta["FLAVOR"]
        retval["NIGHT"] = frame.meta["NIGHT"]

        #- get the skyfibers first
        skyfiber=np.where(frame.fibermap['OBJTYPE']=='SKY')[0]
        nspec_sky=skyfiber.shape[0]
        wminlow,wmaxlow=[float(w) for w in wrange1.split(',')]
        wminhigh,wmaxhigh=[float(w) for w in wrange2.split(',')]
        selectlow=np.where((frame.wave>wminlow) & (frame.wave<wmaxlow))[0]
        selecthigh=np.where((frame.wave>wminhigh) & (frame.wave < wmaxhigh))[0]

        contfiberlow=[]
        contfiberhigh=[]
        meancontfiber=[]
        for ii in skyfiber:
            contlow=continuum(frame.wave[selectlow],frame.flux[ii,selectlow])
            conthigh=continuum(frame.wave[selecthigh],frame.flux[ii,selecthigh])
            contfiberlow.append(contlow)
            contfiberhigh.append(conthigh)
            meancontfiber.append(np.mean((contlow,conthigh)))
        skycont=np.mean(meancontfiber) #- over the entire CCD (skyfibers)

        if amps:

            leftmax = dict_countbins["LEFT_MAX_FIBER"]
            rightmin = dict_countbins["RIGHT_MIN_FIBER"]
            bottommax = dict_countbins["BOTTOM_MAX_WAVE_INDEX"]
            topmin = dict_countbins["TOP_MIN_WAVE_INDEX"]

            fidboundary = slice_fidboundary(frame,leftmax,rightmin,bottommax,topmin)

            k1=np.where(skyfiber < fidboundary[0][0].stop)[0]
            maxsky_index=max(k1)

            contamp1=np.mean(contfiberlow[:maxsky_index])
            contamp3=np.mean(contfiberhigh[:maxsky_index])

            if fidboundary[1][0].start >=fidboundary[0][0].stop:
                k2=np.where(skyfiber > fidboundary[1][0].start)[0]
                minsky_index=min(k2)
                contamp2=np.mean(contfiberlow[minsky_index:])
                contamp4=np.mean(contfiberhigh[minsky_index:])
            else:
                contamp2=0
                contamp4=0

            skycont_amps=np.array((contamp1,contamp2,contamp3,contamp4)) #- in four amps regions

            retval["METRICS"]={"SKYFIBERID": skyfiber.tolist(), "SKYCONT":skycont, "SKYCONT_FIBER":meancontfiber, "SKYCONT_AMP":skycont_amps}

        else: 
            retval["METRICS"]={"SKYFIBERID": skyfiber.tolist(), "SKYCONT":skycont, "SKYCONT_FIBER":meancontfiber}

        if qlf:
            qlf_post(retval)    

        if qafile is not None:
            yaml.dump(retval,open(qafile,"wb"))
            log.info("Output QA data is in {}".format(qafile))

        if qafig is not None:
            from desispec.qa.qa_plots_ql import plot_sky_continuum
            plot_sky_continuum(retval,qafig)
            
            log.info("Output QA fig {}".format(qafig))                   
        
        return retval

    def get_default_config(self):
        return {}


class Sky_Peaks(MonitoringAlg):
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="SKYPEAK"
        from  desispec.frame import Frame as fr
        MonitoringAlg.__init__(self,name,fr,config,logger)
    def run(self,*args,**kwargs):
        if len(args) == 0 :
            raise qlexceptions.ParameterException("Missing input parameter")
        if not self.is_compatible(type(args[0])):
            raise qlexceptions.ParameterException("Incompatible parameter type. Was expecting desispec.image.Image, got {}".format(type(args[0])))

        input_frame=args[0]

        if "paname" not in kwargs:
            paname=None
        else:
            paname=kwargs["paname"]

        amps=False
        if "amps" in kwargs:
            amps=kwargs["amps"]

        psf = None
        if "PSFFile" in kwargs:
            psf=kwargs["PSFFile"]

        if "qlf" in kwargs:
             qlf=kwargs["qlf"]
        else: qlf=False

        if "qafile" in kwargs: qafile = kwargs["qafile"]
        else: qafile = None

        if "qafig" in kwargs:
            qafig=kwargs["qafig"]
        else: qafig = None

        return self.run_qa(input_frame,paname=paname,amps=amps,psf=psf, qafile=qafile, qafig=qafig, qlf=qlf)

    def run_qa(self,frame,paname=None,amps=False,psf=None, qafile=None,qafig=None, qlf=False):
        retval={}
        retval["PANAME"]=paname
        retval["QATIME"]=datetime.datetime.now().isoformat()
        retval["EXPID"]= '{0:08d}'.format(frame.meta["EXPID"])
        retval["CAMERA"] = camera = frame.meta["CAMERA"]
        retval["FLAVOR"] = frame.meta["FLAVOR"]
        retval["NIGHT"] = frame.meta["NIGHT"]

        # define sky peaks and wavelength region around peak flux to be integrated
        dw=2.
        b_peaks=np.array([3914.4,5199.3,5201.8])
        r_peaks=np.array([6301.9,6365.4,7318.2,7342.8,7371.3])
        z_peaks=np.array([8401.5,8432.4,8467.5,9479.4,9505.6,9521.8])

        nspec_counts=[]
        sky_counts=[]
        rms_skyspec_amp=[]
        amp1=[]
        amp2=[]
        amp3=[]
        amp4=[]
        for i in range(frame.flux.shape[0]):
            if camera[0]=="b":
                iwave1=np.argmin(np.abs(frame.wave-b_peaks[0]))
                iwave2=np.argmin(np.abs(frame.wave-b_peaks[1]))
                iwave3=np.argmin(np.abs(frame.wave-b_peaks[2]))
                peak1_flux=np.trapz(frame.flux[i,iwave1-dw:iwave1+dw+1])
                peak2_flux=np.trapz(frame.flux[i,iwave2-dw:iwave2+dw+1])
                peak3_flux=np.trapz(frame.flux[i,iwave3-dw:iwave3+dw+1])
                sum_counts=np.sum(peak1_flux+peak2_flux+peak3_flux)
                nspec_counts.append(sum_counts)
            if camera[0]=="r":
                iwave1=np.argmin(np.abs(frame.wave-r_peaks[0]))
                iwave2=np.argmin(np.abs(frame.wave-r_peaks[1]))
                iwave3=np.argmin(np.abs(frame.wave-r_peaks[2]))
                iwave4=np.argmin(np.abs(frame.wave-r_peaks[3]))
                iwave5=np.argmin(np.abs(frame.wave-r_peaks[4]))
                peak1_flux=np.trapz(frame.flux[i,iwave1-dw:iwave1+dw+1])
                peak2_flux=np.trapz(frame.flux[i,iwave2-dw:iwave2+dw+1])
                peak3_flux=np.trapz(frame.flux[i,iwave3-dw:iwave3+dw+1])
                peak4_flux=np.trapz(frame.flux[i,iwave4-dw:iwave4+dw+1])
                peak5_flux=np.trapz(frame.flux[i,iwave5-dw:iwave5+dw+1])
                sum_counts=np.sum(peak1_flux+peak2_flux+peak3_flux+peak4_flux+peak5_flux)
                nspec_counts.append(sum_counts)
            if camera[0]=="z":
                iwave1=np.argmin(np.abs(frame.wave-z_peaks[0]))
                iwave2=np.argmin(np.abs(frame.wave-z_peaks[1]))
                iwave3=np.argmin(np.abs(frame.wave-z_peaks[2]))
                iwave4=np.argmin(np.abs(frame.wave-z_peaks[3]))
                iwave5=np.argmin(np.abs(frame.wave-z_peaks[4]))
                iwave6=np.argmin(np.abs(frame.wave-z_peaks[5]))
                peak1_flux=np.trapz(frame.flux[i,iwave1-dw:iwave1+dw+1])
                peak2_flux=np.trapz(frame.flux[i,iwave2-dw:iwave2+dw+1])
                peak3_flux=np.trapz(frame.flux[i,iwave3-dw:iwave3+dw+1])
                peak4_flux=np.trapz(frame.flux[i,iwave4-dw:iwave4+dw+1])
                peak5_flux=np.trapz(frame.flux[i,iwave5-dw:iwave5+dw+1])
                peak6_flux=np.trapz(frame.flux[i,iwave6-dw:iwave6+dw+1])
                sum_counts=np.sum(peak1_flux+peak2_flux+peak3_flux+peak4_flux+peak5_flux+peak6_flux)
                nspec_counts.append(sum_counts)

            if frame.fibermap['OBJTYPE'][i]=='SKY':
                sky_counts.append(sum_counts)

                if amps:
                    if frame.fibermap['FIBER'][i]<240:
                        if camera[0]=="b":
                            amp1_flux=peak1_flux
                            amp3_flux=np.sum(peak2_flux+peak3_flux)
                        if camera[0]=="r":
                            amp1_flux=np.sum(peak1_flux+peak2_flux)
                            amp3_flux=np.sum(peak3_flux+peak4_flux+peak5_flux)
                        if camera[0]=="z":
                            amp1_flux=np.sum(peak1_flux+peak2_flux+peak3_flux)
                            amp3_flux=np.sum(peak4_flux+peak5_flux+peak6_flux)
                        amp1.append(amp1_flux)
                        amp3.append(amp3_flux)
                    if frame.fibermap['FIBER'][i]>260:
                        if camera[0]=="b":
                            amp2_flux=peak1_flux
                            amp4_flux=np.sum(peak2_flux+peak3_flux)
                        if camera[0]=="r":
                            amp2_flux=np.sum(peak1_flux+peak2_flux)
                            amp4_flux=np.sum(peak3_flux+peak4_flux+peak5_flux)
                        if camera[0]=="z":
                            amp2_flux=np.sum(peak1_flux+peak2_flux+peak3_flux)
                            amp4_flux=np.sum(peak4_flux+peak5_flux+peak6_flux)
                        amp2.append(amp2_flux)
                        amp4.append(amp4_flux)

        nspec_counts=np.array(nspec_counts)
        sky_counts=np.array(sky_counts)
        rms_nspec=getrms(nspec_counts)
        rms_skyspec=getrms(sky_counts)

        if amps:

            if frame.fibermap['FIBER'].shape[0]<260:
                amp2=np.zeros(len(sky_counts))
                amp4=np.zeros(len(sky_counts))
            else:
                amp2=np.array(amp2)
                amp4=np.array(amp4)
            amp1=np.array(amp1)
            amp3=np.array(amp3)
            amp1_rms=getrms(amp1)
            amp2_rms=getrms(amp2)
            amp3_rms=getrms(amp3)
            amp4_rms=getrms(amp4)
            rms_skyspec_amp=np.array([amp1_rms,amp2_rms,amp3_rms,amp4_rms])

            retval["METRICS"]={"SUMCOUNT":nspec_counts,"SUMCOUNT_RMS":rms_nspec,"SUMCOUNT_RMS_SKY":rms_skyspec,"SUMCOUNT_RMS_AMP":rms_skyspec_amp}
        else:
            retval["METRICS"]={"SUMCOUNT":nspec_counts,"SUMCOUNT_RMS":rms_nspec,"SUMCOUNT_RMS_SKY":rms_skyspec}

        if qlf:
            qlf_post(retval)

        if qafile is not None:
            yaml.dump(retval,open(qafile,"wb"))
            log.info("Output QA data is in {}".format(qafile))

        if qafig is not None:
            from desispec.qa.qa_plots_ql import plot_sky_peaks
            plot_sky_peaks(retval,qafig)

            log.info("Output QA fig {}".format(qafig))

        return retval

    def get_default_config(self):
        return {}


class Calc_XWSigma(MonitoringAlg):
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="XWSIGMA"
        from desispec.image import Image as im
        MonitoringAlg.__init__(self,name,im,config,logger)
    def run(self,*args,**kwargs):
        if len(args) == 0 :
            raise qlexceptions.ParameterException("Missing input parameter")
        if not self.is_compatible(type(args[0])):
            raise qlexceptions.ParameterException("Incompatible parameter type. Was expecting desispec.image.Image got {}".format(type(args[0])))
 
        input_image=args[0]
 
        if "paname" not in kwargs:
            paname=None
        else:
            paname=kwargs["paname"]
 
        amps=False
        if "amps" in kwargs:
            amps=kwargs["amps"]
 
        psf = None
        if "PSFFile" in kwargs:
            psf=kwargs["PSFFile"]
 
        fibermap = None
        if "FiberMap" in kwargs:
            fibermap=kwargs["FiberMap"]
 
        if "qlf" in kwargs:
             qlf=kwargs["qlf"]
        else: qlf=False
 
        if "qafile" in kwargs: qafile = kwargs["qafile"]
        else: qafile = None

        if "qafig" in kwargs: qafig=kwargs["qafig"]
        else: qafig = None
 
        return self.run_qa(input_image,paname=paname,amps=amps,psf=psf,fibermap=fibermap, qafile=qafile,qafig=qafig, qlf=qlf)
 
    def run_qa(self,image,paname=None,amps=False,psf=None,fibermap=None, qafile=None,qafig=None, qlf=False):
        from scipy.optimize import curve_fit
 
        retval={}
        retval["PANAME"]=paname
        retval["QATIME"]=datetime.datetime.now().isoformat() 
        retval["EXPID"]= '{0:08d}'.format(image.meta["EXPID"])
        retval["CAMERA"] = camera = image.meta["CAMERA"]
        retval["FLAVOR"] = image.meta["FLAVOR"]
        retval["NIGHT"] = image.meta["NIGHT"]

        dw=2.
        b_peaks=np.array([3914.4,5199.3,5201.8])
        r_peaks=np.array([6301.9,6365.4,7318.2,7342.8,7371.3])
        z_peaks=np.array([8401.5,8432.4,8467.5,9479.4,9505.6,9521.8])
 
        dp=3
        xsigma=[]
        wsigma=[]
        xsigma_sky=[]
        wsigma_sky=[]
        xsigma_amp1=[]
        wsigma_amp1=[]
        xsigma_amp2=[]
        wsigma_amp2=[]
        xsigma_amp3=[]
        wsigma_amp3=[]
        xsigma_amp4=[]
        wsigma_amp4=[]
        for i in range(fibermap['FIBER'].shape[0]):
            if camera[0]=="b":
                peak_wave=np.array([b_peaks[0]-dw,b_peaks[0]+dw,b_peaks[1]-dw,b_peaks[1]+dw,b_peaks[2]-dw,b_peaks[2]+dw])
 
                xpix=psf.x(ispec=i,wavelength=peak_wave)
                ypix=psf.y(ispec=i,wavelength=peak_wave)
                xpix_peak1=np.arange(int(round(xpix[0]))-dp,int(round(xpix[1]))+dp+1,1)
                ypix_peak1=np.arange(int(round(ypix[0])),int(round(ypix[1])),1)
                xpix_peak2=np.arange(int(round(xpix[2]))-dp,int(round(xpix[3]))+dp+1,1)
                ypix_peak2=np.arange(int(round(ypix[2])),int(round(ypix[3])),1)
                xpix_peak3=np.arange(int(round(xpix[4]))-dp,int(round(xpix[5]))+dp+1,1)
                ypix_peak3=np.arange(int(round(ypix[4])),int(round(ypix[5])),1)
 
                xpopt1,xpcov1=curve_fit(gauss,np.arange(len(xpix_peak1)),image.pix[int(np.mean(ypix_peak1)),xpix_peak1])
                wpopt1,wpcov1=curve_fit(gauss,np.arange(len(ypix_peak1)),image.pix[ypix_peak1
,int(np.mean(xpix_peak1))])
                xpopt2,xpcov2=curve_fit(gauss,np.arange(len(xpix_peak2)),image.pix[int(np.mean(ypix_peak2)),xpix_peak2])
                wpopt2,wpcov2=curve_fit(gauss,np.arange(len(ypix_peak2)),image.pix[ypix_peak2
,int(np.mean(xpix_peak2))])
                xpopt3,xpcov3=curve_fit(gauss,np.arange(len(xpix_peak3)),image.pix[int(np.mean(ypix_peak3)),xpix_peak3])
                wpopt3,wpcov3=curve_fit(gauss,np.arange(len(ypix_peak3)),image.pix[ypix_peak3
,int(np.mean(xpix_peak3))])

                xsigma1=np.abs(xpopt1[2])
                wsigma1=np.abs(wpopt1[2])
                xsigma2=np.abs(xpopt2[2])
                wsigma2=np.abs(wpopt2[2])
                xsigma3=np.abs(xpopt3[2])
                wsigma3=np.abs(wpopt3[2])
 
                xsig=np.array([xsigma1,xsigma2,xsigma3])
                wsig=np.array([wsigma1,wsigma2,wsigma3])
                xsigma_avg=np.mean(xsig)
                wsigma_avg=np.mean(wsig)
                xsigma.append(xsig)
                wsigma.append(wsig)
 
            if camera[0]=="r":
                peak_wave=np.array([r_peaks[0]-dw,r_peaks[0]+dw,r_peaks[1]-dw,r_peaks[1]+dw,r_peaks[2]-dw,r_peaks[2]+dw,r_peaks[3]-dw,r_peaks[3]+dw,r_peaks[4]-dw,r_peaks[4]+dw])
 
                xpix=psf.x(ispec=i,wavelength=peak_wave)
                ypix=psf.y(ispec=i,wavelength=peak_wave)
                xpix_peak1=np.arange(int(round(xpix[0]))-dp,int(round(xpix[1]))+dp+1,1)
                ypix_peak1=np.arange(int(round(ypix[0])),int(round(ypix[1])),1)
                xpix_peak2=np.arange(int(round(xpix[2]))-dp,int(round(xpix[3]))+dp+1,1)
                ypix_peak2=np.arange(int(round(ypix[2])),int(round(ypix[3])),1)
                xpix_peak3=np.arange(int(round(xpix[4]))-dp,int(round(xpix[5]))+dp+1,1)
                ypix_peak3=np.arange(int(round(ypix[4])),int(round(ypix[5])),1)
                xpix_peak4=np.arange(int(round(xpix[6]))-dp,int(round(xpix[7]))+dp+1,1)
                ypix_peak4=np.arange(int(round(ypix[6])),int(round(ypix[7])),1)
                xpix_peak5=np.arange(int(round(xpix[8]))-dp,int(round(xpix[9]))+dp+1,1)
                ypix_peak5=np.arange(int(round(ypix[8])),int(round(ypix[9])),1)

                xpopt1,xpcov1=curve_fit(gauss,np.arange(len(xpix_peak1)),image.pix[int(np.mean(ypix_peak1)),xpix_peak1])
                wpopt1,wpcov1=curve_fit(gauss,np.arange(len(ypix_peak1)),image.pix[ypix_peak1,int(np.mean(xpix_peak1))])
                xpopt2,xpcov2=curve_fit(gauss,np.arange(len(xpix_peak2)),image.pix[int(np.mean(ypix_peak2)),xpix_peak2])
                wpopt2,wpcov2=curve_fit(gauss,np.arange(len(ypix_peak2)),image.pix[ypix_peak2,int(np.mean(xpix_peak2))])
                xpopt3,xpcov3=curve_fit(gauss,np.arange(len(xpix_peak3)),image.pix[int(np.mean(ypix_peak3)),xpix_peak3])
                wpopt3,wpcov3=curve_fit(gauss,np.arange(len(ypix_peak3)),image.pix[ypix_peak3,int(np.mean(xpix_peak3))])
                xpopt4,xpcov4=curve_fit(gauss,np.arange(len(xpix_peak4)),image.pix[int(np.mean(ypix_peak4)),xpix_peak4])
                wpopt4,wpcov4=curve_fit(gauss,np.arange(len(ypix_peak4)),image.pix[ypix_peak4,int(np.mean(xpix_peak4))])
                xpopt5,xpcov5=curve_fit(gauss,np.arange(len(xpix_peak5)),image.pix[int(np.mean(ypix_peak5)),xpix_peak5])
                wpopt5,wpcov5=curve_fit(gauss,np.arange(len(ypix_peak5)),image.pix[ypix_peak5,int(np.mean(xpix_peak5))])

                xsigma1=np.abs(xpopt1[2])
                wsigma1=np.abs(wpopt1[2])
                xsigma2=np.abs(xpopt2[2])
                wsigma2=np.abs(wpopt2[2])
                xsigma3=np.abs(xpopt3[2])
                wsigma3=np.abs(wpopt3[2])
                xsigma4=np.abs(xpopt4[2])
                wsigma4=np.abs(wpopt4[2])
                xsigma5=np.abs(xpopt5[2])
                wsigma5=np.abs(wpopt5[2]) 

                xsig=np.array([xsigma1,xsigma2,xsigma3,xsigma4,xsigma5])
                wsig=np.array([wsigma1,wsigma2,wsigma3,wsigma4,wsigma5])
                xsigma_avg=np.mean(xsig)
                wsigma_avg=np.mean(wsig)
                xsigma.append(xsigma_avg)
                wsigma.append(wsigma_avg)

            if camera[0]=="z":
                peak_wave=np.array([z_peaks[0]-dw,z_peaks[0]+dw,z_peaks[1]-dw,z_peaks[1]+dw,z_peaks[2]-dw,z_peaks[2]+dw,z_peaks[3]-dw,z_peaks[3]+dw,z_peaks[4]-dw,z_peaks[4]+dw,z_peaks[5]-dw,z_peaks[5]+dw])
 
                xpix=psf.x(ispec=i,wavelength=peak_wave)
                ypix=psf.y(ispec=i,wavelength=peak_wave)
                xpix_peak1=np.arange(int(round(xpix[0]))-dp,int(round(xpix[1]))+dp+1,1)
                ypix_peak1=np.arange(int(round(ypix[0])),int(round(ypix[1])),1)
                xpix_peak2=np.arange(int(round(xpix[2]))-dp,int(round(xpix[3]))+dp+1,1)
                ypix_peak2=np.arange(int(round(ypix[2])),int(round(ypix[3])),1)
                xpix_peak3=np.arange(int(round(xpix[4]))-dp,int(round(xpix[5]))+dp+1,1)
                ypix_peak3=np.arange(int(round(ypix[4])),int(round(ypix[5])),1)
                xpix_peak4=np.arange(int(round(xpix[6]))-dp,int(round(xpix[7]))+dp+1,1)
                ypix_peak4=np.arange(int(round(ypix[6])),int(round(ypix[7])),1)
                xpix_peak5=np.arange(int(round(xpix[8]))-dp,int(round(xpix[9]))+dp+1,1)
                ypix_peak5=np.arange(int(round(ypix[8])),int(round(ypix[9])),1)
                xpix_peak6=np.arange(int(round(xpix[10]))-dp,int(round(xpix[11]))+dp+1,1)
                ypix_peak6=np.arange(int(round(ypix[10])),int(round(ypix[11])),1)
 
                xpopt1,xpcov1=curve_fit(gauss,np.arange(len(xpix_peak1)),image.pix[int(np.mean(ypix_peak1)),xpix_peak1])
                wpopt1,wpcov1=curve_fit(gauss,np.arange(len(ypix_peak1)),image.pix[ypix_peak1,int(np.mean(xpix_peak1))])
                xpopt2,xpcov2=curve_fit(gauss,np.arange(len(xpix_peak2)),image.pix[int(np.mean(ypix_peak2)),xpix_peak2])
                wpopt2,wpcov2=curve_fit(gauss,np.arange(len(ypix_peak2)),image.pix[ypix_peak2,int(np.mean(xpix_peak2))])
                xpopt3,xpcov3=curve_fit(gauss,np.arange(len(xpix_peak3)),image.pix[int(np.mean(ypix_peak3)),xpix_peak3])
                wpopt3,wpcov3=curve_fit(gauss,np.arange(len(ypix_peak3)),image.pix[ypix_peak3,int(np.mean(xpix_peak3))])
                xpopt4,xpcov4=curve_fit(gauss,np.arange(len(xpix_peak4)),image.pix[int(np.mean(ypix_peak4)),xpix_peak4])
                wpopt4,wpcov4=curve_fit(gauss,np.arange(len(ypix_peak4)),image.pix[ypix_peak4,int(np.mean(xpix_peak4))])
                xpopt5,xpcov5=curve_fit(gauss,np.arange(len(xpix_peak5)),image.pix[int(np.mean(ypix_peak5)),xpix_peak5])
                wpopt5,wpcov5=curve_fit(gauss,np.arange(len(ypix_peak5)),image.pix[ypix_peak5,int(np.mean(xpix_peak5))])
                xpopt6,xpcov6=curve_fit(gauss,np.arange(len(xpix_peak6)),image.pix[int(np.mean(ypix_peak6)),xpix_peak6])
                wpopt6,wpcov6=curve_fit(gauss,np.arange(len(ypix_peak6)),image.pix[ypix_peak6,int(np.mean(xpix_peak6))])

                xsigma1=np.abs(xpopt1[2])
                wsigma1=np.abs(wpopt1[2])
                xsigma2=np.abs(xpopt2[2])
                wsigma2=np.abs(wpopt2[2])
                xsigma3=np.abs(xpopt3[2])
                wsigma3=np.abs(wpopt3[2])
                xsigma4=np.abs(xpopt4[2])
                wsigma4=np.abs(wpopt4[2])
                xsigma5=np.abs(xpopt5[2])
                wsigma5=np.abs(wpopt5[2])
                xsigma6=np.abs(xpopt6[2])
                wsigma6=np.abs(wpopt6[2])

                xsig=np.array([xsigma1,xsigma2,xsigma3,xsigma4,xsigma5,xsigma6])
                wsig=np.array([wsigma1,wsigma2,wsigma3,wsigma4,wsigma5,wsigma6])
                xsigma_avg=np.mean(xsig)
                wsigma_avg=np.mean(wsig)
                xsigma.append(xsigma_avg)
                wsigma.append(wsigma_avg)
 
            if fibermap['OBJTYPE'][i]=='SKY':
                xsigma_sky=xsigma
                wsigma_sky=wsigma
 
            if amps:
                if fibermap['FIBER'][i]<240:
                    if camera[0]=="b":
                        xsig_amp1=np.array([xsigma1])
                        xsig_amp3=np.array([xsigma2,xsigma3])
                        wsig_amp1=np.array([wsigma1])
                        wsig_amp3=np.array([wsigma2,wsigma3])
                    if camera[0]=="r":
                        xsig_amp1=np.array([xsigma1,xsigma2])
                        xsig_amp3=np.array([xsigma3,xsigma4,xsigma5])
                        wsig_amp1=np.array([wsigma1,wsigma2])
                        wsig_amp3=np.array([wsigma3,wsigma4,wsigma5])
                    if camera[0]=="z":
                        xsig_amp1=np.array([xsigma1,xsigma2,xsigma3])
                        xsig_amp3=np.array([xsigma4,xsigma5,xsigma6])
                        wsig_amp1=np.array([wsigma1,wsigma2,wsigma3])
                        wsig_amp3=np.array([wsigma4,wsigma5,wsigma6])
                    xsigma_amp1.append(xsig_amp1)
                    wsigma_amp1.append(wsig_amp1)
                    xsigma_amp3.append(xsig_amp3)
                    wsigma_amp3.append(wsig_amp3)
                if fibermap['FIBER'][i]>260:
                    if camera[0]=="b":
                        xsig_amp2=np.array([xsigma1])
                        xsig_amp4=np.array([xsigma2,xsigma3])
                        wsig_amp2=np.array([wsigma1])
                        wsig_amp4=np.array([wsigma2,wsigma3])
                    if camera[0]=="r":
                        xsig_amp2=np.array([xsigma1,xsigma2])
                        xsig_amp4=np.array([xsigma3,xsigma4,xsigma5])
                        wsig_amp2=np.array([wsigma1,wsigma2])
                        wsig_amp4=np.array([wsigma3,wsigma4,wsigma5])
                    if camera[0]=="z":
                        xsig_amp2=np.array([xsigma1,xsigma2,xsigma3])
                        xsig_amp4=np.array([xsigma4,xsigma5,xsigma6])
                        wsig_amp2=np.array([wsigma1,wsigma2,wsigma3])
                        wsig_amp4=np.array([wsigma4,wsigma5,wsigma6])
                    xsigma_amp2.append(xsig_amp2)
                    wsigma_amp2.append(wsig_amp2)
                    xsigma_amp4.append(xsig_amp4)
                    wsigma_amp4.append(wsig_amp4)
  
                if fibermap['FIBER'].shape[0]<260:
                    xsigma_amp2=np.zeros(len(xsigma))
                    xsigma_amp4=np.zeros(len(xsigma))
                    wsigma_amp2=np.zeros(len(wsigma))
                    wsigma_amp4=np.zeros(len(wsigma))
 
        xsigma=np.array(xsigma)
        wsigma=np.array(wsigma)
        xsigma_med=np.median(xsigma)
        wsigma_med=np.median(wsigma)
        xsigma_med_sky=np.median(xsigma_sky)
        wsigma_med_sky=np.median(wsigma_sky)
        xamp1_med=np.median(xsigma_amp1)
        xamp2_med=np.median(xsigma_amp2)
        xamp3_med=np.median(xsigma_amp3)
        xamp4_med=np.median(xsigma_amp4)
        wamp1_med=np.median(wsigma_amp1)
        wamp2_med=np.median(wsigma_amp2)
        wamp3_med=np.median(wsigma_amp3)
        wamp4_med=np.median(wsigma_amp4)
        xsigma_amp=np.array([xamp1_med,xamp2_med,xamp3_med,xamp4_med])
        wsigma_amp=np.array([wamp1_med,wamp2_med,wamp3_med,wamp4_med])

        if amps:
            retval["METRICS"]={"XSIGMA":xsigma,"XSIGMA_MED":xsigma_med,"XSIGMA_MED_SKY":xsigma_med_sky,"XSIGMA_AMP":xsigma_amp,"WSIGMA":wsigma,"WSIGMA_MED":wsigma_med,"WSIGMA_MED_SKY":wsigma_med_sky,"WSIGMA_AMP":wsigma_amp}
        else:
            retval["METRICS"]={"XSIGMA":xsigma,"XSIGMA_MED":xsigma_med,"XSIGMA_MED_SKY":xsigma_med_sky,"WSIGMA":wsigma,"WSIGMA_MED":wsigma_med,"WSIGMA_MED_SKY":wsigma_med_sky}

        #- http post if needed
        if qlf:
            qlf_post(retval)    

        if qafile is not None:
            yaml.dump(retval,open(qafile,"wb"))
            log.info("Output QA data is in {}".format(qafile))

        if qafig is not None:
            from desispec.qa.qa_plots_ql import plot_XWSigma
            plot_XWSigma(retval,qafig)

            log.info("Output QA fig {}".format(qafig))

        return retval
 
    def get_default_config(self):
        return {}


class Bias_From_Overscan(MonitoringAlg):
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="BIAS_OVERSCAN"
        import astropy
        rawtype=astropy.io.fits.hdu.hdulist.HDUList
        MonitoringAlg.__init__(self,name,rawtype,config,logger)
    def run(self,*args,**kwargs):
        if len(args) == 0 :
            raise qlexceptions.ParameterException("Missing input parameter")
        if not self.is_compatible(type(args[0])):
            raise qlexceptions.ParameterException("Incompatible input. Was expecting {} got {}".format(type(self.__inpType__),type(args[0])))

        input_raw=args[0]
        camera=kwargs["camera"]

        paname=None
        if "paname" in kwargs:
            paname=kwargs["paname"]

        amps=False
        if "amps" in kwargs:
            amps=kwargs["amps"]

        if "qlf" in kwargs:
             qlf=kwargs["qlf"]
        else: qlf=False

        if "qafile" in kwargs: qafile = kwargs["qafile"]
        else: qafile = None

        if "qafig" in kwargs: qafig=kwargs["qafig"]
        else: qafig=None

        return self.run_qa(input_raw,camera,paname=paname,amps=amps, qafile=qafile,qafig=qafig, qlf=qlf)

    def run_qa(self,raw,camera,paname=None,amps=False,qafile=None,qafig=None, qlf=False):

        rawimage=raw[camera.upper()].data
        header=raw[camera.upper()].header

        retval={}
        retval["EXPID"]= '{0:08d}'.format(header["EXPID"])
        retval["CAMERA"]=camera
        retval["PANAME"]=paname
        retval["QATIME"]=datetime.datetime.now().isoformat()
        retval["FLAVOR"] = header["FLAVOR"]
        retval["NIGHT"] = header["NIGHT"]
        
        rawimage=raw[camera.upper()].data
        header=raw[camera.upper()].header

        if 'INHERIT' in header and header['INHERIT']:
            h0 = raw[0].header
            for key in h0:
                if key not in header:
                    header[key] = h0[key]

        bias_overscan=[]        
        for kk in ['1','2','3','4']:
            from desispec.preproc import _parse_sec_keyword
            
            sel=_parse_sec_keyword(header['BIASSEC'+kk])
            pixdata=rawimage[sel]
            #- Compute statistics of the bias region that only reject
            #  the 0.5% of smallest and largest values. (from sdssproc) 
            isort=np.sort(pixdata.ravel())
            nn=isort.shape[0]
            bias=np.mean(isort[int(0.005*nn) : int(0.995*nn)])
            bias_overscan.append(bias)

        bias=np.mean(bias_overscan)

        if amps:
            bias_amps=np.array(bias_overscan)
            retval["METRICS"]={'BIAS':bias,'BIAS_AMP':bias_amps}
        else:
            retval["METRICS"]={'BIAS':bias}

        #- http post if needed
        if qlf:
            qlf_post(retval)    

        if qafile is not None:
            yaml.dump(retval,open(qafile,"wb"))
            log.info("Output QA data is in {}".format(qafile))

        if qafig is not None:
            from desispec.qa.qa_plots_ql import plot_bias_overscan
            plot_bias_overscan(retval,qafig)
            
            log.info("Output QA fig {}".format(qafig))                   
        
        return retval

    def get_default_config(self):
        return {}

class CountSpectralBins(MonitoringAlg):

    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="COUNTBINS"
        from  desispec.frame import Frame as fr
        MonitoringAlg.__init__(self,name,fr,config,logger)
    def run(self,*args,**kwargs):
        if len(args) == 0 :
            raise qlexceptions.ParameterException("Missing input parameter")
        if not self.is_compatible(type(args[0])):
            raise qlexceptions.ParameterException("Incompatible input. Was expecting {} got {}".format(type(self.__inpType__),type(args[0])))
        input_frame=args[0]

        paname=None
        if "paname" in kwargs:
            paname=kwargs["paname"]

        amps=False
        if "amps" in kwargs:
            amps=kwargs["amps"]

        psf = None
        if "PSFFile" in kwargs: 
            psf=kwargs["PSFFile"]

        if "param" in kwargs: param=kwargs["param"]
        else: param=None

        if "qlf" in kwargs:
             qlf=kwargs["qlf"]
        else: qlf=False

        if "qafile" in kwargs: qafile = kwargs["qafile"]
        else: qafile = None

        if "qafig" in kwargs: qafig=kwargs["qafig"]
        else: qafig=None

        return self.run_qa(input_frame,paname=paname,amps=amps,psf=psf, qafile=qafile,qafig=qafig, param=param, qlf=qlf)


    def run_qa(self,frame,paname=None,psf=None,amps=False,qafile=None,qafig=None,param=None, qlf=False):

        #- qa dictionary 
        retval={}
        retval["PANAME"]=paname
        retval["QATIME"]=datetime.datetime.now().isoformat()
        retval["EXPID"]= '{0:08d}'.format(frame.meta["EXPID"])
        retval["CAMERA"] = frame.meta["CAMERA"]
        retval["FLAVOR"] = frame.meta["FLAVOR"]
        retval["NIGHT"] = frame.meta["NIGHT"]

        grid=np.gradient(frame.wave)
        if not np.all(grid[0]==grid[1:]): 
            log.info("grid_size is NOT UNIFORM")

        if param is None:
            log.info("Param is None. Using default param instead")
            param = dict(
                         CUTLO = 100,   # low threshold for number of counts
                         CUTMED = 250,
                         CUTHI = 500
                         )
        retval["PARAMS"] = param
        
        countslo=countbins(frame.flux,threshold=param['CUTLO'])
        countsmed=countbins(frame.flux,threshold=param['CUTMED'])
        countshi=countbins(frame.flux,threshold=param['CUTHI'])

        goodfibers=np.where(countshi>0)[0] #- fibers with at least one bin higher than 500 counts
        ngoodfibers=goodfibers.shape[0]

        leftmax=None
        rightmax=None
        bottommax=None
        topmin=None

        if amps:
            #- get the pixel boundary and fiducial boundary in flux-wavelength space

            leftmax,rightmin,bottommax,topmin = fiducialregion(frame,psf)  
            fidboundary=slice_fidboundary(frame,leftmax,rightmin,bottommax,topmin)          
            countslo_amp1=countbins(frame.flux[fidboundary[0]],threshold=param['CUTLO'])
            averagelo_amp1=np.mean(countslo_amp1)
            countsmed_amp1=countbins(frame.flux[fidboundary[0]],threshold=param['CUTMED'])
            averagemed_amp1=np.mean(countsmed_amp1)
            countshi_amp1=countbins(frame.flux[fidboundary[0]],threshold=param['CUTHI'])
            averagehi_amp1=np.mean(countshi_amp1)

            countslo_amp3=countbins(frame.flux[fidboundary[2]],threshold=param['CUTLO'])
            averagelo_amp3=np.mean(countslo_amp3)
            countsmed_amp3=countbins(frame.flux[fidboundary[2]],threshold=param['CUTMED'])
            averagemed_amp3=np.mean(countsmed_amp3)
            countshi_amp3=countbins(frame.flux[fidboundary[2]],threshold=param['CUTHI'])
            averagehi_amp3=np.mean(countshi_amp3)


            if fidboundary[1][0].start is not None: #- to the right bottom of the CCD

                countslo_amp2=countbins(frame.flux[fidboundary[1]],threshold=param['CUTLO'])
                averagelo_amp2=np.mean(countslo_amp2)
                countsmed_amp2=countbins(frame.flux[fidboundary[1]],threshold=param['CUTMED'])
                averagemed_amp2=np.mean(countsmed_amp2)
                countshi_amp2=countbins(frame.flux[fidboundary[1]],threshold=param['CUTHI'])
                averagehi_amp2=np.mean(countshi_amp2)

            else:
                averagelo_amp2=0.
                averagemed_amp2=0.
                averagehi_amp2=0.

            if fidboundary[3][0].start is not None: #- to the right top of the CCD

                countslo_amp4=countbins(frame.flux[fidboundary[3]],threshold=param['CUTLO'])
                averagelo_amp4=np.mean(countslo_amp4)
                countsmed_amp4=countbins(frame.flux[fidboundary[3]],threshold=param['CUTMED'])
                averagemed_amp4=np.mean(countsmed_amp4)
                countshi_amp4=countbins(frame.flux[fidboundary[3]],threshold=param['CUTHI'])
                averagehi_amp4=np.mean(countshi_amp4)

            else:
                averagelo_amp4=0.
                averagemed_amp4=0.
                averagehi_amp4=0.

            averagelo_amps=np.array([averagelo_amp1,averagelo_amp2,averagelo_amp3,averagelo_amp4])
            averagemed_amps=np.array([averagemed_amp1,averagemed_amp2,averagemed_amp3,averagemed_amp4])
            averagehi_amps=np.array([averagehi_amp1,averagehi_amp2,averagehi_amp3,averagehi_amp4])

            retval["METRICS"]={"NBINSLOW":countslo,"NBINSMED":countsmed,"NBINSHIGH":countshi, "NBINSLOW_AMP":averagelo_amps,"NBINSMED_AMP":averagemed_amps,"NBINSHIGH_AMP":averagehi_amps, "NGOODFIBERS": ngoodfibers}
        else:
            retval["METRICS"]={"NBINSLOW":countslo,"NBINSMED":countsmed,"NBINSHIGH":countshi,"NGOODFIBERS": ngoodfibers}

        retval["LEFT_MAX_FIBER"]=int(leftmax)
        retval["RIGHT_MIN_FIBER"]=int(rightmin)
        retval["BOTTOM_MAX_WAVE_INDEX"]=int(bottommax)
        retval["TOP_MIN_WAVE_INDEX"]=int(topmin)

        #- http post if needed
        if qlf:
            qlf_post(retval)    

        if qafile is not None:
            yaml.dump(retval,open(qafile,"wb"))
            log.info("Output QA data is in {}".format(qafile))

        if qafig is not None:
            from desispec.qa.qa_plots_ql import plot_countspectralbins
            plot_countspectralbins(retval,qafig)
            
            log.info("Output QA fig {}".format(qafig))                   
        
        return retval

class Sky_Residual(MonitoringAlg):
    """ 
    Use offline sky_residual function to calculate sky residuals
    """
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="RESIDUAL"
        from  desispec.frame import Frame as fr
        MonitoringAlg.__init__(self,name,fr,config,logger)
    def run(self,*args,**kwargs):
        from desispec.io.sky import read_sky
        if len(args) == 0 :
            raise qlexceptions.ParameterException("Missing input parameter")
        if not self.is_compatible(type(args[0])):
            raise qlexceptions.ParameterException("Incompatible input. Was expecting {} got {}".format(type(self.__inpType__),type(args[0])))

        input_frame=args[0]
        skymodel=args[1] #- should be skymodel evaluated
        if "SkyFile" in kwargs:
            from desispec.io.sky import read_sky
            skyfile=kwargs["SkyFile"]    #- Read sky model file itself from an argument
            log.info("Using given sky file {} for subtraction".format(skyfile))

            skymodel=read_sky(skyfile)

        amps=False
        if "amps" in kwargs:
            amps=kwargs["amps"]

        dict_countbins=None
        if "dict_countbins" in kwargs:
            dict_countbins=kwargs["dict_countbins"]
        
        paname=None
        if "paname" in kwargs:
            paname=kwargs["paname"]

        if "param" in kwargs: param=kwargs["param"]
        else: param=None

        if "qlf" in kwargs:
             qlf=kwargs["qlf"]
        else: qlf=False

        if "qafile" in kwargs: qafile = kwargs["qafile"]
        else: qafile = None

        if "qafig" in kwargs: qafig=kwargs["qafig"]
        else: qafig = None
        
        return self.run_qa(input_frame,paname=paname,skymodel=skymodel,amps=amps,
dict_countbins=dict_countbins, qafile=qafile,qafig=qafig, param=param, qlf=qlf)


    def run_qa(self,frame,paname=None,skymodel=None,amps=False,dict_countbins=None, qafile=None,qafig=None, param=None, qlf=False):
        from desispec.sky import qa_skysub
        from desispec import util

        if skymodel is None:
            raise IOError("Must have skymodel to find residual. It can't be None")
        #- return values
        retval={}
        retval["PANAME"]=paname
        retval["QATIME"]=datetime.datetime.now().isoformat()
        retval["EXPID"]= '{0:08d}'.format(frame.meta["EXPID"])
        retval["CAMERA"] = frame.meta["CAMERA"]
        retval["FLAVOR"] = frame.meta["FLAVOR"]
        retval["NIGHT"] = frame.meta["NIGHT"]
        
        if param is None:
            log.info("Param is None. Using default param instead")
            param = dict(
                         PCHI_RESID=0.05, # P(Chi^2) limit for bad skyfiber model residuals
                         PER_RESID=95.,   # Percentile for residual distribution
                        )
        retval["PARAMS"] = param
        qadict=sky_resid(param,frame,skymodel,quick_look=True)

        retval["METRICS"] = {}
        for key in qadict.keys():
            retval["METRICS"][key] = qadict[key]

        if qlf:
            qlf_post(retval)    

        if qafile is not None:
            yaml.dump(retval,open(qafile,"wb"))
            log.info("Output QA data is in {}".format(qafile))

        if qafig is not None:
            from desispec.qa.qa_plots_ql import plot_residuals
            plot_residuals(retval,qafig)
            
            log.info("Output QA fig {}".format(qafig))            

        return retval
        
class Calculate_SNR(MonitoringAlg):
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="SNR"
        from  desispec.frame import Frame as fr
        MonitoringAlg.__init__(self,name,fr,config,logger)
    def run(self,*args,**kwargs):
        from desispec.io.sky import read_sky
        if len(args) == 0 :
            raise qlexceptions.ParameterException("Missing input parameter")
        if not self.is_compatible(type(args[0])):
            raise qlexceptions.ParameterException("Incompatible input. Was expecting {} got {}".format(type(self.__inpType__),type(args[0])))
        input_frame=args[0]

        amps=False
        if "amps" in kwargs:
            amps=kwargs["amps"]

        dict_countbins=None
        if "dict_countbins" in kwargs:
            dict_countbins=kwargs["dict_countbins"]

        paname=None
        if "paname" in kwargs:
            paname=kwargs["paname"]

        if "qlf" in kwargs:
             qlf=kwargs["qlf"]
        else: qlf=False

        if "qafile" in kwargs: qafile = kwargs["qafile"]
        else: qafile = None

        if "qafig" in kwargs: qafig=kwargs["qafig"]
        else: qafig = None

        return self.run_qa(input_frame,paname=paname,amps=amps,dict_countbins=dict_countbins, qafile=qafile,qafig=qafig, qlf=qlf)


    def run_qa(self,frame,paname=None,amps=False,dict_countbins=None, qafile=None,qafig=None, qlf=False, param=None):

        #- return values
        retval={}
        retval["PANAME"]=paname
        retval["QATIME"]=datetime.datetime.now().isoformat()
        retval["EXPID"]= '{0:08d}'.format(frame.meta["EXPID"])
        retval["CAMERA"] = frame.meta["CAMERA"]
        retval["FLAVOR"] = frame.meta["FLAVOR"]
        retval["NIGHT"] = frame.meta["NIGHT"]

        #- select band for mag, using DECAM_R if present

        # filter_pick=["" for x in range(len(frame.fibermap))]        
        # for ii in range(len(frame.fibermap)):
        #    if "DECAM_R" in frame.fibermap["FILTER"][ii]: filter_pick[ii]="DECAM_R"
        #    else: filter_pick[ii]= -1 #- only accepting "DECAM_R" now
        # filter_pick=np.array(filter_pick)

        if param is None:
            log.info("Param is None. Using default param instead")
            params = dict(
                SNR_FLUXTHRESH=0.0, # Minimum value of flux to go into SNR calc. 
                )

        fidboundary=None
        if amps: 
            #- get the pixel boundary and fiducial boundary in flux-wavelength space
            leftmax = dict_countbins["LEFT_MAX_FIBER"]
            rightmin = dict_countbins["RIGHT_MIN_FIBER"]
            bottommax = dict_countbins["BOTTOM_MAX_WAVE_INDEX"]
            topmin = dict_countbins["TOP_MIN_WAVE_INDEX"]
            fidboundary = slice_fidboundary(frame,leftmax,rightmin,bottommax,topmin)
        qadict = SignalVsNoise(frame,params,fidboundary=fidboundary)
        retval["METRICS"] = qadict
        retval["PARAMS"] = params
        #- http post if valid
        if qlf:
            qlf_post(retval)            

        if qafile is not None:
            yaml.dump(retval,open(qafile,"wb"))
            log.info("Output QA data is in {}".format(qafile))

        if qafig is not None:
            from desispec.qa.qa_plots_ql import plot_SNR
            plot_SNR(retval,qafig)         
            log.info("Output QA fig {}".format(qafig))

        return retval

    def get_default_config(self):
        return {}

