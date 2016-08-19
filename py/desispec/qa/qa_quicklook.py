""" 
Monitoring algorithms for Quicklook pipeline

"""

import numpy as np
import scipy.ndimage
from desispec.quicklook.qas import MonitoringAlg
from desispec.quicklook import qlexceptions
from desispec.quicklook import qllogger

qlog=qllogger.QLLogger("QuickLook",0)
log=qlog.getlog()
import datetime
from astropy.time import Time

#- Few utility functions that a corresponding method of a QA class may call

def ampregion(image):
    """
       get the pixel boundary regions for amps
       args: image: desispec.image.Image object
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
       get the fiducial amplifier regions on the CCD pixel to fiber by wavelength space
       args: frame: desispec.frame.Frame object
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
            endspec+=1 #- last entry exclusive in slice, so add 1
            endwave+=1
        fiducialb=(slice(startspec,endspec,None),slice(startwave,endwave,None))  #- Note: y,x --> spec, wavelength 
        fidboundary.append(fiducialb)
 
    return pixboundary,fidboundary


def getrms(image):
    """
    calculate the rms of the pixel values)
    args: image : 2d array
    """
    pixdata=image.ravel()
    rms=np.std(pixdata)
    return rms


def countpix(image,nsig=None,ncounts=None):
    """
    count the pixels above a given threshold
    threshold can be in n times sigma or counts 
    args: image: 2d image array 
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
    count the number of bins above a given threshold on each fiber
    args: flux: 2d (nspec,nwave)
          threshold: threshold counts 
    """
    counts=np.zeros(flux.shape[0])
    for ii in range(flux.shape[0]):
        ok=np.where(flux[ii]> threshold)[0]
        counts[ii]=ok.shape[0]
    return counts

def continuum(wave,flux,wmin=None,wmax=None):
    """
    find the continuum of the spectrum inside a wavelength region"
    args: wave: 1d wavelength array
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
    calculate the integral of the spectrum in the given range using trapezoidal integration
    args: wave: 1d wavelength array
          flux: 1d flux array 
    Note: limits of integration are min and max values of wavelength
    """   
    integral=np.trapz(flux,wave)
    return integral

def SN_ratio(flux,ivar):
    """
    flux: 2d [nspec,nwave] : the signal (typically for spectra, this comes from frame object
    ivar: 2d [nspec,nwave] : corresponding inverse variance

    Note: At current QL setting, can't use offline QA for S/N calculation for sky subtraction, as 
    that requires frame before sky subtration as QA itself does the sky subtration. QL should take frame
    after sky subtration. 
    Also a S/N calculation there needs skymodel object (as it is specific to Sky subtraction), that is not needed for S/N calculation itself.
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

def gauss(x,a,mu,sigma):
    """
    return Gaussian fit of input data
    """
    return a*np.exp(-(x-mu)**2/(2*sigma**2))


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
            raise qlexceptions.ParameterException("Incompatible parameter type. Was expecting desispec.image.Image got %s"%(type(args[0])))

        input_image=args[0]
        camera=kwargs["camera"]
        expid=kwargs["expid"] 

        if "paname" not in kwargs:
            paname=None
        else:
            paname=kwargs["paname"]

        amps=False
        if "amps" in kwargs:
            amps=kwargs["amps"]

        if "url" in kwargs:
             url=kwargs["url"]
        else: 
             url=None

        if "qafig" in kwargs: qafig=kwargs["qafig"]
        else: qafig = None

        return self.run_qa(input_image,camera,expid,paname=paname,amps=amps,url=url,qafig=qafig)

    def run_qa(self,image,camera,expid,paname=None,amps=False,url=None,qafig=None):
        retval={}
        retval["EXPID"]=expid
        retval["ARM"]=camera[0]
        retval["SPECTROGRAPH"]=int(camera[1])
        retval["PANAME"]=paname
        retval["QATIME"]=datetime.datetime.now().isoformat()
        rmsccd=getrms(image.pix) #- should we add dark current and/or readnoise to this as well?
        if amps:
            rms_amps=[]
            rmsover_amps=[]
            overscanregion=[]
            #- get amp/overcan boundary in pixels
            from desispec.preproc import _parse_sec_keyword
            for kk in ['1','2','3','4']:
                ampboundary=_parse_sec_keyword(image.meta["CCDSEC"+kk])
                overscanboundary=_parse_sec_keyword(image.meta["BIASSEC"+kk])
                rmsoveramp=getrms(image.pix[overscanboundary])
                overscan=np.ravel(image.pix[overscanboundary])
                rmsamp=getrms(image.pix[ampboundary])
                rms_amps.append(rmsamp)
                rmsover_amps.append(rmsoveramp)
                overscanregion.append(overscan)
            rmsover=np.std(overscanregion)
            retval["VALUE"]={"RMS":rmsccd,"RMS_OVER":rmsover,"RMS_AMP":np.array(rms_amps),"RMS_OVER_AMP":np.array(rmsover_amps)}
        else:
            retval["VALUE"]={"RMS":rmsccd}     

        if url is not None:
            try: 
                import requests
                response=requests.get(url)
                #- Check if the api has json
                api=response.json()
                #- proceed with post
                job={"name":"QL","status":0,"dictionary":retval} #- QLF should disintegrate dictionary
                response=requests.post(api['job'],json=job,auth=("username","password")) #- username, password not real but placeholder here.
            except:
                log.info("Skipping HTTP post...")    

        if qafig is not None:
            from desispec.qa.qa_plots_ql import plot_RMS
            plot_RMS(retval,qafig)
            
            log.info("Output QA fig %s"%qafig)      

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
            raise qlexceptions.ParameterException("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))

        input_image=args[0]
        camera=kwargs["camera"]
        expid=kwargs["expid"] 

        nsigma=None
        if "nsigma" in kwargs:
            nsigma=kwargs["nsigma"]
       
        ncounts=None
        if "ncounts" in kwargs:
            ncounts=kwargs["ncounts"]

        if "paname" not in kwargs:
            paname=None
        else:
            paname=kwargs["paname"]

        amps=False
        if "amps" in kwargs:
            amps=kwargs["amps"]

        if "url" in kwargs:
             url=kwargs["url"]
        else: 
             url=None

        if "qafig" in kwargs: qafig=kwargs["qafig"]
        else: qafig = None

        return self.run_qa(input_image,camera,expid,paname=paname,amps=amps,url=url,qafig=qafig)

    def run_qa(self,image,camera,expid,paname=None,amps=False,url=None,qafig=None):
        retval={}
        retval["EXPID"]=expid
        retval["ARM"]=camera[0]
        retval["SPECTROGRAPH"]=int(camera[1])
        retval["PANAME"]=paname
        retval["QATIME"]=datetime.datetime.now().isoformat()

        #- get the counts over entire CCD
        npix3sig=countpix(image.pix,nsig=3) #- above 3 sigma
        npix100=countpix(image.pix,ncounts=100) #- above 100 pixel count
        npix500=countpix(image.pix,ncounts=500) #- above 500 pixel count
        #- get the counts for each amp
        if amps:
            npix3sig_amps=[]
            npix100_amps=[]
            npix500_amps=[]
            #- get amp boundary in pixels
            from desispec.preproc import _parse_sec_keyword
            for kk in ['1','2','3','4']:
                ampboundary=_parse_sec_keyword(image.meta["CCDSEC"+kk])
                npix3sig=countpix(image.pix[ampboundary],nsig=3)
                npix3sig_amps.append(npix3sig)
                npix100=countpix(image.pix[ampboundary],ncounts=100)
                npix100_amps.append(npix100)
                npix500=countpix(image.pix[ampboundary],ncounts=500)
                npix500_amps.append(npix500)

            retval["VALUE"]={"NPIX3SIG":npix3sig,"NPIX100":npix100,"NPIX500":npix500, "NPIX3SIG_AMP": npix3sig_amps, "NPIX100_AMP": npix100_amps,"NPIX500_AMP": npix500_amps}
        else:
            retval["VALUE"]={"NPIX3SIG":npix3sig,"NPIX100":npix100,"NPIX500":npix500}     

        if url is not None:
            try: 
                import requests
                response=requests.get(url)
                #- Check if the api has json
                api=response.json()
                #- proceed with post
                job={"name":"QL","status":0,"dictionary":retval} #- QLF should disintegrate dictionary
                response=requests.post(api['job'],json=job,auth=("username","password")) #- username, password not real but placeholder here.
            except:
                log.info("Skipping HTTP post...")    

        if qafig is not None:
            from desispec.qa.qa_plots_ql import plot_countpix
            plot_countpix(retval,qafig)
            
            log.info("Output QA fig %s"%qafig)      

        return retval    

    def get_default_config(self):
        return {}

class Integrate_Spec(MonitoringAlg):
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="INTEG"
        from desispec.image import Image as im
        MonitoringAlg.__init__(self,name,im,config,logger)
    def run(self,*args,**kwargs):
        if len(args) == 0 :
            raise qlexceptions.ParameterException("Missing input parameter")
        if not self.is_compatible(type(args[0])):
            raise qlexceptions.ParameterException("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))

        input_frame=args[0]
        camera=kwargs["camera"]
        expid=kwargs["expid"] 

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

        if "url" in kwargs:
             url=kwargs["url"]
        else: 
             url=None

        if "qafig" in kwargs: qafig=kwargs["qafig"]
        else: qafig = None
        return self.run_qa(input_frame,camera,expid,paname=paname,amps=amps,psf=psf,url=url,qafig=qafig)

    def run_qa(self,frame,camera,expid,paname=None,amps=False,psf=None,url=None,qafig=None):
        retval={}
        retval["EXPID"]=expid
        retval["ARM"]=camera[0]
        retval["SPECTROGRAPH"]=int(camera[1])
        retval["PANAME"]=paname
        retval["QATIME"]=datetime.datetime.now().isoformat()

        #- get the integrals for all fibers
        flux=frame.flux
        wave=frame.wave
        integrals=np.zeros(flux.shape[0])

        for ii in range(len(integrals)):
            integrals[ii]=integrate_spec(wave,flux[ii])
        
        #- average integrals over star fibers
        starfibers=np.where(frame.fibermap['OBJTYPE']=='STD')
        int_stars=integrals[starfibers]
        int_average=np.mean(int_stars)

        #- get the counts for each amp
        if amps:

            #- get the fiducial boundary
            pixboundary,fidboundary=fiducialregion(frame,psf)
            int_amp1=integrate_spec(frame.wave[fidboundary[0][1]],frame.flux[fidboundary[0]][starfibers])
            int_amp3=integrate_spec(frame.wave[fidboundary[2][1]],frame.flux[fidboundary[2]][starfibers])
            
            int_avg_amp1=np.mean(int_amp1)
            int_avg_amp3=np.mean(int_amp3)

            if fidboundary[1][0].start is not None:
                int_amp2=integrate_spec(frame.wave[fidboundary[1][1]],frame.flux[fidboundary[1]][starfibers])
                int_amp4=integrate_spec(frame.wave[fidboundary[3][1]],frame.flux[fidboundary[3]][starfibers])
                int_avg_amp2=np.mean(int_amp2)
                int_avg_amp4=np.mean(int_amp4)

            else:
                int_avg_amp2=0.
                int_avg_amp4=0.

            int_avg_amps=np.array((int_avg_amp1,int_avg_amp2,int_avg_amp3,int_avg_amp4)) #- in four amps regions            

            retval["VALUE"]={"INTEG":int_stars,"INTEG_AVG":int_average,"INTEG_AVG_AMP":int_avg_amps}
        else:
            retval["VALUE"]={"INTEG":int_stars,"INTEG_AVG":int_average}     

        if url is not None:
            try: 
                import requests
                response=requests.get(url)
                #- Check if the api has json
                api=response.json()
                #- proceed with post
                job={"name":"QL","status":0,"dictionary":retval} #- QLF should disintegrate dictionary
                response=requests.post(api['job'],json=job,auth=("username","password")) #- username, password not real but placeholder here.
            except:
                log.info("Skipping HTTP post...")    

        if qafig is not None:
            from desispec.qa.qa_plots_ql import plot_integral
            plot_integral(retval,qafig)
            
            log.info("Output QA fig %s"%qafig)      

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
            raise qlexceptions.ParameterException("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))

        input_frame=args[0]
        camera=kwargs["camera"]
        expid=kwargs["expid"]
        
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
        psf = None
        if "PSFFile" in kwargs: 
            psf=kwargs["PSFFile"]

        url=None
        if "url" in kwargs:
            url=kwargs["url"]

        if "qafig" in kwargs: qafig=kwargs["qafig"]
        else: qafig=None
        return self.run_qa(input_frame,camera,expid,wrange1=wrange1,wrange2=wrange2,paname=paname,amps=amps,psf=psf,url=url,qafig=qafig)

    def run_qa(self,frame,camera,expid,wrange1=None,wrange2=None,paname=None,amps=False,psf=None,url=None,qafig=None):

        #- qa dictionary 
        retval={}
        retval["EXPID"]=expid
        retval["ARM"]=camera[0]
        retval["SPECTROGRAPH"]=int(camera[1])

        retval["PANAME"]=paname
        retval["QATIME"]=datetime.datetime.now().isoformat()

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
   
            pixboundary,fidboundary=fiducialregion(frame,psf)
            maxsky_index=240  #- for amp 1 and 3
            for jj,ii in enumerate(skyfiber):
                if ii < fidboundary[0][0].stop:
                    maxsky_index=jj
            contamp1=np.mean(contfiberlow[:maxsky_index])
            contamp3=np.mean(contfiberhigh[:maxsky_index])

            if fidboundary[1][0].start is not None:
                minsky_index=260 #- for amp 2 and 4
                for jj,ii in enumerate(skyfiber):
                    if ii > fidboundary[1][0].start:
                        minsky_index=jj
                contamp2=np.mean(contfiberlow[minsky_index:])
                contamp4=np.mean(contfiberhigh[minsky_index:])
            else:
                contamp2=0
                contamp4=0

            skycont_amps=np.array((contamp1,contamp2,contamp3,contamp4)) #- in four amps regions

            retval["VALUE"]={"SKYFIBERID": skyfiber.tolist(), "SKYCONT":skycont, "SKYCONT_FIBER":meancontfiber, "SKYCONT_AMP":skycont_amps}

        else: 
            retval["VALUE"]={"SKYFIBERID": skyfiber.tolist(), "SKYCONT":skycont, "SKYCONT_FIBER":meancontfiber}

        if url is not None:
            try: 
                import requests
                response=requests.get(url)
                #- Check if the api has json
                api=response.json()
                #- proceed with post
                job={"name":"QL","status":0,"dictionary":retval} 
                response=requests.post(api['job'],json=job,auth=("username","password"))
            except:
                log.info("Skipping HTTP post...")    

        if qafig is not None:
            from desispec.qa.qa_plots_ql import plot_sky_continuum
            plot_sky_continuum(retval,qafig)
            
            log.info("Output QA fig %s"%qafig)                   
        
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
            raise qlexceptions.ParameterException("Incompatible parameter type. Was expecting desispec.image.Image got %s"%(type(args[0])))

        input_frame=args[0]
        camera=kwargs["camera"]
        expid=kwargs["expid"]

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

        if "url" in kwargs:
            url=kwargs["url"]
        else:
            url=None

        if "qafig" in kwargs:
            qafig=kwargs["qafig"]
        else: qafig = None

        return self.run_qa(input_frame,camera,expid,paname=paname,amps=amps,psf=psf,url=url,qafig=qafig)

    def run_qa(self,frame,camera,expid,paname=None,amps=False,psf=None,url=None,qafig=None):
        retval={}
        retval["EXPID"]=expid
        retval["ARM"]=camera[0]
        retval["SPECTROGRAPH"]=int(camera[1])
        retval["PANAME"]=paname
        retval["QATIME"]=datetime.datetime.now().isoformat()

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
                wave1=np.argmin(np.abs(frame.wave-b_peaks[0]))
                wave2=np.argmin(np.abs(frame.wave-b_peaks[1]))
                wave3=np.argmin(np.abs(frame.wave-b_peaks[2]))
                peak1_flux=np.trapz(frame.flux[i,wave1-dw:wave1+dw+1])
                peak2_flux=np.trapz(frame.flux[i,wave2-dw:wave2+dw+1])
                peak3_flux=np.trapz(frame.flux[i,wave3-dw:wave3+dw+1])
                sum_counts=np.sum(peak1_flux+peak2_flux+peak3_flux)
                nspec_counts.append(sum_counts)
            if camera[0]=="r":
                wave1=np.argmin(np.abs(frame.wave-r_peaks[0]))
                wave2=np.argmin(np.abs(frame.wave-r_peaks[1]))
                wave3=np.argmin(np.abs(frame.wave-r_peaks[2]))
                wave4=np.argmin(np.abs(frame.wave-r_peaks[3]))
                wave5=np.argmin(np.abs(frame.wave-r_peaks[4]))
                peak1_flux=np.trapz(frame.flux[i,wave1-dw:wave1+dw+1])
                peak2_flux=np.trapz(frame.flux[i,wave2-dw:wave2+dw+1])
                peak3_flux=np.trapz(frame.flux[i,wave3-dw:wave3+dw+1])
                peak4_flux=np.trapz(frame.flux[i,wave4-dw:wave4+dw+1])
                peak5_flux=np.trapz(frame.flux[i,wave5-dw:wave5+dw+1])
                sum_counts=np.sum(peak1_flux+peak2_flux+peak3_flux+peak4_flux+peak5_flux)
                nspec_counts.append(sum_counts)
            if camera[0]=="z":
                wave1=np.argmin(np.abs(frame.wave-z_peaks[0]))
                wave2=np.argmin(np.abs(frame.wave-z_peaks[1]))
                wave3=np.argmin(np.abs(frame.wave-z_peaks[2]))
                wave4=np.argmin(np.abs(frame.wave-z_peaks[3]))
                wave5=np.argmin(np.abs(frame.wave-z_peaks[4]))
                wave6=np.argmin(np.abs(frame.wave-z_peaks[5]))
                peak1_flux=np.trapz(frame.flux[i,wave1-dw:wave1+dw+1])
                peak2_flux=np.trapz(frame.flux[i,wave2-dw:wave2+dw+1])
                peak3_flux=np.trapz(frame.flux[i,wave3-dw:wave3+dw+1])
                peak4_flux=np.trapz(frame.flux[i,wave4-dw:wave4+dw+1])
                peak5_flux=np.trapz(frame.flux[i,wave5-dw:wave5+dw+1])
                peak6_flux=np.trapz(frame.flux[i,wave6-dw:wave6+dw+1])
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
                    if frame.fibermap['FIBER'].shape[0]>260:
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

            retval["VALUE"]={"SUMCOUNT":nspec_counts,"SUMCOUNT_RMS":rms_nspec,"SUMCOUNT_RMS_SKY":rms_skyspec,"SUMCOUNT_RMS_AMP":rms_skyspec_amp}
        else:
            retval["VALUE"]={"SUMCOUNT":nspec_counts,"SUMCOUNT_RMS":rms_nspec,"SUMCOUNT_RMS_SKY":rms_skyspec}

        if qafig is not None:
            from desispec.qa.qa_plots_ql import plot_sky_peaks
            plot_sky_peaks(retval,qafig)

            log.info("Output QA fig %s"%qafig)

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
            raise qlexceptions.ParameterException("Incompatible parameter type. Was expecting desispec.image.Image got %s"%(type(args[0])))
 
        input_image=args[0]
        camera=kwargs["camera"]
        expid=kwargs["expid"]
 
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
 
        if "url" in kwargs:
             url=kwargs["url"]
        else:
             url=None
 
        if "qafig" in kwargs: qafig=kwargs["qafig"]
        else: qafig = None
 
        return self.run_qa(input_image,camera,expid,paname=paname,amps=amps,psf=psf,fibermap=fibermap,url=url,qafig=qafig)
 
    def run_qa(self,image,camera,expid,paname=None,amps=False,psf=None,fibermap=None,url=None,qafig=None):
        from scipy.optimize import curve_fit
 
        retval={}
        retval["EXPID"]=expid
        retval["ARM"]=camera[0]
        retval["SPECTROGRAPH"]=int(camera[1])
        retval["PANAME"]=paname
        retval["QATIME"]=datetime.datetime.now().isoformat()
 
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
            retval["VALUE"]={"XSIGMA":xsigma,"XSIGMA_MED":xsigma_med,"XSIGMA_MED_SKY":xsigma_med_sky,"XSIGMA_AMP":xsigma_amp,"WSIGMA":wsigma,"WSIGMA_MED":wsigma_med,"WSIGMA_MED_SKY":wsigma_med_sky,"WSIGMA_AMP":wsigma_amp}
        else:
            retval["VALUE"]={"XSIGMA":xsigma,"XSIGMA_MED":xsigma_med,"XSIGMA_MED_SKY":xsigma_med_sky,"WSIGMA":wsigma,"WSIGMA_MED":wsigma_med,"WSIGMA_MED_SKY":wsigma_med_sky}

        if qafig is not None:
            from desispec.qa.qa_plots_ql import plot_XWSigma
            plot_XWSigma(retval,qafig)

            log.info("Output QA fig %s"%qafig)

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
            raise qlexceptions.ParameterException("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))

        input_raw=args[0]
        camera=kwargs["camera"]
        expid=kwargs["expid"]

        paname=None
        if "paname" in kwargs:
            paname=kwargs["paname"]

        amps=False
        if "amps" in kwargs:
            amps=kwargs["amps"]

        url=None
        if "url" in kwargs:
            url=kwargs["url"]

        if "qafig" in kwargs: qafig=kwargs["qafig"]
        else: qafig=None

        return self.run_qa(input_raw,camera,expid,paname=paname,amps=amps,url=url,qafig=qafig)

    def run_qa(self,raw,camera,expid,paname=None,amps=False,url=None,qafig=None):

        retval={}
        retval["EXPID"]=expid
        retval["ARM"]=camera[0]
        retval["SPECTROGRAPH"]=int(camera[1])
        retval["PANAME"]=paname
        retval["QATIME"]=datetime.datetime.now().isoformat()
        
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
            retval["VALUE"]={'BIAS':bias,'BIAS_AMP':bias_amps}
        else:
            retval["VALUE"]={'BIAS':bias}

        #- http post if needed
        if url is not None:
            try: 
                import requests
                response=requests.get(url)
                #- Check if the api has json
                api=response.json()
                #- proceed with post
                job={"name":"QL","status":0,"dictionary":retval} 
                response=requests.post(api['job'],json=job,auth=("username","password"))
            except:
                log.info("Skipping HTTP post...")    

        if qafig is not None:
            from desispec.qa.qa_plots_ql import plot_bias_overscan
            plot_bias_overscan(retval,qafig)
            
            log.info("Output QA fig %s"%qafig)                   
        
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
            raise qlexceptions.ParameterException("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))
        input_frame=args[0]
        camera=kwargs["camera"]
        expid=kwargs["expid"]

        paname=None
        if "paname" in kwargs:
            paname=kwargs["paname"]

        amps=False
        if "amps" in kwargs:
            amps=kwargs["amps"]

        psf = None
        if "PSFFile" in kwargs: 
            psf=kwargs["PSFFile"]


        if "url" in kwargs:
            url=kwargs["url"]
        else:
            url=None

        if "qafig" in kwargs: qafig=kwargs["qafig"]
        else: qafig=None

        return self.run_qa(input_frame,camera,expid,paname=paname,amps=amps,psf=psf,url=url,qafig=qafig)


    def run_qa(self,input_frame,camera,expid,paname=None,psf=None,amps=False,url=None,qafig=None):

        #- qa dictionary 
        retval={}
        retval["EXPID"]=expid
        retval["ARM"]=camera[0]
        retval["SPECTROGRAPH"]=int(camera[1])
        retval["PANAME"]=paname
        retval["QATIME"]=datetime.datetime.now().isoformat()

        grid=np.gradient(input_frame.wave)
        if not np.all(grid[0]==grid[1:]): 
            log.info("grid_size is NOT UNIFORM")

        counts100=countbins(input_frame.flux,threshold=100)
        counts250=countbins(input_frame.flux,threshold=250)
        counts500=countbins(input_frame.flux,threshold=500)

        goodfibers=np.where(counts500>0)[0] #- fibers with at least one bin higher than 500 counts
        ngoodfibers=goodfibers.shape[0]

        if amps:
            #- get the pixel boundary and fiducial boundary in flux-wavelength space
            pixboundary,fidboundary=fiducialregion(input_frame,psf)
           
            counts100_amp1=countbins(input_frame.flux[fidboundary[0]],threshold=100)
            average100_amp1=np.mean(counts100_amp1)
            counts250_amp1=countbins(input_frame.flux[fidboundary[0]],threshold=250)
            average250_amp1=np.mean(counts250_amp1)
            counts500_amp1=countbins(input_frame.flux[fidboundary[0]],threshold=500)
            average500_amp1=np.mean(counts500_amp1)

            counts100_amp3=countbins(input_frame.flux[fidboundary[2]],threshold=100)
            average100_amp3=np.mean(counts100_amp3)
            counts250_amp3=countbins(input_frame.flux[fidboundary[2]],threshold=250)
            average250_amp3=np.mean(counts250_amp3)
            counts500_amp3=countbins(input_frame.flux[fidboundary[2]],threshold=500)
            average500_amp3=np.mean(counts500_amp3)


            if fidboundary[1][0].start is not None: #- to the right bottom of the CCD

                counts100_amp2=countbins(input_frame.flux[fidboundary[1]],threshold=100)
                average100_amp2=np.mean(counts100_amp2)
                counts250_amp2=countbins(input_frame.flux[fidboundary[1]],threshold=250)
                average250_amp2=np.mean(counts250_amp2)
                counts500_amp2=countbins(input_frame.flux[fidboundary[1]],threshold=500)
                average500_amp2=np.mean(counts500_amp2)

            else:
                average100_amp2=0.
                average250_amp2=0.
                average500_amp2=0.

            if fidboundary[3][0].start is not None: #- to the right top of the CCD

                counts100_amp4=countbins(input_frame.flux[fidboundary[3]],threshold=100)
                average100_amp4=np.mean(counts100_amp4)
                counts250_amp4=countbins(input_frame.flux[fidboundary[3]],threshold=250)
                average250_amp4=np.mean(counts250_amp4)
                counts500_amp4=countbins(input_frame.flux[fidboundary[3]],threshold=500)
                average500_amp4=np.mean(counts500_amp4)

            else:
                average100_amp4=0.
                average250_amp4=0.
                average500_amp4=0.

            average100_amps=np.array([average100_amp1,average100_amp2,average100_amp3,average100_amp4])
            average250_amps=np.array([average250_amp1,average250_amp2,average250_amp3,average250_amp4])
            average500_amps=np.array([average500_amp1,average500_amp2,average500_amp3,average500_amp4])

            retval["VALUE"]={"NBINS100":counts100,"NBINS250":counts250,"NBINS500":counts500, "NBINS100_AMP":average100_amps,"NBINS250_AMP":average250_amps,"NBINS500_AMP":average500_amps, "NGOODFIBERS": ngoodfibers}
        else:
            retval["VALUE"]={"NBINS100":counts100,"NBINS250":counts250,"NBINS500":counts500,"NGOODFIBERS": ngoodfibers}

        #- http post if needed
        if url is not None:
            try: 
                import requests
                response=requests.get(url)
                #- Check if the api has json
                api=response.json()
                #- proceed with post
                job={"name":"QL","status":0,"dictionary":retval} #- QLF should disintegrate dictionary
                response=requests.post(api['job'],json=job,auth=("username","password"))
            except:
                log.info("Skipping HTTP post...")    

        if qafig is not None:
            from desispec.qa.qa_plots_ql import plot_countspectralbins
            plot_countspectralbins(retval,qafig)
            
            log.info("Output QA fig %s"%qafig)                   
        
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
            raise qlexceptions.ParameterException("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))
        input_frame=args[0]
        camera=kwargs["camera"]
        expid=kwargs["expid"]

        amps=False
        if "amps" in kwargs:
            amps=kwargs["amps"]

        psf = None
        if "PSFFile" in kwargs: 
            psf=kwargs["PSFFile"]
        
        paname=None
        if "paname" in kwargs:
            paname=kwargs["paname"]

        url=None
        if "url" in kwargs:
             url=kwargs["url"]

        if "qafig" in kwargs: qafig=kwargs["qafig"]
        else: qafig = None

        return self.run_qa(input_frame,camera,expid,paname=paname,amps=amps,psf=psf,url=url,qafig=qafig)


    def run_qa(self,input_frame,camera,expid,paname=None,amps=False,psf=None,url=None,qafig=None):

        #- return values
        retval={}
        retval["ARM"]=camera[0]
        retval["SPECTROGRAPH"]=int(camera[1])
        retval["EXPID"]=expid
        retval["PANAME"]=paname
        retval["QATIME"]=datetime.datetime.now().isoformat()

        #- select band for mag, using DECAM_R if present

        filter_pick=["" for x in range(len(input_frame.fibermap))]
        
        for ii in range(len(input_frame.fibermap)):
            if "DECAM_R" in input_frame.fibermap["FILTER"][ii]: filter_pick[ii]="DECAM_R"
            else: filter_pick[ii]= -1 #- only accepting "DECAM_R" now
        filter_pick=np.array(filter_pick)

        medsnr=SN_ratio(input_frame.flux,input_frame.ivar)
        elgfibers=np.where(input_frame.fibermap['OBJTYPE']=='ELG')[0]
        elg_medsnr=medsnr[elgfibers]
        elg_mag=np.zeros(len(elgfibers))
        for ii,fib in enumerate(elgfibers):
            elg_mag[ii]=input_frame.fibermap['MAG'][fib][input_frame.fibermap['FILTER'][fib]==filter_pick[fib]]

        elg_snr_mag=np.array((elg_medsnr,elg_mag)) #- not storing fiber number
      
        
        lrgfibers=np.where(input_frame.fibermap['OBJTYPE']=='LRG')[0]
        lrg_medsnr=medsnr[lrgfibers]
        lrg_mag=np.zeros(len(lrgfibers))
        for ii,fib in enumerate(lrgfibers):
            lrg_mag[ii]=input_frame.fibermap['MAG'][fib][input_frame.fibermap['FILTER'][fib]==filter_pick[fib]]
        lrg_snr_mag=np.array((lrg_medsnr,lrg_mag))

        qsofibers=np.where(input_frame.fibermap['OBJTYPE']=='QSO')[0]
        qso_medsnr=medsnr[qsofibers]
        qso_mag=np.zeros(len(qsofibers))
        for ii,fib in enumerate(qsofibers):
            qso_mag[ii]=input_frame.fibermap['MAG'][fib][input_frame.fibermap['FILTER'][fib]==filter_pick[fib]]
        qso_snr_mag=np.array((qso_medsnr,qso_mag))

        stdfibers=np.where(input_frame.fibermap['OBJTYPE']=='STD')[0]
        std_medsnr=medsnr[stdfibers]
        std_mag=np.zeros(len(stdfibers))
        for ii,fib in enumerate(stdfibers):
            std_mag[ii]=input_frame.fibermap['MAG'][fib][input_frame.fibermap['FILTER'][fib]==filter_pick[fib]] 
        std_snr_mag=np.array((std_medsnr,std_mag))

        if amps:
            
            #- get the pixel boundary and fiducial boundary in flux-wavelength space
            pixboundary,fidboundary=fiducialregion(input_frame,psf)
           
            medsnr1=SN_ratio(input_frame.flux[fidboundary[0]],input_frame.ivar[fidboundary[0]])
            average1=np.mean(medsnr1)

            medsnr3=SN_ratio(input_frame.flux[fidboundary[2]],input_frame.ivar[fidboundary[2]])
            average3=np.mean(medsnr3)

            if fidboundary[1][0].start is not None: #- to the right bottom of the CCD
               
                medsnr2=SN_ratio(input_frame.flux[fidboundary[1]],input_frame.ivar[fidboundary[1]])
                average2=np.mean(medsnr2)
            else:
                average2=0.

            if fidboundary[3][0].start is not None : #- to the right top of the CCD

                medsnr4=SN_ratio(input_frame.flux[fidboundary[3]],input_frame.ivar[fidboundary[3]])
                average4=np.mean(medsnr4)
            else:
                average4=0.

            average_amp=np.array([average1,average2,average3,average4])

            retval["VALUE"]={"MEDIAN_SNR":medsnr,"MEDIAN_AMP_SNR":average_amp, "ELG_FIBERID":elgfibers.tolist(), "ELG_SNR_MAG": elg_snr_mag, "LRG_FIBERID":lrgfibers.tolist(), "LRG_SNR_MAG": lrg_snr_mag, "QSO_FIBERID": qsofibers.tolist(), "QSO_SNR_MAG": qso_snr_mag, "STAR_FIBERID": stdfibers.tolist(), "STAR_SNR_MAG":std_snr_mag}

        else: retval["VALUE"]={"MEDIAN_SNR":medsnr,"ELG_FIBERID": elgfibers, "ELG_SNR_MAG": elg_snr_mag, "LRG_FIBERID":lrgfibers, "LRG_SNR_MAG": lrg_snr_mag, "QSO_FIBERID": qsofibers, "QSO_SNR_MAG": qso_snr_mag, "STAR_FIBERID": stdfibers, "STAR_SNR_MAG":std_snr_mag}
        
        #- http post if valid
        if url is not None:
            try: 
                import requests
                response=requests.get(url)
                #- Check if the api has json
                api=response.json()
                #- proceed with post
                #job={"name":"QL","status":0,"measurements":[{"metric":"SNR","value":retval["VALUE"]["MED_AMP_SNR"][0]}]}
                #response=requests.post(api['job'],json=job, auth=("nobody","nobody")) #- username and password here is temporary for testing. can come from configuration rather than hardcode.

                job={"name":"QL","status":0,"dictionary":retval} #- QLF should disintegrate dictionary
                response=requests.post(api['job'],json=job,auth=("username","password"))
            except:
             
                log.info("Skipping HTTP post...")            

        if qafig is not None:
            from desispec.qa.qa_plots_ql import plot_SNR
            plot_SNR(retval,qafig)         
            log.info("Output QA fig %s"%qafig)

        return retval

    def get_default_config(self):
        return {}

