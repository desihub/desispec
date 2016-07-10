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

# Evaluate rms of pixel values after dark subtraction
class Get_RMS(MonitoringAlg):
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="Get_RMS"
        from desispec.image import Image as im
        MonitoringAlg.__init__(self,name,im,config,logger)
    def run(self,*args,**kwargs):
        if len(args) == 0 :
            raise qlexceptions.ParameterException("Missing input parameter")
        if not self.is_compatible(type(args[0])):
            raise qlexceptions.ParameterException("Incompatible parameter type. Was expecting desispec.image.Image got %s"%(type(args[0])))
        return self.get_rms(args[0])
    def get_default_config(self):
        return {}
    def get_rms(self,image): 
        """ 
        image: desispec.image.Image like object
        attributes: image(image, ivar, mask, readnoise,
        camera, meta)
        
        """
        value=image.pix
    #TODO might need to use mask to filter?
        vals=value.ravel()
        rms=np.std(vals)
        retval={}
        retval["VALUE"]={"RMS":rms}
        retval["EXPERT_LEVEL"]={"EXPERT_LEVEL":"OBSERVER"}
        return retval

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
        
        #ampboundary=[250,input_frame.wave.shape[0]/2] #- TODO propagate amplifier boundary from kwargs. Dividing into quadrants for now. This may come from config also
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
        #retval["TIMESTAMP"]='{:%Y-%b-%d %H:%M:%S}'.format(datetime.datetime.now())
        t=datetime.datetime.now()
        retval["MJD"]=Time(t).mjd
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
        t=datetime.datetime.now()
        retval["MJD"]=Time(t).mjd

        #- get the skyfibers first
        skyfiber=np.where(frame.fibermap['OBJTYPE']=='SKY')[0]
        nspec_sky=skyfiber.shape[0]
        wminlow,wmaxlow=map(float,wrange1.split(','))
        wminhigh,wmaxhigh=map(float,wrange2.split(','))
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

            retval["VALUE"]={"SKY":skycont, "SKY_FIBER":meancontfiber,"SKY_AMPS":skycont_amps}

        else: 
            retval["VALUE"]={"SKY":skycont, "SKY_FIBER":meancontfiber}

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


class Count_Fibers(MonitoringAlg):
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="Count_Fibers"
        from  desispec.frame import Frame as fr
        MonitoringAlg.__init__(self,name,fr,config,logger)
    def run(self,*args,**kwargs):
        if len(args) == 0 :
            raise qlexceptions.ParameterException("Missing input parameter")
        if not self.is_compatible(type(args[0])):
            raise qlexceptions.ParameterException("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))
        return self.count_fibers(args[0])
    def get_default_config(self):
        return {}
                            
    def count_fibers(self,frame): # after extraction, i.e. boxcar
        """
        image: an image that has mask keyword like desispec.image.Image object
        """
    
        good=np.where(frame['MASK'].data[:,1]==0) # Although this seems to be bin mask ?? 
        count=good.shape[0]
        retval={}
        retval["VALUE"]={"Count":count}
        retval["EXPERT_LEVEL"]={"EXPERT_LEVEL":"OBSERVER"}
        return retval


class Bias_From_Overscan(MonitoringAlg):
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="BIAS_OVERSCAN"
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

        return self.run_qa(input_image,camera,expid,paname=paname,amps=amps,url=url,qafig=qafig)

    def run_qa(self,image,camera,expid,paname=None,amps=False,url=None,qafig=None):

        retval={}
        retval["EXPID"]=expid
        retval["ARM"]=camera[0]
        retval["SPECTROGRAPH"]=int(camera[1])
        retval["PANAME"]=paname
        t=datetime.datetime.now()
        retval["MJD"]=Time(t).mjd

        bias_overscan=[]        
        for kk in ['1','2','3','4']:
            from desispec.preproc import _parse_sec_keyword
            sel=_parse_sec_keyword(image.meta['BIASSEC'+kk])
            pix=image.pix[sel]
            #- Compute statistics of the bias region that only reject
            #  the 0.5% of smallest and largest values. (from sdssproc) 
            isort=np.sort(pix.ravel())
            nn=isort.shape[0]
            bias=np.mean(isort[long(0.005*nn) : long(0.995*nn)])
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
        t=datetime.datetime.now()
        retval["MJD"]=Time(t).mjd

        grid=np.gradient(input_frame.wave)
        if not np.all(grid[0]==grid[1:]): 
            log.info("grid_size is NOT UNIFORM")

        counts100=countbins(input_frame.flux,threshold=100)
        counts250=countbins(input_frame.flux,threshold=250)
        counts500=countbins(input_frame.flux,threshold=500)

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

            retval["VALUE"]={"NBINS100":counts100,"NBINS250":counts250,"NBINS500":counts500, "NBINS100_AMP":average100_amps,"NBINS250_AMP":average250_amps,"NBINS500_AMP":average500_amps}
        else:
            retval["VALUE"]={"NBINS100":counts100,"NBINS250":counts250,"NBINS500":counts500}

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
        t=datetime.datetime.now()
        retval["MJD"]=Time(t).mjd

        medsnr=SN_ratio(input_frame.flux,input_frame.ivar)
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

            retval["VALUE"]={"MEDIAN_SNR":medsnr,"MEDIAN_AMP_SNR":average_amp}

        else: retval["VALUE"]={"MEDIAN_SNR":medsnr}


        medsnr, totsnr=SN_ratio(input_frame.flux,input_frame.ivar)
        retval["VALUE"]={"MED_SNR":medsnr,"TOT_SNR":totsnr}
        
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

