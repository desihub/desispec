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

#- Few utility functions that a corresponding method of a QA class may call

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
    totsnr=np.zeros(flux.shape[0])
    for ii in range(flux.shape[0]):
        signalmask=flux[ii,:]>0 #- mask negative values
        snr=flux[ii,signalmask]*np.sqrt(ivar[ii,signalmask])
        medsnr[ii]=np.median(snr)
        totsnr[ii]=np.sqrt(np.sum(snr**2))
    return medsnr, totsnr

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
            name="Count_Pixels"
        from desispec.image import Image as im
        MonitoringAlg.__init__(self,name,im,config,logger)
    def run(self,*args,**kwargs):
        if len(args) == 0 :
            raise qlexceptions.ParameterException("Missing input parameter")
        if not self.is_compatible(type(args[0])):
            raise qlexceptions.ParameterException("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))
        return self.count_pixels(args[0],**kwargs)
    def get_default_config(self):
        return {("Width",3.,"Width in sigmas")}

    def count_pixels(self,image,**kwargs):  #after pixel flat 
        """ 
        counts pixels above given threshold
        may be need to mask first for cosmics etc.
        image: desispec.image.Image like object
        n: number of sigma
        """
        if "Width" in kwargs:
            n=kwargs["Width"]
        rdnoise=image.readnoise
        #rdnoise=0
        darknoise=0 # should get from somewhere
        values=image.pix
        sigma=np.sqrt(values+darknoise**2+rdnoise**2)
        cut=np.where(values > n*sigma)
        count=cut[0].shape
        retval={}
        retval["VALUE"]={"COUNT":count}
        retval["EXPERT_LEVEL"]={"EXPERT_LEVEL":"OBSERVER"}
        return retval

class Find_Sky_Continuum(MonitoringAlg):
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="Find_Sky_Continuum"
        from  desispec.frame import Frame as fr
        MonitoringAlg.__init__(self,name,fr,config,logger)
    def run(self,*args,**kwargs):
        if len(args) == 0 :
            raise qlexceptions.ParameterException("Missing input parameter")
        if not self.is_compatible(type(args[0])):
            raise qlexceptions.ParameterException("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))
        if "FiberMap" not in kwargs:
            raise qlexceptions.ParameterException("Missing fibermap")
        return self.find_continuum(args[0],**kwargs)
    def get_default_config(self):
        return {("FiberMap","%%fibermap","Fibermap object"),
                ("Wmin",None,"Lower limit of wavelength for obtaining continuum"),
                ("Wmax",None,"Upper limit of wavelength for obtaining continuum")                }
    


    def find_continuum(self,Frame,**kwargs):  #after extraction i.e. boxcar 
        """
        frame: frame object: get by reading the frame file obtained from extraction
        fibermap: fibermap object: obtained from reading fibermap file to find sky fibers
        wmin and wmax: wavelength limits to obtain the continuum 
        """
        fibermap=kwargs["FiberMap"]
        wmin=None
        wmax=None
        if "Wmin" in kwargs: wmin=kwargs["Wmin"]
        if "Wmax" in kwargs: wmax=kwargs["Wmax"]
        
        objtype=fibermap['OBJTYPE'].copy()
        skyfiber,=np.where(objtype=='SKY')
        skyfiber=skyfiber[skyfiber<500] # only considering spectrograph 0
        sky_spectra=Frame.flux[skyfiber]
        nspec_sky=sky_spectra.shape[0]
        wave=Frame.wave
        if wmin is None: wmin=wave[0]
        if wmax is None: wmax=wave[-1]
        ## should average multiple sky fibers?
        contin=np.zeros((len(skyfiber)))
        print "No of skyfibers", len(skyfiber)
        def applysmoothingfilter(flux):
            return scipy.ndimage.filters.median_filter(flux,200) # bin range has to be optimized
        for i in range(len(skyfiber)):
            select,=np.where((wave > wmin)&(wave < wmax))
            contin[i]=np.mean(applysmoothingfilter(sky_spectra[i,select])) # smooth and mean from all ith sky spectra
        retval={}
        retval["VALUE"]={"SkyContinuum":contin,"SkyFiber":skyfiber}
        retval["EXPERT_LEVEL"]={"EXPERT_LEVEL":"EXPERT"}
    
        return retval

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
            name="Bias From Overscan"
        from desispec.image import Image as im
        MonitoringAlg.__init__(self,name,im,config,logger)
    def run(self,*args,**kwargs):
        if len(args) == 0 :
            raise qlexceptions.ParameterException("Missing input parameter")
        if not self.is_compatible(type(args[0])):
            raise qlexceptions.ParameterException("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))
        return self.bias_from_overscan(args[0],**kwargs)
    def get_default_config(self):
        return {("Quadrant",None,"CCD quadrant to work on (1-4)")}
    def bias_from_overscan(self,image,**kwargs): # dont use
        """
        image: something like raw image object
        just looking at pixel values now
        quadrant=the number of quadrant 1-4, dtype=int
        """

   # TODO define overscan region for each
   # xlo=[]
   # xhigh=[]
   # ylow=[]
   # yhigh=[]
        quadrant=None
        if "Quadrant" in kwargs: quadrant=kwargs["Quadrant"]
        if (quadrant==None):
   # read these values from configuration file?
            overreg=[xlo,xhigh,ylow, yhigh]
        else: 
            overreg=[xlo[quadrant],xhigh[quadrant],ylow[quadrant],yhigh[quadrant]]
   # put further cuts if apply like end col or rows.
        biasreg=image.image[overreg[0]:overreg[1],overreg[2]:overreg[3]]
   # Compute statistics of the bias region that only reject
   #  the 0.5% of smallest and largest values. (from sdssproc) 
        isort=np.sort(bias)
        nn = biasreg.shape[0]*biasreg.shape[1]
        ii = isort[long(0.005*nn) : long(0.995*nn)]
        biasval = mean(biasreg[ii])
   # also calculate readnoise
        rdnoise=biasreg[ii].std()
        retval={}
        retval["VALUE"]={"Bias Value":biasval,"Read Noise":rdnoise}
        retval["EXPERT_LEVEL"]={"EXPERT_LEVEL":"EXPERT"}
        
        return retval

class CountSpectralBins(MonitoringAlg):

    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="Count Bins above n"
        from  desispec.frame import Frame as fr
        MonitoringAlg.__init__(self,name,fr,config,logger)
    def run(self,*args,**kwargs):
        if len(args) == 0 :
            raise qlexceptions.ParameterException("Missing input parameter")
        if not self.is_compatible(type(args[0])):
            raise qlexceptions.ParameterException("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))
        input_frame=args[0]
        threshold=kwargs["thresh"]
        camera=kwargs["camera"]
        expid=kwargs["expid"]
        if "url" in kwargs:
            url=kwargs["url"]
        else:
            url=None
        if "qafig" in kwargs: qafig=kwargs["qafig"]
        else: qafig=None

        return self.run_qa(input_frame,threshold,camera,expid,url=url,qafig=qafig)


    def run_qa(self,input_frame,thresh,camera,expid,url=None,qafig=None):
        nspec=input_frame.flux.shape[0]
        counts=np.zeros(nspec)
        for ii in range(nspec):
            ok,=np.where(input_frame.flux[ii]>thresh)
            counts[ii]=ok.shape[0]

        #- return the qa dictionary 
        retval={}
        retval["ARM"]=camera[0]
        retval["SPECTROGRAPH"]=int(camera[1])
        retval["EXPID"]=expid
        retval["QANAME"]="Count_Bins"
        retval["THRESHOLD"]=thresh
        grid=np.gradient(input_frame.wave)
        if not np.all(grid[0]==grid[1:]): 
            log.info("grid_size is NOT UNIFORM")
        grid_size=grid[0]
        retval["WAVE_GRID"]=grid_size
        retval["VALUE"]={"CNTS_ABOVE_THRESH":counts}

        #- http post if needed
        if url is not None:
            try: 
                import requests
                response=requests.get(url)
                #- Check if the api has json
                api=response.json()
                #- proceed with post
                response=requests.post(url,json=retval) #- no need of json.dumps as the api has it
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
            name="Calculate Signal-to-Noise ratio"
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
        ampboundary=[250,input_frame.wave.shape[0]/2] #- TODO propagate amplifier boundary from kwargs. Dividing into quadrants for now. This may come from config also
        if "url" in kwargs:
             url=kwargs["url"]
        else: 
             url=None

        if "qafig" in kwargs: qafig=kwargs["qafig"]
        else: qafig = None

        return self.run_qa(input_frame,ampboundary,camera,expid,url=url,qafig=qafig)

    def run_qa(self,input_frame,ampboundary,camera,expid,url=None,qafig=None):

        #- parameters (adopting from offline qa)
        # sky_dict={'PCHI_RESID': 0.05, 'PER_RESID': 95.0}
        # qadict=qa_skysub(sky_dict,input_frame,skymodel)

        #- return values
        retval={}
        retval["ARM"]=camera[0]
        retval["SPECTROGRAPH"]=int(camera[1])
        retval["EXPID"]=expid
        retval["QANAME"]="SNR"

        medsnr, totsnr=SN_ratio(input_frame.flux,input_frame.ivar)
        retval["VALUE"]={"MED_SNR":medsnr,"TOT_SNR":totsnr}
        
        if ampboundary is not None:
            
            import desispec.frame as frame
            import desispec.sky as sky
            
            top_left_frame=frame.Frame(input_frame.wave[ampboundary[1]:],input_frame.flux[:ampboundary[0],ampboundary[1]:], input_frame.ivar[:ampboundary[0], ampboundary[1]:],fibermap=input_frame.fibermap)
           
            medsnr01,totsnr01=SN_ratio(top_left_frame.flux,top_left_frame.ivar)
            average01=np.mean(medsnr01)
            tot01=np.mean(totsnr01)

            bottom_left_frame=frame.Frame(input_frame.wave[:ampboundary[1]],input_frame.flux[:ampboundary[0],:ampboundary[1]], input_frame.ivar[:ampboundary[0],:ampboundary[1]],fibermap=input_frame.fibermap)

            medsnr00,totsnr00=SN_ratio(bottom_left_frame.flux,bottom_left_frame.ivar)
            average00=np.mean(medsnr00)
            tot00=np.mean(totsnr00)
            if ampboundary[0] > 250:

                bottom_right_frame=frame.Frame(input_frame.wave[:ampboundar[1]],input_frame.flux[ampboundary[0]:,:ampboundary[1]], input_frame.ivar[ampboundary[0]:,:ampboundary[1]],fibermap=input_frame.fibermap)
                
                medsnr10,totsnr10=SN_ratio(bottom_right_frame.flux,bottom_right_frame.ivar)
                average10=np.mean(medsnr10)
                tot10=np.mean(totsnr10)
            else:
                average10=0.
                tot10=0.
            if ampboundary[0]> 250: #- only if nspec> 250 for the right quadrants

                top_right_frame=frame.Frame(input_frame.wave[ampboundary[1]:],input_frame.flux[ampboundary[0]:,ampboundary[1]:], input_frame.ivar[ampboundary[0]:,ampboundary[1]:],fibermap=input_frame.fibermap)
                
                medsnr11,totsnr11=SN_ratio(top_right_frame.flux,top_right_frame.ivar)
                average11=np.mean(medsnr11)
                tot11=np.mean(totsnr11)
            else:
                average11=0.
                tot11=0.

            average_amp=np.array([average01,average00,average10,average11])
            tot_amp=np.array([tot01,tot00,tot10,tot11])

            retval["VALUE"]={"MED_SNR":medsnr,"TOT_SNR":totsnr,"TOT_AMP_SNR":tot_amp,"MED_AMP_SNR":average_amp}

        #- http post if valid
        if url is not None:
            try: 
                import requests
                response=requests.get(url)
                #- Check if the api has json
                api=response.json()
                #- proceed with post
                response=requests.post(url,json=retval) #- no need of json.dumps as the api has it
            except:
             
                log.info("Skipping HTTP post...")
            
        return retval

        if qafig is not None:
            from desispec.qa.qa_plots_ql import plot_SNR
            plot_SNR(retval,qafig)         
            log.info("Output QA fig %s"%qafig)

    def get_default_config(self):
        return {}

