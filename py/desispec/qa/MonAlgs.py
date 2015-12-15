""" 
Monitoring algorithms for Quicklook pipeline

"""

import numpy as np
import scipy.ndimage
from desispec.qlpipeline.MAs import MonitoringAlg
from desispec.qlpipeline import QLExceptions
# Evaluate rms of pixel values after dark subtraction
class Get_RMS(MonitoringAlg):
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="Get_RMS"
        from desispec.image import Image as im
        MonitoringAlg.__init__(self,name,im,config,logger)
    def run(self,*args,**kwargs):
        if len(args) == 0 :
            raise QLExceptions.ParameterException("Missing input parameter")
        if not self.is_compatible(type(args[0])):
            raise QLExceptions.ParameterException("Incompatible parameter type. Was expecting desispec.image.Image got %s"%(type(args[0])))
        return self.get_rms(args[0])
    def get_default_config(self):
        return {}
    def get_rms(self,image): #after dark
        """ 
        image: desispec.image.Image like object
        attributes: image(image, ivar, mask, readnoise,
        camera, meta)
        
        """
        value=image.pix
    #TODO might need to use mask to filter?
        vals=value.ravel()
        rms=np.std(vals)
        retVal={}
        retVal["OBSERVER"]={"RMS":rms}
        retVal["EXPERT"]={"RMS":rms}
        retVal["USER"]={"RMS":rms}
        return retVal

class Count_Pixels(MonitoringAlg):
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="Count_Pixels"
        from desispec.image import Image as im
        MonitoringAlg.__init__(self,name,im,config,logger)
    def run(self,*args,**kwargs):
        if len(args) == 0 :
            raise QLExceptions.ParameterException("Missing input parameter")
        if not self.is_compatible(type(args[0])):
            raise QLExceptions.ParameterException("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))
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
        retVal={}
        retVal["OBSERVER"]={"COUNT":count}
        retVal["EXPERT"]={"COUNT":count}
        retVal["USER"]={"COUNT":count}
        return retVal

class Find_Sky_Continuum(MonitoringAlg):
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="Find_Sky_Continuum"
        from  desispec.frame import Frame as fr
        MonitoringAlg.__init__(self,name,fr,config,logger)
    def run(self,*args,**kwargs):
        if len(args) == 0 :
            raise QLExceptions.ParameterException("Missing input parameter")
        if not self.is_compatible(type(args[0])):
            raise QLExceptions.ParameterException("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))
        if "FiberMap" not in kwargs:
            raise QLExceptions.ParameterException("Missing fibermap")
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
        def applySmoothingFilter(flux):
            return scipy.ndimage.filters.median_filter(flux,200) # bin range has to be optimized
        for i in range(len(skyfiber)):
            select,=np.where((wave > wmin)&(wave < wmax))
            contin[i]=np.mean(applySmoothingFilter(sky_spectra[i,select])) # smooth and mean from all ith sky spectra
        retVal={}
        retVal["OBSERVER"]={"SkyContinuum":contin,"SkyFiber":skyfiber}
        retVal["EXPERT"]={"SkyContinuum":contin,"SkyFiber":skyfiber}
        retVal["USER"]={"SkyContinuum":contin,"SkyFiber":skyfiber}

    
        return retVal

class Count_Fibers(MonitoringAlg):
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="Count_Fibers"
        from  desispec.frame import Frame as fr
        MonitoringAlg.__init__(self,name,fr,config,logger)
    def run(self,*args,**kwargs):
        if len(args) == 0 :
            raise QLExceptions.ParameterException("Missing input parameter")
        if not self.is_compatible(type(args[0])):
            raise QLExceptions.ParameterException("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))
        return self.count_fibers(args[0])
    def get_default_config(self):
        return {}
                            
    def count_fibers(self,frame): # after extraction, i.e. boxcar
        """
        image: an image that has mask keyword like desispec.image.Image object
        """
    
        good=np.where(frame['MASK'].data[:,1]==0) # Although this seems to be bin mask ?? 
        count=good.shape[0]
        retVal={}
        retVal["OBSERVER"]={"Count":count}
        retVal["EXPERT"]={"Count":count}
        retVal["USER"]={"Count":count}
        return retVal


class Bias_From_Overscan(MonitoringAlg):
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="Bias From Overscan"
        from desispec.image import Image as im
        MonitoringAlg.__init__(self,name,im,config,logger)
    def run(self,*args,**kwargs):
        if len(args) == 0 :
            raise QLExceptions.ParameterException("Missing input parameter")
        if not self.is_compatible(type(args[0])):
            raise QLExceptions.ParameterException("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))
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
        retVal={}
        retVal["OBSERVER"]={"Bias Value":biasval,"Read Noise":rdnoise}
        retVal["EXPERT"]={"Bias Value":biasval,"Read Noise":rdnoise}
        retVal["USER"]={"Bias Value":biasval,"Read Noise":rdnoise}
        
        return retVal

class Calculate_SNR(MonitoringAlg):
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="Calculate Signal-to-Noise ratio"
        from  desispec.frame import Frame as fr
        MonitoringAlg.__init__(self,name,fr,config,logger)
    def run(self,*args,**kwargs):
        if len(args) == 0 :
            raise QLExceptions.ParameterException("Missing input parameter")
        if not self.is_compatible(type(args[0])):
            raise QLExceptions.ParameterException("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))
        return self.calculate_snr(args[0])
    def get_default_config(self):
        return {}

    def calculate_snr(self,frame): #after extraction, boxcar
        """ input is desispec.frame like object
        output: median of S/N (bin) for each fiber
        total snr calculated considering bin by bin uncorrelated S/N 
        """
        flux=frame.flux
        wave=frame.wave
        ivar=frame.ivar
        medsnr=np.zeros(flux.shape[0])
        snrtot=np.zeros(flux.shape[0])
        for ii in range(flux.shape[0]):
            signalmask=flux[ii,:]>0
            snr=flux[ii,signalmask]*np.sqrt(ivar[ii,signalmask]) # actually flux should be sky subtracted for true S/N?
            medsnr[ii]=np.median(snr)
            snrtot[ii]=np.sqrt(np.sum(snr**2))
        retVal={}
        retVal["OBSERVER"]={"Median SNR":medsnr,"Total SNR":snrtot}
        retVal["EXPERT"]={"Median SNR":medsnr,"Total SNR":snrtot}
        retVal["USER"]={"Median SNR":medsnr,"Total SNR":snrtot}
        return retVal
