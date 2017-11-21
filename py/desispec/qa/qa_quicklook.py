""" 
Monitoring algorithms for Quicklook pipeline

"""

import numpy as np
import scipy.ndimage
import yaml
from desispec.quicklook.qas import MonitoringAlg, QASeverity
from desispec.quicklook import qlexceptions
from desispec.quicklook import qllogger
import os,sys
import datetime
from astropy.time import Time
from desispec.qa import qalib
from desispec.io import qa

qlog=qllogger.QLLogger("QuickLook",0)
log=qlog.getlog()


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
            log.debug("Environment variables are set for QLF. Now trying HTTP post.")
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
                log.error("Skipping HTTP post... Exception",exc_info=true)

    else:   
        log.warning("Skipping QLF. QLF_API_URL must be set as environment variable")

class Get_RMS(MonitoringAlg):
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="RMS"
        from desispec.image import Image as im
        kwargs=config['kwargs']
        parms=kwargs['param']
        key=kwargs['refKey']
        key="NOISE_AMP"
        kwargs["SAMI_RESULTKEY"]=key
        kwargs["SAMI_QASTATUSKEY"]="NOISE_STAT"

        if "ReferenceMetrics" in kwargs:
            r=kwargs["ReferenceMetrics"]
            if key in r:
                kwargs["REFERENCE"]=r[key]
        if "NOISE_WARN_RANGE" in parms and "NOISE_NORMAL_RANGE" in parms:
            kwargs["RANGES"]=[(np.asarray(parms["NOISE_WARN_RANGE"]),QASeverity.WARNING),
                              (np.asarray(parms["NOISE_NORMAL_RANGE"]),QASeverity.NORMAL)]# sorted by most severe to least severe 
        
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

        if "ReferenceMetrics" in kwargs: refmetrics=kwargs["ReferenceMetrics"]
        else: refmetrics=None

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
        
        return self.run_qa(input_image,paname=paname,amps=amps,qafile=qafile,qafig=qafig, param=param, qlf=qlf, refmetrics=refmetrics)

    def run_qa(self,image,paname=None,amps=False,qafile=None, qafig=None,param=None,qlf=False, refmetrics=None):
        retval={}
        retval["EXPID"] = '{0:08d}'.format(image.meta["EXPID"])
        retval["PANAME"] = paname
        retval["QATIME"] = datetime.datetime.now().isoformat()
        retval["CAMERA"] = image.meta["CAMERA"]
        retval["PROGRAM"] = image.meta["PROGRAM"]
        retval["FLAVOR"] = image.meta["FLAVOR"]
        retval["NIGHT"] = image.meta["NIGHT"]
        kwargs=self.config['kwargs']
        if "REFERENCE" in kwargs:
            retval['NOISE_AMP_REF']=kwargs["REFERENCE"]

        # return rms values in rms/sqrt(exptime)
        rmsccd=qalib.getrms(image.pix/np.sqrt(image.meta["EXPTIME"])) #- should we add dark current and/or readnoise to this as well?

        if param is None:
            log.debug("Param is None. Using default param instead")
            param = {
                "NOISE_NORMAL_RANGE":[-1.0, 1.0],
                "NOISE_WARN_RANGE":[-2.0, 2.0]
                }

        retval["PARAMS"] = param

        expnum=[]
        rms_row=[]
        rms_amps=[]
        rms_over_amps=[]
        overscan_values=[]
        #- get amp/overcan boundary in pixels
        from desispec.preproc import _parse_sec_keyword
        for kk in ['1','2','3','4']:
            thisampboundary=_parse_sec_keyword(image.meta["CCDSEC"+kk])
            thisoverscanboundary=_parse_sec_keyword(image.meta["BIASSEC"+kk])
            for i in range(image.pix[thisoverscanboundary].shape[0]):
                rmsrow = qalib.getrms(image.pix[thisoverscanboundary][i]/np.sqrt(image.meta["EXPTIME"]))
                rms_row.append(rmsrow)
            rms_thisover_thisamp=qalib.getrms(image.pix[thisoverscanboundary]/np.sqrt(image.meta["EXPTIME"]))
            rms_thisamp=qalib.getrms(image.pix[thisampboundary]/np.sqrt(image.meta["EXPTIME"]))
            rms_amps.append(rms_thisamp)
            rms_over_amps.append(rms_thisover_thisamp)
        rmsover=np.max(rms_over_amps)

        rmsdiff_err=[]
        if amps:
            rms_amps=[]
            rms_over_amps=[]
            overscan_values=[]
            #- get amp/overcan boundary in pixels
            from desispec.preproc import _parse_sec_keyword
            for kk in ['1','2','3','4']:
                thisampboundary=_parse_sec_keyword(image.meta["CCDSEC"+kk])
                thisoverscanboundary=_parse_sec_keyword(image.meta["BIASSEC"+kk])
                rms_thisover_thisamp=qalib.getrms(image.pix[thisoverscanboundary]/np.sqrt(image.meta["EXPTIME"]))
                thisoverscan_values=np.ravel(image.pix[thisoverscanboundary]/np.sqrt(image.meta["EXPTIME"]))
                rms_thisamp=qalib.getrms(image.pix[thisampboundary]/np.sqrt(image.meta["EXPTIME"]))
                rms_amps.append(rms_thisamp)
                rms_over_amps.append(rms_thisover_thisamp)
                overscan_values+=thisoverscan_values.tolist()
            rmsover=np.std(overscan_values)

            retval["METRICS"]={"RMS":rmsccd,"NOISE":rmsover,"RMS_AMP":np.array(rms_amps),"RESULT":rmsccd,"NOISE_AMP":np.array(rms_over_amps),"RMS_ROW":rms_row,"NOISE_STAT":rmsdiff_err,"EXPNUM_WARN":expnum}
        else:
            retval["METRICS"]={"RMS":rmsccd,"NOISE":rmsover,"RMS_ROW":rms_row,"NOISE_STAT":rmsdiff_err,"EXPNUM_WARN":expnum}

        if qlf:
            qlf_post(retval)  

        if qafile is not None:
            outfile = qa.write_qa_ql(qafile,retval)
            log.debug("Output QA data is in {}".format(outfile))
        if qafig is not None:
            from desispec.qa.qa_plots_ql import plot_RMS
            plot_RMS(retval,qafig)            
            log.debug("Output QA fig {}".format(qafig))      

        return retval    

    def get_default_config(self):
        return {}

class Count_Pixels(MonitoringAlg):
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="COUNTPIX"
        from desispec.image import Image as im
        kwargs=config['kwargs']
        parms=kwargs['param']
        key="NPIX_AMP"
        kwargs["SAMI_RESULTKEY"]=key
        kwargs["SAMI_QASTATUSKEY"]="NPIX_STAT"
        if "ReferenceMetrics" in kwargs:
            r=kwargs["ReferenceMetrics"]
            if key in r:
                kwargs["REFERENCE"]=r[key]
        if "NPIX_WARN_RANGE" in parms and "NPIX_NORMAL_RANGE" in parms:
            kwargs["RANGES"]=[(np.asarray(parms["NPIX_WARN_RANGE"]),QASeverity.WARNING),
                              (np.asarray(parms["NPIX_NORMAL_RANGE"]),QASeverity.NORMAL)]# sorted by most severe to least severe 
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

        if "ReferenceMetrics" in kwargs: refmetrics=kwargs["ReferenceMetrics"]
        else: refmetrics=None

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

        return self.run_qa(input_image,paname=paname,amps=amps,qafile=qafile,qafig=qafig, param=param, qlf=qlf, refmetrics=refmetrics)

    def run_qa(self,image,paname=None,amps=False,qafile=None,qafig=None, param=None, qlf=False, refmetrics=None):
        retval={}
        retval["PANAME"] = paname
        retval["QATIME"] = datetime.datetime.now().isoformat()
        retval["EXPID"] = '{0:08d}'.format(image.meta["EXPID"])
        retval["CAMERA"] = image.meta["CAMERA"]
        retval["PROGRAM"] = image.meta["PROGRAM"]
        retval["FLAVOR"] = image.meta["FLAVOR"]
        retval["NIGHT"] = image.meta["NIGHT"]
        kwargs=self.config['kwargs']
        if "REFERENCE" in kwargs:
            retval['NPIX_AMP_REF']=kwargs["REFERENCE"]

        if param is None:
            log.debug("Param is None. Using default param instead")
            param = {
                 "CUTLO":3,   # low threshold for number of counts in sigmas
                 "CUTHI":10, 
                 "NPIX_NORMAL_RANGE":[200.0, 500.0],
                 "NPIX_WARN_RANGE":[50.0, 650.0]
                 }

        retval["PARAMS"] = param

        #- get the counts over entire CCD in counts per second
        npixlo=qalib.countpix(image.pix,nsig=param['CUTLO']) #- above 3 sigma in counts
        npixhi=qalib.countpix(image.pix,nsig=param['CUTHI']) #- above 10 sigma in counts

        npix_err=[]
        #- get the counts for each amp
        if amps:
            npixlo_amps=[]
            npixhi_amps=[]
            #- get amp boundary in pixels
            from desispec.preproc import _parse_sec_keyword
            for kk in ['1','2','3','4']:
                ampboundary=_parse_sec_keyword(image.meta["CCDSEC"+kk])
                npixlo_thisamp=qalib.countpix(image.pix[ampboundary]/image.meta["EXPTIME"],nsig=param['CUTLO'])
                npixlo_amps.append(npixlo_thisamp)
                npixhi_thisamp=qalib.countpix(image.pix[ampboundary]/image.meta["EXPTIME"],nsig=param['CUTHI'])
                npixhi_amps.append(npixhi_thisamp)

            retval["METRICS"]={"NPIX_LOW":npixlo,"NPIX_HIGH":npixhi,"NPIX_AMP": npixlo_amps,"NPIX_HIGH_AMP": npixhi_amps,"NPIX_STAT":npix_err}
        else:
            retval["METRICS"]={"NPIX_LOW":npixlo,"NPIX_HIGH":npixhi,"NPIX_STAT":npix_err}

        if qlf:
            qlf_post(retval)      

        if qafile is not None:
            outfile = qa.write_qa_ql(qafile,retval)
            log.debug("Output QA data is in {}".format(outfile))
        if qafig is not None:
            from desispec.qa.qa_plots_ql import plot_countpix
            plot_countpix(retval,qafig)
            
            log.debug("Output QA fig {}".format(qafig))      

        return retval    

    def get_default_config(self):
        return {}

class Integrate_Spec(MonitoringAlg):
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="INTEG"
        from desispec.frame import Frame as fr
        kwargs=config['kwargs']
        parms=kwargs['param']
        key="INTEG_AVG"
        kwargs["SAMI_RESULTKEY"]=key
        kwargs["SAMI_QASTATUSKEY"]="MAGDIFF_STAT"
        if "ReferenceMetrics" in kwargs:
            r=kwargs["ReferenceMetrics"]
            if key in r:
                kwargs["REFERENCE"]=r[key]

        if "MAGDIFF_WARN_RANGE" in parms and "MAGDIFF_NORMAL_RANGE" in parms:
            kwargs["RANGES"]=[(np.asarray(parms["MAGDIFF_WARN_RANGE"]),QASeverity.WARNING),
                              (np.asarray(parms["MAGDIFF_NORMAL_RANGE"]),QASeverity.NORMAL)]# sorted by most severe to least severe 
        MonitoringAlg.__init__(self,name,fr,config,logger)
    def run(self,*args,**kwargs):
        if len(args) == 0 :
            raise qlexceptions.ParameterException("Missing input parameter")
        if not self.is_compatible(type(args[0])):
            raise qlexceptions.ParameterException("Incompatible input. Was expecting {}, got {}".format(type(self.__inpType__),type(args[0])))

        fibermap=kwargs['FiberMap']
        input_frame=args[0]

        if "paname" not in kwargs:
            paname=None
        else:
            paname=kwargs["paname"]

        if "ReferenceMetrics" in kwargs: refmetrics=kwargs["ReferenceMetrics"]
        else: refmetrics=None

        amps=False
        if "amps" in kwargs:
            amps=kwargs["amps"]

        if "param" in kwargs: param=kwargs["param"]
        else: param=None

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
        return self.run_qa(fibermap,input_frame,paname=paname,amps=amps, dict_countbins=dict_countbins, qafile=qafile,qafig=qafig, param=param, qlf=qlf, refmetrics=refmetrics)

    def run_qa(self,fibermap,frame,paname=None,amps=False,dict_countbins=None, qafile=None,qafig=None, param=None, qlf=False, refmetrics=None):
        retval={}
        retval["PANAME" ] = paname
        retval["QATIME"] = datetime.datetime.now().isoformat()
        retval["EXPID"] = '{0:08d}'.format(frame.meta["EXPID"])
        retval["CAMERA"] = frame.meta["CAMERA"]
        retval["PROGRAM"] = frame.meta["PROGRAM"]
        retval["FLAVOR"] = frame.meta["FLAVOR"]
        retval["NIGHT"] = frame.meta["NIGHT"]
        kwargs=self.config['kwargs']
        if "REFERENCE" in kwargs:
            retval['MAGDIFF_TGT_REF']=kwargs["REFERENCE"]

        ra = fibermap["RA_TARGET"]
        dec = fibermap["DEC_TARGET"]

        #- get the integrals for all fibers
        flux=frame.flux
        wave=frame.wave
        integrals=np.zeros(flux.shape[0])

        for ii in range(len(integrals)):
            integrals[ii]=qalib.integrate_spec(wave,flux[ii])
        
        #- average integrals over fibers of each object type and get imaging magnitudes
        integ_avg_tgt=[]
        mag_avg_tgt=[]
        for T in ["ELG","QSO","LRG","STD"]:
            fibers=np.where(frame.fibermap['OBJTYPE']==T)[0]
            if len(fibers) < 1:
                log.warning("no {} fibers found.".format(T))
            magnitudes=frame.fibermap['MAG'][fibers]
            mag_avg=np.mean(magnitudes)
            mag_avg_tgt.append(mag_avg)
            integ=integrals[fibers]
            integ_avg=np.mean(integ)
            integ_avg_tgt.append(integ_avg)
            if T == "STD":
                starfibers=fibers
                int_stars=integ
                int_average=integ_avg

        # simple, temporary magdiff calculation (to be corrected...)
        magdiff_avg=[]
        for i in range(len(mag_avg_tgt)):
            mag_fib=-2.5*np.log(integ_avg_tgt[i]/frame.meta["EXPTIME"])+30.
            if mag_avg_tgt[i] != np.nan:
                magdiff=mag_fib-mag_avg_tgt[i]
            else:
                magdiff=nan
            magdiff_avg.append(magdiff)

        if param is None:
            log.debug("Param is None. Using default param instead")
            param = {
                "MAGDIFF_NORMAL_RANGE":[-0.5, 0.5],
                "MAGDIFF_WARN_RANGE":[-1.0, 1.0]
                }

        retval["PARAMS"] = param

        magdiff_avg_amp = [0.0]

        magdiff_err=[]
        #- get the counts for each amp
        if amps:

            #- get the fiducial boundary
            leftmax = dict_countbins["LEFT_MAX_FIBER"]
            rightmin = dict_countbins["RIGHT_MIN_FIBER"]
            bottommax = dict_countbins["BOTTOM_MAX_WAVE_INDEX"]
            topmin = dict_countbins["TOP_MIN_WAVE_INDEX"]

            fidboundary = qalib.slice_fidboundary(frame,leftmax,rightmin,bottommax,topmin)

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
                        integ_thisamp[ii]=qalib.integrate_spec(wave,stdflux_thisamp[ii])
                    int_avg_amps[amp]=np.mean(integ_thisamp)

            retval["METRICS"]={"RA":ra,"DEC":dec, "INTEG":int_stars, "INTEG_AVG":int_average,"INTEG_AVG_AMP":int_avg_amps, "STD_FIBERID": starfibers.tolist(),"MAGDIFF_TGT":magdiff_avg,"MAGDIFF_AVG_AMP":magdiff_avg_amp,"MAGDIFF_STAT":magdiff_err}

        else:
            retval["METRICS"]={"RA":ra,"DEC":dec, "INTEG":int_stars,"INTEG_AVG":int_average,"STD_FIBERID":starfibers.tolist(),"MAGDIFF_TGT":magdiff_avg,"MAGDIFF_STAT":magdiff_err}

        if qlf:
            qlf_post(retval) 

        if qafile is not None:
            outfile = qa.write_qa_ql(qafile,retval)
            log.debug("Output QA data is in {}".format(outfile))
        if qafig is not None:
            from desispec.qa.qa_plots_ql import plot_integral
            plot_integral(retval,qafig)
            
            log.debug("Output QA fig {}".format(qafig))      

        return retval    

    def get_default_config(self):
        return {}
 
 
class Sky_Continuum(MonitoringAlg):
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="SKYCONT"
        from  desispec.frame import Frame as fr
        kwargs=config['kwargs']
        parms=kwargs['param']
        key="SKYCONT"
        kwargs["SAMI_RESULTKEY"]=key
        kwargs["SAMI_QASTATUSKEY"]="SKYCONT_STAT"
        if "ReferenceMetrics" in kwargs:
            r=kwargs["ReferenceMetrics"]
            if key in r:
                kwargs["REFERENCE"]=r[key]
    
        if "SKYCONT_WARN_RANGE" in parms and "SKYCONT_NORMAL_RANGE" in parms:
            kwargs["RANGES"]=[(np.asarray(parms["SKYCONT_WARN_RANGE"]),QASeverity.WARNING),
                              (np.asarray(parms["SKYCONT_NORMAL_RANGE"]),QASeverity.NORMAL)]# sorted by most severe to least severe 
        MonitoringAlg.__init__(self,name,fr,config,logger)
    def run(self,*args,**kwargs):
        if len(args) == 0 :
            raise qlexceptions.ParameterException("Missing input parameter")
        if not self.is_compatible(type(args[0])):
            raise qlexceptions.ParameterException("Incompatible input. Was expecting {}, got {}".format(type(self.__inpType__),type(args[0])))

        fibermap=kwargs['FiberMap']
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

        if "ReferenceMetrics" in kwargs: refmetrics=kwargs["ReferenceMetrics"]
        else: refmetrics=None

        amps=False
        if "amps" in kwargs:
            amps=kwargs["amps"]

        if "param" in kwargs: param=kwargs["param"]
        else: param=None

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
        return self.run_qa(fibermap,input_frame,wrange1=wrange1,wrange2=wrange2,paname=paname,amps=amps, dict_countbins=dict_countbins,qafile=qafile,qafig=qafig, param=param, qlf=qlf, refmetrics=refmetrics)

    def run_qa(self,fibermap,frame,wrange1=None,wrange2=None,
               paname=None,amps=False,dict_countbins=None,
               qafile=None,qafig=None, param=None, qlf=False, 
               refmetrics=None):

        #- qa dictionary 
        retval={}
        retval["PANAME" ]= paname
        retval["QATIME"] = datetime.datetime.now().isoformat()
        retval["EXPID"] = '{0:08d}'.format(frame.meta["EXPID"])
        retval["CAMERA"] = frame.meta["CAMERA"]
        retval["PROGRAM"] = frame.meta["PROGRAM"]
        retval["FLAVOR"] = frame.meta["FLAVOR"]
        retval["NIGHT"] = frame.meta["NIGHT"]
        kwargs=self.config['kwargs']
        if "REFERENCE" in kwargs:
            retval['SKYCONT_REF']=kwargs["REFERENCE"]

        ra = fibermap["RA_TARGET"]
        dec = fibermap["DEC_TARGET"]

        if param is None:
            log.debug("Param is None. Using default param instead")
            param = {
                "B_CONT":[(4000, 4500), (5250, 5550)],
                "R_CONT":[(5950, 6200), (6990, 7230)],
                "Z_CONT":[(8120, 8270), (9110, 9280)],
                "SKYCONT_NORMAL_RANGE":[100.0, 400.0],
                "SKYCONT_WARN_RANGE":[50.0, 600.0]
                }

        retval["PARAMS"] = param

        skyfiber, contfiberlow, contfiberhigh, meancontfiber, skycont = qalib.sky_continuum(
            frame, wrange1, wrange2)

        skycont_err = []
        if amps:
            leftmax = dict_countbins["LEFT_MAX_FIBER"]
            rightmin = dict_countbins["RIGHT_MIN_FIBER"]
            bottommax = dict_countbins["BOTTOM_MAX_WAVE_INDEX"]
            topmin = dict_countbins["TOP_MIN_WAVE_INDEX"]

            fidboundary = qalib.slice_fidboundary(frame,leftmax,rightmin,bottommax,topmin)

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

            retval["METRICS"]={"RA":ra,"DEC":dec, "SKYFIBERID": skyfiber.tolist(), "SKYCONT":skycont, "SKYCONT_FIBER":meancontfiber, "SKYCONT_AMP":skycont_amps, "SKYCONT_STAT":skycont_err}

        else: 
            retval["METRICS"]={"RA":ra,"DEC":dec, "SKYFIBERID": skyfiber.tolist(), "SKYCONT":skycont, "SKYCONT_FIBER":meancontfiber, "SKYCONT_STAT":skycont_err}

        if qlf:
            qlf_post(retval)    

        if qafile is not None:
            outfile = qa.write_qa_ql(qafile,retval)
            log.debug("Output QA data is in {}".format(outfile))

        if qafig is not None:
            from desispec.qa.qa_plots_ql import plot_sky_continuum
            plot_sky_continuum(retval,qafig)
            
            log.debug("Output QA fig {}".format(qafig))                   
        
        return retval

    def get_default_config(self):
        return {}


class Sky_Peaks(MonitoringAlg):
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="SKYPEAK"
        from  desispec.frame import Frame as fr
        kwargs=config['kwargs']
        parms=kwargs['param']
        key="SUMCOUNT_MED_SKY"
        kwargs["SAMI_RESULTKEY"]=key
        kwargs["SAMI_QASTATUSKEY"]="PEAKCOUNT_STAT"
        if "ReferenceMetrics" in kwargs:
            r=kwargs["ReferenceMetrics"]
            if key in r:
                kwargs["REFERENCE"]=r[key]

        if "PEAKCOUNT_WARN_RANGE" in parms and "PEAKCOUNT_NORMAL_RANGE" in parms:
            kwargs["RANGES"]=[(np.asarray(parms["PEAKCOUNT_WARN_RANGE"]),QASeverity.WARNING),
                              (np.asarray(parms["PEAKCOUNT_NORMAL_RANGE"]),QASeverity.NORMAL)]# sorted by most severe to least severe
        MonitoringAlg.__init__(self,name,fr,config,logger)
    def run(self,*args,**kwargs):
        if len(args) == 0 :
            raise qlexceptions.ParameterException("Missing input parameter")
        if not self.is_compatible(type(args[0])):
            raise qlexceptions.ParameterException("Incompatible parameter type. Was expecting desispec.image.Image, got {}".format(type(args[0])))

        fibermap=kwargs['FiberMap']
        input_frame=args[0]

        if "paname" not in kwargs:
            paname=None
        else:
            paname=kwargs["paname"]

        if "ReferenceMetrics" in kwargs: refmetrics=kwargs["ReferenceMetrics"]
        else: refmetrics=None

        amps=False
        if "amps" in kwargs:
            amps=kwargs["amps"]

        if "param" in kwargs: param=kwargs["param"]
        else: param=None

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

        return self.run_qa(fibermap,input_frame,paname=paname,amps=amps,psf=psf, qafile=qafile, qafig=qafig, param=param, qlf=qlf, refmetrics=refmetrics)

    def run_qa(self,fibermap,frame,paname=None,amps=False,psf=None, qafile=None,qafig=None, param=None, qlf=False, refmetrics=None):
        retval={}
        retval["PANAME"] = paname
        retval["QATIME"] = datetime.datetime.now().isoformat()
        retval["EXPID"] = '{0:08d}'.format(frame.meta["EXPID"])
        retval["CAMERA"] = camera = frame.meta["CAMERA"]
        retval["PROGRAM"] = frame.meta["PROGRAM"]
        retval["FLAVOR"] = frame.meta["FLAVOR"]
        retval["NIGHT"] = frame.meta["NIGHT"]
        kwargs=self.config['kwargs']
        if "REFERENCE" in kwargs:
            retval['PEAKCOUNT_REF']=kwargs["REFERENCE"]

        ra = fibermap["RA_TARGET"]
        dec = fibermap["DEC_TARGET"]

        # define sky peaks and wavelength region around peak flux to be integrated
        dw=2
        b_peaks=np.array([3914.4,5199.3,5201.8])
        r_peaks=np.array([6301.9,6365.4,7318.2,7342.8,7371.3])
        z_peaks=np.array([8401.5,8432.4,8467.5,9479.4,9505.6,9521.8])

        nspec_counts=[]
        sky_counts=[]
        nspec_counts_rms=[]
        sky_counts_rms=[]
        rms_skyspec_amp=[]
        amp1=[]
        amp2=[]
        amp3=[]
        amp4=[]
        rmsamp1=[]
        rmsamp2=[]
        rmsamp3=[]
        rmsamp4=[]
        for i in range(frame.flux.shape[0]):
            if camera[0]=="b":
                iwave1=np.argmin(np.abs(frame.wave-b_peaks[0]))
                iwave2=np.argmin(np.abs(frame.wave-b_peaks[1]))
                iwave3=np.argmin(np.abs(frame.wave-b_peaks[2]))
                peak1_flux=np.trapz(frame.flux[i,iwave1-dw:iwave1+dw+1])
                peak2_flux=np.trapz(frame.flux[i,iwave2-dw:iwave2+dw+1])
                peak3_flux=np.trapz(frame.flux[i,iwave3-dw:iwave3+dw+1])
                sum_counts=np.sum((peak1_flux+peak2_flux+peak3_flux)/frame.meta["EXPTIME"])
                sum_counts_rms=np.sum((peak1_flux+peak2_flux+peak3_flux)/np.sqrt(frame.meta["EXPTIME"]))
                nspec_counts.append(sum_counts)
                nspec_counts_rms.append(sum_counts_rms)
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
                sum_counts=np.sum((peak1_flux+peak2_flux+peak3_flux+peak4_flux+peak5_flux)/frame.meta["EXPTIME"])
                sum_counts_rms=np.sum((peak1_flux+peak2_flux+peak3_flux+peak4_flux+peak5_flux)/np.sqrt(frame.meta["EXPTIME"]))
                nspec_counts.append(sum_counts)
                nspec_counts_rms.append(sum_counts_rms)
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
                sum_counts=np.sum((peak1_flux+peak2_flux+peak3_flux+peak4_flux+peak5_flux+peak6_flux)/frame.meta["EXPTIME"])
                sum_counts_rms=np.sum((peak1_flux+peak2_flux+peak3_flux+peak4_flux+peak5_flux+peak6_flux)/np.sqrt(frame.meta["EXPTIME"]))
                nspec_counts.append(sum_counts)
                nspec_counts_rms.append(sum_counts_rms)

            if frame.fibermap['OBJTYPE'][i]=='SKY':
                sky_counts.append(sum_counts)

                if amps:
                    if frame.fibermap['FIBER'][i]<240:
                        if camera[0]=="b":
                            amp1_flux=peak1_flux/frame.meta["EXPTIME"]
                            amp3_flux=np.sum((peak2_flux+peak3_flux)/frame.meta["EXPTIME"])
                            rmsamp1_flux=peak1_flux/np.sqrt(frame.meta["EXPTIME"])
                            rmsamp3_flux=np.sum((peak2_flux+peak3_flux)/np.sqrt(frame.meta["EXPTIME"]))
                        if camera[0]=="r":
                            amp1_flux=np.sum((peak1_flux+peak2_flux)/frame.meta["EXPTIME"])
                            amp3_flux=np.sum((peak3_flux+peak4_flux+peak5_flux)/frame.meta["EXPTIME"])
                            rmsamp1_flux=np.sum((peak1_flux+peak2_flux)/np.sqrt(frame.meta["EXPTIME"]))
                            rmsamp3_flux=np.sum((peak3_flux+peak4_flux+peak5_flux)/np.sqrt(frame.meta["EXPTIME"]))
                        if camera[0]=="z":
                            amp1_flux=np.sum((peak1_flux+peak2_flux+peak3_flux)/frame.meta["EXPTIME"])
                            amp3_flux=np.sum((peak4_flux+peak5_flux+peak6_flux)/frame.meta["EXPTIME"])
                            rmsamp1_flux=np.sum((peak1_flux+peak2_flux+peak3_flux)/np.sqrt(frame.meta["EXPTIME"]))
                            rmsamp3_flux=np.sum((peak4_flux+peak5_flux+peak6_flux)/np.sqrt(frame.meta["EXPTIME"]))
                        amp1.append(amp1_flux)
                        amp3.append(amp3_flux)
                        rmsamp1.append(rmsamp1_flux)
                        rmsamp3.append(rmsamp3_flux)
                    if frame.fibermap['FIBER'][i]>260:
                        if camera[0]=="b":
                            amp2_flux=peak1_flux/frame.meta["EXPTIME"]
                            amp4_flux=np.sum((peak2_flux+peak3_flux)/frame.meta["EXPTIME"])
                            rmsamp2_flux=peak1_flux/np.sqrt(frame.meta["EXPTIME"])
                            rmsamp4_flux=np.sum((peak2_flux+peak3_flux)/np.sqrt(frame.meta["EXPTIME"]))
                        if camera[0]=="r":
                            amp2_flux=np.sum((peak1_flux+peak2_flux)/frame.meta["EXPTIME"])
                            amp4_flux=np.sum((peak3_flux+peak4_flux+peak5_flux)/frame.meta["EXPTIME"])
                            rmsamp2_flux=np.sum((peak1_flux+peak2_flux)/np.sqrt(frame.meta["EXPTIME"]))
                            rmsamp4_flux=np.sum((peak3_flux+peak4_flux+peak5_flux)/np.sqrt(frame.meta["EXPTIME"]))
                        if camera[0]=="z":
                            amp2_flux=np.sum((peak1_flux+peak2_flux+peak3_flux)/frame.meta["EXPTIME"])
                            amp4_flux=np.sum((peak4_flux+peak5_flux+peak6_flux)/frame.meta["EXPTIME"])
                            rmsamp2_flux=np.sum((peak1_flux+peak2_flux+peak3_flux)/np.sqrt(frame.meta["EXPTIME"]))
                            rmsamp4_flux=np.sum((peak4_flux+peak5_flux+peak6_flux)/np.sqrt(frame.meta["EXPTIME"]))
                        amp2.append(amp2_flux)
                        amp4.append(amp4_flux)
                        rmsamp2.append(rmsamp2_flux)
                        rmsamp4.append(rmsamp4_flux)

        nspec_counts=np.array(nspec_counts)
        sky_counts=np.array(sky_counts)
        rms_nspec=qalib.getrms(nspec_counts)
        rms_skyspec=qalib.getrms(sky_counts)

        sumcount_med_sky=[]

        if param is None:
            log.debug("Param is None. Using default param instead")
            param ={
                "B_PEAKS":[3914.4, 5199.3, 5201.8],
                "R_PEAKS":[6301.9, 6365.4, 7318.2, 7342.8, 7371.3],
                "Z_PEAKS":[8401.5, 8432.4, 8467.5, 9479.4, 9505.6, 9521.8],
                "PEAKCOUNT_NORMAL_RANGE":[1000.0, 20000.0],
                "PEAKCOUNT_WARN_RANGE":[500.0, 40000.0]
                }

        retval["PARAMS"] = param

        sumcount_err=[]
        if amps:
            if frame.fibermap['FIBER'].shape[0]<260:
                amp2=np.zeros(len(sky_counts))
                amp4=np.zeros(len(sky_counts))
            else:
                amp2=np.array(rmsamp2)
                amp4=np.array(rmsamp4)
            amp1=np.array(rmsamp1)
            amp3=np.array(rmsamp3)
            amp1_rms=qalib.getrms(amp1)
            amp2_rms=qalib.getrms(amp2)
            amp3_rms=qalib.getrms(amp3)
            amp4_rms=qalib.getrms(amp4)
            rms_skyspec_amp=np.array([amp1_rms,amp2_rms,amp3_rms,amp4_rms])

            retval["METRICS"]={"RA":ra,"DEC":dec, "PEAKCOUNT":nspec_counts,"PEAKCOUNT_RMS":rms_nspec,"PEAKCOUNT":sumcount_med_sky,"PEAKCOUNT_RMS_SKY":rms_skyspec,"PEAKCOUNT_RMS_AMP":rms_skyspec_amp,"PEAKCOUNT_STAT":sumcount_err}
        else:
            retval["METRICS"]={"RA":ra,"DEC":dec, "PEAKCOUNT":nspec_counts,"PEAKCOUNT_RMS":rms_nspec,"SUMCOUNT_MED_SKY":sumcount_med_sky,"PEAKCOUNT_RMS_SKY":rms_skyspec,"PEAKCOUNT_STAT":sumcount_err}

        if qlf:
            qlf_post(retval)

        if qafile is not None:
            outfile = qa.write_qa_ql(qafile,retval)
            log.debug("Output QA data is in {}".format(outfile))
        if qafig is not None:
            from desispec.qa.qa_plots_ql import plot_sky_peaks
            plot_sky_peaks(retval,qafig)

            log.debug("Output QA fig {}".format(qafig))

        return retval

    def get_default_config(self):
        return {}


class Calc_XWSigma(MonitoringAlg):
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="XWSIGMA"
        from desispec.image import Image as im
        kwargs=config['kwargs']
        parms=kwargs['param']
        key="WSIGMA_MED_SKY"
        kwargs["SAMI_RESULTKEY"]=key
        kwargs["SAMI_QASTATUSKEY"]="XWSIGMA_STAT"
        if "ReferenceMetrics" in kwargs:
            r=kwargs["ReferenceMetrics"]
            if key in r:
                kwargs["REFERENCE"]=r[key]

        if "XWSIGMA_WARN_RANGE" in parms and "XWSIGMA_NORMAL_RANGE" in parms:
            kwargs["RANGES"]=[(np.asarray(parms["XWSIGMA_WARN_RANGE"]),QASeverity.WARNING),
                              (np.asarray(parms["XWSIGMA_NORMAL_RANGE"]),QASeverity.NORMAL)]# sorted by most severe to least severe
        MonitoringAlg.__init__(self,name,im,config,logger)
    def run(self,*args,**kwargs):
        if len(args) == 0 :
            raise qlexceptions.ParameterException("Missing input parameter")
        if not self.is_compatible(type(args[0])):
            raise qlexceptions.ParameterException("Incompatible parameter type. Was expecting desispec.image.Image got {}".format(type(args[0])))

        fibermap=kwargs['FiberMap'] 
        input_image=args[0]
 
        if "paname" not in kwargs:
            paname=None
        else:
            paname=kwargs["paname"]
 
        if "ReferenceMetrics" in kwargs: refmetrics=kwargs["ReferenceMetrics"]
        else: refmetrics=None

        amps=False
        if "amps" in kwargs:
            amps=kwargs["amps"]

        if "param" in kwargs: param=kwargs["param"]
        else: param=None
 
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
 
        return self.run_qa(fibermap,input_image,paname=paname,amps=amps,psf=psf, qafile=qafile,qafig=qafig, param=param, qlf=qlf, refmetrics=refmetrics)
 
    def run_qa(self,fibermap,image,paname=None,amps=False,psf=None, qafile=None,qafig=None, param=None, qlf=False, refmetrics=None):
        from scipy.optimize import curve_fit

        retval={}
        retval["PANAME"] = paname
        retval["QATIME"] = datetime.datetime.now().isoformat() 
        retval["EXPID"] = '{0:08d}'.format(image.meta["EXPID"])
        retval["CAMERA"] = camera = image.meta["CAMERA"]
        retval["PROGRAM"] = image.meta["PROGRAM"]
        retval["FLAVOR"] = image.meta["FLAVOR"]
        retval["NIGHT"] = image.meta["NIGHT"]
        kwargs=self.config['kwargs']
        if "REFERENCE" in kwargs:
            retval['XWSIGMA_REF']=kwargs["REFERENCE"]

        ra = fibermap["RA_TARGET"]
        dec = fibermap["DEC_TARGET"]

        if param is None:
            log.debug("Param is None. Using default param instead")
            if image.meta["FLAVOR"] == 'arc':
                param = {
                    "B_PEAKS":[4047.7, 4359.6, 5087.2],
                    "R_PEAKS":[6144.8, 6508.3, 6600.8, 6718.9, 6931.4, 7034.4,],
                    "Z_PEAKS":[8379.9, 8497.7, 8656.8, 8783.0],
                    "XWSIGMA_NORMAL_RANGE":[-2.0, 2.0],
                    "XWSIGMA_WARN_RANGE":[-4.0, 4.0]
                    }
            else:
                param = {
                    "B_PEAKS":[3914.4, 5199.3, 5578.9],
                    "R_PEAKS":[6301.9, 6365.4, 7318.2, 7342.8, 7371.3],
                    "Z_PEAKS":[8401.5, 8432.4, 8467.5, 9479.4, 9505.6, 9521.8],
                    "XWSIGMA_NORMAL_RANGE":[-2.0, 2.0],
                    "XWSIGMA_WARN_RANGE":[-4.0, 4.0]
                    }

        dw=2.
        dp=3
        b_peaks=param['B_PEAKS']
        r_peaks=param['R_PEAKS']
        z_peaks=param['Z_PEAKS']

        if fibermap["OBJTYPE"][0] == 'ARC':
            import desispec.psf
            psf=desispec.psf.PSF(psf)

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
        if fibermap['FIBER'].shape[0] >= 500:
            fibers = 500
        else:
            fibers = fibermap['FIBER'].shape[0]
        for i in range(fibers):
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
 
                xpopt1,xpcov1=curve_fit(qalib.gauss,np.arange(len(xpix_peak1)),image.pix[int(np.mean(ypix_peak1)),xpix_peak1])
                wpopt1,wpcov1=curve_fit(qalib.gauss,np.arange(len(ypix_peak1)),image.pix[ypix_peak1,int(np.mean(xpix_peak1))])
                xpopt2,xpcov2=curve_fit(qalib.gauss,np.arange(len(xpix_peak2)),image.pix[int(np.mean(ypix_peak2)),xpix_peak2])
                wpopt2,wpcov2=curve_fit(qalib.gauss,np.arange(len(ypix_peak2)),image.pix[ypix_peak2,int(np.mean(xpix_peak2))])
                xpopt3,xpcov3=curve_fit(qalib.gauss,np.arange(len(xpix_peak3)),image.pix[int(np.mean(ypix_peak3)),xpix_peak3])
                wpopt3,wpcov3=curve_fit(qalib.gauss,np.arange(len(ypix_peak3)),image.pix[ypix_peak3,int(np.mean(xpix_peak3))])

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
                xsigma.append(xsigma_avg)
                wsigma.append(wsigma_avg)
 
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

                xpopt1,xpcov1=curve_fit(qalib.gauss,np.arange(len(xpix_peak1)),image.pix[int(np.mean(ypix_peak1)),xpix_peak1])
                wpopt1,wpcov1=curve_fit(qalib.gauss,np.arange(len(ypix_peak1)),image.pix[ypix_peak1,int(np.mean(xpix_peak1))])
                xpopt2,xpcov2=curve_fit(qalib.gauss,np.arange(len(xpix_peak2)),image.pix[int(np.mean(ypix_peak2)),xpix_peak2])
                wpopt2,wpcov2=curve_fit(qalib.gauss,np.arange(len(ypix_peak2)),image.pix[ypix_peak2,int(np.mean(xpix_peak2))])
                xpopt3,xpcov3=curve_fit(qalib.gauss,np.arange(len(xpix_peak3)),image.pix[int(np.mean(ypix_peak3)),xpix_peak3])
                wpopt3,wpcov3=curve_fit(qalib.gauss,np.arange(len(ypix_peak3)),image.pix[ypix_peak3,int(np.mean(xpix_peak3))])
                xpopt4,xpcov4=curve_fit(qalib.gauss,np.arange(len(xpix_peak4)),image.pix[int(np.mean(ypix_peak4)),xpix_peak4])
                wpopt4,wpcov4=curve_fit(qalib.gauss,np.arange(len(ypix_peak4)),image.pix[ypix_peak4,int(np.mean(xpix_peak4))])
                xpopt5,xpcov5=curve_fit(qalib.gauss,np.arange(len(xpix_peak5)),image.pix[int(np.mean(ypix_peak5)),xpix_peak5])
                wpopt5,wpcov5=curve_fit(qalib.gauss,np.arange(len(ypix_peak5)),image.pix[ypix_peak5,int(np.mean(xpix_peak5))])

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
                peak_wave=np.array([z_peaks[0]-dw,z_peaks[0]+dw,z_peaks[1]-dw,z_peaks[1]+dw,z_peaks[2]-dw,z_peaks[2]+dw,z_peaks[3]-dw,z_peaks[3]+dw])
 
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
 
                xpopt1,xpcov1=curve_fit(qalib.gauss,np.arange(len(xpix_peak1)),image.pix[int(np.mean(ypix_peak1)),xpix_peak1])
                wpopt1,wpcov1=curve_fit(qalib.gauss,np.arange(len(ypix_peak1)),image.pix[ypix_peak1,int(np.mean(xpix_peak1))])
                xpopt2,xpcov2=curve_fit(qalib.gauss,np.arange(len(xpix_peak2)),image.pix[int(np.mean(ypix_peak2)),xpix_peak2])
                wpopt2,wpcov2=curve_fit(qalib.gauss,np.arange(len(ypix_peak2)),image.pix[ypix_peak2,int(np.mean(xpix_peak2))])
                xpopt3,xpcov3=curve_fit(qalib.gauss,np.arange(len(xpix_peak3)),image.pix[int(np.mean(ypix_peak3)),xpix_peak3])
                wpopt3,wpcov3=curve_fit(qalib.gauss,np.arange(len(ypix_peak3)),image.pix[ypix_peak3,int(np.mean(xpix_peak3))])
                xpopt4,xpcov4=curve_fit(qalib.gauss,np.arange(len(xpix_peak4)),image.pix[int(np.mean(ypix_peak4)),xpix_peak4])
                wpopt4,wpcov4=curve_fit(qalib.gauss,np.arange(len(ypix_peak4)),image.pix[ypix_peak4,int(np.mean(xpix_peak4))])

                xsigma1=np.abs(xpopt1[2])
                wsigma1=np.abs(wpopt1[2])
                xsigma2=np.abs(xpopt2[2])
                wsigma2=np.abs(wpopt2[2])
                xsigma3=np.abs(xpopt3[2])
                wsigma3=np.abs(wpopt3[2])
                xsigma4=np.abs(xpopt4[2])
                wsigma4=np.abs(wpopt4[2])

                xsig=np.array([xsigma1,xsigma2,xsigma3,xsigma4])
                wsig=np.array([wsigma1,wsigma2,wsigma3,wsigma4])
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
                        xsig_amp3=np.array([xsigma4])
                        wsig_amp1=np.array([wsigma1,wsigma2,wsigma3])
                        wsig_amp3=np.array([wsigma4])

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
                        xsig_amp4=np.array([xsigma4])
                        wsig_amp2=np.array([wsigma1,wsigma2,wsigma3])
                        wsig_amp4=np.array([wsigma4])

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
        xwsigma=np.array([xsigma_med_sky,wsigma_med_sky])
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

        xshift=0.0
        wshift=0.0
        xshift_fib=[]
        wshift_fib=[]
        xshift_amp=[]
        wshift_amp=[]
        shift_warn=[]

        retval["PARAMS"] = param

        shift_err=[]
        if amps:
            retval["METRICS"]={"RA":ra,"DEC":dec, "XSIGMA":xsigma,"XSIGMA_MED":xsigma_med,"XSIGMA_AMP":xsigma_amp,"XSHIFT":xshift,"XSHIFT_FIB":xshift_fib,"XSHIFT_AMP":xshift_amp,"WSIGMA":wsigma,"WSIGMA_MED":wsigma_med,"WSIGMA_AMP":wsigma_amp,"WSHIFT":wshift,"WSHIFT_FIB":wshift_fib,"WSHIFT_AMP":wshift_amp,"XWSIGMA":xwsigma,"XWSIGMA_STAT":shift_err}

        else:
            retval["METRICS"]={"RA":ra,"DEC":dec, "XSIGMA":xsigma,"XSIGMA_MED":xsigma_med,"XSIGMA_MED_SKY":xsigma_med_sky,"XSHIFT":xshift,"XSHIFT_FIB":xshift_fib,"WSIGMA":wsigma,"WSIGMA_MED":wsigma_med,"WSIGMA_MED_SKY":wsigma_med_sky,"WSHIFT":wshift,"WSHIFT_FIB":wshift_fib,"XWSIGMA_STAT":shift_err}

        #- http post if needed
        if qlf:
            qlf_post(retval)    

        if qafile is not None:
            outfile = qa.write_qa_ql(qafile,retval)
            log.debug("Output QA data is in {}".format(outfile))
        if qafig is not None:
            from desispec.qa.qa_plots_ql import plot_XWSigma
            plot_XWSigma(retval,qafig)

            log.debug("Output QA fig {}".format(qafig))

        return retval
 
    def get_default_config(self):
        return {}


class Bias_From_Overscan(MonitoringAlg):
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="BIAS_OVERSCAN"
        import astropy
        rawtype=astropy.io.fits.hdu.hdulist.HDUList
        kwargs=config['kwargs']
        parms=kwargs['param']
        key="BIAS_AMP"
        kwargs["SAMI_RESULTKEY"]=key
        kwargs["SAMI_QASTATUSKEY"]="BIAS_STAT"

        if "ReferenceMetrics" in kwargs:
            r=kwargs["ReferenceMetrics"]
            if key in r:
                kwargs["REFERENCE"]=r[key]

        if "BIAS_WARN_RANGE" in parms and "BIAS_NORMAL_RANGE" in parms:
            kwargs["RANGES"]=[(np.asarray(parms["BIAS_WARN_RANGE"]),QASeverity.WARNING),
                              (np.asarray(parms["BIAS_NORMAL_RANGE"]),QASeverity.NORMAL)]# sorted by most severe to least severe 
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

        if "ReferenceMetrics" in kwargs: refmetrics=kwargs["ReferenceMetrics"]
        else: refmetrics=None

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
        else: qafig=None

        return self.run_qa(input_raw,camera,paname=paname,amps=amps, qafile=qafile,qafig=qafig, param=param, qlf=qlf, refmetrics=refmetrics)

    def run_qa(self,raw,camera,paname=None,amps=False,qafile=None,qafig=None, param=None, qlf=False, refmetrics=None):

        rawimage=raw[camera.upper()].data
        header=raw[camera.upper()].header

        retval={}
        retval["EXPID"]= '{0:08d}'.format(header["EXPID"])
        retval["CAMERA"] = camera
        retval["PANAME"] = paname
        retval["QATIME"] = datetime.datetime.now().isoformat()
        retval["FLAVOR"] = header["FLAVOR"]
        if retval["FLAVOR"] == 'arc':
            pass
        else:
            retval["PROGRAM"] = header["PROGRAM"]
        retval["NIGHT"] = header["NIGHT"]
        kwargs=self.config['kwargs']
        if "REFERENCE" in kwargs:
            retval['BIAS_AMP_REF']=kwargs["REFERENCE"]

        rawimage=raw[camera.upper()].data
        header=raw[camera.upper()].header

        if 'INHERIT' in header and header['INHERIT']:
            h0 = raw[0].header
            for key in h0:
                if key not in header:
                    header[key] = h0[key]

        data=[]
        row_data_amp1=[]
        row_data_amp2=[]
        row_data_amp3=[]
        row_data_amp4=[]
        bias_overscan=[]        
        for kk in ['1','2','3','4']:
            from desispec.preproc import _parse_sec_keyword
            
            sel=_parse_sec_keyword(header['BIASSEC'+kk])
            #- Obtain counts/second in bias region
            pixdata=rawimage[sel]/header["EXPTIME"]
            if kk == '1':
                for i in range(pixdata.shape[0]):
                    row_amp1=pixdata[i]
                    row_data_amp1.append(row_amp1)
            if kk == '2':
                for i in range(pixdata.shape[0]):
                    row_amp2=pixdata[i]
                    row_data_amp2.append(row_amp2)
            if kk == '3':
                for i in range(pixdata.shape[0]):
                    row_amp3=pixdata[i]
                    row_data_amp3.append(row_amp3)
            if kk == '4':
                for i in range(pixdata.shape[0]):
                    row_amp4=pixdata[i]
                    row_data_amp4.append(row_amp4)
            #- Compute statistics of the bias region that only reject
            #  the 0.5% of smallest and largest values. (from sdssproc) 
            isort=np.sort(pixdata.ravel())
            nn=isort.shape[0]
            bias=np.mean(isort[int(0.005*nn) : int(0.995*nn)])
            bias_overscan.append(bias)
            data.append(isort)

        row_data_bottom=[]
        row_data_top=[]
        for i in range(len(row_data_amp1)):
            row_data_lower=np.concatenate((row_data_amp1[i],row_data_amp2[i]))
            row_data_upper=np.concatenate((row_data_amp3[i],row_data_amp4[i]))
            row_data_bottom.append(row_data_lower)
            row_data_top.append(row_data_upper)
        row_data=np.concatenate((row_data_bottom,row_data_top))

        mean_row=[]
        for i in range(len(row_data)):
            mean=np.mean(row_data[i])
            mean_row.append(mean)

        full_data=np.concatenate((data[0],data[1],data[2],data[3])).ravel()
        bias=np.mean(bias_overscan)

        if param is None:
            log.debug("Param is None. Using default param instead")
            param = {
                "PERCENTILES":[68.2,95.4,99.7],
                "BIAS_NORMAL_RANGE":[-1.0, 1.0],
                "BIAS_WARN_RANGE:":[-2.0, 2.0]
                }

        sig1_lo = np.percentile(full_data,(100.-param['PERCENTILES'][0])/2.)
        sig1_hi = np.percentile(full_data,100.-sig1_lo)
        sig2_lo = np.percentile(full_data,(100.-param['PERCENTILES'][1])/2.)
        sig2_hi = np.percentile(full_data,100.-sig2_lo)
        sig3_lo = np.percentile(full_data,(100.-param['PERCENTILES'][2])/2.)
        sig3_hi = np.percentile(full_data,100.-sig3_lo)

        diff1sig = sig1_hi - sig1_lo
        diff2sig = sig2_hi - sig2_lo
        diff3sig = sig3_hi - sig3_lo

        sig5_value = np.percentile(full_data,100.-99.99994)
        data5sig = len(np.where(full_data <= sig5_value)[0])

        retval["PARAMS"] = param

        biasdiff_err=[]
        if amps:
            bias_amps=np.array(bias_overscan)

            retval["METRICS"]={'BIAS':bias,'BIAS_AMP':bias_amps,"DIFF1SIG":diff1sig,"DIFF2SIG":diff2sig,"DIFF3SIG":diff3sig,"DATA5SIG":data5sig,"MEANBIAS_ROW":mean_row,"BIAS_STAT":biasdiff_err}

        else:
            retval["METRICS"]={'BIAS':bias,"DIFF1SIG":diff1sig,"DIFF2SIG":diff2sig,"DIFF3SIG":diff3sig,"DATA5SIG":data5sig,"MEANBIAS_ROW":mean_row,"BIAS_STAT":biasdiff_err}

        #- http post if needed
        if qlf:
            qlf_post(retval)    

        if qafile is not None:
            outfile = qa.write_qa_ql(qafile,retval)
            log.debug("Output QA data is in {}".format(outfile))
        if qafig is not None:
            from desispec.qa.qa_plots_ql import plot_bias_overscan
            plot_bias_overscan(retval,qafig)
            
            log.debug("Output QA fig {}".format(qafig))                   
        
        return retval

    def get_default_config(self):
        return {}

class CountSpectralBins(MonitoringAlg):

    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="COUNTBINS"
        from  desispec.frame import Frame as fr
        kwargs=config['kwargs']
        parms=kwargs['param']
        key="NGOODFIB"
        kwargs["SAMI_RESULTKEY"]=key
        kwargs["SAMI_QASTATUSKEY"]="NGOODFIB_STAT"

        if "ReferenceMetrics" in kwargs:
            r=kwargs["ReferenceMetrics"]
            if key in r:
                kwargs["REFERENCE"]=r[key]

        if "NGOODFIB_WARN_RANGE" in parms and "NGOODFIB_NORMAL_RANGE" in parms:
            kwargs["RANGES"]=[(np.asarray(parms["NGOODFIB_WARN_RANGE"]),QASeverity.WARNING),
                              (np.asarray(parms["NGOODFIB_NORMAL_RANGE"]),QASeverity.NORMAL)]# sorted by most severe to least severe 
        MonitoringAlg.__init__(self,name,fr,config,logger)
    def run(self,*args,**kwargs):
        if len(args) == 0 :
            raise qlexceptions.ParameterException("Missing input parameter")
        if not self.is_compatible(type(args[0])):
            raise qlexceptions.ParameterException("Incompatible input. Was expecting {} got {}".format(type(self.__inpType__),type(args[0])))

        fibermap=kwargs['FiberMap']
        input_frame=args[0]

        paname=None
        if "paname" in kwargs:
            paname=kwargs["paname"]

        if "ReferenceMetrics" in kwargs: refmetrics=kwargs["ReferenceMetrics"]
        else: refmetrics=None

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

        return self.run_qa(fibermap,input_frame,paname=paname,amps=amps,psf=psf, qafile=qafile,qafig=qafig, param=param, qlf=qlf, refmetrics=refmetrics)


    def run_qa(self,fibermap,frame,paname=None,psf=None,amps=False,qafile=None,qafig=None,param=None, qlf=False, refmetrics=None):

        #- qa dictionary 
        retval={}
        retval["PANAME"] = paname
        retval["QATIME"] = datetime.datetime.now().isoformat()
        retval["EXPID"] = '{0:08d}'.format(frame.meta["EXPID"])
        retval["CAMERA"] = frame.meta["CAMERA"]
        retval["PROGRAM"] = frame.meta["PROGRAM"]
        retval["FLAVOR"] = frame.meta["FLAVOR"]
        retval["NIGHT"] = frame.meta["NIGHT"]
        kwargs=self.config['kwargs']
        if "REFERENCE" in kwargs:
            retval['NGOODFIB_REF']=kwargs["REFERENCE"]

        ra = fibermap["RA_TARGET"]
        dec = fibermap["DEC_TARGET"]

        if fibermap["OBJTYPE"][0] == 'ARC':
            import desispec.psf
            psf=desispec.psf.PSF(psf)

        grid=np.gradient(frame.wave)
        if not np.all(grid[0]==grid[1:]): 
            log.debug("grid_size is NOT UNIFORM")

        if param is None:
            log.debug("Param is None. Using default param instead")
            param = {
                 "CUTLO":100,   # low threshold for number of counts
                 "CUTMED":250,
                 "CUTHI":500,
                 "NGOODFIB_NORMAL_RANGE":[490, 500],
                 "NGOODFIB_WARN_RANGE":[480, 500]
                 }

        retval["PARAMS"] = param

        nbinshi_temp=[]
        
        countslo=qalib.countbins(frame.flux,threshold=param['CUTLO'])
        countsmed=qalib.countbins(frame.flux,threshold=param['CUTMED'])
        countshi=qalib.countbins(frame.flux,threshold=param['CUTHI'])

        goodfibers=np.where(countshi>0)[0] #- fibers with at least one bin higher than cuthi counts
        ngoodfibers=goodfibers.shape[0]

        leftmax=None
        rightmax=None
        bottommax=None
        topmin=None

        ngood_err="UNKNOWN"

        if amps:
            #- get the pixel boundary and fiducial boundary in flux-wavelength space

            leftmax,rightmin,bottommax,topmin = qalib.fiducialregion(frame,psf)  
            fidboundary=qalib.slice_fidboundary(frame,leftmax,rightmin,bottommax,topmin)          
            countslo_amp1=qalib.countbins(frame.flux[fidboundary[0]],threshold=param['CUTLO'])
            averagelo_amp1=np.mean(countslo_amp1)
            countsmed_amp1=qalib.countbins(frame.flux[fidboundary[0]],threshold=param['CUTMED'])
            averagemed_amp1=np.mean(countsmed_amp1)
            countshi_amp1=qalib.countbins(frame.flux[fidboundary[0]],threshold=param['CUTHI'])
            averagehi_amp1=np.mean(countshi_amp1)

            countslo_amp3=qalib.countbins(frame.flux[fidboundary[2]],threshold=param['CUTLO'])
            averagelo_amp3=np.mean(countslo_amp3)
            countsmed_amp3=qalib.countbins(frame.flux[fidboundary[2]],threshold=param['CUTMED'])
            averagemed_amp3=np.mean(countsmed_amp3)
            countshi_amp3=qalib.countbins(frame.flux[fidboundary[2]],threshold=param['CUTHI'])
            averagehi_amp3=np.mean(countshi_amp3)


            if fidboundary[1][0].start is not None: #- to the right bottom of the CCD

                countslo_amp2=qalib.countbins(frame.flux[fidboundary[1]],threshold=param['CUTLO'])
                averagelo_amp2=np.mean(countslo_amp2)
                countsmed_amp2=qalib.countbins(frame.flux[fidboundary[1]],threshold=param['CUTMED'])
                averagemed_amp2=np.mean(countsmed_amp2)
                countshi_amp2=qalib.countbins(frame.flux[fidboundary[1]],threshold=param['CUTHI'])
                averagehi_amp2=np.mean(countshi_amp2)

            else:
                averagelo_amp2=0.
                averagemed_amp2=0.
                averagehi_amp2=0.

            if fidboundary[3][0].start is not None: #- to the right top of the CCD

                countslo_amp4=qalib.countbins(frame.flux[fidboundary[3]],threshold=param['CUTLO'])
                averagelo_amp4=np.mean(countslo_amp4)
                countsmed_amp4=qalib.countbins(frame.flux[fidboundary[3]],threshold=param['CUTMED'])
                averagemed_amp4=np.mean(countsmed_amp4)
                countshi_amp4=qalib.countbins(frame.flux[fidboundary[3]],threshold=param['CUTHI'])
                averagehi_amp4=np.mean(countshi_amp4)

            else:
                averagelo_amp4=0.
                averagemed_amp4=0.
                averagehi_amp4=0.

            averagelo_amps=np.array([averagelo_amp1,averagelo_amp2,averagelo_amp3,averagelo_amp4])
            averagemed_amps=np.array([averagemed_amp1,averagemed_amp2,averagemed_amp3,averagemed_amp4])
            averagehi_amps=np.array([averagehi_amp1,averagehi_amp2,averagehi_amp3,averagehi_amp4])

            retval["METRICS"]={"RA":ra,"DEC":dec, "NBINSLOW":countslo,"NBINSMED":countsmed,"NBINSHIGH":countshi, "NBINSLOW_AMP":averagelo_amps,"NBINSMED_AMP":averagemed_amps,"NBINSHIGH_AMP":averagehi_amps, "NGOODFIB": ngoodfibers, "NBINSHI_TEMP":nbinshi_temp,"NGOODFIB_STAT":ngood_err}
        else:
            retval["METRICS"]={"RA":ra,"DEC":dec, "NBINSLOW":countslo,"NBINSMED":countsmed,"NBINSHIGH":countshi,"NGOODFIB": ngoodfibers, "NBINSHI_TEMP":nbinshi_temp,"NGOODFIB_STAT":ngood_err}

        retval["LEFT_MAX_FIBER"]=int(leftmax)
        retval["RIGHT_MIN_FIBER"]=int(rightmin)
        retval["BOTTOM_MAX_WAVE_INDEX"]=int(bottommax)
        retval["TOP_MIN_WAVE_INDEX"]=int(topmin)

        #- http post if needed
        if qlf:
            qlf_post(retval)    

        if qafile is not None:
            outfile = qa.write_qa_ql(qafile,retval)
            log.debug("Output QA data is in {}".format(outfile))
        if qafig is not None:
            from desispec.qa.qa_plots_ql import plot_countspectralbins
            plot_countspectralbins(retval,qafig)
            
            log.debug("Output QA fig {}".format(qafig))                   
        
        return retval

class Sky_Residual(MonitoringAlg):
    """ 
    Use offline sky_residual function to calculate sky residuals
    """
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="RESIDUAL"
        from  desispec.frame import Frame as fr
        kwargs=config['kwargs']
        parms=kwargs['param']
        key="RESIDRMS"
        kwargs["SAMI_RESULTKEY"]=key
        kwargs["SAMI_QASTATUSKEY"]="RESIDRMS_STAT"

        if "ReferenceMetrics" in kwargs:
            r=kwargs["ReferenceMetrics"]
            if key in r:
                kwargs["REFERENCE"]=r[key]

        if "RESID_WARN_RANGE" in parms and "RESID_NORMAL_RANGE" in parms:
            kwargs["RANGES"]=[(np.asarray(parms["RESID_WARN_RANGE"]),QASeverity.WARNING),
                              (np.asarray(parms["RESID_NORMAL_RANGE"]),QASeverity.NORMAL)]# sorted by most severe to least severe 
        MonitoringAlg.__init__(self,name,fr,config,logger)
    def run(self,*args,**kwargs):
        from desispec.io.sky import read_sky
        if len(args) == 0 :
            raise qlexceptions.ParameterException("Missing input parameter")
        if not self.is_compatible(type(args[0])):
            raise qlexceptions.ParameterException("Incompatible input. Was expecting {} got {}".format(type(self.__inpType__),type(args[0])))

        fibermap=kwargs['FiberMap']
        input_frame=args[0] #- should be sky subtracted
        skymodel=args[1] #- should be skymodel evaluated
        if "SkyFile" in kwargs:
            from desispec.io.sky import read_sky
            skyfile=kwargs["SkyFile"]    #- Read sky model file itself from an argument
            log.debug("Using given sky file {} for subtraction".format(skyfile))

            skymodel=read_sky(skyfile)

        if "ReferenceMetrics" in kwargs: refmetrics=kwargs["ReferenceMetrics"]
        else: refmetrics=None

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
        
        return self.run_qa(fibermap,input_frame,paname=paname,skymodel=skymodel,amps=amps, dict_countbins=dict_countbins, qafile=qafile,qafig=qafig, param=param, qlf=qlf, refmetrics=refmetrics)


    def run_qa(self,fibermap,frame,paname=None,skymodel=None,amps=False,dict_countbins=None, qafile=None,qafig=None, param=None, qlf=False, refmetrics=None):
        from desispec.sky import qa_skysub

        if skymodel is None:
            raise IOError("Must have skymodel to find residual. It can't be None")
        #- return values
        retval={}
        retval["PANAME"] = paname
        retval["QATIME"] = datetime.datetime.now().isoformat()
        retval["EXPID"] = '{0:08d}'.format(frame.meta["EXPID"])
        retval["CAMERA"] = frame.meta["CAMERA"]
        retval["PROGRAM"] = frame.meta["PROGRAM"]
        retval["FLAVOR"] = frame.meta["FLAVOR"]
        retval["NIGHT"] = frame.meta["NIGHT"]
        kwargs=self.config['kwargs']
        if "REFERENCE" in kwargs:
            retval['RESIDRMS_REF']=kwargs["REFERENCE"]

        ra = fibermap["RA_TARGET"]
        dec = fibermap["DEC_TARGET"]

        if param is None:
            log.debug("Param is None. Using default param instead")
            param = {
                "BIN_SZ":0.1, #- Bin size for histograms
                "PCHI_RESID":0.05, # P(Chi^2) limit for bad skyfiber model residuals
                "PER_RESID":95.,   # Percentile for residual distribution
                "RESID_NORMAL_RANGE":[-5.0, 5.0],
                "RESID_WARN_RANGE":[-10.0, 10.0]
                }

        qadict=qalib.sky_resid(param,frame,skymodel,quick_look=True)

        retval["METRICS"] = {}
        for key in qadict.keys():
            retval["METRICS"][key] = qadict[key]

        if qlf:
            qlf_post(retval)    

        skyresid_err=[]
        retval["PARAMS"] = param
        retval["METRICS"]["RESIDRMS_STAT"]=skyresid_err

        if qafile is not None:
            outfile = qa.write_qa_ql(qafile,retval)
            log.debug("Output QA data is in {}".format(outfile))
        if qafig is not None:
            from desispec.qa.qa_plots_ql import plot_residuals
            plot_residuals(retval,qafig)
            
            log.debug("Output QA fig {}".format(qafig))            

        return retval
        
class Calculate_SNR(MonitoringAlg):
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="SNR"
        from  desispec.frame import Frame as fr
        kwargs=config['kwargs']
        parms=kwargs['param']
        key="ELG_FIDSNR"
        kwargs["SAMI_RESULTKEY"]=key
        kwargs["SAMI_QASTATUSKEY"]="FIDSNR_STAT"
        if "ReferenceMetrics" in kwargs:
            r=kwargs["ReferenceMetrics"]
            if key in r:
                kwargs["REFERENCE"]=r[key]

        if "FIDSNR_WARN_RANGE" in parms and "FIDSNR_NORMAL_RANGE" in parms:
            kwargs["RANGES"]=[(np.asarray(parms["FIDSNR_WARN_RANGE"]),QASeverity.WARNING),
                              (np.asarray(parms["FIDSNR_NORMAL_RANGE"]),QASeverity.NORMAL)]# sorted by most severe to least severe 
        MonitoringAlg.__init__(self,name,fr,config,logger)
    def run(self,*args,**kwargs):
        from desispec.io.sky import read_sky
        if len(args) == 0 :
            raise qlexceptions.ParameterException("Missing input parameter")
        if not self.is_compatible(type(args[0])):
            raise qlexceptions.ParameterException("Incompatible input. Was expecting {} got {}".format(type(self.__inpType__),type(args[0])))

        fibermap=kwargs['FiberMap']
        input_frame=args[0]

        if "ReferenceMetrics" in kwargs: refmetrics=kwargs["ReferenceMetrics"]
        else: refmetrics=None

        amps=False
        if "amps" in kwargs:
            amps=kwargs["amps"]

        dict_countbins=None
        if "dict_countbins" in kwargs:
            dict_countbins=kwargs["dict_countbins"]

        if "param" in kwargs: param=kwargs["param"]
        else: param=None
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

        return self.run_qa(fibermap,input_frame,paname=paname,amps=amps,dict_countbins=dict_countbins, qafile=qafile,qafig=qafig, param=param, qlf=qlf, refmetrics=refmetrics)


    def run_qa(self,fibermap,frame,paname=None,amps=False,dict_countbins=None, qafile=None,qafig=None, qlf=False, param=None, refmetrics=None):

        #- return values
        retval={}
        retval["PANAME"] = paname
        retval["QATIME"] = datetime.datetime.now().isoformat()
        retval["EXPID"] = '{0:08d}'.format(frame.meta["EXPID"])
        retval["CAMERA"] = frame.meta["CAMERA"]
        retval["PROGRAM"] = frame.meta["PROGRAM"]
        retval["FLAVOR"] = frame.meta["FLAVOR"]
        retval["NIGHT"] = frame.meta["NIGHT"]
        kwargs=self.config['kwargs']
        if "REFERENCE" in kwargs:
            retval['FIDSNR_TGT_REF']=kwargs["REFERENCE"]

        ra = fibermap["RA_TARGET"]
        dec = fibermap["DEC_TARGET"]

        #- select band for mag, using DECAM_R if present
        if param is None:
            log.debug("Param is None. Using default param instead")
            param = {
                "SNR_FLUXTHRESH":0.0, # Minimum value of flux to go into SNR calc. 
                "FIDSNR_NORMAL_RANGE":[6.5, 7.5],
                "FIDSNR_WARN_RANGE":[6.0, 8.0],
                "FIDMAG":22.
                }

        fidboundary=None
        if amps: 
            #- get the pixel boundary and fiducial boundary in flux-wavelength space
            leftmax = dict_countbins["LEFT_MAX_FIBER"]
            rightmin = dict_countbins["RIGHT_MIN_FIBER"]
            bottommax = dict_countbins["BOTTOM_MAX_WAVE_INDEX"]
            topmin = dict_countbins["TOP_MIN_WAVE_INDEX"]
            fidboundary = qalib.slice_fidboundary(frame,leftmax,rightmin,bottommax,topmin)
        #qadict = qalib.SignalVsNoise(frame,param,fidboundary=fidboundary)
        qadict = qalib.SNRFit(frame,param,fidboundary=fidboundary)

        #- Check for inf and nans in missing magnitudes for json support of QLF #TODO review this later
        for mag in [qadict["ELG_SNR_MAG"][1],qadict["LRG_SNR_MAG"][1],qadict["QSO_SNR_MAG"][1],qadict["STAR_SNR_MAG"][1]]:
            k=np.where(~np.isfinite(mag))[0]
            if len(k) > 0:
                log.warning("{} objects have no or unphysical magnitudes".format(len(k)))
            mag=np.array(mag)
            mag[k]=26.  #- Putting 26, so as to make sure within reasonable range for plots.
        retval["METRICS"] = qadict
        retval["PARAMS"] = param

        snrwarn=[]
        retval["METRICS"]["FIDSNR_STAT"] = snrwarn

        #- http post if valid
        if qlf:
            qlf_post(retval)            

        if qafile is not None:
            outfile = qa.write_qa_ql(qafile,retval)
            log.debug("Output QA data is in {}".format(outfile))
        if qafig is not None:
            from desispec.qa.qa_plots_ql import plot_SNR
            plot_SNR(retval,qafig)         
            log.debug("Output QA fig {}".format(qafig))

        return retval

    def get_default_config(self):
        return {}

