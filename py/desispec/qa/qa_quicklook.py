""" 
Monitoring algorithms for Quicklook pipeline
"""

import os,sys
import datetime
import numpy as np
import scipy.ndimage
import yaml
import astropy.io.fits as fits
import desispec.qa.qa_plots_ql as plot
from desispec.quicklook.qas import MonitoringAlg, QASeverity
from desispec.quicklook import qlexceptions
from desispec.quicklook import qllogger
from astropy.time import Time
from desispec.qa import qalib
from desispec.io import qa, read_params
from desispec.io.meta import findfile
from desispec.io.sky import read_sky
from desispec.image import Image as im
from desispec.frame import Frame as fr
from desispec.preproc import _parse_sec_keyword

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

def get_image(filetype,night,expid,camera,rawdir):
    '''
    Make image object from file if in development mode
    '''
    #- Find correct file for QA
    imagefile = findfile(filetype,int(night),int(expid),camera,rawdata_dir=rawdir)

    #- Create necessary input for desispec.image
    image = fits.open(imagefile)
    pix = image[0].data
    ivar = image[1].data
    mask = image[2].data
    readnoise = image[3].data
    meta = image[0].header

    #- Create image object
    imageobj = im(pix,ivar,mask=mask,readnoise=readnoise,camera=camera,meta=meta)

    return imageobj

def get_frame(filetype,night,expid,camera,specdir):
    '''
    Make frame object from file if in development mode
    '''
    #- Find correct file for QA
    framefile = findfile(filetype,int(night),int(expid),camera,specprod_dir=specdir)

    #- Create necessary input for desispec.frame
    frame = fits.open(framefile)
    wave = frame[3].data
    flux = frame[0].data
    ivar = frame[1].data
    fibermap = frame[5].data
    fibers = fibermap['FIBER']
    meta = frame[0].header

    #- Create frame object
    frameobj = fr(wave,flux,ivar,fibers=fibers,fibermap=fibermap,meta=meta)

    return frameobj

class Bias_From_Overscan(MonitoringAlg):
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="BIAS_OVERSCAN"
        import astropy
        rawtype=astropy.io.fits.hdu.hdulist.HDUList
        kwargs=config['kwargs']
        parms=kwargs['param']
        key=kwargs['refKey'] if 'refKey' in kwargs else "BIAS_AMP"
        status=kwargs['statKey'] if 'statKey' in kwargs else "BIAS_STAT"
        kwargs["SAMI_RESULTKEY"]=key
        kwargs["SAMI_QASTATUSKEY"]=status

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

        if kwargs["singleqa"] == 'Bias_From_Ovescan':
            rawfile = findfile(filetype,int(night),int(expid),camera,rawdata_dir=kwargs["rawdir"])
            raw = fits.open(rawfile)
        else:
            raw=args[0]

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

        return self.run_qa(raw,camera,paname=paname,amps=amps, qafile=qafile,qafig=qafig, param=param, qlf=qlf, refmetrics=refmetrics)

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

        #- Combine data from each row and take average
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

        #- Calculate upper and lower bounds of 1, 2, and 3 sigma
        sig1_lo = np.percentile(full_data,50.-(param['PERCENTILES'][0]/2.))
        sig1_hi = np.percentile(full_data,50.+(param['PERCENTILES'][0]/2.))
        sig2_lo = np.percentile(full_data,50.-(param['PERCENTILES'][1]/2.))
        sig2_hi = np.percentile(full_data,50.+(param['PERCENTILES'][1]/2.))
        sig3_lo = np.percentile(full_data,50.-(param['PERCENTILES'][2]/2.))
        sig3_hi = np.percentile(full_data,50.+(param['PERCENTILES'][2]/2.))

        #- Find difference between upper and lower sigma bounds
        diff1sig = sig1_hi - sig1_lo
        diff2sig = sig2_hi - sig2_lo
        diff3sig = sig3_hi - sig3_lo

        #- Calculate number of pixels below 5 sigma
        sig5_value = np.percentile(full_data,3e-5)
        data5sig = len(np.where(full_data <= sig5_value)[0])

        retval["PARAMS"] = param
        if "REFERENCE" in kwargs:
            retval['PARAMS']['BIAS_AMP_REF']=kwargs["REFERENCE"]

        biasdiff_err='NORMAL'
        if amps:
            bias_amps=np.array(bias_overscan)

#            retval["METRICS"]={'BIAS':bias,'BIAS_AMP':bias_amps,"DIFF1SIG":diff1sig,"DIFF2SIG":diff2sig,"DIFF3SIG":diff3sig,"DATA5SIG":data5sig,"MEANBIAS_ROW":mean_row}
            retval["METRICS"]={'BIAS':bias,'BIAS_AMP':bias_amps,"DIFF1SIG":diff1sig,"DIFF2SIG":diff2sig,"DIFF3SIG":diff3sig,"DATA5SIG":data5sig,"MEANBIAS_ROW":mean_row,"BIAS_STAT":biasdiff_err}
        else:
#            retval["METRICS"]={'BIAS':bias,"DIFF1SIG":diff1sig,"DIFF2SIG":diff2sig,"DIFF3SIG":diff3sig,"DATA5SIG":data5sig,"MEANBIAS_ROW":mean_row}
            retval["METRICS"]={'BIAS':bias,"DIFF1SIG":diff1sig,"DIFF2SIG":diff2sig,"DIFF3SIG":diff3sig,"DATA5SIG":data5sig,"MEANBIAS_ROW":mean_row,"BIAS_STAT":biasdiff_err}

        #- http post if needed
        if qlf:
            qlf_post(retval)    

        if qafile is not None:
            outfile = qa.write_qa_ql(qafile,retval)
            log.debug("Output QA data is in {}".format(outfile))
        if qafig is not None:
            plot.plot_bias_overscan(retval,qafig)
            
            log.debug("Output QA fig {}".format(qafig))
        
        return retval

    def get_default_config(self):
        return {}


class Get_RMS(MonitoringAlg):
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="RMS"
        kwargs=config['kwargs']
        parms=kwargs['param']
        key=kwargs['refKey'] if 'refKey' in kwargs else "NOISE_AMP" 
        status=kwargs['statKey'] if 'statKey' in kwargs else "NOISE_STAT" 
        kwargs["SAMI_RESULTKEY"]=key
        kwargs["SAMI_QASTATUSKEY"]=status
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

        if kwargs["singleqa"] == 'Get_RMS':
            night = kwargs['night']
            expid = '{:08d}'.format(kwargs['expid'])
            camera = kwargs['camera']
            image = get_image('pix',night,expid,camera,kwargs["rawdir"])
        else:
            image=args[0]

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

        return self.run_qa(image,paname=paname,amps=amps,qafile=qafile,qafig=qafig, param=param, qlf=qlf, refmetrics=refmetrics)

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

        # return rms values in rms/sqrt(exptime)
        rmsccd=qalib.getrms(image.pix/np.sqrt(image.meta["EXPTIME"])) #- should we add dark current and/or readnoise to this as well?

        if param is None:
            log.debug("Param is None. Using default param instead")
            param = {
                "NOISE_NORMAL_RANGE":[-1.0, 1.0],
                "NOISE_WARN_RANGE":[-2.0, 2.0]
                }

        retval["PARAMS"] = param
        if "REFERENCE" in kwargs:
            retval['PARAMS']['NOISE_AMP_REF']=kwargs["REFERENCE"]

        expnum=[]
        rms_row=[]
        rms_amps=[]
        rms_over_amps=[]
        overscan_values=[]
        #- get amp/overcan boundary in pixels
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

        rmsdiff_err='NORMAL'
        if amps:
            rms_amps=[]
            rms_over_amps=[]
            overscan_values=[]
            #- get amp/overcan boundary in pixels
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

#            retval["METRICS"]={"RMS":rmsccd,"NOISE":rmsover,"RMS_AMP":np.array(rms_amps),"NOISE_AMP":np.array(rms_over_amps),"RMS_ROW":rms_row,"EXPNUM_WARN":expnum}
            retval["METRICS"]={"RMS":rmsccd,"NOISE":rmsover,"RMS_AMP":np.array(rms_amps),"NOISE_AMP":np.array(rms_over_amps),"RMS_ROW":rms_row,"NOISE_STAT":rmsdiff_err,"EXPNUM_WARN":expnum}
        else:
#            retval["METRICS"]={"RMS":rmsccd,"NOISE":rmsover,"RMS_ROW":rms_row,"EXPNUM_WARN":expnum}
            retval["METRICS"]={"RMS":rmsccd,"NOISE":rmsover,"RMS_ROW":rms_row,"NOISE_STAT":rmsdiff_err,"EXPNUM_WARN":expnum}

        if qlf:
            qlf_post(retval)  

        if qafile is not None:
            outfile = qa.write_qa_ql(qafile,retval)
            log.debug("Output QA data is in {}".format(outfile))
        if qafig is not None:
            plot.plot_RMS(retval,qafig)            
            log.debug("Output QA fig {}".format(qafig))

        return retval    

    def get_default_config(self):
        return {}


class Calc_XWSigma(MonitoringAlg):
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="XWSIGMA"
        kwargs=config['kwargs']
        parms=kwargs['param']
        key=kwargs['refKey'] if 'refKey' in kwargs else "WSIGMA_MED_SKY"
        status=kwargs['statKey'] if 'statKey' in kwargs else "XWSIGMA_STAT"
        kwargs["SAMI_RESULTKEY"]=key
        kwargs["SAMI_QASTATUSKEY"]=status
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

        if kwargs["singleqa"] == 'Calc_XWSigma':
            night = kwargs['night']
            expid = '{:08d}'.format(kwargs['expid'])
            camera = kwargs['camera']
            image = get_image('pix',night,expid,camera,kwargs["rawdir"])
        else:
            image=args[0]

        fibermap=kwargs['FiberMap'] 
 
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
 
        return self.run_qa(fibermap,image,paname=paname,amps=amps,psf=psf, qafile=qafile,qafig=qafig, param=param, qlf=qlf, refmetrics=refmetrics)
 
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

        #- Ensure that the QA will run even if 500 spectra aren't present
        if fibermap['FIBER'].shape[0] >= 500:
            fibers = 500
        else:
            fibers = fibermap['FIBER'].shape[0]

        #- dw and dp are wavelength/pixel ranges used as region to calculate Gaussian over peaks
        #- hardcoded until a better method is found, perhaps configurable (R.S.)
        dw=2.
        dp=3
        #- Get wavelength ranges around peaks
        peak_wave=[]
        peaks=param['{}_PEAKS'.format(camera[0].upper())]
        for p in range(len(peaks)):
            peak_lower = peaks[p] - dw
            peak_upper = peaks[p] + dw
            peak_wave.append(peak_lower)
            peak_wave.append(peak_upper)

        if fibermap["OBJTYPE"][0] == 'ARC':
            import desispec.psf
            psf=desispec.psf.PSF(psf)

        xfails=[]
        wfails=[]
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
        for i in range(fibers):
            xsig=[]
            wsig=[]
            #- Use psf information to convert wavelength to pixel values
            xpix=psf.x(ispec=i,wavelength=peak_wave)
            ypix=psf.y(ispec=i,wavelength=peak_wave)
            for peak in range(len(peaks)):
                #- Find x and y pixel values around sky lines
                xpix_peak=np.arange(int(round(xpix[2*peak]))-dp,int(round(xpix[2*peak+1]))+dp+1,1)
                ypix_peak=np.arange(int(round(ypix[2*peak])),int(round(ypix[2*peak+1])),1)
                #- Fit gaussian to counts in pixels around sky line
                #- If any values fail, store x/w, wavelength, and fiber
                try:
                    xpopt,xpcov=curve_fit(qalib.gauss,np.arange(len(xpix_peak)),image.pix[int(np.mean(ypix_peak)),xpix_peak])
                except:
                    xfail=[i,peaks[peak]]
                    xfails.append(xfail)
                    pass
                try:
                    wpopt,wpcov=curve_fit(qalib.gauss,np.arange(len(ypix_peak)),image.pix[ypix_peak,int(np.mean(xpix_peak))])
                except:
                    wfail=[i,peaks[peak]]
                    wfails.append(wfail)
                    pass

                #- Save sigmas from fits
                xs=np.abs(xpopt[2])
                ws=np.abs(wpopt[2])
                xsig.append(xs)
                wsig.append(ws)

                xsigma_avg=np.mean(xsig)
                wsigma_avg=np.mean(wsig)
                xsigma.append(xsigma_avg)
                wsigma.append(wsigma_avg)
 
            if fibermap['OBJTYPE'][i]=='SKY':
                xsigma_sky=xsigma
                wsigma_sky=wsigma

            #- Excluding fibers 240-260 in case some fibers overlap amps
            #- This shouldn't cause a significant loss of information 
            if amps:
                if fibermap['FIBER'][i]<240:
                    if camera[0]=="b":
                        xsig_amp1=np.array([xsig[0]])
                        xsig_amp3=np.array([xsig[1],xsig[2]])
                        wsig_amp1=np.array([wsig[0]])
                        wsig_amp3=np.array([wsig[1],wsig[2]])
                    if camera[0]=="r":
                        xsig_amp1=np.array([xsig[0],xsig[1]])
                        xsig_amp3=np.array([xsig[2],xsig[3],xsig[4]])
                        wsig_amp1=np.array([wsig[0],wsig[1]])
                        wsig_amp3=np.array([wsig[2],wsig[3],wsig[4]])
                    if camera[0]=="z":
                        xsig_amp1=np.array([xsig[0],xsig[1],xsig[2]])
                        xsig_amp3=np.array([xsig[3]])
                        wsig_amp1=np.array([wsig[0],wsig[1],wsig[2]])
                        wsig_amp3=np.array([wsig[3]])
    
                    xsigma_amp1.append(xsig_amp1)
                    wsigma_amp1.append(wsig_amp1)
                    xsigma_amp3.append(xsig_amp3)
                    wsigma_amp3.append(wsig_amp3)

                if fibermap['FIBER'][i]>260:
                    if camera[0]=="b":
                        xsig_amp2=np.array([xsig[0]])
                        xsig_amp4=np.array([xsig[1],xsig[2]])
                        wsig_amp2=np.array([wsig[0]])
                        wsig_amp4=np.array([wsig[1],wsig[2]])
                    if camera[0]=="r":
                        xsig_amp2=np.array([xsig[0],xsig[1]])
                        xsig_amp4=np.array([xsig[2],xsig[3],xsig[4]])
                        wsig_amp2=np.array([wsig[0],wsig[1]])
                        wsig_amp4=np.array([wsig[2],wsig[3],wsig[4]])
                    if camera[0]=="z":
                        xsig_amp2=np.array([xsig[0],xsig[1],xsig[2]])
                        xsig_amp4=np.array([xsig[3]])
                        wsig_amp2=np.array([wsig[0],wsig[1],wsig[2]])
                        wsig_amp4=np.array([wsig[3]])
    
                    xsigma_amp2.append(xsig_amp2)
                    wsigma_amp2.append(wsig_amp2)
                    xsigma_amp4.append(xsig_amp4)
                    wsigma_amp4.append(wsig_amp4)
  
                if fibermap['FIBER'].shape[0]<260:
                    xsigma_amp2=np.zeros(len(xsigma))
                    xsigma_amp4=np.zeros(len(xsigma))
                    wsigma_amp2=np.zeros(len(wsigma))
                    wsigma_amp4=np.zeros(len(wsigma))

        #- Calculate desired output metrics 
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

        xwfails=[xfails,wfails]

        retval["PARAMS"] = param
        if "REFERENCE" in kwargs:
            retval['PARAMS']['XWSIGMA_REF']=kwargs["REFERENCE"]

        shift_err='NORMAL'
        if amps:
#            retval["METRICS"]={"RA":ra,"DEC":dec, "XSIGMA":xsigma,"XSIGMA_MED":xsigma_med,"XSIGMA_AMP":xsigma_amp,"XSIGMA_MED_SKY":xsigma_med_sky,"XSHIFT":xshift,"XSHIFT_FIB":xshift_fib,"XSHIFT_AMP":xshift_amp,"WSIGMA":wsigma,"WSIGMA_MED":wsigma_med,"WSIGMA_AMP":wsigma_amp,"WSIGMA_MED_SKY":wsigma_med_sky,"WSHIFT":wshift,"WSHIFT_FIB":wshift_fib,"WSHIFT_AMP":wshift_amp,"XWSIGMA":xwsigma}
            retval["METRICS"]={"RA":ra,"DEC":dec, "XSIGMA":xsigma,"XSIGMA_MED":xsigma_med,"XSIGMA_AMP":xsigma_amp,"XSHIFT":xshift,"XSHIFT_FIB":xshift_fib,"XSHIFT_AMP":xshift_amp,"WSIGMA":wsigma,"WSIGMA_MED":wsigma_med,"WSIGMA_AMP":wsigma_amp,"WSHIFT":wshift,"WSHIFT_FIB":wshift_fib,"WSHIFT_AMP":wshift_amp,"XWSIGMA":xwsigma,"XWFIT_FAIL":xwfails,"XWSIGMA_STAT":shift_err}
        else:
#            retval["METRICS"]={"RA":ra,"DEC":dec, "XSIGMA":xsigma,"XSIGMA_MED":xsigma_med,"XSIGMA_MED_SKY":xsigma_med_sky,"XSHIFT":xshift,"XSHIFT_FIB":xshift_fib,"WSIGMA":wsigma,"WSIGMA_MED":wsigma_med,"WSIGMA_MED_SKY":wsigma_med_sky,"WSHIFT":wshift,"WSHIFT_FIB":wshift_fib,"XWSIGMA":xwsigma}
            retval["METRICS"]={"RA":ra,"DEC":dec, "XSIGMA":xsigma,"XSIGMA_MED":xsigma_med,"XSIGMA_MED_SKY":xsigma_med_sky,"XSHIFT":xshift,"XSHIFT_FIB":xshift_fib,"WSIGMA":wsigma,"WSIGMA_MED":wsigma_med,"WSIGMA_MED_SKY":wsigma_med_sky,"WSHIFT":wshift,"WSHIFT_FIB":wshift_fib,"XWFIT_FAIL":xwfails,"XWSIGMA_STAT":shift_err}

        #- http post if needed
        if qlf:
            qlf_post(retval)    

        if qafile is not None:
            outfile = qa.write_qa_ql(qafile,retval)
            log.debug("Output QA data is in {}".format(outfile))
        if qafig is not None:
            plot.plot_XWSigma(retval,qafig)

            log.debug("Output QA fig {}".format(qafig))

        return retval
 
    def get_default_config(self):
        return {}


class Count_Pixels(MonitoringAlg):
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="COUNTPIX"
        kwargs=config['kwargs']
        parms=kwargs['param']
        key=kwargs['refKey'] if 'refKey' in kwargs else "NPIX_AMP"
        status=kwargs['statKey'] if 'statKey' in kwargs else "NPIX_STAT"
        kwargs["SAMI_RESULTKEY"]=key
        kwargs["SAMI_QASTATUSKEY"]=status
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

        if kwargs["singleqa"] == 'Count_Pixels':
            night = kwargs['night']
            expid = '{:08d}'.format(kwargs['expid'])
            camera = kwargs['camera']
            image = get_image('pix',night,expid,camera,kwargs["rawdir"])
        else:
            image=args[0]

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

        return self.run_qa(image,paname=paname,amps=amps,qafile=qafile,qafig=qafig, param=param, qlf=qlf, refmetrics=refmetrics)

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

        if param is None:
            log.debug("Param is None. Using default param instead")
            param = {
                 "CUTLO":3,   # low threshold for number of counts in sigmas
                 "CUTHI":10,
                 "NPIX_NORMAL_RANGE":[200.0, 500.0],
                 "NPIX_WARN_RANGE":[50.0, 650.0]
                 }

        retval["PARAMS"] = param
        if "REFERENCE" in kwargs:
            retval['PARAMS']['NPIX_AMP_REF']=kwargs["REFERENCE"]

        #- get the counts over entire CCD in counts per second
        npixlo=qalib.countpix(image.pix,nsig=param['CUTLO']) #- above 3 sigma in counts
        npixhi=qalib.countpix(image.pix,nsig=param['CUTHI']) #- above 10 sigma in counts

        npix_err='NORMAL'
        #- get the counts for each amp
        if amps:
            npixlo_amps=[]
            npixhi_amps=[]
            #- get amp boundary in pixels
            for kk in ['1','2','3','4']:
                ampboundary=_parse_sec_keyword(image.meta["CCDSEC"+kk])
                npixlo_thisamp=qalib.countpix(image.pix[ampboundary]/image.meta["EXPTIME"],nsig=param['CUTLO'])
                npixlo_amps.append(npixlo_thisamp)
                npixhi_thisamp=qalib.countpix(image.pix[ampboundary]/image.meta["EXPTIME"],nsig=param['CUTHI'])
                npixhi_amps.append(npixhi_thisamp)

#            retval["METRICS"]={"NPIX_LOW":npixlo,"NPIX_HIGH":npixhi,"NPIX_AMP": npixlo_amps,"NPIX_HIGH_AMP": npixhi_amps}
            retval["METRICS"]={"NPIX_LOW":npixlo,"NPIX_HIGH":npixhi,"NPIX_AMP": npixlo_amps,"NPIX_HIGH_AMP": npixhi_amps,"NPIX_STAT":npix_err}
        else:
#            retval["METRICS"]={"NPIX_LOW":npixlo,"NPIX_HIGH":npixhi}
            retval["METRICS"]={"NPIX_LOW":npixlo,"NPIX_HIGH":npixhi,"NPIX_STAT":npix_err}

        if qlf:
            qlf_post(retval)      

        if qafile is not None:
            outfile = qa.write_qa_ql(qafile,retval)
            log.debug("Output QA data is in {}".format(outfile))
        if qafig is not None:
            plot.plot_countpix(retval,qafig)
            
            log.debug("Output QA fig {}".format(qafig))

        return retval    

    def get_default_config(self):
        return {}


class CountSpectralBins(MonitoringAlg):

    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="COUNTBINS"
        kwargs=config['kwargs']
        parms=kwargs['param']
        key=kwargs['refKey'] if 'refKey' in kwargs else "NGOODFIB"
        status=kwargs['statKey'] if 'statKey' in kwargs else "NGOODFIB_STAT"
        kwargs["SAMI_RESULTKEY"]=key
        kwargs["SAMI_QASTATUSKEY"]=status

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

        if kwargs["singleqa"] == 'CountSpectralBins':
            night = kwargs['night']
            expid = '{:08d}'.format(kwargs['expid'])
            camera = kwargs['camera']
            frame = get_frame('frame',night,expid,camera,kwargs["specdir"])
        else:
            frame=args[0]

        fibermap=kwargs['FiberMap']

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

        return self.run_qa(fibermap,frame,paname=paname,amps=amps,psf=psf, qafile=qafile,qafig=qafig, param=param, qlf=qlf, refmetrics=refmetrics)


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
        if "REFERENCE" in kwargs:
            retval['PARAMS']['NGOODFIB_REF']=kwargs["REFERENCE"]

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

        ngood_err='NORMAL'
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

#            retval["METRICS"]={"RA":ra,"DEC":dec, "NBINSLOW":countslo,"NBINSMED":countsmed,"NBINSHIGH":countshi, "NBINSLOW_AMP":averagelo_amps,"NBINSMED_AMP":averagemed_amps,"NBINSHIGH_AMP":averagehi_amps, "NGOODFIB": ngoodfibers, "NBINSHI_TEMP":nbinshi_temp}
            retval["METRICS"]={"RA":ra,"DEC":dec, "NBINSLOW":countslo,"NBINSMED":countsmed,"NBINSHIGH":countshi, "NBINSLOW_AMP":averagelo_amps,"NBINSMED_AMP":averagemed_amps,"NBINSHIGH_AMP":averagehi_amps, "NGOODFIB": ngoodfibers, "NBINSHI_TEMP":nbinshi_temp,"NGOODFIB_STAT":ngood_err}
        else:
#            retval["METRICS"]={"RA":ra,"DEC":dec, "NBINSLOW":countslo,"NBINSMED":countsmed,"NBINSHIGH":countshi,"NGOODFIB": ngoodfibers, "NBINSHI_TEMP":nbinshi_temp}
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
            plot.plot_countspectralbins(retval,qafig)
            
            log.debug("Output QA fig {}".format(qafig))
        
        return retval

    def get_default_config(self):
        return {}


class Sky_Continuum(MonitoringAlg):
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="SKYCONT"
        kwargs=config['kwargs']
        parms=kwargs['param']
        key=kwargs['refKey'] if 'refKey' in kwargs else "SKYCONT"
        status=kwargs['statKey'] if 'statKey' in kwargs else "SKYCONT_STAT"
        kwargs["SAMI_RESULTKEY"]=key
        kwargs["SAMI_QASTATUSKEY"]=status
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

        dict_countbins=None
        if kwargs["singleqa"] == 'Sky_Continuum':
            night = kwargs['night']
            expid = '{:08d}'.format(kwargs['expid'])
            camera = kwargs['camera']
            frame = get_frame('fframe',night,expid,camera,kwargs["specdir"])
            reduxpath = os.path.join(os.environ['QL_SPEC_REDUX'],'exposures',night,expid)
            with open(os.path.join(reduxpath,'ql-countbins-{}-{}.yaml'.format(camera,expid))) as f:
                cdict = yaml.load(f)
            dict_countbins = {'LEFT_MAX_FIBER' : cdict['LEFT_MAX_FIBER'], 'RIGHT_MIN_FIBER' : cdict['RIGHT_MIN_FIBER'], 'BOTTOM_MAX_WAVE_INDEX' : cdict['BOTTOM_MAX_WAVE_INDEX'], 'TOP_MIN_WAVE_INDEX' : cdict['TOP_MIN_WAVE_INDEX']}
        else:
            frame=args[0]
            if "dict_countbins" in kwargs:
                dict_countbins=kwargs["dict_countbins"]

        fibermap=kwargs['FiberMap']
        camera=frame.meta["CAMERA"]
        
        if "paname" in kwargs:
            paname=kwargs["paname"]

        if "ReferenceMetrics" in kwargs: refmetrics=kwargs["ReferenceMetrics"]
        else: refmetrics=None

        amps=False
        if "amps" in kwargs:
            amps=kwargs["amps"]

        if "param" in kwargs: param=kwargs["param"]
        else: param=None

        if "{}_CONT".format(camera[0].upper()) in param:
            wrange1=param["{}_CONT".format(camera[0].upper())][0]
            wrange2=param["{}_CONT".format(camera[0].upper())][1]
        else:
            if camera[0]=="b":
                wrange1= "4000,4500"
                wrange2= "5250,5550"
            if camera[0]=="r":
                wrange1= "5950,6200"
                wrange2= "6990,7230"
            if camera[0]=="z":
                wrange1= "8120,8270"
                wrange2= "9110,9280"

        if "qlf" in kwargs:
             qlf=kwargs["qlf"]
        else: qlf=False

        if "qafile" in kwargs: qafile = kwargs["qafile"]
        else: qafile = None

        if "qafig" in kwargs: qafig=kwargs["qafig"]
        else: qafig=None
        return self.run_qa(fibermap,frame,wrange1=wrange1,wrange2=wrange2,paname=paname,amps=amps, dict_countbins=dict_countbins,qafile=qafile,qafig=qafig, param=param, qlf=qlf, refmetrics=refmetrics)

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

        ra = fibermap["RA_TARGET"]
        dec = fibermap["DEC_TARGET"]

        if param is None:
            log.debug("Param is None. Using default param instead")
            desi_params = read_params()
            param = {}
            for key in ['B_CONT','R_CONT', 'Z_CONT', 'SKYCONT_WARN_RANGE', 'SKYCONT_ALARM_RANGE']:
                param[key] = desi_params['qa']['skysub']['PARAMS'][key]
        retval["PARAMS"] = param
        if "REFERENCE" in kwargs:
            retval['PARAMS']['SKYCONT_REF']=kwargs["REFERENCE"]

        skyfiber, contfiberlow, contfiberhigh, meancontfiber, skycont = qalib.sky_continuum(
            frame, wrange1, wrange2)

        skycont_err = 'NORMAL'
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

#            retval["METRICS"]={"RA":ra,"DEC":dec, "SKYFIBERID": skyfiber.tolist(), "SKYCONT":skycont, "SKYCONT_FIBER":meancontfiber, "SKYCONT_AMP":skycont_amps}
            retval["METRICS"]={"RA":ra,"DEC":dec, "SKYFIBERID": skyfiber.tolist(), "SKYCONT":skycont, "SKYCONT_FIBER":meancontfiber, "SKYCONT_AMP":skycont_amps, "SKYCONT_STAT":skycont_err}
        else: 
#            retval["METRICS"]={"RA":ra,"DEC":dec, "SKYFIBERID": skyfiber.tolist(), "SKYCONT":skycont, "SKYCONT_FIBER":meancontfiber}
            retval["METRICS"]={"RA":ra,"DEC":dec, "SKYFIBERID": skyfiber.tolist(), "SKYCONT":skycont, "SKYCONT_FIBER":meancontfiber, "SKYCONT_STAT":skycont_err}

        if qlf:
            qlf_post(retval)    

        if qafile is not None:
            outfile = qa.write_qa_ql(qafile,retval)
            log.debug("Output QA data is in {}".format(outfile))

        if qafig is not None:
            plot.plot_sky_continuum(retval,qafig)
            
            log.debug("Output QA fig {}".format(qafig))
        
        return retval

    def get_default_config(self):
        return {}


class Sky_Peaks(MonitoringAlg):
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="SKYPEAK"
        kwargs=config['kwargs']
        parms=kwargs['param']
        key=kwargs['refKey'] if 'refKey' in kwargs else "PEAKCOUNT_MED_SKY"
        status=kwargs['statKey'] if 'statKey' in kwargs else "PEAKCOUNT_STAT"
        kwargs["SAMI_RESULTKEY"]=key
        kwargs["SAMI_QASTATUSKEY"]=status
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

        if kwargs["singleqa"] == 'Sky_Peaks':
            night = kwargs['night']
            expid = '{:08d}'.format(kwargs['expid'])
            camera = kwargs['camera']
            frame = get_frame('fframe',night,expid,camera,kwargs["specdir"])
        else:
            frame=args[0]

        fibermap=kwargs['FiberMap']

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

        return self.run_qa(fibermap,frame,paname=paname,amps=amps,psf=psf, qafile=qafile, qafig=qafig, param=param, qlf=qlf, refmetrics=refmetrics)

    def run_qa(self,fibermap,frame,paname=None,amps=False,psf=None, qafile=None,qafig=None, param=None, qlf=False, refmetrics=None):
        from desispec.qa.qalib import sky_peaks
        retval={}
        retval["PANAME"] = paname
        retval["QATIME"] = datetime.datetime.now().isoformat()
        retval["EXPID"] = '{0:08d}'.format(frame.meta["EXPID"])
        retval["CAMERA"] = camera = frame.meta["CAMERA"]
        retval["PROGRAM"] = frame.meta["PROGRAM"]
        retval["FLAVOR"] = frame.meta["FLAVOR"]
        retval["NIGHT"] = frame.meta["NIGHT"]
        kwargs=self.config['kwargs']

        ra = fibermap["RA_TARGET"]
        dec = fibermap["DEC_TARGET"]

        # Parameters
        if param is None:
            log.info("Param is None. Using default param instead")
            desi_params = read_params()
            param = desi_params['qa']['skypeaks']['PARAMS']

        # Run
        nspec_counts, sky_counts = sky_peaks(param, frame, amps=amps)
        rms_nspec = qalib.getrms(nspec_counts)
        rms_skyspec = qalib.getrms(sky_counts)
        sumcount_med_sky=[]

        retval["PARAMS"] = param
        if "REFERENCE" in kwargs:
            retval['PARAMS']['PEAKCOUNT_REF']=kwargs["REFERENCE"]

#        retval["METRICS"]={"RA":ra,"DEC":dec, "PEAKCOUNT":nspec_counts,"PEAKCOUNT_RMS":rms_nspec,"PEAKCOUNT_MED_SKY":sumcount_med_sky,"PEAKCOUNT_RMS_SKY":rms_skyspec}
        sumcount_err='NORMAL'
        retval["METRICS"]={"RA":ra,"DEC":dec, "PEAKCOUNT":nspec_counts,"PEAKCOUNT_RMS":rms_nspec,"PEAKCOUNT_MED_SKY":sumcount_med_sky,"PEAKCOUNT_RMS_SKY":rms_skyspec,"PEAKCOUNT_STAT":sumcount_err}

        if qlf:
            qlf_post(retval)

        if qafile is not None:
            outfile = qa.write_qa_ql(qafile,retval)
            log.debug("Output QA data is in {}".format(outfile))
        if qafig is not None:
            plot.plot_sky_peaks(retval,qafig)

            log.debug("Output QA fig {}".format(qafig))

        return retval

    def get_default_config(self):
        return {}


class Sky_Residual(MonitoringAlg):
    """ 
    Use offline sky_residual function to calculate sky residuals
    """
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="RESIDUAL"
        kwargs=config['kwargs']
        parms=kwargs['param']
        key=kwargs['refKey'] if 'refKey' in kwargs else "RESIDRMS"
        status=kwargs['statKey'] if 'statKey' in kwargs else "RESIDRMS_STAT"
        kwargs["SAMI_RESULTKEY"]=key
        kwargs["SAMI_QASTATUSKEY"]=status

        if "ReferenceMetrics" in kwargs:
            r=kwargs["ReferenceMetrics"]
            if key in r:
                kwargs["REFERENCE"]=r[key]

        if "RESID_WARN_RANGE" in parms and "RESID_NORMAL_RANGE" in parms:
            kwargs["RANGES"]=[(np.asarray(parms["RESID_WARN_RANGE"]),QASeverity.WARNING),
                              (np.asarray(parms["RESID_NORMAL_RANGE"]),QASeverity.NORMAL)]# sorted by most severe to least severe 

        MonitoringAlg.__init__(self,name,fr,config,logger)
    def run(self,*args,**kwargs):
        if len(args) == 0 :
            raise qlexceptions.ParameterException("Missing input parameter")
        if not self.is_compatible(type(args[0])):
            raise qlexceptions.ParameterException("Incompatible input. Was expecting {} got {}".format(type(self.__inpType__),type(args[0])))

        dict_countbins=None
        if kwargs["singleqa"] == 'Sky_Residual':
            night = kwargs['night']
            expid = '{:08d}'.format(kwargs['expid'])
            camera = kwargs['camera']
            frame = get_frame('sframe',night,expid,camera,kwargs["specdir"])
            reduxpath = os.path.join(os.environ['QL_SPEC_REDUX'],'exposures',night,expid)
            with open(os.path.join(reduxpath,'ql-countbins-{}-{}.yaml'.format(camera,expid))) as f:
                cdict = yaml.load(f)
            dict_countbins = {'LEFT_MAX_FIBER' : cdict['LEFT_MAX_FIBER'], 'RIGHT_MIN_FIBER' : cdict['RIGHT_MIN_FIBER'], 'BOTTOM_MAX_WAVE_INDEX' : cdict['BOTTOM_MAX_WAVE_INDEX'], 'TOP_MIN_WAVE_INDEX' : cdict['TOP_MIN_WAVE_INDEX']}
        else:
            frame=args[0]
            if "dict_countbins" in kwargs:
                dict_countbins=kwargs["dict_countbins"]

        fibermap=kwargs['FiberMap']
        skymodel=args[1] #- should be skymodel evaluated
        if "SkyFile" in kwargs:
            skyfile=kwargs["SkyFile"]    #- Read sky model file itself from an argument
            log.debug("Using given sky file {} for subtraction".format(skyfile))

            skymodel=read_sky(skyfile)

        if "ReferenceMetrics" in kwargs: refmetrics=kwargs["ReferenceMetrics"]
        else: refmetrics=None

        amps=False
        if "amps" in kwargs:
            amps=kwargs["amps"]

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
        
        return self.run_qa(fibermap,frame,paname=paname,skymodel=skymodel,amps=amps, dict_countbins=dict_countbins, qafile=qafile,qafig=qafig, param=param, qlf=qlf, refmetrics=refmetrics)


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

        skyresid_err='NORMAL'
        retval["PARAMS"] = param
        retval["METRICS"]["RESIDRMS_STAT"]=skyresid_err
        if "REFERENCE" in kwargs:
            retval['PARAMS']['RESIDRMS_REF']=kwargs["REFERENCE"]

        if qafile is not None:
            outfile = qa.write_qa_ql(qafile,retval)
            log.debug("Output QA data is in {}".format(outfile))
        if qafig is not None:
            plot.plot_residuals(retval,qafig)
            
            log.debug("Output QA fig {}".format(qafig))

        return retval

    def get_default_config(self):
        return {}


class Integrate_Spec(MonitoringAlg):
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="INTEG"
        kwargs=config['kwargs']
        parms=kwargs['param']
        key=kwargs['refKey'] if 'refKey' in kwargs else "INTEG_AVG"
        status=kwargs['statKey'] if 'statKey' in kwargs else "MAGDIFF_STAT"
        kwargs["SAMI_RESULTKEY"]=key
        kwargs["SAMI_QASTATUSKEY"]=status
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

        dict_sountbins=None
        if kwargs["singleqa"] == 'Integrate_Spec':
            night = kwargs['night']
            expid = '{:08d}'.format(kwargs['expid'])
            camera = kwargs['camera']
            frame = get_frame('sframe',night,expid,camera,kwargs["specdir"])
            reduxpath = os.path.join(os.environ['QL_SPEC_REDUX'],'exposures',night,expid)
            with open(os.path.join(reduxpath,'ql-countbins-{}-{}.yaml'.format(camera,expid))) as f:
                cdict = yaml.load(f)
            dict_countbins = {'LEFT_MAX_FIBER' : cdict['LEFT_MAX_FIBER'], 'RIGHT_MIN_FIBER' : cdict['RIGHT_MIN_FIBER'], 'BOTTOM_MAX_WAVE_INDEX' : cdict['BOTTOM_MAX_WAVE_INDEX'], 'TOP_MIN_WAVE_INDEX' : cdict['TOP_MIN_WAVE_INDEX']}
        else:
            frame=args[0]
            if "dict_countbins" in kwargs:
                dict_countbins=kwargs["dict_countbins"]

        fibermap=kwargs['FiberMap']

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
        return self.run_qa(fibermap,frame,paname=paname,amps=amps, dict_countbins=dict_countbins, qafile=qafile,qafig=qafig, param=param, qlf=qlf, refmetrics=refmetrics)

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
        if "REFERENCE" in kwargs:
            retval['PARAMS']['MAGDIFF_TGT_REF']=kwargs["REFERENCE"]

        magdiff_avg_amp = [0.0]

        magdiff_err='NORMAL'
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

#            retval["METRICS"]={"RA":ra,"DEC":dec, "INTEG":int_stars, "INTEG_AVG":int_average,"INTEG_AVG_AMP":int_avg_amps, "STD_FIBERID": starfibers.tolist(),"MAGDIFF_TGT":magdiff_avg,"MAGDIFF_AVG_AMP":magdiff_avg_amp}
            retval["METRICS"]={"RA":ra,"DEC":dec, "INTEG":int_stars, "INTEG_AVG":int_average,"INTEG_AVG_AMP":int_avg_amps, "STD_FIBERID": starfibers.tolist(),"MAGDIFF_TGT":magdiff_avg,"MAGDIFF_AVG_AMP":magdiff_avg_amp,"MAGDIFF_STAT":magdiff_err}
        else:
#            retval["METRICS"]={"RA":ra,"DEC":dec, "INTEG":int_stars,"INTEG_AVG":int_average,"STD_FIBERID":starfibers.tolist(),"MAGDIFF_TGT":magdiff_avg}
            retval["METRICS"]={"RA":ra,"DEC":dec, "INTEG":int_stars,"INTEG_AVG":int_average,"STD_FIBERID":starfibers.tolist(),"MAGDIFF_TGT":magdiff_avg,"MAGDIFF_STAT":magdiff_err}

        if qlf:
            qlf_post(retval) 

        if qafile is not None:
            outfile = qa.write_qa_ql(qafile,retval)
            log.debug("Output QA data is in {}".format(outfile))
        if qafig is not None:
            plot.plot_integral(retval,qafig)
            
            log.debug("Output QA fig {}".format(qafig))

        return retval    

    def get_default_config(self):
        return {}
 
class Calculate_SNR(MonitoringAlg):
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="SNR"
        kwargs=config['kwargs']
        parms=kwargs['param']
        key=kwargs['refKey'] if 'refKey' in kwargs else "ELG_FIDSNR"
        status=kwargs['statKey'] if 'statKey' in kwargs else "FIDSNR_STAT"
        kwargs["SAMI_RESULTKEY"]=key
        kwargs["SAMI_QASTATUSKEY"]=status
        if "ReferenceMetrics" in kwargs:
            r=kwargs["ReferenceMetrics"]
            if key in r:
                kwargs["REFERENCE"]=r[key]

        if "FIDSNR_WARN_RANGE" in parms and "FIDSNR_NORMAL_RANGE" in parms:
            kwargs["RANGES"]=[(np.asarray(parms["FIDSNR_WARN_RANGE"]),QASeverity.WARNING),
                              (np.asarray(parms["FIDSNR_NORMAL_RANGE"]),QASeverity.NORMAL)]# sorted by most severe to least severe
        MonitoringAlg.__init__(self,name,fr,config,logger)
    def run(self,*args,**kwargs):
        if len(args) == 0 :
            raise qlexceptions.ParameterException("Missing input parameter")
        if not self.is_compatible(type(args[0])):
            raise qlexceptions.ParameterException("Incompatible input. Was expecting {} got {}".format(type(self.__inpType__),type(args[0])))

        dict_countbins=None
        if kwargs["singleqa"] == 'Calculate_SNR':
            night = kwargs['night']
            expid = '{:08d}'.format(kwargs['expid'])
            camera = kwargs['camera']
            frame = get_frame('sframe',night,expid,camera,kwargs["specdir"])
            reduxpath = os.path.join(os.environ['QL_SPEC_REDUX'],'exposures',night,expid)
            with open(os.path.join(reduxpath,'ql-countbins-{}-{}.yaml'.format(camera,expid))) as f:
                cdict = yaml.load(f)
            dict_countbins = {'LEFT_MAX_FIBER' : cdict['LEFT_MAX_FIBER'], 'RIGHT_MIN_FIBER' : cdict['RIGHT_MIN_FIBER'], 'BOTTOM_MAX_WAVE_INDEX' : cdict['BOTTOM_MAX_WAVE_INDEX'], 'TOP_MIN_WAVE_INDEX' : cdict['TOP_MIN_WAVE_INDEX']}
        else:
            frame=args[0]
            if "dict_countbins" in kwargs:
                dict_countbins=kwargs["dict_countbins"]
        fibermap=kwargs['FiberMap']

        if "ReferenceMetrics" in kwargs: refmetrics=kwargs["ReferenceMetrics"]
        else: refmetrics=None

        amps=False
        if "amps" in kwargs:
            amps=kwargs["amps"]

        if "param" in kwargs: param=kwargs["param"]
        else: param=None
        paname=None
        if "paname" in kwargs:
            paname=kwargs["paname"]

        if "sigmacut" in kwargs: sigmacut=kwargs["sigmacut"]
        else: sigmacut=None

        if "qlf" in kwargs:
             qlf=kwargs["qlf"]
        else: qlf=False

        if "qafile" in kwargs: qafile = kwargs["qafile"]
        else: qafile = None

        if "qafig" in kwargs: qafig=kwargs["qafig"]
        else: qafig = None

        return self.run_qa(fibermap,frame,paname=paname,amps=amps,dict_countbins=dict_countbins, qafile=qafile,qafig=qafig, param=param, qlf=qlf, refmetrics=refmetrics, sigmacut=sigmacut)


    def run_qa(self,fibermap,frame,paname=None,amps=False,dict_countbins=None, qafile=None,qafig=None, qlf=False, param=None, refmetrics=None, sigmacut=None):

        #- return values
        retval={}
        retval["PANAME"] = paname
        retval["QATIME"] = datetime.datetime.now().isoformat()
        retval["EXPID"] = expid = '{0:08d}'.format(frame.meta["EXPID"])
        retval["CAMERA"] = camera = frame.meta["CAMERA"]
        retval["PROGRAM"] = frame.meta["PROGRAM"]
        retval["FLAVOR"] = frame.meta["FLAVOR"]
        retval["NIGHT"] = night = frame.meta["NIGHT"]
        kwargs=self.config['kwargs']

        ra = fibermap["RA_TARGET"]
        dec = fibermap["DEC_TARGET"]
        objlist = set(fibermap["OBJTYPE"])
        if 'SKY' in objlist:
            objlist.remove('SKY')

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
        qadict,badfibs = qalib.SNRFit(frame,night,camera,expid,objlist,param,fidboundary=fidboundary)

        #- Check for inf and nans in missing magnitudes for json support of QLF #TODO review this later

        for obj in range(len(qadict["SNR_MAG_TGT"])):
            for mag in [qadict["SNR_MAG_TGT"][obj]]:
                k=np.where(~np.isfinite(mag))[0]
                if len(k) > 0:
                    log.warning("{} objects have no or unphysical magnitudes".format(len(k)))
            mag=np.array(mag)
            mag[k]=26.  #- Putting 26, so as to make sure within reasonable range for plots.
        retval["METRICS"] = qadict
        retval["PARAMS"] = param
        if "REFERENCE" in kwargs:
            retval['PARAMS']['FIDSNR_TGT_REF']=kwargs["REFERENCE"]

        snrwarn='NORMAL'
        retval["METRICS"]["FIDSNR_STAT"] = snrwarn

        #- http post if valid
        if qlf:
            qlf_post(retval)            

        if qafile is not None:
            outfile = qa.write_qa_ql(qafile,retval)
            log.debug("Output QA data is in {}".format(outfile))
        if qafig is not None:
            plot.plot_SNR(retval,qafig,objlist,badfibs,sigmacut)
            log.debug("Output QA fig {}".format(qafig))

        return retval

    def get_default_config(self):
        return {}


