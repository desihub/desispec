"""
desispec.qa.qa_quicklook
========================

Monitoring algorithms for Quicklook pipeline.
"""

import os,sys
import datetime
import numpy as np
import scipy.ndimage
import yaml
import re
import astropy.io.fits as fits
import desispec.qa.qa_plots_ql as plot
import desispec.quicklook.qlpsf
import desispec.qa.qa_plots_ql as fig
from desispec.quicklook.qas import MonitoringAlg, QASeverity
from desispec.quicklook import qlexceptions
from desispec.quicklook import qllogger
from desispec.quicklook.palib import resample_spec
from astropy.time import Time
from desispec.qa import qalib
from desispec.io import qa, read_params
from desispec.io.meta import findfile
from desispec.io.sky import read_sky
from desispec.image import Image as im
from desispec.frame import Frame as fr
from desispec.preproc import parse_sec_keyword
from desispec.util import runcmd
from desispec.qproc.qframe import QFrame
from desispec.fluxcalibration import isStdStar
from desitarget.targetmask import desi_mask
import astropy
from astropy.io import fits

qlog=qllogger.QLLogger("QuickLook",0)
log=qlog.getlog()

def get_inputs(*args,**kwargs):
    '''
    Get inputs required for each QA
    '''
    inputs={}
    inputs["camera"]=kwargs["camera"]

    if "paname" not in kwargs: inputs["paname"]=None
    else: inputs["paname"]=kwargs["paname"]

    if "ReferenceMetrics" in kwargs: inputs["refmetrics"]=kwargs["ReferenceMetrics"]
    else: inputs["refmetrics"]=None

    inputs["amps"]=False
    if "amps" in kwargs: inputs["amps"]=kwargs["amps"]

    if "param" in kwargs: inputs["param"]=kwargs["param"]
    else: inputs["param"]=None

    inputs["psf"]=None
    if "PSFFile" in kwargs: inputs["psf"]=kwargs["PSFFile"]

    inputs["fibermap"]=None
    if "FiberMap" in kwargs: inputs["fibermap"]=kwargs["FiberMap"]

    if "Peaks" in kwargs: inputs["Peaks"]=kwargs["Peaks"]

    if "qafile" in kwargs: inputs["qafile"] = kwargs["qafile"]
    else: inputs["qafile"]=None

    if "qafig" in kwargs: inputs["qafig"]=kwargs["qafig"]
    else: inputs["qafig"]=None

    if "plotconf" in kwargs: inputs["plotconf"]=kwargs["plotconf"]
    else: inputs["plotconf"]=None

    if "hardplots" in kwargs: inputs["hardplots"]=kwargs["hardplots"]
    else: inputs["hardplots"]=False

    return inputs

def get_image(filetype,night,expid,camera,specdir):
    '''
    Make image object from file if in development mode
    '''
    #- Find correct file for QA
    imagefile = findfile(filetype,int(night),int(expid),camera,specprod_dir=specdir)

    #- Create necessary input for desispec.image
    image = fits.open(imagefile)
    pix = image['IMAGE'].data
    ivar = image['IVAR'].data
    mask = image['MASK'].data
    readnoise = image['READNOISE'].data
    meta = image['IMAGE'].header

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
    wave = frame['WAVE'].data
    flux = frame['FLUX'].data
    ivar = frame['IVAR'].data
    fibermap = frame['FIBERMAP'].data
    fibers = fibermap['FIBER']
    meta = frame['FLUX'].header

    #- Create frame object
    frameobj = QFrame(wave,flux,ivar,fibers=fibers,fibermap=fibermap,meta=meta)

    return frameobj


class Check_HDUs(MonitoringAlg):
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="CHECKHDUS"
        import astropy
        rawtype=astropy.io.fits.hdu.hdulist.HDUList
        kwargs=config['kwargs']
        parms=kwargs['param']
        key=kwargs['refKey'] if 'refKey' in kwargs else "CHECKHDUS"
        status=kwargs['statKey'] if 'statKey' in kwargs else "CHECKHDUS_STATUS"
        kwargs["RESULTKEY"]=key
        kwargs["QASTATUSKEY"]=status

        if "ReferenceMetrics" in kwargs:
            r=kwargs["ReferenceMetrics"]
            if key in r:
                kwargs["REFERENCE"]=r[key]

        MonitoringAlg.__init__(self,name,rawtype,config,logger)
    def run(self,*args,**kwargs):
        if len(args) == 0 :
            log.critical("No parameter is given for this QA! ")
            sys.exit("Check the configuration file")

        if not self.is_compatible(type(args[0])):
            #raise qlexceptions.ParameterException("Incompatible input. Was expecting {} got {}".format(type(self.__inpType__),type(args[0])))
            log.critical("Incompatible input!")
            sys.exit("Was expecting {} got {}".format(type(self.__inpType__),type(args[0])))


        if kwargs["singleqa"] == 'Check_HDUs':
            night = kwargs['night']
            expid = '{:08d}'.format(kwargs['expid'])
            camera = kwargs['camera']
            rawfile = findfile('raw',int(night),int(expid),camera,rawdata_dir=kwargs["rawdir"])
            raw = fits.open(rawfile)
        else: raw=args[0]
        inputs=get_inputs(*args,**kwargs)

        return self.run_qa(raw,inputs)

    def run_qa(self,raw,inputs):
        camera=inputs["camera"]
        paname=inputs["paname"]
        qafile=inputs["qafile"]
        qafig=inputs["qafig"]
        param=inputs["param"]
        refmetrics=inputs["refmetrics"]

        rawimage=raw[camera.upper()].data
        header=raw[camera.upper()].header

        retval={}
        retval["EXPID"]= '{0:08d}'.format(header["EXPID"])
        retval["CAMERA"] = camera
        retval["PANAME"] = paname
        retval["QATIME"] = datetime.datetime.now().isoformat()
        retval["FLAVOR"] = header["FLAVOR"]
        #SE: quicklook to crash when a mismatched config file with the one in fits header
        from desispec.scripts import quicklook

        args=quicklook.parse()
        ad,fl = args.config.split("qlconfig_")
        flvr = fl.split(".yaml")[0]
        #if flvr in ['darksurvey','graysurvey','brightsurvey']: flvr = 'science'
        if header["FLAVOR"] == 'science':
           flvr = flvr.split("survey")[0]
           if (header["FLAVOR"] == flvr or header["FLAVOR"] == format(flvr.upper()) or flvr == 'test'):
                    log.info("The correct configuration file is being used!")
           else:
                    log.critical("Wrong configuration file is being used!")
                    sys.exit("Wrong configuration file! use the one for "+str(header["FLAVOR"]))

        elif (header["FLAVOR"] == flvr or flvr == 'test'):
                    log.info("The correct configuration file is being used!")
        else:
                    log.critical("Wrong configuration file is being used!")
                    sys.exit("Wrong configuration file! use the one for "+str(header["FLAVOR"]))


        if retval["FLAVOR"] == 'science':
            retval["PROGRAM"] = header["PROGRAM"]
        else:
            pass
        retval["NIGHT"] = header["NIGHT"]
        kwargs=self.config['kwargs']


        HDUstat = "NORMAL"
        EXPNUMstat = "NORMAL"

        param['EXPTIME'] = header["EXPTIME"]

        if camera != header["CAMERA"]:
                log.critical("The raw FITS file is missing camera "+camera)
                sys.exit("QuickLook Abort: CHECK THE RAW FITS FILE :"+rawfile)
                HDUstat = 'ALARM'

        if header["EXPID"] != kwargs['expid'] :
                log.critical("The raw FITS file is missing camera "+camera)
                sys.exit("QuickLook Abort: EXPOSURE NUMBER DOES NOT MATCH THE ONE IN THE HEADER")
                EXPNUMstat = "ALARM"



        if header["FLAVOR"] != "science" :

           retval["METRICS"] = {"CHECKHDUS_STATUS":HDUstat,"EXPNUM_STATUS":EXPNUMstat}

        else :
           retval["METRICS"] = {"CHECKHDUS_STATUS":HDUstat,"EXPNUM_STATUS":EXPNUMstat}
           param['SEEING'] = header["SEEING"]
           param['AIRMASS'] = header["AIRMASS"]
           param['PROGRAM'] = header["PROGRAM"]


        retval["PARAMS"] = param

        if 'INHERIT' in header and header['INHERIT']:
            h0 = raw[0].header
            for key in h0:
                if key not in header:
                    header[key] = h0[key]

        return retval

    def get_default_config(self):
        return {}


class Trace_Shifts(MonitoringAlg):
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="XYSHIFTS"
        kwargs=config['kwargs']
        parms=kwargs['param']
        key=kwargs['refKey'] if 'refKey' in kwargs else "XYSHIFTS"
        status=kwargs['statKey'] if 'statKey' in kwargs else "XYSHIFTS_STATUS"
        kwargs["RESULTKEY"]=key
        kwargs["QASTATUSKEY"]=status
        if "ReferenceMetrics" in kwargs:
            r=kwargs["ReferenceMetrics"]
            if key in r:
                kwargs["REFERENCE"]=r[key]
        if "XYSHIFTS_WARN_RANGE" in parms and "XYSHIFTS_NORMAL_RANGE" in parms:
            kwargs["RANGES"]=[(np.asarray(parms["XYSHIFTS_WARN_RANGE"]),QASeverity.WARNING),
                              (np.asarray(parms["XYSHIFTS_NORMAL_RANGE"]),QASeverity.NORMAL)]
        MonitoringAlg.__init__(self,name,im,config,logger)
    def run(self,*args,**kwargs):
        if len(args) == 0 :
            log.critical("No parameter is found for this QA")
            sys.exit("Update the configuration file for the parameters")

        if not self.is_compatible(type(args[0])):
            #raise qlexceptions.ParameterException("Incompatible input. Was expecting {} got {}".format(type(self.__inpType__),type(args[0])))
            log.critical("Incompatible input!")
            sys.exit("Was expecting {} got {}".format(type(self.__inpType__),type(args[0])))

        if kwargs["singleqa"] == 'Trace_Shifts':
            night = kwargs['night']
            expid = '{:08d}'.format(kwargs['expid'])
            camera = kwargs['camera']
            image = get_image('preproc',night,expid,camera,kwargs["specdir"])
        else: image=args[0]
        inputs=get_inputs(*args,**kwargs)

        return self.run_qa(image,inputs)

    def run_qa(self,image,inputs):
        camera=inputs["camera"]
        paname=inputs["paname"]
        qafile=inputs["qafile"]
        qafig=inputs["qafig"]
        param=inputs["param"]
        refmetrics=inputs["refmetrics"]

        #- qa dictionary
        retval={}
        retval["PANAME" ]= paname
        retval["QATIME"] = datetime.datetime.now().isoformat()
        retval["EXPID"] = expid = '{0:08d}'.format(image.meta["EXPID"])
        retval["CAMERA"] = camera
        retval["FLAVOR"] = image.meta["FLAVOR"]
        kwargs=self.config['kwargs']

        if image.meta["FLAVOR"] == 'science':
            fibmap =fits.open(kwargs['FiberMap'])
            retval["PROGRAM"]=fibmap[1].header['PROGRAM']


        retval["NIGHT"] = night = image.meta["NIGHT"]


        if param is None:
                log.critical("No parameter is found for this QA")
                sys.exit("Update the configuration file for the parameters")

        # create xytraceset object

        from desispec.calibfinder import findcalibfile
        from desispec.xytraceset import XYTraceSet
        #SE: all next lines till the dashed line exist just so that we get the psf name without hardcoding any address -> there must be a better way
        rawfile = findfile('raw',int(night),int(expid),camera,rawdata_dir=os.environ["QL_SPEC_DATA"])
        hdulist=fits.open(rawfile)
        primary_header=hdulist[0].header
        camera_header =hdulist[camera].header
        hdulist.close()
        #--------------------------------------------------------
        psffile=findcalibfile([camera_header,primary_header],"PSF")
        psf=fits.open(psffile)
        xcoef=psf['XTRACE'].data
        ycoef=psf['YTRACE'].data
        wavemin=psf["XTRACE"].header["WAVEMIN"]
        wavemax=psf["XTRACE"].header["WAVEMAX"]
        npix_y=image.meta['NAXIS2']
        psftrace=XYTraceSet(xcoef,ycoef,wavemin,wavemax,npix_y=npix_y)

        # compute dx and dy
        from desispec.trace_shifts import compute_dx_from_cross_dispersion_profiles as compute_dx
        from desispec.trace_shifts import compute_dy_using_boxcar_extraction as compute_dy
        fibers=np.arange(500) #RS: setting nfibers to 500 for now
        ox,oy,odx,oex,of,ol=compute_dx(xcoef,ycoef,wavemin,wavemax,image,fibers=fibers)
        x_for_dy,y_for_dy,ody,ey,fiber_for_dy,wave_for_dy=compute_dy(psftrace,image,fibers)

        # return average shifts in x and y
        dx=np.mean(odx)
        dy=np.mean(ody)
        xyshift=np.array([dx,dy])

        retval["METRICS"]={"XYSHIFTS":xyshift}
        retval["PARAMS"]=param

        #get_outputs(qafile,qafig,retval,'plot_traceshifts')
#        outfile = qa.write_qa_ql(qafile,retval)
#        log.debug("Output QA data is in {}".format(outfile))
        return retval

    def get_default_config(self):
        return {}


class Bias_From_Overscan(MonitoringAlg):
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="BIAS_OVERSCAN"

        kwargs=config['kwargs']
        parms=kwargs['param']
        key=kwargs['refKey'] if 'refKey' in kwargs else "BIAS_AMP"
        status=kwargs['statKey'] if 'statKey' in kwargs else "BIAS_AMP_STATUS"
        kwargs["RESULTKEY"]=key
        kwargs["QASTATUSKEY"]=status

        if "ReferenceMetrics" in kwargs:
            r=kwargs["ReferenceMetrics"]
            if key in r:
                kwargs["REFERENCE"]=r[key]

        if "BIAS_WARN_RANGE" in parms and "BIAS_NORMAL_RANGE" in parms:
            kwargs["RANGES"]=[(np.asarray(parms["BIAS_WARN_RANGE"]),QASeverity.WARNING),
                              (np.asarray(parms["BIAS_NORMAL_RANGE"]),QASeverity.NORMAL)]# sorted by most severe to least severe

        MonitoringAlg.__init__(self,name,im,config,logger)
    def run(self,*args,**kwargs):
        if len(args) == 0 :

            log.critical("No parameter is found for this QA")
            sys.exit("Update the configuration file for the parameters")

        if not self.is_compatible(type(args[0])):
            #raise qlexceptions.ParameterException("Incompatible input. Was expecting {} got {}".format(type(self.__inpType__),type(args[0])))
            log.critical("Incompatible input!")
            sys.exit("Was expecting {} got {}".format(type(self.__inpType__),type(args[0])))

        if kwargs["singleqa"] == 'Bias_From_Overscan':
            night = kwargs['night']
            expid = '{:08d}'.format(kwargs['expid'])
            camera = kwargs['camera']
            image = get_image('preproc',night,expid,camera,kwargs["specdir"])
        else: image=args[0]
        inputs=get_inputs(*args,**kwargs)

        return self.run_qa(image,inputs)

    def run_qa(self,image,inputs):
        camera=inputs["camera"]
        paname=inputs["paname"]
        amps=inputs["amps"]
        qafile=inputs["qafile"]
        qafig=inputs["qafig"]
        param=inputs["param"]
        refmetrics=inputs["refmetrics"]
        plotconf=inputs["plotconf"]
        hardplots=inputs["hardplots"]

        retval={}
        retval["EXPID"] = '{0:08d}'.format(image.meta["EXPID"])
        retval["PANAME"] = paname
        retval["QATIME"] = datetime.datetime.now().isoformat()
        retval["CAMERA"] = camera
        retval["NIGHT"] = image.meta["NIGHT"]
        retval["FLAVOR"] = flavor = image.meta["FLAVOR"]
        kwargs=self.config['kwargs']

        if image.meta["FLAVOR"] == 'science':
            fibmap =fits.open(kwargs['FiberMap'])
            retval["PROGRAM"]=fibmap[1].header['PROGRAM']

        retval["EXPTIME"] = image.meta["EXPTIME"]


        if retval["FLAVOR"] == 'arc':
            pass
        else:
            retval["FLAVOR"] = image.meta["FLAVOR"]
        retval["NIGHT"] = image.meta["NIGHT"]
        kwargs=self.config['kwargs']

        #SE: this would give the desispec version stored in DEPVER07 key of the raw simulated fits file :0.16.0.dev1830
        #RS: don't look for this if not using simulated files, differences in simulated headers vs. data headers cause this to crash
        if flavor == 'science':
            param['FITS_DESISPEC_VERSION'] = image.meta['DEPVER07']
            import desispec
            from desispec import quicklook
            param['PROC_DESISPEC_VERSION']= desispec.__version__
            param['PROC_QuickLook_VERSION']= quicklook.__qlversion__


        if 'INHERIT' in image.meta and image.meta['INHERIT']:

            h0 = image.meta
            #h0 = header
            for key in h0:
                if key not in image.meta:
                    image.meta[key] = h0[key]

        #RS: look for values in simulated data, if not found try finding data values
        try:
            bias_overscan = [image.meta['OVERSCN1'],image.meta['OVERSCN2'],image.meta['OVERSCN3'],image.meta['OVERSCN4']]
        except:
            bias_overscan = [image.meta['OVERSCNA'],image.meta['OVERSCNB'],image.meta['OVERSCNC'],image.meta['OVERSCND']]

        bias = np.mean(bias_overscan)

        if param is None:
            log.critical("No parameter is found for this QA")
            sys.exit("Update the configuration file for the parameters")


        retval["PARAMS"] = param

        if amps:
            bias_amps=np.array(bias_overscan)
            retval["METRICS"]={'BIAS_AMP':bias_amps}
        else:
            #retval["METRICS"]={'BIAS':bias,"DIFF1SIG":diff1sig,"DIFF2SIG":diff2sig,"DIFF3SIG":diff3sig,"DATA5SIG":data5sig,"BIAS_ROW":mean_row}
            retval["METRICS"]={}

        ###############################################################
        # This section is for adding QA metrics for plotting purposes #
        ###############################################################

        ###############################################################

#        if qafile is not None:
#            outfile=qa.write_qa_ql(qafile,retval)
#            log.debug("Output QA data is in {}".format(outfile))
        if qafig is not None:
            fig.plot_bias_overscan(retval,qafig,plotconf=plotconf,hardplots=hardplots)
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
        status=kwargs['statKey'] if 'statKey' in kwargs else "NOISE_AMP_STATUS"
        kwargs["RESULTKEY"]=key
        kwargs["QASTATUSKEY"]=status

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
            log.critical("No parameter is found for this QA")
            sys.exit("Update the configuration file for the parameters")

        if not self.is_compatible(type(args[0])):
            #raise qlexceptions.ParameterException("Incompatible input. Was expecting {} got {}".format(type(self.__inpType__),type(args[0])))
            log.critical("Incompatible input!")
            sys.exit("Was expecting {} got {}".format(type(self.__inpType__),type(args[0])))

        if kwargs["singleqa"] == 'Get_RMS':
            night = kwargs['night']
            expid = '{:08d}'.format(kwargs['expid'])
            camera = kwargs['camera']
            image = get_image('preproc',night,expid,camera,kwargs["specdir"])
        else: image=args[0]
        inputs=get_inputs(*args,**kwargs)

        return self.run_qa(image,inputs)

    def run_qa(self,image,inputs):
        camera=inputs["camera"]
        paname=inputs["paname"]
        amps=inputs["amps"]
        qafile=inputs["qafile"]
        qafig=inputs["qafig"]
        param=inputs["param"]
        refmetrics=inputs["refmetrics"]
        plotconf=inputs["plotconf"]
        hardplots=inputs["hardplots"]

        retval={}
        retval["EXPID"] = '{0:08d}'.format(image.meta["EXPID"])
        retval["PANAME"] = paname
        retval["QATIME"] = datetime.datetime.now().isoformat()
        retval["CAMERA"] = camera
        retval["FLAVOR"] = flavor = image.meta["FLAVOR"]
        kwargs=self.config['kwargs']

        if flavor == 'science':
            fibmap =fits.open(kwargs['FiberMap'])
            retval["PROGRAM"]=fibmap[1].header['PROGRAM']

        retval["NIGHT"] = image.meta["NIGHT"]

        # return rms values in rms/sqrt(exptime)
        #rmsccd=qalib.getrms(image.pix/np.sqrt(image.meta["EXPTIME"])) #- should we add dark current and/or readnoise to this as well?
        #rmsccd = np.mean([image.meta['RDNOISE1'],image.meta['RDNOISE2'],image.meta['RDNOISE3'],image.meta['RDNOISE4']]) #--> "NOISE":rmsccd

        if param is None:
            log.critical("No parameter is given for this QA! ")
            sys.exit("Check the configuration file")

        retval["PARAMS"] = param

        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # SE: this section is moved from BIAS_FROM_OVERSCAN to header

        data=[]
        row_data_amp1=[]
        row_data_amp2=[]
        row_data_amp3=[]
        row_data_amp4=[]
        bias_patnoise=[]
        #bias_overscan=[]
        #RS: loop through amps based on header info
        loop_amps = get_amp_ids(image.meta)
        exptime=image.meta["EXPTIME"]
        if exptime == 0.:
            exptime = 1.
        for kk in loop_amps:
            sel=parse_sec_keyword(image.meta['BIASSEC'+kk])
            #- Obtain counts/second in bias region
#            pixdata=image[sel]/header["EXPTIME"]
            pixdata=image.pix[sel]/exptime
            if kk == '1' or kk == 'A':
                for i in range(pixdata.shape[0]):
                    row_amp1=pixdata[i]
                    row_data_amp1.append(row_amp1)
            if kk == '2' or kk == 'B':

                for i in range(pixdata.shape[0]):
                    row_amp2=pixdata[i]
                    row_data_amp2.append(row_amp2)
            if kk == '3' or kk == 'C':

                for i in range(pixdata.shape[0]):
                    row_amp3=pixdata[i]
                    row_data_amp3.append(row_amp3)
            if kk == '4' or kk == 'D':

                for i in range(pixdata.shape[0]):
                    row_amp4=pixdata[i]
                    row_data_amp4.append(row_amp4)
            #- Compute statistics of the bias region that only reject
            #  the 0.5% of smallest and largest values. (from sdssproc)
            isort=np.sort(pixdata.ravel())
            nn=isort.shape[0]
            bias=np.mean(isort[int(0.005*nn) : int(0.995*nn)])
            #bias_overscan.append(bias)
            data.append(isort)

        #- Combine data from each row per amp and take average
        # BIAS_ROW = mean_row
        median_row_amp1=[]
        for i in range(len(row_data_amp1)):
            median=np.median(row_data_amp1[i])
            median_row_amp1.append(median)

        rms_median_row_amp1= np.std(median_row_amp1)
        try:
            noise1 = image.meta['RDNOISE1']
        except:
            noise1 = image.meta['OBSRDNA']
        bias_patnoise.append(rms_median_row_amp1/noise1)

        median_row_amp2=[]
        for i in range(len(row_data_amp2)):
            median=np.median(row_data_amp2[i])
            median_row_amp2.append(median)

        rms_median_row_amp2= np.std(median_row_amp2)
        try:
            noise2 = image.meta['RDNOISE2']
        except:
            noise2 = image.meta['OBSRDNB']
        bias_patnoise.append(rms_median_row_amp2/noise2)


        median_row_amp3=[]
        for i in range(len(row_data_amp3)):
            median=np.median(row_data_amp3[i])
            median_row_amp3.append(median)

        rms_median_row_amp3= np.std(median_row_amp3)
        try:
            noise3 = image.meta['RDNOISE3']
        except:
            noise3 = image.meta['OBSRDNC']
        bias_patnoise.append(rms_median_row_amp3/noise3)

        median_row_amp4=[]
        for i in range(len(row_data_amp4)):
            median=np.median(row_data_amp4[i])
            median_row_amp4.append(median)

        rms_median_row_amp4= np.std(median_row_amp4)
        try:
            noise4 = image.meta['RDNOISE4']
        except:
            noise4 = image.meta['OBSRDND']
        bias_patnoise.append(rms_median_row_amp4/noise4)


        #- Calculate upper and lower bounds of 1, 2, and 3 sigma
        full_data=np.concatenate((data[0],data[1],data[2],data[3])).ravel()
        sig1_lo = np.percentile(full_data,50.-(param['PERCENTILES'][0]/2.))
        sig1_hi = np.percentile(full_data,50.+(param['PERCENTILES'][0]/2.))
        sig2_lo = np.percentile(full_data,50.-(param['PERCENTILES'][1]/2.))
        sig2_hi = np.percentile(full_data,50.+(param['PERCENTILES'][1]/2.))
        sig3_lo = np.percentile(full_data,50.-(param['PERCENTILES'][2]/2.))
        sig3_hi = np.percentile(full_data,50.+(param['PERCENTILES'][2]/2.))

        #- Find difference between upper and lower sigma bounds
        # DIFF1SIG: The number of counts separating the 1 sigma percentiles in the noise distribution (from the overscan region)
        diff1sig = sig1_hi - sig1_lo
        # DIFF2SIG: The number of counts separating 2 or 3 sigma in the noise distribution
        diff2sig = sig2_hi - sig2_lo
        diff3sig = sig3_hi - sig3_lo

        #-DATA5SIG: number of pixels more than 5 sigma below the bias level
        sig5_value = np.percentile(full_data,3e-5)
        data5sig = len(np.where(full_data <= sig5_value)[0])

        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        if amps:
            rms_over_amps = [noise1,noise2,noise3,noise4]
            try:
                rms_amps = [image.meta['OBSRDN1'],image.meta['OBSRDN2'],image.meta['OBSRDN3'],image.meta['OBSRDN4']]
            except:
                rms_amps = [image.meta['OBSRDNA'],image.meta['OBSRDNB'],image.meta['OBSRDNC'],image.meta['OBSRDND']]
            retval["METRICS"]={"NOISE_AMP":np.array(rms_amps),"NOISE_OVERSCAN_AMP":np.array(rms_over_amps),"DIFF1SIG":diff1sig,"DIFF2SIG":diff2sig,"DATA5SIG":data5sig,"BIAS_PATNOISE":bias_patnoise}#,"NOISE_ROW":noise_row,"EXPNUM_WARN":expnum,"NOISE_OVER":rmsover
        else:
            retval["METRICS"]={"DIFF1SIG":diff1sig,"DIFF2SIG":diff2sig,"DATA5SIG":data5sig, "BIAS_PATNOISE":bias_patnoise} # Dropping "NOISE_OVER":rmsover,"NOISE_ROW":noise_row,"EXPNUM_WARN":expnum


        ###############################################################
        # This section is for adding QA metrics for plotting purposes #
        ###############################################################

        ###############################################################

#        if qafile is not None:
#            outfile=qa.write_qa_ql(qafile,retval)
#            log.debug("Output QA data is in {}".format(outfile))
        if qafig is not None:
            fig.plot_RMS(retval,qafig,plotconf=plotconf,hardplots=hardplots)
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
        key=kwargs['refKey'] if 'refKey' in kwargs else "XWSIGMA"
        status=kwargs['statKey'] if 'statKey' in kwargs else "XWSIGMA_STATUS"
        kwargs["RESULTKEY"]=key
        kwargs["QASTATUSKEY"]=status
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
            log.critical("No parameter is found for this QA")
            sys.exit("Update the configuration file for the parameters")

        if not self.is_compatible(type(args[0])):
            #raise qlexceptions.ParameterException("Incompatible input. Was expecting {} got {}".format(type(self.__inpType__),type(args[0])))
            log.critical("Incompatible input!")
            sys.exit("Was expecting {} got {}".format(type(self.__inpType__),type(args[0])))

        if kwargs["singleqa"] == 'Calc_XWSigma':
            night = kwargs['night']
            expid = '{:08d}'.format(kwargs['expid'])
            camera = kwargs['camera']
            image = get_image('preproc',night,expid,camera,kwargs["specdir"])
        else: image=args[0]
        inputs=get_inputs(*args,**kwargs)

        return self.run_qa(image,inputs)

    def run_qa(self,image,inputs):
        import desispec.quicklook.qlpsf
        from scipy.optimize import curve_fit
        camera=inputs["camera"]
        paname=inputs["paname"]
        fibermap=inputs["fibermap"]
        psffile=inputs["psf"]
        psf=desispec.quicklook.qlpsf.PSF(psffile)
        amps=inputs["amps"]
        allpeaks=inputs["Peaks"]
        qafile=inputs["qafile"]
        qafig=inputs["qafig"]
        param=inputs["param"]
        refmetrics=inputs["refmetrics"]
        plotconf=inputs["plotconf"]
        hardplots=inputs["hardplots"]

        retval={}
        retval["PANAME"] = paname
        retval["QATIME"] = datetime.datetime.now().isoformat()
        retval["EXPID"] = '{0:08d}'.format(image.meta["EXPID"])
        retval["CAMERA"] = camera
        retval["FLAVOR"] = image.meta["FLAVOR"]
        kwargs=self.config['kwargs']

        if image.meta["FLAVOR"] == 'science':
            fibmap =fits.open(kwargs['FiberMap'])
            retval["PROGRAM"]=program=fibmap[1].header['PROGRAM']

        retval["NIGHT"] = image.meta["NIGHT"]

        if param is None:
            log.critical("No parameter is given for this QA! ")
            sys.exit("Check the configuration file")

        retval["PARAMS"] = param
        #- Ensure that the QA will run even if 500 spectra aren't present
        if fibermap['FIBER'].shape[0] >= 500:
            fibers = 500
        else:
            fibers = fibermap['FIBER'].shape[0]

        #- Define number of pixels to be fit
        dp=param['PIXEL_RANGE']/2
        #- Get wavelength ranges around peaks
        peaks=allpeaks['{}_PEAKS'.format(camera[0].upper())]
        #- Maximum allowed fit sigma value
        maxsigma=param['MAX_SIGMA']

        xfails=[]
        wfails=[]
        xsigma=[]
        wsigma=[]
        xsigma_amp1=[]
        wsigma_amp1=[]
        xsigma_amp2=[]
        wsigma_amp2=[]
        xsigma_amp3=[]
        wsigma_amp3=[]
        xsigma_amp4=[]
        wsigma_amp4=[]

        for fiber in range(fibers):

            xs = -1 # SE: this prevents crash in "XWSIGMA_AMP" for when xs or ws is empty list -> try b9 of 20200515/00000001
            ws = -1
            xsig=[]
            wsig=[]
            for peak in range(len(peaks)):
                #- Use psf information to convert wavelength to pixel values
                xpixel=desispec.quicklook.qlpsf.PSF.x(psf,ispec=fiber,wavelength=peaks[peak])[0][0]
                ypixel=desispec.quicklook.qlpsf.PSF.y(psf,ispec=fiber,wavelength=peaks[peak])[0][0]
                #- Find x and y pixel values around sky lines
                xpix_peak=np.arange(int(xpixel-dp),int(xpixel+dp),1)
                ypix_peak=np.arange(int(ypixel-dp),int(ypixel+dp),1)
                #- Fit gaussian to counts in pixels around sky line
                #- If any values fail, store x/w, wavelength, and fiber
                try:
                    xpopt,xpcov=curve_fit(qalib.gauss,np.arange(len(xpix_peak)),image.pix[int(ypixel),xpix_peak])
                    xs=np.abs(xpopt[2])
                    if xs <= maxsigma:
                        xsig.append(xs)
                    else:
                        xfail=[fiber,peaks[peak]]
                        xfails.append(xfail)
                except:
                    xfail=[fiber,peaks[peak]]
                    xfails.append(xfail)
                    pass
                try:
                    wpopt,wpcov=curve_fit(qalib.gauss,np.arange(len(ypix_peak)),image.pix[ypix_peak,int(xpixel)])
                    ws=np.abs(wpopt[2])
                    if ws <= maxsigma:
                        wsig.append(ws)
                    else:
                        wfail=[fiber,peaks[peak]]
                        wfails.append(wfail)
                except:
                    wfail=[fiber,peaks[peak]]
                    wfails.append(wfail)
                    pass

                #- Excluding fibers 240-260 in case some fibers overlap amps
                #- Excluding peaks in the center of image in case peak overlaps two amps
                #- This shouldn't cause a significant loss of information

                if amps:

                    if fibermap['FIBER'][fiber]<240:
                        if ypixel < 2000.:
                            xsigma_amp1.append(xs)
                            wsigma_amp1.append(ws)
                        if ypixel > 2100.:
                            xsigma_amp3.append(xs)
                            wsigma_amp3.append(ws)

                    if fibermap['FIBER'][fiber]>260:
                        if ypixel < 2000.:
                            xsigma_amp2.append(xs)
                            wsigma_amp2.append(ws)
                        if ypixel > 2100.:
                            xsigma_amp4.append(xs)
                            wsigma_amp4.append(ws)


            if len(xsig)!=0:
                xsigma.append(np.mean(xsig))
            if len(wsig)!=0:
                wsigma.append(np.mean(wsig))

        if fibermap['FIBER'].shape[0]<260:
            xsigma_amp2=[]
            xsigma_amp4=[]
            wsigma_amp2=[]
            wsigma_amp4=[]

        #- Calculate desired output metrics
        xsigma_med=np.median(np.array(xsigma))
        wsigma_med=np.median(np.array(wsigma))
        xsigma_amp=np.array([np.median(xsigma_amp1),np.median(xsigma_amp2),np.median(xsigma_amp3),np.median(xsigma_amp4)])
        wsigma_amp=np.array([np.median(wsigma_amp1),np.median(wsigma_amp2),np.median(wsigma_amp3),np.median(wsigma_amp4)])
        xwfails=np.array([xfails,wfails])


        #SE: mention the example here when the next lines are ineffective and when they are effective in removing the NaN from XWSIGMA_AMP--> XWSIGMA itself no longer includes any NaN value. As we both know, this is not the way to properly deal with NaNs -->let's see if switching to non-scipy fuction would bring about a better solution
        if len(xsigma)==0:
            xsigma=[param['XWSIGMA_{}_REF'.format(program.upper())][0]]

        if len(wsigma)==0:
            wsigma=[param['XWSIGMA_{}_REF'.format(program.upper())][1]]

        #- Combine metrics for x and w
        xwsigma_fib=np.array((xsigma,wsigma)) #- (2,nfib)
        xwsigma_med=np.array((xsigma_med,wsigma_med)) #- (2)
        xwsigma_amp=np.array((xsigma_amp,wsigma_amp))

        if amps:
            #if len(xsigma_amp1)==0 :
                #xsigma_amp1 = [param['XWSIGMA_REF'][0]]
            #if len(xsigma_amp2)==0 :
                #xsigma_amp2 = [param['XWSIGMA_REF'][0]]
            #if len(xsigma_amp3)==0 :
                #xsigma_amp3 = [param['XWSIGMA_REF'][0]]
            #if len(xsigma_amp4)==0 :
                #xsigma_amp4 = [param['XWSIGMA_REF'][0]]

            #if len(wsigma_amp1)==0 :
                #wsigma_amp1 = [param['XWSIGMA_REF'][1]]
            #if len(wsigma_amp2)==0 :
                #wsigma_amp2 = [param['XWSIGMA_REF'][1]]
            #if len(wsigma_amp3)==0 :
                #wsigma_amp3 = [param['XWSIGMA_REF'][1]]
            #if len(wsigma_amp4)==0 :
                #wsigma_amp4 = [param['XWSIGMA_REF'][1]]

            retval["METRICS"]={"XWSIGMA":xwsigma_med,"XWSIGMA_FIB":xwsigma_fib,"XWSIGMA_AMP":xwsigma_amp}#,"XWSHIFT":xwshift,"XWSHIFT_AMP":xwshift_amp,"XWSIGMA_SHIFT": xwsigma_shift}
        else:
            retval["METRICS"]={"XWSIGMA":xwsigma_med,"XWSIGMA_FIB":xwsigma_fib}#,"XWSHIFT":xwshift,"XWSIGMA_SHIFT": xwsigma_shift}

        ###############################################################
        # This section is for adding QA metrics for plotting purposes #
        ###############################################################

        ###############################################################

#        if qafile is not None:
#            outfile=qa.write_qa_ql(qafile,retval)
#            log.debug("Output QA data is in {}".format(outfile))
        if qafig is not None:
            fig.plot_XWSigma(retval,qafig,plotconf=plotconf,hardplots=hardplots)
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
        key=kwargs['refKey'] if 'refKey' in kwargs else "LITFRAC_AMP"
        status=kwargs['statKey'] if 'statKey' in kwargs else "LITFRAC_AMP_STATUS"
        kwargs["RESULTKEY"]=key
        kwargs["QASTATUSKEY"]=status
        if "ReferenceMetrics" in kwargs:
            r=kwargs["ReferenceMetrics"]
            if key in r:
                kwargs["REFERENCE"]=r[key]

        if "LITFRAC_AMP_WARN_RANGE" in parms and "LITFRAC_AMP_NORMAL_RANGE" in parms:
            kwargs["RANGES"]=[(np.asarray(parms["LITFRAC_AMP_WARN_RANGE"]),QASeverity.WARNING),
                              (np.asarray(parms["LITFRAC_AMP_NORMAL_RANGE"]),QASeverity.NORMAL)]# sorted by most severe to least severe
        MonitoringAlg.__init__(self,name,im,config,logger)
    def run(self,*args,**kwargs):
        if len(args) == 0 :
            log.critical("No parameter is found for this QA")
            sys.exit("Update the configuration file for the parameters")

        if not self.is_compatible(type(args[0])):
            #raise qlexceptions.ParameterException("Incompatible input. Was expecting {} got {}".format(type(self.__inpType__),type(args[0])))
            log.critical("Incompatible input!")
            sys.exit("Was expecting {} got {}".format(type(self.__inpType__),type(args[0])))

        if kwargs["singleqa"] == 'Count_Pixels':
            night = kwargs['night']
            expid = '{:08d}'.format(kwargs['expid'])
            camera = kwargs['camera']
            image = get_image('preproc',night,expid,camera,kwargs["specdir"])
        else: image=args[0]
        inputs=get_inputs(*args,**kwargs)

        return self.run_qa(image,inputs)

    def run_qa(self,image,inputs):
        camera=inputs["camera"]
        paname=inputs["paname"]
        amps=inputs["amps"]
        qafile=inputs["qafile"]
        qafig=inputs["qafig"]
        param=inputs["param"]
        refmetrics=inputs["refmetrics"]
        plotconf=inputs["plotconf"]
        hardplots=inputs["hardplots"]

        retval={}
        retval["PANAME"] = paname
        retval["QATIME"] = datetime.datetime.now().isoformat()
        retval["EXPID"] = '{0:08d}'.format(image.meta["EXPID"])
        retval["CAMERA"] = camera
        retval["FLAVOR"] = image.meta["FLAVOR"]
        kwargs=self.config['kwargs']

        if image.meta["FLAVOR"] == 'science':
            fibmap =fits.open(kwargs['FiberMap'])
            retval["PROGRAM"]=fibmap[1].header['PROGRAM']


        retval["NIGHT"] = image.meta["NIGHT"]


        if param is None:
            log.critical("No parameter is given for this QA! ")
            sys.exit("Check the configuration file")


        retval["PARAMS"] = param

        #- get the counts for each amp
        npix_amps=[]
        litfrac_amps=[]

        from desispec.preproc import parse_sec_keyword
        #RS: loop through amps based on header info
        try:
            header_test=parse_sec_keyword(image.meta['CCDSEC1'])
            loop_amps=['1','2','3','4']
        except:
            loop_amps=['A','B','C','D']
        #- get amp boundary in pixels
        for kk in loop_amps:
            ampboundary=parse_sec_keyword(image.meta["CCDSEC"+kk])
            try:
                rdnoise_thisamp=image.meta["RDNOISE"+kk]
            except:
                rdnoise_thisamp=image.meta["OBSRDN"+kk]
            npix_thisamp= image.pix[ampboundary][image.pix[ampboundary] > param['CUTPIX'] * rdnoise_thisamp].size #- no of pixels above threshold
            npix_amps.append(npix_thisamp)
            size_thisamp=image.pix[ampboundary].size
            litfrac_thisamp=round(np.float64(npix_thisamp)/size_thisamp,2) #- fraction of pixels getting light above threshold
            litfrac_amps.append(litfrac_thisamp)
	#        retval["METRICS"]={"NPIX_AMP",npix_amps,'LITFRAC_AMP': litfrac_amps}
        retval["METRICS"]={"LITFRAC_AMP": litfrac_amps}

        ###############################################################
        # This section is for adding QA metrics for plotting purposes #
        ###############################################################

        ###############################################################

#        if qafile is not None:
#            outfile=qa.write_qa_ql(qafile,retval)
#            log.debug("Output QA data is in {}".format(outfile))
        if qafig is not None:
            fig.plot_countpix(retval,qafig,plotconf=plotconf,hardplots=hardplots)
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
        status=kwargs['statKey'] if 'statKey' in kwargs else "NGOODFIB_STATUS"
        kwargs["RESULTKEY"]=key
        kwargs["QASTATUSKEY"]=status

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
            log.critical("No parameter is found for this QA")
            sys.exit("Update the configuration file for the parameters")

        if not self.is_compatible(type(args[0])):
            #raise qlexceptions.ParameterException("Incompatible input. Was expecting {} got {}".format(type(self.__inpType__),type(args[0])))
            log.critical("Incompatible input!")
            sys.exit("Was expecting {} got {}".format(type(self.__inpType__),type(args[0])))

        if kwargs["singleqa"] == 'CountSpectralBins':
            night = kwargs['night']
            expid = '{:08d}'.format(kwargs['expid'])
            camera = kwargs['camera']
            frame = get_frame('frame',night,expid,camera,kwargs["specdir"])
        else: frame=args[0]
        inputs=get_inputs(*args,**kwargs)

        return self.run_qa(frame,inputs)

    def run_qa(self,frame,inputs):
        camera=inputs["camera"]
        paname=inputs["paname"]
        fibermap=inputs["fibermap"]
        amps=inputs["amps"]
        psf=inputs["psf"]
        qafile=inputs["qafile"]
        qafig=None #inputs["qafig"]
        param=inputs["param"]
        refmetrics=inputs["refmetrics"]
        plotconf=inputs["plotconf"]
        hardplots=inputs["hardplots"]

        if isinstance(frame,QFrame):
            frame = frame.asframe()

        #- qa dictionary
        retval={}
        retval["PANAME"] = paname
        retval["QATIME"] = datetime.datetime.now().isoformat()
        retval["EXPID"] = '{0:08d}'.format(frame.meta["EXPID"])
        retval["CAMERA"] = camera
        retval["FLAVOR"] = frame.meta["FLAVOR"]
        kwargs=self.config['kwargs']

        if frame.meta["FLAVOR"] == 'science':
            fibmap =fits.open(kwargs['FiberMap'])
            retval["PROGRAM"]=fibmap[1].header['PROGRAM']

        retval["NIGHT"] = frame.meta["NIGHT"]

        grid=np.gradient(frame.wave)
        if not np.all(grid[0]==grid[1:]):
            log.debug("grid_size is NOT UNIFORM")

        if param is None:
            log.critical("No parameter is given for this QA! ")
            sys.exit("Check the configuration file")


        retval["PARAMS"] = param
        #- get the effective readnoise for the fibers
        #- readnoise per fib = readnoise per pix * sqrt(box car width)* sqrt(no. of bins in the amp) * binsize/pix size scale
        nspec=frame.nspec
        rdnoise_fib=np.zeros(nspec)
        if nspec > 250: #- upto 250 - amp 1 and 3, beyond that 2 and 4
            rdnoise_fib[:250]=[(frame.meta['RDNOISE1']+frame.meta['RDNOISE3'])*np.sqrt(5.)*np.sqrt(frame.flux.shape[1]/2)*frame.meta['WAVESTEP']/0.5]*250
            rdnoise_fib[250:]=[(frame.meta['RDNOISE2']+frame.meta['RDNOISE4'])*np.sqrt(5.)*np.sqrt(frame.flux.shape[1]/2)*frame.meta['WAVESTEP']/0.5]*(nspec-250)
        else:
            rdnoise_fib=[(frame.meta['RDNOISE1']+frame.meta['RDNOISE3'])*np.sqrt(5.)*np.sqrt(frame.flux.shape[1]/2)*frame.meta['WAVESTEP']/0.5]*nspec
        threshold=[param['CUTBINS']*ii for ii in rdnoise_fib]
        #- compare the flux sum to threshold

        totcounts=frame.flux.sum(axis=1)
        passfibers=np.where(totcounts>threshold)[0]
        ngoodfibers=passfibers.shape[0]
        good_fibers=np.array([0]*frame.nspec)
        good_fibers[passfibers]=1 #- assign 1 for good fiber

        #- leaving the amps granularity needed for caching as defunct. If needed in future, this needs to be propagated through.
        amps=False
        leftmax=None
        rightmax=None
        bottommax=None
        topmin=None

        if amps: #- leaving this for now
            leftmax,rightmin,bottommax,topmin = qalib.fiducialregion(frame,psf)
            retval["LEFT_MAX_FIBER"]=int(leftmax)
            retval["RIGHT_MIN_FIBER"]=int(rightmin)
            retval["BOTTOM_MAX_WAVE_INDEX"]=int(bottommax)
            retval["TOP_MIN_WAVE_INDEX"]=int(topmin)

        retval["METRICS"]={"NGOODFIB": ngoodfibers, "GOOD_FIBERS": good_fibers, "TOTCOUNT_FIB": totcounts}

        ###############################################################
        # This section is for adding QA metrics for plotting purposes #
        ###############################################################

        ###############################################################

#        if qafile is not None:
#            outfile=qa.write_qa_ql(qafile,retval)
#            log.debug("Output QA data is in {}".format(outfile))
        if qafig is not None:
            fig.plot_countspectralbins(retval,qafig,plotconf=plotconf,hardplots=hardplots)
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
        status=kwargs['statKey'] if 'statKey' in kwargs else "SKYCONT_STATUS"
        kwargs["RESULTKEY"]=key
        kwargs["QASTATUSKEY"]=status
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
            log.critical("No parameter is found for this QA")
            sys.exit("Update the configuration file for the parameters")

        if not self.is_compatible(type(args[0])):
            #raise qlexceptions.ParameterException("Incompatible input. Was expecting {} got {}".format(type(self.__inpType__),type(args[0])))
            log.critical("Incompatible input!")
            sys.exit("Was expecting {} got {}".format(type(self.__inpType__),type(args[0])))

        if kwargs["singleqa"] == 'Sky_Continuum':
            night = kwargs['night']
            expid = '{:08d}'.format(kwargs['expid'])
            camera = kwargs['camera']
            frame = get_frame('fframe',night,expid,camera,kwargs["specdir"])
        else: frame=args[0]
        inputs=get_inputs(*args,**kwargs)

        return self.run_qa(frame,inputs)

    def run_qa(self,frame,inputs):
        camera=inputs["camera"]
        paname=inputs["paname"]
        fibermap=inputs["fibermap"]
        amps=inputs["amps"]
        qafile=inputs["qafile"]
        qafig=inputs["qafig"]
        param=inputs["param"]
        refmetrics=inputs["refmetrics"]
        plotconf=inputs["plotconf"]
        hardplots=inputs["hardplots"]

        if isinstance(frame,QFrame):
            frame = frame.asframe()

        #- qa dictionary
        retval={}
        retval["PANAME" ]= paname
        retval["QATIME"] = datetime.datetime.now().isoformat()
        retval["EXPID"] = '{0:08d}'.format(frame.meta["EXPID"])
        retval["CAMERA"] = camera
        retval["FLAVOR"] = frame.meta["FLAVOR"]
        kwargs=self.config['kwargs']

        if frame.meta["FLAVOR"] == 'science':
            fibmap =fits.open(kwargs['FiberMap'])
            retval["PROGRAM"]=fibmap[1].header['PROGRAM']

        retval["NIGHT"] = frame.meta["NIGHT"]

        camera=frame.meta["CAMERA"]

        if param is None:
            log.critical("No parameter is given for this QA! ")
            sys.exit("Check the configuration file")


        wrange1=param["{}_CONT".format(camera[0].upper())][0]
        wrange2=param["{}_CONT".format(camera[0].upper())][1]

        retval["PARAMS"] = param

        skyfiber, contfiberlow, contfiberhigh, meancontfiber, skycont = qalib.sky_continuum(
            frame, wrange1, wrange2)


        retval["METRICS"]={"SKYFIBERID": skyfiber.tolist(), "SKYCONT":skycont, "SKYCONT_FIBER":meancontfiber}


        ###############################################################
        # This section is for adding QA metrics for plotting purposes #
        ###############################################################

        ###############################################################

#        if qafile is not None:
#            outfile=qa.write_qa_ql(qafile,retval)
#            log.debug("Output QA data is in {}".format(outfile))
        if qafig is not None:
            fig.plot_sky_continuum(retval,qafig,plotconf=plotconf,hardplots=hardplots)
            log.debug("Output QA fig {}".format(qafig))

        return retval

    def get_default_config(self):
        return {}


class Sky_Rband(MonitoringAlg):
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="SKYRBAND"
        kwargs=config['kwargs']
        parms=kwargs['param']
        key=kwargs['refKey'] if 'refKey' in kwargs else "SKYRBAND"
        status=kwargs['statKey'] if 'statKey' in kwargs else "SKYRBAND_STATUS"
        kwargs["RESULTKEY"]=key
        kwargs["QASTATUSKEY"]=status
        if "ReferenceMetrics" in kwargs:
            r=kwargs["ReferenceMetrics"]
            if key in r:
                kwargs["REFERENCE"]=r[key]

        if "SKYRBAND_WARN_RANGE" in parms and "SKYRBAND_NORMAL_RANGE" in parms:
            kwargs["RANGES"]=[(np.asarray(parms["SKYRBAND_WARN_RANGE"]),QASeverity.WARNING),
                              (np.asarray(parms["SKYRBAND_NORMAL_RANGE"]),QASeverity.NORMAL)]# sorted by most severe to least severe
        MonitoringAlg.__init__(self,name,fr,config,logger)
    def run(self,*args,**kwargs):
        if len(args) == 0 :
            log.critical("No parameter is found for this QA")
            sys.exit("Update the configuration file for the parameters")

        if not self.is_compatible(type(args[0])):
            #raise qlexceptions.ParameterException("Incompatible input. Was expecting {} got {}".format(type(self.__inpType__),type(args[0])))
            log.critical("Incompatible input!")
            sys.exit("Was expecting {} got {}".format(type(self.__inpType__),type(args[0])))

        if kwargs["singleqa"] == 'Sky_Rband':
            night = kwargs['night']
            expid = '{:08d}'.format(kwargs['expid'])
            camera = kwargs['camera']
            frame = get_frame('cframe',night,expid,camera,kwargs["specdir"])
        else: frame=args[0]
        inputs=get_inputs(*args,**kwargs)

        return self.run_qa(frame,inputs)

    def run_qa(self,frame,inputs):
        camera=inputs["camera"]
        paname=inputs["paname"]
        fibermap=inputs["fibermap"]
        amps=inputs["amps"]
        qafile=inputs["qafile"]
        qafig=inputs["qafig"]
        param=inputs["param"]
        refmetrics=inputs["refmetrics"]
        plotconf=inputs["plotconf"]
        hardplots=inputs["hardplots"]

        if isinstance(frame,QFrame):
            frame = frame.asframe()

        #- qa dictionary
        retval={}
        retval["NIGHT"] = frame.meta["NIGHT"]
        retval["PANAME" ]= paname
        retval["QATIME"] = datetime.datetime.now().isoformat()
        retval["EXPID"] = '{0:08d}'.format(frame.meta["EXPID"])
        retval["CAMERA"] = camera
        retval["FLAVOR"] = frame.meta["FLAVOR"]
        kwargs=self.config['kwargs']

        if frame.meta["FLAVOR"] == 'science':
            fibmap =fits.open(kwargs['FiberMap'])
            retval["PROGRAM"]=program=fibmap[1].header['PROGRAM']

        if param is None:
            log.critical("No parameter is given for this QA! ")
            sys.exit("Check the configuration file")

        retval["PARAMS"] = param

        #- Find sky fibers
        objects=frame.fibermap['OBJTYPE']
        skyfibers=np.where(objects=="SKY")[0]

        flux=frame.flux
        wave=frame.wave
        #- Set appropriate filter and zero point
        if camera[0].lower() == 'r':
            responsefilter='decam2014-r'

            #- Get filter response information from speclite
            try:
                from pkg_resources import resource_filename
                responsefile=resource_filename('speclite','data/filters/{}.ecsv'.format(responsefilter))
                #- Grab wavelength and response information from file
                rfile=np.genfromtxt(responsefile)
                rfile=rfile[1:] # remove wavelength/response labels
                rwave=np.zeros(rfile.shape[0])
                response=np.zeros(rfile.shape[0])
                for i in range(rfile.shape[0]):
                    rwave[i]=10.*rfile[i][0] # convert to angstroms
                    response[i]=rfile[i][1]
            except:
                log.critical("Could not find filter response file, can't compute spectral magnitudes")

            #- Convole flux with response information
            res=np.zeros(frame.wave.shape)
            for w in range(response.shape[0]):
                if w >= 1 and w<= response.shape[0]-2:
                    ind=np.abs(frame.wave-rwave[w]).argmin()
                    lo=(rwave[w]-rwave[w-1])/2
                    wlo=rwave[w]-lo
                    indlo=np.abs(frame.wave-wlo).argmin()
                    hi=(rwave[w+1]-rwave[w])/2
                    whi=rwave[w]+hi
                    indhi=np.abs(frame.wave-whi).argmin()
                    res[indlo:indhi]=response[w]
            skyrflux=res*flux[skyfibers]

            #- Calculate integrals for sky fibers
            integrals=[]
            for ii in range(len(skyrflux)):
                integrals.append(qalib.integrate_spec(frame.wave,skyrflux[ii]))
            integrals=np.array(integrals)

            #- Convert calibrated flux to fiber magnitude
            specmags=np.zeros(integrals.shape)
            specmags[integrals>0]=21.1-2.5*np.log10(integrals[integrals>0]/frame.meta["EXPTIME"])
            avg_skyrband=np.mean(specmags[specmags>0])

            retval["METRICS"]={"SKYRBAND_FIB":specmags,"SKYRBAND":avg_skyrband}

        #- If not in r channel, set reference and metrics to zero
        else:
            retval["PARAMS"]["SKYRBAND_{}_REF".format(program.upper())]=[0.]
            zerospec=np.zeros_like(skyfibers)
            zerorband=0.
            retval["METRICS"]={"SKYRBAND_FIB":zerospec,"SKYRBAND":zerorband}

#        if qafile is not None:
#            outfile=qa.write_qa_ql(qafile,retval)
#            log.debug("Output QA data is in {}".format(outfile))

        return retval

    def get_default_config(self):
        return {}


class Sky_Peaks(MonitoringAlg):
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="PEAKCOUNT"
        kwargs=config['kwargs']
        parms=kwargs['param']
        key=kwargs['refKey'] if 'refKey' in kwargs else "PEAKCOUNT"
        status=kwargs['statKey'] if 'statKey' in kwargs else "PEAKCOUNT_STATUS"
        kwargs["RESULTKEY"]=key
        kwargs["QASTATUSKEY"]=status
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
            log.critical("No parameter is given for this QA! ")
            sys.exit("Check the configuration file")

        if not self.is_compatible(type(args[0])):
            #raise qlexceptions.ParameterException("Incompatible input. Was expecting {} got {}".format(type(self.__inpType__),type(args[0])))
            log.critical("Incompatible input!")
            sys.exit("Was expecting {} got {}".format(type(self.__inpType__),type(args[0])))

        if kwargs["singleqa"] == 'Sky_Peaks':
            night = kwargs['night']
            expid = '{:08d}'.format(kwargs['expid'])
            camera = kwargs['camera']
            frame = get_frame('fframe',night,expid,camera,kwargs["specdir"])
        else: frame=args[0]
        inputs=get_inputs(*args,**kwargs)

        return self.run_qa(frame,inputs)

    def run_qa(self,frame,inputs):
        from desispec.qa.qalib import sky_peaks
        camera=inputs["camera"]
        paname=inputs["paname"]
        fibermap=inputs["fibermap"]
        amps=inputs["amps"]
        allpeaks=inputs["Peaks"]
        qafile=inputs["qafile"]
        qafig=inputs["qafig"]
        param=inputs["param"]
        refmetrics=inputs["refmetrics"]
        plotconf=inputs["plotconf"]
        hardplots=inputs["hardplots"]

        if isinstance(frame,QFrame):
            frame = frame.asframe()

        retval={}
        retval["PANAME"] = paname
        retval["QATIME"] = datetime.datetime.now().isoformat()
        retval["EXPID"] = '{0:08d}'.format(frame.meta["EXPID"])
        retval["CAMERA"] = camera
        retval["FLAVOR"] = frame.meta["FLAVOR"]
        kwargs=self.config['kwargs']

        if frame.meta["FLAVOR"] == 'science':
            fibmap =fits.open(kwargs['FiberMap'])
            retval["PROGRAM"]=fibmap[1].header['PROGRAM']

        retval["NIGHT"] = frame.meta["NIGHT"]

        # Parameters
        if param is None:
            log.critical("No parameter is given for this QA! ")
            sys.exit("Check the configuration file")

        param['B_PEAKS']=allpeaks['B_PEAKS']
        param['R_PEAKS']=allpeaks['R_PEAKS']
        param['Z_PEAKS']=allpeaks['Z_PEAKS']

        #nspec_counts, sky_counts, tgt_counts, tgt_counts_rms = sky_peaks(param, frame)
        nspec_counts, sky_counts, skyfibers, nskyfib= sky_peaks(param, frame)
        rms_nspec = np.std(nspec_counts)#qalib.getrms(nspec_counts)
        rms_skyspec = np.std(sky_counts)#qalib.getrms(sky_counts)

        sumcount_med_sky=np.median(sky_counts)

        retval["PARAMS"] = param

        fiberid=frame.fibermap['FIBER']

        retval["METRICS"]={"FIBERID":fiberid,"PEAKCOUNT":sumcount_med_sky,"PEAKCOUNT_NOISE":rms_skyspec,"PEAKCOUNT_FIB":nspec_counts,"SKYFIBERID":skyfibers, "NSKY_FIB":nskyfib}#,"PEAKCOUNT_TGT":tgt_counts,"PEAKCOUNT_TGT_NOISE":tgt_counts_rms}

        ###############################################################
        # This section is for adding QA metrics for plotting purposes #
        ###############################################################

        ###############################################################

#        if qafile is not None:
#            outfile=qa.write_qa_ql(qafile,retval)
#            log.debug("Output QA data is in {}".format(outfile))
        if qafig is not None:
            fig.plot_sky_peaks(retval,qafig,plotconf=plotconf,hardplots=hardplots)
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
        key=kwargs['refKey'] if 'refKey' in kwargs else "RESIDNOISE"
        status=kwargs['statKey'] if 'statKey' in kwargs else "RESID_STATUS"
        kwargs["RESULTKEY"]=key
        kwargs["QASTATUSKEY"]=status

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
            log.critical("No parameter is given for this QA! ")
            sys.exit("Check the configuration file")

        if not self.is_compatible(type(args[0])):
            #raise qlexceptions.ParameterException("Incompatible input. Was expecting {} got {}".format(type(self.__inpType__),type(args[0])))
            log.critical("Incompatible input!")
            sys.exit("Was expecting {} got {}".format(type(self.__inpType__),type(args[0])))

        if kwargs["singleqa"] == 'Sky_Residual':
            night = kwargs['night']
            expid = '{:08d}'.format(kwargs['expid'])
            camera = kwargs['camera']
            frame = get_frame('sframe',night,expid,camera,kwargs["specdir"])
        else: frame=args[0]
        inputs=get_inputs(*args,**kwargs)
        skymodel=args[1]

        return self.run_qa(frame,skymodel,inputs)

    def run_qa(self,frame,skymodel,inputs):
        from desispec.sky import qa_skysub
        camera=inputs["camera"]
        paname=inputs["paname"]
        fibermap=inputs["fibermap"]
        amps=inputs["amps"]
        qafile=inputs["qafile"]
        qafig=inputs["qafig"]
        param=inputs["param"]
        refmetrics=inputs["refmetrics"]
        plotconf=inputs["plotconf"]
        hardplots=inputs["hardplots"]

        if isinstance(frame,QFrame):
            frame = frame.asframe()

        if skymodel is None:
            raise IOError("Must have skymodel to find residual. It can't be None")
        #- return values
        retval={}
        retval["PANAME"] = paname
        retval["QATIME"] = datetime.datetime.now().isoformat()
        retval["EXPID"] = '{0:08d}'.format(frame.meta["EXPID"])
        retval["CAMERA"] = camera
        retval["FLAVOR"] = frame.meta["FLAVOR"]
        kwargs=self.config['kwargs']

        if frame.meta["FLAVOR"] == 'science':
            fibmap =fits.open(kwargs['FiberMap'])
            retval["PROGRAM"]=fibmap[1].header['PROGRAM']

        retval["NIGHT"] = frame.meta["NIGHT"]

        if param is None:
            log.critical("No parameter is given for this QA! ")
            sys.exit("Check the configuration file")

        retval["PARAMS"] = param

        qadict=qalib.sky_resid(param,frame,skymodel,quick_look=True)

        retval["METRICS"] = {}
        for key in qadict.keys():
            retval["METRICS"][key] = qadict[key]

        ###############################################################
        # This section is for adding QA metrics for plotting purposes #
        ###############################################################

        ###############################################################

#        if qafile is not None:
#            outfile=qa.write_qa_ql(qafile,retval)
#            log.debug("Output QA data is in {}".format(outfile))
        if qafig is not None:
            fig.plot_residuals(retval,qafig,plotconf=plotconf,hardplots=hardplots)
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
        key=kwargs['refKey'] if 'refKey' in kwargs else "DELTAMAG_TGT"
        status=kwargs['statKey'] if 'statKey' in kwargs else "DELTAMAG_TGT_STATUS"
        kwargs["RESULTKEY"]=key
        kwargs["QASTATUSKEY"]=status
        if "ReferenceMetrics" in kwargs:
            r=kwargs["ReferenceMetrics"]
            if key in r:
                kwargs["REFERENCE"]=r[key]

        if "DELTAMAG_WARN_RANGE" in parms and "DELTAMAG_NORMAL_RANGE" in parms:
            kwargs["RANGES"]=[(np.asarray(parms["DELTAMAG_WARN_RANGE"]),QASeverity.WARNING),
                              (np.asarray(parms["DELTAMAG_NORMAL_RANGE"]),QASeverity.NORMAL)]# sorted by most severe to least severe
        MonitoringAlg.__init__(self,name,fr,config,logger)
    def run(self,*args,**kwargs):
        if len(args) == 0 :
            log.critical("No parameter is given for this QA! ")
            sys.exit("Check the configuration file")

        if not self.is_compatible(type(args[0])):
            #raise qlexceptions.ParameterException("Incompatible input. Was expecting {} got {}".format(type(self.__inpType__),type(args[0])))
            log.critical("Incompatible input!")
            sys.exit("Was expecting {} got {}".format(type(self.__inpType__),type(args[0])))

        if kwargs["singleqa"] == 'Integrate_Spec':
            night = kwargs['night']
            expid = '{:08d}'.format(kwargs['expid'])
            camera = kwargs['camera']
            frame = get_frame('cframe',night,expid,camera,kwargs["specdir"])
        else: frame=args[0]
        inputs=get_inputs(*args,**kwargs)

        return self.run_qa(frame,inputs)

    def run_qa(self,frame,inputs):
        camera=inputs["camera"]
        paname=inputs["paname"]
        fibermap=inputs["fibermap"]
        amps=inputs["amps"]
        qafile=inputs["qafile"]
        qafig=inputs["qafig"]
        param=inputs["param"]
        refmetrics=inputs["refmetrics"]
        plotconf=inputs["plotconf"]
        hardplots=inputs["hardplots"]

        if isinstance(frame,QFrame):
            frame = frame.asframe()

        flux=frame.flux
        ivar=frame.ivar
        wave=frame.wave

        retval={}
        retval["PANAME" ] = paname
        retval["QATIME"] = datetime.datetime.now().isoformat()
        retval["NIGHT"] = frame.meta["NIGHT"]
        retval["EXPID"] = '{0:08d}'.format(frame.meta["EXPID"])
        retval["CAMERA"] = camera
        retval["FLAVOR"] = frame.meta["FLAVOR"]
        kwargs=self.config['kwargs']
        if frame.meta["FLAVOR"] == 'science':
            fibmap =fits.open(kwargs['FiberMap'])
            retval["PROGRAM"]=program=fibmap[1].header['PROGRAM']

        retval["NIGHT"] = frame.meta["NIGHT"]

        flux=frame.flux
        wave=frame.wave
        #- Grab magnitudes for appropriate filter
        if camera[0].lower() == 'b':
            band = 'G'
            responsefilter='decam2014-g'
        elif camera[0].lower() == 'r':
            band = 'R'
            responsefilter='decam2014-r'
        elif camera[0].lower() == 'z':
            band = 'Z'
            responsefilter='decam2014-z'
        else:
            raise ValueError("Camera {} not in b, r, or z channels...".format(camera))

        #- Find fibers per target type
        elgfibers = np.where((frame.fibermap['DESI_TARGET'] & desi_mask.ELG) != 0)[0]
        lrgfibers = np.where((frame.fibermap['DESI_TARGET'] & desi_mask.LRG) != 0)[0]
        qsofibers = np.where((frame.fibermap['DESI_TARGET'] & desi_mask.QSO) != 0)[0]
        bgsfibers = np.where((frame.fibermap['DESI_TARGET'] & desi_mask.BGS_ANY) != 0)[0]
        mwsfibers = np.where((frame.fibermap['DESI_TARGET'] & desi_mask.MWS_ANY) != 0)[0]
        stdfibers = np.where(isStdStar(frame.fibermap))[0]
        skyfibers = np.where(frame.fibermap['OBJTYPE'] == 'SKY')[0]

        #- Setup target fibers per program
        if program == 'dark':
            objfibers = [elgfibers,lrgfibers,qsofibers,stdfibers]
        elif program == 'gray':
            objfibers = [elgfibers,stdfibers]
        elif program == 'bright':
            objfibers = [bgsfibers,mwsfibers,stdfibers]

        magnitudes=np.zeros(frame.nspec)
        key = 'FLUX_'+band
        magnitudes = 22.5 - 2.5*np.log10(frame.fibermap[key])
        #- Set objects with zero flux to 30 mag
        zeroflux = np.where(frame.fibermap[key]==0.)[0]
        magnitudes[zeroflux] = 30.

        #- Get filter response information from speclite
        try:
            from pkg_resources import resource_filename
            responsefile=resource_filename('speclite','data/filters/{}.ecsv'.format(responsefilter))
            #- Grab wavelength and response information from file
            rfile=np.genfromtxt(responsefile)
            rfile=rfile[1:] # remove wavelength/response labels
            rwave=np.zeros(rfile.shape[0])
            response=np.zeros(rfile.shape[0])
            for i in range(rfile.shape[0]):
                rwave[i]=10.*rfile[i][0] # convert to angstroms
                response[i]=rfile[i][1]
        except:
            log.critical("Could not find filter response file, can't compute spectral magnitudes")

        #- Convole flux with response information
        res=np.zeros(frame.wave.shape)
        for w in range(response.shape[0]):
            if w >= 1 and w<= response.shape[0]-2:
                ind=np.abs(frame.wave-rwave[w]).argmin()
                lo=(rwave[w]-rwave[w-1])/2
                wlo=rwave[w]-lo
                indlo=np.abs(frame.wave-wlo).argmin()
                hi=(rwave[w+1]-rwave[w])/2
                whi=rwave[w]+hi
                indhi=np.abs(frame.wave-whi).argmin()
                res[indlo:indhi]=response[w]
        rflux=res*flux

        #- Calculate integrals for all fibers
        integrals=[]
        for ii in range(len(rflux)):
            integrals.append(qalib.integrate_spec(frame.wave,rflux[ii]))
        integrals=np.array(integrals)

        #- Convert calibrated flux to spectral magnitude
        specmags=np.zeros(integrals.shape)
        specmags[integrals>0]=21.1-2.5*np.log10(integrals[integrals>0]/frame.meta["EXPTIME"])

        #- Save number of negative flux fibers
        negflux=np.where(specmags==0.)[0]
        num_negflux=len(negflux)

        #- Set sky and negative flux fibers to 30 mag
        specmags[skyfibers]=30.
        specmags[negflux]=30.

        #- Calculate integrals for each target type
        tgt_specmags=[]
        for T in objfibers:
            if num_negflux != 0:
                T=np.array(list(set(T) - set(negflux)))
            obj_integ=[]
            for ii in range(len(rflux[T])):
                obj_integ.append(qalib.integrate_spec(frame.wave,rflux[T][ii]))
            obj_integ = np.array(obj_integ)

            #- Convert calibrated flux to spectral magnitude per terget type
            #- Using ST magnitude system because frame flux is in units ergs/s/cm**2/A
            obj_specmags = np.zeros(obj_integ.shape)
            obj_specmags[obj_integ>0] = 21.1-2.5*np.log10(obj_integ[obj_integ>0]/frame.meta["EXPTIME"])
            tgt_specmags.append(obj_specmags)

        tgt_specmags = np.array(tgt_specmags)

        #- Fiber magnitudes per target type
        tgt_mags=[]
        for obj in objfibers:
            if num_negflux != 0:
                obj=np.array(list(set(obj) - set(negflux)))
            tgt_mags.append(magnitudes[obj])

        tgt_mags = np.array(tgt_mags)

        #- Calculate delta mag, remove sky/negative flux fibers first
        remove_fib = np.array(list(set(skyfibers) | set(negflux)))
        nosky_specmags = np.delete(specmags,remove_fib)
        nosky_mags = np.delete(magnitudes,remove_fib)
        deltamag = nosky_specmags - nosky_mags

        #- Calculate avg delta mag per target type
        deltamag_tgt = tgt_specmags - tgt_mags
        deltamag_tgt_avg=[]
        for tgt in range(len(deltamag_tgt)):
            deltamag_tgt_avg.append(np.mean(deltamag_tgt[tgt]))

        if param is None:
            log.critical("No parameter is given for this QA! ")
            sys.exit("Check the configuration file")

        retval["PARAMS"] = param

        fiberid=frame.fibermap['FIBER']

        #SE: should not have any nan or inf at this point but let's keep it for safety measures here
        retval["METRICS"]={"FIBERID":fiberid,"NFIBNOTGT":num_negflux,"SPEC_MAGS":specmags, "DELTAMAG":np.nan_to_num(deltamag), "STD_FIBERID":stdfibers, "DELTAMAG_TGT":np.nan_to_num(deltamag_tgt_avg)}

        ###############################################################
        # This section is for adding QA metrics for plotting purposes #
        ###############################################################

        ###############################################################

#        if qafile is not None:
#            outfile=qa.write_qa_ql(qafile,retval)
#            log.debug("Output QA data is in {}".format(outfile))
        if qafig is not None:
            fig.plot_integral(retval,qafig,plotconf=plotconf,hardplots=hardplots)
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
        key=kwargs['refKey'] if 'refKey' in kwargs else "FIDSNR_TGT"
        status=kwargs['statKey'] if 'statKey' in kwargs else "FIDSNR_TGT_STATUS"
        kwargs["RESULTKEY"]=key
        kwargs["QASTATUSKEY"]=status
        if "ReferenceMetrics" in kwargs:
            r=kwargs["ReferenceMetrics"]
            if key in r:
                kwargs["REFERENCE"]=r[key]

        if "FIDSNR_TGT_WARN_RANGE" in parms and "FIDSNR_TGT_NORMAL_RANGE" in parms:
            kwargs["RANGES"]=[(np.asarray(parms["FIDSNR_TGT_WARN_RANGE"]),QASeverity.WARNING),
                              (np.asarray(parms["FIDSNR_TGT_NORMAL_RANGE"]),QASeverity.NORMAL)]# sorted by most severe to least severe
        MonitoringAlg.__init__(self,name,fr,config,logger)
    def run(self,*args,**kwargs):
        if len(args) == 0 :
            log.critical("No parameter is given for this QA! ")
            sys.exit("Check the configuration file")

        if not self.is_compatible(type(args[0])):
            #raise qlexceptions.ParameterException("Incompatible input. Was expecting {} got {}".format(type(self.__inpType__),type(args[0])))
            log.critical("Incompatible input!")
            sys.exit("Was expecting {} got {}".format(type(self.__inpType__),type(args[0])))

        if kwargs["singleqa"] == 'Calculate_SNR':
            night = kwargs['night']
            expid = '{:08d}'.format(kwargs['expid'])
            camera = kwargs['camera']
            frame = get_frame('sframe',night,expid,camera,kwargs["specdir"])
        else: frame=args[0]
        inputs=get_inputs(*args,**kwargs)

        return self.run_qa(frame,inputs)

    def run_qa(self,frame,inputs):
        camera=inputs["camera"]
        paname=inputs["paname"]
        fibermap=inputs["fibermap"]
        amps=inputs["amps"]
        qafile=inputs["qafile"]
        qafig=inputs["qafig"]
        param=inputs["param"]
        refmetrics=inputs["refmetrics"]
        plotconf=inputs["plotconf"]
        hardplots=inputs["hardplots"]

        if isinstance(frame,QFrame):
            frame = frame.asframe()

        #- return values
        retval={}
        retval["PANAME"] = paname
        retval["QATIME"] = datetime.datetime.now().isoformat()
        retval["EXPID"] = expid = '{0:08d}'.format(frame.meta["EXPID"])
        retval["CAMERA"] = camera
        retval["FLAVOR"] = frame.meta["FLAVOR"]
        kwargs=self.config['kwargs']

        if frame.meta["FLAVOR"] == 'science':
            fibmap =fits.open(kwargs['FiberMap'])
            retval["PROGRAM"]=program=fibmap[1].header['PROGRAM']

        objlist=[]
        if program == 'dark':
            objlist = ['ELG','LRG','QSO','STAR']
        elif program == 'gray':
            objlist = ['ELG','STAR']
        elif program == 'bright':
            objlist = ['BGS','MWS','STAR']

        retval["NIGHT"] = night = frame.meta["NIGHT"]

        ra = fibermap["TARGET_RA"]
        dec = fibermap["TARGET_DEC"]

        #- select band for mag, using DECAM_R if present
        if param is None:
            log.critical("No parameter is given for this QA! ")
            sys.exit("Check the configuration file")


        fidboundary=None

        qadict,fitsnr = qalib.orig_SNRFit(frame,night,camera,expid,param,fidboundary=fidboundary)

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

        rescut=param["RESIDUAL_CUT"]
        sigmacut=param["SIGMA_CUT"]

        ###############################################################
        # This section is for adding QA metrics for plotting purposes #
        ###############################################################

        ###############################################################

#        if qafile is not None:
#            outfile=qa.write_qa_ql(qafile,retval)
#            log.debug("Output QA data is in {}".format(outfile))
        if qafig is not None:
            fig.plot_SNR(retval,qafig,objlist,fitsnr,rescut,sigmacut,plotconf=plotconf,hardplots=hardplots)
            log.debug("Output QA fig {}".format(qafig))

        return retval

    def get_default_config(self):
        return {}

class Check_Resolution(MonitoringAlg):
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="CHECKARC"
        kwargs=config['kwargs']
        parms=kwargs['param']
        key=kwargs['refKey'] if 'refKey' in kwargs else "CHECKARC"
        status=kwargs['statKey'] if 'statKey' in kwargs else "CHECKARC_STATUS"
        kwargs["RESULTKEY"]=key
        kwargs["QASTATUSKEY"]=status

        if "ReferenceMetrics" in kwargs:
            r=kwargs["ReferenceMetrics"]
            if key in r:
                kwargs["REFERENCE"]=r[key]

        if "CHECKARC_WARN_RANGE" in parms and "CHECKARC_NORMAL_RANGE" in parms:
            kwargs["RANGES"]=[(np.asarray(parms["CHECKARC_WARN_RANGE"]),QASeverity.WARNING),
                              (np.asarray(parms["CHECKARC_NORMAL_RANGE"]),QASeverity.NORMAL)]

        MonitoringAlg.__init__(self,name,fr,config,logger)
    def run(self,*args,**kwargs):
        if len(args) == 0 :
            log.critical("No parameter is given for this QA! ")
            sys.exit("Check the configuration file")

        if not self.is_compatible(type(args[0])):
            #raise qlexceptions.ParameterException("Incompatible input. Was expecting {} got {}".format(type(self.__inpType__),type(args[0])))
            log.critical("Incompatible input!")
            sys.exit("Was expecting {} got {}".format(type(self.__inpType__),type(args[0])))

        if kwargs["singleqa"] == 'Check_Resolution':
            night = kwargs['night']
            expid = '{:08d}'.format(kwargs['expid'])
            camera = kwargs['camera']
            #- Finding psf file for QA
            #file_psf = get_psf('psf',night,expid,camera,kwargs["specdir"])
        else: file_psf = args[0]
        inputs=get_inputs(*args,**kwargs)

        return self.run_qa(file_psf,inputs)

    def run_qa(self,file_psf,inputs):
        camera=inputs["camera"]
        paname=inputs["paname"]
        fibermap=inputs["fibermap"]
        amps=inputs["amps"]
        qafile=inputs["qafile"]
        qafig=inputs["qafig"]
        param=inputs["param"]
        refmetrics=inputs["refmetrics"]
        plotconf=inputs["plotconf"]
        hardplots=inputs["hardplots"]

        retval={}
        retval['PANAME'] = paname
        kwargs=self.config['kwargs']
        retval["QATIME"] = datetime.datetime.now().isoformat()
        retval["EXPID"] = '{:08d}'.format(kwargs['expid'])
        retval["CAMERA"] = camera
        retval["PROGRAM"] = 'ARC'
        retval["FLAVOR"] = 'arc'
        retval["NIGHT"] = kwargs['night']


        # file_psf.ycoeff is not the wsigma_array.
        # FIX later.TEST QA with file_psf.ycoeff

        wsigma_array = file_psf.ysig_vs_wave_traceset._coeff
        p0 = wsigma_array[0:, 0:1]
        p1 = wsigma_array[0:, 1:2]
        p2 = wsigma_array[0:, 2:3]

        #- Save array of ones and zeros for good/no fits
        nfib = len(p0)
        nofit = np.where(p0 == 0.)[0]
        allfibs=np.ones(nfib)
        allfibs[nofit] = 0.
        #- Total number of fibers fit used as scalar metric
        ngoodfits = len(np.where(allfibs == 1.)[0])

        # Medians of Legendre Coeffs to be used as 'Model'
        medlegpolcoef = np.median(wsigma_array,axis = 0)

        wsigma_rms = np.sqrt(np.mean((wsigma_array - medlegpolcoef)**2,axis = 0))

        # Check how many of each parameter are outside of +- 2 RMS of the median.
        toperror = np.array([medlegpolcoef[val] + 2*wsigma_rms[val] for val in [0,1,2]])
        bottomerror = np.array([medlegpolcoef[val] - 2*wsigma_rms[val] for val in [0,1,2]])

        badparamrnum0 = list(np.where(np.logical_or(p0>toperror[0], p0<bottomerror[0]))[0])
        badparamrnum1 = list(np.where(np.logical_or(p1>toperror[1], p1<bottomerror[1]))[0])
        badparamrnum2 = list(np.where(np.logical_or(p2>toperror[2], p2<bottomerror[2]))[0])
        nbadparam = np.array([len(badparamrnum0), len(badparamrnum1), len(badparamrnum2)])

        retval["METRICS"]={"CHECKARC":ngoodfits, "GOODPSFS":allfibs, "CHECKPSF":nbadparam}
        retval["DATA"]={"LPolyCoef0":p0, "LPolyCoef1":p1, "LPolyCoef2":p2}

        if param is None:
            log.critical("No parameter is given for this QA! ")
            sys.exit("Check the configuration file")

        retval["PARAMS"] = param

        ###############################################################
        # This section is for adding QA metrics for plotting purposes #
        ###############################################################

        ###############################################################

#        if qafile is not None:
#            outfile=qa.write_qa_ql(qafile,retval)
#            log.debug("Output QA data is in {}".format(outfile))
        if qafig is not None:
            fig.plot_lpolyhist(retval,qafig,plotconf=plotconf,hardplots=hardplots)
            log.debug("Output QA fig {}".format(qafig))

        return retval

    def get_default_config(self):
        return {}

class Check_FiberFlat(MonitoringAlg):
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="CHECKFLAT"
        kwargs=config['kwargs']
        parms=kwargs['param']
        key=kwargs['refKey'] if 'refKey' in kwargs else "CHECKFLAT"
        status=kwargs['statKey'] if 'statKey' in kwargs else "CHECKFLAT_STATUS"
        kwargs["RESULTKEY"]=key
        kwargs["QASTATUSKEY"]=status

        if "ReferenceMetrics" in kwargs:
            r=kwargs["ReferenceMetrics"]
            if key in r:
                kwargs["REFERENCE"]=r[key]

        if "CHECKFLAT_WARN_RANGE" in parms and "CHECKFLAT_NORMAL_RANGE" in parms:
            kwargs["RANGES"]=[(np.asarray(parms["CHECKFLAT_WARN_RANGE"]),QASeverity.WARNING),
                              (np.asarray(parms["CHECKFLAT_NORMAL_RANGE"]),QASeverity.NORMAL)]

        MonitoringAlg.__init__(self,name,fr,config,logger)
    def run(self,*args,**kwargs):
        if len(args) == 0 :
            log.critical("No parameter is given for this QA! ")
            sys.exit("Check the configuration file")

        if not self.is_compatible(type(args[0])):
            #raise qlexceptions.ParameterException("Incompatible input. Was expecting {} got {}".format(type(self.__inpType__),type(args[0])))
            log.critical("Incompatible input!")
            sys.exit("Was expecting {} got {}".format(type(self.__inpType__),type(args[0])))

        if kwargs["singleqa"] == 'Check_FiberFlat':
            night = kwargs['night']
            expid = '{:08d}'.format(kwargs['expid'])
            camera = kwargs['camera']
        else: fibflat=args[0]
        inputs=get_inputs(*args,**kwargs)

        return self.run_qa(fibflat,inputs)

    def run_qa(self,fibflat,inputs):
        camera=inputs["camera"]
        paname=inputs["paname"]
        fibermap=inputs["fibermap"]
        amps=inputs["amps"]
        qafile=inputs["qafile"]
        qafig=inputs["qafig"]
        param=inputs["param"]
        refmetrics=inputs["refmetrics"]

        kwargs=self.config['kwargs']
        retval={}
        retval["PANAME"] = paname
        retval["QATIME"] = datetime.datetime.now().isoformat()
        retval["PROGRAM"] = 'FLAT'
        retval["FLAVOR"] = 'flat'
        retval["NIGHT"] = kwargs['night']
        retval["CAMERA"] = fibflat.header['CAMERA']
        retval["EXPID"] = '{:08d}'.format(kwargs['expid'])

        if param is None:
            log.critical("No parameter is given for this QA! ")
            sys.exit("Check the configuration file")

        retval["PARAMS"] = param

        #- Calculate mean and rms fiberflat value for each fiber
        fiberflat = fibflat.fiberflat
        avg_fiberflat=[]
        rms=[]
        for fib in range(len(fiberflat)):
            avg_fiberflat.append(np.mean(fiberflat[fib]))
            rms.append(np.std(fiberflat[fib]))

        #- Calculate mean of the fiber means for scalar metric
        avg_all = np.mean(avg_fiberflat)

        retval['METRICS'] = {"FLATMEAN":avg_fiberflat, "FLATRMS":rms, "CHECKFLAT":avg_all}

        ###############################################################
        # This section is for adding QA metrics for plotting purposes #
        ###############################################################

        ###############################################################

        return retval

    def get_default_config(self):
        return {}
