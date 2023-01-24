"""
desispec.quicklook.quicklook
============================

Please add module-level documentation.
"""
from __future__ import absolute_import, division, print_function

import sys,os,time,signal
import threading,string
import subprocess
import importlib
import yaml
import astropy.io.fits as fits
import desispec.io.fibermap as fibIO
import desispec.io.sky as skyIO
import desispec.io.fiberflat as ffIO
import desispec.fiberflat as ff
import desispec.io.image as imIO
import desispec.image as im
import desispec.io.frame as frIO
import desispec.frame as dframe
from desispec.quicklook import qllogger
from desispec.quicklook import qlheartbeat as QLHB
from desispec.io import qa as qawriter
from desispec.quicklook.merger import QL_QAMerger
from desispec.quicklook import procalgs
from desiutil.io import yamlify

def get_chan_spec_exp(inpname,camera=None):
    """
    Get channel, spectrograph and expid from the filename itself

    Args:
        inpname: can be raw or pix, or frame etc filename
        camera: is required for raw case, eg, r0, b5, z8
                irrelevant for others
    """
    basename=os.path.basename(inpname)
    if basename == "":
        print("can't parse input file name")
        sys.exit("can't parse input file name {}".format(inpname))
    brk=string.split(inpname,'-')
    if len(brk)!=3: #- for raw files
        if camera is None:
            raise IOError("Must give camera for raw file")
        else:
            expid=int(string.replace(brk[1],".fits.fz",""))

    elif len(brk)==3: #- for pix,frame etc. files
        camera=brk[1]
        expid=int(string.replace(brk[2],".fits",""))
    chan=camera[0]
    spectrograph=int(camera[1:])
    return (chan,spectrograph,expid)

def getobject(conf,log):
     #qlog=qllogger("QuickLook",20)
     #log=qlog.getlog()
    log.debug("Running for {} {} {}".format(conf["ModuleName"],conf["ClassName"],conf))
    try:
        mod=__import__(conf["ModuleName"],fromlist=[conf["ClassName"]])
        klass=getattr(mod,conf["ClassName"])
        if "Name" in conf.keys():
            return klass(conf["Name"],conf)
        else:
            return klass(conf["ClassName"],conf)
    except Exception as e:
        log.error("Failed to import {} from {}. Error was '{}'".format(conf["ClassName"],conf["ModuleName"],e))
        return None

def mapkeywords(kw,kwmap):
    """
    Maps the keyword in the configuration to the corresponding object
    returned by the desispec.io module.
    e.g  Bias Image file is mapped to biasimage object... for the same keyword "BiasImage"
    """

    newmap={}
    # qlog=qllogger.QLLogger()
    # log=qlog.getlog()
    for k,v in kw.items():
        if isinstance(v,str) and len(v)>=3 and  v[0:2]=="%%": #- For direct configuration
            if v[2:] in kwmap:
                newmap[k]=kwmap[v[2:]]
            else:
                log.warning("Can't find key {} in conversion map. Skipping".format(v[2:]))
        if k in kwmap: #- for configs generated via desispec.quicklook.qlconfig
            newmap[k]=kwmap[k]
        else:
            newmap[k]=v
    return newmap

def runpipeline(pl,convdict,conf):
    """
    Runs the quicklook pipeline as configured

    Args:
        pl: is a list of [pa,qas] where pa is a pipeline step and qas the corresponding
            qas for that pa
        convdict: converted dictionary e.g : conf["IMAGE"] is the real psf file
            but convdict["IMAGE"] is like desispec.image.Image object and so on.
            details in setup_pipeline method below for examples.
        conf: a configured dictionary, read from the configuration yaml file.
            e.g: conf=configdict=yaml.safe_load(open('configfile.yaml','rb'))
    """

    qlog=qllogger.QLLogger()
    log=qlog.getlog()
    hb=QLHB.QLHeartbeat(log,conf["Period"],conf["Timeout"])

    inp=convdict["rawimage"]
    singqa=conf["singleqa"]
    paconf=conf["PipeLine"]
    qlog=qllogger.QLLogger()
    log=qlog.getlog()
    passqadict=None #- pass this dict to QAs downstream
    schemaMerger=QL_QAMerger(conf['Night'],conf['Expid'],conf['Flavor'],conf['Camera'],conf['Program'],convdict)
    QAresults=[]
    if singqa is None:
        for s,step in enumerate(pl):
            log.info("Starting to run step {}".format(paconf[s]["StepName"]))
            pa=step[0]
            pargs=mapkeywords(step[0].config["kwargs"],convdict)
            schemaStep=schemaMerger.addPipelineStep(paconf[s]["StepName"])
            try:
                hb.start("Running {}".format(step[0].name))
                oldinp=inp #-  copy for QAs that need to see earlier input
                inp=pa(inp,**pargs)
                if step[0].name == 'Initialize':
                    schemaStep.addMetrics(inp[1])
            except Exception as e:
                log.critical("Failed to run PA {} error was {}".format(step[0].name,e),exc_info=True)
                sys.exit("Failed to run PA {}".format(step[0].name))
            qaresult={}
            for qa in step[1]:
                try:
                    qargs=mapkeywords(qa.config["kwargs"],convdict)
                    hb.start("Running {}".format(qa.name))
                    qargs["dict_countbins"]=passqadict #- pass this to all QA downstream

                    if qa.name=="RESIDUAL" or qa.name=="Sky_Residual":
                        res=qa(inp[0],inp[1],**qargs)
                    else:
                        if isinstance(inp,tuple):
                            res=qa(inp[0],**qargs)
                        else:
                            res=qa(inp,**qargs)

                    if qa.name=="COUNTBINS" or qa.name=="CountSpectralBins":
                        passqadict=res
                    if "qafile" in qargs:
                        qawriter.write_qa_ql(qargs["qafile"],res)
                    log.debug("{} {}".format(qa.name,inp))
                    qaresult[qa.name]=res
                    schemaStep.addParams(res['PARAMS'])
                    schemaStep.addMetrics(res['METRICS'])
                except Exception as e:
                    log.warning("Failed to run QA {}. Got Exception {}".format(qa.name,e),exc_info=True)
            hb.stop("Step {} finished.".format(paconf[s]["StepName"]))
            QAresults.append([pa.name,qaresult])
        hb.stop("Pipeline processing finished. Serializing result")
    else:
        import numpy as np
        qa=None
        qas=[[],['Bias_From_Overscan','Get_RMS','Count_Pixels','Calc_XWSigma'],'Trace_Shifts','CountSpectralBins',['Sky_Continuum','Sky_Peaks'],['Calculate_SNR'],['Sky_Rband','Integrate_Spec']]

        singleqaperpa=['Bias_From_Overscan','Check_HDUs','Trace_Shifts','CountSpectralBins']
        for palg in range(len(qas)):
            if singqa in qas[palg]:
                pa=pl[palg][0]
                pac=paconf[palg]
                if singqa in singleqaperpa:
                    qa = pl[palg][1][0]
                else:
                    for qalg in range(len(qas[palg])):
                        if qas[palg][qalg] == singqa:
                            qa=pl[palg][1][qalg]
        if qa is None:
            log.critical("Unknown input QA... Valid QAs are: {}".format(qas))
            sys.exit()

        log.info("Starting to run step {}".format(pac["StepName"]))
        pargs=mapkeywords(pa.config["kwargs"],convdict)
        schemaStep=schemaMerger.addPipelineStep(pac["StepName"])
        qaresult={}
        try:
            qargs=mapkeywords(qa.config["kwargs"],convdict)
            hb.start("Running {}".format(qa.name))
            if singqa=="Sky_Residual":
                res=qa(inp[0],inp[1],**qargs)
            else:
                if isinstance(inp,tuple):
                    res=qa(inp[0],**qargs)
                else:
                    res=qa(inp,**qargs)
            if singqa=="CountSpectralBins":
                passqadict=res
            if "qafile" in qargs:
                qawriter.write_qa_ql(qargs["qafile"],res)
            log.debug("{} {}".format(qa.name,inp))
            schemaStep.addMetrics(res['METRICS'])
        except Exception as e:
            log.warning("Failed to run QA {}. Got Exception {}".format(qa.name,e),exc_info=True)
        if len(qaresult):
           if conf["DumpIntermediates"]:
                f = open(pac["OutputFile"],"w")
                f.write(yaml.dump(yamlify(qaresult)))
                log.info("{} finished".format(qa.name))

    #- merge QAs for this pipeline execution
    #- RS: don't write merged file if running single QA
    if singqa is None:
        log.debug("Dumping mergedQAs")
        from desispec.io import findfile
        specprod_dir=os.environ['QL_SPEC_REDUX'] if 'QL_SPEC_REDUX' in os.environ else ""
        destFile=findfile('ql_mergedQA_file',night=conf['Night'],
                          expid=conf['Expid'],
                          camera=conf['Camera'],
                          specprod_dir=specprod_dir)

        schemaMerger.writeTojsonFile(destFile)
        log.info("Wrote merged QA file {}".format(destFile))
        if isinstance(inp,tuple):
           return inp[0]
        else:
           return inp

#- Setup pipeline from configuration

def setup_pipeline(config):
    """
    Given a configuration from QLF, this sets up a pipeline [pa,qa] and also returns a
    conversion dictionary from the configuration dictionary so that Pipeline steps (PA) can
    take them. This is required for runpipeline.
    """
    qlog=qllogger.QLLogger()
    log=qlog.getlog()
    if config is None:
        return None
    log.debug("Reading Configuration")
    flavor=config["Flavor"]
    if "RawImage" not in config:
        log.critical("Config is missing \"RawImage\" key.")
        sys.exit("Missing \"RawImage\" key.")
    inpname=config["RawImage"]
    if flavor != 'bias' and flavor != 'dark':
        if "FiberMap" not in config:
            log.critical("Config is missing \"FiberMap\" key.")
            sys.exit("Missing \"FiberMap\" key.")
    fibname=config["FiberMap"]
    proctype="Exposure"
    if "Camera" in config:
        camera=config["Camera"]
    if "DataType" in config:
        proctype=config["DataType"]
    debuglevel=20
    if "DebugLevel" in config:
        debuglevel=config["DebugLevel"]
        log.setLevel(debuglevel)
    hbeat=QLHB.QLHeartbeat(log,config["Period"],config["Timeout"])
    if config["Timeout"]> 200.0:
        log.warning("Heartbeat timeout exceeding 200.0 seconds")
    dumpintermediates=False
    if "DumpIntermediates" in config:
        dumpintermediates=config["DumpIntermediates"]

    biasimage=None #- This will be the converted dictionary key
    biasfile=None
    if "BiasImage" in config:
        biasfile=config["BiasImage"]

    darkimage=None
    darkfile=None
    if "DarkImage" in config:
        darkfile=config["DarkImage"]

    pixelflatfile=None
    pixflatimage=None
    if "PixelFlat" in config:
        pixelflatfile=config["PixelFlat"]

    fiberflatimagefile=None
    fiberflatimage=None
    if "FiberFlatImage" in config:
        fiberflatimagefile=config["FiberFlatImage"]

    arclampimagefile=None
    arclampimage=None
    if "ArcLampImage" in config:
        arclampimagefile=config["ArcLampImage"]

    fiberflatfile=None
    fiberflat=None
    if config["Flavor"] == 'science':
        if "FiberFlatFile" in config:
            fiberflatfile=config["FiberFlatFile"]

    skyfile=None
    skyimage=None
    if "SkyFile" in config:
        skyfile=config["SkyFile"]

    psf_filename=None
    if "PSFFile" in config:
        psf_filename=config["PSFFile"]
        #import desispec.psf
        #psf=desispec.psf.PSF(config["PSFFile"])

    if "basePath" in config:
        basePath=config["basePath"]

    hbeat.start("Reading input file {}".format(inpname))
    inp=fits.open(inpname) #- reading raw image directly from astropy.io.fits
    hbeat.start("Reading fiberMap file {}".format(fibname))

    convdict={}

    if flavor != 'bias' and flavor != 'dark':
        fibfile=fibIO.read_fibermap(fibname)
        fibhdr=fibfile.meta
        convdict["FiberMap"]=fibfile

    if psf_filename is not None:
        convdict["PSFFile"]=psf_filename

    if biasfile is not None:
        hbeat.start("Reading Bias Image {}".format(biasfile))
        biasimage=imIO.read_image(biasfile)
        convdict["BiasImage"]=biasimage

    if darkfile is not None:
        hbeat.start("Reading Dark Image {}".format(darkfile))
        darkimage=imIO.read_image(darkfile)
        convdict["DarkImage"]=darkimage

    if pixelflatfile:
        hbeat.start("Reading PixelFlat Image {}".format(pixelflatfile))
        pixelflatimage=imIO.read_image(pixelflatfile)
        convdict["PixelFlat"]=pixelflatimage

    if fiberflatfile:
        hbeat.start("Reading FiberFlat {}".format(fiberflatfile))
        fiberflat=ffIO.read_fiberflat(fiberflatfile)
        convdict["FiberFlatFile"]=fiberflat

    if skyfile:
        hbeat.start("Reading SkyModel file {}".format(skyfile))
        skymodel=skyIO.read_sky(skyfile)
        convdict["SkyFile"]=skymodel

    if dumpintermediates:
        convdict["DumpIntermediates"]=dumpintermediates

    hbeat.stop("Finished reading all static files")

    img=inp
    convdict["rawimage"]=img
    pipeline=[]
    for step in config["PipeLine"]:
        pa=getobject(step["PA"],log)
        if len(pipeline) == 0:
            if not pa.is_compatible(type(img)):
                log.critical("Pipeline configuration is incorrect! check configuration {} {}".format(img,pa.is_compatible(img)))
                sys.exit("Wrong pipeline configuration")
        else:
            if not pa.is_compatible(pipeline[-1][0].get_output_type()):
                log.critical("Pipeline configuration is incorrect! check configuration")
                log.critical("Can't connect input of {} to output of {}. Incompatible types".format(pa.name,pipeline[-1][0].name))
                sys.exit("Wrong pipeline configuration")
        qas=[]
        for q in step["QAs"]:
            qa=getobject(q,log)
            if not qa.is_compatible(pa.get_output_type()):
                log.warning("QA {} can not be used for output of {}. Skipping expecting {} got {} {}".format(qa.name,pa.name,qa.__inpType__,pa.get_output_type(),qa.is_compatible(pa.get_output_type())))
            else:
                qas.append(qa)
        pipeline.append([pa,qas])
    return pipeline,convdict
