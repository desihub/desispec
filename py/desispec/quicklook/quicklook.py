#!/usr/bin/env python
import sys,os,time,signal
import threading,string
import subprocess
import importlib
import yaml
from desispec.quicklook import qllogger


def testconfig(outfilename="qlconfig.yaml"):

    """ 
    Make a test Config file, should be provided by the QL framework
    Below the %% variables are replaced by actual object when the respective
    algorithm is executed.
    """
    qlog=qllogger.QLLogger("QuickLook",20)
    log=qlog.getlog()

    conf={'BiasImage':os.environ['BIASIMAGE'],# path to bias image
          'DarkImage':os.environ['DARKIMAGE'],# path to dark image
          'DataType':'Exposure',# type of input ['Exposure','Arc','Dark']
          'DebugLevel':20, # debug level
          'Period':5.0, # Heartbeat Period (Secs)
          'Timeout': 120.0, # Heartbeat Timeout (Secs)
          'DumpIntermediates':False, # whether to dump output of each step
          'FiberFlatFrame':os.environ['FIBERFLATFRAME'], # path to fiber flat frame
          'FiberFlatImage':os.environ['FIBERFLATIMAGE'], # for psf calibration
          'ArcLampImage':os.environ['ARCLAMPIMAGE'], # for psf calibration
          'SkyFile':os.environ['SKYFILE'], # path to Sky file
          'FiberMap':os.environ['FIBERMAP'],# path to fiber map
          'RawImage':os.environ['PIXIMAGE'],#path to input image
          'PixelFlat':os.environ['PIXELFLAT'], #path to pixel flat image
          'PSFFile':os.environ['PSFFILE'],  # .../desimodel/data/specpsf/psf-r.fits
          'basePath':os.environ['DESIMODEL'],
          'OutputFile':'lastframe_QL-r0-00000004.fits', # output file from last pipeline step. Need to output intermediate steps? Most likely after boxcar extraction?
          'PipeLine':[{'PA':{"ModuleName":"desispec.procalgs",
                             "ClassName":"BiasSubtraction",
                             "Name":"Bias Subtraction",
                             "kwargs":{"BiasImage":"%%BiasImage"}
                             },
                       'QAs':[{"ModuleName":"desispec.qa.qa_quicklook",
                               "ClassName":"Get_RMS",
                               "Name":"Get RMS",
                               "kwargs":{},
                               },
                              {"ModuleName":"desispec.qa.qa_quicklook",
                               "ClassName":"Count_Pixels",
                               "Name":"Count Pixels",
                               "kwargs":{'Width':3.}
                               }
                              ],
                       "StepName":"Preprocessing-Bias Subtraction",
                       "OutputFile":"QA_biassubtraction.yaml"
                       },
                      {'PA':{"ModuleName":"desispec.procalgs",
                             "ClassName":"DarkSubtraction",
                             "Name":"Dark Subtraction",
                             "kwargs":{"DarkImage":"%%DarkImage"}
                             },
                       'QAs':[{"ModuleName":"desispec.qa.qa_quicklook",
                               "ClassName":"Get_RMS",
                               "Name":"Get RMS",
                               "kwargs":{},
                               },
                              {"ModuleName":"desispec.qa.qa_quicklook",
                               "ClassName":"Count_Pixels",
                               "Name":"Count Pixels",
                               "kwargs":{'Width':3.},
                               }
                              ],
                       "StepName":"Preprocessing-Dark Subtraction",
                       "OutputFile":"QA_darksubtraction.yaml"
                       },
                      {'PA':{"ModuleName":"desispec.procalgs",
                             "ClassName":"PixelFlattening",
                             "Name":"Pixel Flattening",
                             "kwargs":{"PixelFlat":"%%PixelFlat"}
                             },
                       'QAs':[{"ModuleName":"desispec.qa.qa_quicklook",
                               "ClassName":"Get_RMS",
                               "Name":"Get RMS",
                               "kwargs":{},
                               },
                              {"ModuleName":"desispec.qa.qa_quicklook",
                               "ClassName":"Count_Pixels",
                               "Name":"Count Pixels",
                               "kwargs":{'Width':3.},
                               }
                              ],
                       "StepName":"Preprocessing-Pixel Flattening",
                       "OutputFile":"QA_pixelflattening.yaml"
                       },
                      {'PA':{"ModuleName":"desispec.procalgs",
                             "ClassName":"BoxcarExtraction",
                             "Name":"Boxcar Extraction",
                             "kwargs":{"PSFFile":"%%PSFFile",
                                       "BoxWidth":2.5,
                                       "DeltaW":0.5,
                                       "Nspec":500
                                       }
                             },
                       'QAs':[],
                       "StepName":"Boxcar Extration",
                       "OutputFile":"QA_boxcarextraction.yaml"
                       }
                      ]
          }
    
    if "yaml" in outfilename:
        yaml.dump(conf,open(outfilename,"wb"))
    else:
        log.warning("Only yaml defined. Use yaml format in the output config file")
        sys.exit(0)

def get_chan_cam_exp(inpname):
    basename=os.path.basename(inpname)
    if basename =="":
        print "can't parse input file name"
        sys.exit("can't parse input file name %s"%inpname)
    mod,cid,expid=string.split(basename,'-')

    expid=int(string.replace(expid,".fits",""))
    chan=cid[0]
    cam=int(cid[1:])
    return (chan,cam,expid)

def getobject(conf,log):
     #qlog=qllogger("QuickLook",20)
     #log=qlog.getlog()
    log.debug("Running for %s %s %s"%(conf["ModuleName"],conf["ClassName"],conf))
    try:
        mod=__import__(conf["ModuleName"],fromlist=[conf["ClassName"]])
        klass=getattr(mod,conf["ClassName"])
        return klass(conf["Name"],conf)
    except Exception as e:
        log.error("Failed to import %s from %s. Error was '%s'"%(conf["ClassName"],conf["ModuleName"],e))
        return None

def mapkeywords(kw,kwmap):
    """
    Maps the keyword in the configuration to the corresponding object 
    returned by the desispec.io module.
    e.g  Bias Image file is mapped to biasimage object... for the same keyword "BiasImage" 
    """ 

    newmap={}
    qlog=qllogger.QLLogger("QuickLook",20)
    log=qlog.getlog()
    for k,v in kw.iteritems():
        if isinstance(v,basestring) and len(v)>=3 and  v[0:2]=="%%":
            if v[2:] in kwmap:
                newmap[k]=kwmap[v[2:]]
            else:
                log.warning("Can't find key %s in conversion map. Skipping"%(v[2:]))
        else:
            newmap[k]=v
    return newmap

def runpipeline(pl,convdict,conf,hb):
    inp=convdict["rawimage"]
    paconf=conf["PipeLine"]
    qlog=qllogger.QLLogger("QuickLook",0)
    log=qlog.getlog()
    for s,step in enumerate(pl):
        log.info("Starting to run step %s"%(paconf[s]["StepName"]))
        pa=step[0]
        pargs=mapkeywords(step[0].config["kwargs"],convdict)
        print inp
        print pargs
        try:
            hb.start("Running %s"%(step[0].name))
            inp=pa(inp,**pargs)
            print inp
        except Exception as e:
            log.critical("Failed to run PA %s error was %s"%(step[0].name,e))
            sys.exit("Failed to run PA %s"%(step[0].name))
        qaresult={}
        for qa in step[1]:
            try:
                qargs=mapkeywords(qa.config["kwargs"],convdict)
                hb.start("Running %s"%(qa.name))
                res=qa(inp,**qargs)
                log.debug("%s %s"%(qa.name,inp))
                qaresult[qa.name]=res
            except Exception as e:
                log.warning("Failed to run QA %s error was %s"%(qa.name,e))
        if len(qaresult):
            yaml.dump(qaresult,open(paconf[s]["OutputFile"],"wb"))
            hb.stop("Step %s finished. Output is in %s "%(paconf[s]["StepName"],paconf[s]["OutputFile"]))
        else:
            hb.stop("Step %s finished."%(paconf[s]["StepName"]))
    hb.stop("Pipeline processing finished. Serializing result")
    return inp

#- Setup pipeline from configuration

def setup_pipeline(config):
    import desispec.io.fibermap as fibIO
    import desispec.io.sky as skyIO
    import desispec.io.fiberflat as ffIO
    import desispec.fiberflat as ff
    import desispec.io.image as imIO
    import desispec.image as im
    import desispec.io.frame as frIO
    import desispec.frame as dframe
    import desispec.procalgs as procalgs
    from desispec.boxcar import do_boxcar
    from desispec.quicklook import qlheartbeat as QLHB

    qlog=qllogger.QLLogger("QuickLook",20)
    log=qlog.getlog()
    if config is None:
        return None
    log.info("Reading Configuration")
    if "RawImage" not in config:
        log.critical("Config is missing \"RawImage\" key.")
        sys.exit("Missing \"RawImage\" key.")
    inpname=config["RawImage"]
    if "FiberMap" not in config:
        log.critical("Config is missing \"FiberMap\" key.")
        sys.exit("Missing \"FiberMap\" key.")
    fibname=config["FiberMap"]
    proctype="Exposure"
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

    fiberflatframefile=None
    fiberflatframe=None
    if "FiberFlatFrame" in config:
        fiberflatframefile=config["FiberFlatFrame"]

    skyfile=None
    skyimage=None
    if "SkyFile" in config:
        skyfile=config["SkyFile"]
    

    if "PSFFile" in config:
        from specter.psf import load_psf
        import desispec.psf
        psf=desispec.psf.PSF(config["PSFFile"])
        #psf=load_psf(config["PSFFile"])

    if "basePath" in config:
        basePath=config["basePath"]

    hbeat.start("Reading input file %s"%inpname)
    inp=imIO.read_image(inpname)
    hbeat.start("Reading fiberMap file %s"%fibname)
    fibfile,fibhdr=fibIO.read_fibermap(fibname,header=True)
    convdict={"FiberMap":fibfile,"PSFFile":psf}

    if biasfile is not None:
        hbeat.start("Reading Bias Image %s"%biasfile)
        biasimage=imIO.read_image(biasfile)
        convdict["BiasImage"]=biasimage

    if darkfile is not None:
        hbeat.start("Reading Dark Image %s"%darkfile)
        darkimage=imIO.read_image(darkfile)
        convdict["DarkImage"]=darkimage

    if pixelflatfile:
        hbeat.start("Reading PixelFlat Image %s"%pixelflatfile)
        pixelflatimage=imIO.read_image(pixelflatfile)
        convdict["PixelFlat"]=pixelflatimage     
   
    if fiberflatimagefile:
        hbeat.start("Reading FiberFlat Image %s"%fiberflatimagefile)
        fiberflatimage=imIO.read_image(fiberflatimagefile)
        convdict["FiberFlatImage"]=fiberflatimage       
 
    if arclampimagefile:
        hbeat.start("Reading ArcLampImage %s"%arclampimagefile)
        arclampimage=imIO.read_image(arclampimagefile)
        convdict["ArcLampImage"]=arclampimage

    if fiberflatframefile: 
        hbeat.start("Reading FiberFlat frame %s"%fiberflatframefile)
        fiberflatframe=ffIO.read_fiberflat(fiberflatframefile)
        convdict["FiberFlatFrame"]=fiberflatframe

    if skyfile:
        hbeat.start("Reading SkyModel file %s"%skyfile)
        skymodel=skyIO.read_sky(skyfile)
        convdict["SkyFile"]=skymodel

    img=inp
    convdict["rawimage"]=img
    pipeline=[]
    for step in config["PipeLine"]:
        pa=getobject(step["PA"],log)
        if len(pipeline) == 0:
            if not pa.is_compatible(type(img)):
                log.critical("Pipeline configuration is incorrect! check configuration %s %s"%(img,pa.is_compatible(img)))
                sys.exit("Wrong pipeline configuration")
        else:
            if not pa.is_compatible(pipeline[-1][0].get_output_type()):
                log.critical("Pipeline configuration is incorrect! check configuration")
                log.critical("Can't connect input of %s to output of %s. Incompatible types"%(pa.name,pipeline[-1][0].name))
                sys.exit("Wrong pipeline configuration")
        qas=[]
        for q in step["QAs"]:
            qa=getobject(q,log)
            if not qa.is_compatible(pa.get_output_type()):
                log.warning("QA %s can not be used for output of %s. Skipping expecting %s got %s %s"%(qa.name,pa.name,qa.__inpType__,pa.get_output_type(),qa.is_compatible(pa.get_output_type())))
            else:
                qas.append(qa)
        pipeline.append([pa,qas])
    
    chan,cam,expid=get_chan_cam_exp(inpname)
    res=runpipeline(pipeline,convdict,config,hbeat)
    if isinstance(res,im.Image):
        if config["OutputFile"]: finalname=config["OutputFile"]
        else: finalname="image-%s%d-%08d.fits"%(chan,cam,expid)
        imIO.write_image(finalname,res,meta=None)        
    elif isinstance(res,dframe.Frame):
        if config["OutputFile"]: finalname=config["OutputFile"]
        else: finalname="frame-%s%d-%08d.fits"%(chan,cam,expid)
        frIO.write_frame(finalname,res,header=None)
    else:
        log.error("Result of pipeline is in unkown type %s. Don't know how to write"%(type(res)))
        sys.exit("Unknown pipeline result type %s."%(type(res)))
    log.info("Pipeline completed. Final result is in %s"%finalname)
    return 

