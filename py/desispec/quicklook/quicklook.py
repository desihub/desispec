#!/usr/bin/env python
import sys,os,time,signal
import threading,string
import subprocess
import importlib
import yaml
from desispec.quicklook import qllogger
from desispec.quicklook import qlheartbeat as QLHB


def testconfig(outfilename="qlconfig.yaml"):

    """ 
    Make a test Config file, should be provided by the QL framework
    Below the %% variables are replaced by actual object when the respective
    algorithm is executed.
    """
    qlog=qllogger.QLLogger("QuickLook",20)
    log=qlog.getlog()
    url=None #- QA output will be posted to QLF if set true

    conf={'BiasImage':os.environ['BIASIMAGE'],# path to bias image
          'DarkImage':os.environ['DARKIMAGE'],# path to dark image
          'DataType':'Exposure',# type of input ['Exposure','Arc','Dark']
          'DebugLevel':20, # debug level
          'Period':5.0, # Heartbeat Period (Secs)
          'Timeout': 120.0, # Heartbeat Timeout (Secs)
          'DumpIntermediates':False, # whether to dump output of each step
          'FiberFlatFile':os.environ['FIBERFLATFILE'], # path to fiber flat field file
          'FiberFlatImage':os.environ['FIBERFLATIMAGE'], # for psf calibration
          'ArcLampImage':os.environ['ARCLAMPIMAGE'], # for psf calibration
          'SkyFile':os.environ['SKYFILE'], # path to Sky file
          'FiberMap':os.environ['FIBERMAP'],# path to fiber map
          'RawImage':os.environ['PIXIMAGE'],#path to input image
          'PixelFlat':os.environ['PIXELFLAT'], #path to pixel flat image
          'PSFFile':os.environ['PSFFILE'],  # for boxcar this can be bootcalib psf or specter psf file
          #'PSFFile_sp':os.environ['PSFFILE_sp'], # .../desimodel/data/specpsf/psf-r.fits (for running 2d extraction)
          'basePath':os.environ['DESIMODEL'],
          'OutputFile':'lastframe_QL-r0-00000004.fits', # output file from last pipeline step. Need to output intermediate steps? Most likely after boxcar extraction?
          'PipeLine':[{'PA':{"ModuleName":"desispec.quicklook.procalgs",
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
                      {'PA':{"ModuleName":"desispec.quicklook.procalgs",
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
                      {'PA':{"ModuleName":"desispec.quicklook.procalgs",
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
                      #{'PA':{"ModuleName":"desispec.quicklook.procalgs",
                      #       "ClassName":"BoxcarExtraction",
                      #       "Name":"Boxcar Extraction",
                      #       "kwargs":{"PSFFile":"%%PSFFile",
                      #                 "BoxWidth":2.5,
                      #                 "DeltaW":0.5,
                      #                 "Nspec":500
                      #                 }
                      #       },
                      # 'QAs':[],
                      # "StepName":"Boxcar Extration",
                      # "OutputFile":"QA_boxcarextraction.yaml"
                      # },
                      {'PA':{"ModuleName":"desispec.quicklook.procalgs",
                             "ClassName":"Extraction_2d",
                             "Name":"2D Extraction",
                             "kwargs":{"PSFFile_sp":"/home/govinda/Desi/desimodel/data/specpsf/psf-r.fits",
                                       "Nspec":10,
                                       "Wavelength": "5630,7740,0.5",
                                       "FiberMap":"%%FiberMap" #need this for qa_skysub downstream as well.
                                       }
                             },
                       'QAs':[{"ModuleName":"desispec.qa.qa_quicklook",
                               "ClassName":"CountSpectralBins",
                               "Name":"Count Bins above n",
                               "kwargs":{'thresh':100,
                                         'camera':"r0",
                                         'expid':"%08d"%2,
                                         'url':url
                                        }
                               }
                             ],
                       "StepName":"2D Extraction",
                       "OutputFile":"qa-extract-r0-00000002.yaml"
                       },
                      {'PA':{"ModuleName":"desispec.quicklook.procalgs",
                             "ClassName": "ApplyFiberFlat",
                             "Name": "Apply Fiberflat",
                             "kwargs":{"FiberFlatFile":"%%FiberFlatFile"
                                      }
                             },
                       'QAs':[],
                       "StepName":"Apply Fiberflat",
                       "Outputfile":"apply_fiberflat_QA.yaml"
                      },
                      {'PA':{"ModuleName":"desispec.quicklook.procalgs",
                             "ClassName":"SubtractSky",
                             "Name": "Sky Subtraction",
                             "kwargs":{"SkyFile":"%%SkyFile"
                                      }
                             },
                       'QAs':[{"ModuleName":"desispec.qa.qa_quicklook",
                               "ClassName":"Calculate_SNR",
                               "Name":"Calculate Signal-to-Noise ratio",
                               "kwargs":{'SkyFile':"%%SkyFile",
                                         'camera':"r0",
                                         'expid':"%08d"%2,
                                         'url':url
                                        }
                               }
                             ],
                       "StepName": "Sky Subtraction",
                       "OutputFile":"qa-r0-00000002.yaml"
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

def runpipeline(pl,convdict,conf):
    """
    runs the quicklook pipeline as configured
    args:- pl: is a list of [pa,qas] where pa is a pipeline step and qas the cor
responding 
               qas for that pa
           conf: a configured dictionary, read from the configuration yaml file.
                 e.g: conf=configdict=yaml.load(open('configfile.yaml','rb'))
           convdict: converted dictionary
                 e.g : conf["IMAGE"] is the real psf file
                       but convdict["IMAGE"] is like desispec.image.Image object
 and so on.
                       details in setup_pipeline method below for examples.
    """
    
   
    qlog=qllogger.QLLogger("QuickLook",20)
    log=qlog.getlog()
    hb=QLHB.QLHeartbeat(log,conf["Period"],conf["Timeout"])

    inp=convdict["rawimage"]
    paconf=conf["PipeLine"]
    qlog=qllogger.QLLogger("QuickLook",0)
    log=qlog.getlog()
    for s,step in enumerate(pl):
        log.info("Starting to run step %s"%(paconf[s]["StepName"]))
        pa=step[0]
        pargs=mapkeywords(step[0].config["kwargs"],convdict)
        try:
            hb.start("Running %s"%(step[0].name))
            inp=pa(inp,**pargs)
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
    """
       Given a configuration from QLF, this sets up a pipeline [pa,qa] and also returns a     
       conversion dictionary from the configuration dictionary so that Pipeline steps (PA) can   
       take them. This is required for runpipeline.
    """
       
    import desispec.io.fibermap as fibIO
    import desispec.io.sky as skyIO
    import desispec.io.fiberflat as ffIO
    import desispec.fiberflat as ff
    import desispec.io.image as imIO
    import desispec.image as im
    import desispec.io.frame as frIO
    import desispec.frame as dframe
    from desispec.quicklook import procalgs
    from desispec.boxcar import do_boxcar

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

    fiberflatfile=None
    fiberflat=None
    if "FiberFlatFile" in config:
        fiberflatfile=config["FiberFlatFile"]

    skyfile=None
    skyimage=None
    if "SkyFile" in config:
        skyfile=config["SkyFile"]
    
    psf=None
    if "PSFFile" in config:
        #from specter.psf import load_psf
        import desispec.psf
        psf=desispec.psf.PSF(config["PSFFile"])
        #psf=load_psf(config["PSFFile"])

    if "basePath" in config:
        basePath=config["basePath"]

    hbeat.start("Reading input file %s"%inpname)
    inp=imIO.read_image(inpname)
    hbeat.start("Reading fiberMap file %s"%fibname)
    fibfile,fibhdr=fibIO.read_fibermap(fibname,header=True)

    convdict={"FiberMap":fibfile}

    if psf is not None:
        convdict["PSFFile"]=psf

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

    if fiberflatfile: 
        hbeat.start("Reading FiberFlat %s"%fiberflatfile)
        fiberflat=ffIO.read_fiberflat(fiberflatfile)
        convdict["FiberFlatFile"]=fiberflat

    if skyfile:
        hbeat.start("Reading SkyModel file %s"%skyfile)
        skymodel=skyIO.read_sky(skyfile)
        convdict["SkyFile"]=skymodel

    hbeat.stop("Finished reading all static files")

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
    return pipeline,convdict

