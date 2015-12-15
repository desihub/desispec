#!/usr/bin/env python
import sys,os,time,signal
import threading,string
import subprocess
import importlib
import cPickle as pickle
import yaml
# import logging
# from datetime import datetime
from desispec.qlpipeline import QLLogger

def testConfig(outFileName="QLConfig.yaml"):
    # make a test Config file, should be provided by the QL framework
    QLog=QLLogger.QLLogger("QuickLook",20)
    log=QLog.getLog()

    conf={'BiasImage':'/home/govinda/Desi/simulate/spectro/sim/exposures/20151127/quicklook/bias-r0.fits',# path to bias image
          'DarkImage':'/home/govinda/Desi/simulate/spectro/sim/exposures/20151127/quicklook/dark-r0.fits',# path to dark image
          'DataType':'Exposure',#type of input ['Exposure','Arc','Dark']
          'DebugLevel':20, #debug level
          'DumpIntermediates':False, #whether to dump output of each step
          'FiberFlat':None, #path to fiber flat image (frame?)
          'FiberMap':'/home/govinda/Desi/simulate/spectro/sim/exposures/20151127/fibermap-00000004.fits',#path to fiber map
          #'Input':'/media/DATA/DESI/20151210/pix-r0-00000001.fits',#path to input
          'Input':'/home/govinda/Desi/simulate/spectro/sim/exposures/20151127/pix-r0-00000004.fits',#path to input
          'PixelFlat':'/home/govinda/Desi/simulate/spectro/sim/exposures/20151127/quicklook/pixflat-r0.fits', #path to pixel flat image
          'PSFFile':'/home/govinda/Desi/desimodel/data/specpsf/psf-r.fits',
          'OutputFile':'boxframe_new-r0-00000004.fits',
          'PipeLine':[{'PA':{"ModuleName":"desispec.ProcAlgs",
                             "ClassName":"BiasSubtraction",
                             "Name":"Bias Subtraction",
                             "kwargs":{"BiasImage":"%%BiasImage"}
                             },
                       'MAs':[{"ModuleName":"desispec.qa.MonAlgs",
                               "ClassName":"Get_RMS",
                               "Name":"Get RMS",
                               "kwargs":{},
                               },
                              {"ModuleName":"desispec.qa.MonAlgs",
                               "ClassName":"Count_Pixels",
                               "Name":"Count Pixels",
                               "kwargs":{'Width':3.}
                               }
                              ],
                       "StepName":"Preprocessing-Bias Subtraction",
                       "OutputFile":"step1.yaml"
                       },
                      {'PA':{"ModuleName":"desispec.ProcAlgs",
                             "ClassName":"DarkSubtraction",
                             "Name":"Dark Subtraction",
                             "kwargs":{"DarkImage":"%%DarkImage"}
                             },
                       'MAs':[{"ModuleName":"desispec.qa.MonAlgs",
                               "ClassName":"Get_RMS",
                               "Name":"Get RMS",
                               "kwargs":{},
                               },
                              {"ModuleName":"desispec.qa.MonAlgs",
                               "ClassName":"Count_Pixels",
                               "Name":"Count Pixels",
                               "kwargs":{'Width':3.},
                               }
                              ],
                       "StepName":"Preprocessing-Dark Subtraction",
                       "OutputFile":"step2.yaml"
                       },
                      {'PA':{"ModuleName":"desispec.ProcAlgs",
                             "ClassName":"PixelFlattening",
                             "Name":"Pixel Flattening",
                             "kwargs":{"PixelFlat":"%%PixelFlat"}
                             },
                       'MAs':[{"ModuleName":"desispec.qa.MonAlgs",
                               "ClassName":"Get_RMS",
                               "Name":"Get RMS",
                               "kwargs":{},
                               },
                              {"ModuleName":"desispec.qa.MonAlgs",
                               "ClassName":"Count_Pixels",
                               "Name":"Count Pixels",
                               "kwargs":{'Width':3.},
                               }
                              ],
                       "StepName":"Preprocessing-Pixel Flattening",
                       "OutputFile":"step3.yaml"
                       },
                      {'PA':{"ModuleName":"desispec.ProcAlgs",
                             "ClassName":"BoxcarExtraction",
                             "Name":"Boxcar Extraction",
                             "kwargs":{"FiberMap":"%%FiberMap",
                                       "PSFFile":"%%PSFFile",
                                       #"Band":"z",
                                       "Band":"r",
                                       #"Band":"b",
                                       "Spectrograph":0,
                                       "BoxWidth":2.5,
                                       "Wmin":5625, 
                                       "Wmax":7741, 
                                       #"Wmin":7435,
                                       #"Wmax":9834,
                                       #"Wmin":3569,
                                       #"Wmax":5949,
                                       "DeltaW":0.5
                                       }
                             },
                       'MAs':[{"ModuleName":"desispec.qa.MonAlgs",
                               "ClassName":"Find_Sky_Continuum",
                               "Name":"Find Sky Continuum",
                               "kwargs":{"FiberMap":"%%FiberMap",
                                         "Wmin":None,
                                         "Wmax":None},
                               },
                              {"ModuleName":"desispec.qa.MonAlgs",
                               "ClassName":"Calculate_SNR",
                               "Name":"Calculate Signal-to-Noise ratio",
                               "kwargs":{},
                              },
                              ],
                       "StepName":"Boxcar Extration",
                       "OutputFile":"step4.yaml"
                       }
                      ]
          }
    
    if "pkl" in outFileName:
        pickle.dump(conf,open(outFileName,"wb"))
    elif "yaml" in outFileName:
        yaml.dump(conf,open(outFileName,"wb"))

def getChanCamExp(inpName):
    baseName=os.path.basename(inpName)
    if baseName =="":
        print "can't parse input file name"
        sys.exit("can't parse input file name %s"%inpName)
    mod,cid,expid=string.split(baseName,'-')

    expid=int(string.replace(expid,".fits",""))
    chan=cid[0]
    cam=int(cid[1:])
    return (chan,cam,expid)

def getObject(conf,log):
    # QLog=QLLogger("QuickLook",20)
    # log=QLog.getLog()
    log.debug("Running for %s %s %s"%(conf["ModuleName"],conf["ClassName"],conf))
    try:
        mod=__import__(conf["ModuleName"],fromlist=[conf["ClassName"]])
        klass=getattr(mod,conf["ClassName"])
        return klass(conf["Name"],conf)
    except Exception as e:
        log.error("Failed to import %s from %s. Error was '%s'"%(conf["ClassName"],conf["ModuleName"],e))
        return None

def replaceKeywords(kw,kwmap):
    newMap={}
    QLog=QLLogger.QLLogger("QuickLook",20)
    log=QLog.getLog()
    for k,v in kw.iteritems():
        if isinstance(v,basestring) and len(v)>=3 and  v[0:2]=="%%":
            if v[2:] in kwmap:
                newMap[k]=kwmap[v[2:]]
            else:
                log.warning("Can't find key %s in conversion map. Skipping"%(v[2:]))
        else:
            newMap[k]=v
    return newMap

def runPipeline(pl,convDict,conf,hb):
    inp=convDict["Input"]
    paconf=conf["PipeLine"]
    QLog=QLLogger.QLLogger("QuickLook",0)
    log=QLog.getLog()
    for s,step in enumerate(pl):
        log.info("Starting to run step %s"%(paconf[s]["StepName"]))
        PA=step[0]
        pargs=replaceKeywords(step[0].config["kwargs"],convDict)
        try:
            hb.start("Running %s"%(step[0].name))
            inp=PA(inp,**pargs)
        except Exception as e:
            log.critical("Failed to run PA %s error was %s"%(step[0].name,e))
            sys.exit("Failed to run PA %s"%(step[0].name))
        maResult={}
        for ma in step[1]:
            try:
                margs=replaceKeywords(ma.config["kwargs"],convDict)
                hb.start("Running %s"%(ma.name))
                res=ma(inp,**margs)
                log.debug("%s %s"%(ma.name,inp))
                maResult[ma.name]=res
            except Exception as e:
                log.warning("Failed to run MA %s error was %s"%(ma.name,e))
        if len(maResult):
            pickle.dump(maResult,open(paconf[s]["OutputFile"],"wb"))
            hb.stop("Step %s finished. Output is in %s "%(paconf[s]["StepName"],paconf[s]["OutputFile"]))
        else:
            hb.stop("Step %s finished."%(paconf[s]["StepName"]))
    hb.stop("Pipeline processing finished. Serializing result")
    return inp

def setupPipeLine(config):
    import desispec.io.fibermap as fibIO
    import desispec.io.image as imIO
    import desispec.image as im
    import desispec.io.frame as frIO
    import desispec.frame as dframe
    import desispec.ProcAlgs as ProcAlgs
    from desispec.qlpipeline import QLHeartbeat as QLHB

    QLog=QLLogger.QLLogger("QuickLook",20)
    log=QLog.getLog()
    if config is None:
        return None
    log.info("Reading Configuration")
    if "Input" not in config:
        log.critical("Config is missing \"Input\" key.")
        sys.exit("Missing \"Input\" key.")
    inpName=config["Input"]
    if "FiberMap" not in config:
        log.critical("Config is missing \"FiberMap\" key.")
        sys.exit("Missing \"FiberMap\" key.")
    fibName=config["FiberMap"]
    procType="Exposure"
    if "DataType" in config:
        procType=config["DataType"]
    debugLevel=20
    if "DebugLevel" in config:
        debugLevel=config["DebugLevel"]
        log.setLevel(debugLevel)
    hbeat=QLHB.QLHeartbeat(log,5.0,120.0)
    dumpIntermediates=False
    if "DumpIntermediates" in config:
        dumpIntermediates=config["DumpIntermediates"]
    biasImage=None
    biasFile=None
    if "BiasImage" in config:
        biasFile=config["BiasImage"]
    darkImage=None
    darkFile=None
    if "DarkImage" in config:
        darkFile=config["DarkImage"]
    pixelFlatFile=None
    pixFlatImage=None
    if "PixelFlat" in config:
        pixelFlatFile=config["PixelFlat"]
    fiberFlatFile=None
    fiberFlatImage=None
    if "FiberFlat" in config:
        fiberFlatFile=config["FiberFlat"]
    psfFileName=None
    psfFile=None
    if "PSFFile" in config:
        from specter.psf import load_psf
        psfFile=load_psf(config["PSFFile"])
    hbeat.start("Reading input file %s"%inpName)
    inp=imIO.read_image(inpName)
    #log.info("Reading fiberMap file %s"%fibName)
    hbeat.start("Reading fiberMap file %s"%fibName)
    fibFile,fibHdr=fibIO.read_fibermap(fibName,header=True)
    convDict={"FiberMap":fibFile,"PSFFile":psfFile}
    if biasFile is not None:
        hbeat.start("Reading Bias Image %s"%biasFile)
        biasImage=imIO.read_image(biasFile)
        convDict["BiasImage"]=biasImage
    if darkFile is not None:
        hbeat.start("Reading Dark Image %s"%darkFile)
        darkImage=imIO.read_image(darkFile)
        convDict["DarkImage"]=darkImage
    if pixelFlatFile:
        hbeat.start("Reading PixelFlat Image %s"%pixelFlatFile)
        pixFlatImage=imIO.read_image(pixelFlatFile)
        convDict["PixelFlat"]=pixFlatImage        
    if fiberFlatFile:
        hbeat.start("Reading FiberFlat Image %s"%fiberFlatFile)
        fiberFlatImage=imIO.read_image(fiberFlatFile)
        convDict["FiberFlat"]=fiberFlatImage        
    img=inp
    convDict["Input"]=img
    pipeline=[]
    for step in config["PipeLine"]:
        PA=getObject(step["PA"],log)
        if len(pipeline) == 0:
            if not PA.is_compatible(type(img)):
                log.critical("Pipeline configuration is incorrect! check configuration %s %s"%(img,PA.is_compatible(img)))
                sys.exit("Wrong pipeline configuration")
        else:
            if not PA.is_compatible(pipeline[-1][0].get_output_type()):
                log.critical("Pipeline configuration is incorrect! check configuration")
                log.critical("Can't connect input of %s to output of %s. Incompatible types"%(PA.name,pipeline[-1][0].name))
                sys.exit("Wrong pipeline configuration")
        mas=[]
        for m in step["MAs"]:
            MA=getObject(m,log)
            if not MA.is_compatible(PA.get_output_type()):
                log.warning("MA %s can not be used for output of %s. Skipping expecting %s got %s %s"%(MA.name,PA.name,MA.__inpType__,PA.get_output_type(),MA.is_compatible(PA.get_output_type())))
            else:
                mas.append(MA)
        pipeline.append([PA,mas])
    
    chan,cam,expid=getChanCamExp(inpName)
    res=runPipeline(pipeline,convDict,config,hbeat)
    if isinstance(res,im.Image):
        finalName="image-%s%d-%08d.fits"%(chan,cam,expid)
        imIO.write_image(finalName,res,meta=None)        
    elif isinstance(res,dframe.Frame):
        finalName="frame-%s%d-%08d.fits"%(chan,cam,expid)
        frIO.write_frame(finalName,res,header=None)
    else:
        log.error("Result of pipeline is in unkown type %s. Don't know how to write"%(type(res)))
        sys.exit("Unknown pipeline result type %s."%(type(res)))
    log.info("Pipeline completed final result is in %s"%finalName)
    return 
    #
    # start processing pipeline 
    # 
    
    # find channel from input name
    if biasImage:
        hbeat.start("Running Bias subtraction")
        img=do_biasSubtract(img,biasImage)
        hbeat.stop("Bias subtraction done")
        if dumpIntermediates:
            imIO.write_image("AfterBias-%s%d-%08d.fits"%(chan,cam,expid),img,meta=None)
    if darkImage:
        hbeat.start("Running Dark subtraction")
        img=do_darkSubtract(img,darkImage)
        hbeat.stop("Dark subtraction done")
        if dumpIntermediates:
            imIO.write_image("AfterDark-%s%d-%08d.fits"%(chan,cam,expid),img,meta=None)
        # Apply Monitoring
        hbeat.start("Running MAs after Dark subtraction")
        res={}
        res["get_rms"]=get_rms(img)
        fnam="DarkMon-%s%d-%08d.pkl"%(chan,cam,expid)
        pickle.dump(res,open(fnam,"wb"))
        hbeat.stop("Dark subtraction monitoring finished. Output is written to %s"%(fnam))
    # do pixel flat
       #do count_pixels
    if pixFlatImage:
        hbeat.start("Applying Pixel Flat ")
        img=do_pixelFlat(img,pixFlatImage)
        hbeat.stop("Pixel Flat application done")
        if dumpIntermediates:
            imIO.write_image("AfterPixelFlat-%s%d-%08d.fits"%(chan,cam,expid),img,meta=None)
        # Apply Monitoring
        hbeat.start("Running MAs after Pixel Flat")
        res={}
        res["count_pixels"]=count_pixels(img)
        fnam="PixelFlatMon-%s%d-%08d.pkl"%(chan,cam,expid)
        pickle.dump(res,open(fnam,"wb"))
        hbeat.stop("PixelFlat monitoring finished. Output is written to %s"%(fnam))
    hbeat.start("Running Boxcar Extraction ")
    frame=do_boxcar(img,chan,cam)
    hbeat.stop("Boxcar Extraction Finished")
    if(dumpIntermediates):
        frIO.write_frame("AfterBoxcar-%s%d-%08d.fits"%(chan,cam,expid),frame,header=None)
    hbeat.start("Running MAs after Boxcar Extraction")
    # Apply Monitoring
    res={}
    res["find_continuum"]=find_continuum(frame,fibFile)
    res["calculate_snr"]=calculate_snr(frame)
    fnam="BoxcarExtractionMon-%s%d-%08d.pkl"%(chan,cam,expid)
    pickle.dump(res,open(fnam,"wb"))
    hbeat.stop("Boxcar monitoring finished. Output is written to %s"%(fnam))
    finalName="frame-%s%d-%08d.fits"%(chan,cam,expid)
    frIO.write_frame(finalName,frame,header=None)
    log.info("QuickLook pipeline finished. Final output is written to %s"%finalName)

def basicPipeline(config):
    import desispec.io.fibermap as fibIO
    import desispec.io.image as imIO
    import desispec.image as im
    import desispec.io.frame as frIO
    from desispec.qa.MonAlgs import get_rms,count_pixels,find_continuum,count_fibers,calculate_snr
    from desispec.ProcAlgs import do_darkSubtract,do_biasSubtract,do_pixelFlat,do_boxcar
    from desispec.QLHeartbeat import QLHeartbeat as QLHB

    QLog=QLLogger.QLLogger("QuickLook",20)
    log=QLog.getLog()
    if config is None:
        return None
    log.info("Reading Configuration")
    if "Input" not in config:
        log.critical("Config is missing \"Input\" key.")
        sys.exit("Missing \"Input\" key.")
    inpName=config["Input"]
    if "FiberMap" not in config:
        log.critical("Config is missing \"FiberMap\" key.")
        sys.exit("Missing \"FiberMap\" key.")
    fibName=config["FiberMap"]
    procType="Exposure"
    if "DataType" in config:
        procType=config["DataType"]
    debugLevel=20
    if "DebugLevel" in config:
        debugLevel=config["DebugLevel"]
        log.setLevel(debugLevel)
    hbeat=QLHB.QLHeartbeat(log,5.0,120.0)
    dumpIntermediates=False
    if "DumpIntermediates" in config:
        dumpIntermediates=config["DumpIntermediates"]
    biasImage=None
    biasFile=None
    if "BiasImage" in config:
        biasFile=config["BiasImage"]
    darkImage=None
    darkFile=None
    if "DarkImage" in config:
        darkFile=config["DarkImage"]
    pixelFlatFile=None
    pixFlatImage=None
    if "PixelFlat" in config:
        pixelFlatFile=config["PixelFlat"]
    fiberFlatFile=None
    fiberFlatImage=None
    if "FiberFlat" in config:
        fiberFlatFile=config["FiberFlat"]
    hbeat.start("Reading input file %s"%inpName)
    inp=imIO.read_image(inpName)
    #log.info("Reading fiberMap file %s"%fibName)
    hbeat.start("Reading fiberMap file %s"%fibName)
    fibFile,fibHdr=fibIO.read_fibermap(fibName,header=True)
    if biasFile is not None:
        hbeat.start("Reading Bias Image %s"%biasFile)
        biasImage=imIO.read_image(biasFile)
    if darkFile is not None:
        hbeat.start("Reading Dark Image %s"%darkFile)
        darkImage=imIO.read_image(darkFile)
    if pixelFlatFile:
        hbeat.start("Reading PixelFlat Image %s"%pixelFlatFile)
        pixFlatImage=imIO.read_image(pixelFlatFile)
    if fiberFlatFile:
        hbeat.start("Reading FiberFlat Image %s"%fiberFlatFile)
        fiberFlatImage=imIO.read_image(fiberFlatFile)
    img=inp
    #
    # start processing pipeline 
    # 

    # find channel from input name
    chan,cam,expid=getChanCamExp(inpName)
    if biasImage:
        hbeat.start("Running Bias subtraction")
        img=do_biasSubtract(img,biasImage)
        hbeat.stop("Bias subtraction done")
        if dumpIntermediates:
            imIO.write_image("AfterBias-%s%d-%08d.fits"%(chan,cam,expid),img,meta=None)
    if darkImage:
        hbeat.start("Running Dark subtraction")
        img=do_darkSubtract(img,darkImage)
        hbeat.stop("Dark subtraction done")
        if dumpIntermediates:
            imIO.write_image("AfterDark-%s%d-%08d.fits"%(chan,cam,expid),img,meta=None)
        # Apply Monitoring
        hbeat.start("Running MAs after Dark subtraction")
        res={}
        res["get_rms"]=get_rms(img)
        fnam="DarkMon-%s%d-%08d.pkl"%(chan,cam,expid)
        pickle.dump(res,open(fnam,"wb"))
        hbeat.stop("Dark subtraction monitoring finished. Output is written to %s"%(fnam))
    # do pixel flat
       #do count_pixels
    if pixFlatImage:
        hbeat.start("Applying Pixel Flat ")
        img=do_pixelFlat(img,pixFlatImage)
        hbeat.stop("Pixel Flat application done")
        if dumpIntermediates:
            imIO.write_image("AfterPixelFlat-%s%d-%08d.fits"%(chan,cam,expid),img,meta=None)
        # Apply Monitoring
        hbeat.start("Running MAs after Pixel Flat")
        res={}
        res["count_pixels"]=count_pixels(img)
        fnam="PixelFlatMon-%s%d-%08d.pkl"%(chan,cam,expid)
        pickle.dump(res,open(fnam,"wb"))
        hbeat.stop("PixelFlat monitoring finished. Output is written to %s"%(fnam))
    hbeat.start("Running Boxcar Extraction ")
    frame=do_boxcar(img,chan,cam)
    hbeat.stop("Boxcar Extraction Finished")
    if(dumpIntermediates):
        frIO.write_frame("AfterBoxcar-%s%d-%08d.fits"%(chan,cam,expid),frame,header=None)
    hbeat.start("Running MAs after Boxcar Extraction")
    # Apply Monitoring
    res={}
    res["find_continuum"]=find_continuum(frame,fibFile)
    res["calculate_snr"]=calculate_snr(frame)
    fnam="BoxcarExtractionMon-%s%d-%08d.pkl"%(chan,cam,expid)
    pickle.dump(res,open(fnam,"wb"))
    hbeat.stop("Boxcar monitoring finished. Output is written to %s"%(fnam))
    finalName="frame-%s%d-%08d.fits"%(chan,cam,expid)
    frIO.write_frame(finalName,frame,header=None)
    log.info("QuickLook pipeline finished. Final output is written to %s"%finalName)

# This should go to desispec/bin?
if __name__ == '__main__':
    import optparse as op
    p = op.OptionParser(usage = "%")
    p.add_option("-c", "--config_file", type=str, help="Pickle file containing config dictionary",dest="config")
    p.add_option("-g", "--gen_testconfig", type=str, help="generate test configuration",dest="dotest")
    QLog=QLLogger.QLLogger("QuickLook",20)
    log=QLog.getLog()
    opts, args = p.parse_args()

    if opts.dotest is not None:
        testConfig(opts.dotest)
    if opts.config is None:
        log.critical("Need config file")
        sys.exit("Missing config parameter")
    if os.path.exists(opts.config):
        if "yaml" in opts.config:
            configDict=yaml.load(open(opts.config,'rb'))
        elif "pkl" in opts.config:
            configDict=pickle.load(open(opts.config,'rb'))
    else:
        log.critical("Can't open config file %s"%(opts.config))
        sys.exit("Can't open config file")
    #basicPipeline(configDict)
    setupPipeLine(configDict)
