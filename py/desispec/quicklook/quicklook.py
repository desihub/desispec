#!/usr/bin/env python
import sys,os,time,signal
import threading,string
import subprocess
import importlib
import cPickle as pickle
import yaml
# import logging
# from datetime import datetime
from desispec.quicklook import qllogger


def testconfig(outfilename="qlconfig.yaml"):
    # make a test Config file, should be provided by the QL framework
    qlog=qllogger.QLLogger("QuickLook",20)
    log=qlog.getlog()

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
                       "OutputFile":"newstep1.yaml"
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
                       "OutputFile":"newstep2.yaml"
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
                       "OutputFile":"newstep3.yaml"
                       },
                      {'PA':{"ModuleName":"desispec.procalgs",
                             "ClassName":"BoxcarExtraction",
                             "Name":"Boxcar Extraction",
                             "kwargs":{"PSFFile":"%%PSFFile",
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
                       'QAs':[{"ModuleName":"desispec.qa.qa_quicklook",
                               "ClassName":"Find_Sky_Continuum",
                               "Name":"Find Sky Continuum",
                               "kwargs":{"FiberMap":"%%FiberMap",
                                         "Wmin":None,
                                         "Wmax":None},
                               },
                              {"ModuleName":"desispec.qa.qa_quicklook",
                               "ClassName":"Calculate_SNR",
                               "Name":"Calculate Signal-to-Noise ratio",
                               "kwargs":{},
                              },
                              ],
                       "StepName":"Boxcar Extration",
                       "OutputFile":"newstep4.yaml"
                       }
                      ]
          }
    
    if "pkl" in outfilename:
        pickle.dump(conf,open(outfilename,"wb"))
    elif "yaml" in outfilename:
        yaml.dump(conf,open(outfilename,"wb"))

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

def replacekeywords(kw,kwmap):
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
    inp=convdict["Input"]
    paconf=conf["PipeLine"]
    qlog=qllogger.QLLogger("QuickLook",0)
    log=qlog.getlog()
    for s,step in enumerate(pl):
        log.info("Starting to run step %s"%(paconf[s]["StepName"]))
        pa=step[0]
        pargs=replacekeywords(step[0].config["kwargs"],convdict)
        #print "Check",pa.name
        #inp=pa(inp,**pargs)
        #print inp
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
                qargs=replacekeywords(qa.config["kwargs"],convdict)
                hb.start("Running %s"%(qa.name))
                res=qa(inp,**qargs)
                log.debug("%s %s"%(qa.name,inp))
                qaresult[qa.name]=res
            except Exception as e:
                log.warning("Failed to run QA %s error was %s"%(qa.name,e))
        if len(qaresult):
            pickle.dump(qaresult,open(paconf[s]["OutputFile"],"wb"))
            hb.stop("Step %s finished. Output is in %s "%(paconf[s]["StepName"],paconf[s]["OutputFile"]))
        else:
            hb.stop("Step %s finished."%(paconf[s]["StepName"]))
    hb.stop("Pipeline processing finished. Serializing result")
    return inp

def setup_pipeline(config):
    import desispec.io.fibermap as fibIO
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
    if "Input" not in config:
        log.critical("Config is missing \"Input\" key.")
        sys.exit("Missing \"Input\" key.")
    inpname=config["Input"]
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
    hbeat=QLHB.QLHeartbeat(log,5.0,120.0)
    dumpintermediates=False
    if "DumpIntermediates" in config:
        dumpintermediates=config["DumpIntermediates"]
    biasimage=None
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
    fiberflatfile=None
    fiberflatimage=None
    if "FiberFlat" in config:
        fiberflatfile=config["FiberFlat"]
    psffilename=None
    psffile=None
    if "PSFFile" in config:
        from specter.psf import load_psf
        psf=load_psf(config["PSFFile"])
    hbeat.start("Reading input file %s"%inpname)
    inp=imIO.read_image(inpname)
    #log.info("Reading fiberMap file %s"%fibName)
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
        pixflatimage=imIO.read_image(pixelflatfile)
        convdict["PixelFlat"]=pixflatimage        
    if fiberflatfile:
        hbeat.start("Reading FiberFlat Image %s"%fiberflatfile)
        fiberflatimage=imIO.read_image(fiberflatfile)
        convdict["FiberFlat"]=fiberflatimage        
    img=inp
    convdict["Input"]=img
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
        finalname="image-%s%d-%08d.fits"%(chan,cam,expid)
        imIO.write_image(finalname,res,meta=None)        
    elif isinstance(res,dframe.Frame):
        finalname="frame-%s%d-%08d.fits"%(chan,cam,expid)
        frIO.write_frame(finalname,res,header=None)
    else:
        log.error("Result of pipeline is in unkown type %s. Don't know how to write"%(type(res)))
        sys.exit("Unknown pipeline result type %s."%(type(res)))
    log.info("Pipeline completed final result is in %s"%finalname)
    return 
    #
    # start processing pipeline 
    # 
    
    # find channel from input name
    if biasimage:
        hbeat.start("Running Bias subtraction")
        img=do_biasSubtract(img,biasimage)
        hbeat.stop("Bias subtraction done")
        if dumpintermediates:
            imIO.write_image("AfterBias-%s%d-%08d.fits"%(chan,cam,expid),img,meta=None)
    if darkimage:
        hbeat.start("Running Dark subtraction")
        img=do_darkSubtract(img,darkimage)
        hbeat.stop("Dark subtraction done")
        if dumpintermediates:
            imIO.write_image("AfterDark-%s%d-%08d.fits"%(chan,cam,expid),img,meta=None)
        # Apply Monitoring
        hbeat.start("Running QAs after Dark subtraction")
        res={}
        res["get_rms"]=get_rms(img)
        fnam="Dark_qa-%s%d-%08d.yaml"%(chan,cam,expid)
        yaml.dump(res,open(fnam,"wb"))
        hbeat.stop("Dark subtraction quality assurance finished. Output is written to %s"%(fnam))
    # do pixel flat
       #do count_pixels
    if pixflatimage:
        hbeat.start("Applying Pixel Flat ")
        img=do_pixelFlat(img,pixflatimage)
        hbeat.stop("Pixel Flat application done")
        if dumpintermediates:
            imIO.write_image("AfterPixelFlat-%s%d-%08d.fits"%(chan,cam,expid),img,meta=None)
        # Apply Monitoring
        hbeat.start("Running QAs after Pixel Flat")
        res={}
        res["count_pixels"]=count_pixels(img)
        fnam="PixelFlat_qa-%s%d-%08d.yaml"%(chan,cam,expid)
        yaml.dump(res,open(fnam,"wb"))
        hbeat.stop("PixelFlat quality assurance finished. Output is written to %s"%(fnam))
    hbeat.start("Running Boxcar Extraction ")
    frame=do_boxcar(img,chan,psf,cam,boxwidth=2.5,dw=0.5,nspec=500)
    hbeat.stop("Boxcar Extraction Finished")
    if(dumpintermediates):
        frIO.write_frame("AfterBoxcar-%s%d-%08d.fits"%(chan,cam,expid),frame,header=None)
    hbeat.start("Running QAs after Boxcar Extraction")
    # Apply Monitoring
    res={}
    res["find_continuum"]=find_continuum(frame,fibFile)
    res["calculate_snr"]=calculate_snr(frame)
    fnam="BoxcarExtraction_qa-%s%d-%08d.yaml"%(chan,cam,expid)
    yaml.dump(res,open(fnam,"wb"))
    hbeat.stop("Boxcar monitoring finished. Output is written to %s"%(fnam))
    finalname="frame-%s%d-%08d.fits"%(chan,cam,expid)
    frIO.write_frame(finalname,frame,header=None)
    log.info("QuickLook pipeline finished. Final output is written to %s"%finalname)

def basic_pipeline(config):
    import desispec.io.fibermap as fibIO
    import desispec.io.image as imIO
    import desispec.image as im
    import desispec.io.frame as frIO
    from desispec.qa.qa_quicklook import get_rms,count_pixels,find_continuum,count_fibers,calculate_snr
    from desispec.procalgs import do_darkSubtract,do_biasSubtract,do_pixelFlat,boxcar_extract
    #from desispec.boxcar import do_boxcar
    from desispec.qlheartbeat import QLHeartbeat as QLHB

    qlog=qllogger.QLLogger("QuickLook",20)
    log=qlog.getlog()
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
    fibname=config["FiberMap"]
    proctype="Exposure"
    if "DataType" in config:
        proctype=config["DataType"]
    debuglevel=20
    if "DebugLevel" in config:
        debuglevel=config["DebugLevel"]
        log.setlevel(debuglevel)
    hbeat=QLHB.QLHeartbeat(log,5.0,120.0)
    dumpintermediates=False
    if "DumpIntermediates" in config:
        dumpintermediates=config["DumpIntermediates"]
    biasimage=None
    biasfile=None
    if "BiasImage" in config:
        biasfile=config["BiasImage"]
    darkimage=None
    darkfile=None
    if "DarkImage" in config:
        darkfile=config["DarkImage"]
    pixelflatfile=None
    pixFlatimage=None
    if "PixelFlat" in config:
        pixelflatfile=config["PixelFlat"]
    fiberflatfile=None
    fiberflatimage=None
    if "FiberFlat" in config:
        fiberflatfile=config["FiberFlat"]
    hbeat.start("Reading input file %s"%inpname)
    inp=imIO.read_image(inpName)
    #log.info("Reading fiberMap file %s"%fibName)
    hbeat.start("Reading fiberMap file %s"%fibname)
    fibfile,fibHdr=fibIO.read_fibermap(fibname,header=True)
    if biasfile is not None:
        hbeat.start("Reading Bias Image %s"%biasfile)
        biasimage=imIO.read_image(biasfile)
    if darkfile is not None:
        hbeat.start("Reading Dark Image %s"%darkfile)
        darkimage=imIO.read_image(darkfile)
    if pixelflatfile:
        hbeat.start("Reading PixelFlat Image %s"%pixelflatfile)
        pixflatimage=imIO.read_image(pixelflatfile)
    if fiberflatfile:
        hbeat.start("Reading FiberFlat Image %s"%fiberflatfile)
        fiberflatimage=imIO.read_image(fiberflatfile)
    img=inp
    #
    # start processing pipeline 
    # 

    # find channel from input name
    chan,cam,expid=get_chan_cam_exp(inpname)
    if biasimage:
        hbeat.start("Running Bias subtraction")
        img=do_biasSubtract(img,biasimage)
        hbeat.stop("Bias subtraction done")
        if dumpintermediates:
            imIO.write_image("AfterBias-%s%d-%08d.fits"%(chan,cam,expid),img,meta=None)
    if darkimage:
        hbeat.start("Running Dark subtraction")
        img=do_darkSubtract(img,darkimage)
        hbeat.stop("Dark subtraction done")
        if dumpintermediates:
            imIO.write_image("AfterDark-%s%d-%08d.fits"%(chan,cam,expid),img,meta=None)
        # Apply Monitoring
        hbeat.start("Running QAs after Dark subtraction")
        res={}
        res["get_rms"]=get_rms(img)
        fnam="Dark_qa-%s%d-%08d.yaml"%(chan,cam,expid)
        yaml.dump(res,open(fnam,"wb"))
        hbeat.stop("Dark subtraction QA finished. Output is written to %s"%(fnam))
    # do pixel flat
       #do count_pixels
    if pixflatimage:
        hbeat.start("Applying Pixel Flat ")
        img=do_pixelFlat(img,pixflatimage)
        hbeat.stop("Pixel Flat application done")
        if dumpintermediates:
            imIO.write_image("AfterPixelFlat-%s%d-%08d.fits"%(chan,cam,expid),img,meta=None)
        # Apply Monitoring
        hbeat.start("Running QAs after Pixel Flat")
        res={}
        res["count_pixels"]=count_pixels(img)
        fnam="PixelFlat_qa-%s%d-%08d.yaml"%(chan,cam,expid)
        yaml.dump(res,open(fnam,"wb"))
        hbeat.stop("PixelFlat monitoring finished. Output is written to %s"%(fnam))
    hbeat.start("Running Boxcar Extraction ")
    frame=boxcar_extract(img,chan,psf,cam,boxwidth=2.5,dw=0.5,nspec=500)
    hbeat.stop("Boxcar Extraction Finished")
    if(dumpintermediates):
        frIO.write_frame("AfterBoxcar-%s%d-%08d.fits"%(chan,cam,expid),frame,header=None)
    hbeat.start("Running QAs after Boxcar Extraction")
    # Apply Monitoring
    res={}
    res["find_continuum"]=find_continuum(frame,fibfile)
    res["calculate_snr"]=calculate_snr(frame)
    fnam="BoxcarExtractionMon-%s%d-%08d.yaml"%(chan,cam,expid)
    yaml.dump(res,open(fnam,"wb"))
    hbeat.stop("Boxcar QA finished. Output is written to %s"%(fnam))
    finalname="frame-%s%d-%08d.fits"%(chan,cam,expid)
    frIO.write_frame(finalname,frame,header=None)
    log.info("QuickLook extraction finished. Final frame output is written to %s"%finalname)

# This should go to desispec/bin?
"""
if __name__ == '__main__':
    import optparse as op
    p = op.OptionParser(usage = "%")
    p.add_option("-c", "--config_file", type=str, help="Pickle file containing config dictionary",dest="config")
    p.add_option("-g", "--gen_testconfig", type=str, help="generate test configuration",dest="dotest")
    qlog=qllogger.QLLogger("QuickLook",20)
    log=qlog.getlog()
    opts, args = p.parse_args()

    if opts.dotest is not None:
        testconfig(opts.dotest)
    if opts.config is None:
        log.critical("Need config file")
        sys.exit("Missing config parameter")
    if os.path.exists(opts.config):
        if "yaml" in opts.config:
            configdict=yaml.load(open(opts.config,'rb'))
        elif "pkl" in opts.config:
            configdict=pickle.load(open(opts.config,'rb'))
    else:
        log.critical("Can't open config file %s"%(opts.config))
        sys.exit("Can't open config file")
    #basic_pipeline(configdict)
    setup_pipeline(configdict)
"""
