"""
desispec.scripts.quicklook
===========================
Command line wrapper for running a QL pipeline 

S. Kama, G. Dhungana 
SMU
Spring 2016
"""

from __future__ import absolute_import, division, print_function

from desispec.quicklook import quicklook,qllogger,qlconfig
import desispec.image as image
import desispec.frame as frame
import desispec.io.frame as frIO
import desispec.io.image as imIO

import os,sys
import yaml

import argparse

def parse():
    """
        Should have either a pre existing config file, or need to generate one using config module
    """
    parser=argparse.ArgumentParser(description="Run QL on DESI data")
    parser.add_argument("-i", "--config_file", type=str, required=False,help="yaml file containing config dictionary",dest="config")
    parser.add_argument("-g", "--gen_testconfig", type=str, required=False, help="generate test configuration",dest="dotest")
    parser.add_argument("-n","--night", type=str, required=False, help="night for the data")
    parser.add_argument("-c", "--camera", type=str, required=False, help= "camera for the raw data")
    parser.add_argument("-e","--expid", type=int, required=False, help="exposure id")
    parser.add_argument("-f","--flavor", type=str, required=False, help="flavor of exposure",default="dark")
    parser.add_argument("--psfboot",type=str,required=False,help="psf boot file")
    parser.add_argument("--fiberflat",type=str, required=False, help="fiberflat file",default=None)
    parser.add_argument("--rawdata_dir", type=str, required=False, help="rawdata directory. overrides $DESI_SPECTRO_DATA in config")
    parser.add_argument("--specprod_dir",type=str, required=False, help="specprod directory, overrides $DESI_SPECTRO_REDUX/$SPECPROD in config")
    parser.add_argument("--save",type=str, required=False,help="save this config to a file")
    parser.add_argument("--qlf",type=str,required=False,help="setup for QLF run", default=False)
    
    args=parser.parse_args()
    return args

def ql_main(args=None):

    qlog=qllogger.QLLogger("QuickLook",20)
    log=qlog.getlog()

    if args is None:
        args = parse()

    if args.dotest is not None:
        quicklook.testconfig(args.dotest)

    if args.config is not None:
        if os.path.exists(args.config):
            if "yaml" in args.config:
                configdict=yaml.load(open(args.config,'rb'))
        else:
            log.critical("Can't open config file %s"%(args.config))
            sys.exit("Can't open config file")
    else:
        log.warning("No config file given. Trying to create config from other options")
        PAs=qlconfig.Palist(args.flavor)

        config=qlconfig.Make_Config(args.night,args.flavor,args.expid,args.camera, PAs,psfboot=args.psfboot,rawdata_dir=args.rawdata_dir, specprod_dir=args.specprod_dir,fiberflat=args.fiberflat, qlf=args.qlf)
        configdict=qlconfig.build_config(config)

        #- save this config to a file
        if args.save:
            if "yaml" in args.save:
                yaml.dump(configdict,open(args.save,"wb"))
                log.info("Output saved for this configuration to %s "%args.save)
            else:
                log.info("Can save config to only yaml output. Put a yaml in the argument")
        
    pipeline, convdict = quicklook.setup_pipeline(configdict)
    res=quicklook.runpipeline(pipeline,convdict,configdict)
    inpname=configdict["RawImage"]
    camera=configdict["Camera"]
    chan,spectrograph,expid=quicklook.get_chan_spec_exp(inpname,camera=camera) #- may be other ways to get it as well

    if isinstance(res,image.Image):
        if configdict["OutputFile"]: finalname=configdict["OutputFile"]
        else: finalname="image-%s%d-%08d.fits"%(chan,spectrograph,expid)
        imIO.write_image(finalname,res,meta=None)        
    elif isinstance(res,frame.Frame):
        if configdict["OutputFile"]: finalname=configdict["OutputFile"]
        else: finalname="frame-%s%d-%08d.fits"%(chan,spectrograph,expid)
        frIO.write_frame(finalname,res,header=None)
    else:
        log.error("Result of pipeline is in unkown type %s. Don't know how to write"%(type(res)))
        sys.exit("Unknown pipeline result type %s."%(type(res)))
    log.info("Pipeline completed. Final result is in %s"%finalname)
if __name__=='__main__':
    ql_main()    
