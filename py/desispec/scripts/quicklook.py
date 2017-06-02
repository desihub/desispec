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
    parser.add_argument("--rawdata_dir", type=str, required=False, help="rawdata directory. overrides $QL_SPEC_DATA in config")
    parser.add_argument("--specprod_dir",type=str, required=False, help="specprod directory, overrides $QL_SPEC_REDUX in config")
    parser.add_argument("--save",type=str, required=False,help="save this config to a file")
    parser.add_argument("--save_full",type=str, required=False,help="save this config to a file including all inputs")
    parser.add_argument("--qlf",type=str,required=False,help="setup for QLF run", default=False)
    parser.add_argument("--mergeQA", default=False, action='store_true',help="output Merged QA file")

    args=parser.parse_args()
    return args

def ql_main(args=None):

    qlog=qllogger.QLLogger("QuickLook",20)
    log=qlog.getlog()

    if args is None:
        args = parse()

    if args.dotest is not None:
        quicklook.testconfig(args.dotest)

    if args.rawdata_dir:
        rawdata_dir = args.rawdata_dir
    else:
        if 'QL_SPEC_DATA' not in os.environ:
            sys.exit("must set ${} environment variable or provide data directory".format('QL_SPEC_DATA'))
        rawdata_dir=os.getenv('QL_SPEC_DATA')

    if args.specprod_dir:
        specprod_dir = args.specprod_dir
    else:
        if 'QL_SPEC_REDUX' not in os.environ:
            sys.exit("must set ${} environment variable or provide output directory".format('QL_SPEC_REDUX'))
        specprod_dir=os.getenv('QL_SPEC_REDUX')

    if args.config is not None:
        if os.path.exists(args.config):
            if "yaml" in args.config:
                file=yaml.load(open(args.config,'r'))
                PAs=qlconfig.Palist(file['Flavor'])
                config=qlconfig.Make_Config(args.night,file['Flavor'],args.expid,args.camera,PAs,psfboot=file['PSFFile'],fiberflat=file['FiberFlatFile'],rawdata_dir=rawdata_dir,specprod_dir=specprod_dir)
                configdict=qlconfig.build_config(config)
        else:
            log.critical("Can't open config file {}".format(args.config))
            sys.exit("Can't open config file")
    else:
        log.warning("No config file given. Trying to create config from other options")
        PAs=qlconfig.Palist(args.flavor)

        config=qlconfig.Make_Config(args.night,args.flavor,args.expid,args.camera, PAs,psfboot=args.psfboot,rawdata_dir=rawdata_dir, specprod_dir=specprod_dir,fiberflat=args.fiberflat, qlf=args.qlf)
        configdict=qlconfig.build_config(config)
        configfile=qlconfig.build_config_short(config)

        #- save this config to a file
        if args.save:
            if "yaml" in args.save:
                f=open(args.save,"w")
                yaml.dump(configfile,f)
                log.info("Output saved for this configuration to {}".format(args.save))
                f.close()
            else:
                log.info("Can save config to only yaml output. Put a yaml in the argument")

        #- save this config to a file with all inputs
        if args.save_full:
            if "yaml" in args.save_full:
                f=open(args.save_full,"w")
                yaml.dump(configdict,f)
                log.info("Output saved for this configuration to {}".format(args.save_full))
                f.close()
            else:
                log.info("Can save config to only yaml output. Put a yaml in the argument")
        
    pipeline, convdict = quicklook.setup_pipeline(configdict)
    res=quicklook.runpipeline(pipeline,convdict,configdict,mergeQA=args.mergeQA)
    inpname=configdict["RawImage"]
    camera=configdict["Camera"]
    expid=configdict["Expid"]

    if isinstance(res,image.Image):
        if configdict["OutputFile"]: finalname=configdict["OutputFile"]
        else: finalname="image-{}-{:08d}.fits".format(camera,expid)
        imIO.write_image(finalname,res,meta=None)        
    elif isinstance(res,frame.Frame):
        if configdict["OutputFile"]: finalname=configdict["OutputFile"]
        else: finalname="frame-{}-{:08d}.fits".format(camera,expid)
        frIO.write_frame(finalname,res,header=None)
    else:
        log.error("Result of pipeline is in unkown type {}. Don't know how to write".format(type(res)))
        sys.exit("Unknown pipeline result type {}.".format(type(res)))
    log.info("Pipeline completed. Final result is in {}".format(finalname))
if __name__=='__main__':
    ql_main()    
