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
from desispec.io.meta import findfile
import desispec.image as image
import desispec.frame as frame
import desispec.io.frame as frIO
import desispec.io.image as imIO

import os,sys
import yaml

import argparse

def quietDesiLogger(loglvl=20):
    from desiutil.log import get_logger
    get_logger(level=loglvl)

def parse():
    """
        Should have either a pre existing config file, or need to generate one using config module
    """
    parser=argparse.ArgumentParser(description="Run QL on DESI data")
    parser.add_argument("-i", "--config_file", type=str, required=False,help="yaml file containing config dictionary",dest="config")
    parser.add_argument("-n","--night", type=str, required=False, help="night for the data")
    parser.add_argument("-c", "--camera", type=str, required=False, help= "camera for the raw data")
    parser.add_argument("-e","--expid", type=int, required=False, help="exposure id")
    parser.add_argument("--rawdata_dir", type=str, required=False, help="rawdata directory. overrides $QL_SPEC_DATA in config")
    parser.add_argument("--specprod_dir",type=str, required=False, help="specprod directory, overrides $QL_SPEC_REDUX in config")
    parser.add_argument("--fullconfig", type=str, required=False, help="full expanded configfile")
    parser.add_argument("--save",type=str, required=False,help="save this full config to a file")
    parser.add_argument("--qlf",type=str,required=False,help="setup for QLF run", default=False)
    parser.add_argument("--mergeQA", default=False, action='store_true',help="output Merged QA file")
    parser.add_argument("--loglvl",default=20,type=int,help="log level for quicklook (0=verbose, 50=Critical)")

    args=parser.parse_args()
    return args

def ql_main(args=None):
    from desispec.quicklook import quicklook,qllogger,qlconfig
    import desispec.image as image
    import desispec.frame as frame
    import desispec.io.frame as frIO
    import desispec.io.image as imIO

    if args is None:
        args = parse()

    qlog=qllogger.QLLogger(name="QuickLook",loglevel=args.loglvl)
    log=qlog.getlog()
    # Sami
    # quiet down DESI logs. We don't want DESI_LOGGER to print messages unless they are important
    # initalize singleton with WARNING level
    quietDesiLogger(args.loglvl+10)
    if args.config is not None:

        if args.rawdata_dir:
            rawdata_dir = args.rawdata_dir
        else:
            if 'QL_SPEC_DATA' not in os.environ:
                sys.exit("must set ${} environment variable or provide rawdata_dir".format('QL_SPEC_DATA'))
            rawdata_dir=os.getenv('QL_SPEC_DATA')

        if args.specprod_dir:
            specprod_dir = args.specprod_dir
        else:
            if 'QL_SPEC_REDUX' not in os.environ:
                sys.exit("must set ${} environment variable or provide specprod_dir".format('QL_SPEC_REDUX'))
            specprod_dir=os.getenv('QL_SPEC_REDUX')

        log.debug("Running Quicklook using configuration file {}".format(args.config))
        if os.path.exists(args.config):
            if "yaml" in args.config:
                config=qlconfig.Config(args.config, args.night,args.camera, args.expid,rawdata_dir=rawdata_dir, specprod_dir=specprod_dir)
                configdict=config.expand_config()
            else:
                log.critical("Can't open config file {}".format(args.config))
                sys.exit("Can't open config file")
        else:
            sys.exit("File does not exist: {}".format(args.config)) 
    
    elif args.fullconfig is not None: #- This is mostly for development/debugging purpose
       log.debug("Running Quicklook using full configuration file {}".format(args.fullconfig))
       if os.path.exists(args.fullconfig):
           if "yaml" in args.fullconfig:
               configdict=yaml.load(open(args.fullconfig,"r"))
           else:
               log.critical("Can't open config file {}".format(args.config))
               sys.exit("Can't open config file")
       else:
           sys.exit("File does not exist: {}".format(args.config))
    else:
        sys.exit("Must provide a valid config file. See desispec/data/quicklook for an example")

    #- save the expanded config to a file
    if args.save:
        if "yaml" in args.save:
            f=open(args.save,"w")
            yaml.dump(configdict,f)
            log.info("Output saved for this configuration to {}".format(args.save))
            f.close()
        else:
            log.warning("Can save config to only yaml output. Put a yaml in the argument")
        
    pipeline, convdict = quicklook.setup_pipeline(configdict)
    res=quicklook.runpipeline(pipeline,convdict,configdict,mergeQA=args.mergeQA)
    inpname=configdict["RawImage"]
    night=configdict["Night"]
    camera=configdict["Camera"]
    expid=configdict["Expid"]

    if configdict["OutputFile"] is None:
        log.warning("Output filename is None and has a object of {}. SKIPPING FINAL OUTPUT".format(type(res)))
        return
    if isinstance(res,image.Image):
        if configdict["OutputFile"]: 
            finalname=configdict["OutputFile"]
        else:
            finalname="image-{}-{:08d}.fits".format(camera,expid)
            log.critical("No final outputname given. Writing to a image file {}".format(finalname))
        imIO.write_image(finalname,res,meta=None)
    elif isinstance(res,frame.Frame):
        if configdict["OutputFile"]: 
            finalname=configdict["OutputFile"]
        else:
            finalname="frame-{}-{:08d}.fits".format(camera,expid)
            log.critical("No final outputname given. Writing to a frame file {}".format(finalname))
        frIO.write_frame(finalname,res,header=None)
    elif configdict["Flavor"] == 'arcs':
        if configdict["OutputFile"]:
            finalname=configdict["OutputFile"]
        else:
            finalname="psfnight-{}.fits".format(camera)
    elif configdict["Flavor"] == 'flat':
        if configdict["OutputFile"]:
            finalname=configdict["OutputFile"]
        else:
            finalname="fiberflat-{}-{:08d}.fits".format(camera,expid)
    else:
        log.error("Result of pipeline is an unknown type {}. Don't know how to write".format(type(res)))
        sys.exit("Unknown pipeline result type {}.".format(type(res)))
    log.info("Pipeline completed. Final result is in {}".format(finalname))

if __name__=='__main__':
    ql_main()    
