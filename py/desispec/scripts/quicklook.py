"""
desispec.scripts.quicklook
==========================

Command line wrapper for running a QL pipeline

QuickLook team @Southern Methodist University (SMU)
First version Spring 2016
Latest revision July 2018

Running QuickLook::

    desi_quicklook -i qlconfig_science.yaml -n 20191001 -c r0 -e 3577

This requires having necessary input files and setting the following environment variables::

    QL_SPEC_DATA: directory containing raw/fibermap files (full path: $QL_SPEC_DATA/night/expid)
    QL_SPEC_REDUX: directory for QL output (full path: $QL_SPEC_REDUX/exposures/night/expid)
    DESI_CALIBRATION_DATA: directory containing calibration files

Necessary Quicklook command line arguments::

    -i,--config_file : path to QL configuration file
    -n,--night : night to be processed
    -c,--camera : camera to be processed
    -e,--expid : exposure ID to be processed

Optional QuickLook arguments::

    --rawdata_dir : directory containing raw/fibermap files (overrides $QL_SPEC_DATA)
    --specprod_dir : directory for QL output (overrides $QL_SPEC_REDUX)

Plotting options::

    -p (including path to plotting configuration file) : generate configured plots
    -p (only using -p with no configuration file) : generate QL hardcoded plots
"""

from __future__ import absolute_import, division, print_function

from desispec.quicklook import quicklook,qllogger,qlconfig
from desispec.io.meta import findfile
import desispec.image as image
import desispec.frame as frame
import desispec.io.frame as frIO
import desispec.io.image as imIO
from desispec.qproc.qframe import QFrame
from desispec.qproc.io import write_qframe


import os,sys
import yaml
import json
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
    parser.add_argument("--psfid", type=int, required=False, help="psf id")
    parser.add_argument("--flatid", type=int, required=False, help="flat id")
    parser.add_argument("--templateid", type=int, required=False, help="template id")
    parser.add_argument("--templatenight", type=int, required=False, help="template night")
    parser.add_argument("--rawdata_dir", type=str, required=False, help="rawdata directory. overrides $QL_SPEC_DATA in config")
    parser.add_argument("--specprod_dir",type=str, required=False, help="specprod directory, overrides $QL_SPEC_REDUX in config")
    parser.add_argument("--singleQA",type=str,required=False,help="choose one QA to run",default=None,dest="singqa")
    parser.add_argument("--loglvl",default=20,type=int,help="log level for quicklook (0=verbose, 50=Critical)")
    parser.add_argument("-p",dest='qlplots',nargs='?',default='noplots',help="generate QL static plots")
    parser.add_argument("--resolution",action='store_true', help="store full resolution information")
    args=parser.parse_args()
    return args

def ql_main(args=None):

    from desispec.util import set_backend
    _matplotlib_backend = None
    set_backend()
    from desispec.quicklook import quicklook,qllogger,qlconfig

    if args is None:
        args = parse()

    qlog=qllogger.QLLogger(name="QuickLook",loglevel=args.loglvl)
    log=qlog.getlog()

    # quiet down DESI logs. We don't want DESI_LOGGER to print messages unless they are important
    # initalize singleton with WARNING level
    quietDesiLogger(args.loglvl+10)

    if args.config is not None:
        #RS: have command line arguments for finding files via old datamodel
        psfid=None
        if args.psfid:
            psfid=args.psfid
        flatid=None
        if args.flatid:
            flatid=args.flatid
        templateid=None
        if args.templateid:
            templateid=args.templateid
        templatenight=None
        if args.templatenight:
            templatenight=args.templatenight

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
                config=qlconfig.Config(args.config, args.night,args.camera, args.expid, args.singqa, rawdata_dir=rawdata_dir, specprod_dir=specprod_dir,psfid=psfid,flatid=flatid,templateid=templateid,templatenight=templatenight,qlplots=args.qlplots,store_res=args.resolution)
                configdict=config.expand_config()
            else:
                log.critical("Can't open config file {}".format(args.config))
                sys.exit("Can't open config file")
        else:
            sys.exit("File does not exist: {}".format(args.config))
    else:
        sys.exit("Must provide a valid config file. See desispec/data/quicklook for an example")

    pipeline, convdict = quicklook.setup_pipeline(configdict)
    res=quicklook.runpipeline(pipeline,convdict,configdict)
    log.info("QuickLook Pipeline completed")

if __name__=='__main__':
    ql_main()
