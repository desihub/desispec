#
# See top-level LICENSE.rst file for Copyright information
#
# -*- coding: utf-8 -*-
"""
desispec.scripts.exposure_qa
============================

Please add module-level documentation.
"""

"""
This script computes QA scores per exposure, after the cframe are done
"""

#- enforce a batch-friendly matplotlib backend
from desispec.util import set_backend
set_backend()

import os,sys
import argparse
import glob
import numpy as np
import multiprocessing
from astropy.table import Table
import fitsio

from desiutil.log import get_logger
from desispec.io import specprod_root,findfile,read_exposure_qa,write_exposure_qa
from desispec.exposure_qa import compute_exposure_qa
from desispec.util import parse_int_args

def parse(options=None):
    parser = argparse.ArgumentParser(
                description="Calculate exposure QA")
    parser.add_argument('-o','--outfile', type=str, default=None, required=False,
                        help = 'Output summary file (optional)')
    parser.add_argument('--recompute', action = 'store_true',
                        help = 'recompute')
    parser.add_argument('--prod', type = str, default = None, required=False,
                        help = 'Path to input reduction, e.g. /global/cfs/cdirs/desi/spectro/redux/blanc/,  or simply prod version, like blanc, but requires env. variable DESI_SPECTRO_REDUX. Default is $DESI_SPECTRO_REDUX/$SPECPROD.')
    parser.add_argument('--outdir', type = str, default = None, required=False,
                        help = 'Path to ouput directory, default is the input prod directory. Files written in {outdir}/exposures/{NIGHT}/')
    parser.add_argument('-e','--expids', type = str, default = None, required=False,
                        help = 'Comma, or colon separated list of nights to process. ex: 12,14 or 12:23')
    parser.add_argument('-n','--nights', type = str, default = None, required=False,
                        help = 'Comma, or colon separated list of nights to process. ex: 20210501,20210502 or 20210501:20210531')
    parser.add_argument('--nproc', type = int, default = 1,
                        help = 'Multiprocessing.')

    args = None
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args


def func(night,expid,specprod_dir,outfile=None) :
    """
    Wrapper function to compute_exposure_qa for multiprocessing
    """
    log = get_logger()
    fiberqa_table, petalqa_table = compute_exposure_qa(night,expid,specprod_dir)
    if fiberqa_table is None :
        return None

    write_exposure_qa(outfile,fiberqa_table,petalqa_table)
    log.info("wrote {}".format(outfile))

    if "EXTNAME" in fiberqa_table.meta :
        fiberqa_table.meta.pop("EXTNAME")

    return(fiberqa_table.meta)

def _func(arg) :
    """
    Wrapper function to compute_exposure_qa for multiprocessing
    """
    return func(**arg)

def main(args=None):

    if args is None:
        args=parse()

    log = get_logger()

    if args.prod is None:
        args.prod = specprod_root()
    elif args.prod.find("/")<0 :
        args.prod = specprod_root(args.prod)
    if args.outdir is None :
        args.outdir = args.prod

    log.info('prod    = {}'.format(args.prod))
    log.info('outfile = {}'.format(args.outfile))

    if args.expids is not None:
        expids = parse_int_args(args.expids)
    else:
        expids = None

    dirnames = sorted(glob.glob('{}/exposures/????????'.format(args.prod)))
    nights=[]
    for dirname in dirnames :
        try :
            night=int(os.path.basename(dirname))
            nights.append(night)
        except ValueError as e :
            log.warning("ignore {}".format(dirname))

    if args.nights :
        requested_nights = parse_int_args(args.nights)
        nights=np.intersect1d(nights,requested_nights)

    log.info("nights = {}".format(nights))
    if expids is not None : log.info('expids = {}'.format(expids))

    summary_rows  = list()
    for count,night in enumerate(nights) :

        dirnames = sorted(glob.glob('{}/exposures/{}/*'.format(args.prod,night)))
        night_expids=[]
        for dirname in dirnames :
            try :
                expid=int(os.path.basename(dirname))
                night_expids.append(expid)
            except ValueError as e :
                log.warning("ignore {}".format(dirname))
        if expids is not None :
            night_expids = np.intersect1d(expids,night_expids)
            if night_expids.size == 0 :
                continue
        log.info("{} {}".format(night,night_expids))

        func_args = []
        for expid in night_expids :
            filename = findfile("exposureqa",night=night,expid=expid,specprod_dir=args.outdir)
            if not args.recompute :
                if os.path.isfile(filename) :
                    log.info("skip existing {}".format(filename))
                    head = fitsio.read_header(filename,"FIBERQA")
                    entry=dict()
                    for r in head.records() :
                        k=r['name']
                        if k in ['SIMPLE','XTENSION','BITPIX','NAXIS','NAXIS1','NAXIS2','EXTEND','PCOUNT','GCOUNT','TFIELDS','EXTNAME','CHECKSUM','DATASUM'] : continue
                        if k.find('TTYPE')>=0 or k.find('TFORM')>=0 : continue
                        entry[k]=r['value']
                    if len(list(entry.keys())) == 0 :
                        log.error(f"empty dictionnary for exposure {expid}")
                    else :
                        summary_rows.append(entry)
                    continue
            func_args.append({'night':night,'expid':expid,'specprod_dir':args.prod,'outfile':filename})

        if args.nproc == 1 :
            for func_arg in func_args :
                entry = func(**func_arg)
                if entry is not None :
                    summary_rows.append(entry)
        else :
            log.info("Multiprocessing with {} procs".format(args.nproc))
            pool = multiprocessing.Pool(args.nproc)
            results  =  pool.map(_func, func_args)
            for entry in results :
                if entry is not None :
                    summary_rows.append(entry)
            pool.close()
            pool.join()

    if args.outfile is not None and len(summary_rows)>0 :
        colnames=None
        good_rows=[]
        for i,row in enumerate(summary_rows) :
            keys=list(row.keys())
            if len(keys)>0 :
                if colnames is not None :
                    if len(keys) != len(colnames) :
                        log.error("mismatch")
                        log.error(keys)
                        log.error(colnames)
                        continue
                good_rows.append(row)
                colnames=keys
            else :
                print("empty row",i)
        table = Table(rows=good_rows, names=colnames)
        print(table)

        table.write(args.outfile,overwrite=True)
        log.info("wrote {}".format(args.outfile))

    if len(summary_rows)==0 :
        print("no data")

if __name__ == '__main__':
    main()
