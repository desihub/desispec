"""
desispec.scripts.fit_cte_night
==============================

"""

import os
import argparse
import multiprocessing as mp
import numpy as np
from desiutil.log import get_logger
import desispec.correct_cte
from desispec.io.util import (decode_camword, camword_union, camword_intersection,
                              difference_camwords, parse_cameras, erow_to_goodcamword,
                              get_tempfilename)
from desispec.io import findfile
from desispec.parallel import default_nproc
from desispec.workflow.tableio import load_table
import yaml

def parse(options=None):
    parser = argparse.ArgumentParser(description="Fit charge transfer efficiency (CTE) model for a given night")

    parser.add_argument('-n','--night', type = int, default = None, required=True,
                        help = 'night')
    parser.add_argument('-c','--cameras', type = str, default = 'r0123456789z0123456789', required=False,
                        help = 'list of cameras to process')
    parser.add_argument('-e','--expids', type = str, default = None, required=False,
                        help = 'comma separated list of flat expids to use'
                        + ' the cte should be first followed by a 120s flat')
    parser.add_argument('-o','--outfile', type = str, default = None, required=False,
                        help = 'path of output cvs table (default is the calibnight directory of the prod)')
    parser.add_argument('--ncpu', type=int, default=default_nproc,
                        help = f"number of parallel processes to use [{default_nproc}]")
    parser.add_argument('--specprod-dir', type=str, default=None, required=False,
                        help = "specify another specprod dir for debugging")
    parser.add_argument('--append', action='store_true', required=False,
                        help = "append to pre-existing output instead of overwriting; removes duplicates")

    args = parser.parse_args(options)

    if args.expids is not None:
        args.expids = [int(e) for e in args.expids.split(',')]

    #- Convert cameras into list
    args.cameras = decode_camword(parse_cameras(args.cameras, loglevel='WARNING'))

    return args

def _fit_cte_night_kwargs_wrapper(opts):
    """
    This function just unpacks opts dict for fit_cte_night so that it can be
    used with multiprocessing.Pool.map
    """

    out = desispec.correct_cte.fit_cte_night(night=opts["night"],camera=opts["camera"],expids=opts["expids"])
    return out

def main(args=None, comm=None):

    log = get_logger()
    if not isinstance(args, argparse.Namespace):
        args = parse(args)

    if args.outfile is None:
        args.outfile = findfile("ctecorrnight", night=args.night, specprod_dir=args.specprod_dir)

    #- Create output directory if needed
    if comm is None or comm.rank == 0:
        outdir = os.path.dirname(os.path.abspath(args.outfile))
        os.makedirs(outdir, exist_ok=True)

    #- Check what cameras are actually needed by science exposures
    if args.expids is None:
        etablefile = findfile('exptable', night=args.night)
        etable = load_table(etablefile, tabletype='exptable')
        keep = etable['OBSTYPE'] == 'science'
        sci_etable = etable[keep]
        if len(sci_etable) == 0:
            if comm is None or comm.rank == 0:
                log.warning(f'No science exposures on {args.night}, but calculating CTE corrections anyway...')
        else:
            #- List of good camwords per exposure
            all_goodcamwords = [erow_to_goodcamword(row, suppress_logging=True) for row in etable]

            #- union = camword of any camera that was good on any exposure
            any_goodcamword = camword_union(all_goodcamwords)
            if comm is None or comm.rank == 0:
                log.debug('any_goodcamword = %s', any_goodcamword)

            #- trim to just the ones we are asking for in args
            args_camword = parse_cameras(args.cameras, loglevel='WARNING')
            final_camword = camword_intersection([args_camword, any_goodcamword])
            if comm is None or comm.rank == 0:
                log.debug('args_camword=%s x any_goodcamword=%s -> final_camword=%s',
                          args_camword, any_goodcamword, final_camword)

            if args_camword != final_camword:
                if comm is None or comm.rank == 0:
                    log.warning(f'Trimming {args_camword} to {any_goodcamword} needed by science exposures')
                args.cameras = decode_camword(final_camword)

    #- Assemble options to pass for each camera
    #- so that they can be optionally parallelized
    opts_array = list()
    for camera in args.cameras:
        opts_array.append(dict(night=args.night, camera=camera, expids=args.expids, specprod_dir=args.specprod_dir))

    num_cameras = len(args.cameras)
    if comm is not None:
        from mpi4py.futures import MPICommExecutor
        if comm.rank == 0:
            log.info(f'Processing {num_cameras} cameras with MPI')

        with MPICommExecutor(comm, root=0) as pool:
            cte_dicts = pool.map(_fit_cte_night_kwargs_wrapper, opts_array)

    elif args.ncpu > 1 and num_cameras>1:
        n = min(args.ncpu, num_cameras)
        log.info(f'Processing {num_cameras} cameras with {n} multiprocessing processes')
        with mp.Pool(n) as pool:
            cte_dicts = pool.map(_fit_cte_night_kwargs_wrapper, opts_array)

    else:
        log.info(f'Not using multiprocessing for {num_cameras} cameras')
        cte_dicts = list()
        for opts in opts_array:
            cte_dicts.append(_fit_cte_night_kwargs_wrapper(opts))

    #- Write output with rank 0
    if comm is None or comm.rank == 0:
        #- filter for None just in case, then stack into one table
        cte_dicts = [t for t in cte_dicts if t is not None]
        cte_dicts = sum(cte_dicts, [])

        if os.path.isfile(args.outfile):
            if args.append:
                log.info(f'Merging CTE params with existing results in {args.outfile}')
                orig_cte_dicts = yaml.safe_load(args.outfile)
                keys = ['NIGHT', 'CAMERA', 'AMPLIFIER', 'SECTOR']
                orig_tuples = [(x[k] for k in keys) for x in orig_cte_dicts]
                tuples = [(x[k] for k in keys) for x in cte_dicts]
                only_in_orig = np.isin(orig_tuples, tuples, invert=True)
                cte_dicts = ([orig_cte_dicts[i] for i in only_in_orig] +
                             cte_dicts)
            else:
                log.warning(f'Overwriting pre-existing {args.outfile}')

        tmpfile = get_tempfilename(args.outfile)
        with open(tmpfile, 'w') as f:
            f.write(yaml.dump(cte_dicts))
        os.rename(tmpfile, args.outfile)
        log.info(f"wrote {args.outfile}")

    if comm is not None:
        comm.barrier()
