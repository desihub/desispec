"""
desispec.scripts.fit_cte_night
==============================

"""

import os
import argparse
import multiprocessing as mp
from desiutil.log import get_logger
import desispec.correct_cte
from desispec.io.util import decode_camword, parse_cameras, get_tempfilename
from desispec.io import findfile
from desispec.parallel import default_nproc
from astropy.table import Table,vstack

log = get_logger()

def parse(options=None):
    parser = argparse.ArgumentParser(description="Fit CTE model for a given night")

    parser.add_argument('-n','--night', type = int, default = None, required=True,
                        help = 'night')
    parser.add_argument('-c','--cameras', type = str, default = 'r0123456789z0123456789', required=False,
                        help = 'list of cameras to process')
    parser.add_argument('-e','--expids', type = str, default = None, required=False,
                        help = 'comma separated list of flat expids to use')
    parser.add_argument('-o','--outfile', type = str, default = None, required=False,
                        help = 'path of output cvs table (default is the calibnight directory of the prod)')
    parser.add_argument('--ncpu', type=int, default=default_nproc,
                        help = f"number of parallel processes to use [{default_nproc}]")
    parser.add_argument('--specprod-dir', type=str, default=None, required=False,
                        help = "specify another specprod dir for debugging")

    args = parser.parse_args(options)

    if args.expids is not None:
        args.expids = [int(e) for e in args.expids.split(',')]

    #- Convert cameras into list
    args.cameras = decode_camword(parse_cameras(args.cameras))

    return args

def _fit_cte_night_kwargs_wrapper(opts):
    """
    This function just unpacks opts dict for fit_cte_night so that it can be
    used with multiprocessing.Pool.map
    """

    table = desispec.correct_cte.fit_cte_night(night=opts["night"],camera=opts["camera"],expids=opts["expids"])
    filename = findfile("ctecorrnight",night=opts["night"],camera=opts["camera"],specprod_dir = opts["specprod_dir"])
    tmpfile = get_tempfilename(filename)
    table.write(tmpfile)
    os.rename(tmpfile, filename)
    log.info(f"wrote {filename}")
    return filename

def main(args=None) :

    if not isinstance(args, argparse.Namespace):
        args = parse(args)

    #- Create output directory if needed
    filename = findfile("ctecorrnight", args.night, camera=args.cameras[0], specprod_dir=args.specprod_dir)
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    #- Assemble options to pass for each camera
    #- so that they can be optionally parallelized
    opts_array = list()
    for camera in args.cameras:
        opts_array.append(dict(night=args.night, camera=camera, expids=args.expids, specprod_dir= args.specprod_dir))

    num_cameras = len(args.cameras)
    if args.ncpu > 1 and num_cameras>1:
        n = min(args.ncpu, num_cameras)
        log.info(f'Processing {num_cameras} cameras with {n} multiprocessing processes')
        pool = mp.Pool(n)
        pool.map(_fit_cte_night_kwargs_wrapper, opts_array)
        pool.close()
        pool.join()
    else:
        log.info(f'Not using multiprocessing for {num_cameras} cameras')
        for opts in opts_array:
            _fit_cte_night_kwargs_wrapper(opts)
