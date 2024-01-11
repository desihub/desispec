"""
desispec.scripts.fit_cte_night
==============================

"""

from __future__ import absolute_import, division

import argparse
import multiprocessing as mp
from desiutil.log import get_logger
from desispec.correct_cte import fit_cte_night
from desispec.io.util import decode_camword,parse_cameras
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
    parser.add_argument('-o','--outfile', type = str, default = None, required=False,
                        help = 'path of output cvs table (default is the calibnight directory of the prod)')
    parser.add_argument('--ncpu', type=int, default=default_nproc,
                        help = f"number of parallel processes to use [{default_nproc}]")
    parser.add_argument('--specprod-dir', type=str, default=None, required=False,
                        help = "specify another specprod dir for debugging")
    args = parser.parse_args(options)

    #- Convert cameras into list
    args.cameras = decode_camword(parse_cameras(args.cameras))

    return args

def _fit_cte_night_kwargs_wrapper(opts):
    """
    This function just unpacks opts dict for fit_cte_night so that it can be
    used with multiprocessing.Pool.map
    """

    table = fit_cte_night(night=opts["night"],camera=opts["camera"])
    filename = findfile("ctecorrnight",night=opts["night"],camera=opts["camera"],specprod_dir = opts["specprod_dir"])
    table.write(filename,overwrite=True)
    log.info(f"wrote {filename}")
    return filename

def main(args=None) :

    if not isinstance(args, argparse.Namespace):
        args = parse(args)

        #- Assemble options to pass for each camera
    #- so that they can be optionally parallelized
    opts_array = [ dict(night = args.night, camera = camera, specprod_dir = args.specprod_dir) for  camera in args.cameras ]

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

    #table=fit_cte_night(night=args.night,camera=args.camera)
    #table.write(args.outfile,overwrite=True)
    #log.info("successfully wrote %s"%args.outfile)
