"""
functions for bin/desi_compute_nightly_bias script
"""

import argparse
from desispec.ccdcalib import compute_nightly_bias
from desispec.io.util import decode_camword, parse_cameras

def parse(options=None):
    p = argparse.ArgumentParser(
            description="Compute nightly bias from ZEROs")
    p.add_argument('-n', '--night', type=int, required=True,
            help='YEARMMDD to process')
    p.add_argument('-c', '--cameras', type=str,
            default='a0123456789', help='list of cameras to process')
    p.add_argument('-o', '--outdir', type=str,
            help='output directory')
    p.add_argument('--nzeros', type=int, default=25,
            help='number of input ZEROS to use (saves memory)')
    p.add_argument('--minzeros', type=int, default=20,
            help='minimum number of good ZEROs required')
    p.add_argument('--mpi', action='store_true',
            help='use_mpi')

    args = p.parse_args(options)  #- uses sys.argv if options is None

    #- Convert cameras into list
    args.cameras = decode_camword(parse_cameras(args.cameras))

    return args

def main(args=None, comm=None):
    if args is None:
        args = parse()
    elif isinstance(args, (list, tuple)):
        args = parse(args)

    if comm is None and args.mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD

    del args.__dict__['mpi']
    compute_nightly_bias(**args.__dict__, comm=comm)

