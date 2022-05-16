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
            help='max number of input ZEROS to use (saves memory) [%(default)s]')
    p.add_argument('--minzeros', type=int, default=15,
            help='minimum number of good ZEROs required [%(default)s]')
    p.add_argument('--nskip', type=int, default=2,
            help='Number of zeros at start to skip [%(default)s]')
    p.add_argument('--anyzeros', action='store_true',
            help='allow non-calib ZEROs to be used')
    p.add_argument('--mpi', action='store_true',
            help='use MPI for parallelism')

    args = p.parse_args(options)  #- uses sys.argv if options is None

    #- Convert cameras into list
    args.cameras = decode_camword(parse_cameras(args.cameras))

    return args

def main(args=None, comm=None):
    if not isinstance(args, argparse.Namespace):
        args = parse(args)

    if comm is None and args.mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD

    del args.__dict__['mpi']
    compute_nightly_bias(**args.__dict__, comm=comm)
