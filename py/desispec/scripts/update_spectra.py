'''
desispec.scripts.update_spectra
===============================

Update healpix-grouped spectra from a set of input cframe files
'''

import os, sys
import argparse

import desispec.io
from desispec import pixgroup

def parse(options=None):
    parser = argparse.ArgumentParser(usage = "{prog} [options]")
    parser.add_argument("-i", "--infiles", type=str, required=True, nargs='+', help="Input cframe files")
    parser.add_argument("-o", "--outfile", type=str, help="output spectra file")
    parser.add_argument("-p", "--healpix", type=int, required=True, help="Nested HEALPix pixel number")
    parser.add_argument("-n", "--nside", type=int, default=64, help="Nested HEALPix nside")
    args = None
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args

def main(args) :

    #- Get output file location if not provided
    if args.outfile is None:
        args.outfile = desispec.io.findfile(
            'spectra', groupname=str(args.healpix), nside=args.nside)

    #- Read input frames
    frames = dict()
    for filename in args.infiles:
        frame = pixgroup.FrameLite.read(filename)
        key = (frame.header['NIGHT'], frame.header['EXPID'], frame.header['CAMERA'])
        frames[key] = frame

    #- Add any missing frames with blank data
    pixgroup.add_missing_frames(frames)

    #- Regroup into spectra
    spectra = pixgroup.frames2spectra(frames, args.healpix, nside=args.nside)

    if len(spectra.fibermap) == 0:
        raise ValueError('No spectra for healpix {} found in input cframe files'.format(args.healpix))

    #- Combine with prior output
    if os.path.exists(args.outfile):
        oldspectra = pixgroup.SpectraLite.read(args.outfile)
        #- TODO: remove any spectra that exist in both
        spectra = oldspectra + spectra

    #- Write output
    spectra.write(args.outfile)

