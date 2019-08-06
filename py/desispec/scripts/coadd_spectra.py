"""
Coadd spectra
"""

from __future__ import absolute_import, division, print_function
import os, sys, time

import numpy as np

from desiutil.log import get_logger

from ..io import read_spectra,write_spectra

def parse(options=None):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--infile", type=str,  help="input spectra file")
    parser.add_argument("-o","--outfile", type=str,  help="output spectra file")
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)

    return args

def main(args=None):

    log = get_logger()

    if args is None:
        args = parse()

    spectra = read_spectra(args.infile)
    
    write_spectra(args.outfile,spectra)
