
"""
Calculate the signal-to-noise ration for each object in this brick. 
"""

import sys
import os
import numpy as np

from desispec import io
from desispec.interpolation import resample_flux
from desispec.log import get_logger
from desispec.zfind.redmonster import RedMonsterZfind
from desispec.zfind import ZfindBase
from desispec.io.qa import load_qa_brick, write_qa_brick
from desispec.util import default_nproc

import argparse


def parse(options=None):
    parser = argparse.ArgumentParser(description="Fit redshifts and classifications on bricks.")
    parser.add_argument("-o", "--outfile", type=str, required=True,
        help="output file name")
    parser.add_argument("brickfiles", nargs="*")
    
    args = None
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args

def main(args) :

    log = get_logger()

    #- Read brick files for each channel
    log.info("Reading bricks")
    brick = dict()
    for filename in args.brickfiles:
        bx = io.Brick(filename)
        if bx.channel not in brick:
            brick[bx.channel] = bx
        else:
            log.error('Channel {} in multiple input files'.format(bx.channel))
	    sys.exit(2)

    filters=brick.keys()
    for fil in filters:
        log.info("Filter found: "+fil)

    #- Coadd individual exposures and combine channels
    #- Full coadd code is a bit slow, so try something quick and dirty for
    #- now to get something going for redshifting
    log.info("Combining individual channels and exposures")
    wave=[]
    for fil in filters:
        wave=np.concatenate([wave,brick[fil].get_wavelength_grid()])
    np.ndarray.sort(wave)
    nwave = len(wave)

    #- flux and ivar arrays to fill for all targets
    #flux = np.zeros((nspec, nwave))
    #ivar = np.zeros((nspec, nwave))
    targetids = brick['b'].get_target_ids()

    fpinfo=open(args.outfile,"w")

    for i, targetid in enumerate(targetids):
        #- wave, flux, and ivar for this target; concatenate
        xwave = list()
        xflux = list()
        xivar = list()

        for channel in filters:
            exp_flux, exp_ivar, resolution, info = brick[channel].get_target(targetid)
            weights = np.sum(exp_ivar, axis=0)
	    ii, = np.where(weights > 0)
            xwave.extend(brick[channel].get_wavelength_grid()[ii])
	    xflux.extend(np.average(exp_flux[:,ii], weights=exp_ivar[:,ii], axis=0))
	    xivar.extend(weights[ii])

        xwave = np.array(xwave)
        xivar = np.array(xivar)
        xflux = np.array(xflux)

        ii = np.argsort(xwave)
	if len(ii)==0:
		continue
	fl, iv = resample_flux(wave, xwave[ii], xflux[ii], xivar[ii])
	s2n=np.median(fl[:-1]*np.sqrt(iv[:-1])/np.sqrt(wave[1:]-wave[:-1]))
	print targetid,s2n
	fpinfo.write(str(targetid)+" "+str(s2n)+"\n")

    fpinfo.close()

