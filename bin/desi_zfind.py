#!/usr/bin/env python
# See top-level LICENSE.rst file for Copyright information

"""
Fit redshifts and classifications on DESI bricks
"""

import sys
import os
import numpy as np

from desispec import io
from desispec.interpolation import resample_flux
from desispec.log import get_logger
from desispec.zfind.redmonster import RedMonsterZfind

import optparse

parser = optparse.OptionParser(usage = "%prog [options]")
parser.add_option("-b", "--brick", type=str,  help="input brickname")
parser.add_option("-n", "--nspec", type=int,  help="number of spectra to fit [default: all]")
parser.add_option("-o", "--outfile", type=str,  help="output file name")
parser.add_option("--zspec",   help="also include spectra in output file", action="store_true")

opts, args = parser.parse_args()

log = get_logger()

#- Read brick files for each channel
log.info("Reading bricks")
brick = dict()
for channel in ('b', 'r', 'z'):
    filename = io.findfile('brick', band=channel, brickid=opts.brick)
    brick[channel] = io.Brick(filename)

#- Assume all channels have the same number of targets
#- TODO: generalize this to allow missing channels
if opts.nspec is None:
    opts.nspec = brick['b'].get_num_targets()
    log.info("Fitting {} targets".format(opts.nspec))
else:
    log.info("Fitting {} of {} targets".format(opts.nspec, brick['b'].get_num_targets()))

nspec = opts.nspec

#- Coadd individual exposures and combine channels
#- Full coadd code is a bit slow, so try something quick and dirty for
#- now to get something going for redshifting
log.info("Combining individual channels and exposures")
wb = brick['b'].get_wavelength_grid()
wr = brick['r'].get_wavelength_grid()
wz = brick['z'].get_wavelength_grid()
wave = np.concatenate([wb, wr, wz])
np.ndarray.sort(wave)
nwave = len(wave)

#- flux and ivar arrays to fill for all targets
flux = np.zeros((nspec, nwave))
ivar = np.zeros((nspec, nwave))
targetids = brick['b'].get_target_ids()[0:nspec]

for i, targetid in enumerate(targetids):
    #- wave, flux, and ivar for this target; concatenate
    xwave = list()
    xflux = list()
    xivar = list()
    for channel in ('b', 'r', 'z'):
        exp_flux, exp_ivar, resolution, info = brick[channel].get_target(targetid)
        weights = np.sum(exp_ivar, axis=0)
        ii, = np.where(weights > 0)
        xwave.extend(brick[channel].get_wavelength_grid()[ii])
        #- Average multiple exposures on the same wavelength grid for each channel
        xflux.extend(np.average(exp_flux[:,ii], weights=exp_ivar[:,ii], axis=0))
        xivar.extend(weights[ii])

    xwave = np.array(xwave)
    xivar = np.array(xivar)
    xflux = np.array(xflux)

    ii = np.argsort(xwave)
    flux[i], ivar[i] = resample_flux(wave, xwave[ii], xflux[ii], xivar[ii])

#- Do the redshift fit
zf = RedMonsterZfind(wave, flux, ivar)

#- Write some output
if opts.outfile is None:
    opts.outfile = io.findfile('zbest', brickid=opts.brick)

log.info("Writing "+opts.outfile)
io.write_zbest(opts.outfile, opts.brick, targetids, zf, zspec=opts.zspec)
