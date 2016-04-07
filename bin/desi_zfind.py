#!/usr/bin/env python
# See top-level LICENSE.rst file for Copyright information

"""
Fit redshifts and classifications on DESI bricks
"""

import sys
import os
import numpy as np
import multiprocessing

from desispec import io
from desispec.interpolation import resample_flux
from desispec.log import get_logger
from desispec.zfind.redmonster import RedMonsterZfind
from desispec.zfind import ZfindBase

import optparse

parser = optparse.OptionParser(usage = "%prog [options] [brickfile-b brickfile-r brickfile-z]")
parser.add_option("-b", "--brick", type=str,  help="input brickname")
parser.add_option("-n", "--nspec", type=int,  help="number of spectra to fit [default: all]")
parser.add_option("-o", "--outfile", type=str,  help="output file name")
parser.add_option(      "--objtype", type=str,  help="only use templates for these objtypes (comma separated elg,lrg,qso,star)")
parser.add_option("--zspec",   help="also include spectra in output file", action="store_true")
parser.add_option("--ncpu", type = int, default = 1, help = "number of cores for multiprocessing")

opts, brickfiles = parser.parse_args()

#- function for multiprocessing
def _func(arg) :
    return RedMonsterZfind(**arg)

log = get_logger()

if opts.objtype is not None:
    opts.objtype = opts.objtype.split(',')

#- Read brick files for each channel
log.info("Reading bricks")
brick = dict()
if opts.brick is not None:
    if len(brickfiles) != 0:
        log.error('Give -b/--brick or input brickfiles but not both')
        sys.exit(1)
        
    for channel in ('b', 'r', 'z'):
        filename = io.findfile('brick', band=channel, brickname=opts.brick)
        brick[channel] = io.Brick(filename)
else:
    for filename in brickfiles:
        bx = io.Brick(filename)
        if bx.channel not in brick:
            brick[bx.channel] = bx
        else:
            log.error('Channel {} in multiple input files'.format(bx.channel))
            sys.exit(2)
            
assert set(brick.keys()) == set(['b', 'r', 'z'])

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

func_args = []

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

# sequence of args for multiprocessing
for i in range(nspec):
    arguments={"wave":wave, "flux":flux[i:i+1],"ivar":ivar[i:i+1],"objtype":opts.objtype}
    func_args.append( arguments )

#- Do the redshift fit

if opts.ncpu==1 : # No parallelization, simple loop over arguments

    zf = list()
    for arg in func_args:
        zff = RedMonsterZfind(**arg)
        zf.append(zff)

else: # Multiprocessing
    log.info("starting multiprocessing with %d cpus"%opts.ncpu)
    
    pool = multiprocessing.Pool(opts.ncpu)
    zf =  pool.map(_func, func_args)

# reformat results
dtype = list()

dtype = [
    ('Z',         zf[0].z.dtype),
    ('ZERR',      zf[0].zerr.dtype),
    ('ZWARN',     zf[0].zwarn.dtype),
    ('TYPE',      zf[0].type.dtype),
    ('SUBTYPE',   zf[0].subtype.dtype),    
]

formatted_data = np.empty(len(zf), dtype=dtype)

for i in range(nspec):
    formatted_data['Z'][i]         = zf[i].z
    formatted_data['ZERR'][i]      = zf[i].zerr
    formatted_data['ZWARN'][i]     = zf[i].zwarn
    formatted_data['TYPE'][i]      = zf[i].type[0]
    formatted_data['SUBTYPE'][i]   = zf[i].subtype[0]


# Create a ZfinBase object with formatted results
zfi = ZfindBase(None, None, None, results=formatted_data)
zfi.nspec = nspec

#- Write some output
if opts.outfile is None:
    opts.outfile = io.findfile('zbest', brickname=opts.brick)

log.info("Writing "+opts.outfile)
io.write_zbest(opts.outfile, opts.brick, targetids, zfi, zspec=opts.zspec)
