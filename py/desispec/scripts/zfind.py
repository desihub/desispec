
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
from desispec.io.qa import load_qa_brick, write_qa_brick
from desispec.util import default_nproc

import argparse


def parse(options=None):
    parser = argparse.ArgumentParser(description="Fit redshifts and classifications on bricks.")
    parser.add_argument("-b", "--brick", type=str, required=False,
        help="input brickname")
    parser.add_argument("-n", "--nspec", type=int, required=False,
        help="number of spectra to fit [default: all]")
    parser.add_argument("-o", "--outfile", type=str, required=False,
        help="output file name")
    parser.add_argument("--specprod_dir", type=str, required=False, default=None, 
        help="override $DESI_SPECTRO_REDUX/$PRODNAME environment variable path")
    parser.add_argument("--objtype", type=str, required=False,
        help="only use templates for these objtypes (comma separated elg,lrg,qso,star)")
    parser.add_argument('--zrange-galaxy', type=float, default=(0.0, 1.6), nargs=2, 
                        help='minimum and maximum galaxy redshifts to consider')
    parser.add_argument('--zrange-qso', type=float, default=(0.0, 3.5), nargs=2, 
                        help='minimum and maximum QSO redshifts to consider')
    parser.add_argument('--zrange-star', type=float, default=(-0.005, 0.005), nargs=2, 
                        help='minimum and maximum stellar redshifts to consider')
    parser.add_argument("--zspec", action="store_true",
        help="also include spectra in output file")
    parser.add_argument('--qafile', type=str, 
        help='path of QA file.')
    parser.add_argument('--qafig', type=str, 
        help='path of QA figure file')
    parser.add_argument("--nproc", type=int, default=default_nproc,
        help="number of parallel processes for multiprocessing")
    parser.add_argument("brickfiles", nargs="*")

    args = None
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args


#- function for multiprocessing
def _func(arg) :
    return RedMonsterZfind(**arg)


def main(args) :

    log = get_logger()

    if args.objtype is not None:
        args.objtype = args.objtype.split(',')

    #- Read brick files for each channel
    log.info("Reading bricks")
    brick = dict()
    if args.brick is not None:
        if len(args.brickfiles) != 0:
            raise RuntimeError('Give -b/--brick or input brickfiles but not both')
        for channel in ('b', 'r', 'z'):
            filename = io.findfile('brick', band=channel, brickname=args.brick,
                                   specprod_dir=args.specprod_dir)
            brick[channel] = io.Brick(filename)
    else:
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

    #- Assume all channels have the same number of targets
    #- TODO: generalize this to allow missing channels
    if args.nspec is None:
        args.nspec = brick['b'].get_num_targets()
        log.info("Fitting {} targets".format(args.nspec))
    else:
        log.info("Fitting {} of {} targets".format(args.nspec, brick['b'].get_num_targets()))

    nspec = args.nspec

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
    flux = np.zeros((nspec, nwave))
    ivar = np.zeros((nspec, nwave))
    targetids = brick['b'].get_target_ids()[0:nspec]

    func_args = []

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
            #- Average multiple exposures on the same wavelength grid for each channel
            xflux.extend(np.average(exp_flux[:,ii], weights=exp_ivar[:,ii], axis=0))
            xivar.extend(weights[ii])

        xwave = np.array(xwave)
        xivar = np.array(xivar)
        xflux = np.array(xflux)

        ii = np.argsort(xwave)
        flux[i], ivar[i] = resample_flux(wave, xwave[ii], xflux[ii], xivar[ii])

    #- distribute the spectra in nspec groups
    if args.nproc > nspec:
        args.nproc = nspec

    ii = np.linspace(0, nspec, args.nproc+1).astype(int)
    for i in range(args.nproc):
        lo, hi = ii[i], ii[i+1]
        log.debug('CPU {} spectra {}:{}'.format(i, lo, hi))
        arguments = {"wave": wave, "flux": flux[lo:hi], "ivar": ivar[lo:hi],
                     "objtype": args.objtype, "zrange_galaxy": args.zrange_galaxy,
                     "zrange_qso": args.zrange_qso, "zrange_star": args.zrange_star}
        func_args.append( arguments )

    #- Do the redshift fit

    if args.nproc==1 : # No parallelization, simple loop over arguments

        zf = list()
        for arg in func_args:
            zff = RedMonsterZfind(**arg)
            zf.append(zff)

    else: # Multiprocessing
        log.info("starting multiprocessing with {} cpus for {} spectra in {} groups".format(args.nproc, nspec, len(func_args)))
        pool = multiprocessing.Pool(args.nproc)
        zf = pool.map(_func, func_args)
        pool.close()
        pool.join()

    # reformat results
    dtype = list()

    dtype = [
        ('Z',         zf[0].z.dtype),
        ('ZERR',      zf[0].zerr.dtype),
        ('ZWARN',     zf[0].zwarn.dtype),
        ('TYPE',      zf[0].type.dtype),
        ('SUBTYPE',   zf[0].subtype.dtype),    
    ]

    formatted_data = np.empty(nspec, dtype=dtype)

    i = 0
    for result in zf:
        n = result.nspec
        formatted_data['Z'][i:i+n]       = result.z
        formatted_data['ZERR'][i:i+n]    = result.zerr
        formatted_data['ZWARN'][i:i+n]   = result.zwarn
        formatted_data['TYPE'][i:i+n]    = result.type
        formatted_data['SUBTYPE'][i:i+n] = result.subtype
        i += n

    assert i == nspec

    # Create a ZfindBase object with formatted results
    zfi = ZfindBase(None, None, None, results=formatted_data)
    zfi.nspec = nspec

    # QA
    if (args.qafile is not None) or (args.qafig is not None):
        log.info("performing skysub QA")
        # Load
        qabrick = load_qa_brick(args.qafile)
        # Run
        qabrick.run_qa('ZBEST', (zfi,brick))
        # Write
        if args.qafile is not None:
            write_qa_brick(args.qafile, qabrick)
            log.info("successfully wrote {:s}".format(args.qafile))
        # Figure(s)
        if args.qafig is not None:
            raise IOError("Not yet implemented")
            qa_plots.brick_zbest(args.qafig, zfi, qabrick)


    #- Write some output
    if args.outfile is None:
        args.outfile = io.findfile('zbest', brickname=args.brick)

    log.info("Writing "+args.outfile)
    io.write_zbest(args.outfile, args.brick, targetids, zfi, zspec=args.zspec)


