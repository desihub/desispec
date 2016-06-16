"""
Run PSF estimation.
"""

from __future__ import print_function, absolute_import, division

import sys
import os
import re
import argparse
import numpy as np

import subprocess as sp

from desispec.log import get_logger


def parse(options=None):
    parser = argparse.ArgumentParser(description="Estimate the PSF for one frame with specex")
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="input image")
    parser.add_argument("-b", "--bootfile", type=str, required=True,
                        help="input bootcalib psf file")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="output extracted spectra")
    parser.add_argument("--bundlesize", type=int, required=False, default=25,
                        help="number of spectra per bundle")
    parser.add_argument("-s", "--specmin", type=int, required=False, default=0,
                        help="first spectrum to extract")
    parser.add_argument("-n", "--nspec", type=int, required=False, default=500,
                        help="number of spectra to extract")
    parser.add_argument("--extra", type=str, required=False, default=None,
                        help="quoted string of arbitrary options to pass to specex_desi_psf_fit")
    parser.add_argument("-v", "--verbose", action="store_true", help="print more stuff")

    args = None
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args


def main(args, comm=None):

    log = get_logger()

    imgfile = args.input
    outfile = args.output
    bootfile = args.bootfile

    optarray = []
    if args.extra is not None:
        optarray = args.extra.split()

    specmin = int(args.specmin)
    nspec = int(args.nspec)
    bundlesize = int(args.bundlesize)

    verbose = args.verbose

    specmax = specmin + nspec

    # Now we divide our spectra into bundles

    checkbundles = set()
    checkbundles.update(np.floor_divide(np.arange(specmin, specmax), bundlesize*np.ones(nspec)).astype(int))
    bundles = sorted(list(checkbundles))
    nbundle = len(bundles)

    bspecmin = {}
    bnspec = {}
    for b in bundles:
        if specmin > b * bundlesize:
            bspecmin[b] = specmin
        else:
            bspecmin[b] = b * bundlesize
        if (b+1) * bundlesize > specmax:
            bnspec[b] = specmax - bspecmin[b]
        else:
            bnspec[b] = bundlesize

    # Now we assign bundles to processes

    nproc = 1
    rank = 0
    if comm is not None:
        nproc = comm.size
        rank = comm.rank

    mynbundle = int(nbundle / nproc)
    myfirstbundle = 0
    leftover = nbundle % nproc
    if rank < leftover:
        mynbundle += 1
        myfirstbundle = rank * mynbundle
    else:
        myfirstbundle = ((mynbundle + 1) * leftover) + (mynbundle * (rank - leftover))

    if rank == 0:
        # Print parameters
        log.info("specex:  using {} processes".format(nproc))
        log.info("specex:  input image = {}".format(imgfile))
        log.info("specex:  bootcalib PSF = {}".format(bootfile))
        log.info("specex:  output = {}".format(outfile))
        log.info("specex:  bundlesize = {}".format(bundlesize))
        log.info("specex:  specmin = {}".format(specmin))
        log.info("specex:  specmax = {}".format(specmax))

    # get the root output file

    outpat = re.compile(r'(.*)\.fits')
    outmat = outpat.match(outfile)
    if outmat is None:
        raise RuntimeError("specex output file should have .fits extension")
    outroot = outmat.group(1)

    outdir = os.path.dirname(outroot)
    if rank == 0:
        if not os.path.isdir(outdir):
            os.makedirs(outdir)

    failcount = 0

    for b in range(myfirstbundle, myfirstbundle+mynbundle):
        outbundle = "{}_{:02d}".format(outroot, b)
        outbundlefits = "{}.fits".format(outbundle)
        outbundlexml = "{}.xml".format(outbundle)
        outbundlespot = "{}-spots.fits".format(outbundle)

        com = ['specex_desi_psf_fit']
        com.extend(['-a', imgfile])
        com.extend(['--xcoord-file', bootfile])
        com.extend(['--ycoord-file', bootfile])
        com.extend(['--out_xml', outbundlexml])
        com.extend(['--out_fits', outbundlefits])
        com.extend(['--out_spots', outbundlespot])
        com.extend(['--first_bundle', "{}".format(b)])
        com.extend(['--last_bundle', "{}".format(b)])
        if verbose:
            com.extend(['--verbose'])

        com.extend(optarray)

        log.debug("proc {} spawning {}".format(rank, " ".join(com)))

        proc = sp.Popen(com, bufsize=8192)
        outs, errs = proc.communicate()
        retval = proc.returncode

        if retval != 0:
            comstr = " ".join(com)
            log.error("specex_desi_psf_fit on process {} failed with return value {} running {}".format(rank, retval, comstr))
            failcount += 1

    if comm is not None:
        failcount = comm.allreduce(failcount)

    if failcount > 0:
        # all processes throw
        raise RuntimeError("some bundles failed specex_desi_psf_fit")

    if comm is not None:
        comm.barrier()

    failcount = 0

    if rank == 0:
        outfits = "{}.fits".format(outroot)
        outxml = "{}.xml".format(outroot)
        outspots = "{}-spots.fits".format(outroot)

        com = ['specex_merge_psf']
        com.extend(['--out-fits', outfits])
        com.extend(['--out-xml', outxml])
        com.extend([ "{}_{:02d}.xml".format(outroot, x) for x in bundles ])

        proc = sp.Popen(com, bufsize=8192)
        outs, errs = proc.communicate()
        retval = proc.returncode
        if retval != 0:
            comstr = " ".join(com)
            log.error("specex_merge_psf failed with return value {} running {}".format(retval, comstr))
            failcount += 1

        com = ['specex_merge_spot']
        com.extend(['--out', outspots])
        com.extend([ "{}_{:02d}-spots.fits".format(outroot, x) for x in bundles ])

        proc = sp.Popen(com, bufsize=8192)
        outs, errs = proc.communicate()
        retval = proc.returncode
        if retval != 0:
            comstr = " ".join(com)
            log.error("specex_merge_spot failed with return value {} running {}".format(retval, comstr))
            failcount += 1

        com = ['rm', '-f']
        com.extend([ "{}_{:02d}.fits".format(outroot, x) for x in bundles ])
        com.extend([ "{}_{:02d}-spots.fits".format(outroot, x) for x in bundles ])
        com.extend([ "{}_{:02d}.xml".format(outroot, x) for x in bundles ])

        if failcount == 0:
            # only remove the per-bundle files if the merge was good
            proc = sp.Popen(com, bufsize=8192)
            outs, errs = proc.communicate()
            retval = proc.returncode
            if retval != 0:
                comstr = " ".join(com)
                log.error("removal of per-bundle files failed with return value {} running {}".format(retval, comstr))
                failcount += 1

    failcount = comm.bcast(failcount, root=0)

    if failcount > 0:
        # all processes throw
        raise RuntimeError("merging of per-bundle files failed")

    return
