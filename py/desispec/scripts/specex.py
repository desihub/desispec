"""
Run PSF estimation.
"""

from __future__ import absolute_import, division

import sys
import os
import re
import numpy as np

import subprocess as sp

from desispec.pipeline.utils import option_list


def run_frame(imgfile, bootfile, outfile, opts, specmin=0, nspec=500, bundlesize=25, comm=None):

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
        print "specex:  input image = {}".format(imgfile)
        print "specex:  bootcalib PSF = {}".format(bootfile)
        print "specex:  output = {}".format(outfile)
        print "specex:  bundlesize = {}".format(bundlesize)

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

        optarray = option_list(opts)
        com.extend(optarray)
        sp.check_call(com)

    if comm is not None:
        comm.barrier()

    if rank == 0:
        outfits = "{}.fits".format(outroot)
        outxml = "{}.xml".format(outroot)
        outspots = "{}-spots.fits".format(outroot)

        com = ['specex_merge_psf']
        com.extend(['--out-fits', outfits])
        com.extend(['--out-xml', outxml])
        com.extend([ "{}_{:02d}.xml".format(outroot, x) for x in bundles ])
        sp.check_call(com)

        com = ['specex_merge_spot']
        com.extend(['--out', outspots])
        com.extend([ "{}_{:02d}-spots.fits".format(outroot, x) for x in bundles ])
        sp.check_call(com)

        com = ['rm', '-f']
        com.extend([ "{}_{:02d}.fits".format(outroot, x) for x in bundles ])
        com.extend([ "{}_{:02d}-spots.fits".format(outroot, x) for x in bundles ])
        com.extend([ "{}_{:02d}.xml".format(outroot, x) for x in bundles ])
        sp.check_call(com)


