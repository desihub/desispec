"""
Run PSF estimation.
"""

from __future__ import print_function, absolute_import, division

import sys
import os
import re
import argparse
import numpy as np

import ctypes as ct
from ctypes.util import find_library

from astropy.io import fits

from desispec.log import get_logger


libspecex = None
try:
    libspecex = ct.CDLL('libspecex.so')
except:
    path = find_library('specex')
    if path is not None:
        libspecex = ct.CDLL(path)


if libspecex is not None:
    libspecex.cspecex_desi_psf_fit.restype = ct.c_int
    libspecex.cspecex_desi_psf_fit.argtypes = [
        ct.c_int,
        ct.POINTER(ct.POINTER(ct.c_char))
    ]
    libspecex.cspecex_psf_merge.restype = ct.c_int
    libspecex.cspecex_psf_merge.argtypes = [
        ct.c_int,
        ct.POINTER(ct.POINTER(ct.c_char))
    ]
    libspecex.cspecex_spot_merge.restype = ct.c_int
    libspecex.cspecex_spot_merge.argtypes = [
        ct.c_int,
        ct.POINTER(ct.POINTER(ct.c_char))
    ]


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

        log.debug("proc {} calling {}".format(rank, " ".join(com)))

        argc = len(com)
        arg_buffers = [ct.create_string_buffer(com[i]) for i in range(argc)]
        addrlist = [ ct.cast(x, ct.POINTER(ct.c_char)) for x in map(ct.addressof, arg_buffers) ]
        arg_pointers = (ct.POINTER(ct.c_char) * argc)(*addrlist)

        retval = libspecex.cspecex_desi_psf_fit(argc, arg_pointers)

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

        argc = len(com)
        arg_buffers = [ct.create_string_buffer(com[i]) for i in range(argc)]
        addrlist = [ ct.cast(x, ct.POINTER(ct.c_char)) for x in map(ct.addressof, arg_buffers) ]
        arg_pointers = (ct.POINTER(ct.c_char) * argc)(*addrlist)

        retval = libspecex.cspecex_psf_merge(argc, arg_pointers)

        if retval != 0:
            comstr = " ".join(com)
            log.error("specex_merge_psf failed with return value {} running {}".format(retval, comstr))
            failcount += 1

        com = ['specex_merge_spot']
        com.extend(['--out', outspots])
        com.extend([ "{}_{:02d}-spots.fits".format(outroot, x) for x in bundles ])

        argc = len(com)
        arg_buffers = [ct.create_string_buffer(com[i]) for i in range(argc)]
        addrlist = [ ct.cast(x, ct.POINTER(ct.c_char)) for x in map(ct.addressof, arg_buffers) ]
        arg_pointers = (ct.POINTER(ct.c_char) * argc)(*addrlist)

        retval = libspecex.cspecex_spot_merge(argc, arg_pointers)

        if retval != 0:
            comstr = " ".join(com)
            log.error("specex_merge_spot failed with return value {} running {}".format(retval, comstr))
            failcount += 1

        com = []
        com.extend([ "{}_{:02d}.fits".format(outroot, x) for x in bundles ])
        com.extend([ "{}_{:02d}-spots.fits".format(outroot, x) for x in bundles ])
        com.extend([ "{}_{:02d}.xml".format(outroot, x) for x in bundles ])

        if failcount == 0:
            # only remove the per-bundle files if the merge was good
            for f in com:
                if os.path.isfile(f):
                    os.remove(f)

    failcount = comm.bcast(failcount, root=0)

    if failcount > 0:
        # all processes throw
        raise RuntimeError("merging of per-bundle files failed")

    return


def compatible(head1, head2) :
    log = get_logger()
    for k in ["PSFTYPE","NPIX_X","NPIX_Y","HSIZEX","HSIZEY","BUNDLMIN","BUNDLMAX","FIBERMAX","FIBERMIN","FIBERMAX","NPARAMS","LEGDEG","GHDEGX","GHDEGY"] :
        if (head1[k] != head2[k]) :
            log.warning("different {} : {}, {}".format(k, head1[k], head2[k]))
            return False
    return True


def mean_psf(inputs, output):

    log = get_logger()

    npsf = len(inputs)
    log.info("Will compute the average of {} PSFs".format(npsf))

    refhead = None
    tables = []
    hdulist = None
    bundle_rchi2 = []
    nbundles = None
    nfibers_per_bundle = None

    for input in inputs :
        psf = fits.open(input)
        if refhead is None :
            hdulist = psf
            refhead = psf[1].header            
            nbundles = (psf[1].header["BUNDLMAX"]-psf[1].header["BUNDLMIN"])+1
            nfibers = (psf[1].header["FIBERMAX"]-psf[1].header["FIBERMIN"])+1
            nfibers_per_bundle = nfibers/nbundles
            log.debug("nbundles = {}".format(nbundles))
            log.debug("nfibers_per_bundle = {}".format(nfibers_per_bundle))
        else :
            if not compatible(psf[1].header,refhead) :
                raise RuntimeError("psfs {} and {} are not compatible".format(inputs[0], input))
        tables.append(psf[1].data)
        
        rchi2 = np.zeros(nbundles)
        for b in range(nbundles) :
            rchi2[b] = psf[1].header["B{:02d}RCHI2".format(b)]
        bundle_rchi2.append(rchi2)
    
    bundle_rchi2 = np.array(bundle_rchi2)
    log.debug("bundle_rchi2 = {}".format(bundle_rchi2))

    for entry in range(tables[0].size):
        PARAM = tables[0][entry]["PARAM"]
        log.info("Averaging {} coefficients".format(PARAM))
        # check WAVEMIN WAVEMAX compatibility
        WAVEMIN = tables[0][entry]["WAVEMIN"]
        WAVEMAX = tables[0][entry]["WAVEMAX"]
        
        # for p in range(1,npsf) :
        #     if tables[p][entry]["WAVEMIN"] != WAVEMIN :
        #         log.error("WAVEMIN not compatible for param %s : %f!=%f"%(PARAM,tables[p][entry]["WAVEMIN"],WAVEMIN)) 
        #         sys.exit(12)
        #     if tables[p][entry]["WAVEMAX"] != WAVEMAX :
        #         log.error("WAVEMAX not compatible for param %s : %f!=%f"%(PARAM,tables[p][entry]["WAVEMAX"],WAVEMAX))
        #         sys.exit(12)
        
        # will need to readdress coefs ...         
        coeff = [tables[0][entry]["COEFF"]]
        npar = coeff[0][1].size
        for p in range(1, npsf) :
            if tables[p][entry]["WAVEMIN"] == WAVEMIN and tables[p][entry]["WAVEMAX"] == WAVEMAX:
                coeff.append(tables[p][entry]["COEFF"])
            else:
                icoeff = tables[p][entry]["COEFF"]
                ocoeff = np.zeros(icoeff.shape)
                # need to reshape legpol
                iu = np.linspace(-1,1,npar+3)
                iwavemin = tables[p][entry]["WAVEMIN"]
                iwavemax = tables[p][entry]["WAVEMAX"]
                wave = (iu+1.)/2.*(iwavemax-iwavemin)+iwavemin
                ou = (wave-WAVEMIN)/(WAVEMAX-WAVEMIN)*2.-1.                
                for f in range(icoeff.shape[0]) :
                    val = legval(iu,icoeff[f])
                    ocoeff[f] = legfit(ou,val,deg=npar-1)
                #print ""
                #print icoeff[2]
                #print ocoeff[2]
                coeff.append(ocoeff)
        coeff = np.array(coeff)
        
        output_rchi2 = np.zeros((bundle_rchi2.shape[1]))
        output_coeff = np.zeros(tables[0][entry]["COEFF"].shape)
        
        #log.info("input coeff.shape  = %d"%coeff.shape)
        #log.info("output coeff.shape = %d"%output_coeff.shape)
        
        # now merge, using rchi2 as selection score
        rchi2_threshold = 2.0
        for bundle in range(bundle_rchi2.shape[1]) :
            
            ok = np.where(bundle_rchi2[:,bundle] < rchi2_threshold)[0]
            #ok=np.array([0,1]) # debug
            if entry == 0:
                log.info("for fiber bundle {}, {} valid PSFs".format(bundle, ok.size))
            
            fibers = np.arange(bundle*nfibers_per_bundle,(bundle+1)*nfibers_per_bundle)
            if ok.size >= 2: # use median
                for f in fibers :
                    output_coeff[f] = np.median(coeff[ok,f],axis=0)
                output_rchi2[bundle] = np.median(bundle_rchi2[ok,bundle])
            elif ok.size == 1: # copy
                for f in fibers :
                    output_coeff[f] = coeff[ok[0],f]
                output_rchi2[bundle] = bundle_rchi2[ok[0],bundle]
                    
            else: # we have a problem here, take the smallest rchi2
                i = np.argmin(bundle_rchi2[:,bundle])
                for f in fibers :
                    output_coeff[f] = coeff[i,f]
                output_rchi2[bundle] = bundle_rchi2[i,bundle]
        
        # now copy this in output table
        hdulist[1].data["COEFF"][entry] = output_coeff
        # change bundle chi2
        for bundle in range(output_rchi2.size) :
            hdulist[1].header["B{:02d}RCHI2".format(bundle)] = output_rchi2[bundle]
        
        # alter other keys in header
        hdulist[1].header["EXPID"] = 0.0 # it's a mix , need to add the expids here
        
    # save output PSF
    hdulist.writeto(output, clobber=True)
    log.info("wrote {}".format(output))

    return
