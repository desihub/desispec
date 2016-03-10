#!/usr/bin/env python
#
# See top-level LICENSE.rst file for Copyright information
#
# -*- coding: utf-8 -*-

"""
This script runs bootcalib scripts for one spectrograph given a flat, arc combination
"""

import pdb
import numpy as np
from desispec.log import get_logger
from desispec import bootcalib as desiboot
from desiutil import funcfits as dufits
from matplotlib.backends.backend_pdf import PdfPages

import argparse

from astropy.io import fits

def main() :

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--fiberflat', type = str, default = None, required=False,
                        help = 'path of DESI fiberflat fits file')
    parser.add_argument('--psffile', type = str, default = None, required=False,
                        help = 'path of DESI PSF fits file')
    parser.add_argument('--arcfile', type = str, default = None, required=False,
                        help = 'path of DESI fiberflat fits file')
    parser.add_argument('--outfile', type = str, default = None, required=True,
                        help = 'path of DESI sky fits file')
    parser.add_argument('--qafile', type = str, default = None, required=False,
                        help = 'path of QA figure file')
    parser.add_argument("--test", help="Debug?", default=False, action="store_true")
    parser.add_argument("--debug", help="Debug?", default=False, action="store_true")
    parser.add_argument("--trace_only", help="Quit after tracing?", default=False, action="store_true")
    parser.add_argument("--legendre-degree", type = int, default=6, required=False, help="Legendre polynomial degree for traces")

    args = parser.parse_args()
    log=get_logger()

    log.info("starting")

    if (args.psffile is None) and (args.fiberflat is None):
        raise IOError("Must provide either a PSF file or a fiberflat")

    # Start QA
    try:
        pp = PdfPages(args.qafile)
    except ValueError:
        QA = False
    else:
        QA = True

    if args.psffile is None:
        ###########
        # Read flat
        flat_hdu = fits.open(args.fiberflat)
        header = flat_hdu[0].header
        flat = flat_hdu[0].data
        ny = flat.shape[0]

        ###########
        # Find fibers
        log.info("finding the fibers")
        xpk, ypos, cut = desiboot.find_fiber_peaks(flat)
        if QA:
            desiboot.qa_fiber_peaks(xpk, cut, pp)

        # Test?
        if args.test:
            log.warning("cutting down fibers for testing..")
            xpk = xpk[87:95]
            #xpk = xpk[0:5]

        ###########
        # Trace the fiber flat spectra
        log.info("tracing the fiber flat spectra")
        # Crude first
        log.info("crudely..")
        xset, xerr = desiboot.trace_crude_init(flat,xpk,ypos)
        # Polynomial fits
        log.info("fitting the traces")
        xfit, fdicts = desiboot.fit_traces(xset,xerr)
        # QA
        if QA:
            desiboot.qa_fiber_Dx(xfit, fdicts, pp)

        ###########
        # Model the PSF with Gaussian
        log.info("modeling the PSF with a Gaussian, be patient..")
        gauss = desiboot.fiber_gauss(flat,xfit,xerr)
        if QA:
            desiboot.qa_fiber_gauss(gauss, pp)
        XCOEFF = None
    else: # Load PSF file and generate trace info
        log.warning("Not tracing the flat.  Using the PSF file.")
        psf_hdu = fits.open(args.psffile)
        psf_head = psf_hdu[0].header
        # Gaussians
        gauss = psf_hdu[2].data
        # Traces
        WAVEMIN = psf_head['WAVEMIN']
        WAVEMAX = psf_head['WAVEMAX']
        XCOEFF = psf_hdu[0].data
        xfit = None
        fdicts = None

    # ARCS
    if not args.trace_only:

        ###########
        # Read arc
        log.info("reading arc")
        arc_hdu = fits.open(args.arcfile)
        arc = arc_hdu[0].data
        header = arc_hdu[0].header
        ny = arc.shape[0]

        #####################################
        # Extract arc spectra (one per fiber)
        log.info("extracting arcs")
        if xfit is None:
            wv_array = np.linspace(WAVEMIN, WAVEMAX, num=arc.shape[0])
            nfiber = XCOEFF.shape[0]
            ncoeff = XCOEFF.shape[1]
            xfit = np.zeros((arc.shape[0], nfiber))
            # Generate a fit_dict
            fit_dict = dufits.mk_fit_dict(XCOEFF[:,0], ncoeff, 'legendre', WAVEMIN, WAVEMAX)
            for ii in range(nfiber):
                fit_dict['coeff'] = XCOEFF[ii,:]
                xfit[:,ii] = dufits.func_val(wv_array, fit_dict)


        all_spec = desiboot.extract_sngfibers_gaussianpsf(arc, xfit, gauss)

        ############################
        # Line list
        camera = header['CAMERA']
        log.info("Loading line list")
        llist = desiboot.load_arcline_list(camera)
        dlamb, wmark, gd_lines, line_guess = desiboot.load_gdarc_lines(camera)

        #####################################
        # Loop to solve for wavelengths
        all_wv_soln = []
        all_dlamb = []
        debug=False
        for ii in range(all_spec.shape[1]):
            spec = all_spec[:,ii]
            if (ii % 20) == 0:
                log.info("working on spectrum {:d}".format(ii))
            # Find Lines
            pixpk = desiboot.find_arc_lines(spec)
            # Match a set of 5 gd_lines to detected lines
            id_dict = desiboot.id_arc_lines(pixpk, gd_lines, dlamb, wmark, line_guess=line_guess)
            id_dict['fiber'] = ii
            # Find the other good ones
            if camera == 'z':
                inpoly = 3  # The solution in the z-camera has greater curvature
            else:
                inpoly = 2
            desiboot.add_gdarc_lines(id_dict, pixpk, gd_lines, inpoly=inpoly, debug=debug)
            # Now the rest
            desiboot.id_remainder(id_dict, pixpk, llist)
            # Final fit wave vs. pix too
            final_fit, mask = dufits.iter_fit(np.array(id_dict['id_wave']), np.array(id_dict['id_pix']), 'polynomial', 3, xmin=0., xmax=1.)
            rms = np.sqrt(np.mean((dufits.func_val(np.array(id_dict['id_wave'])[mask==0], final_fit)-np.array(id_dict['id_pix'])[mask==0])**2))
            final_fit_pix,mask2 = dufits.iter_fit(np.array(id_dict['id_pix']), np.array(id_dict['id_wave']),'legendre',args.legendre_degree , niter=5)
            # Check RMS and dispersion
            wave = dufits.func_val(np.arange(spec.size),final_fit_pix)
            dlamb = np.median(np.abs(wave-np.roll(wave,1)))
            if ii > 0:
                med_dlamb = np.median(all_dlamb)
                if (np.abs(dlamb - med_dlamb)/med_dlamb > 0.1) or (rms > 0.7):
                    log.warn('Bad wavelength solution.  Using previous to guide..')
                    # Bad solution; shifting to previous
                    desiboot.use_previous_wave(id_dict, sv_iddict, pixpk, sv_pixpk)
                    final_fit, mask = dufits.iter_fit(np.array(id_dict['id_wave']), np.array(id_dict['id_pix']), 'polynomial', 3, xmin=0., xmax=1.)
                    rms = np.sqrt(np.mean((dufits.func_val(np.array(id_dict['id_wave'])[mask==0], final_fit)-np.array(id_dict['id_pix'])[mask==0])**2))
                    final_fit_pix,mask2 = dufits.iter_fit(np.array(id_dict['id_pix']), np.array(id_dict['id_wave']),'legendre',args.legendre_degree , niter=5)
                    wave = dufits.func_val(np.arange(spec.size),final_fit_pix)
                    dlamb = np.median(np.abs(wave-np.roll(wave,1)))
                    #from xastropy.xutils import xdebug as xdb
                    #xdb.set_trace()
            all_dlamb.append(dlamb)
            # Save
            id_dict['final_fit'] = final_fit
            id_dict['rms'] = rms
            id_dict['final_fit_pix'] = final_fit_pix
            id_dict['wave_min'] = dufits.func_val(0,final_fit_pix)
            id_dict['wave_max'] = dufits.func_val(ny-1,final_fit_pix)
            id_dict['mask'] = mask
            all_wv_soln.append(id_dict)
            # Save for next fiber
            sv_pixpk = pixpk
            sv_iddict = id_dict

        if QA:
            desiboot.qa_arc_spec(all_spec, all_wv_soln, pp)
            desiboot.qa_fiber_arcrms(all_wv_soln, pp)
            desiboot.qa_fiber_dlamb(all_spec, all_wv_soln, pp)
    else:
        all_wv_soln = None

    ###########
    # Write PSF file
    log.info("writing PSF file")
    desiboot.write_psf(args.outfile, xfit, fdicts, gauss, all_wv_soln, ncoeff=args.legendre_degree , without_arc=args.trace_only,
                       XCOEFF=XCOEFF)
    log.info("successfully wrote {:s}".format(args.outfile))

    ###########
    # All done
    if QA:
        log.info("successfully wrote {:s}".format(args.qafile))
        pp.close()
    log.info("finishing..")


if __name__ == '__main__':
    main()
