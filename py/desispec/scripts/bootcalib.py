"""
desispec.bootcalib
==================

Utility functions to perform a quick calibration of DESI data

TODO:
1. Expand to r, i cameras
2. QA plots
3. Test with CR data
"""
from __future__ import print_function, absolute_import, division, unicode_literals

import numpy as np
from desispec.log import get_logger
from desispec import bootcalib as desiboot
from desiutil import funcfits as dufits

from desispec.util import set_backend
set_backend()

from matplotlib.backends.backend_pdf import PdfPages
import sys

import argparse

from astropy.io import fits


def parse(options=None):
    parser = argparse.ArgumentParser(description="Bootstrap DESI PSF.")

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
    parser.add_argument('--lamps', type = str, default = None, required=False,
                        help = 'comma-separated used lamp elements, ex: HgI,NeI,ArI,CdI,KrI')
    parser.add_argument("--test", help="Debug?", default=False, action="store_true")
    parser.add_argument("--debug", help="Debug?", default=False, action="store_true")
    parser.add_argument("--trace_only", help="Quit after tracing?", default=False, action="store_true")
    parser.add_argument("--legendre-degree", type = int, default=6, required=False, help="Legendre polynomial degree for traces")
    parser.add_argument("--triplet-matching", default=False, action="store_true", help="use triplet matching method for line identification (slower but expected more robust)")
    parser.add_argument("--ntrack", type = int, default=5, required=False, help="Number of solutions to be tracked (only used with triplet-matching, more is safer but slower)")
    
    args = None
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args


def main(args):
    
    log=get_logger()

    log.info("starting")

    if args.triplet_matching :
        log.warning("triplet_matching option deprecated, this algorithm is now used for all cases")
    

    lamps=None
    if args.lamps :
        lamps=np.array(args.lamps.split(","))
        log.info("Using lamps = %s"%str(lamps))
    else :
        log.info("Using default set of lamps")
    
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
        if len(flat_hdu)>=3 :
            flat = flat_hdu[0].data*(flat_hdu[1].data>0)*(flat_hdu[2].data==0)
        else :
            flat = flat_hdu[0].data
            log.warning("found only %d HDU in flat, do not use ivar"%len(flat_hdu))
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
            #xpk = xpk[0:100]
            xpk = xpk[0:50]
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
        if len(arc_hdu)>=3 :
            # set to zero ivar of masked pixels, force positive or null ivar
            arc_ivar = arc_hdu[1].data*(arc_hdu[2].data==0)*(arc_hdu[1].data>0)
            # and mask pixels below -5 sigma (cures unmasked dead columns in sims.)
            arc_ivar *= (arc_hdu[0].data*np.sqrt(arc_hdu[1].data)>-5.)
            # set to zero pixel values with null ivar              
            arc = arc_hdu[0].data*(arc_ivar>0)
        else :
            arc = arc_hdu[0].data
            arc_ivar = np.ones(arc.shape)
            log.warning("found only %d HDU in arc, do not use ivar"%len(arc_hdu))
        
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
        
        all_spec = desiboot.extract_sngfibers_gaussianpsf(arc, arc_ivar, xfit, gauss)

        ############################
        # Line list
        camera = header['CAMERA'].lower()
        log.info("Loading line list")
        llist = desiboot.load_arcline_list(camera,vacuum=True,lamps=lamps)
        dlamb, gd_lines = desiboot.load_gdarc_lines(camera,llist,vacuum=True,lamps=lamps)
        
        #####################################
        # Loop to solve for wavelengths
        all_wv_soln = []
        all_dlamb = []
        debug=False

        id_dict_of_fibers=[]
        # first loop to find arc lines and do a first matching
        for ii in range(all_spec.shape[1]):
            spec = all_spec[:,ii]
            if (ii % 20) == 0:
                log.info("working on spectrum {:d}".format(ii))
            pixpk = desiboot.find_arc_lines(spec)
            
            try:
                id_dict = desiboot.id_arc_lines_using_triplets(pixpk, gd_lines, dlamb,ntrack=args.ntrack)
            except : 
                log.warn(sys.exc_info())
                log.warn("ID_ARC failed on fiber {:d}".format(ii))
                id_dict = dict(status='junk')
                id_dict['rms'] = 999.
                id_dict['id_wave'] = []
                id_dict['id_pix'] = []
                id_dict['id_idx'] = []
                id_dict["fit"] = None
                id_dict["fiber"] = ii
                id_dict_of_fibers.append(id_dict)
                continue
            
            # Add to dict
            id_dict['fiber'] = ii
            
            # Add lines 
            if len(id_dict['pixpk'])>len(id_dict['id_pix']) :
                desiboot.id_remainder(id_dict, llist, deg=args.legendre_degree)

            # Save
            id_dict_of_fibers.append(id_dict)

        
       
        
        # now record the list of waves identified in several fibers 
        matched_lines=np.array([])
        for ii in range(all_spec.shape[1]):
            matched_lines = np.append(matched_lines,id_dict_of_fibers[ii]['id_wave'])
        matched_lines = np.unique(matched_lines)
        number_of_detections = []
        for line in matched_lines :
            ndet=0
            for ii in range(all_spec.shape[1]):
                if np.sum(id_dict_of_fibers[ii]['id_wave']==line) >0 :
                    ndet += 1
            print(line,"ndet=",ndet)
            number_of_detections.append(ndet)
        
        # choose which lines are ok and 
        # ok if 5 detections (coincidental error very low)
        min_number_of_detections=min(5,all_spec.shape[1])
        number_of_detections=np.array(number_of_detections)
        good_matched_lines = matched_lines[number_of_detections>=min_number_of_detections]
        bad_matched_lines  = matched_lines[number_of_detections<min_number_of_detections]
        
        log.info("good_matched_lines = {:s}".format(str(good_matched_lines)))
        log.info("bad_matched_lines  = {:s}".format(str(bad_matched_lines)))
        
        
        # loop again on all fibers
        for ii in range(all_spec.shape[1]):
            spec = all_spec[:,ii]
            if (ii % 20) == 0:
                log.info("working on spectrum {:d}".format(ii))

            id_dict = id_dict_of_fibers[ii]
            n_matched_lines=len(id_dict['id_wave'])
            n_detected_lines=len(id_dict['pixpk'])
            
            # did we find any bad line for this fiber?
            n_bad = np.intersect1d(id_dict['id_wave'],bad_matched_lines).size
            # how many good lines did we find
            n_good = np.intersect1d(id_dict['id_wave'],good_matched_lines).size
            

            if id_dict['status']=="ok" and ( n_bad>0 or n_good < good_matched_lines.size-1 ) :
                log.info("Try to refit fiber {:d} with n_bad={:d} and n_good={:d} when n_good_all={:d}".format(ii,n_bad,n_good,good_matched_lines.size))
                pixpk = id_dict['pixpk']
                try:
                    id_dict = desiboot.id_arc_lines_using_triplets(pixpk,  good_matched_lines, dlamb,ntrack=args.ntrack)
                except:
                    log.warn(sys.exc_info())
                    log.warn("ID_ARC failed on fiber {:d}".format(ii))
                    id_dict = dict(status='junk')
                    id_dict['rms'] = 999.
                if id_dict['status']=="ok" and  len(id_dict['pixpk'])>len(id_dict['id_pix']) :
                    desiboot.id_remainder(id_dict, llist, deg=args.legendre_degree)
            else :
                log.info("Do not refit fiber {:d} with n_bad={:d} and n_good={:d} when n_good_all={:d}".format(ii,n_bad,n_good,good_matched_lines.size))
                    
            if id_dict['status'] == 'junk':
                all_wv_soln.append(id_dict)
                all_dlamb.append(0.)
                continue

            
            
            # Final fit wave vs. pix too
            id_wave=np.array(id_dict['id_wave'])
            id_pix=np.array(id_dict['id_pix'])
            
            deg=max(1,min(args.legendre_degree,id_wave.size-2))
            
            final_fit, mask = dufits.iter_fit(id_wave,id_pix, 'polynomial', deg, xmin=0., xmax=1., sig_rej=3.)
            rms = np.sqrt(np.mean((dufits.func_val(id_wave[mask==0], final_fit)-id_pix[mask==0])**2))
            final_fit_pix,mask2 = dufits.iter_fit(id_pix[mask==0],id_wave[mask==0],'legendre',deg , sig_rej=100000000.)
            rms_pix = np.sqrt(np.mean((dufits.func_val(id_pix[mask==0], final_fit_pix)-id_wave[mask==0])**2))
            
            # Append
            wave = dufits.func_val(np.arange(spec.size),final_fit_pix)
            idlamb = np.median(np.abs(wave-np.roll(wave,1)))
            all_dlamb.append(idlamb)
            # Save
            id_dict['final_fit'] = final_fit
            id_dict['rms'] = rms
            id_dict['final_fit_pix'] = final_fit_pix
            id_dict['wave_min'] = dufits.func_val(0,final_fit_pix)
            id_dict['wave_max'] = dufits.func_val(ny-1,final_fit_pix)
            id_dict['mask'] = mask
            
            log.info("Fiber #{:d} final fit rms(y->wave) = {:g} A ; rms(wave->y) = {:g} pix ; nlines = {:d}".format(ii,rms,rms_pix,id_pix.size))
    
            all_wv_soln.append(id_dict)

        # Fix solutions with poor RMS (failures)
        # desiboot.fix_poor_solutions(all_wv_soln, all_dlamb, ny, args.legendre_degree)

        if QA:
            desiboot.qa_arc_spec(all_spec, all_wv_soln, pp)
            desiboot.qa_fiber_arcrms(all_wv_soln, pp)
            desiboot.qa_fiber_dlamb(all_spec, all_wv_soln, pp)
    else:
        all_wv_soln = None

    ###########
    # Write PSF file
    log.info("writing PSF file")
    desiboot.write_psf(args.outfile, xfit, fdicts, gauss, all_wv_soln, legendre_deg=args.legendre_degree , without_arc=args.trace_only,
                       XCOEFF=XCOEFF)
    log.info("successfully wrote {:s}".format(args.outfile))

    ###########
    # All done
    if QA:
        log.info("successfully wrote {:s}".format(args.qafile))
        pp.close()
    log.info("finishing..")
    print("finished bootcalib {}".format(args.outfile))
    return
