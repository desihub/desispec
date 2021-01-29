"""
This script generates a master NEA file for a given camera.
"""

from   specter.psf.gausshermite  import  GaussHermitePSF
from   desiutil.log import get_logger
from   desispec.parallel import default_nproc as numprocesses
from   functools import partial

import numpy as np
import argparse
import sys
import copy
import astropy.io.fits as fits

# https://github.com/desihub/desispec/issues/1006
sampling           = 500

wmin, wmax, wdelta = 3600., 9824., 0.8
fullwave           = np.round(np.arange(wmin, wmax + wdelta, wdelta), 1)
cslice             = {"b": slice(0, 2751, sampling), "r": slice(2700, 5026, sampling), "z": slice(4900, 7781, sampling)}

def parse(options=None):
    parser = argparse.ArgumentParser(description="Generate master NEA file for a given camera.")
    parser.add_argument('-i','--infile', type = str, default = None, required=True,
                        help = 'path of DESI psf fits file.')
    parser.add_argument('--outdir', type = str, default = None, required=True,
			help = 'dir. of output maser nea file for a given camera')
    args = None
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args

def process_one(w, psf, ifiber):    
    psf_2d = psf.pix(ispec=ifiber, wavelength=w)
    psf_1d = np.sum(psf_2d, axis=0)
    
    # Normalized to one by definition (TBC, again).                                                                                                                                                                                    
    # dA = 1.0 [pixel units]                                                                                                                                                                                                           
    norm = np.sum(psf_1d)
    psf_1d /= norm
    
    try:
        # Automatically raises an assertion.                                                                                                                                                                                              
        # np.testing.assert_almost_equal(norm, 1.0, decimal=7)

        # http://articles.adsabs.harvard.edu/pdf/1983PASP...95..163K                                                                                                                                                                
        nea       = 1. / np.sum(psf_1d**2.)  # [pixel units].                                                                                                                                                                           
        angperpix = psf.angstroms_per_pixel(ifiber, w)

        return  [nea, angperpix]

    except:
        print('Failed on fiber {} and wavelength {} [{} to {} limit] with norm {}.'.format(ifiber, w, psf._wmin_spec[ifiber], psf._wmax_spec[ifiber], norm))        
        return [-99., -99.]

def main(args):
    log  = get_logger()

    cam  = args.infile.split('/')[-1].split('-')[1]
    band = cam[0]
    
    log.info("calculating master nea for camera {}.".format(cam))

    psf   = GaussHermitePSF(args.infile)
    nspec = psf.nspec 
    
    # Sampled to every 50x 0.8A (in sclice).
    wave = fullwave[cslice[band]]

    neas = []
    angperpix = []
    
    for ifiber in range(psf.nspec):
        row_nea = []
        row_angperpix = []

        results = [process_one(w, psf, ifiber) for w in wave]
        results = np.array(results)
        
        log.info('Solved for row {} of {}.'.format(ifiber, nspec))
            
        neas.append(results[:,0].tolist())
        angperpix.append(results[:,1].tolist())

    neas = np.array(neas)
    angperpix = np.array(angperpix)

    # Patch failures.
    log.info('Patching failures of psf norm. with median nea.')
    
    med_nea = np.median(neas)
    med_angperpix = np.median(angperpix)

    neas[neas == -99.] = med_nea
    angperpix[angperpix == -99.] = med_angperpix

    # Convert to float 32.
    neas = neas.astype(np.float32)
    angperpix = angperpix.astype(np.float32)

    print('MEDIAN NEA: {:.3f}'.format(np.median(neas)))
    print('MEDIAN ANG PER PIX: {:.3f}'.format(np.median(angperpix)))
    
    hdr  = fits.Header()
    hdr['MASTERPSF'] = args.infile
    hdr['CAMERA'] = cam
    
    hdu0 = fits.PrimaryHDU()
    hdu1 = fits.ImageHDU(wave, name='WAVELENGTH') 
    hdu2 = fits.ImageHDU(neas, name='NEA')
    hdu3 = fits.ImageHDU(angperpix, name='ANGPERPIX')

    hdul = fits.HDUList([hdu0, hdu1, hdu2, hdu3])

    hdul.writeto(args.outdir + '/masternea_{}.fits'.format(cam), overwrite=True)

    log.info("successfully wrote {}".format(args.outdir + '/masternea_{}.fits'.format(cam)))

    
if __name__ == '__main__':
    args = parse()
    
    main(args)
