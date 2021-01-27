"""
This script generates a master NEA file for a given camera.
"""

from   specter.psf.gausshermite  import  GaussHermitePSF
from   desiutil.log import get_logger

import numpy as np
import argparse
import sys
import copy
import astropy.io.fits as fits

# https://github.com/desihub/desispec/issues/1006
wmin, wmax, wdelta = 3600, 9824, 0.8
fullwave           = np.round(np.arange(wmin, wmax + wdelta, wdelta), 1)
cslice             = {"b": slice(0, 2751, 50), "r": slice(2700, 5026, 50), "z": slice(4900, 7781, 50)}

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

def main(args):
    log  = get_logger()

    cam  = args.infile.split('/')[-1].split('-')[1]
    band = cam[0]
    
    log.info("calculating master nea for camera {}.".format(cam))

    # construct PSF from file. 
    psf  = GaussHermitePSF(args.infile)

    # Sampled to every 50x 0.8A (in sclice).
    wave = fullwave[cslice[band]]

    neas = []
    angperpix = []
    
    for ifiber in range(psf.nspec):
        row_nea = []
        row_angperpix = []
        
        for w in wave: 
            psf_2d = psf.pix(ispec=ifiber, wavelength=w)

	    # Normalized to one by definition (TBC, again).                                                                                                                                                       
            # dA = 1.0 [pixel units]                                                                                                                                                                              
            norm = np.sum(psf_2d)
                
	    # Automatically raises an assertion.                                                                                                                                                                  
            np.testing.assert_almost_equal(norm, 1.0, decimal=7)
                
	    # http://articles.adsabs.harvard.edu/pdf/1983PASP...95..163K                                                                                                                                          
            row_nea.append(1. / np.sum(psf_2d ** 2.))  # [pixel units].                                                                                                                                              
            row_angperpix.append(psf.angstroms_per_pixel(ifiber, w))
            
        log.info('Solved for row {} of {}.'.format(ifiber, psf.nspec))
            
        neas.append(row_nea)
        angperpix.append(row_angperpix)

    neas = np.array(neas)
    angperpix = np.array(angperpix)

    hdr  = fits.Header()
    hdr['MASTERPSF'] = args.infile

    hdu0 = fits.PrimaryHDU()
    hdu1 = fits.ImageHDU(wave, name='WAVELENGTH') 
    hdu2 = fits.ImageHDU(neas, name='NEA')
    hdu3 = fits.ImageHDU(angperpix, name='ANGPERPIX')

    hdul = fits.HDUList([hdu0, hdu1, hdu2, hdu3])

    hdul.writeto(args.outdir + '/masternea_{}.fits'.format(cam))

    log.info("successfully wrote {}".format(args.outdir + '/masternea_{}.fits'.format(cam)))

    
if __name__ == '__main__':
    args = parse()
    
    main(args)
