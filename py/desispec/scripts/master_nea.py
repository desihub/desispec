"""
This script generates a master NEA (Noise Equivalent Area) file for a given camera.
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


log                = get_logger()

# https://github.com/desihub/desispec/issues/1006
sampling           = 500

def parse(options=None):
    parser = argparse.ArgumentParser(description="Generate master NEA file for a given camera.")
    parser.add_argument('-i','--infile', type = str, default = None, required=True,
                        help = 'path of DESI psf fits file.')
    parser.add_argument('--outdir', type = str, default = None, required=True,
			help = 'dir. of output maser nea file for a given camera')
    parser.add_argument('--blue_lim', type=float, default=3600., required=False,
                        help = 'Blue wavelength limit [Angstroms]')
    parser.add_argument('--red_lim', type=float, default=9824., required=False,
                        help = 'Red wavelength limit [Angstroms]')
    args = None
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args

def process_one(w, psf, ifiber):
    '''
    Compute the 1D NEA for a given wavelength, fiber and 
    desispec psf instance.  

    input:
        w:  wavelength, Angstroms. 
        psf: desispec psf instance. 
        ifiber:  fiber indexing integer [0,500].

    returns:
        list of 1D nea value [pixles] and 
        angstroms per pix. for this fiber and wavelength. 
    '''
    
    psf_2d = psf.pix(ispec=ifiber, wavelength=w)
    psf_1d = np.sum(psf_2d, axis=0)
    
    # Normalized to one by definition (TBC, again).                                                                                                                                                                                    
    # dA = 1.0 [pixel units]                                                                                                                                                                                                           
    norm = np.sum(psf_1d)
    psf_1d /= norm
    
    # NOTE: PSf is unexpectedly unnormailzed for the first few fibers
    # at the edges of the wavelength grid.  Given we renomalize after
    # marginalizing over wavelength, we ignore this fact. 

    # http://articles.adsabs.harvard.edu/pdf/1983PASP...95..163K                                                                                                                                                                
    nea       = 1. / np.sum(psf_1d**2.)  # [pixel units].                                                                                                                                                                           
    angperpix = psf.angstroms_per_pixel(ifiber, w)

    return  [nea, angperpix]

def main(args):
    cam  = args.infile.split('/')[-1].split('-')[1]
    band = cam[0]
    
    log.info("calculating master nea for camera {}.".format(cam))

    wmin, wmax, wdelta = args.blue_lim, args.red_lim, 0.8
    fullwave           = np.round(np.arange(wmin, wmax + wdelta, wdelta), 1)
    cslice             = {"b": slice(0, 2751, sampling), "r": slice(2700, 5026, sampling), "z": slice(4900, 7781, sampling)}

    log.info('Assuming blue wavelength of {} A.'.format(wmin))
    log.info('Assuming  red wavelength of {} A.'.format(wmax))
    
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

    # Convert to float 32 for smaller files.
    neas = neas.astype(np.float32)
    angperpix = angperpix.astype(np.float32)

    log.info('MEDIAN NEA: {:.3f}'.format(np.median(neas)))
    log.info('MEDIAN ANG PER PIX: {:.3f}'.format(np.median(angperpix)))
    
    hdr  = fits.Header()
    hdr['MASTPSF'] = args.infile
    hdr['CAMERA'] = cam
    
    hdu0 = fits.PrimaryHDU(header=hdr)
    hdu1 = fits.ImageHDU(wave, name='WAVELENGTH') 
    hdu2 = fits.ImageHDU(neas, name='NEA')
    hdu3 = fits.ImageHDU(angperpix, name='ANGPERPIX')

    hdul = fits.HDUList([hdu0, hdu1, hdu2, hdu3])

    hdul.writeto(args.outdir + '/masternea_{}.fits'.format(cam), overwrite=True)

    log.info("Successfully wrote {}".format(args.outdir + '/masternea_{}.fits'.format(cam)))

    
if __name__ == '__main__':
    args = parse()
    
    main(args)
