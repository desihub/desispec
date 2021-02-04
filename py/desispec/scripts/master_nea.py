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

def parse(options=None):
    parser = argparse.ArgumentParser(description="Generate master NEA file for a given camera.")
    parser.add_argument('-i','--infile', type = str, default = None, required=True,
                        help = 'path of DESI psf fits file.')
    parser.add_argument('--outdir', type = str, default = None, required=True,
			help = 'dir. of output maser nea file for a given camera')
    parser.add_argument('--sampling', type=int, default=500, required=False,
                        help = 'Sampling in wavelength bins, typically 0.8A')
    parser.add_argument('--blue_lim', type=float, default=3520., required=False,
                        help = 'Blue wavelength limit [Angstroms]')
    parser.add_argument('--red_lim', type=float, default=9950., required=False,
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

    # If beyond the wavelength limit, return nea at the limit.
    wmax   = psf.wavelength(ifiber, psf.npix_y - 0.5)
    w      = np.minimum(w, wmax)
    
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
    wdelta = 0.8
    
    cam  = args.infile.split('/')[-1].split('-')[1]
    band = cam[0]
    
    log.info("calculating master nea for camera {}.".format(cam))

    sample_length = args.sampling * wdelta
    
    # https://github.com/desihub/desispec/blob/8dccacdd9b35efc2a5c771269fc2b28dc742caef/bin/desi_proc#L703
    # Note: Extend by one sampling length to ensure limit remains interpolation. 
    if cam.startswith('b'):
        wave = np.round(np.arange(args.blue_lim, 5800. + wdelta + sample_length, wdelta), 1) 

    elif cam.startswith('r'):
        wave = np.round(np.arange(5760., 7620.0 + wdelta + sample_length, wdelta), 1)

    elif cam.startswith('z'):
        wave = np.round(np.arange(7520., args.red_lim + wdelta + sample_length, wdelta), 1)

    else:
        raise ValueError('Erroneous camera found: {}'.format(cam))
        
    log.info('Assuming blue wavelength of {} A.'.format(wave.min()))
    log.info('Assuming  red wavelength of {} A.'.format(wave.max()))
    log.info('Assuming {} A sampling'.format(sample_length))

    wave = wave[::args.sampling]
        
    psf   = GaussHermitePSF(args.infile)
    nspec = psf.nspec 
        
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
