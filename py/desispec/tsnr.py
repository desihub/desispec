import numpy as np


def quadrant(x, y, frame):
    ccdsizes = np.array(frame.meta['CCDSIZE'].split(',')).astype(np.int)

    if (x < (ccdsizes[0] / 2)):
        if (y < (ccdsizes[1] / 2)):
            return  'A'
        else:
            return  'C'

    else:
        if (y < (ccdsizes[1] / 2)):
            return  'B'
        else:
            return  'D'
        
def calc_tsnr(frame, psf, fluxcalib, fiberflat, skymodel, nea, angperpix):
    rdnoise = []

    for ifiber in range(len(frame.flux)):
        # quadrants for readnoise.                                                                                                                   
        psf_wave = np.median(frame.wave)
        
        x, y     = psf.xy(ifiber, psf_wave)
        ccd_quad = quadrant(x, y, frame)
        rdnoise.append(frame.meta['OBSRDN{}'.format(ccd_quad)])

    # rdnoise by fiber. 
    rdnoise = np.array(rdnoise)

    '''
    dflux   = np.array(dtemplate_flux, copy=True)

    # Work in uncalibrated flux units (electrons per angstrom); flux_calib includes exptime. tau.
    dflux  *= flux_calib    # [e/A]                                                                                                                                                                

    # Wavelength dependent fiber flat;  Multiply or divide - check with Julien.
    result  = dflux * fiberflat
    result  = result**2.

    # RDNOISE & NPIX assumed wavelength independent.
    denom   = readnoise**2 * npix / angstroms_per_pixel + fiberflat * sky_flux
    result /= denom
    
    # Eqn. (1) of https://desi.lbl.gov/DocDB/cgi-bin/private/RetrieveFile?docid=4723;filename=sky-monitor-mc-study-v1.pdf;version=2
    return  np.sum(result)
    '''
    
    return  None
