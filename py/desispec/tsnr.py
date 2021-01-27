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

def get_tsnr(frame, psf):
    rdnoise = []

    for ifiber in range(len(frame.flux)):
        # quadrants for readnoise.                                                                                                                   
        psf_wave = np.median(frame.wave)
        
        x, y     = psf.xy(ifiber, psf_wave)
        ccd_quad = quadrant(x, y, frame)
        rdnoise.append(frame.meta['OBSRDN{}'.format(ccd_quad)])

    rdnoise = np.array(rdnoise)

    return  None
