import os

import sys
import glob
import fitsio

import itertools
import warnings

import numpy as np
import pylab as pl

import desisim.templates
import astropy.io.fits           as      fits

import desispec.io
import redrock.templates
import matplotlib.pyplot         as      plt

from   astropy.convolution       import  convolve, Box1DKernel
from   desispec.spectra          import  Spectra
from   desispec.frame            import  Frame
from   desispec.resolution       import  Resolution
from   desispec.io.meta          import  findfile
from   desispec.io               import  read_frame, read_fiberflat, read_flux_calibration, read_sky, read_fibermap 
from   desispec.interpolation    import  resample_flux
from   astropy.table             import  Table
from   desispec.io.image         import  read_image
from   specter.psf.gausshermite  import  GaussHermitePSF
from   scipy.signal              import  medfilt
from   desispec.calibfinder      import  CalibFinder
from   astropy.utils.exceptions  import  AstropyWarning
from   scipy                     import  stats
from   pathlib                   import  Path
from   templateSNR               import	 templateSNR
from   RadLSS                    import  RadLSS

def get_expids(night, andes='/global/cfs/cdirs/desi/spectro/redux/andes'):
    return  np.sort(np.array([x.split('/')[-1] for x in glob.glob(andes + '/exposures/{}/*'.format(night))]).astype(np.int))

nmax      = 1
night     = '20200315'
tracers   = ['ELG']

expids    = get_expids(night)

camera    = ['b5', 'r5', 'z5']

print('Number of exposures to process: {}'.format(len(expids)))

blacklist = [] # [55668, 55670, 55672, 55674, 55676, 55678, 55680, 55682, 55684, 55686, 55688, 55690, 55692, 55705, 55706, 55707, 55708, 55709, 55713, 55714, 55715, 55716, 55717]

for nexp, expid in enumerate(expids[20:]):
  if expid not in blacklist:
    print('Solving for EXPID {:08d}'.format(expid))

    try: 
        rads = RadLSS(night, expid, cameras=None)
    
        # rads.compute(tracers=tracers)

        #  if nexp == nmax:
        #    break

        del rads

    except:
        blacklist.append(expid)

        print('Blacklisted {}'.format(expid))
        
    sys.stdout.flush()
        
print(blacklist)
