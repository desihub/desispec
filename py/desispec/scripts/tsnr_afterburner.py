import os
import glob
import itertools
import argparse
import astropy.io.fits as fits
import numpy as np

from   desispec.io import read_sky
from   desispec.io import read_fiberflat
from   pathlib import Path
from   desispec.io.meta import findfile, specprod_root
from   desispec.calibfinder import CalibFinder
from   desispec.io import read_frame
from   desispec.io import read_fibermap
from   desispec.io.fluxcalibration import read_flux_calibration
from   desiutil.log import get_logger
from   desispec.tsnr import calc_tsnr
from   astropy.table import Table, vstack

def parse(options=None):
    parser = argparse.ArgumentParser(description="Apply fiberflat, sky subtraction and calibration.")
    parser.add_argument('--outdir', type = str, default = None, required=True,
                        help = 'Dir. to write to.')
    parser.add_argument('--prod', type = str, default = None, required=True,
                        help = 'Path to reduction, e.g. /global/cfs/cdirs/desi/spectro/redux/blanc/')
    parser.add_argument('--camera', type = str, default = None, required=False,
                        help = 'Camera to reduce.')
    parser.add_argument('--summary_only', type = int, default = 0, required=False,
                        help = 'Write only summary file.')
    parser.add_argument('--expids', type = str, default = None, required=False,
                        help = 'Comma separated list of exp ids to be reduced.')
    args = None
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args

args=parse()

#
log = get_logger()

# Set SPEC_PROD also.
calibdir = os.environ['DESI_SPECTRO_CALIB']

petals = np.arange(10).astype(str)

if args.camera is None:
    cameras = [x[0] + x[1] for x in itertools.product(['b', 'r', 'z'], petals.astype(np.str))]
else:
    cameras = [args.camera]

if args.expids is not None:
    expids = [np.int(x) for x in args.expids.split(',')]
else:
    expids = None
    
# 
cframes = {}

for cam in cameras:
    cframes[cam] = list(glob.glob('{}/exposures/*/*/cframe-{}-*.fits'.format(args.prod, cam)))

sci_frames = {}

for cam in cameras:
    sci_frames[cam] = []
    
    for cframe in cframes[cam]:
        hdul = fits.open(cframe)
        hdr  = hdul[0].header 
        
        flavor = hdr['FLAVOR']
        prog = hdr['PROGRAM']
        expid = hdr['EXPID']
        
        if expids is not None:
            if expid not in expids:
                continue
            
        if flavor == 'science':
            sci_frames[cam].append(cframe)

        hdul.close()
        
    print('{} science frames to reduce for {}.'.format(len(sci_frames[cam]), cam))

exit(0)
    
# 
for cam in cameras:
    summary  = None
    
    for kk, x in enumerate(sci_frames[cam]):
        hdul = fits.open(x)
        hdr  = hdul[0].header

        flavor = hdr['FLAVOR']
        prog = hdr['PROGRAM']
        
        parts = prog.split(' ')

        if parts[0] == 'SV1':
            parts  = x.split('/')
                
            night  = parts[9]
            expid  = np.int(parts[10])

            name   = parts[-1]

            parts  = name.split('-')
            camera = parts[1]
            
            calib  = findfile('fluxcalib', night=night, expid=expid, camera=camera, specprod_dir=None)
            
            cframe = fits.open(x)
            hdr    = cframe[0].header['FIBERFLT']
            flat   = hdr.replace('SPECPROD', args.prod)

            tileid = cframe[0].header['TILEID']
        
            iin = x.replace('cframe', 'frame')
            sky = x.replace('cframe', 'sky')
            psf = sky.replace('sky', 'psf')
            nea = '/project/projectdirs/desi/users/mjwilson/master_nea/masternea_{}.fits'.format(camera)  
            ens = '/project/projectdirs/desi/users/mjwilson/tsnr-ensemble/'
            out = args.outdir + '/tsnr/{}/{:08d}/tsnr-{}-{:08d}.fits'.format(night, expid, camera, expid)

            if os.path.exists(out):
                continue
            
            Path(os.path.dirname(out)).mkdir(parents=True, exist_ok=True)
                
            frame=read_frame(iin)
            fiberflat=read_fiberflat(flat)
            fluxcalib=read_flux_calibration(calib)
            skymodel=read_sky(sky)
            
            results = calc_tsnr(frame, fiberflat=fiberflat, skymodel=skymodel, fluxcalib=fluxcalib)

            if not args.summary_only:
                # Write individual.
                table=Table()
                
                for k in results:
                    if k != 'ALPHA':
                        table[k] = results[k].astype(np.float32)

                table.meta['NIGHT']  = night
                table.meta['EXPID']  = '{:08d}'.format(expid)
                table.meta['ALPHA']  = results['ALPHA']
                table.meta['TILEID'] = tileid
            
                table.write(out, format='fits', overwrite=True)
            
            # Append to summary. 
            entry = Table(data=np.array(['{:08d}'.format(expid)]), names=['EXPID'])
            
            entry['NIGHT']  = night
            entry['CAMERA'] = camera

            keys = list(results.keys())
            
            for k in keys:
                if 'TSNR' in k:
                    sk = k.split('_')[0]    
                    results[sk] = results[k]
                    del results[k]
                                
            for k in results:                                                                                                                                                                                                
                entry[k] = np.median(results[k].astype(np.float32))

            if summary is None:
                summary = entry

            else:
                summary = vstack((summary, entry))
                
            print('{:08d}  {}: Reduced {} of {}.'.format(expid, cam, kk, len(sci_frames[cam])))

        hdul.close()
        
    # 
    summary.write('/global/cscratch1/sd/mjwilson/trash/afterburner/tsnr/summary_{}.fits'.format(cam), format='fits', overwrite=True)
