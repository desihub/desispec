"""
Run integration tests from pixsim through redshifts

python -m desispec.test.integration_test
"""
from __future__ import absolute_import, print_function
import sys
import os
import random
import time

import numpy as np
from astropy.io import fits

from desispec.pipeline import runcmd
from desispec import io
from desispec.log import get_logger

#- prevent nose from trying to run this test since it takes too long
__test__ = False

def check_env():
    """
    Check required environment variables; raise RuntimeException if missing
    """
    log = get_logger()
    #- template locations
    missing_template_env = False
    for objtype in ('ELG', 'LRG', 'STD', 'QSO'):
        name = 'DESI_'+objtype+'_TEMPLATES'
        if name not in os.environ:
            log.warning('missing ${0} needed for simulating spectra'.format(name))
            missing_template_env = True

    if missing_template_env:
        log.warning('    e.g. see NERSC:/project/projectdirs/desi/datachallenge/dc2/templates/')

    missing_env = False
    for name in (
        'DESI_SPECTRO_SIM', 'DESI_SPECTRO_DATA', 'DESI_SPECTRO_REDUX', 'PIXPROD', 'PRODNAME'):
        if name not in os.environ:
            log.warning("missing ${0}".format(name))
            missing_env = True

    if missing_env:
        log.warning("Why are these needed?")
        log.warning("    Simulations written to $DESI_SPECTRO_SIM/$PIXPROD/")
        log.warning("    Raw data read from $DESI_SPECTRO_DATA/")
        log.warning("    Spectro pipeline output written to $DESI_SPECTRO_REDUX/$PRODNAME/")

    #- Wait until end to raise exception so that we report everything that
    #- is missing before actually failing
    if missing_env or missing_template_env:
        log.critical("missing env vars; exiting without running pipeline")
        sys.exit(1)


#- TODO: fix usage of night to be something other than today
def integration_test(night=None, nspec=5, clobber=False):
    """Run an integration test from raw data simulations through redshifts
    
    Args:
        night (str, optional): YEARMMDD, defaults to current night
        nspec (int, optional): number of spectra to include
        clobber (bool, optional): rerun steps even if outputs already exist
        
    Raises:
        RuntimeError if any script fails
      
    """
    log = get_logger()
    #- YEARMMDD string, rolls over at noon not midnight
    if night is None:
        night = time.strftime('%Y%m%d', time.localtime(time.time()-12*3600))

    #- check for required environment variables
    check_env()

    #- parameter dictionary that will later be used for formatting commands
    params = dict(night=night, nspec=nspec)

    #-----
    #- Input fibermaps, spectra, and pixel-level raw data
    for expid, flavor in zip([0,1,2], ['flat', 'arc', 'science']):
        cmd = "newexp-desi --flavor {flavor} --nspec {nspec} --night {night} --expid {expid}".format(
            expid=expid, flavor=flavor, **params)
        fibermap = io.findfile('fibermap', night, expid)
        simspec = '{}/simspec-{:08d}.fits'.format(os.path.dirname(fibermap), expid)
        inputs = []
        outputs = [fibermap, simspec]
        if runcmd(cmd, inputs, outputs, clobber) != 0:
            raise RuntimeError('pixsim newexp failed for {} exposure {}'.format(flavor, expid))

        cmd = "pixsim-desi --nspec {nspec} --night {night} --expid {expid}".format(expid=expid, **params)
        inputs = [fibermap, simspec]
        outputs = list()
        for camera in ['b0', 'r0', 'z0']:
            pixfile = io.findfile('pix', night, expid, camera)
            outputs.append(pixfile)
            outputs.append(os.path.join(os.path.dirname(pixfile), os.path.basename(pixfile).replace('pix-', 'simpix-')))
        if runcmd(cmd, inputs, outputs, clobber) != 0:
            raise RuntimeError('pixsim failed for {} exposure {}'.format(flavor, expid))

    #-----
    #- Extract

    waverange = dict(
        b = "3570,5940,1.0",
        r = "5630,7740,1.0",
        z = "7440,9830,1.0",
        )
    for expid in [0,1,2]:
        for channel in ['b', 'r', 'z']:
            camera = channel+'0'
            pixfile = io.findfile('pix', night, expid, camera)
            psffile = '{}/data/specpsf/psf-{}.fits'.format(os.getenv('DESIMODEL'), channel)
            framefile = io.findfile('frame', night, expid, camera)
            cmd = "exspec -i {pix} -p {psf} --specrange 0,{nspec} -w {wave} -o {frame}".format(
                pix=pixfile, psf=psffile, wave=waverange[channel], frame=framefile, **params)

            inputs = [pixfile, psffile]
            outputs = [framefile,]
            if runcmd(cmd, inputs, outputs, clobber) != 0:
                raise RuntimeError('extraction failed for {} expid {}'.format(camera, expid))

    #-----
    #- Fiber flat
    expid = 0
    for channel in ['b', 'r', 'z']:
        camera = channel+"0"
        framefile = io.findfile('frame', night, expid, camera)
        fiberflat = io.findfile('fiberflat', night, expid, camera)
        cmd = "desi_compute_fiberflat.py --infile {frame} --outfile {fiberflat}".format(
            frame=framefile, fiberflat=fiberflat, **params)
        inputs = [framefile,]
        outputs = [fiberflat,]
        if runcmd(cmd, inputs, outputs, clobber) != 0:
            raise RuntimeError('fiberflat failed for '+camera)

    #-----
    #- Sky model
    flat_expid = 0
    expid = 2
    for channel in ['b', 'r', 'z']:
        camera = channel+"0"
        framefile = io.findfile('frame', night, expid, camera)
        fibermap = io.findfile('fibermap', night, expid)
        fiberflat = io.findfile('fiberflat', night, flat_expid, camera)
        skyfile = io.findfile('sky', night, expid, camera)
        cmd="desi_compute_sky.py --infile {frame} --fibermap {fibermap} --fiberflat {fiberflat} --outfile {sky}".format(
            frame=framefile, fibermap=fibermap, fiberflat=fiberflat, sky=skyfile, **params)
        inputs = [framefile, fibermap, fiberflat]
        outputs = [skyfile, ]
        if runcmd(cmd, inputs, outputs, clobber) != 0:
            raise RuntimeError('sky model failed for '+camera)


    #-----
    #- Fit standard stars
    if 'STD_TEMPLATES' in os.environ:
        std_templates = os.getenv('STD_TEMPLATES')
    else:
        std_templates = os.getenv('DESI_ROOT')+'/spectro/templates/stellar_templates/v1.0/stdstar_templates_v1.0.fits'

    stdstarfile = io.findfile('stdstars', night, expid, spectrograph=0)
    cmd = """desi_fit_stdstars.py --spectrograph 0 \
      --fibermap {fibermap} \
      --fiberflatexpid {flat_expid} \
      --models {std_templates} --outfile {stdstars}""".format(
        flat_expid=flat_expid, fibermap=fibermap, std_templates=std_templates,
        stdstars=stdstarfile)

    inputs = [fibermap, std_templates]
    outputs = [stdstarfile,]
    if runcmd(cmd, inputs, outputs, clobber) != 0:
        raise RuntimeError('fitting stdstars failed')


    #-----
    #- Flux calibration
    for channel in ['b', 'r', 'z']:
        camera = channel+"0"
        framefile = io.findfile('frame', night, expid, camera)
        fibermap  = io.findfile('fibermap', night, expid)
        fiberflat = io.findfile('fiberflat', night, flat_expid, camera)
        skyfile   = io.findfile('sky', night, expid, camera)
        calibfile = io.findfile('calib', night, expid, camera)

        #- Compute flux calibration vector
        cmd = """desi_compute_fluxcalibration.py \
          --infile {frame} --fibermap {fibermap} --fiberflat {fiberflat} --sky {sky} \
          --models {stdstars} --outfile {calib}""".format(
            frame=framefile, fibermap=fibermap, fiberflat=fiberflat, sky=skyfile,
            stdstars=stdstarfile, calib=calibfile,
            )
        inputs = [framefile, fibermap, fiberflat, skyfile, stdstarfile]
        outputs = [calibfile,]
        if runcmd(cmd, inputs, outputs, clobber) != 0:
            raise RuntimeError('flux calibration failed for '+camera)

        #- Apply the flux calibration to write a cframe file
        cframefile = io.findfile('cframe', night, expid, camera)
        cmd = """desi_process_exposure.py \
          --infile {frame} --fiberflat {fiberflat} --sky {sky} --calib {calib} \
          --outfile {cframe}""".format(frame=framefile, fibermap=fibermap,
            fiberflat=fiberflat, sky=skyfile, calib=calibfile, cframe=cframefile)
        inputs = [framefile, fiberflat, skyfile, calibfile]
        outputs = [cframefile, ]
        if runcmd(cmd, inputs, outputs, clobber) != 0:
            raise RuntimeError('combining calibration steps failed for '+camera)

    #-----
    #- Bricks
    inputs = list()
    for camera in ['b0', 'r0', 'z0']:
        inputs.append( io.findfile('cframe', night, expid, camera) )

    outputs = list()
    fibermap = io.read_fibermap(io.findfile('fibermap', night, expid))
    bricks = set(fibermap['BRICKNAME'])
    for b in bricks:
        for channel in ['b', 'r', 'z']:
            outputs.append( io.findfile('brick', brickid=b, band=channel))

    cmd = "desi_make_bricks.py --night "+night
    if runcmd(cmd, inputs, outputs, clobber) != 0:
        raise RuntimeError('brick generation failed')

    #-----
    #- Redshifts!
    for b in bricks:
        inputs = [io.findfile('brick', brickid=b, band=channel) for channel in ['b', 'r', 'z']]
        zbestfile = io.findfile('zbest', brickid=b)
        outputs = [zbestfile, ]
        cmd = "desi_zfind.py --brick {} -o {}".format(b, zbestfile)
        if runcmd(cmd, inputs, outputs, clobber) != 0:
            raise RuntimeError('redshifts failed for brick '+b)

    #-----
    #- Did it work?
    #- (this combination of fibermap, simspec, and zbest is a pain)
    simdir = os.path.dirname(io.findfile('fibermap', night=night, expid=expid))
    simspec = '{}/simspec-{:08d}.fits'.format(simdir, expid)
    siminfo = fits.getdata(simspec, 'METADATA')

    print()
    print("--------------------------------------------------")
    print("Brick     True  z        ->  Class  z        zwarn")
    # print("3338p190  SKY   0.00000  ->  QSO    1.60853   12   - ok")
    for b in bricks:
        zbest = io.read_zbest(io.findfile('zbest', brickid=b))
        for i in range(len(zbest.z)):
            if zbest.type[i] == 'ssp_em_galaxy':
                objtype = 'GAL'
            elif zbest.type[i] == 'spEigenStar':
                objtype = 'STAR'
            else:
                objtype = zbest.type[i]

            z, zwarn = zbest.z[i], zbest.zwarn[i]

            j = np.where(fibermap['TARGETID'] == zbest.targetid[i])[0][0]
            truetype = siminfo['OBJTYPE'][j]
            truez = siminfo['REDSHIFT'][j]
            dv = 3e5*(z-truez)/(1+truez)
            if truetype == 'SKY' and zwarn > 0:
                status = 'ok'
            elif zwarn == 0:
                if truetype == 'LRG' and objtype == 'GAL' and abs(dv) < 150:
                    status = 'ok'
                elif truetype == 'ELG' and objtype == 'GAL' and abs(dv) < 150:
                    status = 'ok'
                elif truetype == 'QSO' and objtype == 'QSO' and abs(dv) < 750:
                    status = 'ok'
                elif truetype == 'STD' and objtype == 'STAR':
                    status = 'ok'
                else:
                    status = 'oops'
            else:
                status = 'oops'
            print('{0}  {1:4s} {2:8.5f}  -> {3:5s} {4:8.5f} {5:4d}  - {6}'.format(
                b, truetype, truez, objtype, z, zwarn, status))

    print("--------------------------------------------------")

if __name__ == '__main__':
    integration_test()
