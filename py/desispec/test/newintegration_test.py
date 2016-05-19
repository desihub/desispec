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

import desispec.pipeline as pipe
import desispec.io as io
import desispec.log as logging

import desispec.scripts.makebricks as makebricks


#- prevent nose from trying to run this test since it takes too long
__test__ = False


def check_env():
    """
    Check required environment variables; raise RuntimeException if missing
    """
    log = logging.get_logger()
    #- template locations
    missing_env = False
    if 'DESI_BASIS_TEMPLATES' not in os.environ:
        log.warning('missing $DESI_BASIS_TEMPLATES needed for simulating spectra'.format(name))
        missing_env = True

    if not os.path.isdir(os.getenv('DESI_BASIS_TEMPLATES')):
        log.warning('missing $DESI_BASIS_TEMPLATES directory')
        log.warning('e.g. see NERSC:/project/projectdirs/desi/spectro/templates/basis_templates/v1.0')
        missing_env = True

    for name in (
        'DESI_SPECTRO_SIM', 'DESI_SPECTRO_REDUX', 'PIXPROD', 'PRODNAME', 'DESIMODEL'):
        if name not in os.environ:
            log.warning("missing ${0}".format(name))
            missing_env = True

    if missing_env:
        log.warning("Why are these needed?")
        log.warning("    Simulations written to $DESI_SPECTRO_SIM/$PIXPROD/")
        log.warning("    Raw data read from $DESI_SPECTRO_DATA/")
        log.warning("    Spectro pipeline output written to $DESI_SPECTRO_REDUX/$PRODNAME/")
        log.warning("    Templates are read from $DESI_BASIS_TEMPLATES")

    #- Wait until end to raise exception so that we report everything that
    #- is missing before actually failing
    if missing_env:
        log.critical("missing env vars; exiting without running pipeline")
        sys.exit(1)

    #- Override $DESI_SPECTRO_DATA to match $DESI_SPECTRO_SIM/$PIXPROD
    os.environ['DESI_SPECTRO_DATA'] = os.path.join(os.getenv('DESI_SPECTRO_SIM'), os.getenv('PIXPROD'))


# Simulate raw data

def sim(night, nspec=5, clobber=False):
    """
    Simulate data as part of the integration test.

    Args:
        night (str): YEARMMDD
        nspec (int, optional): number of spectra to include
        clobber (bool, optional): rerun steps even if outputs already exist
        
    Raises:
        RuntimeError if any script fails
    """
    log = logging.get_logger()

    # Create input fibermaps, spectra, and pixel-level raw data

    for expid, flavor in zip([0,1,2], ['flat', 'arc', 'dark']):
        cmd = "newexp-desi --flavor {flavor} --nspec {nspec} --night {night} --expid {expid}".format(
            expid=expid, flavor=flavor, nspec=nspec, night=night)
        fibermap = io.findfile('fibermap', night, expid)
        simspec = '{}/simspec-{:08d}.fits'.format(os.path.dirname(fibermap), expid)
        inputs = []
        outputs = [fibermap, simspec]
        if pipe.runcmd(cmd, inputs, outputs, clobber) != 0:
            raise RuntimeError('pixsim newexp failed for {} exposure {}'.format(flavor, expid))

        cmd = "pixsim-desi --nspec {nspec} --night {night} --expid {expid}".format(expid=expid, nspec=nspec, night=night)
        inputs = [fibermap, simspec]
        outputs = list()
        for camera in ['b0', 'r0', 'z0']:
            pixfile = io.findfile('pix', night, expid, camera)
            outputs.append(pixfile)
            outputs.append(os.path.join(os.path.dirname(pixfile), os.path.basename(pixfile).replace('pix-', 'simpix-')))
        if pipe.runcmd(cmd, inputs, outputs, clobber) != 0:
            raise RuntimeError('pixsim failed for {} exposure {}'.format(flavor, expid))

    return


def integration_test(night=None, nspec=5, clobber=False):
    """Run an integration test from raw data simulations through redshifts
    
    Args:
        night (str, optional): YEARMMDD, defaults to current night
        nspec (int, optional): number of spectra to include
        clobber (bool, optional): rerun steps even if outputs already exist
        
    Raises:
        RuntimeError if any script fails
      
    """
    log = logging.get_logger()
    log.setLevel(logging.DEBUG)

    # YEARMMDD string, rolls over at noon not midnight
    # TODO: fix usage of night to be something other than today
    if night is None:
        #night = time.strftime('%Y%m%d', time.localtime(time.time()-12*3600))
        night = "20160726"

    # check for required environment variables
    check_env()

    # simulate inputs
    sim(night, nspec=nspec, clobber=clobber)

    # raw and production locations

    rawdir = os.environ['DESI_SPECTRO_DATA']
    proddir = os.path.join(os.environ['DESI_SPECTRO_REDUX'], os.environ['PRODNAME'])

    # create production output directories and modify the options to
    # restrict the test to a smaller number of spectra.

    pipe.create_prod(rawdir, proddir)

    optfile = os.path.join(proddir, "run", "options.yaml")
    opts = pipe.read_options(optfile)

    opts['extract']['specmin'] = 0
    opts['extract']['nspec'] = nspec
    opts['stdstars']['models'] = '/home/kisner/scratch/desi/spectro/templates/star_templates/v1.1/star_templates_v1.1.fits'

    pipe.write_options(optfile, opts)

    # For this small size of dataset, bootcalib and specex do not yet work.
    # instead we make symlinks to true PSFs used in the simulation.

    # bootcalib

    cal2d = os.path.join(proddir, 'calib2d')
    calpsf = os.path.join(cal2d, 'psf')
    calpsfnight = os.path.join(calpsf, night)
    if not os.path.isdir(calpsfnight):
        os.makedirs(calpsfnight)

    expnight = os.path.join(proddir, 'exposures', night)

    for band in ['b', 'r', 'z']:
        for spec in range(10):
            cam = "{}{}".format(band, spec)
            target = os.path.join(os.environ['DESIMODEL'], 'data', 'specpsf', "psf-{}.fits".format(band))
            lnk = os.path.join(calpsfnight, "psfboot-{}{}.fits".format(band, spec))
            print("ln -s {} {}".format(target, lnk))
            if not os.path.islink(lnk):
                os.symlink(target, lnk)

    # PSF estimation

    for expid in [0, 2]:
        expdir = os.path.join(expnight, "{:08d}".format(expid))
        if not os.path.isdir(expdir):
            os.makedirs(expdir)
        for band in ['b', 'r', 'z']:
            for spec in range(1):
                target = os.path.join(calpsfnight, "psfboot-{}{}.fits".format(band, spec))
                lnk = os.path.join(expdir, "psf-{}{}-{:08d}.fits".format(band, spec, expid))
                print("ln -s {} {}".format(target, lnk))
                if not os.path.islink(lnk):
                    os.symlink(target, lnk)
    for band in ['b', 'r', 'z']:
        for spec in range(1):
            target = os.path.join(calpsfnight, "psfboot-{}{}.fits".format(band, spec))
            lnk = os.path.join(calpsfnight, "psfnight-{}{}.fits".format(band, spec))
            if not os.path.islink(lnk):
                os.symlink(target, lnk)

    # run the pipeline up to cframes

    pipe.run_steps('extract', 'procexp', rawdir, proddir, nights=[night], comm=None)

    # make bricks

    args = makebricks.parse(['--night', night])
    makebricks.main(args)

    # run redshift fitting

    pipe.run_steps('zfind', 'zfind', rawdir, proddir, nights=[night], comm=None)


    # #-----
    # #- Did it work?
    # #- (this combination of fibermap, simspec, and zbest is a pain)
    # simdir = os.path.dirname(io.findfile('fibermap', night=night, expid=expid))
    # simspec = '{}/simspec-{:08d}.fits'.format(simdir, expid)
    # siminfo = fits.getdata(simspec, 'METADATA')

    # print()
    # print("--------------------------------------------------")
    # print("Brick     True  z        ->  Class  z        zwarn")
    # # print("3338p190  SKY   0.00000  ->  QSO    1.60853   12   - ok")
    # for b in bricks:
    #     zbest = io.read_zbest(io.findfile('zbest', brickname=b))
    #     for i in range(len(zbest.z)):
    #         if zbest.type[i] == 'ssp_em_galaxy':
    #             objtype = 'GAL'
    #         elif zbest.type[i] == 'spEigenStar':
    #             objtype = 'STAR'
    #         else:
    #             objtype = zbest.type[i]

    #         z, zwarn = zbest.z[i], zbest.zwarn[i]

    #         j = np.where(fibermap['TARGETID'] == zbest.targetid[i])[0][0]
    #         truetype = siminfo['OBJTYPE'][j]
    #         truez = siminfo['REDSHIFT'][j]
    #         dv = 3e5*(z-truez)/(1+truez)
    #         if truetype == 'SKY' and zwarn > 0:
    #             status = 'ok'
    #         elif zwarn == 0:
    #             if truetype == 'LRG' and objtype == 'GAL' and abs(dv) < 150:
    #                 status = 'ok'
    #             elif truetype == 'ELG' and objtype == 'GAL' and abs(dv) < 150:
    #                 status = 'ok'
    #             elif truetype == 'QSO' and objtype == 'QSO' and abs(dv) < 750:
    #                 status = 'ok'
    #             elif truetype == 'STD' and objtype == 'STAR':
    #                 status = 'ok'
    #             else:
    #                 status = 'OOPS'
    #         else:
    #             status = 'OOPS'
    #         print('{0}  {1:4s} {2:8.5f}  -> {3:5s} {4:8.5f} {5:4d}  - {6}'.format(
    #             b, truetype, truez, objtype, z, zwarn, status))

    # print("--------------------------------------------------")




if __name__ == '__main__':
    integration_test()
