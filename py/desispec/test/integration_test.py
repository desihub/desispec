"""
Run integration tests from pixsim through redshifts

python -m desispec.test.integration_test
"""
from __future__ import absolute_import, print_function
import sys
import os
import random
import time
import subprocess as sp
import glob
import shutil

import numpy as np
from astropy.io import fits

try:
    from scipy import constants
    C_LIGHT = constants.c/1000.0
except TypeError: # This can happen during documentation builds.
    C_LIGHT = 299792458.0/1000.0

from desispec.util import runcmd
import desispec.pipeline as pipe
import desispec.io as io
import desiutil.log as logging

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
        log.warning('missing $DESI_BASIS_TEMPLATES needed for simulating spectra')
        missing_env = True
    elif not os.path.isdir(os.getenv('DESI_BASIS_TEMPLATES')):
        log.warning('missing $DESI_BASIS_TEMPLATES directory')
        log.warning('e.g. see NERSC:/project/projectdirs/desi/spectro/templates/basis_templates/v2.2')
        missing_env = True

    if 'DESI_SPECTRO_CALIB' not in os.environ:
        log.warning('missing $DESI_SPECTRO_CALIB needed for preprocessing images and PSF starting point')
        missing_env = True
    elif not os.path.isdir(os.getenv('DESI_SPECTRO_CALIB')):
        log.warning('missing $DESI_SPECTRO_CALIB directory')
        log.warning('e.g. see NERSC:/project/projectdirs/desi/spectro/desi_spectro_calib/trunk')
        missing_env = True

    for name in (
        'DESI_SPECTRO_SIM', 'DESI_SPECTRO_REDUX', 'PIXPROD', 'SPECPROD', 'DESIMODEL'):
        if name not in os.environ:
            log.warning("missing ${0}".format(name))
            missing_env = True

    if missing_env:
        log.warning("Why are these needed?")
        log.warning("    Simulations written to $DESI_SPECTRO_SIM/$PIXPROD/")
        log.warning("    Raw data read from $DESI_SPECTRO_DATA/")
        log.warning("    Spectro pipeline output written to $DESI_SPECTRO_REDUX/$SPECPROD/")
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

    for expid, program in zip([0,1,2], ['flat', 'arc', 'dark']):
        cmd = "newexp-random --program {program} --nspec {nspec} --night {night} --expid {expid}".format(
            expid=expid, program=program, nspec=nspec, night=night)
        fibermap = io.findfile('fibermap', night, expid)
        simspec = '{}/simspec-{:08d}.fits'.format(os.path.dirname(fibermap), expid)
        inputs = []
        outputs = [fibermap, simspec]
        if runcmd(cmd, inputs=inputs, outputs=outputs, clobber=clobber) != 0:
            raise RuntimeError('newexp-random failed for {} exposure {}'.format(program, expid))

        cmd = "pixsim --nspec {nspec} --night {night} --expid {expid}".format(expid=expid, nspec=nspec, night=night)
        inputs = [fibermap, simspec]
        outputs = list()
        outputs.append(fibermap.replace('fibermap-', 'simpix-'))
        outputs.append(io.findfile('raw', night, expid))
        if runcmd(cmd, inputs=inputs, outputs=outputs, clobber=clobber) != 0:
            raise RuntimeError('pixsim failed for {} exposure {}'.format(program, expid))

    return

def run_pipeline_step(tasktype):
    """Convenience wrapper to run a pipeline step"""
    #- First count the number of tasks that are ready
    log = logging.get_logger()

    dbpath = io.get_pipe_database()
    db = pipe.load_db(dbpath, mode="r")
    task_count = db.count_task_states(tasktype)
    count_string = ', '.join(['{:2d} {}'.format(x[1], x[0]) for x in task_count.items()])

    nready = task_count['ready']
    if nready > 0:
        log.info('{:16s}: {}'.format(tasktype, count_string))
        com = "desi_pipe tasks --tasktypes {tasktype} | grep -v DEBUG | desi_pipe script --shell".format(tasktype=tasktype)
        log.info('Running {}'.format(com))
        script = sp.check_output(com, shell=True)
        log.info('Running {}'.format(script))
        sp.check_call(script, shell=True)
    else:
        log.warning('{:16s}: {} -- SKIPPING'.format(tasktype, count_string))

def integration_test(night=None, nspec=5, clobber=False):
    """Run an integration test from raw data simulations through redshifts

    Args:
        night (str, optional): YEARMMDD, defaults to current night
        nspec (int, optional): number of spectra to include
        clobber (bool, optional): rerun steps even if outputs already exist

    Raises:
        RuntimeError if any script fails

    """

    import argparse
    parser = argparse.ArgumentParser(usage = "{prog} [options]")
    # parser.add_argument("-i", "--input", type=str,  help="input data")
    # parser.add_argument("-o", "--output", type=str,  help="output data")
    parser.add_argument("--skip-psf", action="store_true", help="Skip PSF fitting step")
    args = parser.parse_args()

    from desiutil.iers import freeze_iers
    freeze_iers()

    log = logging.get_logger()

    # YEARMMDD string, rolls over at noon not midnight
    if night is None:
        night = "20160726"

    # check for required environment variables
    check_env()

    # simulate inputs
    sim(night, nspec=nspec, clobber=clobber)

    # raw and production locations

    rawdir = os.path.abspath(io.rawdata_root())
    proddir = os.path.abspath(io.specprod_root())

    # create production

    if clobber and os.path.isdir(proddir):
        shutil.rmtree(proddir)

    dbfile = io.get_pipe_database()
    if not os.path.exists(dbfile):
        com = "desi_pipe create --db-sqlite"
        log.info('Running {}'.format(com))
        sp.check_call(com, shell=True)
    else:
        log.info("Using pre-existing production database {}".format(dbfile))

    # Modify options file to restrict the spectral range

    optpath = os.path.join(proddir, "run", "options.yaml")
    opts = pipe.prod.yaml_read(optpath)
    opts['extract']['specmin'] = 0
    opts['extract']['nspec'] = nspec
    opts['psf']['specmin'] = 0
    opts['psf']['nspec'] = nspec
    opts['traceshift']['nfibers'] = nspec
    pipe.prod.yaml_write(optpath, opts)

    if args.skip_psf:
        #- Copy desimodel psf into this production instead of fitting psf
        import shutil
        for channel in ['b', 'r', 'z']:
            refpsf = '{}/data/specpsf/psf-{}.fits'.format(
                    os.getenv('DESIMODEL'), channel)
            nightpsf = io.findfile('psfnight', night, camera=channel+'0')
            shutil.copy(refpsf, nightpsf)
            for expid in [0,1,2]:
                exppsf = io.findfile('psf', night, expid, camera=channel+'0')
                shutil.copy(refpsf, exppsf)

        #- Resync database to current state
        dbpath = io.get_pipe_database()
        db = pipe.load_db(dbpath, mode="w")
        db.sync(night)

    # Run the pipeline tasks in order
    from desispec.pipeline.tasks.base import default_task_chain
    for tasktype in default_task_chain:
        #- if we skip psf/psfnight/traceshift, update state prior to extractions
        if tasktype == 'traceshift' and args.skip_psf:
            db.getready()
        run_pipeline_step(tasktype)

    # #-----
    # #- Did it work?
    # #- (this combination of fibermap, simspec, and redrock is a pain)
    expid = 2
    fmfile = io.findfile('fibermap', night=night, expid=expid)
    fibermap = io.read_fibermap(fmfile)
    simdir = os.path.dirname(fmfile)
    simspec = '{}/simspec-{:08d}.fits'.format(simdir, expid)
    siminfo = fits.getdata(simspec, 'TRUTH')
    try:
        elginfo = fits.getdata(simspec, 'TRUTH_ELG')
    except:
        elginfo = None

    from desimodel.footprint import radec2pix
    nside=64
    pixels = np.unique(radec2pix(nside, fibermap['TARGET_RA'], fibermap['TARGET_DEC']))

    num_missing = 0
    for pix in pixels:
        zfile = io.findfile('redrock', groupname=pix)
        if not os.path.exists(zfile):
            log.error('Missing {}'.format(zfile))
            num_missing += 1

    if num_missing > 0:
        log.critical('{} redrock files missing'.format(num_missing))
        sys.exit(1)

    print()
    print("--------------------------------------------------")
    print("Pixel     True  z        ->  Class  z        zwarn")
    # print("3338p190  SKY   0.00000  ->  QSO    1.60853   12   - ok")
    for pix in pixels:
        zfile = io.findfile('redrock', groupname=pix)
        if not os.path.exists(zfile):
            log.error('Missing {}'.format(zfile))
            continue

        zfx = fits.open(zfile, memmap=False)
        redrock = zfx['REDSHIFTS'].data
        for i in range(len(redrock['Z'])):
            objtype = redrock['SPECTYPE'][i]
            z, zwarn = redrock['Z'][i], redrock['ZWARN'][i]

            j = np.where(fibermap['TARGETID'] == redrock['TARGETID'][i])[0][0]
            truetype = siminfo['OBJTYPE'][j]
            oiiflux = 0.0
            if truetype == 'ELG':
                k = np.where(elginfo['TARGETID'] == redrock['TARGETID'][i])[0][0]
                oiiflux = elginfo['OIIFLUX'][k]

            truez = siminfo['REDSHIFT'][j]
            dv = C_LIGHT*(z-truez)/(1+truez)
            status = None
            if truetype == 'SKY' and zwarn > 0:
                status = 'ok'
            elif truetype == 'ELG' and zwarn > 0 and oiiflux < 8e-17:
                status = 'ok ([OII] flux {:.2g})'.format(oiiflux)
            elif zwarn == 0:
                if truetype == 'LRG' and objtype == 'GALAXY' and abs(dv) < 150:
                    status = 'ok'
                elif truetype == 'ELG' and objtype == 'GALAXY':
                    if abs(dv) < 150:
                        status = 'ok'
                    elif oiiflux < 8e-17:
                        status = 'ok ([OII] flux {:.2g})'.format(oiiflux)
                    else:
                        status = 'OOPS ([OII] flux {:.2g})'.format(oiiflux)
                elif truetype == 'QSO' and objtype == 'QSO' and abs(dv) < 750:
                    status = 'ok'
                elif truetype in ('STD', 'FSTD') and objtype == 'STAR':
                    status = 'ok'
                else:
                    status = 'OOPS'
            else:
                status = 'OOPS'
            print('{0:<8d}  {1:4s} {2:8.5f}  -> {3:5s} {4:8.5f} {5:4d}  - {6}'.format(
                pix, truetype, truez, objtype, z, zwarn, status))

    print("--------------------------------------------------")


if __name__ == '__main__':
    integration_test()
