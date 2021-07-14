"""
Run integration tests from pixsim through redshifts

python -m desispec.test.old_integration_test
"""

from __future__ import absolute_import, print_function
import os
import time

import numpy as np
from astropy.io import fits

try:
    from scipy import constants
    C_LIGHT = constants.c/1000.0
except TypeError: # This can happen during documentation builds.
    C_LIGHT = 299792458.0/1000.0

from ..util import runcmd
from .. import io
from ..qa import QA_Exposure
from ..database.redshift import get_options, setup_db, load_redrock

from desiutil.log import get_logger

#- prevent nose from trying to run this test since it takes too long
__test__ = False

def check_env():
    """Check required environment variables.

    Raises:
        RuntimeError if any script fails
    """
    log = get_logger()
    #- template locations
    missing_env = False
    if 'DESI_BASIS_TEMPLATES' not in os.environ:
        log.warning('missing $DESI_BASIS_TEMPLATES needed for simulating spectra')
        missing_env = True

    if not os.path.isdir(os.getenv('DESI_BASIS_TEMPLATES')):
        log.warning('missing $DESI_BASIS_TEMPLATES directory')
        log.warning('e.g. see NERSC:/project/projectdirs/desi/spectro/templates/basis_templates/v2.2')
        missing_env = True

    for name in (
        'DESI_SPECTRO_SIM', 'DESI_SPECTRO_REDUX', 'PIXPROD', 'SPECPROD'):
        if name not in os.environ:
            log.warning("missing ${0}".format(name))
            missing_env = True

    if missing_env:
        log.warning("Why are these needed?")
        log.warning("    Simulations written to $DESI_SPECTRO_SIM/$PIXPROD/")
        log.warning("    Raw data read from $DESI_SPECTRO_DATA/")
        log.warning("    Spectro pipeline output written to $DESI_SPECTRO_REDUX/$SPECPROD/")
        log.warning("    Templates are read from $DESI_BASIS_TEMPLATES")
        log.critical("missing env vars; exiting without running pipeline")
        raise RuntimeError("missing env vars; exiting without running pipeline")

    #- Override $DESI_SPECTRO_DATA to match $DESI_SPECTRO_SIM/$PIXPROD
    os.environ['DESI_SPECTRO_DATA'] = os.path.join(os.getenv('DESI_SPECTRO_SIM'), os.getenv('PIXPROD'))


#- TODO: fix usage of night to be something other than today
def integration_test(night=None, nspec=5, clobber=False):
    """Run an integration test from raw data simulations through redshifts.

    Args:
        night (str, optional): YEARMMDD, defaults to current night
        nspec (int, optional): number of spectra to include
        clobber (bool, optional): rerun steps even if outputs already exist

    Raises:
        RuntimeError if any script fails
    """
    from desiutil.iers import freeze_iers
    freeze_iers()

    log = get_logger()
    #- YEARMMDD string, rolls over at noon not midnight
    #- Simulate 8 years ago, prior to start of survey
    if night is None:
        night = time.strftime('%Y%m%d', time.localtime(time.time()-12*3600-(8*365*24*3600)))

    #- check for required environment variables
    check_env()

    #- parameter dictionary that will later be used for formatting commands
    params = dict(night=night, nspec=nspec)

    #-----
    #- Input fibermaps, spectra, and pixel-level raw data
    # raw_dict = {0: 'flat', 1: 'arc', 2: 'dark'}
    programs = ('flat', 'arc', 'dark')
    channels = ('b', 'r', 'z')
    cameras = ('b0', 'r0', 'z0')
    # for expid, program in raw_dict.items():
    for expid, program in enumerate(programs):
        cmd = "newexp-random --program {program} --nspec {nspec} --night {night} --expid {expid}".format(
            expid=expid, program=program, **params)

        fibermap = io.findfile('fibermap', night, expid)
        simspec = '{}/simspec-{:08d}.fits'.format(os.path.dirname(fibermap), expid)
        inputs = []
        outputs = [fibermap, simspec]
        if runcmd(cmd, inputs=inputs, outputs=outputs, clobber=clobber) != 0:
            raise RuntimeError('pixsim newexp failed for {} exposure {}'.format(program, expid))

        cmd = "pixsim --nspec {nspec} --night {night} --expid {expid}".format(expid=expid, **params)
        inputs = [fibermap, simspec]
        outputs = [fibermap.replace('fibermap-', 'simpix-'), ]
        if runcmd(cmd, inputs=inputs, outputs=outputs, clobber=clobber) != 0:
            raise RuntimeError('pixsim failed for {} exposure {}'.format(program, expid))

    #-----
    #- Preproc

    for expid, program in enumerate(programs):
        rawfile = io.findfile('desi', night, expid)
        outdir = os.path.dirname(io.findfile('preproc', night, expid, 'b0'))
        cmd = "desi_preproc --cameras b0,r0,z0 --infile {} --outdir {} --ncpu 1".format(rawfile, outdir)

        inputs = [rawfile,]
        outputs = list()
        for camera in cameras:
            outputs.append(io.findfile('preproc', night, expid, camera))

        if runcmd(cmd, inputs=inputs, outputs=outputs, clobber=clobber) != 0:
            raise RuntimeError('preproc failed for expid {}'.format(expid))

    #-----
    #- Extract

    waverange = dict(b="3570,5940,1.0", r="5630,7740,1.0", z="7440,9830,1.0")
    for expid, program in enumerate(programs):
        for ic, channel in enumerate(channels):
            pixfile = io.findfile('preproc', night, expid, cameras[ic])
            fiberfile = io.findfile('fibermap', night, expid)
            psffile = '{}/data/specpsf/psf-{}.fits'.format(os.getenv('DESIMODEL'), channel)
            framefile = io.findfile('frame', night, expid, cameras[ic])
            # cmd = "exspec -i {pix} -p {psf} --specmin 0 --nspec {nspec} -w {wave} -o {frame}".format(
            #     pix=pixfile, psf=psffile, wave=waverange[channel], frame=framefile, **params)
            cmd = "desi_extract_spectra -i {pix} -p {psf} -f {fibermap} --specmin 0 --nspec {nspec} -o {frame}".format(
                pix=pixfile, psf=psffile, frame=framefile, fibermap=fiberfile, **params)

            inputs = [pixfile, psffile, fiberfile]
            outputs = [framefile,]
            if runcmd(cmd, inputs=inputs, outputs=outputs, clobber=clobber) != 0:
                raise RuntimeError('extraction failed for {} expid {}'.format(cameras[ic], expid))

    #-----
    #- Fiber flat
    expid = 0
    for ic, channel in enumerate(channels):
        framefile = io.findfile('frame', night, expid, cameras[ic])
        fiberflat = io.findfile('fiberflat', night, expid, cameras[ic])
        fibermap = io.findfile('fibermap', night, expid)  # for QA
        qafile = io.findfile('qa_calib', night, expid, cameras[ic])
        qafig = io.findfile('qa_flat_fig', night, expid, cameras[ic])
        cmd = "desi_compute_fiberflat --infile {frame} --outfile {fiberflat} --qafile {qafile} --qafig {qafig}".format(
            frame=framefile, fiberflat=fiberflat, qafile=qafile, qafig=qafig, **params)
        inputs = [framefile,fibermap,]
        outputs = [fiberflat,qafile,qafig,]
        if runcmd(cmd, inputs=inputs, outputs=outputs, clobber=clobber) != 0:
            raise RuntimeError('fiberflat failed for '+cameras[ic])

    #-----
    #- Sky model
    flat_expid = 0
    expid = 2
    for ic, channel in enumerate(channels):
        framefile = io.findfile('frame', night, expid, cameras[ic])
        fibermap = io.findfile('fibermap', night, expid)
        fiberflat = io.findfile('fiberflat', night, flat_expid, cameras[ic])
        skyfile = io.findfile('sky', night, expid, cameras[ic])
        qafile = io.findfile('qa_data', night, expid, cameras[ic])
        qafig = io.findfile('qa_sky_fig', night, expid, cameras[ic])
        cmd="desi_compute_sky --infile {frame} --fiberflat {fiberflat} --outfile {sky} --qafile {qafile} --qafig {qafig}".format(
            frame=framefile, fiberflat=fiberflat, sky=skyfile, qafile=qafile, qafig=qafig, **params)
        inputs = [framefile, fibermap, fiberflat]
        outputs = [skyfile, qafile, qafig,]
        if runcmd(cmd, inputs=inputs, outputs=outputs, clobber=clobber) != 0:
            raise RuntimeError('sky model failed for '+cameras[ic])


    #-----
    #- Fit standard stars
    if 'STD_TEMPLATES' in os.environ:
        std_templates = os.getenv('STD_TEMPLATES')
    else:
        std_templates = os.getenv('DESI_ROOT')+'/spectro/templates/star_templates/v1.1/star_templates_v1.1.fits'

    stdstarfile = io.findfile('stdstars', night, expid, spectrograph=0)
    flats = list()
    frames = list()
    skymodels = list()
    for ic, channel in enumerate(channels):
        frames.append( io.findfile('frame', night, expid, cameras[ic]) )
        flats.append( io.findfile('fiberflat', night, flat_expid, cameras[ic]) )
        skymodels.append( io.findfile('sky', night, expid, cameras[ic]) )

    frames = ' '.join(frames)
    flats = ' '.join(flats)
    skymodels = ' '.join(skymodels)

    cmd = """desi_fit_stdstars \
      --frames {frames} \
      --fiberflats {flats} \
      --skymodels {skymodels} \
      --starmodels {std_templates} \
      -o {stdstars}""".format(
        frames=frames, flats=flats, skymodels=skymodels,
        std_templates=std_templates, stdstars=stdstarfile)

    inputs = [fibermap, std_templates]
    outputs = [stdstarfile,]
    if runcmd(cmd, inputs=inputs, outputs=outputs, clobber=clobber) != 0:
        raise RuntimeError('fitting stdstars failed')


    #-----
    #- Flux calibration
    for ic, channel in enumerate(channels):
        framefile = io.findfile('frame', night, expid, cameras[ic])
        fibermap  = io.findfile('fibermap', night, expid)
        fiberflat = io.findfile('fiberflat', night, flat_expid, cameras[ic])
        skyfile   = io.findfile('sky', night, expid, cameras[ic])
        calibfile = io.findfile('calib', night, expid, cameras[ic])
        qafile = io.findfile('qa_data', night, expid, cameras[ic])
        qafig = io.findfile('qa_flux_fig', night, expid, cameras[ic])

        #- Compute flux calibration vector
        cmd = """desi_compute_fluxcalibration \
          --infile {frame} --fiberflat {fiberflat} --sky {sky} \
          --models {stdstars} --outfile {calib} --qafile {qafile} --qafig {qafig}""".format(
            frame=framefile, fiberflat=fiberflat, sky=skyfile,
            stdstars=stdstarfile, calib=calibfile, qafile=qafile, qafig=qafig
            )
        inputs = [framefile, fibermap, fiberflat, skyfile, stdstarfile]
        outputs = [calibfile, qafile, qafig]
        if runcmd(cmd, inputs=inputs, outputs=outputs, clobber=clobber) != 0:
            raise RuntimeError('flux calibration failed for '+cameras[ic])

        #- Apply the flux calibration to write a cframe file
        cframefile = io.findfile('cframe', night, expid, cameras[ic])
        cmd = """desi_process_exposure \
          --infile {frame} --fiberflat {fiberflat} --sky {sky} --calib {calib} \
          --outfile {cframe}""".format(frame=framefile, fibermap=fibermap,
            fiberflat=fiberflat, sky=skyfile, calib=calibfile, cframe=cframefile)
        inputs = [framefile, fiberflat, skyfile, calibfile]
        outputs = [cframefile, ]
        if runcmd(cmd, inputs=inputs, outputs=outputs, clobber=clobber) != 0:
            raise RuntimeError('combining calibration steps failed for '+cameras[ic])

    #-----
    #- Collate QA
    # Collate data QA
    program2flavor = dict(arc='arc', flat='flat')
    for program in ('dark', 'gray', 'bright', 'elg', 'lrg', 'qso', 'bgs', 'mws'):
        program2flavor[program] = 'science'

    expid = 2
    qafile = io.findfile('qa_data_exp', night, expid)
    if clobber or not os.path.exists(qafile):
        flavor = program2flavor[programs[expid]]
        qaexp_data = QA_Exposure(expid, night, flavor)  # Removes camera files
        io.write_qa_exposure(os.path.splitext(qafile)[0], qaexp_data)
        if not os.path.exists(qafile):
            raise RuntimeError('FAILED data QA_Exposure({},{}, ...) -> {}'.format(expid, night, qafile))
    # Collate calib QA
    calib_expid = [0,1]
    for expid in calib_expid:
        qafile = io.findfile('qa_calib_exp', night, expid)
        if clobber or not os.path.exists(qafile):
            qaexp_calib = QA_Exposure(expid, night, programs[expid])
            io.write_qa_exposure(os.path.splitext(qafile)[0], qaexp_calib)
            if not os.path.exists(qafile):
                raise RuntimeError('FAILED calib QA_Exposure({},{}, ...) -> {}'.format(expid, night, qafile))

    #-----
    #- Regroup cframe -> spectra
    expid = 2
    inputs = list()
    for camera in cameras:
        inputs.append( io.findfile('cframe', night, expid, camera) )

    outputs = list()
    fibermap = io.read_fibermap(io.findfile('fibermap', night, expid))
    from desimodel.footprint import radec2pix
    nside=64
    pixels = np.unique(radec2pix(nside, fibermap['TARGET_RA'], fibermap['TARGET_DEC']))
    for pix in pixels:
        outputs.append( io.findfile('spectra', groupname=pix) )

    cmd = "desi_group_spectra"
    if runcmd(cmd, inputs=inputs, outputs=outputs, clobber=clobber) != 0:
        raise RuntimeError('spectra regrouping failed')

    #-----
    #- Redshifts!
    for pix in pixels:
        specfile = io.findfile('spectra', groupname=pix)
        redrockfile = io.findfile('redrock', groupname=pix)
        inputs = [specfile, ]
        outputs = [redrockfile, ]
        cmd = "rrdesi {} --outfile {}".format(specfile, redrockfile)
        if runcmd(cmd, inputs=inputs, outputs=outputs, clobber=clobber) != 0:
            raise RuntimeError('rrdesi failed for healpixel {}'.format(pix))

    #
    # Load redshifts into database
    #
    options = get_options('--overwrite', '--filename', 'dailytest.db',
                          os.path.join(os.environ['DESI_SPECTRO_REDUX'],
                                       os.environ['SPECPROD']))
    postgresql = setup_db(options)
    load_redrock(options.datapath)
    # ztruth QA
    # qafile = io.findfile('qa_ztruth', night)
    # qafig = io.findfile('qa_ztruth_fig', night)
    # cmd = "desi_qa_zfind --night {night} --qafile {qafile} --qafig {qafig} --verbose".format(
    #     night=night, qafile=qafile, qafig=qafig)
    # inputs = []
    # outputs = [qafile, qafig]
    # if runcmd(cmd, inputs=inputs, outputs=outputs, clobber=clobber) != 0:
    #     raise RuntimeError('redshift QA failed for night '+night)

    #-----
    #- Did it work?
    #- (this combination of fibermap, simspec, and redrock is a pain)
    simdir = os.path.dirname(io.findfile('fibermap', night=night, expid=expid))
    simspec = '{}/simspec-{:08d}.fits'.format(simdir, expid)
    siminfo = fits.getdata(simspec, 'TRUTH')
    try:
        elginfo = fits.getdata(simspec, 'TRUTH_ELG')
    except:
        elginfo = None

    print()
    print("--------------------------------------------------")
    print("Pixel     True  z        -> Class   z        zwarn")
    # print("3338p190  SKY   0.00000  ->  QSO    1.60853   12   - ok")
    for pix in pixels:
        redrock = fits.getdata(io.findfile('redrock', groupname=pix))
        for i in range(len(redrock)):
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
            print('{0:<8d}  {1:4s} {2:8.5f}  -> {3:6s} {4:8.5f} {5:4d}  - {6}'.format(
                pix, truetype, truez, objtype, z, zwarn, status))

    print("--------------------------------------------------")

if __name__ == '__main__':
    from sys import exit
    status = 0
    try:
        integration_test()
    except RuntimeError:
        status = 1
    exit(status)
