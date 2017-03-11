"""
Run integration test for quicklook pipeline

python -m desispec.test.integration_test_quicklook
"""
import os
import sys
import desisim.io
import desispec.pipeline as pipe
import desispec.log as logging
from desispec.util import runcmd
from desispec.quicklook import qlconfig

desi_templates_available = 'DESI_ROOT' in os.environ
desi_root_available = 'DESI_ROOT' in os.environ

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
        'DESI_SPECTRO_SIM', 'PIXPROD', 'DESI_PRODUCT_ROOT'):
        if name not in os.environ:
            log.warning("missing ${0}".format(name))
            missing_env = True

    if missing_env:
        log.warning("Why are these needed?")
        log.warning("    Simulations written to $DESI_SPECTRO_SIM/$PIXPROD")
        log.warning("    Raw data read from $DESI_SPECTRO_DATA")
        log.warning("    PSF file is found inside $DESI_PRODUCT_ROOT")
        log.warning("    Templates are read from $DESI_BASIS_TEMPLATES")

    #- Wait until end to raise exception so that we report everything that
    #- is missing before actually failing
    if missing_env:
        log.critical("missing env vars; exiting without running quicklook pipeline")
        sys.exit(1)

def sim(night, nspec, clobber=False):
    """
    Simulate data as part of the integration test.

    Args:
        night (str): YEARMMDD
        nspec (int): number of spectra to include
        clobber (bool, optional): rerun steps even if outputs already exist
 
    Raises:
        RuntimeError if any script fails
    """
    log = logging.get_logger()
    psf_b = os.path.join('$DESI_PRODUCT_ROOT','desimodel','data','specpsf','psf-b.fits')
    psf_r = os.path.join('$DESI_PRODUCT_ROOT','desimodel','data','specpsf','psf-r.fits')
    psf_z = os.path.join('$DESI_PRODUCT_ROOT','desimodel','data','specpsf','psf-z.fits')

    # Create necessary quicklook files

    for expid, flavor in zip([0,1,2], ['arc', 'flat', 'dark']):

        cmd = "newexp-desi --flavor {flavor} --nspec {nspec} --night {night} --expid {expid}".format(expid=expid, flavor=flavor, nspec=nspec, night=night)
        simspec = desisim.io.findfile('simspec', night, expid)
        fibermap = '{}/fibermap-{:08d}.fits'.format(os.path.dirname(simspec),expid)
        if runcmd(cmd, clobber=clobber) != 0:
            raise RuntimeError('newexp failed for {} exposure {}'.format(flavor, expid))

        if flavor=='dark':
            cmd = "pixsim-desi --night {night} --expid {expid} --nspec {nspec}".format(night=night,expid=expid,nspec=nspec)
            if runcmd(cmd, clobber=clobber) != 0:
                raise RuntimeError('pixsim failed for {} exposure {}'.format(flavor, expid))

        if flavor=='arc' or flavor=='flat':
            cmd = "pixsim-desi --night {night} --expid {expid} --nspec {nspec} --preproc".format(night=night,expid=expid,nspec=nspec)
            if runcmd(cmd, clobber=clobber) != 0:
                raise RuntimeError('pixsim failed for {} exposure {}'.format(flavor, expid))

        if flavor=='flat':

            cmd = "desi_extract_spectra -i {}/pix-b0-00000001.fits -o {}/frame-b0-00000001.fits -f {}/fibermap-00000001.fits -p {} -w 3550,5730,0.8 -n {}".format(os.path.dirname(simspec),os.path.dirname(simspec),os.path.dirname(simspec),psf_b,nspec)
            if runcmd(cmd, clobber=clobber) != 0:
                raise RuntimeError('desi_extract_spectra failed for camera b0')

            cmd = "desi_extract_spectra -i {}/pix-r0-00000001.fits -o {}/frame-r0-00000001.fits -f {}/fibermap-00000001.fits -p {} -w 5630,7740,0.8 -n {}".format(os.path.dirname(simspec),os.path.dirname(simspec),os.path.dirname(simspec),psf_r,nspec)
            if runcmd(cmd, clobber=clobber) != 0:
                raise RuntimeError('desi_extract_spectra failed for camera r0')

            cmd = "desi_extract_spectra -i {}/pix-z0-00000001.fits -o {}/frame-z0-00000001.fits -f {}/fibermap-00000001.fits -p {} -w 7650,9830,0.8 -n {}".format(os.path.dirname(simspec),os.path.dirname(simspec),os.path.dirname(simspec),psf_z,nspec)
            if runcmd(cmd, clobber=clobber) != 0:
                raise RuntimeError('desi_extract_spectra failed for camera z0')

    for camera in ['b0','r0','z0']:

        cmd = "desi_compute_fiberflat --infile {}/frame-{}-00000001.fits --outfile {}/fiberflat-{}-00000001.fits".format(os.path.dirname(simspec),camera,os.path.dirname(simspec),camera)
        if runcmd(cmd, clobber=clobber) != 0:
            raise RuntimeError('desi_compute_fiberflat failed for camera {}'.format(camera))

        cmd = "desi_bootcalib --fiberflat {}/pix-{}-00000001.fits --arcfile {}/pix-{}-00000000.fits --outfile {}/psfboot-{}.fits".format(os.path.dirname(simspec),camera,os.path.dirname(simspec),camera,os.path.dirname(simspec),camera)
        if runcmd(cmd, clobber=clobber) != 0:
            raise RuntimeError('desi_bootcalib failed for camera {}'.format(camera))

    return

def integration_test(night="20160728", nspec=5, clobber=False):
    """Run an integration test from raw data simulations through quicklook pipeline

    Args:
        night (str): YEARMMDD
        nspec (int): number of spectra to include
        clobber (bool, optional): rerun steps even if outputs already exist

    Raises:
        RuntimeError if quicklook fails
    """
    log = logging.get_logger()
    log.setLevel(logging.DEBUG)
    datadir = os.path.join('$DESI_SPECTRO_SIM','exposures')

    expid = 2
    flat_expid = 1

    # check for required environment variables and simulate inputs
    check_env()
    sim(night=night, nspec=nspec, clobber=False)

    for camera in ['r0','z0']:

        # find all necessary input and output files
        psffile = os.path.join(datadir,night,'psfboot-{}.fits'.format(camera))
        fiberflatfile = os.path.join(datadir,night,'fiberflat-{}-{:08d}.fits'.format(camera,flat_expid))

        # verify that quicklook pipeline runs
        com = "desi_quicklook -n {} -c {} -e {} -f {} --psfboot {} --fiberflat {} --rawdata_dir {} --specprod_dir {}".format(night,camera,expid,'dark',psffile,fiberflatfile,datadir,'$DESI_SPECTRO_SIM')
        if runcmd(com, clobber=clobber) != 0:
            raise RuntimeError('quicklook pipeline failed for camera {}'.format(camera))

        os.remove('lastframe-{}-{:08d}.fits'.format(camera,expid))

    # remove all output
    output_dir = os.path.join(os.environ['DESI_SPECTRO_SIM'],'exposures',night)
    if os.path.exists(output_dir):
        exp_dir = os.path.join(output_dir,'{:08d}'.format(expid))
        qa_dir = os.path.join(exp_dir,'qa')
        if os.path.exists(qa_dir):
            qa_files = os.listdir(qa_dir)
            for file in range(len(qa_files)):
                qa_file = os.path.join(qa_dir,qa_files[file])
                os.remove(qa_file)
            os.rmdir(qa_dir)
        if os.path.exists(exp_dir):
            exp_files = os.listdir(exp_dir)
            for file in range(len(exp_files)):
                exp_file = os.path.join(exp_dir,exp_files[file])
                os.remove(exp_file)
            os.rmdir(exp_dir)
        output_files = os.listdir(output_dir)
        for file in range(len(output_files)):
            output_file = os.path.join(output_dir,output_files[file])
            os.remove(output_file)
        os.rmdir(output_dir)

if __name__ == '__main__':
    integration_test()

