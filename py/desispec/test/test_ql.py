"""
Test capabilities of  QuickLook pipeline

python -m desispec.test.test_ql
"""
import os
import sys
import shutil
import unittest
from pkg_resources import resource_filename
import desiutil.log as logging
from desispec.util import runcmd

desi_templates_available = 'DESI_ROOT' in os.environ
desi_root_available = 'DESI_ROOT' in os.environ

class TestQL(unittest.TestCase):
    #- Set simulation input
    def setUp(self):
        self.configFile=resource_filename('desispec','data/quicklook/qlconfig_dark.yaml')
        self.night = '20150105'
        self.camera = 'r0'
        self.arcExpid = 312
        self.flatExpid = 313
        self.expid = 314
        self.nspec = 5

        #- Override default environment variables
        self.testDir = os.path.join(os.environ['HOME'],'ql_test_io')
        if 'PIXPROD' in os.environ:
            os.environ['PIXPROD'] = '.'
        if 'DESI_SPECTRO_SIM' in os.environ:
            os.environ['DESI_SPECTRO_SIM'] = self.testDir
        if 'QL_SPEC_DATA' in os.environ:
            os.environ['QL_SPEC_DATA'] = self.testDir
        if 'QL_SPEC_REDUX' in os.environ:
            os.environ['QL_SPEC_REDUX'] = self.testDir

   #- Clean up test files and directories if they exist
    def tearDown(self):
        if os.path.exists(self.testDir):
            shutil.rmtree(self.testDir)

    #- Simulate test inputs for QuickLook
    def sim(self):
        night = self.night
        camera = self.camera
        expid = self.expid
        arcid = self.arcExpid
        flatid = self.flatExpid
        nspec = self.nspec
        simDir = self.testDir

        psf_b = os.path.join(os.environ['DESIMODEL'],'data','specpsf','psf-b.fits')
        psf_r = os.path.join(os.environ['DESIMODEL'],'data','specpsf','psf-r.fits')
        psf_z = os.path.join(os.environ['DESIMODEL'],'data','specpsf','psf-z.fits')

        cmd = "newarc --nspec {} --night {} --expid {} --outdir {}".format(nspec,night,arcid,simDir)
        if runcmd(cmd) != 0:
            raise RuntimeError('newexp failed for arc exposure')
    
        cmd = "newflat --nspec {} --night {} --expid {} --outdir {}".format(nspec,night,flatid,simDir)
        if runcmd(cmd) != 0:
            raise RuntimeError('newexp failed for flat exposure')
    
        cmd = "newexp-random --program dark --nspec {} --night {} --expid {} --outdir {}".format(nspec,night,expid,simDir)
        if runcmd(cmd) != 0:
            raise RuntimeError('newexp failed for dark exposure')
    
        cmd = "pixsim --night {} --cameras {} --expid {} --nspec {} --rawfile {}/desi-00000000.fits.fz --preproc --preproc_dir {}".format(night,camera,arcid,nspec,simDir,simDir)
        if runcmd(cmd) != 0:
            raise RuntimeError('pixsim failed for arc exposure')
    
        cmd = "pixsim --night {} --cameras {} --expid {} --nspec {} --rawfile {}/desi-00000001.fits.fz --preproc --preproc_dir {}".format(night,camera,flatid,nspec,sim0Dir,simDir)
        if runcmd(cmd) != 0:
            raise RuntimeError('pixsim failed for flat exposure')
    
        cmd = "pixsim --night {} --cameras {} --expid {} --nspec {} --rawfile {}/desi-00000002.fits.fz".format(night,camera,expid,nspec,simDir)
        if runcmd(cmd) != 0:
            raise RuntimeError('pixsim failed for dark exposure')

        cmd = "desi_extract_spectra -i {}/pix-r0-00000001.fits -o {}/frame-r0-00000001.fits -f {}/fibermap-00000001.fits -p {} -w 5630,7740,0.8 -n {}".format(simDir,simDir,simDir,psf_r,nspec)
        if runcmd(cmd) != 0:
            raise RuntimeError('desi_extract_spectra failed for camera r0')

        cmd = "desi_compute_fiberflat --infile {}/frame-{}-00000001.fits --outfile {}/fiberflat-{}-00000001.fits".format(simDir,camera,simDir,camera)
        if runcmd(cmd) != 0:
            raise RuntimeError('desi_compute_fiberflat failed for camera {}'.format(camera))

        cmd = "desi_bootcalib --fiberflat {}/pix-{}-00000001.fits --arcfile {}/pix-{}-00000000.fits --outfile {}/psfboot-{}.fits".format(simDir,camera,simDir,camera,simDir,camera)
        if runcmd(cmd) != 0:
            raise RuntimeError('desi_bootcalib failed for camera {}'.format(camera))

    #- Generate inputs
    def test_run_sim(self):
        TestQL.sim(self)

    #- Test if QuickLook outputs merged QA file
    def test_mergeQA(self):
        cmd = "desi_quicklook -i {} -n {} -c {} -e {} --mergeQA".format(self.configFile,self.night,self.camera,self.expid)
        if runcmd(cmd) != 0:
            raise RuntimeError('quicklook pipeline failed'.format(self.camera))


#- This runs all test* functions in any TestCase class in this file
if __name__ == '__main__':
    unittest.main()
