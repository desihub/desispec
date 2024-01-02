"""
tests desispec.sky
"""

import unittest
import pdb

import numpy as np
import os
from desispec.frame import Frame
#from desispec.qa import QA_Frame, QA_Exposure, QA_Brick, QA_Prod
from desispec.qa.qa_frame import QA_Frame
from desispec.qa.qa_exposure import QA_Exposure
from desispec.qa.qa_brick import QA_Brick
from desispec.qa.qa_prod import QA_Prod
from desispec.qa.qa_night import QA_Night
from desispec.io import write_qa_frame, write_qa_brick, load_qa_frame, write_qa_exposure, findfile, write_frame
from desispec.io import write_fiberflat, specprod_root
from desispec.test.util import get_frame_data, get_calib_from_frame, get_fiberflat_from_frame
#from uuid import uuid4
from shutil import rmtree

class TestQA(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.nspec = 6
        cls.nwave = 20
        cls.id = 1
        # Run
        cls.nights = ['20160101']*2 + ['20160102']*2
        cls.expids = [1,2,3,4]
        cls.cameras = ['b0','b1']
        # Files
        cls.files_written = []
        # Paths
        os.environ['DESI_SPECTRO_REDUX'] = os.environ['HOME']
        os.environ['SPECPROD'] = 'desi_test_qa'
        cls.testDir = specprod_root()
        # Files
        cls.qafile_brick = cls.testDir+'/brick/3582m005/qa-3582m005.yaml'
        cls.flux_pdf = cls.testDir+'/exposures/'+cls.nights[0]+'/{:08d}/qa-flux-{:08d}.pdf'.format(cls.id,cls.id)
        cls.frame_pdf = cls.testDir+'/exposures/'+cls.nights[0]+'/{:08d}/qa-frame-{:08d}.pdf'.format(cls.id,cls.id)
        # Files for exposure fibermap QA figure
        cls.exp_fmap_plot = cls.testDir+'/test_exp_fibermap_plot.png'

    @classmethod
    def tearDownClass(cls):
        """Cleanup in case tests crashed and left files behind"""
        for filename in cls.files_written:
            if os.path.exists(filename):
                os.remove(filename)
                #testpath = os.path.normpath(os.path.dirname(filename))
                #if testpath != '.':
                #    os.removedirs(testpath)
        if os.path.exists(cls.testDir):
            rmtree(cls.testDir)

    def _make_frame(self, camera='b0', flavor='science', night=None, expid=None, nspec=3):
        # Init
        if night is None:
            night = self.nights[0]
        if expid is None:
            expid = self.expids[0]
        # Generate
        frame = get_frame_data(nspec=nspec)
        frame.meta = dict(CAMERA=camera, FLAVOR=flavor, NIGHT=night, EXPID=expid)
        if flavor in ('arc', 'flat', 'zero', 'dark'):
            frame.fibermap['OBJTYPE'] = 'CAL'
            frame.fibermap['DESI_TARGET'] = 0

        return frame

    def _write_flat_file(self, camera='b0', night=None, expid=None):
        # Init
        if night is None:
            night = self.nights[0]
        if expid is None:
            expid = self.expids[0]
        # Filename
        frame_file = findfile('frame', night=night, expid=expid, specprod_dir=self.testDir, camera=camera)
        fflat_file = findfile('fiberflat', night=night, expid=expid, specprod_dir=self.testDir, camera=camera)
        # Frames
        fb = self._make_frame(camera=camera, flavor='flat', nspec=10)
        _ = write_frame(frame_file, fb)
        self.files_written.append(frame_file)
        # Fiberflats
        ff = get_fiberflat_from_frame(fb)
        write_fiberflat(fflat_file, ff)
        self.files_written.append(fflat_file)
        # Return
        return frame_file, fflat_file

    def _write_flat_files(self):
        for expid, night in zip(self.expids, self.nights):
            for camera in self.cameras:
                self._write_flat_file(camera=camera, night=night, expid=expid)

    def _write_qaframe(self, camera='b0', expid=1, night='20160101', ZPval=24., flavor='science'):
        """Write QA data frame files"""
        frm = self._make_frame(camera=camera, expid=expid, night=night, flavor=flavor)
        qafrm = QA_Frame(frm)
        # SKY
        qafrm.init_skysub()
        qafrm.qa_data['SKYSUB']['METRICS'] = {}
        qafrm.qa_data['SKYSUB']['METRICS']['NSKY_FIB'] = 10
        # FLUX
        qafrm.init_fluxcalib()
        qafrm.qa_data['FLUXCALIB']['METRICS'] = {}
        qafrm.qa_data['FLUXCALIB']['METRICS']['ZP'] = ZPval
        qafrm.qa_data['FLUXCALIB']['METRICS']['RMS_ZP'] = 0.05
        # Outfile
        qafile = findfile('qa_data', night=night, expid=expid,
                         specprod_dir=self.testDir, camera=camera)
        # WRITE
        write_qa_frame(qafile, qafrm)
        self.files_written.append(qafile)

        # Generate frame too (for QA_Exposure)
        frame = self._make_frame(camera=camera, flavor=flavor, night=night, expid=expid)
        frame_file = findfile('frame', night=night, expid=expid, specprod_dir=self.testDir, camera=camera)
        _ = write_frame(frame_file, frame)
        self.files_written.append(frame_file)
        #
        return qafile

    def _write_qaframes(self, **kwargs):
        """ Build the standard set of qaframes
        and the accompanying frames for QA_Exposure

        Args:
            **kwargs:  passed to _write_qaframe

        Returns:

        """
        for expid, night in zip(self.expids, self.nights):
            for camera in self.cameras:
                self._write_qaframe(camera=camera, expid=expid, night=night, **kwargs)

    def _write_qabrick(self):
        """Write a QA data brick file"""
        qabrck = QA_Brick()
        # REDROCK
        qabrck.init_redrock()
        qabrck.data['REDROCK']['METRICS'] = {}
        qabrck.data['REDROCK']['METRICS']['NFAIL'] = 10
        write_qa_brick(self.qafile_brick, qabrck)
        self.files_written.append(self.qafile_brick)

    def test_init_qa_frame(self):
        #- Simple Init call
        qafrm1 = QA_Frame(self._make_frame(flavor='science'))
        assert qafrm1.flavor == 'science'

    def test_init_qa_fiberflat(self):
        #- Init FiberFlat dict
        qafrm = QA_Frame(self._make_frame(flavor='flat'))
        qafrm.init_fiberflat()
        assert qafrm.qa_data['FIBERFLAT']['PARAMS']['MAX_RMS'] > 0.

        #- ReInit FiberFlat dict
        qafrm.init_fiberflat(re_init=True)
        assert qafrm.qa_data['FIBERFLAT']['PARAMS']['MAX_RMS'] > 0.

    def test_init_qa_fluxcalib(self):
        #- Init FluxCalib dict
        qafrm = QA_Frame(self._make_frame(flavor='science'))
        qafrm.init_fluxcalib()
        assert qafrm.qa_data['FLUXCALIB']['PARAMS']['MAX_ZP_OFF'] > 0.

        #- ReInit FluxCalib dict
        qafrm.init_fluxcalib(re_init=True)
        assert qafrm.qa_data['FLUXCALIB']['PARAMS']['MAX_ZP_OFF'] > 0.

    def test_init_qa_skysub(self):
        #- Init SkySub dict
        qafrm = QA_Frame(self._make_frame(flavor='science'))
        qafrm.init_skysub()
        assert qafrm.qa_data['SKYSUB']['PARAMS']['PCHI_RESID'] > 0.

        #- ReInit SkySub dict
        qafrm.init_skysub(re_init=True)
        assert qafrm.qa_data['SKYSUB']['PARAMS']['PCHI_RESID'] > 0.

    def test_qa_frame_write_load_data(self):
        # Write
        frm0 = self._make_frame()
        qafrm0 = QA_Frame(frm0)
        # Write
        outfile = findfile('qa_data', night=self.nights[0], expid=self.expids[0],
                           specprod_dir=self.testDir, camera='b0')
        write_qa_frame(outfile, qafrm0)
        self.files_written.append(outfile)
        # Load
        qafrm2 = load_qa_frame(outfile, frame_meta=frm0.meta)
        assert qafrm2.night == qafrm0.night


    def test_init_qa_exposure(self):
        """Test simple init.
        """
        from os import environ
        cache_env = {'SPECPROD': None, 'DESI_SPECTRO_REDUX': None}
        for k in cache_env:
            if k in environ:
                cache_env[k] = environ[k]
            environ[k] = './'
        qaexp = QA_Exposure(1, '20150211', flavor='arc')
        self.assertEqual(qaexp.expid, 1)
        for k in cache_env:
            if cache_env[k] is None:
                del environ[k]
            else:
                environ[k] = cache_env[k]

    def test_qa_exposure_load_write_data(self):
        #- Test loading data
        self._write_qaframes()
        expid, night = self.expids[0], self.nights[0]
        qaexp = QA_Exposure(expid, night, specprod_dir=self.testDir)
        assert 'b0' in qaexp.data['frames']
        assert 'b1' in qaexp.data['frames']
        assert qaexp.flavor == 'science'
        # Write
        qafile_exp_file = self.testDir+'/exposures/'+night+'/{:08d}/qa-{:08d}'.format(self.id,self.id)
        write_qa_exposure(qafile_exp_file, qaexp)
        self.files_written.append(qafile_exp_file)

    def test_exposure_fibermap_plot(self):
        from desispec.qa.qa_plots import exposure_fiberflat
        self._write_flat_files()
        exposure_fiberflat('b', self.expids[0], 'meanflux', outfile=self.exp_fmap_plot)
        self.files_written.append(self.exp_fmap_plot)

    """
    # This needs to run as a script for the figure generation to pass Travis..
    def test_qa_exposure_fluxcalib(self):
        #- Perform fluxcalib QA on Exposure (including figure)
        self._write_qaframes()
        qaexp = QA_Exposure(1, self.night, specprod_dir=self.testDir,
                            flavor='dark')
        qaexp.fluxcalib(self.flux_pdf)
    """

    def test_init_qa_brick(self):
        #- Simple Init calls
        qabrck = QA_Brick(name='tst_brick')
        assert qabrck.brick_name == 'tst_brick'
        #
        qabrck.init_redrock()
        assert qabrck.data['REDROCK']['PARAMS']['MAX_NFAIL'] > 0

    def test_init_qa_prod(self):
        self._write_qaframes()
        qaprod = QA_Prod(self.testDir)
        # Load
        qaprod.make_frameqa()
        _ = qaprod.slurp_nights(write_nights=True)
        qaprod.build_data()
        # Build a Table
        tbl = qaprod.get_qa_table('FLUXCALIB', 'RMS_ZP')
        # Test
        assert len(tbl) == 8
        assert tbl['FLAVOR'][0] == 'science'
        assert len(qaprod.qa_nights) == 2
        assert '20160101' in qaprod.mexp_dict.keys()
        assert isinstance(qaprod.data, dict)
        # Load from night JSON QA dicts
        qaprod2 = QA_Prod(self.testDir)
        qaprod2.load_data()
        tbl2 = qaprod.get_qa_table('FLUXCALIB', 'RMS_ZP')
        assert len(tbl2) == 8

    def test_init_qa_night(self):
        self._write_qaframes()  # Generate a set of science QA frames
        night = self.nights[0]
        qanight = QA_Night(night, specprod_dir=self.testDir)
        # Load
        qanight.make_frameqa()
        _ = qanight.slurp()
        qanight.build_data()
        # Build an empty Table
        tbl = qanight.get_qa_table('FIBERFLAT', 'CHI2PDF')
        assert len(tbl) == 0
        # Build a useful Table
        tbl2 = qanight.get_qa_table('FLUXCALIB', 'RMS_ZP')
        # Test
        assert len(tbl2) == 4
        assert tbl2['FLAVOR'][0] == 'science'
        # More tests
        assert len(qanight.qa_exps) == 2
        assert night in qanight.mexp_dict.keys()
        assert isinstance(qanight.data, dict)

    def test_qa_frame_plot(self):
        from desispec.qa import qa_plots
        from desispec.qa import qa_frame
        # Frame
        frame = get_frame_data(500)
        # Load calib
        fluxcalib = get_calib_from_frame(frame)
        # QA Frame
        tdict = {}
        tdict['20190829'] = {}
        dint = 20
        tdict['20190829'][dint] = {}
        tdict['20190829'][dint]['flavor'] = 'science'
        tdict['20190829'][dint]['b'] = {}
        tdict['20190829'][dint]['b']['FLUXCALIB'] = {}
        tdict['20190829'][dint]['b']['FLUXCALIB']['METRICS'] = {}
        tdict['20190829'][dint]['b']['FLUXCALIB']['METRICS']['BLAH'] = 1
        qaframe = qa_frame.QA_Frame(tdict)
        # Plot
        qa_plots.frame_fluxcalib(self.frame_pdf, qaframe, frame, fluxcalib)

    def runTest(self):
        pass


#- This runs all test* functions in any TestCase class in this file
if __name__ == '__main__':
    unittest.main()
    #qa_frame_plot_test()
