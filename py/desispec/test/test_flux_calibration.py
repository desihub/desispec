"""
test desispec.fluxcalibration
"""

from __future__ import division

import unittest
import copy

import numpy as np
#import scipy.sparse

from desispec.maskbits import fibermask
from desispec.frame import Frame
from desispec.fluxcalibration import normalize_templates
from desispec.fluxcalibration import FluxCalib
from desispec.fluxcalibration import compute_flux_calibration, apply_flux_calibration
from desiutil.log import get_logger
import desispec.io
from desispec.io.filters import load_legacy_survey_filter
from desispec.test.util import get_frame_data, get_models
from desitarget.targetmask import desi_mask

import speclite.filters


class TestFluxCalibration(unittest.TestCase):

    def test_match_templates(self):
        """
        Test with simple interface check for matching best templates with the std star flux
        """
        from desispec.fluxcalibration import match_templates
        frame=get_frame_data()
        # first define dictionaries
        flux={"b":frame.flux,"r":frame.flux*1.1,"z":frame.flux*1.2}
        wave={"b":frame.wave,"r":frame.wave+10,"z":frame.wave+20}
        ivar={"b":frame.ivar,"r":frame.ivar/1.1,"z":frame.ivar/1.2}
        # resol_data={"b":np.mean(frame.resolution_data,axis=0),"r":np.mean(frame.resolution_data,axis=0),"z":np.mean(frame.resolution_data,axis=0)}
        resol_data={"b":frame.resolution_data,"r":frame.resolution_data,"z":frame.resolution_data}

        #model

        nmodels = 10
        modelwave,modelflux=get_models(nmodels)
        teff = np.random.uniform(5000, 7000, nmodels)
        logg = np.random.uniform(4.0, 5.0, nmodels)
        feh = np.random.uniform(-2.5, -0.5, nmodels)
        # say there are 3 stdstars
        stdfibers=np.random.choice(9,3,replace=False)
        frame.fibermap['OBJTYPE'][stdfibers] = 'STD'

        #pick fluxes etc for each stdstars find the best match
        bestid=-np.ones(len(stdfibers))
        bestwave=np.zeros((bestid.shape[0],modelflux.shape[1]))
        bestflux=np.zeros((bestid.shape[0],modelflux.shape[1]))
        red_chisq=np.zeros(len(stdfibers))

        for i in range(len(stdfibers)):

            stdflux={"b":flux["b"][i],"r":flux["r"][i],"z":flux["z"][i]}
            stdivar={"b":ivar["b"][i],"r":ivar["r"][i],"z":ivar["z"][i]}
            stdresol_data={"b":resol_data["b"][i],"r":resol_data["r"][i],"z":resol_data["z"][i]}

            bestid, redshift, chi2 = \
                match_templates(wave, stdflux, stdivar, stdresol_data,
                    modelwave, modelflux, teff, logg, feh)

            #- TODO: come up with assertions for new return values

    def test_normalize_templates(self):
        """
        Test for normalization to a given magnitude for calibration
        """

        stdwave=np.linspace(3000,11000,10000)
        stdflux=np.cos(stdwave)+100.
        mag = 20.0
        normflux=normalize_templates(stdwave,stdflux,mag,'R','S')
        self.assertEqual(stdflux.shape, normflux.shape)

        r = load_legacy_survey_filter('R','S')
        rmag = r.get_ab_magnitude(1e-17*normflux, stdwave)
        self.assertAlmostEqual(rmag, mag)

    def test_compute_fluxcalibration(self):
        """ Test compute_fluxcalibration interface
        """

        #get frame data
        frame=get_frame_data()
        #get model data
        modelwave,modelflux=get_models()
        # pick std star fibers
        stdfibers=np.random.choice(9,3,replace=False) # take 3 std stars fibers
        frame.fibermap['DESI_TARGET'][stdfibers] = desi_mask.STD_FAINT

        input_model_wave=modelwave
        input_model_flux=modelflux[0:3] # assuming the first three to be best models,3 is exclusive here
        fluxCalib =compute_flux_calibration(frame, input_model_wave,input_model_flux,input_model_fibers=stdfibers,nsig_clipping=4.)
        # assert the output
        self.assertTrue(np.array_equal(fluxCalib.wave, frame.wave))
        self.assertEqual(fluxCalib.calib.shape,frame.flux.shape)

        #- nothing should be masked for this test case
        self.assertFalse(np.any(fluxCalib.mask))

    def test_outliers(self):
        '''Test fluxcalib when input starts with large outliers'''
        frame = get_frame_data()
        modelwave, modelflux = get_models()
        nstd = 5
        frame.fibermap['OBJTYPE'][0:nstd] = 'STD'
        nstd = np.count_nonzero(frame.fibermap['OBJTYPE'] == 'STD')

        frame.flux[0] = np.mean(frame.flux[0])        
        fluxCalib = compute_flux_calibration(frame, modelwave, modelflux[0:nstd],input_model_fibers=np.arange(nstd))

    def test_masked_data(self):
        """Test compute_fluxcalibration with some ivar=0 data
        """
        frame = get_frame_data()
        modelwave, modelflux = get_models()
        nstd = 1
        frame.fibermap['OBJTYPE'][2:2+nstd] = 'STD'
        frame.ivar[2:2+nstd, 20:22] = 0

        fluxCalib = compute_flux_calibration(frame, modelwave, modelflux[2:2+nstd], input_model_fibers=np.arange(2,2+nstd), debug=True)
        
        self.assertTrue(np.array_equal(fluxCalib.wave, frame.wave))
        self.assertEqual(fluxCalib.calib.shape,frame.flux.shape)

    def test_apply_fluxcalibration(self):
        #get frame_data
        wave = np.arange(5000, 6000)
        nwave = len(wave)
        nspec = 3
        flux = np.random.uniform(0.9, 1.0, size=(nspec, nwave))
        ivar = np.random.uniform(0.9, 1.1, size=flux.shape)
        origframe = Frame(wave, flux.copy(), ivar.copy(), spectrograph=0)

        # efine fluxcalib object
        calib = np.random.uniform(.5, 1.5, size=origframe.flux.shape)
        mask = np.zeros(origframe.flux.shape, dtype=np.uint32)

        ivar_big = 1e20 * np.ones_like(origframe.flux)

        # fc with essentially no error
        fc = FluxCalib(origframe.wave, calib, ivar_big, mask)
        frame = copy.deepcopy(origframe)
        apply_flux_calibration(frame, fc)
        self.assertTrue(np.allclose(frame.ivar, calib**2 * ivar))

        # spectrum with essentially no error
        # but large calibration error
        # in this case the S/N should be the same as of the
        # calibration vector
        fc = FluxCalib(origframe.wave, calib, ivar, mask)
        frame = copy.deepcopy(origframe)
        frame.ivar = ivar_big
        apply_flux_calibration(frame, fc)
        self.assertTrue(np.allclose(frame.flux**2 * frame.ivar,
                                    calib**2 * ivar))

        # origframe.flux=0 should result in frame.flux=0
        fcivar = np.ones_like(origframe.flux)
        calib = np.ones_like(origframe.flux)
        fc = FluxCalib(origframe.wave, calib, fcivar, mask)
        frame = copy.deepcopy(origframe)
        frame.flux[0,0:10]=0.0
        apply_flux_calibration(frame, fc)
        self.assertTrue(np.all(frame.flux[0, 0:10] == 0.0))

        #fcivar=0 should result in frame.ivar=0
        fcivar=np.ones_like(origframe.flux)
        calib=np.ones_like(origframe.flux)
        fcivar[0,0:10]=0.0
        fc=FluxCalib(origframe.wave,calib,fcivar,mask)
        frame=copy.deepcopy(origframe)
        apply_flux_calibration(frame,fc)
        self.assertTrue(np.all(frame.ivar[0,0:10]==0.0))

        # should also work even the calib =0  ??
        #fcivar=np.ones_like(origframe.flux)
        #calib=np.ones_like(origframe.flux)
        #fcivar[0,0:10]=0.0
        #calib[0,0:10]=0.0
        #fc=FluxCalib(origframe.wave,calib,fcivar,mask)
        #frame=copy.deepcopy(origframe)
        #apply_flux_calibration(frame,fc)
        #self.assertTrue(np.all(frame.ivar[0,0:10]==0.0))

        # test different wavelength bins
        frame=copy.deepcopy(origframe)
        calib = np.ones_like(frame.flux)
        fcivar=np.ones_like(frame.ivar)
        mask=np.zeros(origframe.flux.shape, dtype=np.uint32)
        fc=FluxCalib(origframe.wave+0.01,calib,fcivar,mask)
        with self.assertRaises(SystemExit):  #should be ValueError instead?
            apply_flux_calibration(frame,fc)

    def test_isStdStar(self):
        """test isStdStar works for cmx, main, and sv1 fibermaps"""
        from desispec.fluxcalibration import isStdStar
        from desitarget.targetmask import desi_mask, mws_mask
        from desitarget.sv1.sv1_targetmask import desi_mask as sv1_desi_mask
        from desitarget.sv1.sv1_targetmask import mws_mask as sv1_mws_mask
        from desitarget.cmx.cmx_targetmask import cmx_mask
        from desitarget.targets import main_cmx_or_sv
        from astropy.table import Table

        #- CMX
        fm = Table()
        fm['CMX_TARGET'] = np.zeros(10, dtype=int)
        fm['CMX_TARGET'][0:2] = cmx_mask.STD_FAINT
        fm['CMX_TARGET'][2:4] = cmx_mask.SV0_STD_FAINT
        fm['FIBERSTATUS'] = 0
        self.assertEqual(main_cmx_or_sv(fm)[2], 'cmx')
        self.assertEqual(np.count_nonzero(isStdStar(fm)), 4)

        #- SV1
        fm = Table()
        fm['SV1_DESI_TARGET'] = np.zeros(10, dtype=int)
        fm['SV1_MWS_TARGET'] = np.zeros(10, dtype=int)
        fm['SV1_DESI_TARGET'][0:2] = sv1_desi_mask.STD_FAINT
        fm['SV1_MWS_TARGET'][2:4] = sv1_mws_mask.GAIA_STD_FAINT
        fm['FIBERSTATUS'] = 0
        self.assertEqual(main_cmx_or_sv(fm)[2], 'sv1')
        self.assertEqual(np.count_nonzero(isStdStar(fm)), 4)

        #- Main
        fm = Table()
        fm['DESI_TARGET'] = np.zeros(10, dtype=int)
        fm['MWS_TARGET'] = np.zeros(10, dtype=int)
        fm['DESI_TARGET'][0:2] = desi_mask.STD_FAINT
        fm['DESI_TARGET'][2:4] = desi_mask.STD_BRIGHT
        fm['DESI_TARGET'][4:6] |= desi_mask.MWS_ANY
        fm['MWS_TARGET'][4:6] = sv1_mws_mask.GAIA_STD_FAINT
        fm['FIBERSTATUS'] = 0
        self.assertEqual(main_cmx_or_sv(fm)[2], 'main')
        self.assertEqual(np.count_nonzero(isStdStar(fm)), 6)
        self.assertEqual(np.count_nonzero(isStdStar(fm, bright=False)), 4)
        self.assertEqual(np.count_nonzero(isStdStar(fm, bright=True)), 2)

        #- VARIABLETHRU should be excluded
        fm['FIBERSTATUS'] = 0
        ii = isStdStar(fm)
        fm['FIBERSTATUS'][0] = fibermask.VARIABLETHRU
        jj = isStdStar(fm)
        self.assertTrue(ii[0])
        self.assertFalse(jj[0])



    def test_main(self):
        pass
