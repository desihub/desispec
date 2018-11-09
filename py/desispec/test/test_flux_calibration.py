"""
test desispec.fluxcalibration
"""

from __future__ import division

import unittest
import copy

import numpy as np
#import scipy.sparse

#from desispec.maskbits import specmask
from desispec.frame import Frame
from desispec.fluxcalibration import normalize_templates
from desispec.fluxcalibration import FluxCalib
from desispec.fluxcalibration import compute_flux_calibration, apply_flux_calibration
from desiutil.log import get_logger
import desispec.io
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
        normflux=normalize_templates(stdwave,stdflux,mag,'DECAM_R')
        self.assertEqual(stdflux.shape, normflux.shape)

        r = speclite.filters.load_filter('decam2014-r')
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
        ivar = np.ones_like(flux)
        origframe = Frame(wave, flux, ivar, spectrograph=0)

        #define fluxcalib object
        calib = np.ones_like(origframe.flux)
        mask = np.zeros(origframe.flux.shape, dtype=np.uint32)
        calib[0] *= 0.5
        calib[1] *= 1.5

        # fc with essentially no error
        fcivar = 1e20 * np.ones_like(origframe.flux)
        fc = FluxCalib(origframe.wave, calib, fcivar,mask)
        frame = copy.deepcopy(origframe)
        apply_flux_calibration(frame, fc)
        self.assertTrue(np.allclose(frame.ivar, calib**2))

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

    def test_main(self):
        pass



#- This runs all test* functions in any TestCase class in this file
if __name__ == '__main__':
    unittest.main()
