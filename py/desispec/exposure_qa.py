"""
desispec.exposure_qa
============

Utility functions to compute an exposure QA scores.
"""

import os,sys
import numpy as np
from astropy.table import Table
import fitsio
import yaml
from pkg_resources import resource_filename

from desiutil.log import get_logger

from desispec.io import findfile,specprod_root,read_fibermap,read_xytraceset,read_stdstar_models,read_frame
from desispec.maskbits import fibermask
from desispec.interpolation import resample_flux

# only read it once per process
_qa_params = None
def get_qa_params() :
    global _qa_params
    if _qa_params is None :
        param_filename =resource_filename('desispec', 'data/qa/qa-params.yaml')
        with open(param_filename) as f:
            _qa_params = yaml.safe_load(f)
    return _qa_params

def compute_exposure_qa(night, expid, specprod_dir):
    """
    Computes the exposure_qa
    Args:
       night: int, YYYYMMDD
       expid: int, exposure id
       specprod_dir: str, optional, specify the production directory.
                     default is $DESI_SPECTRO_REDUX/$SPECPROD
    returns an astropy.table.Table with one row per target and at least a TARGETID column
    """

    log=get_logger()

    ##################################################################
    qa_params=get_qa_params()["exposure_qa"]
    ##################################################################

    fibermap_filename=f'{specprod_dir}/preproc/{night}/{expid:08d}/fibermap-{expid:08d}.fits'
    if not os.path.isfile(fibermap_filename) :
        log.warning("no {}".format(fibermap_filename))
        return Table()

    fibermap = read_fibermap(fibermap_filename)
    print(fibermap.dtype.names)
    petal_locs=np.unique(fibermap["PETAL_LOC"])

    table = Table()
    for k in ['TARGETID', 'PETAL_LOC', 'DEVICE_LOC', 'LOCATION', 'FIBER', 'FIBERSTATUS', 'TARGET_RA', 'TARGET_DEC',
              'FIBER_X', 'FIBER_Y', 'DELTA_X', 'DELTA_Y'] :
        table[k]=fibermap[k]

    table.meta["NIGHT"]=night
    table.meta["EXPID"]=expid
    #table.meta["PRODDIR"]=specprod_dir


    x_mm  = fibermap["FIBER_X"]
    y_mm  = fibermap["FIBER_Y"]
    dx_mm = fibermap["DELTA_X"]
    dy_mm = fibermap["DELTA_Y"]

    nan_positions = np.isnan(x_mm)|np.isnan(y_mm)
    x_mm[nan_positions]=0.
    y_mm[nan_positions]=0.

    nan_positions |= np.isnan(dx_mm)|np.isnan(dy_mm)
    dx_mm[nan_positions]=0.
    dy_mm[nan_positions]=0.

    # nan = no data
    table['FIBERSTATUS'][nan_positions] |= fibermask.mask('MISSINGPOSITION')

    dist_mm = np.sqrt(dx_mm**2+dy_mm**2)
    poorposition=(dist_mm>qa_params["max_fiber_offset_mm"])
    table['FIBERSTATUS'][poorposition] |= fibermask.mask('POORPOSITION')

    petal_tsnr2=np.zeros(10)
    worst_rdnoise = 0

    for petal in petal_locs :
        spectro=petal # same number
        log.info("spectro {}".format(spectro))
        entries = np.where(table['PETAL_LOC'] == petal)[0]

        # checking readnoise level
        ####################################################################
        bad_rdnoise_mask = fibermask.mask('BADREADNOISE')
        max_rdnoise      = qa_params["max_readnoise"]
        for band in ["b","r","z"] :
            camera=f"{band}{spectro}"
            cframe_filename=findfile('cframe',night,expid,camera,specprod_dir=specprod_dir)
            head=fitsio.read_header(cframe_filename)

            readnoise_is_bad = False
            for amp in ["A","B","C","D"] :
                rdnoise=head['OBSRDN'+amp]
                worst_rdnoise = max(worst_rdnoise,rdnoise)
                if rdnoise > max_rdnoise :
                    log.warning("readnoise is bad in camera {} amplifier {} : {}".format(camera,amp,rdnoise))
                    readnoise_is_bad = True


            if readnoise_is_bad :

                rdnoise_left  = max(head['OBSRDNA'],head['OBSRDNC'])
                rdnoise_right = max(head['OBSRDNB'],head['OBSRDND'])

                log.warning("readnoise is bad in at least one amplifier, flag affected fibers")
                psf_filename=findfile('psf',night,expid,camera,specprod_dir=specprod_dir)
                tset = read_xytraceset(psf_filename)
                twave=np.linspace(tset.wavemin,tset.wavemax,20)
                xtrans=float(head['CCDSIZE'].split(',')[0])/2.
                xfiber=tset.x_vs_wave(fiber=np.arange(tset.nspec),wavelength=twave)[:,0]
                print(xfiber.shape)
                if rdnoise_left>max_rdnoise :
                    table['FIBERSTATUS'][entries[xfiber<xtrans]] |= bad_rdnoise_mask
                elif rdnoise_right>max_rdnoise :
                    table['FIBERSTATUS'][entries[xfiber>=xtrans]] |= bad_rdnoise_mask

        # checking statistics of positioning
        ####################################################################
        bad_positions = fibermask.mask("STUCKPOSITIONER|BROKENFIBER|RESTRICTED|MISSINGPOSITION|BADPOSITION|POORPOSITION")
        n_bad_positions = np.sum((table['FIBERSTATUS'][entries]&bad_positions)>0)
        if n_bad_positions > qa_params["max_frac_of_bad_positions_per_petal"]*entries.size :
            log.warning("petal #{} has {} fibers with bad positions".format(petal,n_bad_positions))
            table['FIBERSTATUS'][entries] |= fibermask.mask("BADPETALPOS")

        # checking standard stars
        ####################################################################
        stdstars_filename = findfile("stdstars",night,expid,spectrograph=spectro,specprod_dir=specprod_dir)
        if os.path.isfile(stdstars_filename) :
            t = fitsio.read(stdstars_filename,'METADATA')
            # SNR cut is same as in stdstars.py, this is redundant,
            # but for clarity on the selection, I repeat the cuts here
            # CHI2DOF and color cut are used in flux calibration
            # generous color cut here.
            good=(t["CHI2DOF"]<2.)&(t["BLUE_SNR"]>=4.)&(np.abs(t["MODEL_G-R"]-t["DATA_G-R"])<0.3)
            ngood=np.sum(good)
            if ngood < qa_params["min_number_of_good_stdstars_per_petal"] :
                log.warning("petal #{} has only {} good std stars for calibration".format(petal,ngood))
                table['FIBERSTATUS'][entries] |= fibermask.mask("BADPETALSTDSTAR")
            else :
                log.info("petal #{} has {} good std stars for calibration".format(petal,ngood))

                # measure RMS
                goodindices = np.where(good)[0]
                modelwave = fitsio.read(stdstars_filename,'WAVELENGTH')
                modelflux = fitsio.read(stdstars_filename,'FLUX')
                modelflux = modelflux[goodindices]

                starfibers = fitsio.read(stdstars_filename,'FIBERS')
                goodfibers = starfibers[goodindices]%500

                camera=f"r{spectro}"
                cframe_filename=findfile('cframe',night,expid,camera,specprod_dir=specprod_dir)
                cframe = read_frame(cframe_filename)
                frameflux=cframe.flux[goodfibers]
                scale=np.zeros(ngood)
                wave=np.linspace(6000,7500,100) # coarse
                #import matplotlib.pyplot as plt
                for i in range(ngood) :
                    mflux=resample_flux(wave,modelwave,modelflux[i])
                    dflux,ivar=resample_flux(wave,cframe.wave,cframe.flux[goodfibers[i]],cframe.ivar[goodfibers[i]])
                    #plt.plot(cframe.wave,frameflux[i])
                    #plt.plot(wave,dflux,"o",color="C{}".format(i))
                    #plt.plot(wave,mflux,"-",color="C{}".format(i))
                    scale[i] = np.sum(dflux*mflux)/np.sum(mflux**2)
                #print(scale)
                calib_rms=np.sqrt(np.mean((scale-1)**2))
                if calib_rms>qa_params["max_rms_of_rflux_ratio_of_stdstars"] :
                    log.warning("petal #{} has std stars calib rms={:3.2f}>{:3.2f}".format(petal,calib_rms,qa_params["max_rms_of_rflux_ratio_of_stdstars"]))
                    table['FIBERSTATUS'][entries] |= fibermask.mask("BADPETALSTDSTAR")
                else :
                    log.info("petal #{} has std stars calib rms={:3.2f}".format(petal,calib_rms))

                #plt.show()



        else :
            log.warning("petal #{} does not have a standard star file. expected path='{}'".format(petal,stdstars_filename))
            table['FIBERSTATUS'][entries] |= fibermask.mask("BADPETALSTDSTAR")

        # checking fluxcalibration vs GFA?
        ####################################################################

        # record TSNR2
        ####################################################################
        camera="{}{}".format(qa_params["tsnr2_band"],spectro)
        cframe_filename=findfile('cframe',night,expid,camera,specprod_dir=specprod_dir)
        scores = fitsio.read(cframe_filename,"SCORES")
        tsnr2_key  = qa_params["tsnr2_key"]
        tsnr2_vals = scores[tsnr2_key]
        good = ((table['FIBERSTATUS'][entries]&bad_positions)==0)
        petal_tsnr2[petal] = np.median(tsnr2_vals[good])
        log.info("petal #{} median {} = {}".format(petal,tsnr2_key,petal_tsnr2[petal]))

    if np.all(petal_tsnr2==0) :
         table['FIBERSTATUS'] |= fibermask.mask("BADPETALTSNR")
         log.error("all petals have TSNR2=0")
    else :
        mean_tsnr2 = np.mean(petal_tsnr2[petal_tsnr2!=0])
        petal_tsnr2_frac = petal_tsnr2/mean_tsnr2
        badpetals=(petal_tsnr2_frac<qa_params["tsnr2_petal_minfrac"])|(petal_tsnr2_frac>qa_params["tsnr2_petal_maxfrac"])
        for petal in np.where(badpetals)[0] :
            entries=(table['PETAL_LOC'] == petal)
            table['FIBERSTATUS'][entries] |= fibermask.mask("BADPETALSNR")
            log.warning("petal #{} TSNR2 frac = {:3.2f}".format(petal,petal_tsnr2_frac[petal]))

    bad_fibers_mask=bad_positions
    good_fibers = np.where((table['FIBERSTATUS']&bad_fibers_mask)==0)[0]
    good_petals = np.unique(table['PETAL_LOC'][good_fibers])
    table.meta["NGOODFIBERS"]=good_fibers.size
    table.meta["NGOODPETALS"]=good_petals.size
    table.meta["WORSTREADNOISE"]=worst_rdnoise
    table.meta["FPRMS2D"]=np.sqrt(np.mean(dist_mm[good_fibers]**2))
    table.meta["PETALMINEXPFRAC"]=np.min(petal_tsnr2_frac[good_petals])
    table.meta["PETALMAXEXPFRAC"]=np.max(petal_tsnr2_frac[good_petals])

    return table
