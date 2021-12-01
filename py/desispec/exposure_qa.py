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

from desispec.io import findfile,specprod_root,read_fibermap,read_xytraceset,read_stdstar_models,read_frame,read_flux_calibration
from desispec.maskbits import fibermask
from desispec.interpolation import resample_flux
from desispec.tsnr import tsnr2_to_efftime
from desispec.preproc import get_amp_ids,parse_sec_keyword
_qa_params = None
def get_qa_params() :
    """
    Returns a dictionnary with the content of data/qa/qa-params.yaml
    """
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
    returns two tables (astropy.table.Table), fiberqa (with one row per target and at least a TARGETID column)
            and petalqa (with one row per petal and at least a PETAL_LOC column)
    """

    log=get_logger()

    ##################################################################
    qa_params=get_qa_params()["exposure_qa"]
    ##################################################################

    fibermap_filename=f'{specprod_dir}/preproc/{night}/{expid:08d}/fibermap-{expid:08d}.fits'
    if not os.path.isfile(fibermap_filename) :
        log.warning("no {}".format(fibermap_filename))
        return None , None

    fibermap = read_fibermap(fibermap_filename)
    petal_locs=np.unique(fibermap["PETAL_LOC"])

    fiberqa_table = Table()
    for k in ['TARGETID', 'PETAL_LOC', 'DEVICE_LOC', 'LOCATION', 'FIBER', 'TARGET_RA', 'TARGET_DEC',
              'FIBER_X', 'FIBER_Y', 'DELTA_X', 'DELTA_Y', 'EBV'] :
        fiberqa_table[k]=fibermap[k]

    fiberqa_table['QAFIBERSTATUS']=fibermap['FIBERSTATUS']  # copy because content will be different

    fiberqa_table.meta["NIGHT"]=night
    fiberqa_table.meta["EXPID"]=expid
    #fiberqa_table.meta["PRODDIR"]=specprod_dir


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
    fiberqa_table['QAFIBERSTATUS'][nan_positions] |= fibermask.mask('MISSINGPOSITION')

    dist_mm = np.sqrt(dx_mm**2+dy_mm**2)
    poorposition=(dist_mm>qa_params["poor_fiber_offset_mm"])
    fiberqa_table['QAFIBERSTATUS'][poorposition] |= fibermask.mask('POORPOSITION')

    petal_tsnr2=np.zeros(10)
    worst_rdnoise = 0

    fiberqa_table["EFFTIME_SPEC"]=np.zeros(fiberqa_table["TARGETID"].size, dtype=np.float32)

    petalqa_table = Table()
    npetal=10
    petalqa_table["PETAL_LOC"]=np.arange(npetal, dtype=np.int16)
    petalqa_table["WORSTREADNOISE"]=np.zeros(npetal,dtype=np.float32)
    petalqa_table["NGOODPOS"]=np.zeros(npetal,dtype=np.int16)
    petalqa_table["NGOODFIB"]=np.zeros(npetal,dtype=np.int16)
    petalqa_table["NSTDSTAR"]=np.zeros(npetal,dtype=np.int16)
    petalqa_table["STARRMS"]=np.zeros(npetal,dtype=np.float32)
    petalqa_table["TSNR2FRA"]=np.zeros(npetal,dtype=np.float32)
    petalqa_table["EFFTIME_SPEC"]=np.zeros(npetal,dtype=np.float32)
    petalqa_table["NCFRAME"]=np.zeros(npetal,dtype=np.int16)
    petalqa_table["BSKYTHRURMS"]=np.zeros(npetal,dtype=np.float32)
    petalqa_table["BSKYCHI2PDF"]=np.zeros(npetal,dtype=np.float32)
    petalqa_table["RSKYTHRURMS"]=np.zeros(npetal,dtype=np.float32)
    petalqa_table["RSKYCHI2PDF"]=np.zeros(npetal,dtype=np.float32)
    petalqa_table["ZSKYTHRURMS"]=np.zeros(npetal,dtype=np.float32)
    petalqa_table["ZSKYCHI2PDF"]=np.zeros(npetal,dtype=np.float32)
    petalqa_table["BTHRUFRAC"]=np.zeros(npetal,dtype=np.float32)
    petalqa_table["RTHRUFRAC"]=np.zeros(npetal,dtype=np.float32)
    petalqa_table["ZTHRUFRAC"]=np.zeros(npetal,dtype=np.float32)

    # need to add things

    frame_header = None

    # EFFTIME
    goaltype="dark"
    if "GOALTYPE" in fibermap.meta :
        goaltype=fibermap.meta["GOALTYPE"]
    else :
        log.warning("no GOALTYPE info, assume 'dark'")
    param_name="tsnr2_for_efftime_{}".format(goaltype.lower())
    if param_name in qa_params :
        tsnr2_for_efftime_key = qa_params[param_name]
    else :
        tsnr2_for_efftime_key = "TSNR2_ELG"
        log.warning("no parameter '{}', use '{}'".format(param_name,tsnr2_for_efftime_key))

    for petal in petal_locs :
        worst_rdnoise_per_petal = 0

        spectro=petal # same number
        log.info("spectro {}".format(spectro))
        entries = np.where(fiberqa_table['PETAL_LOC'] == petal)[0]

        # checking readnoise level
        ####################################################################
        bad_rdnoise_mask = fibermask.mask('BADREADNOISE')
        max_rdnoise      = qa_params["max_readnoise"]
        for band in ["b","r","z"] :
            camera=f"{band}{spectro}"
            cframe_filename=findfile('cframe',night,expid,camera,specprod_dir=specprod_dir)
            if not os.path.isfile(cframe_filename) :
                continue
            petalqa_table["NCFRAME"][petal]+=1
            head=fitsio.read_header(cframe_filename)
            if frame_header is None :
                frame_header = head

            readnoise_is_bad = False

            amp_ids = get_amp_ids(head)

            for amp in amp_ids :
                rdnoise=head['OBSRDN'+amp]
                worst_rdnoise = max(worst_rdnoise,rdnoise)
                worst_rdnoise_per_petal = max(worst_rdnoise_per_petal,rdnoise)
                if rdnoise > max_rdnoise :
                    log.warning("readnoise is bad in camera {} amplifier {} : {}".format(camera,amp,rdnoise))
                    readnoise_is_bad = True

            petalqa_table["WORSTREADNOISE"][petal]=worst_rdnoise_per_petal

            if readnoise_is_bad :
                log.warning("readnoise is bad in at least one amplifier, flag affected fibers")
                psf_filename=findfile('psf',night,expid,camera,specprod_dir=specprod_dir)
                tset = read_xytraceset(psf_filename)
                fibers = fiberqa_table['FIBER'][entries]%500 # in case the ordering has changed
                x = tset.x_vs_wave(fiber=fibers,wavelength=(tset.wavemin+tset.wavemax)/2.)
                for amp in amp_ids :
                    if head['OBSRDN'+amp] > max_rdnoise :
                        sec = parse_sec_keyword(head['CCDSEC'+amp])
                        ii  = (x>=sec[1].start)&(x<sec[1].stop)
                        fiberqa_table['QAFIBERSTATUS'][entries[ii]] |= bad_rdnoise_mask

        # masks
        ################
        bad_positions_mask = fibermask.mask(qa_params["bad_positions_mask"]) # only positioning issues
        bad_fibers_mask    = fibermask.mask(qa_params["bad_qafstatus_mask"]) # all possible issues


        # checking statistics of positioning
        ####################################################################
        n_bad_positions = np.sum((fiberqa_table['QAFIBERSTATUS'][entries]&bad_positions_mask)>0)
        if n_bad_positions > qa_params["max_frac_of_bad_positions_per_petal"]*entries.size :
            log.warning("petal #{} has {} fibers with bad positions".format(petal,n_bad_positions))
            fiberqa_table['QAFIBERSTATUS'][entries] |= fibermask.mask("BADPETALPOS")

        petalqa_table["NGOODPOS"][petal]=np.sum((fiberqa_table['QAFIBERSTATUS'][entries]&bad_positions_mask)==0)


        # checking standard stars
        ####################################################################
        stdstars_filename = findfile("stdstars",night,expid,spectrograph=spectro,specprod_dir=specprod_dir)
        if os.path.isfile(stdstars_filename) :

            starfibers = fitsio.read(stdstars_filename,'FIBERS')

            # New reductions have list of used standard stars in calibration files
            camera=f"r{spectro}"
            fluxcal_filename=findfile('fluxcalib',night,expid,camera,specprod_dir=specprod_dir)

            if not os.path.isfile(fluxcal_filename) :
                log.warning("no file {}".format(fluxcal_filename))
                continue
            fluxcal = read_flux_calibration(fluxcal_filename)

            cframe_filename=findfile('cframe',night,expid,camera,specprod_dir=specprod_dir)
            if not os.path.isfile(cframe_filename) :
                continue
            cframe = read_frame(cframe_filename)

            if fluxcal.stdstar_fibermap is not None :
                log.info("Use the list of stars from the fluxcalibration file")
                goodfibers  = fluxcal.stdstar_fibermap["FIBER"]
                goodindices = []
                for fiber in goodfibers :
                    goodindices.append( np.where(starfibers==fiber)[0][0])
            else :
                log.info("Apply the same cuts as in compute_flux_calibration to get the list of stars")
                t = fitsio.read(stdstars_filename,'METADATA')
                # SNR cut is same as in stdstars.py, this is redundant,
                # but for clarity on the selection, I repeat the cuts here
                # CHI2DOF and color cut are used in flux calibration
                # generous color cut here.
                good=(t["CHI2DOF"]<2.)&(t["BLUE_SNR"]>=4.)
                if "MODEL_G-R" in t.dtype.names :
                    good &= (np.abs(t["MODEL_G-R"]-t["DATA_G-R"])<0.1) # 0.1 is the selection cut used in prod
                goodindices = np.where(good)[0]
                goodfibers = starfibers[goodindices]

            ngood=goodfibers.size
            petalqa_table["NSTDSTAR"][petal]=ngood

            if ngood < qa_params["min_number_of_good_stdstars_per_petal"] :
                log.warning("petal #{} has only {} good std stars for calibration".format(petal,ngood))
                if ngood <= 1 :
                    fiberqa_table['QAFIBERSTATUS'][entries] |= fibermask.mask("BADPETALSTDSTAR")
                    # else we will keep the data if the few stars we have give similar calibration
            if ngood > 1 :
                log.info("petal #{} has {} good std stars for calibration".format(petal,ngood))

                # measure RMS
                modelwave = fitsio.read(stdstars_filename,'WAVELENGTH')
                modelflux = fitsio.read(stdstars_filename,'FLUX')
                modelflux = modelflux[goodindices]

                log.debug("good fibers = {}".format(goodfibers))
                log.debug("star fibers = {}".format(starfibers))
                log.debug("goodindices = {}".format(goodindices))




                goodfibers_indices=goodfibers%500
                scale=np.zeros(ngood)
                wave=np.linspace(6000,7500,100) # coarse
                for i in range(ngood) :
                    mflux=resample_flux(wave,modelwave,modelflux[i])
                    dflux,ivar=resample_flux(wave,cframe.wave,cframe.flux[goodfibers_indices[i]],cframe.ivar[goodfibers_indices[i]])
                    scale[i] = np.sum(ivar*dflux*mflux)/np.sum(ivar*mflux**2)
                log.debug("scale={}".format(scale))
                calib_rms=np.sqrt(np.mean((scale-1)**2))*np.sqrt(ngood/(ngood-1.))
                petalqa_table["STARRMS"][petal]=calib_rms

                if ngood >= qa_params["min_number_of_good_stdstars_per_petal"] :
                    max_rms = qa_params["max_rms_of_rflux_ratio_of_stdstars"]
                else : # stricter requirement because only few stars
                    max_rms = qa_params["max_rms_of_rflux_ratio_of_stdstars_if_few_stars"]

                if calib_rms>max_rms :
                    log.warning("petal #{} has std stars calib rms={:3.2f}>{:3.2f}".format(petal,calib_rms,qa_params["max_rms_of_rflux_ratio_of_stdstars"]))
                    fiberqa_table['QAFIBERSTATUS'][entries] |= fibermask.mask("BADPETALSTDSTAR")
                else :
                    log.info("petal #{} has std stars calib rms={:3.2f}".format(petal,calib_rms))

        else :
            log.warning("petal #{} does not have a standard star file. expected path='{}'".format(petal,stdstars_filename))
            fiberqa_table['QAFIBERSTATUS'][entries] |= fibermask.mask("BADPETALSTDSTAR")

        # checking fluxcalibration vs GFA?
        ####################################################################


        # record TSNR2
        ####################################################################
        camera="{}{}".format(qa_params["tsnr2_band"],spectro)
        cframe_filename=findfile('cframe',night,expid,camera,specprod_dir=specprod_dir)
        if not os.path.isfile(cframe_filename) :
            continue
        scores = fitsio.read(cframe_filename,"SCORES")
        print(scores.dtype.names)
        tsnr2_key  = qa_params["tsnr2_key"]
        tsnr2_vals = scores[tsnr2_key]
        good = ((fiberqa_table['QAFIBERSTATUS'][entries]&bad_positions_mask)==0)
        petal_tsnr2[petal] = np.median(tsnr2_vals[good])
        log.info("petal #{} median {} = {}".format(petal,tsnr2_key,petal_tsnr2[petal]))

        tsnr2_for_efftime_vals = np.zeros(entries.size)
        for band in ["B","R","Z"] :
             camera="{}{}".format(band.lower(),spectro)
             cframe_filename=findfile('cframe',night,expid,camera,specprod_dir=specprod_dir)
             scores = fitsio.read(cframe_filename,"SCORES")
             tsnr2_for_efftime_vals += scores[tsnr2_for_efftime_key+"_"+band]
        target_type=tsnr2_for_efftime_key.split("_")[1].upper()
        efftime = tsnr2_to_efftime(tsnr2_for_efftime_vals,target_type)
        fiberqa_table['EFFTIME_SPEC'][entries]=efftime
        petalqa_table['EFFTIME_SPEC'][petal]=np.median(efftime)

        # checking sky rms
        ####################################################################
        for band in ["b","r","z"]:
            camera="{}{}".format(band,spectro)
            sky_filename=findfile('sky',night,expid,camera,specprod_dir=specprod_dir)
            if not os.path.isfile(sky_filename) :
                continue
            sky_throughput_corr=fitsio.read(sky_filename,"THRPUTCORR")
            petalqa_table[band.upper()+'SKYTHRURMS'][petal]=1.48*np.median(np.abs(sky_throughput_corr-1))
            log.info("petal #{} {} sky throughput rms={:4.3f}".format(petal,band,petalqa_table[band.upper()+"SKYTHRURMS"][petal]))
            cframe_filename=findfile('cframe',night,expid,camera,specprod_dir=specprod_dir)
            if not os.path.isfile(cframe_filename) :
                continue
            cframe=read_frame(cframe_filename)
            skyfibers=cframe.fibermap["OBJTYPE"]=="SKY"
            chi2=np.sum(cframe.ivar[skyfibers]*cframe.flux[skyfibers]**2*(cframe.mask[skyfibers]==0))
            ndata=np.sum((cframe.ivar[skyfibers]>0)*(cframe.mask[skyfibers]==0))
            npar=cframe.wave.size
            ndf=ndata-npar
            if ndf>0 :
                petalqa_table[band.upper()+"SKYCHI2PDF"][petal]=chi2/ndf
                log.info("petal #{} {} sky chi2pdf={:4.3f}".format(petal,band,petalqa_table[band.upper()+"SKYCHI2PDF"][petal]))

        # check calib
        ####################################################################
        for band in ["b","r","z"]:
            camera="{}{}".format(band,spectro)
            calib_filename=findfile('fluxcalib',night,expid,camera,specprod_dir=specprod_dir)
            if os.path.isfile(calib_filename) :
                # calib value of central fibers, central wavelength
                calib=fitsio.read(calib_filename,0)
                nwave=calib.shape[1]
                # mean of half of the wavelength array to avoid dichroic regions
                cal=np.mean(calib[:,nwave//4:nwave-nwave//4],axis=1)
                # median over fibers because some fibers have large positioning offsets
                cal=np.median(cal)
                petalqa_table[band.upper()+"THRUFRAC"][petal]=cal
            else :
                log.warning("missing {}".format(calib_filename))

    for band in ["b","r","z"]:
        k=band.upper()+"THRUFRAC"
        mval=np.mean(petalqa_table[k][petalqa_table[k]!=0])
        if mval!=0 :
            petalqa_table[k] /= mval
            log.info("{} = {}".format(k,list(petalqa_table[k])))

    petal_tsnr2_frac = np.zeros(npetal)
    if np.all(petal_tsnr2==0) :
         fiberqa_table['QAFIBERSTATUS'] |= fibermask.mask("BADPETALSNR")
         log.error("all petals have TSNR2=0")
    else :
        mean_tsnr2 = np.mean(petal_tsnr2[petal_tsnr2!=0])
        petal_tsnr2_frac = petal_tsnr2/mean_tsnr2
        for petal in petal_locs :
            petalqa_table["TSNR2FRA"][petal]=petal_tsnr2_frac[petal]
        badpetals=(petal_tsnr2_frac<qa_params["tsnr2_petal_minfrac"])|(petal_tsnr2_frac>qa_params["tsnr2_petal_maxfrac"])
        for petal in np.where(badpetals)[0] :
            entries=(fiberqa_table['PETAL_LOC'] == petal)
            fiberqa_table['QAFIBERSTATUS'][entries] |= fibermask.mask("BADPETALSNR")
            log.warning("petal #{} TSNR2 frac = {:3.2f}".format(petal,petal_tsnr2_frac[petal]))

    # count bad fibers
    for petal in petal_locs :
        entries=(fiberqa_table['PETAL_LOC'] == petal)
        petalqa_table["NGOODFIB"][petal]=np.sum((fiberqa_table['QAFIBERSTATUS'][entries]&bad_fibers_mask)==0)

    bad_fibers_mask = fibermask.mask(qa_params["bad_qafstatus_mask"])
    good_fibers = np.where((fiberqa_table['QAFIBERSTATUS']&bad_fibers_mask)==0)[0]
    good_petals = np.unique(fiberqa_table['PETAL_LOC'][good_fibers])
    fiberqa_table.meta["NGOODFIB"]=good_fibers.size
    fiberqa_table.meta["NGOODPET"]=good_petals.size
    fiberqa_table.meta["WORSTRDN"]=worst_rdnoise
    if len(good_fibers) > 0 :
        fiberqa_table.meta["FPRMS2D"]=np.sqrt(np.mean(dist_mm[good_fibers]**2))
        fiberqa_table.meta["PMINEXPF"]=np.min(petal_tsnr2_frac[good_petals])
        fiberqa_table.meta["PMAXEXPF"]=np.max(petal_tsnr2_frac[good_petals])
        fiberqa_table.meta['EFFTIME']=np.mean(petalqa_table['EFFTIME_SPEC'][good_petals])

    if frame_header is not None :
        # copy some keys from the frame header
        keys=["EXPID","TILEID","EXPTIME","MJD-OBS","TARGTRA","TARGTDEC","MOUNTEL","MOUNTHA","AIRMASS","ETCTEFF"]
        for k in keys :
            if k in frame_header :
                fiberqa_table.meta[k] = frame_header[k]

    # copy some keys from the fibermap header
    keys=["TILEID","TILERA","TILEDEC","GOALTIME","GOALTYPE","FAPRGRM","SURVEY","EBVFAC","MINTFRAC"]
    for k in keys :
        if k in fibermap.meta :
            fiberqa_table.meta[k] = fibermap.meta[k]

    return fiberqa_table , petalqa_table
