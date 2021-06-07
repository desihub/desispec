"""
desispec.exposure_qa
============

Utility functions to compute an exposure QA scores.
"""

import os,sys
import numpy as np
from astropy.table import Table
import fitsio

from desiutil.log import get_logger

from desispec.io import findfile,specprod_root,read_fibermap,read_xytraceset
from desispec.maskbits import fibermask

def compute_exposure_qa(night, expid, specprod_dir=None):
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
    qa_params={} # would read from yaml file
    qa_params["max_offset_mm"]=0.01
    qa_params["max_readnoise"]=10 # electron
    qa_params["max_frac_of_bad_positions"]=0.5
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
    poorposition=(dist_mm>qa_params["max_offset_mm"])
    table['FIBERSTATUS'][poorposition] |= fibermask.mask('POORPOSITION')

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
                if head['OBSRDN'+amp] > max_rdnoise :
                    log.warning("readnoise is bad in camera {} amplifier {} : {}".format(camera,amp,head['OBSRDN'+amp]))
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
        if n_bad_positions > qa_params["max_frac_of_bad_positions"]*entries.size :
            log.warning("petal #{} has {} fibers with bad positions".format(petal,n_bad_positions))
            table['FIBERSTATUS'][entries] |= fibermask.mask("BADPETALPOS")








    log.warning("empty for now")
    return table
