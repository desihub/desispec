import os, sys, glob, json
import traceback
import datetime
import subprocess
import yaml

import astropy.io.fits as pyfits
from astropy.table import vstack as table_vstack
from astropy.time import Time
import numpy as np
from scipy.signal import savgol_filter

from desispec import io
from desispec.preproc import masked_median
# from desispec.preproc import parse_sec_keyword, calc_overscan
from desispec.preproc import parse_sec_keyword, get_amp_ids
from desispec.preproc import subtract_peramp_overscan
from desispec.calibfinder import CalibFinder, sp2sm, sm2sp
from desispec.io.util import get_tempfilename, parse_cameras, decode_camword, difference_camwords,create_camword
from desispec.workflow.exptable import get_exposure_table_pathname
from desispec.workflow.tableio import load_table, load_tables, write_table

from desiutil.log import get_logger
from desiutil.depend import add_dependencies

from desispec.workflow.batch import get_config


def compute_dark_file(rawfiles, outfile, camera, bias=None, nocosmic=False,
                 scale=False, exptime=None):
    """
    Compute classic dark model from input dark images

    Args:
        rawfiles (list of str): list of input raw data files (desi-*.fits.fz)
        outfile (str): output file with dark model to write
        camera (str): camera to process, e.g. b0, r1, z9

    Options:
        bias (str or list): bias file to use, or list of bias files
        nocosmic (bool): use medians instead of cosmic identification
        scale (bool): apply scale correction for EM0 teststand data
        exptime (float): write EXPTIME header keyword; all inputs must match

    Note: if bias is None, no bias correction is applied.  If it is a single
    file, then use that bias for all darks.  If it is a list, it must have
    len(rawfiles) and gives the per-file bias to use.

    Note: this computes a classic dark model without any non-linear terms.
    see bin/compute_dark_nonlinear for current DESI dark model.

    TODO: separate algorithm from I/O
    """
    log = get_logger()
    log.info("read images ...")

    shape=None
    images=[]
    first_image_header = None
    if nocosmic :
        masks=None
    else :
        masks=[]

    for ifile, filename in enumerate(rawfiles):
        log.info(f'Reading {filename} camera {camera}')

        # collect exposure times
        fitsfile=pyfits.open(filename)
        primary_header = fitsfile[0].header
        if not "EXPTIME" in primary_header :
            primary_header = fitsfile[1].header
        if "EXPREQ" in primary_header :
            thisexptime = primary_header["EXPREQ"]
            log.warning("Using EXPREQ and not EXPTIME, because a more accurate quantity on teststand")
        else :
            thisexptime = primary_header["EXPTIME"]

        flavor = primary_header['FLAVOR'].upper()
        if flavor != 'DARK':
            message = f'Input {filename} flavor {flavor} != DARK'
            log.error(message)
            raise ValueError(message)

        if exptime is not None:
            if round(exptime)  != round(thisexptime):
                message = f'Input {filename} exptime {thisexptime} != requested exptime {exptime}'
                log.error(message)
                raise ValueError(message)

        if first_image_header is None :
            first_image_header = fitsfile[camera].header

        fitsfile.close()

        if bias is not None:
            if isinstance(bias, str):
                thisbias = bias
            elif isinstance(bias, (list, tuple, np.array)):
                thisbias = bias[ifile]
            else:
                message = 'bias should be None, str, list, or tuple, not {}'.format(type(bias))
                log.error(message)
                raise RuntimeError(message)
        else:
            thisbias = False

        # read raw data and preprocess them
        img = io.read_raw(filename, camera, bias=thisbias, nocosmic=nocosmic,
                mask=False, dark=False, pixflat=False)

        # propagate gains to first_image_header
        for a in get_amp_ids(img.meta) :
            k="GAIN"+a
            if k in img.meta and k not in first_image_header:
                first_image_header[k] = img.meta[k]

        if shape is None :
            shape=img.pix.shape
        log.info("adding dark %s divided by exposure time %f s"%(filename,thisexptime))
        images.append(img.pix.ravel()/thisexptime)
        if masks is not None :
            masks.append(img.mask.ravel())

    images=np.array(images)
    if masks is not None :
        masks=np.array(masks)
        smask=np.sum(masks,axis=0)
    else :
        smask=np.zeros(images[0].shape)

    log.info("compute median image ...")
    medimage=masked_median(images,masks)

    if scale :
        log.info("compute a scale per image ...")
        sm2=np.sum((smask==0)*medimage**2)
        ok=(medimage>0.6*np.median(medimage))*(smask==0)
        for i,image in enumerate(rawfiles) :
            s=np.sum((smask==0)*medimage*image)/sm2
            #s=np.median(image[ok]/medimage[ok])
            log.info("image %d scale = %f"%(i,s))
            images[i] /= s
        log.info("recompute median image after scaling ...")
        medimage=masked_median(images,masks)

    if True :
        log.info("compute mask ...")
        ares=np.abs(images-medimage)
        nsig=4.
        mask=(ares<nsig*1.4826*np.median(ares,axis=0))
        # average (not median)
        log.info("compute average ...")
        meanimage=np.sum(images*mask,axis=0)/np.sum(mask,axis=0)
        meanimage=meanimage.reshape(shape)
    else :
        meanimage=medimage.reshape(shape)

    log.info("write result in %s ..."%outfile)
    hdulist=pyfits.HDUList([pyfits.PrimaryHDU(meanimage.astype('float32'))])

    # copy some keywords
    for key in [
        "TELESCOP","INSTRUME","SPECGRPH","SPECID","DETECTOR","CAMERA",
        "CCDNAME","CCDPREP","CCDSIZE","CCDTEMP","CPUTEMP","CASETEMP",
        "CCDTMING","CCDCFG","SETTINGS","VESSEL","FEEVER","FEEBOX",
        "PRESECA","PRRSECA","DATASECA","TRIMSECA","BIASSECA","ORSECA",
        "CCDSECA","DETSECA","AMPSECA",
        "PRESECB","PRRSECB","DATASECB","TRIMSECB","BIASSECB","ORSECB",
        "CCDSECB","DETSECB","AMPSECB",
        "PRESECC","PRRSECC","DATASECC","TRIMSECC","BIASSECC","ORSECC",
        "CCDSECC","DETSECC","AMPSECC",
        "PRESECD","PRRSECD","DATASECD", "TRIMSECD","BIASSECD","ORSECD",
        "CCDSECD","DETSECD","AMPSECD",
        "DAC0","DAC1","DAC2","DAC3","DAC4","DAC5","DAC6","DAC7",
        "DAC8","DAC9","DAC10","DAC11","DAC12","DAC13","DAC14","DAC15",
        "DAC16","DAC17","CLOCK0","CLOCK1","CLOCK2","CLOCK3","CLOCK4",
        "CLOCK5","CLOCK6","CLOCK7","CLOCK8","CLOCK9","CLOCK10",
        "CLOCK11","CLOCK12","CLOCK13","CLOCK14","CLOCK15","CLOCK16",
        "CLOCK17","CLOCK18","OFFSET0","OFFSET1","OFFSET2","OFFSET3",
        "OFFSET4","OFFSET5","OFFSET6","OFFSET7","DELAYS","CDSPARMS",
        "PGAGAIN","OCSVER","DOSVER","CONSTVER",
        "GAINA", "GAINB", "GAINC", "GAIND",
        ] :
        if key in first_image_header :
            hdulist[0].header[key] = (first_image_header[key],first_image_header.comments[key])

    if exptime is not None:
        hdulist[0].header['EXPTIME'] = exptime

    hdulist[0].header["BUNIT"] = "electron/s"
    hdulist[0].header["EXTNAME"] = "DARK"

    for i, filename in enumerate(rawfiles):
        hdulist[0].header["INPUT%03d"%i]=os.path.basename(filename)

    hdulist.writeto(outfile, overwrite=True)
    log.info(f"Wrote {outfile}")

    log.info(f"done")


def compute_bias_file(rawfiles, outfile, camera, explistfile=None,
        extraheader=None):
    """
    Compute a bias file from input ZERO rawfiles

    Args:
        rawfiles: list of input raw file names
        outfile (str): output filename
        camera (str): camera, e.g. b0, r1, z9

    Options:
        explistfile: filename with text list of NIGHT EXPID to use
        extraheader: dict-like key/value header keywords to add

    Notes: explistfile is only used if rawfiles=None; it should have
    one NIGHT EXPID entry per line.
    """
    log = get_logger()

    if explistfile is not None:
        if rawfiles is not None:
            msg = "specify rawfiles or explistfile, but not both"
            log.error(msg)
            raise ValueError(msg)

        rawfiles = list()
        with open(explistfile, 'r') as fx:
            for line in fx:
                line = line.strip()
                if line.startswith('#') or len(line)<2:
                    continue
                night, expid = map(int, line.split())
                filename = io.findfile('raw', night, expid)
                if not os.path.exists(filename):
                    msg = f'Missing {filename}'
                    log.critical(msg)
                    raise RuntimeError(msg)

                rawfiles.append(filename)

    log.info("read %s images ...", camera)
    images=[]
    shape=None
    first_image_header = None
    for filename in rawfiles :
        log.info("reading %s %s", filename, camera)
        fitsfile=pyfits.open(filename)

        primary_header=fitsfile[0].header
        image_header=fitsfile[camera].header

        if first_image_header is None :
            first_image_header = image_header

        flavor = image_header['FLAVOR'].upper()
        if flavor != 'ZERO':
            message = f'Input {filename} flavor {flavor} != ZERO'
            log.error(message)
            raise ValueError(message)

        # subtract overscan region
        cfinder=CalibFinder([image_header,primary_header])

        image=fitsfile[camera].data.astype("float64")

        subtract_peramp_overscan(image, image_header)

        if shape is None :
            shape=image.shape
        images.append(image.ravel())

        fitsfile.close()

    images=np.array(images)
    log.debug('%s images.shape=%s', camera, str(images.shape))

    # compute a mask
    log.info(f"compute median {camera} image ...")
    medimage=np.median(images,axis=0) #.reshape(shape)
    log.info(f"compute {camera} mask ...")
    ares=np.abs(images-medimage)
    nsig=4.
    mask=(ares<nsig*1.4826*np.median(ares,axis=0))
    # average (not median)
    log.info(f"compute {camera} average ...")
    meanimage=np.sum(images*mask,axis=0)/np.sum(mask,axis=0)
    meanimage=meanimage.reshape(shape)

    # cleanup memory
    del images
    del mask
    del medimage

    log.info(f"write {camera} result to {outfile} ...")
    hdus=pyfits.HDUList([pyfits.PrimaryHDU(meanimage.astype('float32'))])

    # copy some keywords
    for key in [
            "TELESCOP","INSTRUME","SPECGRPH","SPECID","DETECTOR","CAMERA",
            "CCDNAME","CCDPREP","CCDSIZE","CCDTEMP","CPUTEMP","CASETEMP",
            "CCDTMING","CCDCFG","SETTINGS","VESSEL","FEEVER","FEEBOX",
            "PRESECA","PRRSECA","DATASECA","TRIMSECA","BIASSECA","ORSECA",
            "CCDSECA","DETSECA","AMPSECA",
            "PRESECB","PRRSECB","DATASECB","TRIMSECB","BIASSECB","ORSECB",
            "CCDSECB","DETSECB","AMPSECB",
            "PRESECC","PRRSECC","DATASECC","TRIMSECC","BIASSECC","ORSECC",
            "CCDSECC","DETSECC","AMPSECC",
            "PRESECD","PRRSECD","DATASECD","TRIMSECD","BIASSECD","ORSECD",
            "CCDSECD","DETSECD","AMPSECD",
            "DAC0","DAC1","DAC2","DAC3","DAC4","DAC5","DAC6","DAC7","DAC8",
            "DAC9","DAC10","DAC11","DAC12","DAC13","DAC14","DAC15","DAC16",
            "DAC17","CLOCK0","CLOCK1","CLOCK2","CLOCK3","CLOCK4","CLOCK5",
            "CLOCK6","CLOCK7","CLOCK8","CLOCK9","CLOCK10","CLOCK11","CLOCK12",
            "CLOCK13","CLOCK14","CLOCK15","CLOCK16","CLOCK17","CLOCK18",
            "OFFSET0","OFFSET1","OFFSET2","OFFSET3","OFFSET4","OFFSET5",
            "OFFSET6","OFFSET7","DELAYS","CDSPARMS","PGAGAIN","OCSVER",
            "DOSVER","CONSTVER"] :
        if key in first_image_header :
            hdus[0].header[key] = (first_image_header[key],first_image_header.comments[key])

    hdus[0].header["BUNIT"] = "adu"
    hdus[0].header["EXTNAME"] = "BIAS"
    if extraheader is not None:
        for key, value in extraheader.items():
            hdus[0].header[key] = value

    add_dependencies(hdus[0].header)

    for filename in rawfiles :
        #- keep only NIGHT/EXPID/filename.fits part of path
        fullpath = os.path.abspath(filename)
        tmp = fullpath.split(os.path.sep)
        shortpath = os.path.sep.join(tmp[-3:])
        hdus[0].header["COMMENT"] = "Inc. {}".format(shortpath)

    #- write via temporary file, then rename
    tmpfile = get_tempfilename(outfile)
    hdus.writeto(tmpfile, overwrite="True")
    os.rename(tmpfile, outfile)

    log.info(f"done with {camera}")

def _find_zeros(night, cameras, nzeros=25, nskip=2, anyzeros=False):
    """Find all OBSTYPE=ZERO exposures on a given night

    Args:
        night (int): YEARMMDD night to search
        cameras (str): list of cameras to process

    Options:
        nzeros (int): number of zeros desired from valid all-cam observations to not worry about partials
        nskip (int): number of initial zeros to skip
        anyzeros (bool): allow any ZEROs, not just those taken for CCD calib seq

    Returns array of expids that are OBSTYPE=ZERO

    Uses production exposure tables to veto known bad ZEROs, but it will also
    find any ZEROs on disk for that night, regardless of whether they are in
    the exposures table or not.
    """

    #- Find all ZERO exposures on this night
    log = get_logger()
    nightdir = io.rawdata_root() + f'/{night}'
    requestfiles = sorted(glob.glob(f'{nightdir}/*/request*.json'))
    expids = list()
    for filename in requestfiles:
        with open(filename) as fx:
            r = json.load(fx)

        #- CALIB ZEROs, or any ZEROs if anyzeros=True,
        #- while being robust to missing OBSTYPE or PROGRAM
        if (('OBSTYPE' in r) and (r['OBSTYPE'] == 'ZERO') and
            ((('PROGRAM' in r) and  r['PROGRAM'].startswith('CALIB ZEROs')) or
             anyzeros)):
            expids.append(int(os.path.basename(os.path.dirname(filename))))
        else:
            continue

    expids = np.array(expids)

    #- drop first two zeros because they are sometimes still stabilizing
    if nskip > 0:
       log.info('Dropping first {} ZEROs: {}'.format(nskip, expids[0:2]))
       expids = expids[nskip:]

    #- Remove ZEROs that are flagged as bad, but allow for the possibility
    #- of ZEROs that aren't in the exposure table for whatever reason
    log.debug('Checking for pre-identified bad ZEROs')
    expfile = get_exposure_table_pathname(night)
    exptable = load_table(expfile, tabletype='exptable')
    select_zeros=exptable['OBSTYPE']=='zero'
    bad = select_zeros & (exptable['LASTSTEP']!='all')
    badcam = select_zeros & (exptable['BADCAMWORD']!='')
    badamp = select_zeros & (exptable['BADAMPS']!='')
    notallcams = select_zeros & (exptable['CAMWORD']!='a0123456789')
    if np.any(bad):
        #this discards observations that are bad for all cams
        drop = np.isin(expids, exptable['EXPID'][bad])
        ndrop = np.sum(drop)
        drop_expids = expids[drop]
        log.info(f'Dropping {ndrop}/{len(expids)} bad ZEROs: {drop_expids}')
        expids = expids[~drop]
        
    if np.any(badcam|badamp|notallcams):
        #do the by spectrograph evaluation of bad spectra
        drop = np.isin(expids, exptable['EXPID'][badcam|badamp|notallcams])
        ndrop = np.sum(drop)
        drop_expids = expids[drop]
        expids = expids[~drop]
        #need lists here so we can append good observations on some spectrographs
        expdict={f'{cam}':list(expids) for cam in cameras}
        if len(expids) >= nzeros:
            #in this case we can just drop all partially bad exposures as we have enough that are good on all cams
            log.info(f'Additionally dropped {ndrop} partially bad ZEROs for all cams because of BADCAM/BADAMP/CAMWORD: {drop_expids}')
        else:
            #in this case we want to recover as many as possible
            log.info(f'additionally dropped {ndrop} bad ZEROs for some cams because of BADCAM/BADAMP/CAMWORD: {drop_expids}')
            
            for expid in drop_expids:
                select_exp=exptable['EXPID']==expid
                badampstring=exptable['BADAMPS'][select_exp][0]
                goodcamword=difference_camwords(exptable['CAMWORD'][select_exp][0],exptable['BADCAMWORD'][select_exp][0])
                goodcamlist=decode_camword(goodcamword)
                for camera in goodcamlist:
                    if camera in cameras and camera not in badampstring:
                        expdict[camera].append(expid)
    else:
        expdict={f'{cam}':expids for cam in cameras}
    
    for camera,expids in expdict.items():
        log.info(f'Keeping {len(expids)} calibration ZEROs for camera {camera}')
        #make sure everything is in np arrays again
        expdict[camera]=np.sort(expids)


    return expdict

def compute_nightly_bias(night, cameras, outdir=None, nzeros=25, minzeros=15,
        nskip=2, anyzeros=False, comm=None):
    """Create nightly biases for cameras on night

    Args:
        night (int): YEARMMDD night to process
        cameras (str): list of cameras to process

    Options:
        outdir (str): write files to this output directory
        nzeros (int): number of OBSTYPE=ZERO exposures to use
        minzeros (int): minimum number of OBSTYPE=ZERO exposures required
        nskip (int): number of initial zeros to skip
        anyzeros (bool): allow any ZEROs, not just those taken for CCD calib seq
        comm: MPI communicator for parallelism

    Returns:
        nfail (int): number of cameras that failed across all ranks

    Writes biasnight*.fits files in outdir or
    $DESI_SPECTRO_REDUX/$SPECPROD/calibnight/night/

    Note: compute_bias_file requires ~12 GB memory per camera, so limit
    the size of the MPI communicator depending upon the memory available.
    """
    #- only import fitsio if needed, not upon package import
    import fitsio

    log = get_logger()

    if comm is not None:
        rank, size = comm.rank, comm.size
    else:
        rank, size = 0, 1

    #- Find all zeros for the night
    expdict = None
    if rank == 0:
        expdict = _find_zeros(night, cameras=cameras, nzeros=nzeros,
                nskip=nskip, anyzeros=anyzeros)
        used_expdict = {}
        for cam,expids in expdict.items():
            if len(expids) < minzeros:
                msg = f'Only {len(expids)} ZEROS on {night} and cam {cam}; need at least {minzeros}'
                log.error(msg)
                continue

            if len(expids) > nzeros:
                nexps = len(expids)
                n = (nexps - nzeros)//2
                used_expdict[cam] = expids[n:n+nzeros]
            else:
                used_expdict[cam] = expids

            log.info(f'Using {len(used_expdict[cam])} ZEROs for nightly bias {night} and cam {cam}')

        if len(used_expdict)==0:
            log.critical("No camera has enough zeros")
            raise RuntimeError("No camera has enough zeros")
        expdict=used_expdict


    if comm is not None:
        expdict = comm.bcast(expdict, root=0)

    #- Rank 0 create output directory if needed
    if rank == 0:
        outfile = io.findfile('biasnight', night=night,
                              camera=cameras[0], outdir=outdir)
        os.makedirs(os.path.dirname(outfile), exist_ok=True)

    #- wait for directory creation before continuing
    if comm is not None:
        comm.barrier()

    nfail = 0
    for camera in cameras[rank::size]:
        if camera not in expdict.keys():
            log.error(f'execution was skipped for camera {camera} due to lack of usable zeros')
            nfail+=1
            continue
        expids=expdict[camera]
        rawfiles=[io.findfile('raw', night, e) for e in expids]

        outfile = io.findfile('biasnight', night=night, camera=camera,
                              outdir=outdir)

        #- write to preliminary file until validated as better than default bias
        head, tail = os.path.split(outfile)
        tail = tail.replace('biasnight-', 'biasnighttest-')
        testbias = os.path.join(head, tail)

        if os.path.exists(outfile):
            log.info(f'{outfile} already exists; skipping')
        elif os.path.exists(testbias):
            log.info(f'{testbias} already exists; skipping')
        else:
            log.info(f'Rank {rank} computing nightly bias for {night} {camera}')
            try:
                compute_bias_file(rawfiles, testbias, camera,
                                  extraheader=dict(NIGHT=night))
            except Exception as ex:
                nfail+=1
                log.error(f'Rank {rank} camera {camera} raised {type(ex)} exception {ex}')
                for line in traceback.format_exception(*sys.exc_info()):
                    log.error('  '+line.strip())

        #- Validate that the new nightlybias is better than default
        if os.path.exists(testbias):
            rawtestfile = rawfiles[-1]
            with fitsio.FITS(rawtestfile) as fx:
                rawhdr = fx['SPEC'].read_header()
                camhdr = fx[camera].read_header()

            cf = CalibFinder([rawhdr, camhdr])
            defaultbias = cf.findfile('BIAS')

            log.info(f'Comparing {night} {camera} nightly bias to {defaultbias} using {os.path.basename(rawtestfile)}')
            mdiff1, mdiff2 = compare_bias(rawtestfile, testbias, defaultbias)
            maxabs1 = np.max(np.abs(mdiff1))
            std1 = np.std(mdiff1)
            maxabs2 = np.max(np.abs(mdiff2))
            std2 = np.std(mdiff2)
            log.info(f'Nightly bias {camera}: maxabsdiff {maxabs1:.2f}, stddev {std1:.2f}')
            log.info(f'Default bias {camera}: maxabsdiff {maxabs2:.2f}, stddev {std2:.2f}')

            if maxabs1 < maxabs2 + 0.5 : # add handicap of 0.5 elec to favor nightly bias that also fixes bad columns
                log.info(f'Selecting nightly bias for {night} {camera}')
                os.rename(testbias, outfile)
            else:
                log.warning(f'Nightly bias worse than default; leaving {testbias} for inspection')

    if comm is not None:
        nfail = np.sum(comm.gather(nfail, root=0))
        nfail = comm.bcast(nfail, root=0)

    if rank == 0 and nfail > 0:
        log.error(f'{nfail}/{len(cameras)} failed')

    return nfail


def compare_bias(rawfile, biasfile1, biasfile2, ny=8, nx=40):
    """Compare rawfile image to bias images in biasfile1 and biasfile2

    Args:
        rawfile: full filepath to desi*.fits.fz raw data file
        biasfile1: filepath to bias model made from OBSTYPE=ZERO exposures
        biasfile2: filepath to bias model made from OBSTYPE=ZERO exposures

    Options:
        ny (even int): number of patches in y (row) direction
        nx (even int): number of patches in x (col) direction

    Returns tuple (mdiff1[ny,nx], mdiff2[ny,nx]) median(image-bias) in patches

    The rawfile camera is derived from the biasfile CAMERA header keyword.

    median(raw-bias) is calculated in ny*nx patches using only the DATASEC
    portion of the images.  Since the DESI CCD bias features tend to vary
    faster with row than column, default patches are (4k/8 x 4k/40) = 500x100.
    """
    #- only import fitsio if needed, not upon package import
    import fitsio

    log = get_logger()
    bias1, biashdr1 = fitsio.read(biasfile1, header=True)
    bias2, biashdr2 = fitsio.read(biasfile2, header=True)

    #- bias cameras must match
    cam1 = biashdr1['CAMERA'].strip().upper()
    cam2 = biashdr2['CAMERA'].strip().upper()
    if cam1 != cam2:
        msg  = f'{biasfile1} camera {cam1} != {biasfile2} camera {cam2}'
        log.critical(msg)
        raise ValueError(msg)

    image, hdr = fitsio.read(rawfile, ext=cam1, header=True)

    #- subtract constant per-amp overscan region
    image = image.astype(float)
    subtract_peramp_overscan(image, hdr)

    #- calculate differences per-amp, thus //2
    ny_groups = ny//2
    nx_groups = nx//2
    diff1 = image - bias1
    diff2 = image - bias2

    median_diff1 = list()
    median_diff2 = list()

    amp_ids = get_amp_ids(hdr)
    for amp in amp_ids:
        ampdiff1 = np.zeros((ny_groups, nx_groups))
        ampdiff2 = np.zeros((ny_groups, nx_groups))
        yy, xx = parse_sec_keyword(hdr['DATASEC'+amp])
        iiy = np.linspace(yy.start, yy.stop, ny_groups+1).astype(int)
        jjx = np.linspace(xx.start, xx.stop, nx_groups+1).astype(int)
        for i in range(ny_groups):
            for j in range(nx_groups):
                aa = slice(iiy[i], iiy[i+1])
                bb = slice(jjx[j], jjx[j+1])

                #- median of differences
                ampdiff1[i,j] = np.median(image[aa,bb] - bias1[aa,bb])
                ampdiff2[i,j] = np.median(image[aa,bb] - bias2[aa,bb])

                #- Note: diff(medians) is less sensitive
                ## ampdiff1[i,j] = np.median(image[aa,bb]) - np.median(bias1[aa,bb])
                ## ampdiff2[i,j] = np.median(image[aa,bb]) - np.median(bias2[aa,bb])

        median_diff1.append(ampdiff1)
        median_diff2.append(ampdiff2)

    #- put back into 2D array by amp
    d1 = median_diff1
    d2 = median_diff2
    mdiff1 = np.vstack([np.hstack([d1[2],d1[3]]), np.hstack([d1[0],d1[2]])])
    mdiff2 = np.vstack([np.hstack([d2[2],d2[3]]), np.hstack([d2[0],d2[2]])])

    assert mdiff1.shape == (ny,nx)
    assert mdiff2.shape == (ny,nx)

    return mdiff1, mdiff2


def compare_dark(preprocfile1, preprocfile2, ny=8, nx=40):
    """Compare preprocessed dark images based on different dark models

    Args:
        preprocfile1: filepath to bias model made from OBSTYPE=ZERO exposures
        preprocfile2: filepath to bias model made from OBSTYPE=ZERO exposures

    Options:
        ny (even int): number of patches in y (row) direction
        nx (even int): number of patches in x (col) direction

    Returns tuple (mdiff1[ny,nx], mdiff2[ny,nx]) median(image-bias) in patches

    median(raw-bias) is calculated in ny*nx patches using only the DATASEC
    portion of the images.  Since the DESI CCD bias features tend to vary
    faster with row than column, default patches are (4k/8 x 4k/40) = 500x100.
    """
    #- only import fitsio if needed, not upon package import
    import fitsio

    log = get_logger()
    diff1, preprochdr1 = fitsio.read(preprocfile1, header=True)
    diff2, preprochdr2 = fitsio.read(preprocfile2, header=True)

    #- bias cameras must match
    cam1 = preprochdr1['CAMERA'].strip().upper()
    cam2 = preprochdr2['CAMERA'].strip().upper()
    if cam1 != cam2:
        msg  = f'{preprocfile1} camera {cam1} != {preprocfile2} camera {cam2}'
        log.critical(msg)
        raise ValueError(msg)

    #- calculate differences per-amp, thus //2
    ny_groups = ny//2
    nx_groups = nx//2
    
    median_diff1 = list()
    median_diff2 = list()

    amp_ids = get_amp_ids(preprochdr1)
    for amp in amp_ids:
        ampdiff1 = np.zeros((ny_groups, nx_groups))
        ampdiff2 = np.zeros((ny_groups, nx_groups))
        #DATASEC is still CCD based, where DETSEC is given the coords in preproc x/y units?
        yy, xx = parse_sec_keyword(preprochdr1['DETSEC'+amp])
        iiy = np.linspace(yy.start, yy.stop, ny_groups+1).astype(int)
        jjx = np.linspace(xx.start, xx.stop, nx_groups+1).astype(int)
        for i in range(ny_groups):
            for j in range(nx_groups):
                aa = slice(iiy[i], iiy[i+1])
                bb = slice(jjx[j], jjx[j+1])

                #- median of differences
                ampdiff1[i,j] = np.median(diff1[aa,bb])
                ampdiff2[i,j] = np.median(diff2[aa,bb])

                #- Note: diff(medians) is less sensitive
                ## ampdiff1[i,j] = np.median(image[aa,bb]) - np.median(bias1[aa,bb])
                ## ampdiff2[i,j] = np.median(image[aa,bb]) - np.median(bias2[aa,bb])

        median_diff1.append(ampdiff1)
        median_diff2.append(ampdiff2)

    #- put back into 2D array by amp
    d1 = median_diff1
    d2 = median_diff2
    mdiff1 = np.vstack([np.hstack([d1[2],d1[3]]), np.hstack([d1[0],d1[2]])])
    mdiff2 = np.vstack([np.hstack([d2[2],d2[3]]), np.hstack([d2[0],d2[2]])])

    assert mdiff1.shape == (ny,nx)
    assert mdiff2.shape == (ny,nx)

    return mdiff1, mdiff2


def fit_const_plus_dark(exp_arr,image_arr):
    """
    fit const + dark*t model given images and exptimes

    Args:
        exp_arr: list of exposure times
        image_arr: list of average dark images

    returns: (const, dark) tuple of images

    NOTE: the image_arr should *not* be divided by the exposure time
    """
    n_images=len(image_arr)
    n0=image_arr[0].shape[0]
    n1=image_arr[0].shape[1]

    # fit dark
    A = np.zeros((2,2))
    b0  = np.zeros((n0,n1))
    b1  = np.zeros((n0,n1))
    for image,exptime in zip(image_arr,exp_arr) :
        res = image
        A[0,0] += 1
        A[0,1] += exptime
        A[1,0] += exptime
        A[1,1] += exptime**2
        b0 += res
        b1 += res*exptime

    # const + exptime * dark
    Ai = np.linalg.inv(A)
    const = Ai[0,0]*b0 + Ai[0,1]*b1
    dark  = Ai[1,0]*b0 + Ai[1,1]*b1

    return const, dark

def fit_dark(exp_arr,image_arr):
    """
    fit dark*t model given images and exptimes

    Args:
        exp_arr: list of exposure times
        image_arr: list of average dark images

    returns: dark_image

    NOTE: the image_arr should *not* be divided by the exposure time
    """
    n_images=len(image_arr)
    n0=image_arr[0].shape[0]
    n1=image_arr[0].shape[1]

    # fit dark
    a = 0
    b  = np.zeros((n0,n1))
    for image,exptime in zip(image_arr,exp_arr) :
        res = image
        a  += exptime**2
        b += res*exptime

    return b/a

def model_y1d(image, smooth=0):
    """
    Model image as a sigma-clipped mean 1D function of row

    Args:
        image: 2D array to model

    Options:
        smooth (int): if >0, Savitzky-Golay filter curve by this window

    Returns 1D model of image with len = image.shape[0]
    """
    ny, nx = image.shape
    median1d = np.median(image, axis=1)
    absdiffimg = np.abs((image.T - median1d).T)
    robust_sigma = 1.4826*np.median(absdiffimg, axis=1)

    keep = (absdiffimg.T < 4*robust_sigma).T
    model1d = np.average(image, weights=keep, axis=1)
    if smooth>0:
        model1d[0:ny//2] = savgol_filter(model1d[0:ny//2], smooth, polyorder=3)
        model1d[ny//2:] = savgol_filter(model1d[ny//2:], smooth, polyorder=3)

    return model1d

def make_dark_scripts(outdir, days=None, nights=None, cameras=None,
                      linexptime=None, nskip_zeros=None, tempdir=None, nosubmit=False,
                      first_expid=None,night_for_name=None, use_exptable=True,queue='realtime',
                      copy_outputs_to_split_dirs=False, prepared_exptable=None, system_name=None):
    """
    Generate batch script to run desi_compute_dark_nonlinear

    Args:
        outdir (str): output directory
        days or nights (list of int): days or nights to include

    Options:
        cameras (list of str): cameras to include, e.g. b0, r1, z9
        linexptime (float): exptime after which dark current is linear
        nskip_zeros (int): number of ZEROs at beginning of day/night to skip
        tempdir (str): tempfile working directory
        nosubmit (bool): generate scripts but don't submit them to batch queue
        first_expid (int): ignore expids prior to this
        use_exptable (bool): use shortened copy of joined exposure tables instead of spectable (need to have right $SPECPROD set)
        queue (str): which batch queue to use for submission
        copy_outputs_to_split_dirs (bool): whether to copy outputs to bias_frames/dark_frames subdirs
        prepared_exptable (exptable): if a table is submitted here, no further spectra will be searched and this will be used instead
        system_name (str): the system for which batch files should be created, defaults to guessing current system

    Args/Options are passed to the desi_compute_dark_nonlinear script
    """
    log = get_logger()
    batch_config=get_config(system_name)

    runtime= 1.5 * batch_config['timefactor']
    runtime_hh = runtime // 60
    runtime_mm = runtime % 60

    if tempdir is None:
        tempdir = os.path.join(outdir, 'temp')
        log.info(f'using tempdir {tempdir}')

    if not os.path.isdir(tempdir):
        os.makedirs(tempdir)

    if not os.path.isdir(outdir):
        os.makdeirs(outdir)

    if cameras is None:
        cameras = list()
        for sp in range(10):
            for c in ['b', 'r', 'z']:
                cameras.append(c+str(sp))

    today = datetime.datetime.now().strftime('%Y%m%d')

    #- Convert list of days to single string to use in command

    lastdayornight=0
    if days is not None:
        days = ' '.join([str(tmp) for tmp in days])
        lastdayornight=sorted(days.split())[-1]
    elif nights is not None:
        nights = ' '.join([str(tmp) for tmp in nights])
        lastdayornight=sorted(nights.split())[-1]
    else:
        msg = 'Must specify days or nights'
        log.critical(msg)
        raise ValueError(msg)

    #- Create exposure log so that N>>1 jobs don't step on each other
    nightlist = [int(tmp) for tmp in nights.split()]

    if prepared_exptable is None:
        if use_exptable:
            #grab all exposures from the exposure log in case some have been marked bad
            #note that some exposures will not be in here, so we'll assume those are all fine
            log.info(f'Using exposure tables for {len(nightlist)} night directories')
            expfiles=[]
            for night in nightlist:
                expfiles.append(get_exposure_table_pathname(night))
            exptables = load_tables(expfiles)
            exptable_all=table_vstack(exptables)
            select = ((exptable_all['OBSTYPE']=='zero')|(exptable_all['OBSTYPE']=='dark'))
            exptable_select=exptable_all[select]
        
        log.info(f'Scanning {len(nightlist)} night directories')
        speclog = io.util.get_speclog(nightlist)
        select_speclog = ((speclog['OBSTYPE']=='ZERO')|(speclog['OBSTYPE']=='DARK'))

        speclog = speclog[select_speclog]
        if use_exptable:
            badcamwords=[]
            laststeps=[]
            badamps=[]
            camwords=[]
            for entry in speclog:
                if entry['EXPID'] in exptable_select['EXPID']:
                    sel=entry['EXPID']==exptable_select['EXPID']
                    badcamwords.append(exptable_select['BADCAMWORD'][sel][0])
                    badamps.append(exptable_select['BADAMPS'][sel][0])
                    camwords.append(exptable_select['CAMWORD'][sel][0])
                    laststeps.append(exptable_select['LASTSTEP'][sel][0])
                else:
                    badcamwords.append("")
                    laststeps.append("all")
                    camwords.append("a0123456789")
                    badamps.append("")
            speclog.add_column(badcamwords,name='BADCAMWORD')
            speclog.add_column(laststeps,name='LASTSTEP')
            speclog.add_column(camwords,name='CAMWORD')
            speclog.add_column(badamps,name='BADAMPS')
    else:
        #TODO: need to check if this works properly, else needs to be adapted
        speclog=prepared_exptable
        speclog.rename_column('MJD-OBS','MJD')
        del speclog['HEADERERR']
        del speclog['EXPFLAG']
        del speclog['COMMENTS']


    speclog['OBSTYPE']=np.char.upper(speclog['OBSTYPE'])

    t = Time(speclog['MJD']-7/24, format='mjd')
    speclog['DAY'] = t.strftime('%Y%m%d').astype(int)
    speclogfile = os.path.join(tempdir, 'speclog.csv')
    
    tmpfile = speclogfile + '.tmp-' + str(os.getpid())
    speclog.write(tmpfile, format='ascii.csv')
    os.rename(tmpfile, speclogfile)
    log.info(f'Wrote speclog to {speclogfile}')

    n_jobs_per_script = int(batch_config['cores_per_node']//32)
    #create scripts that do up to 4 cameras in parallel
    batch_opts = list()
    if 'batch_opts' in batch_config:
        for opt in batch_config['batch_opts']:
            batch_opts.append(f'#SBATCH {opt}')
    batch_opts = '\n'.join(batch_opts)
    n_scripts=int(len(cameras)//n_jobs_per_script)
    if len(cameras)%n_jobs_per_script !=0:
        n_scripts+=1
    for scriptid in range(n_scripts):
        if night_for_name is not None :
            job_filename_key=f'scriptnumber-{scriptid}-{night_for_name}'
        else:
            job_filename_key = f'scriptnumber-{scriptid}-{lastdayornight}'

        batchfile = os.path.join(tempdir, f'dark-{job_filename_key}.slurm')
        logfile = os.path.join(tempdir, f'dark-{job_filename_key}-%j.log')
        #header
        with open(batchfile, 'w') as fx:
            fx.write(f'''#!/bin/bash -l
#SBATCH -N 1
#SBATCH --qos {queue}
#SBATCH --account desi
#SBATCH --job-name dark-{job_filename_key}
#SBATCH --output {logfile}
#SBATCH --time={f"{runtime_hh:02d}:{runtime_mm:02d}:00" if queue!="debug" else "00:30:00"}
#SBATCH --exclusive
{batch_opts}

cd {outdir}''')

        minind=scriptid*n_jobs_per_script
        maxind=(scriptid+1)*n_jobs_per_script
        if maxind>len(cameras):
            maxind=len(cameras)
        darkfile_list=[]
        biasfile_list=[]
        for camera in cameras[minind:maxind]:
            sp = 'sp' + camera[1]
            sm = sp2sm(sp)
            #key = f'{sm}-{camera}-{today}'
            if night_for_name is not None :
                key = f'{sm}-{camera}-{night_for_name}'
            else :
                key = f'{sm}-{camera}-{lastdayornight}'
            darkfile = f'dark-{key}.fits.gz'
            biasfile = f'bias-{key}.fits.gz'

            darkfile_list.append(darkfile)
            biasfile_list.append(biasfile)
            cmd = f"desi_compute_dark_nonlinear"
            cmd += f" \\\n    --camera {camera}"
            cmd += f" \\\n    --tempdir {tempdir}"
            cmd += f" \\\n    --darkfile {darkfile}"
            cmd += f" \\\n    --biasfile {biasfile}"
            if days is not None:
                cmd += f" \\\n    --days {days}"
            if nights is not None:
                cmd += f" \\\n    --nights {nights}"
            if linexptime is not None:
                cmd += f" \\\n    --linexptime {linexptime}"
            if nskip_zeros is not None:
                cmd += f" \\\n    --nskip-zeros {nskip_zeros}"
            if first_expid is not None:
                cmd += f" \\\n    --first-expid {first_expid}"

            with open(batchfile, 'a') as fx:
                fx.write(f"time {cmd} &")
        
        with open(batchfile, 'a') as fx:
            fx.write("wait")
            for darkfile,biasfile in zip(darkfile_list,biasfile_list):
                if copy_outputs_to_split_dirs:
                    fx.write(f"""
    cp {darkfile}  dark_frames/{darkfile}
    cp {biasfile}  bias_frames/{biasfile}
    """)

        if not nosubmit:
            err = subprocess.call(['sbatch', batchfile])
            if err == 0:
                log.info(f'Submitted {batchfile}')
            else:
                log.error(f'Error {err} submitting {batchfile}')
        else:
            log.info(f"Generated but didn't submit {batchfile}")


def make_biweekly_darks(outdir=None, lastnight=None, cameras=None, window=30,
                      linexptime=None, nskip_zeros=None, tempdir=None, nosubmit=False,
                      first_expid=None,night_for_name=None, use_exptable=True,queue='realtime',
                      copy_outputs_to_split_dirs=None, transmit_obslist = True, system_name=None):
    """
    Generate batch script to run desi_compute_dark_nonlinear

    Options:
        outdir (str): output directory
        lastnight (int): last night to take into account (inclusive), defaults to tonight

        window (int): length of time window to take into account
        cameras (list of str): cameras to include, e.g. b0, r1, z9
        linexptime (float): exptime after which dark current is linear
        nskip_zeros (int): number of ZEROs at beginning of day/night to skip
        tempdir (str): tempfile working directory
        nosubmit (bool): generate scripts but don't submit them to batch queue
        first_expid (int): ignore expids prior to this
        use_exptable (bool): use shortened copy of joined exposure tables instead of spectable (need to have right $SPECPROD set)
        queue (str): which batch queue to use for submission
        transmit_obslist(bool): if True will give use the obslist from here downstream
        system_name(str): allows to overwrite the system for which slurm scripts are created, will default to guessing the current system

    Args/Options are passed to the desi_compute_dark_nonlinear script
    """
    log = get_logger()

    if lastnight is None:
        lastnight=datetime.datetime.now().strftime('%Y%m%d')
    if outdir is None:
        outdir=os.getenv('DESI_SPECTRO_DARK')
    if tempdir is None:
        tempdir=outdir+f'/temp_{lastnight}'
    if copy_outputs_to_split_dirs is None:
        copy_outputs_to_split_dirs = True

    #probably run a script here that updates the obslist or checks it's up-to-date

    obslist=load_table(f"{os.getenv('DESI_SPECTRO_DARK')}/exp_dark_zero.csv")

    #TODO: nights does not yet end with lastnight need fix
    startnight=datetime.datetime.strptime(str(lastnight),'%Y%m%d')-datetime.timedelta(days=window)
    nights = [int((startnight+datetime.timedelta(days=i)).strftime('%Y%m%d')) for i in range(window)]


    #TODO: the following steps should probably be done by spectrograph and then marking spectrographs that changed in between as bad for nights before the change (allowing other spectrographs to still use more data...)
    #this could probably use calibfinder instead to find the setups...

    #read all calib files to get dates of changes
    yaml_filenames=glob.glob(os.getenv('DESI_SPECTRO_CALIB')+'/spec/sm*/*.yaml')
    all_config_data={}
    for y_file in yaml_filenames:
        with open(y_file) as f:
            y_data=yaml.safe_load(f)
        all_config_data.update(y_data)

    #extract only the main keys which are dates except for the very first one (could elsewise check on OBS-BEGIN), only mildly more complicated
    change_dates={k:[] for k in all_config_data.keys()}
    for speckey,data in all_config_data.items():
        required_keys=[(k,{k2:v2 for (k2,v2) in v.items() if k2 in ['DATE-OBS-BEGIN','DATE-OBS-END','DETECTOR','CCDTMING','CCDCFG','AMPLIFIERS']}) for k,v in data.items()]
        required_keys.sort(key=lambda x:x[1]['DATE-OBS-BEGIN'],reverse=True)
        usever,useval=required_keys[0]
        for newver,newval in required_keys[1:]:
            usenew=True
            for key in ['DETECTOR','CCDTMING','CCDCFG','AMPLIFIERS']:
                if useval[key]!=newval[key]:
                    usenew = False
                    break
            
            if not usenew:
                change_dates[speckey].append(int(useval['DATE-OBS-BEGIN']))
                useval = newval
                usever = newver
            useval = newval
            usever = newver
    change_dates_any_spectrograph=sorted(np.unique([int(d) for v in change_dates.values() for d in v]))   #this is to not overcomplicate things by tracking per detector yet

    nights = [n for n in nights if n in obslist['NIGHT']]
    if len(nights)==0:
        log.critical("No darks were taken for this time frame, exiting")
        sys.exit(1)
    change_dates_in_nights=[d for d in change_dates_any_spectrograph if d<max(nights) and d>min(nights)]

    #change_dates_relevant={k:v for k,v in change_dates.items() if v in change_dates_in_nights}
    change_dates_relevant={}
    for speckey,dates in change_dates.items():
        dates_relevant= [date for date in dates if date in change_dates_in_nights]
        if len(dates_relevant)>0:
            dates_relevant.sort()
            change_dates_relevant[speckey]=dates_relevant[-1]

    obslist=obslist[[o['NIGHT'] in nights for o in obslist]]
    for i,o in enumerate(obslist):
        for speckey, date in change_dates_relevant.items():
            if o['NIGHT']<date:
                badcamword_decoded=decode_camword(o['BADCAMWORD'])
                spec=sm2sp(speckey.split('-')[0])
                color=speckey[-1]
                mask_sp=f"{color}{spec[-1]}"
                if mask_sp not in badcamword_decoded:
                    badcamword_decoded.append(mask_sp)
                badcamword_encoded=create_camword(badcamword_decoded)
                obslist[i]['BADCAMWORD']=badcamword_encoded

    #if len(change_dates_in_nights)>0:
    #    nights = [n for n in nights if n >= max(change_dates_in_nights)]

    #truncate to the right nights
    if transmit_obslist:
        obslist=obslist[[o['NIGHT'] in nights for o in obslist]]
        if nskip_zeros is None:
            nskip_zeros = 0
    else:
        obslist = None

    #TODO: potentially need to do further selections based on quality, but should not be needed as this is done in desi_compute_dark_nonlinear

    #could in principle parse the obslist for all relevant nights and give to the make_dark_scripts...

    make_dark_scripts(outdir, nights=nights, cameras=cameras,
                      linexptime=linexptime, nskip_zeros=nskip_zeros, tempdir=tempdir, nosubmit=nosubmit,
                      first_expid=first_expid,night_for_name=night_for_name, use_exptable=use_exptable,queue=queue,
                      copy_outputs_to_split_dirs=copy_outputs_to_split_dirs,prepared_exptable=obslist, system_name=system_name)