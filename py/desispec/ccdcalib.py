import os,sys
import datetime
import subprocess

import astropy.io.fits as pyfits
import numpy as np
from scipy.signal import savgol_filter

from desispec import io
from desispec.preproc import masked_median
from desispec.preproc import parse_sec_keyword, _overscan
from desispec.calibfinder import CalibFinder, sp2sm
from desiutil.log import get_logger

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
        if 'GAINA' in img.meta and 'GAINA' not in first_image_header:
            first_image_header['GAINA'] = img.meta['GAINA']
            first_image_header['GAINB'] = img.meta['GAINB']
            first_image_header['GAINC'] = img.meta['GAINC']
            first_image_header['GAIND'] = img.meta['GAIND']

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


def compute_bias_file(rawfiles, outfile, camera):
    """
    Compute a bias file from input ZERO rawfiles

    Args:
        rawfiles: list of input raw file names
        outfile (str): output filename
        camera (str): camera, e.g. b0, r1, z9
    """
    log = get_logger()

    log.info("read images ...")
    images=[]
    shape=None
    first_image_header = None
    for filename in rawfiles :
        log.info("reading %s"%filename)
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

        if cfinder and cfinder.haskey("AMPLIFIERS") :
            amp_ids=list(cfinder.value("AMPLIFIERS"))
        else :
            amp_ids=['A','B','C','D']

        n0=image.shape[0]//2
        n1=image.shape[1]//2

        for a,amp in enumerate(amp_ids) :
            ii = parse_sec_keyword(image_header['BIASSEC'+amp])
            overscan_image = image[ii].copy()
            overscan,rdnoise = _overscan(overscan_image)
            log.info("amp {} overscan = {}".format(amp,overscan))
            if ii[0].start < n0 and ii[1].start < n1 :
                image[:n0,:n1] -= overscan
            elif ii[0].start < n0 and ii[1].start >= n1 :
                image[:n0,n1:] -= overscan
            elif ii[0].start >= n0 and ii[1].start < n1 :
                image[n0:,:n1] -= overscan
            elif ii[0].start >= n0 and ii[1].start >= n1 :
                image[n0:,n1:] -= overscan


        if shape is None :
            shape=image.shape
        images.append(image.ravel())

        fitsfile.close()

    images=np.array(images)
    print(images.shape)

    # compute a mask
    log.info("compute median image ...")
    medimage=np.median(images,axis=0) #.reshape(shape)
    log.info("compute mask ...")
    ares=np.abs(images-medimage)
    nsig=4.
    mask=(ares<nsig*1.4826*np.median(ares,axis=0))
    # average (not median)
    log.info("compute average ...")
    meanimage=np.sum(images*mask,axis=0)/np.sum(mask,axis=0)
    meanimage=meanimage.reshape(shape)

    log.info("write result in %s ..."%outfile)
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
    for filename in rawfiles :
        hdus[0].header["COMMENT"] = "Inc. {}".format(os.path.basename(filename))
    hdus.writeto(outfile,overwrite="True")

    log.info("done")

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

def make_dark_scripts(days, outdir, cameras=None,
        linexptime=None, nskip_zeros=None, tempdir=None, nosubmit=False):
    """
    TODO: document
    """
    log = get_logger()

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
    days = ' '.join([str(tmp) for tmp in days])

    for camera in cameras:
        sp = 'sp' + camera[1]
        sm = sp2sm(sp)
        key = f'{sm}-{camera}-{today}'
        batchfile = os.path.join(tempdir, f'dark-{key}.slurm')
        logfile = os.path.join(tempdir, f'dark-{key}-%j.log')
        darkfile = f'dark-{key}.fits.gz'
        biasfile = f'bias-{key}.fits.gz'

        cmd = f"desi_compute_dark_nonlinear --days {days}"
        cmd += f" \\\n    --camera {camera}"
        cmd += f" \\\n    --tempdir {tempdir}"
        cmd += f" \\\n    --darkfile {darkfile}"
        cmd += f" \\\n    --biasfile {biasfile}"
        if linexptime is not None:
            cmd += f" \\\n    --linexptime {linexptime}"
        if nskip_zeros is not None:
            cmd += f" \\\n    --nskip-zeros {nskip_zeros}"

        with open(batchfile, 'w') as fx:
            fx.write(f'''#!/bin/bash -l

#SBATCH -C haswell
#SBATCH -N 1
#SBATCH --qos realtime
#SBATCH --account desi
#SBATCH --job-name dark-{key}
#SBATCH --output {logfile}
#SBATCH --time=01:00:00
#SBATCH --exclusive

cd {outdir}
time {cmd}
''')

        if not nosubmit:
            err = subprocess.call(['sbatch', batchfile])
            if err == 0:
                log.info(f'Submitted {batchfile}')
            else:
                log.error(f'Error {err} submitting {batchfile}')
        else:
            log.info(f"Generated but didn't submit {batchfile}")

