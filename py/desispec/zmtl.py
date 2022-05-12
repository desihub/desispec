"""
desispec.zmtl
=============

Post-redrock ML processing for LyA Quasar object identification for
creating zmtl files to be used by the merged target list (MTL) updates.
"""

import os
import numpy as np
import time
import fitsio

from operator import itemgetter
from itertools import groupby
from astropy.modeling import fitting
from astropy.modeling import models
from astropy.table import Table
from scipy.signal import medfilt
from astropy.convolution import Gaussian1DKernel
from astropy.convolution import convolve

from quasarnp.io import load_model
from quasarnp.io import load_desi_coadd
from quasarnp.utils import process_preds

from desitarget.geomask import match, match_to
from desitarget.internal import sharedmem
from desitarget.io import write_with_units
from desiutil.depend import add_dependencies
from desitarget.targets import main_cmx_or_sv, switch_main_cmx_or_sv
from desitarget.targetmask import zwarn_mask

from desispec.io import read_spectra, findfile
from desispec.io.util import checkgzip, replace_prefix
from desispec.exposure_qa import get_qa_params
from desispec.maskbits import fibermask

# ADM set up the DESI default logger.
from desiutil.log import get_logger
log = get_logger()

# ADM data models for the various afterburners.
zmtldatamodel = np.array([], [('RA', '>f8'), ('DEC', '>f8'), ('TARGETID','>i8'),
                              ('DESI_TARGET', '>i8'), ('BGS_TARGET', '>i8'),
                              ('MWS_TARGET', '>i8'), ('SCND_TARGET', '>i8'),
                              ('Z', '>f8'), ('ZWARN', '>i8'),
                              ('SPECTYPE', '<U6'), ('DELTACHI2', '>f8'),
                              ('NUMOBS', '>i4'), ('ZTILEID', '>i4')])
qndm = [('Z_QN', '>f8'), ('Z_QN_CONF', '>f8'), ('IS_QSO_QN', '>i2')]
sqdm = [('Z_SQ', '>f8'), ('Z_SQ_CONF', '>f8')]
absdm = [('Z_ABS', '>f8'), ('Z_ABS_CONF', '>f8')]
combdm = [('Z_COMB', '>f8'), ('Z_COMB_PROB', '>f8')]


def tmark(istring):
    """A function to mark the time an operation starts or ends.

    Parameters
    ----------
    istring : :class:'str'
        The input string to print to the terminal.

    Notes
    -----
    - A string with the date and time in ISO 8061 standard followed
      by the 'istring'.
    """
    t0 = time.time()
    t_start = time.strftime('%Y-%m-%d | %H:%M:%S')
    log.info('\n{}: {}'.format(istring, t_start))


def make_new_zmtl(redrockname, qn_flag=False, sq_flag=False, abs_flag=False,
                  zcomb_flag=False):
    """Make the initial zmtl array with redrock data.

    Parameters
    ----------
    redrockname : :class:`str`
        Full filename and path for the redrock file to process.
    qn_flag : :class:`bool'` optional
        Flag to add QuasarNP data (or not) to the zmtl file.
    sq_flag : :class:`bool`, optional
        Flag to add SQUEzE data (or not) to the zmtl file.
    abs_flag : :class:`bool`, optional
        Flag to add MgII Absorption data (or not) to the zmtl file.
    zcomb_flag : :class:`bool`, optional
        Flag if a combined redshift (or not) was added to the zmtl file.

    Returns
    -------
    :class:`~numpy.array` or `bool`
        A zmtl in the official format (`zmtldatamodel`) compiled from
        the `tile', 'night', and 'petal_num', in `zmtldir`. If the redrock
        file for that petal doesn't exist, returns ``False``.
    """
    tmark('    Making redrock zmtl')

    # ADM read in the redrock and fibermap extensions for the RR
    # ADM redshift catalog, if they exist.

    zs=None
    try:
        zs = fitsio.read(redrockname, "REDSHIFTS")
        log.info(f'Read {redrockname}')
    except (FileNotFoundError, OSError):
        log.warning(f'Cannot open hdu REDSHIFTS in {redrockname}')
    if zs is None :
        try:
            log.warning("Trying ZBEST...")
            zs = fitsio.read(redrockname, "ZBEST")
            log.info(f'Read {redrockname}')
        except (FileNotFoundError, OSError):
            log.error(f'Cannot open hdu REDSHIFTS in {redrockname}')
            return False

    try:
        fms = fitsio.read(redrockname, "FIBERMAP")
        log.info(f'Read {redrockname}')
    except (FileNotFoundError, OSError):
        log.error(f'Missing {redrockname}')
        return False

    # SB check assumption that FIBERMAP is row-matched to redrock
    if len(zs) != len(fms) or np.any(zs['TARGETID'] != fms['TARGETID']):
        msg = "REDSHIFTS and FIBERMAP TARGETIDs aren't row-matched"
        log.critical(msg)
        raise ValueError(msg)

    # ADM check for some glitches.
    if len(zs) != len(set(zs["TARGETID"])):
        msg = "a target is duplicated in file {}!!!".format(redrockname)
        log.critical(msg)
        raise ValueError(msg)
    # ADM check for some glitches.
    if len(zs) != len(fms):
        msg = "TARGETID mismatch for extensions in file {}!!!".format(redrockname)
        log.critical(msg)
        raise ValueError(msg)

    # ADM Strictly match the targets in the z catalog and fibermap.
    zid = match_to(fms["TARGETID"], zs["TARGETID"])

    # ADM set up the output zmtl file, which differs depending on which
    # ADM afterburners were specified.
    dtswitched = switch_main_cmx_or_sv(zmtldatamodel, fms)
    dt = dtswitched.dtype.descr
    for flag, dm in zip([qn_flag, sq_flag, abs_flag, zcomb_flag],
                        [qndm, sqdm, absdm, combdm]):
        if flag:
            dt += dm
    zmtl = np.full(len(zs), -1, dtype=dt)

    # ADM add the columns from the original redrock file.
    zmtl["RA"] = fms[zid]["TARGET_RA"]
    zmtl["DEC"] = fms[zid]["TARGET_DEC"]
    zmtl["ZTILEID"] = fms[zid]["TILEID"]
    zmtl["NUMOBS"] = fms["COADD_NUMTILE"]

    # ADM also add the appropriate bit-columns.
    Mxcols, _, _, = main_cmx_or_sv(fms, scnd=True)
    for col in Mxcols:
        if col in fms.dtype.names:
            zmtl[col] = fms[zid][col]
        # SB fail on missing required columns ...
        elif col in zmtldatamodel.dtype.names:
            msg = f'Input fibermap missing {col}, which is required by zmtl datamodel'
            log.critical(msg)
            raise ValueError(msg)
        # SB ... but only log error about unexpectedly missing optional columns
        else:
            log.error(f'Input fibermap missing optional {col}; leaving it blank')

    # ADM write out the unwritten columns.
    allcols = set(dtswitched.dtype.names)
    usedcols = set(['RA', 'DEC', 'NUMOBS', 'ZTILEID'] + Mxcols)
    for col in allcols - usedcols:
        zmtl[col] = zs[col]

    return zmtl


def get_qn_model_fname(qnmodel_fname=None):
    """Convenience function to grab the $QN_MODEL_FILE environment variable.

    Parameters
    ----------
    qnmodel_fname : :class:`str`, optional, defaults to $QN_MODEL_FILE
        If `qnmodel_fname` is passed, it is returned from this function. If it's
        not passed, the $QN_MODEL_FILE variable is returned.

    Returns
    -------
    :class:`str`
        not passed, the directory stored in the $QN_MODEL_FILE environment
        variable is returned prepended to the default filename.
    """
    if qnmodel_fname is None:
        qnmodel_fname = os.environ.get('QN_MODEL_FILE')
        # EBL check that the $QN_MODEL_FILE environment variable is set.
        if qnmodel_fname is None:
            msg = "Pass qnmodel_fname or set $QN_MODEL_FILE environment variable!"
            log.critical(msg)
            raise ValueError(msg)

    return qnmodel_fname


def load_qn_model(model_filename):
    """Convenience function to load the QuasarNP model and line lists.

    Parameters
    ----------
    model_filename : :class:`str`
        The filename and path of the QuasarNP model. Either input by user or defaults
        to get_qn_model_fname().

    Returns
    -------
    :class:`~numpy.array`
        The QuasarNP model file loaded as an array.
    :class:`~numpy.array`
        An array of the emission line names to be used for quasarnp.process_preds().
    :class:`~numpy.array`
        An array of the BAL emission line names to be used by quasarnp.process_preds().
    """
    lines = ['LYA', 'CIV(1548)', 'CIII(1909)', 'MgII(2796)', 'Hbeta', 'Halpha']
    lines_bal = ['CIV(1548)']
    model = load_model(model_filename)

    return model, lines, lines_bal

def add_tileqa_data(zmtl, tileqafile):
    """Modifies zmtl['ZWARN'] in-place to add tile QA flags

    Parameters
    ----------
    zmtl : :class:`~numpy.array`
        The structured array that was created by make_new_zmtl()
    tileqafile : :class:`str`
        The full filepath of the tile-qa file

    Returns boolean array for whether targets were flagged as bad (True) or not
    """
    tileqa = fitsio.read(tileqafile, 'FIBERQA')
    keep = np.isin(tileqa['TARGETID'], zmtl['TARGETID'])
    tileqa = tileqa[keep]

    if not np.all(np.isin(zmtl['TARGETID'], tileqa['TARGETID'])):
        msg = "zmtl has targets not in tileqa"
        log.error(msg)
        raise ValueError(msg)

    assert len(zmtl) == len(tileqa)

    if not np.all(zmtl['TARGETID'] == tileqa['TARGETID']):
        log.debug('Resorting tileqa to match zmtl TARGETID order')
        ii = match_to(tileqa['TARGETID'], zmtl['TARGETID'])
        tileqa = tileqa[ii]

    assert np.all(zmtl['TARGETID'] == tileqa['TARGETID'])

    qaparams = get_qa_params()['exposure_qa']
    badpetalmask = fibermask.mask(qaparams['bad_petal_mask'])
    badfibermask = fibermask.mask(qaparams['bad_qafstatus_mask'])
    badfibermask &= ~badpetalmask  #- remove petal-specific bits

    badfiber = (tileqa['QAFIBERSTATUS'] & badfibermask) != 0
    badpetal = (tileqa['QAFIBERSTATUS'] & badpetalmask) != 0
    zmtl['ZWARN'][badfiber] |= zwarn_mask.BAD_SPECQA
    zmtl['ZWARN'][badpetal] |= zwarn_mask.BAD_PETALQA

    return badfiber | badpetal

def add_qn_data(zmtl, coaddname, qnp_model, qnp_lines, qnp_lines_bal):
    """Apply the QuasarNP model to the input zmtl and add data to columns.

    Parameters
    ----------
    zmtl : :class:`~numpy.array`
        The structured array that was created by make_new_zmtl()
    coaddname : :class:`str`
        The name of the coadd file corresponding to the redrock file used
        in make_new_zmtl()
    qnp_model : :class:`h5.array`
        The array containing the pre-trained QuasarNP model.
    qnp_lines : :class:`list`
        A list containing the names of the emission lines that
        quasarnp.process_preds() should use.
    qnp_lines_bal : :class:`list`
        A list containing the names of the emission lines to check
        for BAL troughs.

    Returns
    -------
    :class:`~numpy.array`
        The zmtl array with QuasarNP data included in the columns:

        * Z_QN        - The best QuasarNP redshift for the object
        * Z_QN_CONF   - The confidence of Z_QN
        * IS_QSO_QN   - A binary flag indicated object is a quasar
    """
    tmark('    Adding QuasarNP data')

    data, w = load_desi_coadd(coaddname)
    data = data[:, :, None]
    p = qnp_model.predict(data)
    c_line, z_line, redrock, *_ = process_preds(p, qnp_lines, qnp_lines_bal,
                                              verbose=False)

    cbest = np.array(c_line[c_line.argmax(axis=0), np.arange(len(redrock))])
    c_thresh = 0.5
    n_thresh = 1
    is_qso = np.sum(c_line > c_thresh, axis=0) >= n_thresh

    zmtl['Z_QN'][w] = redrock
    zmtl['Z_QN_CONF'][w] = cbest
    zmtl['IS_QSO_QN'][w] = is_qso

    return zmtl


def get_sq_model_fname(sqmodel_fname=None):
    """Convenience function to grab the $SQ_MODEL_FILE environment variable.

    Parameters
    ----------
    sqmodel_fname : :class:`str`, optional, defaults to $SQ_MODEL_FILE
        If `sqmodel_fname` is passed, it is returned from this function. If it's
        not passed, the $SQ_MODEL_FILE environment variable is returned.

    Returns
    -------
    :class:`str`
        If `sqmodel_fname` is passed, it is returned from this function. If it's
        not passed, the directory stored in the $SQ_MODEL_FILE environment
        variable is returned.
    """
    if sqmodel_fname is None:
        sqmodel_fname = os.environ.get('SQ_MODEL_FILE')
        # EBL check that the $SQ_MODEL_FILE environment variable is set.
        if sqmodel_fname is None:
            msg = "Pass sqmodel_fname or set $SQ_MODEL_FILE environment variable!"
            log.critical(msg)
            raise ValueError(msg)

    return sqmodel_fname


def load_sq_model(model_filename):
    """Convenience function for loading the SQUEzE model file.

    Parameters
    ----------
    model_filename : :class:`str`
        The filename and path of the SQUEzE model file. Either input by user
        or defaults to get_sq_model_fname().

    Returns
    -------
    :class:`~numpy.array`
        A numpy array of the SQUEzE model.

    Notes
    -----
    - The input model file needs to be in the json file format.
    """
    from squeze.common_functions import load_json
    from squeze.model import Model
    model = Model.from_json(load_json(model_filename))

    return model


def add_sq_data(zmtl, coaddname, squeze_model):
    """Apply the SQUEzE model to the input zmtl and add data to columns.

    Parameters
    ----------
    zmtl : :class:`~numpy.array`
        The structured array that was created by make_new_zmtl()
    coaddname : class:`str`
        The name of the coadd file corresponding to the redrock file used
        in make_new_zmtl()
    squeze_model : :class:`numpy.array`
        The loaded SQUEzE model file

    Returns
    -------
    :class:`~numpy.array`
        The zmtl array with SQUEzE data included in the columns:

        * Z_SQ        - The best redshift from SQUEzE for each object.
        * Z_SQ_CONF   - The confidence value of this redshift.
    """
    tmark('    Adding SQUEzE data')

    from squeze.candidates import Candidates
    from squeze.desi_spectrum import DesiSpectrum
    from squeze.spectra import Spectra

    mdata = ['TARGETID']
    single_exposure = False
    sq_cols_keep = ['PROB', 'Z_TRY', 'TARGETID']

    tmark('      Reading spectra')
    desi_spectra = read_spectra(coaddname)
    # EBL Initialize squeze Spectra class
    squeze_spectra = Spectra([])
    # EBL Get TARGETIDs
    targetid = np.unique(desi_spectra.fibermap['TARGETID'])
    # EBL Loop over TARGETIDs to build the Spectra objects
    for targid in targetid:
        # EBL Select objects
        pos = np.where(desi_spectra.fibermap['TARGETID'] == targid)
        # EBL Prepare column metadata
        metadata = {col.upper(): desi_spectra.fibermap[col][pos[0][0]] for col in mdata}
        # EBL Add the SPECID as the TARGETID
        metadata['SPECID'] = targid
        # EBL Extract the data
        flux = {}
        wave = {}
        ivar = {}
        mask = {}
        for band in desi_spectra.bands:
            flux[band] = desi_spectra.flux[band][pos]
            wave[band] = desi_spectra.wave[band]
            ivar[band] = desi_spectra.ivar[band][pos]
            mask[band] = desi_spectra.mask[band][pos]

        # EBL Format each spectrum for the model application
        spectrum = DesiSpectrum(flux, wave, ivar, mask, metadata, single_exposure)
        # EBL Append the spectrum to the Spectra object
        squeze_spectra.append(spectrum)

    # EBL Initialize candidate object. This takes a while with no feedback
    # so we want a time output for benchmarking purposes.
    tmark('      Initializing candidates')
    candidates = Candidates(mode='operation', model=squeze_model)
    # EBL Look for candidate objects. This also takes a while.
    tmark('      Looking for candidates')
    candidates.find_candidates(squeze_spectra.spectra_list(), save=False)
    # EBL Compute the probabilities of the line/model matches to the spectra
    tmark('      Computing probabilities')
    candidates.classify_candidates(save=False)
    # EBL Filter the results by removing the duplicate entries for each
    # TARGETID. Merge the remaining with the zmtl data.
    tmark('      Merging SQUEzE data with zmtl')
    data_frame = candidates.candidates()
    data_frame = data_frame[~data_frame['DUPLICATED']][sq_cols_keep]
    # EBL Strip the pandas data frame structure and put it into a numpy
    # structured array first.
    sqdata_arr = np.zeros(len(data_frame), dtype=[('TARGETID', 'int64'),
                                                  ('Z_SQ', 'float64'),
                                                  ('Z_SQ_CONF', 'float64')])
    sqdata_arr['TARGETID'] = data_frame['TARGETID'].values
    sqdata_arr['Z_SQ'] = data_frame['Z_TRY'].values
    sqdata_arr['Z_SQ_CONF'] = data_frame['PROB'].values
    # EBL SQUEzE will reorder the objects, so match on TARGETID.
    zmtl_args, sqdata_args = match(zmtl['TARGETID'], sqdata_arr['TARGETID'])
    zmtl['Z_SQ'][zmtl_args] = sqdata_arr['Z_SQ'][sqdata_args]
    zmtl['Z_SQ_CONF'][zmtl_args] = sqdata_arr['Z_SQ_CONF'][sqdata_args]

    return zmtl


def add_abs_data(zmtl, coaddname):
    """Add the MgII absorption line finder data to the input zmtl array.

    Parameters
    ----------
    zmtl : :class:'~numpy.array`
        The structured array that was created by make_new_zmtl()
    coaddname : class:`str`
        The name of the coadd file corresponding to the redrock file used
        in make_new_zmtl()

    Returns
    -------
    :class:`~numpy.array`
        The zmtl array with MgII Absorption data included in the columns:

        * Z_ABS        - The highest redshift of MgII absorption
        * Z_ABS_CONF   - The confidence value for this redshift.

    Notes
    -----
    - The original function was written by Lucas Napolitano (LGN) and
      modified for this script by Eleanor Lyke (EBL).
    """
    #- TODO: replace this with something from desispec.coaddition instead of
    #- using a utility function from a plotting library
    from prospect.mycoaddcam import coadd_brz_cameras

    fitter = fitting.LevMarLSQFitter()
    model = models.Gaussian1D()
    # LGN Define constants
    first_line_wave = 2796.3543
    second_line_wave = 2803.5315
    rf_line_sep = second_line_wave - first_line_wave

    # LGN Define hyperparameters
    rf_err_margain = 0.50
    kernel_smooth = 2
    kernel = Gaussian1DKernel(stddev=kernel_smooth)
    med_filt_size = 19
    snr_threshold = 3.0
    qi_min = 0.01
    sim_fudge = 0.94

    # LGN Intialize output array.
    out_arr = []

    # LGN Read the coadd file and find targetid.
    specobj = read_spectra(coaddname)
    redrockfile = replace_prefix(coaddname, 'coadd', 'redrock').replace('.fits', '.h5')
    # LGN Get all targetids
    tids = specobj.target_ids()
    # LGN Run for every quasar target on the petal.
    num_rows = len(zmtl)
    for specnum in range(num_rows):
        # LGN Grab a single targetid.
        targetid = tids[specnum]
        # LGN Open the redrock file and read in model fits for specific
        # targetid.
        targpath = f'/zfit/{targetid}/zfit'
        zalt = Table.read(redrockfile, path=targpath)
        # LGN If best spectype is a star we shouldn't process it.
        if zalt['spectype'][0] == 'STAR':
            out_arr.append([targetid, 0, 0])
            continue

        # LGN Define wavelength range and flux values.
        # LGN Check to see if b,r, and z cameras are already coadded.
        if "brz" in specobj.wave:
            x_spc = specobj.wave["brz"]
            y_flx = specobj.flux["brz"][specnum]
            y_err = np.sqrt(specobj.ivar["brz"][specnum])**(-1.0)
        # LGN If not, coadd them into "brz" using coadd_brz_cameras from
        # prospect docs.
        else:
            wave_arr = [specobj.wave["b"],
                        specobj.wave["r"],
                        specobj.wave["z"]]
            flux_arr = [specobj.flux["b"][specnum],
                        specobj.flux["r"][specnum],
                        specobj.flux["z"][specnum]]
            noise_arr = [np.sqrt(specobj.ivar["b"][specnum])**(-1.0),
                         np.sqrt(specobj.ivar["r"][specnum])**(-1.0),
                         np.sqrt(specobj.ivar["z"][specnum])**(-1.0)]

            x_spc, y_flx, y_err = coadd_brz_cameras(wave_arr, flux_arr,
                                                    noise_arr)

        # LGN Apply a gaussian smoothing kernel using hyperparameters
        # defined above.
        smooth_yflx = convolve(y_flx, kernel)
        # LGN Estimate the continuum using median filter.
        continuum = medfilt(y_flx, med_filt_size)

        # LGN Run the doublet finder.
        residual = continuum - y_flx

        # LGN Generate groups of data with positive residuals.
        # LGN/EBL: The following is from a stackoverlow thread:
        #     https://stackoverflow.com/questions/3149440/python-splitting-list-based-on-missing-numbers-in-a-sequence
        groups = []
        for k, g in groupby(enumerate(np.where(residual > 0)[0]), lambda x: x[0] - x[1]):
            groups.append(list(map(itemgetter(1), g)))

        # LGN Intialize the absorbtion line list.
        absorb_lines = []

        for group in groups:
            # LGN Skip groups of 1 or 2 data vals, these aren't worthwhile
            #    peaks and cause fitting issues.
            if len(group) < 3:
                continue

            # LGN Calculate the S/N value.
            snr = np.sum(residual[group]) * np.sqrt(np.sum(y_err[group]))**(-1.0)
            if snr > snr_threshold:
                # LGN Fit a gaussian model.
                model = models.Gaussian1D(amplitude=np.nanmax(residual[group]),
                                          mean=np.average(x_spc[group]))
                fm = fitter(model=model, x=x_spc[group], y=residual[group])
                # LGN Unpack the model fit data.
                amp, cen, stddev = fm.parameters

                absorb_lines.append([amp, cen, stddev, snr])

        # LGN Extract the highest z feature and associated quality index (QI)
        hz = 0
        hz_qi = 0
        # LGN This is particuarly poorly implemented, using range(len) so
        # I can slice to higher redshift lines only more easily.
        for counter in range(len(absorb_lines)):
            line1 = absorb_lines[counter]
            # LGN Determine redshift from model parameters.
            ztemp = (line1[1] * first_line_wave**(-1.0)) - 1

            # LGN If redshift is in any of the masked regions ignore it.
            if 2.189 < ztemp < 2.191 or 2.36 < ztemp < 2.40:
                continue
            # LGN Determine line seperation and error margain scaled to
            # redshift.
            line_sep = rf_line_sep * (1 + ztemp)
            err_margain = rf_err_margain * (1 + ztemp)
            # LGN for all lines at higher redshifts.
            for line2 in absorb_lines[counter+1:]:
                # LGN calculate error from expected line seperation
                # given the redshift of the first line.
                sep_err = np.abs(line2[1] - line1[1] - line_sep)
                # LGN Keep if within error margains.
                if sep_err < err_margain:
                    # LGN Calculate the QI.
                    # LGN S/N similarity of lines. sim_fudge is defined
                    #    in the hyperparameters above and
                    #    adjusts for the first line being larger,
                    #    kind of a fudge, won't lie.
                    snr_sim = sim_fudge * line1[3] * line2[3]**(-1.0)
                    # LGN Rescale to peak at lines having exact same S/N.
                    if snr_sim > 1:
                        snr_sim = snr_sim**(-1.0)
                    # LGN seperation accuracy
                    #   Is '1' if expected seperation = actual seperation.
                    #   Decreases to 0 outside this.
                    sep_acc = (1 - sep_err) * err_margain**(-1.0)
                    qi = snr_sim * sep_acc
                    if ztemp > hz and qi > qi_min:
                        hz = ztemp
                        hz_qi = qi

        out_arr.append([targetid, hz, hz_qi])

    # EBL Add the redshift and quality index for each targetid to the
    # zmtl file passed to the function.
    out_arr = np.array(out_arr)
    zmtl_args, abs_args = match(zmtl['TARGETID'], out_arr[0])
    zmtl['Z_ABS'][zmtl_args] = out_arr[1][abs_args]
    zmtl['Z_ABS_CONF'][zmtl_args] = out_arr[2][abs_args]

    return zmtl


def zcomb_selector(zmtl, proc_flag=False):
    """Compare results from redrock, QuasarNP, SQUEzE, and MgII data.

    Parameters
    ----------
    zmtl : :class:`~numpy.array`
        The structured array that was created by make_new_zmtl()
    proc_flag : :class:`bool`
        Turn on extra comparison procedure.

    Returns
    -------
    :class:`~numpy.array`
        The zmtl array with SQUEzE data included in the columns:

        * Z_COMB        - The best models-combined redshift for each object.
        * Z_COMB_PROB   - The combined probability value of that redshift.
    """
    zmtl['Z_COMB'][:] = zmtl['Z']
    zmtl['Z_COMB_PROB'][:] = 0.95

    return zmtl


def write_zmtl(zmtl, outputname,
               qn_flag=False, sq_flag=False, abs_flag=False, zcomb_flag=False,
               qnp_model_file=None, squeze_model_file=None):
    """Writes the zmtl structured array out as a FITS file.

    Parameters
    ----------
    zmtl : :class:`~numpy.array`
        The structured array that was created by make_new_zmtl()
    outputname : :class:`str`
        The full filepathname of the zmtl output file.
    qn_flag : :class:`bool`
        Flag if QuasarNP data (or not) was added to the zmtl file.
    sq_flag : :class:`bool`
        Flag if SQUEzE data (or not) was added to the zmtl file.
    abs_flag : :class:`bool`
        Flag if MgII Absorption data (or not) was added to the zmtl file.
    zcomb_flag : :class:`bool`
        Flag if a combined redshift (or not) was added to the zmtl file.
    qnp_model_file : :class:`str`, optional
        File from which the QuasarNP model was loaded. Written to the
        output header.
    squeze_model_file : :class:`str`, optional
        File from which the SQUEzE model was loaded. Written to the
        output header.

    Returns
    -------
    :class:`str`
        The filename, with path, of the FITS file written out.
    """
    tmark('    Creating output file...')

    # ADM create the necessary output directory, if it doesn't exist.
    outputdir = os.path.dirname(os.path.abspath(outputname))
    os.makedirs(outputdir, exist_ok=True)

    # ADM create the header and add the standard DESI dependencies.
    hdr = {}
    add_dependencies(hdr)
    add_dependencies(hdr, module_names=['quasarnp',])

    # ADM add the specific zmtl dependencies
    hdr['QN_ADDED'] = qn_flag
    hdr['SQ_ADDED'] = sq_flag
    hdr['AB_ADDED'] = abs_flag
    hdr['ZC_ADDED'] = zcomb_flag
    if qn_flag:
        hdr['QNMODFIL'] = qnp_model_file
    if sq_flag:
        hdr['SQMODFIL'] = squeze_model_file

    # SB Check if all fibers were masked due to failing petal QA
    npetalmask = np.sum(zmtl['ZWARN'] & zwarn_mask['BAD_PETALQA'] != 0)
    if npetalmask == len(zmtl):
        hdr['BADPTLQA'] = True
    else:
        hdr['BADPTLQA'] = False

    # ADM write out the data to the full file name.
    write_with_units(outputname, zmtl, extname='ZMTL', header=hdr)

    return outputname


def create_zmtl(zmtldir, outputdir, tile=None, night=None, petal_num=None,
                qn_flag=False, qnp_model=None, qnp_model_file=None,
                qnp_lines=None, qnp_lines_bal=None,
                sq_flag=False, squeze_model=None, squeze_model_file=None,
                abs_flag=False, zcomb_flag=False):
    """This will create a single zmtl file from a set of user inputs.

    Parameters
    ----------
    zmtldir : :class:`str`
        If any of `tile`, `night` or `petal_num` are ``None``:
            The name of a redrock `redrock` file.
        If none of `tile`, `night` and `petal_num` are ``None``:
            The root directory from which to read `redrock` and `coadd`
            spectro files. The full directory is constructed as
            `zmtldir` + `tile` + `night`, with files
            redrock-/coadd-`petal_num`*`night`.fits.
    outputdir : :class:`str`
        If any of `tile`, `night` or `petal_num` are ``None``:
            The name of an output file.
        If none of `tile`, `night` and `petal_num` are ``None``:
            The output directory to which to write the output file.
            The full directory is constructed as `outputdir` + `tile` +
            `night`, with file zmtl-`petal_num`*`night`.fits.
    tile : :class:`int`
        The TILEID of the tile to process.
    night : :class:`int`
        The date associated with the observation of the 'tile' used.
        * Must be in YYYYMMDD format
    petal_num : :class:`int`
        If 'all_petals' isn't used, the single petal to create a zmtl for.
    qn_flag : :class:`bool`, optional
        Flag to add QuasarNP data (or not) to the zmtl file.
    qnp_model : :class:`h5 array`, optional
        The QuasarNP model file to be used for line predictions.
    qnp_model_file : :class:`str`, optional
        File from which to load the QuasarNP model (`qnp_model`),
        `qnp_lines` and `qnp_lines_bal` if `qnp_model` is ``None``. Also
        written to the output header of the zmtl file.
    qnp_lines : :class:`list`, optional
        The list of lines to use in the QuasarNP model to test against.
    qnp_lines_bal : :class:`list`, optional
        The list of BAL lines to use for QuasarNP to identify BALs.
    sq_flag : :class:`bool`, optional
        Flag to add SQUEzE data (or not) to the zmtl file.
    squeze_model : :class:`numpy.array`, optional
        The numpy array for the SQUEzE model file.
    squeze_model_file : :class:`str`, optional
        File from which to load the SQUEzE model if `squeze_model` is
        ``None``. Also written to the output header of the zmtl file.
    abs_flag : :class:`bool`, optional
        Flag to add MgII Absorption data (or not) to the zmtl file.
    zcomb_flag : :class:`bool`, optional
        Flag if a combined redshift (or not) was added to the zmtl file.

    Notes
    -----
    - Writes a FITS catalog that incorporates redrock, and a range of
      afterburner redshifts and confidence values. This will write to the
      same directory of the redrock and coadd files unless a different
      output directory is passed.
    """
    # ADM load the model files, if needed.
    if qn_flag and qnp_model is None:
        tmark('    Loading QuasarNP Model file and lines of interest')
        qnp_model, qnp_lines, qnp_lines_bal = load_qn_model(qnp_model_file)
        tmark('      QNP model file loaded')
    if sq_flag and squeze_model is None:
        tmark('    Loading SQUEzE Model file')
        sq_model = load_sq_model(squeze_model_file)
        tmark('      Model file loaded')

    # ADM simply read/write files if tile/night/petal_num not specified.
    if tile is None or night is None or petal_num is None:
        redrockfn = zmtldir
        dirname, basename = os.path.split(redrockfn)
        coaddfn = os.path.join(dirname, basename.replace("redrock", "coadd"))
        outputfn = os.path.join(dirname, basename.replace("redrock", "zmtl"))

        # SB tile-qa file doesn't have spectro num so requires more parsing
        # /path/redrock-0-1234-20201220.fits -> /path/tile-qa-1234-20201220.fits
        tmp = redrockfn.split('-')
        tileqafn = '-'.join(['tile-qa',] + tmp[2:])
        tileqafn = os.path.join(dirname, tileqafn)

    # EBL Create the filepath for the input tile/night combination
    else:
        ### tiledir = os.path.join(zmtldir, tile)
        ### ymdir = os.path.join(tiledir, night)

        # ADM Create the corresponding output directory.
        ### outputdir = os.path.join(outputdir, tile, night)

        # EBL Create the filename tag that appends to redrock-*, coadd-*,
        # and zmtl-* files.
        ### filename_tag = f'{petal_num}-{tile}-{night}.fits'
        # ADM try a couple of generic options for the file names.
        ### if not os.path.isfile(os.path.join(ymdir, f'redrock-{filename_tag}')):
        ###     filename_tag = f'{petal_num}-{tile}-thru{night}.fits'

        redrockfn = findfile('redrock_tile', tile=tile, night=night, spectrograph=petal_num)
        coaddfn = findfile('coadd_tile', tile=tile, night=night, spectrograph=petal_num)
        tileqafn = findfile('tileqa', tile=tile, night=night, spectrograph=petal_num)
        outputfn = findfile('zmtl', tile=tile, night=night, spectrograph=petal_num)

        ### redrockfn = os.path.join(ymdir, redrockname)
        ### coaddfn = os.path.join(ymdir, coaddname)

    if not os.path.exists(redrockfn):
        log.warning(f'Petal {petal_num} missing redrock file: {redrockfn}')
    elif not os.path.exists(coaddfn):
        log.warning(f'Petal {petal_num} missing coadd file: {coaddfn}')
    elif not os.path.exists(tileqafn):
        log.warning(f'Petal {petal_num} missing tile-qa file: {tileqafn}')
    else:
        zmtl = make_new_zmtl(redrockfn, qn_flag, sq_flag, abs_flag, zcomb_flag)
        if isinstance(zmtl, bool):
            log.warning(f'make_new_zmtl for {redrockfn} failed')
            raise RuntimeError

        add_tileqa_data(zmtl, tileqafn)

        if qn_flag:
            zmtl = add_qn_data(zmtl, coaddfn, qnp_model, qnp_lines, qnp_lines_bal)
        if sq_flag:
            zmtl = add_sq_data(zmtl, coaddfn, squeze_model)
        if abs_flag:
            zmtl = add_abs_data(zmtl, coaddfn)

        if zcomb_flag:
            zmtl = zcomb_selector(zmtl)

        full_outputname = write_zmtl(zmtl, outputfn, qn_flag,
                                     sq_flag, abs_flag, zcomb_flag,
                                     qnp_model_file, squeze_model_file)

        tmark('    --{} written out correctly.'.format(full_outputname))
        log.info('='*79)
