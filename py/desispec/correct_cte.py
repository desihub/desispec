"""
desispec.correct_cte
====================

Methods to fit CTE effects and remove them from images.
"""

import os
import copy
from functools import partial

import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import median_filter
from astropy.stats import sigma_clipped_stats
import yaml
from astropy.table import Table
import fitsio

from desispec.io import findfile
from desispec.util import difference_nights
from desiutil.log import get_logger

import desispec.preproc
from desispec.calibfinder import CalibFinder
from desispec.trace_shifts import compute_dx_from_cross_dispersion_profiles
from desispec.qproc import qfiberflat, qsky, rowbyrowextract
from desispec.image_model import compute_image_model
from desispec.workflow.tableio import load_table

def apply_multiple_cte_effects(amp, locations, ctefuns):
    """Apply a series of traps to an amp.

    This function assumes that it is only being given a
    single amp that has some traps on it.  The readout
    direction is always assumed to be toward smaller numbers
    on the second axis.

    Parameters
    ----------
    amp : np.ndarray
        image of amplifier
    locations : np.ndarray [int]
        locations of traps
    ctefuns : callables
        functions that can be called to apply CTE to an amp

    Returns
    -------
    amplifier after applying traps
    """

    # amp is affected by a series of traps.
    # apply them all.
    # I don't think we want to have more than one, though!
    # assume that the read out is such that we don't have to
    # reverse the amplifier to apply in the right direction;
    # that's been done for us, as has the corresponding
    # reversing of the locations.
    locations = np.array(np.atleast_1d(locations))
    s = np.argsort(-locations)
    amp_out = amp.copy()
    # sequentially step down the serial register,
    # trap-by-trap, applying the appropriate CTE
    # function to all of the pixels affected by that trap.
    # in the picture in my head, the serial register is read
    # out in the downward direction.  All of the pixels
    # above the trap are affected.  So we start with the
    # highest trap and walk down.  Those later traps
    # see the CTE-affected image from the earlier traps.

    # for the one trap case, which is the only one we actually
    # have right now, this just applies the trap function to the
    # portion of the amplifier above the trap.
    for ii in s:
        affected = np.s_[:, locations[ii]:]
        amp_out[affected] = ctefuns[ii](amp_out[affected])
    return amp_out


def correct_amp(image, ctefun, niter=1, apply_only=False):
    """Correct an amp for CTE.

    In the limit that CTE is perturbative, one can correct for CTE
    as follows:
    for i in range(niter):
    I' = CTE(I)
    I = I - (I' - I)
    This function implements that approach.

    However, for large traps, this approach correlates the noise
    and leads to poor performance; we do not use it for DESI.

    Parameters
    ----------
    image : np.ndarray
        CTE-affected amplifier
    ctefun : callable
        function that applies CTE to an amplifier
    niter : int
        number of iterations to run
    apply_only : bool
        Do not correct, just apply the CTE function to the image.

    Returns
    -------
    CTE-corrected image
    """

    # more than one iteration makes the noise properties worse, but
    # does make things more self-consistent.
    corrected_image = image.copy()
    log = get_logger()
    for i in range(niter):
        cte_image = ctefun(corrected_image)
        if apply_only:
            return cte_image
        correction = cte_image - corrected_image
        mn, med, rms = sigma_clipped_stats(correction)
        log.info(f'Correcting CTE, iteration {i}, correction rms {rms:6.3f}')
        corrected_image = image - correction
    return corrected_image


def add_cte(img, cte_transfer_func=None, **cteparam):
    """Add CTE to a region affected by CTE.

    Assumes all of img is affected.

    Parameters
    ----------
    img : np.ndarray
        input image
    cte_transfer_func : callable
        function describing how many electrons trap soaks up and emits
    **cteparam : dictionary
        additional arguments to cte_transfer_func

    Returns
    -------
    np.ndarray
    img with additional CTE applied
    """
    if cte_transfer_func is None:
        cte_transfer_func = simplified_regnault
    out = img.copy()
    in_trap = img[:, 0] * 0.0
    for i in range(img.shape[1]):
        transfer_amount = cte_transfer_func(
            np.clip(img[:, i], 0, np.inf), in_trap, **cteparam)
        in_trap -= transfer_amount
        out[:, i] += transfer_amount
    return out

def needs_ctecorr(header=None, cfinder=None):
    """
    Return True/False for whether this CCD needs CTE corrections

    Args:
        header: dict-like header from raw data HDU for this CCD
        cfinder: CalibFinder object for this CCD

    Returns True/False

    Must provide header or cfinder but not both; providing a pre-generated
    cfinder avoids re-reading the $DESI_SPECTRO_CALIB yaml file.
    """
    if cfinder is not None and header is not None:
        raise ValueError('provide header or cfinder but not both')
    elif header is None and cfinder is None:
        raise ValueError('must provide header or cfinder')

    if cfinder is None:
        cfinder = CalibFinder([header])
    else:
        header = cfinder.header

    docte = False
    for amp in desispec.preproc.get_amp_ids(header):
        key = "CTECOLS"+amp.upper()
        if cfinder.haskey(key):
            docte = True
            break

    return docte

def get_amp_regions_to_cte_correct(header):
    """
    Get CTE corrections parameters for amps on this image

    Parameters
    ----------
    header (dict-like): image header

    Returns
    -------
    amp_regions, cte_regions
    amp_regions : dict
        dictionary of slices defining each amp
    cte_regions : dict
        The keys are the names of the amplifier, matching amp_regions.
        Each entry is a list containing one entry for each trap on that amp.
        Each item in the list is dictionary with the start and stop locations,

    Uses `header` and CalibFinder to identify any regions to correct

    Returns pair of empty dictionaries if there are no amps needing CTE corrections.
    """
    log = get_logger()
    cfinder = CalibFinder([header])
    amps = desispec.preproc.get_amp_ids(header)
    night = desispec.preproc.header2night(header)
    camera = header['CAMERA'].lower()

    amp_regions = dict()
    cte_regions = dict()
    for amp in amps:
        key = "CTECOLS"+amp
        if not cfinder.haskey(key) :
            # that's ok, we don't expect this keyword for each camera and amplifier luckily
            log.debug(f"No {key} for {camera} on {night}")
            continue

        value = cfinder.value(key)
        if cfinder.haskey("CTEFUNC"+amp):
            ctefunc = cfinder.value("CTEFUNC"+amp)
        else:
            ctefunc = "simplified_regnault"

        amp_sec = desispec.preproc.parse_sec_keyword(header["CCDSEC"+amp])

        yb = amp_sec[0].start
        ye = amp_sec[0].stop
        cte_regions_in_amp = list()
        for ctecols in value.split(","):
            if len(ctecols)==0 : continue
            vals  = ctecols.split(":")
            nvals = len(vals)
            if nvals != 2 :
                mess = "cannot decode {}={}".format(key, value)
                log.critical(mess)
                raise KeyError(mess)

            start, stop = int(vals[0]), int(vals[1])

            xb = max(amp_sec[1].start, start)
            xe = min(amp_sec[1].stop, stop)
            sector = [yb, ye, xb, xe]

            cteparam={"ctecols":ctecols, "start":start, "stop":stop,
                      "ctefunc":ctefunc}
            cte_regions_in_amp.append(cteparam)

        # only add if we have a model for it
        if len(cte_regions_in_amp)>0 :
            amp_regions[amp] = amp_sec
            cte_regions[amp] = cte_regions_in_amp

    return amp_regions, cte_regions

def get_cte_params(header, cte_params_filename=None):
    """
    Get CTE corrections parameters for amps on this image

    Parameters
    ----------
    header (dict-like): image header

    Optional
    --------
    cte_params_filename (str):
        Alternate filename with nightly CTE parameters
        instead of default findfile('ctecorrnight', night)

    Returns
    -------
    amp_regions, cte_regions
    amp_regions : dict
        dictionary of slices defining each amp
    cte_regions : dict
        The keys are the names of the amplifier, matching amp_regions.
        Each entry is a list containing one entry for each trap on that amp.
        Each item in the list is dictionary with the start and stop locations,
        and CTE correction parameters.

    Uses `get_amp_regions_to_cte_correct` (which uses CalibFinder and header)
    to identify any regions to correct, then reads parameters for those regions
    from `cte_params_filename` or default ctecorrnight file.

    Returns pair of empty dictionaries if there are no amps needing CTE corrections.
    """
    import desispec.io   # inside func to avoid circular import

    log = get_logger()

    amp_regions, cte_regions = get_amp_regions_to_cte_correct(header)
    if len(cte_regions) == 0:
        return amp_regions, cte_regions

    night = desispec.preproc.header2night(header)
    camera = header['CAMERA'].lower()

    if cte_params_filename is None :
        cte_params_filename = desispec.io.findfile('ctecorrnight', night=night, readonly=True)

    if not os.path.isfile(cte_params_filename):
        msg = f'Missing {cte_params_filename}; Generate it with desi_fit_cte_night -n {night}, or run preproc with --no-cte-correction or specify an alternate file with --cte-params FILENAME'
        log.critical(msg)
        raise RuntimeError(msg)

    # CTE correction files have list of dicts with entries
    # NIGHT CAMERA AMPLIFIER SECTOR to identify regions,
    # FUNC to identify functional form of correction, and
    # CTE parameters like AMPLITUDE FRACLEAK (depends upon FUNC value).
    # If no amps need CTE corrections, file will be a blank (length 0) list.
    ctecorrnight_dicts = yaml.safe_load(open(cte_params_filename, 'r'))

    ## For each row of the input table, check if the row data was derived from
    ## a night within 2 weeks of the current night, otherwise it isn't valid
    valid_night = np.array([
        np.abs(difference_nights(row['NIGHT'], night)) < 14
        for row in ctecorrnight_dicts])
    ## Check for rows that match the camera we want
    valid_camera = np.array([row['CAMERA'] == camera for row in ctecorrnight_dicts])

    #- augment cte_regions with CTE correction parameters
    for amp in cte_regions:
        ## check for rows that are from a valid not and on the desired camera
        ## and amplifier
        valid_amp = np.array([row['AMPLIFIER'] == amp for row in ctecorrnight_dicts])
        selection = valid_night & valid_camera & valid_amp
        if np.sum(selection)==0 :
            # we do expect a set of CTE parameter for the amplifier because we know the effect is there and
            # we asked for the parameters, this is an error
            mess = f"No CTE correction in {cte_params_filename} for night {night} camera {camera} amplifier {amp}"
            log.critical(mess)
            raise RuntimeError(mess)

        for i in range(len(cte_regions[amp])):
            ctecols = cte_regions[amp][i]['ctecols']
            valid_sector = np.array([row['SECTOR'] == ctecols for row in ctecorrnight_dicts])
            selection2 = selection & valid_sector
            if np.sum(selection2)==0 :
                mess = f"No CTE correction in {cte_params_filename} for night {night} camera {camera} amplifier {amp} sector {ctecols}"
                log.critical(mess)
                raise RuntimeError(mess)

            #- Having identified which row we want, add params to cte_regions
            entry=np.where(selection2)[0][0]
            for k in ctecorrnight_dicts[entry].keys():
                if k in ['NIGHT', 'CAMERA', 'AMPLIFIER', 'SECTOR', 'CHI2PDF']:
                    continue
                cte_regions[amp][i][k] = ctecorrnight_dicts[entry][k]

            cteparam = cte_regions[amp][i]
            log.info(f"CTE correction in amplifier {amp}, sector {ctecols}, {cteparam}")

    return amp_regions, cte_regions


def cte_transfer_amount_regnault(pixval, in_trap, alpha, beta,
                                 cmax, fin, fout):
    """CTE transfer function of Regnault+.

    The model is
    transfer = leak_out - leak in
    leak_out = fout * cmax * (pixval / cmax)**alpha
    leak_in = fin * (1 - in_trap / cmax)**alpha * pixval**beta
    with slight elaborations to prevent more electrons than exist
    from entering or leaving the trap.

    Values of fin tend to be large: the trap effectively steals
    electrons, and fout small; the trap leaks out slowly.  The
    alpha and beta parameters change the effective stickiness
    of the trap wrt how full it is.

    Parameters
    ----------
    pixval : float
        value of uncorrupted image
    in_trap : float
        number of electrons presently in trap
    alpha : float
        power law index for how quickly electrons leave the trap
    beta : float
        power law index for how quickly electrons enter the trap
    cmax : float
        trap capacity
    fin : float
        ~attraction strength of trap from pixel
    fout : float
        ~attraction strength of pixel from trap

    Returns
    -------
    int
    number of electrons that leak out of the trap into the pixel
    """
    # leak out
    transfer_amount = np.clip(fout*cmax*(in_trap/cmax)**alpha, 0, in_trap)
    # leak in
    maxin = np.minimum(pixval, cmax - in_trap)
    alpha = np.clip(alpha, 0, np.inf)
    beta = np.clip(beta, 0, np.inf)
    transfer_amount -= np.clip(
        fin*(1 - in_trap/cmax)**alpha * pixval**beta,
        0, maxin)
    return transfer_amount


def chi_regnault(param, cleantraces=None, ctetraces=None, uncertainties=None):
    """Chi loss function for a Regnault transfer function.

    Parameters
    ----------
    param : list
        parameters of model: [cmax, fin, fout, alpha, beta]
    cleantraces : ndarray
        CTE-unaffected cross-dispersion traces
    ctetraces : ndarray
        CTE-affected cross-dispersion traces
    uncertainties : ndarray
        statistical uncertainty in cleantraces - ctetraces.

    Returns
    -------
    chi : ndarray
        (model - data) / uncertainty
    """
    models = [add_cte(trace, cmax=param[0], fin=param[1], fout=param[2], alpha=param[3], beta=param[4],
                      cte_transfer_func=cte_transfer_amount_regnault)
              for trace in cleantraces]
    res = np.array([
        (m - c)/u for (m, c, u) in zip(models, ctetraces, uncertainties)])
    return res.reshape(-1)



def simplified_regnault(pixval, in_trap, amplitude, fracleak):
    """CTE transfer function of Regnault+.

    The model is
    transfer = in_trap * fracleak - pixval * (1 - in_trap / amplitude)
    with slight elaborations to prevent more electrons than exist from entering
    or leaving the trap.

    Parameters
    ----------
    pixval : float
        value of uncorrupted image
    in_trap : float
        number of electrons presently in trap
    amplitude : float
        amplitude of trap
    fracleak : float
        fraction of electrons that leak out of trap each transfer

    Returns
    -------
    int
    number of electrons that leak out of the trap into the pixel
    """
    maxin = np.minimum(pixval, amplitude - in_trap)
    # leak out of trap
    transfer_amount = np.clip(in_trap * fracleak, 0, in_trap)
    # leak into trap
    transfer_amount -= np.clip(pixval * (1 - in_trap / amplitude), 0, maxin)
    return transfer_amount
    # when full, always leaking ~amplitude * fracleak
    # and stealing back up to this amount.


def chi_simplified_regnault(param, cleantraces=None, ctetraces=None,
                            uncertainties=None):
    """Chi loss function for a Regnault transfer function."""
    models = [add_cte(trace, amplitude=param[0], fracleak=param[1])
              for trace in cleantraces]
    res = np.array([
        (m - c)/u for (m, c, u) in zip(models, ctetraces, uncertainties)])
    return res.reshape(-1)


def get_transfer_function(entry) :
    """Returns a CTE correction function from its record.

    This maps the FUNC name in $DESI_SPECTRO_CALIB yaml files to the
    Python function that should be called.
    """
    if entry['FUNC'] == "simplified_regnault" :
        return partial(add_cte, cte_transfer_func=simplified_regnault,
                       amplitude=entry['AMPLITUDE'], fracleak=entry['FRACLEAK'])
    elif entry['FUNC'] == 'regnault':
        return partial(add_cte, cte_transfer_func=cte_transfer_amount_regnault,
                       cmax=entry['CMAX'], fin=entry['FIN'],
                       fout=entry['FOUT'],
                       alpha=entry['ALPHA'], beta=entry['BETA'])
    else :
        raise KeyError(f"No transfer function called '{entry['FUNC']}'")


def get_transfer_function_chi_names_guess_diff(name):
    """Return information needed for fitting CTE.

    Parameters
    ----------
    name : str
        The name of the CTE transfer function, regnault or simplified_regnault

    Returns
    -------
    (chi, names, guess, diff) tuple

    chi: the function that returns the scaled residuals given the model
         parameters
    names: the names of the parameters being optimized
    guess: the initial guess to make for the parameter values, excluding
         the trap capacity for which a few different guesses are made
    diff: the step sizes to make in the different parameters, including
         in the trap capacity.
    names: the names of the
    """
    if name == "simplified_regnault":
        return (chi_simplified_regnault, ['AMPLITUDE', 'FRACLEAK'],
                [0.2], [0.2, 0.01])
    elif name == "regnault":
        return (chi_regnault,
                ['CMAX', 'FIN', 'FOUT', 'ALPHA', 'BETA'],
                [0.2, 0.2, 1, 1],
                [0.2, 0.01, 0.01, 0.01, 0.01])
    else:
        raise KeyError(f"No transfer function called '{name}'.")


def fit_cte(images):
    """Fits CTE models to a list of images.

    This fits the parameters of the Regnault transfer function model
    to an image with a trap.  It works by comparing an unaffected amplifier
    to an amplifier with a trap along the boundary between the two amplifiers,
    solving for the trap parameters that make the unaffected image + the CTE
    effect a good match to the CTE-affected image.

    It will not work if both amplifiers have CTE, or if there are multiple CTE
    effects to solve for.

    It assumes that amp A is across from amp C and amp B is across from amp D.

    It would likely fail badly for a two-amp mode image.

    Parameters
    ----------
    images : list
        A list of images.
        Usually these are all flats with different exposure lengths on the
        same device.

    Returns
    -------
    list of dicts with fields "NIGHT","CAMERA","AMPLIFIER","SECTOR","FUNC",
                              "CHI2PDF" and parameter names

      "NIGHT","CAMERA","AMPLIFIER" are properties of the image
      "SECTOR" is a string of the form 'BEGIN:END' defining a range of CCD cols.
      "FUNC" is a transfer function, only 'simplified_regnault' implemented for now.
      "CHI2PDF" is the reduced chi2 of the fit
      Other keys are parameters of the transfer function
    """
    # take a bunch of preproc images on one camera
    # these should all be flats and the same device
    # compare areas above and below the amp boundary
    # only works if only one of the two amps has a CTE effect!
    # if both have CTE effects then we need another approach.
    # assume that A <-> C and B <-> D are paired for these purposes.
    # take an uncontaminated line from one of the pair.  Apply the CTE
    # model with some parameters.  Get the contaminated line.
    # compare with the actual contaminated line.  Minimize.
    # we need to extract only the relevant region and have
    # some robustness / noise tracking.

    log = get_logger()
    log.debug("begin fit_cte")

    if images is None or len(images)==0:
        # nothing to do
        return list()

    assert len(images) > 0
    night = desispec.preproc.header2night(images[0].meta)
    camera = images[0].meta['CAMERA']
    obstype = images[0].meta['OBSTYPE']

    if obstype != 'FLAT':
        log.warning('Really should not use this function with a non-flat?!')
    # only use with flats; otherwise the matching above and below the
    # boundary is particularly fraught.
    for image in images:
        assert image.meta['CAMERA'] == camera
        assert image.meta['OBSTYPE'] == obstype
    matching_amps = {
        'A': 'C',
        'C': 'A',
        'B': 'D',
        'D': 'B',
    }
    amp, cte = get_amp_regions_to_cte_correct(images[0].meta)

    res = list()

    for ampname in amp:
        tcte = cte[ampname]
        if len(tcte) == 0:
            continue
        if len(tcte) > 1:
            raise ValueError('Two CTE effect fitting not yet implemented.')
        if matching_amps[ampname] in cte and len(cte[matching_amps[ampname]]) != 0:
            raise ValueError('CTE effect on amp and its mirror not yet '
                             'implemented.')
        tcte = tcte[0]
        # okay!  we should be able to do the fit.
        # we need to select out the region near the amp boundary.
        ampreg = amp[ampname]
        on_bottom = ampreg[0].start == 0
        if on_bottom:
            ampbd = ampreg[0].stop
        else:
            ampbd = ampreg[0].start
        npix = 31
        need_to_reverse = ampreg[1].stop == image.pix.shape[1]
        start, stop = tcte['start'], tcte['stop']

        step = 1
        if need_to_reverse:
            step = -1
            start, stop = stop, start
        scte = np.s_[ampbd:ampbd+npix, start:stop:step]
        sclean = np.s_[ampbd-npix:ampbd, start:stop:step]

        if on_bottom:
            scte, sclean = sclean, scte
        cleantraces = [np.median(im.pix[sclean], axis=0, keepdims=True)
                       for im in images]
        ctetraces = [np.median(im.pix[scte], axis=0, keepdims=True)
                     for im in images]
        # variance in median of a normal distribution is sigma^2 * pi / 2 / n
        # this is more careful than it makes sense to be here, but I
        # wanted to look it up again and figured I might as well.
        fac = np.sqrt(np.pi / 2 / npix)
        uncertainties = [
            fac * np.sqrt(np.median(
                im.ivar[sclean]**-1 + im.ivar[scte]**-1, axis=0,
                keepdims=True))
            for im in images]

        chi_function, names, guess, diff_step = (
            get_transfer_function_chi_names_guess_diff(tcte['ctefunc']))
        chi = partial(chi_function,
                      cleantraces=cleantraces,
                      ctetraces=ctetraces,
                      uncertainties=uncertainties)
        startguesses = [1, 20, 50, 100]
        chiguesses = np.array([chi([g] + guess) for g in startguesses])
        bestguess = np.argmin(np.sum(chiguesses**2, axis=1))
        par = least_squares(chi, [startguesses[bestguess], *guess],
                            diff_step=diff_step, loss='huber',
                            f_scale=5)
        offcols = f'{tcte["start"]}:{tcte["stop"]}'
        chi2dof = par.cost / len(par.fun)
        log.info(f'CTE fit {night} {camera} {par.x=} chi^2/dof={chi2dof:5.2f}')
        out = dict(NIGHT=night, CAMERA=camera, AMPLIFIER=ampname,
                   SECTOR=offcols, FUNC=tcte['ctefunc'],
                   CHI2PDF=float(chi2dof))
        for name, value in zip(names, par.x):
            out[name] = float(value)
        res.append(out)
    return res


def get_cte_images(night, camera, expids=None):
    """Get the images needed for a CTE fit for a particular night.

    This function looks up the appropriate exposure tables to find
    the CTE-detection image and the previous image, which is
    usually a normal flat field image.

    Parameters
    ----------
    night : int
        the night YEARMMDD integer
    camera : str
        the camera, e.g., z1

    Options
    -------
    expids : array-like
        list of exposure IDs to use; cte first, 120s flat second
        if None, determine from exposure table

    Returns
    -------
    List of preprocessed Images without CTE correction applied
    """
    # inside func to avoid circular import
    import desispec.io
    from desispec.io.util import decode_camword, erow_to_goodcamword, get_tempfilename

    log = get_logger()

    if expids is not None:
        filename = desispec.io.findfile('raw', night=night, expid=expids[0], readonly=True)
        header   = fitsio.read_header(filename, camera.upper())
        if not needs_ctecorr(header):
            log.info(f"No CTE correction needed for {night} {camera}")
            return None

    else:
        #- Look up exposures in production exposure table
        exptablefn = findfile('exptable', night=night)

        if not os.path.isfile(exptablefn) :
            mess = f"Cannot find exposure table file '{exptablefn}'. Because of that the flat exposures needed for the CTE correction modeling cannot be identified. Maybe check env. variables DESI_SPECTRO_REDUX and SPECPROD?"
            log.critical(mess)
            raise RuntimeError(mess)

        #- Read the exposure table and trim to just the good exposures covering this camera
        exptable = load_table(exptablefn, tabletype='exptable')
        keep = np.ones(len(exptable), dtype=bool)
        for i, erow in enumerate(exptable):
            if ((erow['LASTSTEP'] == 'ignore') or
                (camera not in decode_camword(erow_to_goodcamword(erow, suppress_logging=True)))):
                keep[i] = False

        if np.sum(keep) == 0:
            log.info(f'No good exposures on {night} for {camera}, so no CTE correction needed')
            return None
        else:
            exptable = exptable[keep]

        #- check the first exp of any OBSTYPE covering this camera to see if CTE corr is needed
        filename = desispec.io.findfile('raw', night=night, expid=exptable['EXPID'][0], readonly=True)
        header   = fitsio.read_header(filename, camera.upper())
        if not needs_ctecorr(header):
            log.info(f"No CTE correction needed for {night} {camera}")
            return None

        #- CTE correction is needed, so trim to OBSTYPE=flat
        keep = exptable['OBSTYPE'] == 'flat'
        exptable = exptable[keep]

        lengths = [1, 3, 10, 120]
        indices = []
        for length in lengths:
            selection = (np.abs(exptable['EXPTIME'] - length) < 0.1) & (exptable['OBSTYPE'] == 'flat')
            if np.sum(selection)<1:
                if length not in [3, 10]:
                    mess = f"No flat exposure of approx. {length} found for night {night} (in {exptablefn}). It's a requirement for the CTE correction model fit"
                    log.error(mess)
                    raise RuntimeError(mess)
                else:
                    continue
            indices.append(np.where(selection)[0][0])

        expids = list(exptable['EXPID'][indices])

    log.info(f"Will use exposures {expids} for {night} {camera} CTE corrections")

    images = list()
    for expid in expids:
        preproc_filename = desispec.io.findfile('preproc_for_cte', night=night, expid=expid, camera=camera)
        if not os.path.isfile(preproc_filename) :
            log.info(f"Computing {preproc_filename}")
            infile = desispec.io.findfile('raw', night=night, expid=expid, readonly=True)
            image  = desispec.io.read_raw(infile, camera, no_cte_corr=True)
            tmpfile = get_tempfilename(preproc_filename)
            desispec.io.write_image(tmpfile,image)
            os.rename(tmpfile, preproc_filename)
            log.info(f"Wrote {preproc_filename}")
            images.append(image)
        else :
            images.append(desispec.io.read_image(preproc_filename))

    return images


def fit_cte_night(night, camera, expids=None):
    """Fit the CTE parameters for a particular night.

    Parameters
    ----------
    night : int
        the night YEARMMDD integer
    camera : str
        the camera, e.g., z1

    Options
    -------
    expids : array-like
        list of exposure IDs to use;
        if None, determine from exposure table

    Returns
    -------
    Fit results; see fit_cte for details.
    """
    images = get_cte_images(night, camera, expids=expids)
    return fit_cte(images)


def get_image_model(preproc, psf=None):
    """Compute model for an image using aperture extraction.

    This computes a simple model for an image based on an aperture extraction.

    Parameters
    ----------
    preproc : Image
        Image to model

    Returns
    -------
    np.ndarray
    Model image
    """
    # inside func to avoid circular import
    import desispec.io
    import desispec.io.xytraceset
    import desispec.io.fiberflat

    meta = preproc.meta
    cfinder = CalibFinder([meta])
    psf_filename = cfinder.findfile("PSF")
    xyset = desispec.io.xytraceset.read_xytraceset(psf_filename)
    fiberflat_filename = cfinder.findfile("FIBERFLAT")
    fiberflat = desispec.io.fiberflat.read_fiberflat(fiberflat_filename)
    with_sky_model = True
    with_spectral_smoothing = True
    spectral_smoothing_sigma_length = 71
    no_traceshift = False

    mask = preproc.mask
    mimage = compute_image_model(
        preproc, xyset, fiberflat=fiberflat,
        with_spectral_smoothing=with_spectral_smoothing,
        spectral_smoothing_sigma_length=spectral_smoothing_sigma_length,
        with_sky_model=with_sky_model,
        psf=psf,
        fit_x_shift=(not no_traceshift))
    preproc.mask = mask
    # compute_image_model sets this to None for some reason.
    # we're restoring it.
    return mimage


def get_rowbyrow_image_model(preproc, fibermap=None,
                             spectral_smoothing_sigma_length=31,
                             nspec=500, psf=None, return_frame=False):
    """Compute row-by-row image model.

    This model uses a simultaneous PSF fit in each row to get better
    performance than get_image_model at the expense of reduced speed.
    The extracted fluxes are then combined with the PSF to produce a
    2D model image.

    Parameters
    ----------
    preproc : Image
        image to model
    fibermap : astropy.Table
        fibermap to use with image
    spectral_smoothing_sigma_length : int
        amount to smooth source spectra in model
    nspec : int
        number of spectra to extract and model
    psf : specter.psf.gausshermite.GaussHermitePSF
        PSF to use

    Returns
    -------
    np.ndarray
    Model image.
    """
    # load specter only if needed to simplify required dependencies
    import specter.psf
    import desispec.io   # inside func to avoid circular import

    meta = preproc.meta
    cfinder = CalibFinder([meta])
    if fibermap is None and hasattr(preproc, 'fibermap'):
        fibermap = preproc.fibermap
    if psf is None:
        psf_filename = cfinder.findfile("PSF")
        psf = specter.psf.load_psf(psf_filename)
        # try to update the trace shifts first?
        xytraceset = desispec.io.xytraceset.read_xytraceset(psf_filename)
        x, y, dx, ex, fiber, wave = compute_dx_from_cross_dispersion_profiles(
            xcoef=xytraceset.x_vs_wave_traceset._coeff,
            ycoef=xytraceset.y_vs_wave_traceset._coeff,
            wavemin=xytraceset.wavemin,
            wavemax=xytraceset.wavemax,
            image=preproc,
            fibers=np.arange(xytraceset.nspec, dtype=int))
        dx = np.median(dx)
        psf._x._coeff[:, 0] += dx

    res = rowbyrowextract.extract(preproc, psf, nspec=nspec,
                                  fibermap=fibermap, return_model=True)
    qframe, model, profile, profilepix = res

    fiberflat_filename = cfinder.findfile("FIBERFLAT")
    fiberflat = desispec.io.fiberflat.read_fiberflat(fiberflat_filename)
    fqframe = copy.deepcopy(qframe)
    flat = qfiberflat.qproc_apply_fiberflat(
        fqframe, fiberflat, return_flat=True)
    sfqframe = copy.deepcopy(fqframe)
    sky = qsky.qproc_sky_subtraction(sfqframe, return_skymodel=True)
    if spectral_smoothing_sigma_length > 0:
        sfflux = median_filter(
            sfqframe.flux, size=(1, spectral_smoothing_sigma_length),
            mode='nearest')
    else:
        sfflux = sfqframe.flux.copy()
    modflux = (sfflux + sky) * flat
    mframe = copy.deepcopy(sfqframe)
    mframe.flux = modflux
    mframe.sky = sky
    model = rowbyrowextract.model(mframe, profile, profilepix,
                                  preproc.pix.shape)
    if return_frame:
        model = (model, mframe)
    return model


def correct_image_via_model(image, niter=5, cte_params_filename=None):
    """Correct for CTE via an image model.

    The idea here is that you can roughly extract spectra from a
    CTE-affected image just by extracting as usual.  You can then
    create a noise-free image from that extraction in the context
    of a PSF and traces.  You can then apply CTE to that noise-free
    image.  The difference between the CTE-affected image and the
    original image is a noise-free version of what CTE is doing to your
    data, which you can then subtract.

    You can then re-extract from the corrected image, and repeat, improving
    the CTE correction.

    As pseudocode, this corresponds to:
    for i in range(niter):
    M = get_model(I)
    M' = CTE(M)
    I = I - (M' - M)
    This function implements that approach.

    Parameters
    ----------
    image : Image
        input image
    niter : int
        number of iterations to run

    Optional
    --------
    cte_params_filename : str or None (default)
        if filename is not None, use this one instead
        of the default one found with find_file




    Returns
    -------
    outimage : Image
        image after correction for CTE
    """


    log = get_logger()

    # here we get the list of amplifiers and the list
    # of sectors per amplifiers that are affected by CTE issues
    # and for which we have a model to apply
    # (only amplifers and sectors with a model are in this list)
    amp, cte = get_cte_params(image.meta, cte_params_filename=cte_params_filename)
    if len(cte) == 0 :
        log.info("No CTE correction to do for this image, return original")
        return image

    outimage = copy.deepcopy(image)

    previous_rms = 0.
    for i in range(niter):
        outmodel = get_rowbyrow_image_model(outimage)
        cteimage = outmodel.copy()

        for ampname in amp:
            ampreg = amp[ampname]
            imamp = outmodel[ampreg]
            cteamp = cte[ampname]
            if len(cteamp) == 0:
                # don't need to do anything in this case.
                continue

            need_to_reverse = ampreg[1].stop == image.pix.shape[1]
            if need_to_reverse:
                field, offset, sign = 'stop', ampreg[1].stop, -1
            else:
                field, offset, sign = 'start', 0, 1

            ctelocs = [sign * (x[field] - offset) for x in cteamp]
            individual_ctefuns = []
            for entry in cteamp :
                individual_ctefuns.append(get_transfer_function(entry))
            cteimage[ampreg] = apply_multiple_cte_effects(
                imamp[:, ::sign], locations=ctelocs,
                ctefuns=individual_ctefuns)[:, ::sign]
            correction_amp = imamp - cteimage[ampreg]
            mn, med, rms = sigma_clipped_stats(correction_amp)
            log.info(
                f'Correcting CTE, iteration {i}, correction rms {rms:6.3f}')
            if abs(rms-previous_rms)<0.1 :
                break
            previous_rms =rms
        correction = cteimage - outmodel
        outimage.pix = image.pix - correction
    return outimage
