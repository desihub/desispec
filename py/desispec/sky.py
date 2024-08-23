"""
desispec.sky
============

Utility functions to compute a sky model and subtract it.
"""

import os
import numpy as np
from collections import OrderedDict

from desispec.resolution import Resolution
from desispec.linalg import cholesky_solve
from desispec.linalg import cholesky_invert
from desispec.linalg import spline_fit
from desiutil.log import get_logger
from desispec import util
from desiutil import stats as dustat
import scipy,scipy.sparse,scipy.stats,scipy.ndimage
from scipy.signal import fftconvolve
import sys
from desispec.fiberbitmasking import get_fiberbitmasked_frame_arrays, get_fiberbitmasked_frame
import scipy.ndimage
from desispec.maskbits import specmask, fibermask
from desispec.preproc import get_amp_ids,parse_sec_keyword
from desispec.io import findfile,read_xytraceset
from desispec.calibfinder import CalibFinder
from desispec.preproc import get_amp_ids
from desispec.tpcorrparam import tpcorrmodel
import desispec.skygradpca


def _model_variance(frame,cskyflux,cskyivar,skyfibers) :
    """look at chi2 per wavelength and increase sky variance to reach chi2/ndf=1
    """

    log = get_logger()


    tivar = util.combine_ivar(frame.ivar[skyfibers], cskyivar[skyfibers])

    # the chi2 at a given wavelength can be large because on a cosmic
    # and not a psf error or sky non uniformity
    # so we need to consider only waves for which
    # a reasonable sky model error can be computed

    # mean sky
    msky = np.mean(cskyflux,axis=0)
    dwave = np.mean(np.gradient(frame.wave))
    dskydw = np.zeros(msky.shape)
    dskydw[1:-1]=(msky[2:]-msky[:-2])/(frame.wave[2:]-frame.wave[:-2])
    dskydw = np.abs(dskydw)

    # now we consider a worst possible sky model error (20% error on flat, 0.5A )
    max_possible_var = 1./(tivar+(tivar==0)) + (0.2*msky)**2 + (0.5*dskydw)**2

    # exclude residuals inconsistent with this max possible variance (at 3 sigma)
    bad = (frame.flux[skyfibers]-cskyflux[skyfibers])**2 > 3**2*max_possible_var
    tivar[bad]=0
    ndata = np.sum(tivar>0,axis=0)
    ok=np.where(ndata>1)[0]

    chi2  = np.zeros(frame.wave.size)
    chi2[ok] = np.sum(tivar*(frame.flux[skyfibers]-cskyflux[skyfibers])**2,axis=0)[ok]/(ndata[ok]-1)
    chi2[ndata<=1] = 1. # default

    # now we are going to evaluate a sky model error based on this chi2,
    # but only around sky flux peaks (>0.1*max)
    tmp   = np.zeros(frame.wave.size)
    tmp   = (msky[1:-1]>msky[2:])*(msky[1:-1]>msky[:-2])*(msky[1:-1]>0.1*np.max(msky))
    peaks = np.where(tmp)[0]+1
    dpix  = 2 #eval error range
    dpix2  = 3 # scale error range (larger)

    input_skyvar = 1./(cskyivar+(cskyivar==0))
    skyvar = input_skyvar + 0.

    # loop on peaks
    for peak in peaks :
        b=peak-dpix
        e=peak+dpix+1
        b2=peak-dpix2
        e2=peak+dpix2+1
        mchi2  = np.mean(chi2[b:e]) # mean reduced chi2 around peak
        mndata = np.mean(ndata[b:e]) # mean number of fibers contributing

        # sky model variance = sigma_flat * msky  + sigma_wave * dmskydw
        sigma_flat=0.005 # the fiber flat error is already included in the flux ivar, but empirical evidence we need an extra term
        sigma_wave=0.005 # A, minimum value
        res2=(frame.flux[skyfibers,b:e]-cskyflux[skyfibers,b:e])**2
        var=1./(tivar[:,b:e]+(tivar[:,b:e]==0))
        nd=np.sum(tivar[:,b:e]>0)
        sigma_wave = np.arange(0.005, 2, 0.005)

        #- pivar has shape (nskyfibers, npix, nsigma_wave)
        pivar = (tivar[:, b:e, np.newaxis]>0)/((var+(sigma_flat*msky[b:e])**2)[..., np.newaxis] + ((sigma_wave[np.newaxis,:]*dskydw[b:e, np.newaxis])**2)[np.newaxis, ...])
        #- chi2_of_sky_fibers has shape (nskyfibers, nsigma_wave)
        chi2_of_sky_fibers = np.sum(pivar*res2[..., np.newaxis],axis=1)/np.sum(tivar[:,b:e]>0,axis=1)[:, np.newaxis]
        #- normalization from median to mean for chi2 with 3 d.o.f.
        norm = 0.7888
        #- median_chi2 has shape (nsigma_wave,)
        median_chi2 = np.median(chi2_of_sky_fibers, axis=0)/norm
        if np.any(median_chi2 <= 1):
            #- first sigma_wave with median_chi2 <= 1 is the peak
            sigma_wave_peak = sigma_wave[np.where(median_chi2 <= 1)[0][0]]
        else :
            sigma_wave_peak = 2.
        log.info("peak at {}A : sigma_wave={}".format(int(frame.wave[peak]),sigma_wave_peak))
        skyvar[:,b2:e2] = input_skyvar[:,b2:e2] + (sigma_flat*msky[b2:e2])**2 + (sigma_wave_peak*dskydw[b2:e2])**2

    return (cskyivar>0)/(skyvar+(skyvar==0))


def get_sector_masks(frame):
    # get sector info from metadata

    meta = frame.meta
    cfinder = CalibFinder([meta])
    amps = get_amp_ids(meta)
    log = get_logger()

    sectors = []
    for amp in amps:

        sec = parse_sec_keyword(frame.meta['CCDSEC'+amp])
        yb = sec[0].start
        ye = sec[0].stop

        # fit an offset as part of sky sub if OFFCOLSX or CTECOLSX in calib
        # to correct for CTE issues
        # if CTECOLSX, another correction is also applied at preproc
        # see also doc/cte-correction.rst
        for key in [ "OFFCOLS"+amp , "CTECOLS"+amp ] :
            if cfinder.haskey(key) :
                val = cfinder.value(key)
                for tmp1 in val.split(",") :
                    tmp2 = tmp1.split(":")
                    if len(tmp2) != 2 :
                        mess="cannot decode {}={}".format(key,val)
                        log.error(mess)
                        raise KeyError(mess)
                    xb = max(sec[1].start,int(tmp2[0]))
                    xe = min(sec[1].stop,int(tmp2[1]))
                    sector = [yb,ye,xb,xe]
                    sectors.append(sector)
                    log.info("Adding CCD sector in amp {} with offset: {}".format(
                        amp,sector))

    if len(sectors) == 0:
        return [], [[]]

    psf_filename = findfile('psf', meta["NIGHT"], meta["EXPID"],
                            meta["CAMERA"])
    if not os.path.isfile(psf_filename) :
        log.error("No PSF file "+psf_filename)
        raise IOError("No PSF file "+psf_filename)
    log.info("Using PSF {}".format(psf_filename))
    tset = read_xytraceset(psf_filename)
    tmp_fibers = np.arange(frame. nspec)
    tmp_x = np.zeros(frame.flux.shape, dtype=float)
    tmp_y = np.zeros(frame.flux.shape, dtype=float)
    for fiber in tmp_fibers :
        tmp_x[fiber] = tset.x_vs_wave(fiber=fiber, wavelength=frame.wave)
        tmp_y[fiber] = tset.y_vs_wave(fiber=fiber, wavelength=frame.wave)

    masks = []
    templates = []
    for ymin, ymax, xmin, xmax in sectors:
        mask = ((tmp_y >= ymin) & (tmp_y < ymax) &
                (tmp_x >= xmin) & (tmp_x < xmax))
        masks.append(mask)
        constant_template = 1.0 * mask
        linear_template = (
            np.ones(frame.flux.shape[0])[:, None] *
            np.arange(frame.flux.shape[1])[None, :])
        linear_template -= np.min(linear_template*mask, axis=1, keepdims=True)
        tempmax = np.max(linear_template*mask, axis=1, keepdims=True)
        linear_template /= (tempmax + (tempmax == 0))
        linear_template *= mask
        templates.append([constant_template, linear_template])
    return masks, templates

def get_sky_fibers(fibermap, override_sky_targetids=None, exclude_sky_targetids=None):
    """
    Retrieve the fiber indices of sky fibers

    Args:
        fibermap: Table from frame FIBERMAP HDU (frame.fibermap)

    Options:
        override_sky_targetids (array of int): TARGETIDs to use, overriding fibermap
        exclude_sky_targetids (array of int): TARGETIDs to exclude

    Returns:
        array of indices of sky fibers to use

    By default we rely on fibermap['OBJTYPE']=='SKY', but we can also exclude
    some targetids by providing a list of them through exclude_sky_targetids
    or by just providing all the sky targetids directly (in that case
    the OBJTYPE information is ignored)

    Fibers with FIBERSTATUS bit VARIABLETHRU are also excluded
    """
    log = get_logger()
    # Grab sky fibers on this frame
    if override_sky_targetids is not None:
        log.info('Overriding default sky fiber list using override_sky_targetids')
        skyfibers = np.where(np.in1d(fibermap['TARGETID'], override_sky_targetids))[0]
        # we ignore OBJTYPEs
    else:
        oksky = (fibermap['OBJTYPE'] == 'SKY')
        oksky &= ((fibermap['FIBERSTATUS'] & fibermask.VARIABLETHRU) == 0)
        skyfibers = np.where(oksky)[0]
        if exclude_sky_targetids is not None:
            log.info('Excluding default sky fibers using exclude_sky_targetids')
            bads = np.in1d(fibermap['TARGETID'][skyfibers], exclude_sky_targetids)
            skyfibers = skyfibers[~bads]

    assert np.max(skyfibers) < len(fibermap)  #- indices, not fiber numbers
    return skyfibers

def compute_sky_linear(
        flux, ivar, Rframe, sectors, skyfibers, skygradpca, fibermap,
        fiberflat=None,
        min_iterations=5, max_iterations=100, nsig_clipping=4,
        tpcorrparam=None):
    log = get_logger()
    nfibers, nwave = flux.shape
    nskygradpc = skygradpca.flux.shape[0] if skygradpca is not None else 0
    current_ivar = ivar.copy()
    chi2 = np.zeros(flux.shape)
    nout_tot = 0
    bad_skyfibers = []
    Rsky = Rframe[skyfibers]

    if tpcorrparam is None:
        skytpcorrfixed = np.ones(nfibers)
    else:
        skytpcorrfixed = tpcorrmodel(tpcorrparam,
                                     fibermap['FIBER_X'], fibermap['FIBER_Y'])
        skytpcorrfixed = skytpcorrfixed[skyfibers]

    skytpcorr = skytpcorrfixed.copy()
    if sectors is not None:
        sectors, sectemplates = sectors

    for iteration in range(max_iterations) :
        # the matrix A is 1/2 of the second derivative of the chi2 with respect to the parameters
        # A_ij = 1/2 d2(chi2)/di/dj
        # A_ij = sum_fiber sum_wave_w ivar[fiber,w] d(model)/di[fiber,w] * d(model)/dj[fiber,w]

        # the vector B is 1/2 of the first derivative of the chi2 with respect to the parameters
        # B_i  = 1/2 d(chi2)/di
        # B_i  = sum_fiber sum_wave_w ivar[fiber,w] d(model)/di[fiber,w] * (flux[fiber,w]-model[fiber,w])

        # the model is model[fiber]=R[fiber]*sky
        # and the parameters are the unconvolved sky flux at the wavelength i

        # so, d(model)/di[fiber,w] = R[fiber][w,i]
        # this gives
        # A_ij = sum_fiber  sum_wave_w ivar[fiber,w] R[fiber][w,i] R[fiber][w,j]
        # A = sum_fiber ( diag(sqrt(ivar))*R[fiber] ) ( diag(sqrt(ivar))* R[fiber] )^t
        # A = sum_fiber sqrtwR[fiber] sqrtwR[fiber]^t
        # and
        # B = sum_fiber sum_wave_w ivar[fiber,w] R[fiber][w] * flux[fiber,w]
        # B = sum_fiber sum_wave_w sqrt(ivar)[fiber,w]*flux[fiber,w] sqrtwR[fiber,wave]

        #A=scipy.sparse.lil_matrix((nwave,nwave)).tocsr()

        # Julien can do A^T C^-1 A, A^T C^-1 b himself, but I like to write it
        # out
        # the model is that
        # frame = R*(sky spectrum + sum(PC * (a*(x-<x>) + b*(y-<y>)))) + offsets
        # We could consider adding a mild prior to deal with ill-conditioned
        # matrices.

        # note: the design matrix we set up has the following parameters:
        # first nwave columns: deconvolved flux at each wavelength
        # next nsector columns: sector offsets
        # next 2*nskygradpc columns: sky gradient amplitudes in x & y
        # direction for each PC.

        # in a separate step we also set up a 'tpcorr' model, reflecting
        # different throughputs of each fiber.

        # the full model is:
        # R(sky + amplitudes * skygradpc * dx)*tpcorr + sector

        nsector = len(sectors)
        nsectemplate = sum([len(x) for x in sectemplates])
        npar = nwave + nsectemplate + nskygradpc*2

        yy = np.zeros((nwave*nfibers))

        SD = scipy.sparse.dia_matrix((nwave*nfibers,nwave*nfibers))
        SD.setdiag(current_ivar.reshape(-1))

        # loop on fiber to handle resolution
        allrows = []
        allcols = []
        allvals = []
        for fiber in range(nfibers):
            if fiber % 10 == 0:
                log.info("iter %d sky fiber %d/%d"%(iteration,fiber,nfibers))
            R = Rsky[fiber]
            rows, cols, vals = scipy.sparse.find(R)
            allrows.append(rows+fiber*nwave)
            allcols.append(cols)
            allvals.append(vals)
            yy[fiber*nwave:(fiber+1)*nwave] = flux[fiber]
            if skygradpca is not None:
                dx = skygradpca.dx[skygradpca.skyfibers[fiber]]
                dy = skygradpca.dy[skygradpca.skyfibers[fiber]]
                for skygradpcind in range(nskygradpc):
                    convskygradpc = R.dot(skygradpca.deconvflux[skygradpcind])
                    allrows.append(np.arange(nwave)+fiber*nwave)
                    allcols.append(nwave + nsectemplate + skygradpcind*2 +
                                   np.zeros(nwave, dtype='i4'))
                    allvals.append(convskygradpc * dx)
                    allrows.append(np.arange(nwave)+fiber*nwave)
                    allcols.append(nwave + nsectemplate + skygradpcind*2 + 1 +
                                   np.zeros(nwave, dtype='i4'))
                    allvals.append(convskygradpc * dy)
        # boost model by throughput corrections
        for i in range(len(allvals)):
            allvals[i] *= skytpcorr[allrows[i] // nwave]

        i = 0
        for j, secmask in enumerate(sectors):
            for template in sectemplates[j]:
                rows = np.flatnonzero(secmask[skyfibers])
                cols = np.full(len(rows), nwave+i)
                if fiberflat is not None:
                    flat = (
                        fiberflat.fiberflat[skyfibers][secmask[skyfibers]].ravel())
                else:
                    flat = np.ones(rows.shape)
                vals = (template[skyfibers][secmask[skyfibers]].ravel()/
                        (flat + (flat == 0)))
                allrows.append(rows)
                allcols.append(cols)
                allvals.append(vals)
                i += 1

        design = scipy.sparse.coo_matrix(
            (np.concatenate(allvals),
             (np.concatenate(allrows), np.concatenate(allcols))),
            shape=(nwave*nfibers, npar))
        design = design.tocsr()

        A = design.T.dot(SD.dot(design))
        A = A.toarray()
        B = design.T.dot(SD.dot(yy))

        log.info("iter %d solving"%iteration)
        w = A.diagonal() > 0
        A_pos_def = A[w,:]
        A_pos_def = A_pos_def[:,w]
        param = B*0
        try:
            param[w]=cholesky_solve(A_pos_def,B[w])
        except:
            log.info("cholesky failed, trying svd in iteration {}".format(iteration))
            param[w]=np.linalg.lstsq(A_pos_def,B[w], rcond=None)[0]
        deconvolved_sky = param[:nwave]
        modeled_sky = design.dot(param).reshape(flux.shape)
        modeled_secoffs = (
            design[:, nwave:nwave + nsectemplate].dot(
                param[nwave:nwave + nsectemplate]))
        modeled_secoffs = modeled_secoffs.reshape(flux.shape)

        log.info("iter %d compute chi2"%iteration)

        medflux=np.zeros(nfibers)
        for fiber in range(nfibers) :
            # the parameters are directly the unconvolve sky flux
            # so we simply have to reconvolve it
            chi2[fiber]=current_ivar[fiber]*(flux[fiber]-modeled_sky[fiber])**2
            ok=(current_ivar[fiber]>0)
            if np.sum(ok)>0 :
                medflux[fiber] = np.median((flux[fiber]-modeled_sky[fiber])[ok])

        log.info("rejecting")

        # whole fiber with excess flux
        if np.sum(medflux!=0) > 2 : # at least 3 valid sky fibers
            rms_from_nmad = 1.48*np.median(np.abs(medflux[medflux!=0]))
            # discard fibers that are 7 sigma away
            badfibers=np.where(np.abs(medflux)>7*rms_from_nmad)[0]
            for fiber in badfibers :
                log.warning("discarding fiber {} with median flux = {:.2f} > 7*{:.2f}".format(skyfibers[fiber],medflux[fiber],rms_from_nmad))
                current_ivar[fiber]=0
                # set a mask bit here
                bad_skyfibers.append(skyfibers[fiber])
        nout_iter=0
        if iteration<1 :
            # only remove worst outlier per wave
            # apply rejection iteratively, only one entry per wave among fibers
            # find waves with outlier (fastest way)
            nout_per_wave=np.sum(chi2>nsig_clipping**2,axis=0)
            selection=np.where(nout_per_wave>0)[0]
            for i in selection :
                worst_entry=np.argmax(chi2[:,i])
                current_ivar[worst_entry,i]=0
                nout_iter += 1

        else :
            # remove all of them at once
            bad=(chi2>nsig_clipping**2)
            current_ivar *= (bad==0)
            nout_iter += np.sum(bad)

        if tpcorrparam is not None:
            # the throughput of each fiber varies, usually following
            # the tpcorrparam pca.  We want to find the coefficients
            # for these principal components.
            # the code here is a bit hard to track primarily because
            # in the iterative scheme, we have already applied some PCA correction
            # in the previous iteration.  Here we remove the previous PCA correction,
            # and then re-fit the result.  This may be equivalent to fitting directly
            # and then added the fit results to the existing pca coefficients, but
            # that wasn't the approach taken here.

            # the _current_ pca-tracked bit of the tpcorr we are using is
            # in tppca0.  This is the current total skytpcorr, divided by the fixed
            # bit that comes from the mean and the spatial within-patrol-radius model.

            tppca0 = skytpcorr[:, None]/skytpcorrfixed[:, None]
            tppcam = tpcorrparam.pca[:, skyfibers]
            # in the design matrix and flux residuals, we divide out tppca0 from
            # modeled_sky so that we have only the pre-PCA skies
            # we use the modeled_sky without the offsets since this is a throughput
            # effect and not an instrumental effect.
            sky_no_offsets = modeled_sky - modeled_secoffs
            aa = np.array([(sky_no_offsets*tppcam0[:, None]/tppca0).reshape(-1)
                           for tppcam0 in tppcam]).T
            fluxresid = flux - modeled_secoffs - sky_no_offsets / tppca0
            # then we solve for the PCA coefficients that best take the
            # pre-PCA skies to the pre-PCA sky residuals (fluxresid).
            skytpcorrcoeff = np.linalg.lstsq(
                aa.T.dot(current_ivar.reshape(-1)[:, None]*aa),
                aa.T.dot((current_ivar*fluxresid).reshape(-1)),
                rcond=None)[0]
            skytpcorr = skytpcorrfixed.copy()
            for coeff, vec in zip(skytpcorrcoeff,
                                  tpcorrparam.pca[:, skyfibers]):
                skytpcorr += coeff*vec

        nout_tot += nout_iter

        sum_chi2=float(np.sum(chi2))
        ndf=int(np.sum(chi2>0)-nwave)
        chi2pdf=0.
        if ndf>0 :
            chi2pdf=sum_chi2/ndf
        log.info("iter #%d chi2=%f ndf=%d chi2pdf=%f nout=%d"%(iteration,sum_chi2,ndf,chi2pdf,nout_iter))

        # at least min_iterations
        if (nout_iter == 0) & (iteration >= min_iterations - 1):
            break

    if nsectemplate > 0:
        log.info('sectors: %d sectors fit, values %s' %
                 (nsector, ' '.join(
                     [str(x) for x in param[nwave:nwave+nsectemplate]])))

    if nskygradpc > 0:
        log.info(('Fit with %d spatial PCs, amplitudes ' % nskygradpc) +
                 ' '.join(['%.1f' % x for x in param[nwave+nsectemplate:]]))

    log.info("compute the parameter covariance")
    # we may have to use a different method to compute this
    # covariance
    try :
        parameter_covar=cholesky_invert(A)
        # the above is too slow
        # maybe invert per block, sandwich by R
    except np.linalg.linalg.LinAlgError :
        log.warning("cholesky_solve_and_invert failed, switching to np.linalg.lstsq and np.linalg.pinv")
        parameter_covar = np.linalg.pinv(A)

    if tpcorrparam is None:
        skytpcorr = np.ones(len(fibermap), dtype='f4')
    else:
        skytpcorr = tpcorrmodel(tpcorrparam,
                                fibermap['FIBER_X'], fibermap['FIBER_Y'],
                                skytpcorrcoeff)

    unconvflux = param[:nwave].copy()
    skygradpcacoeff = param[
        nwave + nsectemplate:nwave+nsectemplate+nskygradpc*2]
    if skygradpca is not None:
        modeled_sky = desispec.skygradpca.evaluate_model(
            skygradpca, Rframe, skygradpcacoeff, mean=unconvflux)
    else:
        modeled_sky = np.zeros((len(Rframe), nwave), dtype='f8')
        for i in range(len(Rframe)):
            modeled_sky[i] = Rframe[i].dot(unconvflux)

    sector_offsets = np.zeros((len(fibermap), flux.shape[1]), dtype='f4')
    i = 0
    for j, secmask in enumerate(sectors):
        for sectemplate in sectemplates[j]:
            sector_offsets[secmask] += param[nwave+i] * sectemplate[secmask]
            i += 1
    if len(sectors) > 0 and fiberflat is not None:
        flat = fiberflat.fiberflat + (fiberflat.fiberflat == 0)
        sector_offsets /= flat

    modeled_sky *= skytpcorr[:, None]
    bad_wavelengths = ~(w[:nwave])
    modeled_sky += sector_offsets

    return (param, parameter_covar, modeled_sky, current_ivar, nout_tot,
            skytpcorr, bad_skyfibers, bad_wavelengths, sector_offsets,
            skygradpcacoeff)


def compute_sky(
    frame, nsig_clipping=4., max_iterations=100, model_ivar=False,
    add_variance=True, adjust_wavelength=False, adjust_lsf=False,
    only_use_skyfibers_for_adjustments=True, pcacorr=None,
    fit_offsets=False, fiberflat=None, skygradpca=None,
        min_iterations=5, tpcorrparam=None,
        exclude_sky_targetids=None, override_sky_targetids=None):
    """Compute a sky model.

    Sky[fiber,i] = R[fiber,i,j] Flux[j]

    Input flux are expected to be flatfielded!
    We don't check this in this routine.

    Args:
        frame : Frame object, which includes attributes
          - wave : 1D wavelength grid in Angstroms
          - flux : 2D flux[nspec, nwave] density
          - ivar : 2D inverse variance of flux
          - mask : 2D inverse mask flux (0=good)
          - resolution_data : 3D[nspec, ndiag, nwave]  (only sky fibers)
        nsig_clipping : [optional] sigma clipping value for outlier rejection
        max_iterations : int, maximum number of iterations
        model_ivar : replace ivar by a model to avoid bias due to correlated flux and ivar. this has a negligible effect on sims.
        add_variance : evaluate calibration error and add this to the sky model variance
        adjust_wavelength : adjust the wavelength of the sky model on sky lines to improve the sky subtraction
        adjust_lsf : adjust the LSF width of the sky model on sky lines to improve the sky subtraction
        only_use_skyfibers_for_adjustments : interpolate adjustments using sky fibers only
        pcacorr : SkyCorrPCA object to interpolate the wavelength or LSF adjustment from sky fibers to all fibers
        fit_offsets : fit offsets for regions defined in calib
        fiberflat : desispec.FiberFlat object used for the fit of offsets
        skygradpca : SkyGradPCA object to use to fit sky gradients, or None
        min_iterations : int, minimum number of iterations
        tpcorrparam : TPCorrParam object to use to fit fiber throughput
            variations, or None

    returns SkyModel object with attributes wave, flux, ivar, mask
    """

    log=get_logger()
    log.info("starting")

    skyfibers = get_sky_fibers(frame.fibermap, override_sky_targetids=override_sky_targetids,
                              exclude_sky_targetids=exclude_sky_targetids)

    #- Hack: test tile 81097 (observed 20210430/00086750) had set
    #- FIBERSTATUS bit UNASSIGNED for sky targets on stuck positioners.
    #- Undo that.
    if (frame.meta is not None) and ('TILEID' in frame.meta) and (frame.meta['TILEID'] == 81097):
        log.info('Unsetting FIBERSTATUS UNASSIGNED for tileid 81097 sky fibers')
        frame.fibermap['FIBERSTATUS'][skyfibers] &= ~1

    nwave=frame.nwave

    current_ivar = get_fiberbitmasked_frame_arrays(frame,bitmask='sky',ivar_framemask=True,return_mask=False)

    # checking ivar because some sky fibers have been disabled
    bad=(np.sum(current_ivar[skyfibers]>0,axis=1)==0)
    good=~bad

    if np.any(bad) :
        log.warning("{} sky fibers discarded (because ivar=0 or bad FIBERSTATUS), only {} left.".format(np.sum(bad),np.sum(good)))
        skyfibers = skyfibers[good]

    if np.sum(good)==0 :
        message = "no valid sky fibers"
        log.error(message)
        raise RuntimeError(message)

    nfibers=len(skyfibers)

    current_ivar = current_ivar[skyfibers]
    flux = frame.flux[skyfibers]

    input_ivar=None
    if model_ivar :
        log.info("use a model of the inverse variance to remove bias due to correlated ivar and flux")
        input_ivar=current_ivar.copy()
        median_ivar_vs_wave  = np.median(current_ivar,axis=0)
        median_ivar_vs_fiber = np.median(current_ivar,axis=1)
        median_median_ivar   = np.median(median_ivar_vs_fiber)
        for f in range(current_ivar.shape[0]) :
            threshold=0.01
            current_ivar[f] = median_ivar_vs_fiber[f]/median_median_ivar * median_ivar_vs_wave
            # keep input ivar for very low weights
            ii=(input_ivar[f]<=(threshold*median_ivar_vs_wave))
            #log.info("fiber {} keep {}/{} original ivars".format(f,np.sum(ii),current_ivar.shape[1]))
            current_ivar[f][ii] = input_ivar[f][ii]


    chi2=np.zeros(flux.shape)

    #max_iterations=2 ; log.warning("DEBUGGING LIMITING NUMBER OF ITERATIONS")

    if fit_offsets:
        sectors = get_sector_masks(frame)
    else:
        sectors = [], [[]]

    if skygradpca is not None:
        desispec.skygradpca.configure_for_xyr(
            skygradpca, frame.fibermap['FIBERASSIGN_X'],
            frame.fibermap['FIBERASSIGN_Y'],
            frame.R, skyfibers=skyfibers)

    res = compute_sky_linear(
        flux, current_ivar, frame.R, sectors, skyfibers, skygradpca,
        frame.fibermap,
        fiberflat=fiberflat, min_iterations=min_iterations,
        max_iterations=max_iterations, nsig_clipping=nsig_clipping,
        tpcorrparam=tpcorrparam)
    (param, parameter_covar, modeled_sky, current_ivar, nout_tot, skytpcorr,
     bad_skyfibers, bad_wavelengths, background, skygradpcacoeff) = res
    deconvolved_sky = param[:nwave]

    log.info("compute mean resolution")
    # we make an approximation for the variance to save CPU time
    # we use the average resolution of all fibers in the frame:
    mean_res_data=np.mean(frame.resolution_data,axis=0)
    Rmean = Resolution(mean_res_data)

    log.info("compute convolved sky and ivar")

    parameter_sky_covar = parameter_covar[:nwave, :nwave]

    # The parameters are directly the unconvolved sky
    # First convolve with average resolution :
    convolved_sky_covar=Rmean.dot(parameter_sky_covar).dot(Rmean.T.todense())

    # and keep only the diagonal
    convolved_sky_var=np.diagonal(convolved_sky_covar)

    # inverse
    convolved_sky_ivar=(convolved_sky_var>0)/(convolved_sky_var+(convolved_sky_var==0))

    # and simply consider it's the same for all spectra
    cskyivar = np.tile(convolved_sky_ivar, frame.nspec).reshape(frame.nspec, nwave)

    # remove background for line fitting; add back at end
    cskyflux = modeled_sky - background
    frame.flux -= background

    # See if we can improve the sky model by readjusting the wavelength and/or the width of sky lines
    dwavecoeff = None
    dlsfcoeff = None
    if adjust_wavelength or adjust_lsf :
        log.info("adjust the wavelength of sky spectrum on sky lines to improve sky subtraction ...")

        if adjust_wavelength :
            # compute derivative of sky w.r.t. wavelength
            dskydwave = np.gradient(cskyflux,axis=1)/np.gradient(frame.wave)
        else :
            dskydwave = None

        if adjust_lsf :
            # compute derivative of sky w.r.t. lsf width
            dwave = np.mean(np.gradient(frame.wave))
            dsigma_A   = 0.3 #A
            dsigma_bin = dsigma_A/dwave # consider this extra width for the PSF (sigma' = sqrt(sigma**2+dsigma**2))
            hw=int(4*dsigma_bin)+1
            x=np.arange(-hw,hw+1)
            k=np.zeros((3,x.size)) # a Gaussian kernel
            k[1]=np.exp(-x**2/dsigma_bin**2/2.)
            k/=np.sum(k)
            tmp = fftconvolve(cskyflux,k,mode="same")
            dskydlsf = (tmp-cskyflux)/dsigma_A # variation of line shape with width
        else :
            dskydlsf = None

        # detect peaks in mean sky spectrum
        # peaks = local maximum larger than 10% of max peak
        meansky = np.mean(cskyflux,axis=0)
        tmp   = (meansky[1:-1]>meansky[2:])*(meansky[1:-1]>meansky[:-2])*(meansky[1:-1]>0.1*np.max(meansky))
        peaks = np.where(tmp)[0]+1
        # remove edges
        peaks = peaks[(peaks>10)&(peaks<meansky.size-10)]
        peak_wave=frame.wave[peaks]

        log.info("Number of peaks: {}".format(peaks.size))
        if  peaks.size < 10 :
            log.info("Wavelength of peaks: {}".format(peak_wave))

        # define area around each sky line to adjust
        dwave = np.mean(np.gradient(frame.wave))
        dpix = int(3//dwave)+1

        # number of parameters to fit for each peak: delta_wave , delta_lsf , scale of sky , a background (to absorb source signal)
        nparam = 2
        if adjust_wavelength : nparam += 1
        if adjust_lsf : nparam += 1

        AA=np.zeros((nparam,nparam))
        BB=np.zeros((nparam))

        # temporary arrays with best fit parameters on peaks
        # for each fiber, with errors and chi2/ndf
        peak_scale=np.zeros((frame.nspec,peaks.size))
        peak_scale_err=np.zeros((frame.nspec,peaks.size))
        peak_dw=np.zeros((frame.nspec,peaks.size))
        peak_dw_err=np.zeros((frame.nspec,peaks.size))
        peak_dlsf=np.zeros((frame.nspec,peaks.size))
        peak_dlsf_err=np.zeros((frame.nspec,peaks.size))

        peak_chi2pdf=np.zeros((frame.nspec,peaks.size))

        # interpolated values across peaks, after selection
        # based on precision and chi2
        interpolated_sky_dwave=np.zeros(frame.flux.shape)
        interpolated_sky_dlsf=np.zeros(frame.flux.shape)

        # loop on fibers and then on sky spectrum peaks
        if only_use_skyfibers_for_adjustments :
            fibers_in_fit = skyfibers
        else :
            fibers_in_fit = np.arange(frame.nspec)

        # restrict to fibers with ivar!=0
        ok = np.sum(frame.ivar[fibers_in_fit],axis=1)>0
        fibers_in_fit = fibers_in_fit[ok]

        # loop on sky spectrum peaks, compute for all fibers simultaneously
        for j,peak in enumerate(peaks) :
            b = peak-dpix
            e = peak+dpix+1
            npix = e - b
            flux = frame.flux[fibers_in_fit][:,b:e]
            ivar = frame.ivar[fibers_in_fit][:,b:e]
            if b < 0 or e > frame.flux.shape[1] :
                log.warning("skip peak on edge of spectrum with b={} e={}".format(b,e))
                continue
            M = np.zeros((fibers_in_fit.size, nparam, npix))
            index = 0
            M[:, index] = np.ones(npix); index += 1
            M[:, index] = cskyflux[fibers_in_fit][:, b:e]; index += 1
            if adjust_wavelength : M[:, index] = dskydwave[fibers_in_fit][:, b:e]; index += 1
            if adjust_lsf        : M[:, index] = dskydlsf[fibers_in_fit][:, b:e]; index += 1
            # Solve (M * W * M.T) X = (M * W * flux)
            BB = np.einsum('ijk,ik->ij', M, ivar*flux)
            AA = np.einsum('ijk,ik,ilk->ijl', M, ivar, M)
            # solve linear system
            #- TODO: replace with X = np.linalg.solve(AA, BB) ?
            try:
                AAi=np.linalg.inv(AA)
            except np.linalg.LinAlgError as e:
                log.warning(str(e))
                continue
            # save best fit parameter and errors
            X = np.einsum('ijk,ik->ij', AAi, BB)
            X_err = np.sqrt(AAi*(AAi>0))
            index = 1
            peak_scale[fibers_in_fit,j] = X[:, index]
            peak_scale_err[fibers_in_fit,j] = X_err[:, index, index]
            index += 1
            if adjust_wavelength:
                peak_dw[fibers_in_fit, j] = X[:, index]
                peak_dw_err[fibers_in_fit, j] = X_err[:, index, index]
                index += 1
            if adjust_lsf:
                peak_dlsf[fibers_in_fit, j] = X[:, index]
                peak_dlsf_err[fibers_in_fit, j] = X_err[:, index, index]
                index += 1

            residuals = flux
            for index in range(nparam) :
                #for index in range(3) : # needed for compatibility with master (but this was a bug)
                residuals -= X[:,index][:, np.newaxis]*M[:,index]

            variance = 1.0/(ivar+(ivar==0)) + (0.05*M[:,1])**2
            peak_chi2pdf[fibers_in_fit, j] = np.sum((ivar>0)/variance*(residuals)**2, axis=1)/(npix-nparam)

        for i in fibers_in_fit :
            # for each fiber, select valid peaks and interpolate
            ok=(peak_chi2pdf[i]<2)
            if adjust_wavelength :
                ok &= (peak_dw_err[i]>0.)&(peak_dw_err[i]<0.1) # error on wavelength shift
            if adjust_lsf :
                ok &= (peak_dlsf_err[i]>0.)&(peak_dlsf_err[i]<0.3) # error on line width (quadratic, so 0.3 mean a change of width of 0.3**2/2~5%)
            # piece-wise linear interpolate across the whole spectrum between the sky line peaks
            # this interpolation will be used to alter the whole sky spectrum
            if np.sum(ok)>0 :
                if adjust_wavelength :
                    interpolated_sky_dwave[i]=np.interp(frame.wave,peak_wave[ok],peak_dw[i,ok])
                if adjust_lsf :
                    interpolated_sky_dlsf[i]=np.interp(frame.wave,peak_wave[ok],peak_dlsf[i,ok])
                line=""
                if adjust_wavelength :
                    line += " dlambda mean={:4.3f} rms={:4.3f} A".format(np.mean(interpolated_sky_dwave[i]),np.std(interpolated_sky_dwave[i]))
                if adjust_lsf :
                    line += " dlsf mean={:4.3f} rms={:4.3f} A".format(np.mean(interpolated_sky_dlsf[i]),np.std(interpolated_sky_dlsf[i]))
                log.info(line)

        # we ignore the interpolated_sky_scale which is too sensitive
        # to CCD defects or cosmic rays

        if pcacorr is None :
            if only_use_skyfibers_for_adjustments :
                goodfibers=fibers_in_fit
            else : # keep all except bright objects and interpolate over them
                mflux=np.median(frame.flux,axis=1)
                mmflux=np.median(mflux)
                rms=1.48*np.median(np.abs(mflux-mmflux))
                selection=(mflux<mmflux+2*rms)
                # at least 80% of good pixels
                ngood=np.sum((frame.ivar>0)*(frame.mask==0),axis=1)
                selection &= (ngood>0.8*frame.flux.shape[1])
                goodfibers=np.where(mflux<mmflux+2*rms)[0]
                log.info("number of good fibers=",goodfibers.size)
            allfibers=np.arange(frame.nspec)
            # the actual median filtering
            if adjust_wavelength :
                for j in range(interpolated_sky_dwave.shape[1]) :
                    interpolated_sky_dwave[:,j] = np.interp(np.arange(interpolated_sky_dwave.shape[0]),goodfibers,interpolated_sky_dwave[goodfibers,j])
                cskyflux += interpolated_sky_dwave*dskydwave
            if adjust_lsf : # simple interpolation over fibers
                for j in range(interpolated_sky_dlsf.shape[1]) :
                    interpolated_sky_dlsf[:,j] = np.interp(np.arange(interpolated_sky_dlsf.shape[0]),goodfibers,interpolated_sky_dlsf[goodfibers,j])
                cskyflux += interpolated_sky_dlsf*dskydlsf

        else :


            def fit_and_interpolate(delta,skyfibers,mean,components,label="") :
                mean_and_components = np.zeros((components.shape[0]+1,
                                                components.shape[1],
                                                components.shape[2]))
                mean_and_components[0]  = mean
                mean_and_components[1:] = components
                ncomp=mean_and_components.shape[0]
                log.info("Will fit a linear combination on {} components for {}".format(ncomp,label))
                AA=np.zeros((ncomp,ncomp))
                BB=np.zeros(ncomp)
                for i in range(ncomp) :
                    BB[i] = np.sum(delta[skyfibers]*mean_and_components[i][skyfibers])
                    for j in range(i,ncomp) :
                        AA[i,j] = np.sum(mean_and_components[i][skyfibers]*mean_and_components[j][skyfibers])
                        if j!=i :
                            AA[j,i]=AA[i,j]
                AAi=np.linalg.inv(AA)
                X=AAi.dot(BB)
                log.info("Best fit linear coefficients for {} = {}".format(label,list(X)))
                result = np.zeros_like(delta)
                for i in range(ncomp) :
                    result += X[i]*mean_and_components[i]
                return result, X


            # we are going to fit a linear combination of the PCA coefficients only on the sky fibers
            # and then apply the linear combination to all fibers
            log.info("Use PCA skycorr")

            if adjust_wavelength :
                correction, dwavecoeff = fit_and_interpolate(
                    interpolated_sky_dwave, skyfibers,
                    pcacorr.dwave_mean, pcacorr.dwave_eigenvectors,
                    label="wavelength")
                cskyflux  += correction*dskydwave
            if adjust_lsf :
                correction, dlsfcoeff = fit_and_interpolate(
                    interpolated_sky_dlsf,skyfibers,
                    pcacorr.dlsf_mean,pcacorr.dlsf_eigenvectors,label="LSF")
                cskyflux  += correction*dskydlsf


    # look at chi2 per wavelength and increase sky variance to reach chi2/ndf=1
    if skyfibers.size > 1 and add_variance :
        modified_cskyivar = _model_variance(frame,cskyflux,cskyivar,skyfibers)
    else :
        modified_cskyivar = cskyivar.copy()

    cskyflux += background
    frame.flux += background

    # set sky flux and ivar to zero to poorly constrained regions
    # and add margins to avoid expolation issues with the resolution matrix
    # limit to sky spectrum part of A
    wmask = bad_wavelengths.astype(float)
    # empirically, need to account for the full width of the resolution band
    # (realized here by applying twice the resolution)
    wmask = Rmean.dot(Rmean.dot(wmask))
    bad = np.where(wmask!=0)[0]
    cskyflux[:,bad]=0.
    modified_cskyivar[:,bad]=0.

    # minimum number of fibers at each wavelength
    min_number_of_fibers = min(10,max(1,skyfibers.size//2))
    fibers_with_signal=np.sum(current_ivar>0,axis=0)
    bad = (fibers_with_signal<min_number_of_fibers)
    # increase by 1 pixel
    bad[1:-1] |= bad[2:]
    bad[1:-1] |= bad[:-2]
    cskyflux[:,bad]=0.
    modified_cskyivar[:,bad]=0.

    mask = (modified_cskyivar==0).astype(np.uint32)

    # add mask bits for bad sky fibers
    bad_skyfibers = np.unique(bad_skyfibers)
    if bad_skyfibers.size > 0 :
        mask[bad_skyfibers] |= specmask.mask("BADSKY")

    skymodel = SkyModel(frame.wave.copy(), cskyflux, modified_cskyivar, mask,
                        nrej=nout_tot, stat_ivar = cskyivar,
                        dwavecoeff=dwavecoeff, dlsfcoeff=dlsfcoeff,
                        throughput_corrections_model=skytpcorr,
                        skygradpcacoeff=skygradpcacoeff,
                        skytargetid=frame.fibermap['TARGETID'][skyfibers])
    # keep a record of the statistical ivar for QA
    if adjust_wavelength :
        skymodel.dwave = interpolated_sky_dwave
    if adjust_lsf :
        skymodel.dlsf  = interpolated_sky_dlsf

    skymodel.throughput_corrections = calculate_throughput_corrections(
        frame, skymodel)

    return skymodel


class SkyModel(object):
    def __init__(self, wave, flux, ivar, mask, header=None, nrej=0,
                 stat_ivar=None, throughput_corrections=None,
                 throughput_corrections_model=None,
                 dwavecoeff=None, dlsfcoeff=None, skygradpcacoeff=None,
                 skytargetid=None):
        """Create SkyModel object

        Args:
            wave  : 1D[nwave] wavelength in Angstroms
            flux  : 2D[nspec, nwave] sky model to subtract
            ivar  : 2D[nspec, nwave] inverse variance of the sky model
            mask  : 2D[nspec, nwave] 0=ok or >0 if problems; 32-bit
            header : (optional) header from FITS file HDU0
            nrej : (optional) Number of rejected pixels in fit
            stat_ivar  : 2D[nspec, nwave] inverse variance of the statistical inverse variance
            throughput_corrections : 1D (optional) Residual multiplicative throughput corrections for each fiber
            throughput_corrections_model : 1D (optional) Model multiplicative throughput corrections for each fiber
            dwavecoeff : (optional) 1D[ncoeff] vector of PCA coefficients for wavelength offsets
            dlsfcoeff : (optional) 1D[ncoeff] vector of PCA coefficients for LSF size changes
            skygradpcacoeff : (optional) 1D[ncoeff] vector of gradient amplitudes for
                sky gradient spectra.
            skytargetid : (optional) 1D[nsky] vector of TARGETIDs of fibers used for sky determination
        All input arguments become attributes
        """
        assert wave.ndim == 1
        assert flux.ndim == 2
        assert ivar.shape == flux.shape
        assert mask.shape == flux.shape

        self.nspec, self.nwave = flux.shape
        self.wave = wave
        self.flux = flux
        self.ivar = ivar
        self.mask = util.mask32(mask)
        self.header = header
        self.nrej = nrej
        self.stat_ivar = stat_ivar
        self.throughput_corrections = throughput_corrections
        self.throughput_corrections_model = throughput_corrections_model
        self.dwave = None # wavelength corrections
        self.dlsf  = None # LSF corrections
        self.dwavecoeff = dwavecoeff
        self.dlsfcoeff = dlsfcoeff
        self.skygradpcacoeff = skygradpcacoeff
        self.skytargetid = skytargetid

    def __getitem__(self, index):
        """
        Return a subset of the fibers for this skymodel

        e.g. `stdsky = sky[stdstar_indices]`
        """
        #- convert index to 1d array to maintain dimentionality of sliced arrays
        if not isinstance(index, slice):
            index = np.atleast_1d(index)

        flux = self.flux[index]
        ivar = self.ivar[index]
        mask = self.mask[index]

        if self.stat_ivar is not None:
            stat_ivar = self.stat_ivar[index]
        else:
            stat_ivar = None

        if self.throughput_corrections is not None:
            tcorr = self.throughput_corrections[index]
        else:
            tcorr = None

        sky2 = SkyModel(self.wave, flux, ivar, mask, header=self.header, nrej=self.nrej,
                stat_ivar=stat_ivar, throughput_corrections=tcorr)

        sky2.dwave = self.dwave
        if self.dlsf is not None:
            sky2.dlsf = self.dlsf[index]

        return sky2


def subtract_sky(frame, skymodel, apply_throughput_correction_to_lines = True, apply_throughput_correction = False, zero_ivar=True) :
    """Subtract skymodel from frame, altering frame.flux, .ivar, and .mask

    Args:
        frame : desispec.Frame object
        skymodel : desispec.SkyModel object

    Option:
        apply_throughput_correction : if True, fit for an achromatic throughput
            correction.  This is to absorb variations of Focal Ratio Degradation
            with fiber flexure.  This applies the residual throughput corrections
            on top of the model throughput corrections already included in the sky
            model.

        zero_ivar : if True , set ivar=0 for masked pixels
    """
    assert frame.nspec == skymodel.nspec
    assert frame.nwave == skymodel.nwave

    log=get_logger()
    log.info("starting with apply_throughput_correction_to_lines = {} apply_throughput_correction = {} and zero_ivar = {}".format(apply_throughput_correction_to_lines,apply_throughput_correction, zero_ivar))

    # Set fibermask flagged spectra to have 0 flux and variance
    frame = get_fiberbitmasked_frame(frame,bitmask='sky',ivar_framemask=zero_ivar)

    # check same wavelength, die if not the case
    if not np.allclose(frame.wave, skymodel.wave):
        message = "frame and sky not on same wavelength grid"
        log.error(message)
        raise ValueError(message)


    skymodel_flux = skymodel.flux.copy() # always use a copy to avoid overwriting model

    if skymodel.throughput_corrections is not None :
        # a multiplicative factor + background around
        # each of the bright sky lines has been fit.
        # here we apply this correction to the emission lines only or to the whole
        # sky spectrum

        if apply_throughput_correction  :

            skymodel_flux *= skymodel.throughput_corrections[:,None]

        elif apply_throughput_correction_to_lines :

            if frame.meta is not None and "CAMERA" in frame.meta and frame.meta["CAMERA"] is not None and frame.meta["CAMERA"][0].lower() == "b" :
                log.info("Do not apply throughput correction to sky lines for blue cameras")
            else :
                in_cont_boolean = np.repeat(True,skymodel.wave.shape)
                for line in get_sky_lines() :
                    # ignore b-arm sky lines, because there is really only one significant line
                    # at 5579A. without other lines, we could be erase a target emission line.
                    # (this is a duplication of test on the camera ID above)
                    if line < 5700 : continue
                    in_cont_boolean &= np.abs(skymodel.wave-line)>2. # A
                in_cont = np.where(in_cont_boolean)[0]

                if in_cont.size > 0 :
                    # apply this correction to the sky lines only
                    for fiber in range(frame.flux.shape[0]) :
                        # estimate and subtract continuum for this fiber specifically
                        cont = np.interp(skymodel.wave,skymodel.wave[in_cont],skymodel.flux[fiber][in_cont])
                        skylines = skymodel.flux[fiber] - cont
                        skylines[skylines<0]=0
                        # apply correction to the sky lines only
                        skymodel_flux[fiber] += (skymodel.throughput_corrections[fiber]-1.)*skylines
                else :
                    log.warning("Could not determine sky continuum, do not apply throughput correction on sky lines")

    frame.flux -= skymodel_flux
    frame.ivar = util.combine_ivar(frame.ivar, skymodel.ivar)
    frame.mask |= skymodel.mask

    log.info("done")

def get_sky_lines() :
    # it's more robust to have a hardcoded set of sky lines here
    # these are most of the dark sky lines at KPNO (faint lines are discarded)
    # wavelength are A, in vacuum, (obviously in earth frame)
    return np.array([
        4359.55,5199.27,5462.38,5578.85,5891.47,5897.51,5917.04,5934.63,
        5955.11,5978.80,6172.39,6204.48,6223.56,6237.36,6259.62,6266.96,
        6289.21,6302.06,6308.70,6323.09,6331.63,6350.26,6358.10,6365.57,
        6388.31,6467.83,6472.63,6500.48,6506.85,6524.31,6534.88,6545.95,
        6555.44,6564.59,6570.69,6579.14,6606.02,6831.19,6836.16,6843.66,
        6865.78,6873.06,6883.14,6891.22,6902.79,6914.52,6925.11,6941.43,
        6950.99,6971.84,6980.36,7005.76,7013.36,7050.01,7242.08,7247.18,
        7255.13,7278.23,7286.57,7298.03,7305.76,7318.31,7331.26,7342.93,
        7360.71,7371.44,7394.23,7403.92,7431.82,7440.56,7468.66,7473.57,
        7475.80,7481.74,7485.52,7495.74,7526.02,7532.80,7559.59,7573.90,
        7588.21,7600.49,7620.20,7630.86,7655.22,7664.52,7694.11,7701.75,
        7714.64,7718.99,7725.76,7728.26,7737.35,7752.67,7759.19,7762.17,
        7775.53,7782.58,7794.17,7796.27,7810.62,7823.65,7843.43,7851.81,
        7855.53,7860.06,7862.85,7870.17,7872.94,7880.89,7883.89,7892.04,
        7915.75,7921.90,7923.29,7933.51,7947.89,7951.41,7966.84,7970.48,
        7980.89,7982.00,7995.58,8016.34,8022.30,8028.03,8030.18,8054.15,
        8064.43,8087.35,8096.04,8104.77,8141.35,8149.00,8190.40,8197.67,
        8280.73,8283.97,8290.79,8298.52,8301.21,8305.09,8313.02,8320.68,
        8346.77,8352.10,8355.21,8363.19,8367.11,8384.59,8401.53,8417.58,
        8432.54,8436.62,8448.87,8454.61,8467.72,8477.00,8495.82,8507.19,
        8523.03,8541.04,8551.10,8590.62,8599.43,8620.40,8623.91,8627.34,
        8630.96,8634.50,8638.26,8642.53,8643.38,8649.38,8652.98,8657.70,
        8662.46,8667.36,8672.49,8677.88,8683.25,8689.10,8695.03,8702.18,
        8709.64,8762.40,8763.78,8770.00,8780.36,8793.54,8812.37,8829.35,
        8831.74,8835.98,8838.89,8847.90,8852.27,8864.96,8870.02,8888.31,
        8898.80,8905.60,8911.52,8922.12,8930.34,8945.87,8960.56,8973.21,
        8984.22,8990.85,8994.99,9003.85,9023.44,9030.80,9040.55,9052.03,
        9067.36,9095.13,9105.32,9154.58,9163.70,9219.04,9227.34,9265.26,
        9288.76,9296.23,9309.49,9315.79,9320.36,9326.25,9333.54,9340.43,
        9351.84,9364.59,9370.52,9378.34,9385.63,9399.67,9404.72,9422.34,
        9425.23,9442.27,9450.81,9461.13,9479.60,9486.22,9493.22,9505.47,
        9522.06,9532.66,9539.71,9555.14,9562.95,9570.04,9610.34,9623.30,
        9656.01,9661.88,9671.38,9676.93,9684.39,9693.15,9701.98,9714.38,
        9722.52,9737.51,9740.80,9748.49,9793.49,9799.16,9802.55,9810.34,
        9814.71,9819.99])


def calculate_throughput_corrections(frame,skymodel):
    """
    Calculate the throughput corrections for each fiber based on the skymodel.

    Args:
        frame (Frame object): frame containing the data that may need to be corrected
        skymodel (SkyModel object): skymodel object that contains the information about the sky for the given exposure/frame

    Output:
        corrections (1D array):  1D array where the index corresponds to the fiber % 500 and the values are the multiplicative corrections that would
                             be applied to the fluxes in frame.flux to correct them based on the input skymodel
    """
    # need to fit for a multiplicative factor of the sky model
    # before subtraction
    # we are going to use a set of bright sky lines,
    # and fit a multiplicative factor + background around
    # each of them individually, and then combine the results
    # with outlier rejection in case a source emission line
    # coincides with one of the sky lines.

    skyline=get_sky_lines()

    # half width of wavelength region around each sky line
    # larger values give a better statistical precision
    # but also a larger sensitivity to source features
    # best solution on one dark night exposure obtained with
    # a half width of 4A.
    hw=4#A
    tivar=frame.ivar
    if frame.mask is not None :
        tivar *= (frame.mask==0)
        tivar *= (skymodel.ivar>0)

    # we precompute the quantities needed to fit each sky line + continuum
    # the sky "line profile" is the actual sky model
    # and we consider an additive constant
    sw,swf,sws,sws2,swsf=[],[],[],[],[]
    for line in skyline :
        if line<=frame.wave[0] or line>=frame.wave[-1] : continue
        ii=np.where((frame.wave>=line-hw)&(frame.wave<=line+hw))[0]
        if ii.size<2 : continue
        sw.append(np.sum(tivar[:,ii],axis=1))
        swf.append(np.sum(tivar[:,ii]*frame.flux[:,ii],axis=1))
        swsf.append(np.sum(tivar[:,ii]*frame.flux[:,ii]*skymodel.flux[:,ii],axis=1))
        sws.append(np.sum(tivar[:,ii]*skymodel.flux[:,ii],axis=1))
        sws2.append(np.sum(tivar[:,ii]*skymodel.flux[:,ii]**2,axis=1))

    log=get_logger()
    nlines=len(sw)
    corrections = np.ones(frame.flux.shape[0]).astype('f8')
    for fiber in range(frame.flux.shape[0]) :
        # we solve the 2x2 linear system for each fiber and sky line
        # and save the results for each fiber
        coef=[] # list of scale values
        var=[] # list of variance on scale values
        for line in range(nlines) :
            if sw[line][fiber]<=0 : continue
            A=np.array([[sw[line][fiber],sws[line][fiber]],[sws[line][fiber],sws2[line][fiber]]])
            B=np.array([swf[line][fiber],swsf[line][fiber]])
            try :
                Ai=np.linalg.inv(A)
                X=Ai.dot(B)
                coef.append(X[1]) # the scale coef (marginalized over cst background)
                var.append(Ai[1,1])
            except :
                pass

        if len(coef)==0 :
            log.warning("cannot corr. throughput. for fiber %d"%fiber)
            continue

        coef=np.array(coef)
        var=np.array(var)
        ivar=(var>0)/(var+(var==0)+0.005**2)
        ivar_for_outliers=(var>0)/(var+(var==0)+0.02**2)

        # loop for outlier rejection
        failed=False
        for loop in range(50) :
            a=np.sum(ivar)
            if a <= 0 :
                log.warning("cannot corr. throughput. ivar=0 everywhere on sky lines for fiber %d"%fiber)
                failed=True
                break

            mcoef=np.sum(ivar*coef)/a
            mcoeferr=1/np.sqrt(a)

            nsig=3.
            chi2=ivar_for_outliers*(coef-mcoef)**2
            worst=np.argmax(chi2)
            if chi2[worst]>nsig**2*np.median(chi2[chi2>0]) : # with rough scaling of errors
                #log.debug("discard a bad measurement for fiber %d"%(fiber))
                ivar[worst]=0
                ivar_for_outliers[worst]=0
            else :
                break

        if failed :
            continue


        log.info("fiber #%03d throughput corr = %5.4f +- %5.4f (mean fiber flux=%f)"%(fiber,mcoef,mcoeferr,np.median(frame.flux[fiber])))
        if mcoeferr>0.1 :
            log.warning("throughput corr error = %5.4f > 0.1 is too large for fiber %d, do not apply correction"%(mcoeferr,fiber))
        else :
            corrections[fiber] = mcoef

    return corrections


def qa_skysub(param, frame, skymodel, quick_look=False):
    """Calculate QA on SkySubtraction

    Note: Pixels rejected in generating the SkyModel (as above), are
    not rejected in the stats calculated here.  Would need to carry
    along current_ivar to do so.

    Args:
        param : dict of QA parameters : see qa_frame.init_skysub for example
        frame : desispec.Frame object;  Should have been flat fielded
        skymodel : desispec.SkyModel object
        quick_look : bool, optional
          If True, do QuickLook specific QA (or avoid some)
    Returns:
        qadict: dict of QA outputs
          Need to record simple Python objects for yaml (str, float, int)
    """
    from desispec.qa import qalib
    import copy

    log=get_logger()

    #- QAs
    #- first subtract sky to get the sky subtracted frame. This is only for QA. Pipeline does it separately.
    tempframe=copy.deepcopy(frame) #- make a copy so as to propagate frame unaffected so that downstream pipeline uses it.
    subtract_sky(tempframe,skymodel) #- Note: sky subtract is done to get residuals. As part of pipeline it is done in fluxcalib stage

    # Sky residuals first
    qadict = qalib.sky_resid(param, tempframe, skymodel, quick_look=quick_look)

    # Sky continuum
    if not quick_look:  # Sky continuum is measured after flat fielding in QuickLook
        channel = frame.meta['CAMERA'][0]
        wrange1, wrange2 = param[channel.upper()+'_CONT']
        skyfiber, contfiberlow, contfiberhigh, meancontfiber, skycont = qalib.sky_continuum(frame,wrange1,wrange2)
        qadict["SKYFIBERID"] = skyfiber.tolist()
        qadict["SKYCONT"] = skycont
        qadict["SKYCONT_FIBER"] = meancontfiber

    if quick_look:  # The following can be a *large* dict
        qadict_snr = qalib.SignalVsNoise(tempframe,param)
        qadict.update(qadict_snr)

    return qadict
