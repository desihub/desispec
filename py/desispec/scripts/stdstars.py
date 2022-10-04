

"""
Get the normalized best template to do flux calibration.
"""

#- TODO: refactor algorithmic code into a separate module/function

import argparse
import sys

import numpy as np
from astropy.io import fits
from astropy import units
from astropy.table import Table
import astropy.coordinates as acoo

import desispec.fluxcalibration
from desispec import io
from desispec.fluxcalibration import match_templates,normalize_templates,isStdStar
from desispec.interpolation import resample_flux
from desiutil.log import get_logger
from desispec.parallel import default_nproc
from desispec.io.filters import load_legacy_survey_filter, load_gaia_filter
from desiutil.dust import dust_transmission,extinction_total_to_selective_ratio, SFDMap, gaia_extinction
from desispec.fiberbitmasking import get_fiberbitmasked_frame

from desispec.fiberflat import apply_fiberflat
from desispec.sky import subtract_sky

def parse(options=None):
    parser = argparse.ArgumentParser(description="Fit of standard star spectra in frames.")
    parser.add_argument('--frames', type = str, default = None, required=True, nargs='*',
                        help = 'list of path to DESI frame fits files (needs to be same exposure, spectro)')
    parser.add_argument('--skymodels', type = str, default = None, required=True, nargs='*',
                        help = 'list of path to DESI sky model fits files (needs to be same exposure, spectro)')
    parser.add_argument('--fiberflats', type = str, default = None, required=True, nargs='*',
                        help = 'list of path to DESI fiberflats fits files (needs to be same exposure, spectro)')
    parser.add_argument('--starmodels', type = str, help = 'path of spectro-photometric stellar spectra fits')
    parser.add_argument('-o','--outfile', type = str, help = 'output file for normalized stdstar model flux')
    parser.add_argument('--ncpu', type = int, default = default_nproc, required = False, help = 'use ncpu for multiprocessing')
    parser.add_argument('--delta-color', type = float, default = 0.2, required = False, help = 'max delta-color for the selection of standard stars (on top of meas. errors)')
    parser.add_argument('--min-blue-snr', type = float, default = 4.0, required = False,
            help = 'Minimum required S/N in blue CCD to be used')
    parser.add_argument('--color', type = str, default = None, choices=['G-R', 'R-Z', 'GAIA-BP-RP','GAIA-G-RP'], required = False, help = 'color for selection of standard stars')
    parser.add_argument('--z-max', type = float, default = 0.008, required = False, help = 'max peculiar velocity (blue/red)shift range')
    parser.add_argument('--z-res', type = float, default = 0.00002, required = False, help = 'dz grid resolution')
    parser.add_argument('--template-error', type = float, default = 0.1, required = False, help = 'fractional template error used in chi2 computation (about 0.1 for BOSS b1)')
    parser.add_argument('--maxstdstars', type=int, default=30, \
            help='Maximum number of stdstars to include')
    parser.add_argument('--std-targetids', type=int, default=None,
                         nargs='*',
                         help='List of TARGETIDs of standards overriding the targeting info')
    parser.add_argument('--mpi', action='store_true', help='Use MPI')
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU, if available')

    log = get_logger()

    args = parser.parse_args(options)

    if options is None:
        cmd = ' '.join(sys.argv)
    else:
        cmd = 'desi_fit_stdstars ' + ' '.join(options)

    log.info('RUNNING {}'.format(cmd))

    return args

def safe_read_key(header,key) :
    value = None
    try :
        value=header[key]
    except KeyError :
        value = None
        pass
    if value is None : # second try
        value=header[key.ljust(8).upper()]
    return value

def get_gaia_ab_correction():
    """
Get the dictionary with corrections from AB magnitudes to
Vega magnitudes (as the official gaia catalog is in vega)
"""
    vega_zpt = dict(G=25.6914396869,
                    BP=25.3488107670,
                    RP=24.7626744847)
    ab_zpt=dict(G=25.7915509947,
                BP=25.3861560855,
                RP=25.1161664528)
    # revised dr2 zpts from https://www.cosmos.esa.int/web/gaia/iow_20180316
    ret = {}
    for k in vega_zpt.keys():
        ret['GAIA-'+k] = vega_zpt[k] - ab_zpt[k]
    # these corrections need to be added to convert
    # the simulated ab into vega
    return ret

def get_magnitude(stdwave, model, model_filters, cur_filt):
    """ Obtain magnitude for a filter taking into
account the ab/vega correction if needed.
Wwe assume the flux is in units of 1e-17 erg/s/cm^2/A
    """
    fluxunits = 1e-17 * units.erg / units.s / units.cm**2 / units.Angstrom

    # AB/Vega correction
    if cur_filt[:5] == 'GAIA-':
        corr = get_gaia_ab_correction()[cur_filt]
    else:
        corr = 0
    if not(cur_filt in model_filters):
        raise Exception(('Filter {} is not present in models').format(cur_filt))
    # see https://github.com/desihub/speclite/issues/34
    # to explain copy()
    retmag = model_filters[cur_filt].get_ab_magnitude(model * fluxunits, stdwave.copy())+ corr
    return retmag

def main(args=None, comm=None) :
    """ finds the best models of all standard stars in the frame
    and normlize the model flux. Output is written to a file and will be called for calibration.
    """

    if not isinstance(args, argparse.Namespace):
        args = parse(args)

    log = get_logger()

    log.info("mag delta %s = %f (for the pre-selection of stellar models)"%(args.color,args.delta_color))

    if args.mpi or comm is not None:
        from mpi4py import MPI
        if comm is None:
            comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        if rank == 0:
            log.info('mpi parallelizing with {} ranks'.format(size))
    else:
        comm = None
        rank = 0
        size = 1

    # disable multiprocess by forcing ncpu = 1 when using MPI
    if comm is not None:
        ncpu = 1
        if rank == 0:
            log.info('disabling multiprocess (forcing ncpu = 1)')
    else:
        ncpu = args.ncpu

    if ncpu > 1:
        if rank == 0:
            log.info('multiprocess parallelizing with {} processes'.format(ncpu))

    if not args.use_gpu and desispec.fluxcalibration.use_gpu:
        # Opt-out of GPU usage
        desispec.fluxcalibration.use_gpu = False
        if rank == 0:
            log.info('ignoring GPU')
    elif desispec.fluxcalibration.use_gpu:
        # Nothing to do here, GPU is used by default if available
        if rank == 0:
            log.info('using GPU')
    else:
        if rank == 0:
            log.info('GPU not available')

    # READ DATA
    ############################################
    # First loop through and group by exposure and spectrograph
    frames_by_expid = {}
    rows = list()
    for filename in args.frames :
        log.info("reading %s"%filename)
        frame=io.read_frame(filename)
        night = safe_read_key(frame.meta,"NIGHT")
        expid = safe_read_key(frame.meta,"EXPID")
        camera = safe_read_key(frame.meta,"CAMERA").strip().lower()
        rows.append( (night, expid, camera) )
        spec = camera[1]
        uniq_key = (expid,spec)

        # To save memory, trim to just stdstars as each frame is read;
        # more quality cuts will be applied later
        if args.std_targetids is None:
            keep = isStdStar(frame.fibermap)
        else:
            keep = np.isin(frame.fibermap['TARGETID'], args.std_targetids)

        frame = frame[keep]

        if uniq_key in frames_by_expid.keys():
            frames_by_expid[uniq_key][camera] = frame
        else:
            frames_by_expid[uniq_key] = {camera: frame}

    input_frames_table = Table(rows=rows, names=('NIGHT', 'EXPID', 'CAMERA'))

    frames={}
    flats={}
    skies={}

    spectrograph=None
    starfibers=None
    starindices=None
    fibermap=None
    # For each unique expid,spec pair, get the logical OR of the FIBERSTATUS for all
    # cameras and then proceed with extracting the frame information
    # once we modify the fibermap FIBERSTATUS
    for (expid,spec),camdict in frames_by_expid.items():

        fiberstatus = None
        for frame in camdict.values():
            if fiberstatus is None:
                fiberstatus = frame.fibermap['FIBERSTATUS'].data.copy()
            else:
                fiberstatus |= frame.fibermap['FIBERSTATUS']

        for camera,frame in camdict.items():
            frame.fibermap['FIBERSTATUS'] |= fiberstatus
            # Set fibermask flagged spectra to have 0 flux and variance
            frame = get_fiberbitmasked_frame(frame,bitmask='stdstars',ivar_framemask=True)
            frame_fibermap = frame.fibermap

            # frame was filtered to just stdstars upon reading, so initial list of starindices is full range
            frame_starindices = np.arange(len(frame.fibermap), dtype=int)

            #- Confirm that all fluxes have entries but trust targeting bits
            #- to get basic magnitude range correct
            keep_legacy = np.ones(len(frame_starindices), dtype=bool)

            for colname in ['FLUX_G', 'FLUX_R', 'FLUX_Z']:  #- and W1 and W2?
                keep_legacy &= frame_fibermap[colname][frame_starindices] > 10**((22.5-30)/2.5)
                keep_legacy &= frame_fibermap[colname][frame_starindices] < 10**((22.5-0)/2.5)
            keep_gaia = np.ones(len(frame_starindices), dtype=bool)

            for colname in ['G', 'BP', 'RP']:  #- and W1 and W2?
                keep_gaia &= frame_fibermap['GAIA_PHOT_'+colname+'_MEAN_MAG'][frame_starindices] > 10
                keep_gaia &= frame_fibermap['GAIA_PHOT_'+colname+'_MEAN_MAG'][frame_starindices] < 20
            n_legacy_std = keep_legacy.sum()
            n_gaia_std = keep_gaia.sum()
            keep = keep_legacy | keep_gaia
            # accept both types of standards for the time being

            # keep the indices for gaia/legacy subsets
            is_gaia_std = keep_gaia[keep]
            is_legacy_std = keep_legacy[keep]

            frame_starindices = frame_starindices[keep]

            if spectrograph is None :
                spectrograph = frame.spectrograph
                fibermap = frame_fibermap
                starindices=frame_starindices
                starfibers=np.asarray(fibermap["FIBER"][starindices])

            elif spectrograph != frame.spectrograph :
                log.error("incompatible spectrographs {} != {}".format(spectrograph,frame.spectrograph))
                raise ValueError("incompatible spectrographs {} != {}".format(spectrograph,frame.spectrograph))
            elif starindices.size != frame_starindices.size or np.sum(starindices!=frame_starindices)>0 :
                log.error("incompatible fibermap")
                raise ValueError("incompatible fibermap")

            if not camera in frames :
                frames[camera]=[]

            frames[camera].append(frame)

    # possibly cleanup memory
    del frames_by_expid

    # wait for all ranks to finish reading and trimming before reading more
    if comm is not None:
        comm.barrier()

    #- Read sky models and fiberflats, also trimming to starfibers kept for frames
    for filename in args.skymodels :
        log.info("reading %s"%filename)
        sky=io.read_sky(filename)
        camera=safe_read_key(sky.header,"CAMERA").strip().lower()
        if not camera in skies :
            skies[camera]=[]
        skies[camera].append(sky[starfibers%500])

    for filename in args.fiberflats :
        log.info("reading %s"%filename)
        flat=io.read_fiberflat(filename)
        camera=safe_read_key(flat.header,"CAMERA").strip().lower()

        # NEED TO ADD MORE CHECKS
        if camera in flats:
            log.warning("cannot handle several flats of same camera (%s), will use only the first one"%camera)
            #raise ValueError("cannot handle several flats of same camera (%s)"%camera)
        else :
            flats[camera]=flat[starfibers%500]

    # if color is not specified we decide on the fly
    color = args.color
    if color is not None:
        if color[:4] == 'GAIA':
            legacy_color  = False
            gaia_color = True
        else:
            legacy_color = True
            gaia_color = False
        if n_legacy_std == 0 and legacy_color:
            raise Exception('Specified Legacy survey color, but no legacy standards')
        if n_gaia_std == 0 and gaia_color:
            raise Exception('Specified gaia color, but no gaia stds')

    if starindices.size == 0:
        log.error("no STD star found in fibermap")
        raise ValueError("no STD star found in fibermap")
    log.info("found %d STD stars" % starindices.size)

    if n_legacy_std == 0:
        gaia_std = True
        if color is None:
            color = 'GAIA-BP-RP'
    else:
        gaia_std = False
        if color is None:
            color='G-R'
        if n_gaia_std > 0:
            log.info('Gaia standards found but not used')

    if gaia_std:
        # The name of the reference filter to which we normalize the flux
        ref_mag_name = 'GAIA-G'
        color_band1, color_band2  = ['GAIA-'+ _ for _ in color[5:].split('-')]
        log.info("Using Gaia standards with color {} and normalizing to {}".format(color, ref_mag_name))
        # select appropriate subset of standards
        starindices = np.where(is_gaia_std)[0]
        starfibers = starfibers[is_gaia_std]
    else:
        ref_mag_name = 'R'
        color_band1, color_band2  = color.split('-')
        log.info("Using Legacy standards with color {} and normalizing to {}".format(color, ref_mag_name))
        # select appropriate subset of standards
        starindices = np.where(is_legacy_std)[0]
        starfibers = starfibers[is_legacy_std]


    # excessive check but just in case
    if not color in ['G-R', 'R-Z', 'GAIA-BP-RP', 'GAIA-G-RP']:
        raise ValueError('Unknown color {}'.format(color))

    # log.warning("Not using flux errors for Standard Star fits!")

    # DIVIDE FLAT AND SUBTRACT SKY , TRIM DATA
    ############################################
    # since poping dict, we need to copy keys to iterate over to avoid
    # RuntimeError due to changing dict
    frame_cams = list(frames.keys())
    for cam in frame_cams:

        if not cam in skies:
            log.warning("Missing sky for %s"%cam)
            frames.pop(cam)
            continue
        if not cam in flats:
            log.warning("Missing flat for %s"%cam)
            frames.pop(cam)
            continue

        flat = flats[cam][starindices]

        for i in range(len(frames[cam])):
            frame = frames[cam][i][starindices]
            sky = skies[cam][i][starindices]

            #- don't use masked or ivar=0 data
            frame.ivar *= (frame.mask == 0)
            frame.ivar *= (sky.ivar != 0)
            frame.ivar *= (sky.mask == 0)
            frame.ivar *= (flat.ivar != 0)
            frame.ivar *= (flat.mask == 0)
            frame.flux *= (frame.ivar > 0) # just for clean plots

            apply_fiberflat(frame, flat)
            subtract_sky(frame, sky)

            #- keep newly flat-fielded sky-subtracted frame
            frames[cam][i] = frame

        nframes=len(frames[cam])
        if nframes>1 :
            # optimal weights for the coaddition = ivar*throughput, not directly ivar,
            # we estimate the relative throughput with median fluxes at this stage
            medflux=np.zeros(nframes)
            for i,frame in enumerate(frames[cam]) :
                if np.sum(frame.ivar>0) == 0 :
                    log.error("ivar=0 for all std star spectra in frame {}-{:08d}".format(cam,frame.meta["EXPID"]))
                else :
                    medflux[i] = np.median(frame.flux[frame.ivar>0])
            log.debug("medflux = {}".format(medflux))
            medflux *= (medflux>0)
            if np.sum(medflux>0)==0 :
               log.error("mean median flux = 0, for all stars in fibers {}".format(list(frames[cam][0].fibermap["FIBER"][starindices])))
               sys.exit(12)
            mmedflux = np.mean(medflux[medflux>0])
            weights=medflux/mmedflux
            log.info("coadding {} exposures in cam {}, w={}".format(nframes,cam,weights))

            sw=np.zeros(frames[cam][0].flux.shape)
            swf=np.zeros(frames[cam][0].flux.shape)
            swr=np.zeros(frames[cam][0].resolution_data.shape)

            for i,frame in enumerate(frames[cam]) :
                sw  += weights[i]*frame.ivar
                swf += weights[i]*frame.ivar*frame.flux
                swr += weights[i]*frame.ivar[:,None,:]*frame.resolution_data
            coadded_frame = frames[cam][0]
            coadded_frame.ivar = sw
            coadded_frame.flux = swf/(sw+(sw==0))
            coadded_frame.resolution_data = swr/((sw+(sw==0))[:,None,:])
            frames[cam] = [ coadded_frame ]

    # We're done with skies and flats dict; remove them to possibly save memory
    del skies
    del flats

    # CHECK S/N
    ############################################
    # for each band in 'brz', record quadratic sum of median S/N across wavelength
    snr=dict()
    for band in ['b','r','z'] :
        snr[band]=np.zeros(starindices.size)
    for cam in frames :
        band=cam[0].lower()
        for frame in frames[cam] :
            msnr = np.median( frame.flux * np.sqrt( frame.ivar ) / np.sqrt(np.gradient(frame.wave)) , axis=1 ) # median SNR per sqrt(A.)
            msnr *= (msnr>0)
            snr[band] = np.sqrt( snr[band]**2 + msnr**2 )
    log.info("SNR(B) = {}".format(snr['b']))

    # Sort and trim by blue S/N
    indices=np.argsort(snr['b'])[::-1][:args.maxstdstars]
    validstars = np.where(snr['b'][indices]>args.min_blue_snr)[0]

    #- TODO: later we filter on models based upon color, thus throwing
    #- away very blue stars for which we don't have good models.

    log.info("Number of stars with median stacked blue S/N > {} /sqrt(A) = {}".format(args.min_blue_snr,validstars.size))
    if validstars.size == 0 :
        log.error(f"No valid star for sp{spectrograph}")
        sys.exit(12)

    validstars = indices[validstars]

    for band in ['b','r','z'] :
        snr[band]=snr[band][validstars]

    log.info("BLUE SNR of selected stars={}".format(snr['b']))

    for cam in frames :
        for i in range(len(frames[cam])) :
            frames[cam][i] = frames[cam][i][validstars]

    starindices = starindices[validstars]
    starfibers  = starfibers[validstars]
    nstars = starindices.size
    fibermap = Table(fibermap[starindices])

    # MASK OUT THROUGHPUT DIP REGION
    ############################################
    mask_throughput_dip_region = True
    if mask_throughput_dip_region :
        wmin=4300.
        wmax=4500.
        log.warning("Masking out the wavelength region [{},{}]A in the standard star fit".format(wmin,wmax))
    for cam in frames :
        for frame in frames[cam] :
            ii=np.where( (frame.wave>=wmin)&(frame.wave<=wmax) )[0]
            if ii.size>0 :
                frame.ivar[:,ii] = 0

    # READ MODELS
    ############################################

    log.info("reading star models in %s"%args.starmodels)
    stdwave,stdflux,templateid,teff,logg,feh=io.read_stdstar_templates(args.starmodels)

    # COMPUTE MAGS OF MODELS FOR EACH STD STAR MAG
    ############################################

    #- Support older fibermaps
    if 'PHOTSYS' not in fibermap.colnames:
        log.warning('Old fibermap format; using defaults for missing columns')
        log.warning("    PHOTSYS = 'S'")
        log.warning("    EBV = 0.0")
        fibermap['PHOTSYS'] = 'S'
        fibermap['EBV'] = 0.0

    if not np.in1d(np.unique(fibermap['PHOTSYS']),['','N','S','G']).all():
        log.error('Unknown PHOTSYS found')
        raise Exception('Unknown PHOTSYS found')
    # Fetching Filter curves
    model_filters = dict()
    for band in ["G","R","Z"] :
        for photsys in np.unique(fibermap['PHOTSYS']) :
            if photsys in ['N','S']:
                model_filters[band+photsys] = load_legacy_survey_filter(band=band,photsys=photsys)
    if len(model_filters) == 0:
        log.info('No Legacy survey photometry identified in fibermap')

    # I will always load gaia data even if we are fitting LS standards only
    for band in ["G", "BP", "RP"] :
        model_filters["GAIA-" + band] = load_gaia_filter(band=band, dr=2)

    # Compute model mags on rank 0 and bcast result to other ranks
    # This sidesteps an OOM event on Cori Haswell with "-c 2"
    model_mags = None
    if rank == 0:
        log.info("computing model mags for %s"%sorted(model_filters.keys()))
        model_mags = dict()
        for filter_name in model_filters.keys():
            model_mags[filter_name] = get_magnitude(stdwave, stdflux, model_filters, filter_name)
        log.info("done computing model mags")

    if comm is not None:
        model_mags = comm.bcast(model_mags, root=0)

    # LOOP ON STARS TO FIND BEST MODEL
    ############################################
    star_mags = dict()
    star_unextincted_mags = dict()

    if gaia_std and (fibermap['EBV']==0).all():
        log.info("Using E(B-V) from SFD rather than FIBERMAP")
        # when doing gaia standards, on old tiles the
        # EBV is not set so we fetch from SFD (in original SFD scaling)
        ebv = SFDMap(scaling=1).ebv(acoo.SkyCoord(
            ra = fibermap['TARGET_RA'] * units.deg,
            dec = fibermap['TARGET_DEC'] * units.deg))
    else:
        ebv = fibermap['EBV']

    photometric_systems = np.unique(fibermap['PHOTSYS'])
    if not gaia_std:
        for band in ['G', 'R', 'Z']:
            star_mags[band] = 22.5 - 2.5 * np.log10(fibermap['FLUX_'+band])
            star_unextincted_mags[band] = np.zeros(star_mags[band].shape)
            for photsys in  photometric_systems :
                r_band = extinction_total_to_selective_ratio(band , photsys) # dimensionless
                # r_band = a_band / E(B-V)
                # E(B-V) is a difference of magnitudes (dimensionless)
                # a_band = -2.5*log10(effective dust transmission) , dimensionless
                # effective dust transmission =
                #                  integral( SED(lambda) * filter_transmission(lambda,band) * dust_transmission(lambda,E(B-V)) dlamdba)
                #                / integral( SED(lambda) * filter_transmission(lambda,band) dlamdba)
                selection = (fibermap['PHOTSYS'] == photsys)
                a_band = r_band * ebv[selection]  # dimensionless
                star_unextincted_mags[band][selection] = 22.5 - 2.5 * np.log10(fibermap['FLUX_'+band][selection]) - a_band

    for band in ['G','BP','RP']:
        star_mags['GAIA-'+band] = fibermap['GAIA_PHOT_'+band+'_MEAN_MAG']

    for band, extval in gaia_extinction(star_mags['GAIA-G'],
                                        star_mags['GAIA-BP'],
                                        star_mags['GAIA-RP'], ebv).items():
        star_unextincted_mags['GAIA-'+band] = star_mags['GAIA-'+band] - extval


    star_colors = dict()
    star_unextincted_colors = dict()

    # compute the colors and define the unextincted colors
    # the unextincted colors are filled later
    if not gaia_std:
        for c1,c2 in ['GR', 'RZ']:
            star_colors[c1 + '-' + c2] = star_mags[c1] - star_mags[c2]
            star_unextincted_colors[c1 + '-' + c2] = (
                star_unextincted_mags[c1] - star_unextincted_mags[c2])
    for c1,c2 in [('BP','RP'), ('G','RP')]:
        star_colors['GAIA-' + c1 + '-' + c2] = (
            star_mags['GAIA-' + c1] - star_mags['GAIA-' + c2])
        star_unextincted_colors['GAIA-' + c1 + '-' + c2] = (
            star_unextincted_mags['GAIA-' + c1] -
            star_unextincted_mags['GAIA-' + c2])

    local_comm, head_comm = None, None
    if comm is not None:
        # All ranks in local_comm work on the same stars
        local_comm = comm.Split(rank % nstars, rank)
        # The color 1 in head_comm contains all ranks that are have rank 0 in local_comm
        head_comm = comm.Split(rank < nstars, rank)

    #- Allocate arrays only needed by local_comm.rank == 0 ranks
    if local_comm is None or local_comm.rank == 0:
        linear_coefficients = np.zeros((nstars,stdflux.shape[0]))
        chi2dof = np.zeros((nstars))
        redshift = np.zeros((nstars))
        normflux = np.zeros((nstars, stdwave.size))
        fitted_model_colors = np.zeros(nstars)
        model = np.zeros(stdwave.size)
    else:
        linear_coefficients = None
        chi2dof = None
        redshift = None
        normflux = None
        fitted_model_colors = None
        model = None

    for star in range(rank % nstars, nstars, size):

        log.info("rank %d: finding best model for observed star #%d"%(rank, star))

        # np.array of wave,flux,ivar,resol
        wave = {}
        flux = {}
        ivar = {}
        resolution_data = {}
        for camera in frames :
            for i,frame in enumerate(frames[camera]) :
                identifier="%s-%d"%(camera,i)
                wave[identifier]=frame.wave
                flux[identifier]=frame.flux[star]
                ivar[identifier]=frame.ivar[star]
                resolution_data[identifier]=frame.resolution_data[star]

        # preselect models based on magnitudes
        photsys=fibermap['PHOTSYS'][star]

        if gaia_std:
            model_colors = model_mags[color_band1] - model_mags[color_band2]
        else:
            model_colors = model_mags[color_band1 + photsys] - model_mags[color_band2 + photsys]

        color_diff = model_colors - star_unextincted_colors[color][star]
        selection = np.abs(color_diff) < args.delta_color
        if np.sum(selection) == 0 :
            log.warning("no model in the selected color range for this star")
            continue


        # smallest cube in parameter space including this selection (needed for interpolation)
        new_selection = (teff>=np.min(teff[selection]))&(teff<=np.max(teff[selection]))
        new_selection &= (logg>=np.min(logg[selection]))&(logg<=np.max(logg[selection]))
        new_selection &= (feh>=np.min(feh[selection]))&(feh<=np.max(feh[selection]))
        selection = np.where(new_selection)[0]

        log.info("star#%d fiber #%d, %s = %f, number of pre-selected models = %d/%d"%(
            star, starfibers[star], color, star_unextincted_colors[color][star],
            selection.size, stdflux.shape[0]))

        # Match unextincted standard stars to data
        match_templates_result = match_templates(
            wave, flux, ivar, resolution_data,
            stdwave, stdflux[selection],
            teff[selection], logg[selection], feh[selection],
            ncpu=ncpu, z_max=args.z_max, z_res=args.z_res,
            template_error=args.template_error, comm=local_comm
            )

        # Only local rank 0 can perform the remaining work
        if local_comm is not None and local_comm.Get_rank() != 0:
            continue

        coefficients, redshift[star], chi2dof[star] = match_templates_result
        linear_coefficients[star,selection] = coefficients
        log.info('Star Fiber: {}; TEFF: {:.3f}; LOGG: {:.3f}; FEH: {:.3f}; Redshift: {:g}; Chisq/dof: {:.3f}'.format(
            starfibers[star],
            np.inner(teff,linear_coefficients[star]),
            np.inner(logg,linear_coefficients[star]),
            np.inner(feh,linear_coefficients[star]),
            redshift[star],
            chi2dof[star])
            )

        # Apply redshift to original spectrum at full resolution
        model *= 0.0   #- clear model from previous loop without re-allocating memory
        redshifted_stdwave = stdwave*(1+redshift[star])
        for i,c in enumerate(linear_coefficients[star]) :
            if c != 0 :
                model += c*np.interp(stdwave,redshifted_stdwave,stdflux[i])

        # Apply dust extinction to the model
        log.info("Applying MW dust extinction to star {} with EBV = {}".format(star,ebv[star]))
        model *= dust_transmission(stdwave, ebv[star])

        # Compute final color of dust-extincted model
        photsys=fibermap['PHOTSYS'][star]

        if not gaia_std:
            model_mag1, model_mag2 = [get_magnitude(stdwave, model, model_filters, _ + photsys) for _ in [color_band1, color_band2]]
        else:
            model_mag1, model_mag2 = [get_magnitude(stdwave, model, model_filters, _ ) for _ in [color_band1, color_band2]]

        if color_band1 == ref_mag_name:
            model_magr = model_mag1
        elif color_band2 == ref_mag_name:
            model_magr = model_mag2
        else:
            # if the reference magnitude is not among colours
            # I'm fetching it separately. This will happen when
            # colour is BP-RP and ref magnitude is G
            if gaia_std:
                model_magr = get_magnitude(stdwave, model, model_filters, ref_mag_name)
            else:
                model_magr = get_magnitude(stdwave, model, model_filters, ref_mag_name + photsys)
        fitted_model_colors[star] = model_mag1 - model_mag2

        #- TODO: move this back into normalize_templates, at the cost of
        #- recalculating a model magnitude?

        cur_refmag = star_mags[ref_mag_name][star]

        # Normalize the best model using reported magnitude
        scalefac=10**((model_magr - cur_refmag)/2.5)

        log.info('scaling {} mag {:.3f} to {:.3f} using scale {}'.format(ref_mag_name, model_magr, cur_refmag, scalefac))
        normflux[star] = model*scalefac

    if head_comm is not None and rank < nstars: # head_comm color is 1
        linear_coefficients = head_comm.reduce(linear_coefficients, op=MPI.SUM, root=0)
        redshift = head_comm.reduce(redshift, op=MPI.SUM, root=0)
        chi2dof = head_comm.reduce(chi2dof, op=MPI.SUM, root=0)
        fitted_model_colors = head_comm.reduce(fitted_model_colors, op=MPI.SUM, root=0)
        normflux = head_comm.reduce(normflux, op=MPI.SUM, root=0)

    # Check at least one star was fit. The check is peformed on rank 0 and
    # the result is bcast to other ranks so that all ranks exit together if
    # the check fails.
    atleastonestarfit = False
    if rank == 0:
        fitted_stars = np.where(chi2dof != 0)[0]
        atleastonestarfit = fitted_stars.size > 0
    if comm is not None:
        atleastonestarfit = comm.bcast(atleastonestarfit, root=0)
    if not atleastonestarfit:
        log.error("No star has been fit.")
        sys.exit(12)

    # Now write the normalized flux for all best models to a file
    if rank == 0:

        # get the fibermap from any input frame for the standard stars
        fibermap = Table(frame.fibermap[fitted_stars])
        assert np.all(fibermap['FIBER'] == starfibers[fitted_stars])

        # drop fibermap columns specific to exposures instead of targets
        for col in ['DELTA_X', 'DELTA_Y', 'EXPTIME', 'NUM_ITER',
                'FIBER_RA', 'FIBER_DEC', 'FIBER_X', 'FIBER_Y']:
            if col in fibermap.colnames:
                fibermap.remove_column(col)

        data={}
        data['TARGETID'] = fibermap['TARGETID']
        data['FIBER'] = fibermap['FIBER']
        data['LOGG']=linear_coefficients[fitted_stars,:].dot(logg)
        data['TEFF']= linear_coefficients[fitted_stars,:].dot(teff)
        data['FEH']= linear_coefficients[fitted_stars,:].dot(feh)
        data['CHI2DOF']=chi2dof[fitted_stars]
        data['REDSHIFT']=redshift[fitted_stars]
        data['COEFF']=linear_coefficients[fitted_stars,:]
        data['DATA_%s'%color]=star_colors[color][fitted_stars]
        data['MODEL_%s'%color]=fitted_model_colors[fitted_stars]
        data['BLUE_SNR'] = snr['b'][fitted_stars]
        data['RED_SNR']  = snr['r'][fitted_stars]
        data['NIR_SNR']  = snr['z'][fitted_stars]
        io.write_stdstar_models(args.outfile,normflux,stdwave,
                starfibers[fitted_stars],data,
                fibermap, input_frames_table)

    if comm is not None:
        comm.barrier()

    return 0
