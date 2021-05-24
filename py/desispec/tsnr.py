import os
import numpy as np
import time

import astropy.io.fits as fits
from astropy.table import Table
from astropy.convolution import convolve, Box1DKernel

import glob
import yaml
from pkg_resources import resource_filename

from scipy.optimize import minimize
from scipy.interpolate import RectBivariateSpline,interp1d
from scipy.signal import fftconvolve
from desiutil.log import get_logger
from desiutil.dust import dust_transmission

from desispec.io import findfile,read_frame,read_fiberflat,read_sky,read_flux_calibration,iotime
from desispec.io.spectra import Spectra
from desispec.calibfinder import findcalibfile
from astropy import constants as const

from specter.psf.gausshermite  import  GaussHermitePSF

class Config(object):
    def __init__(self, cpath):
        with open(cpath) as f:
            d = yaml.safe_load(f)

        for key in d:
            setattr(self, key, d[key])

class template_ensemble(object):
    '''
    Generate an ensemble of templates to sample tSNR for a range of points in
    (z, m, OII, etc.) space.

    If conditioned, uses deepfield redshifts and (currently r) magnitudes
    to condition simulated templates.
    '''
    def read_config(self,filename) :
        log = get_logger()
        log.info("Reading config {}".format(filename))
        self.config = Config(filename)

    def __init__(self,tracer, config_filename=None) :

        self.tracer = tracer.lower()  # support ELG or elg, etc.

        # AR/DK DESI spectra wavelengths
        # TODO:  where are brz extraction wavelengths defined?  https://github.com/desihub/desispec/issues/1006.
        self.wmin = 3600
        self.wmax = 9824
        self.wdelta = 0.8
        self.wave               = np.round(np.arange(self.wmin, self.wmax + self.wdelta, self.wdelta), 1)
        self.cslice             = {"b": slice(0, 2751), "r": slice(2700, 5026), "z": slice(4900, 7781)}

        if config_filename is None :
            config_filename = resource_filename('desispec', 'data/tsnr/tsnr-config-{}.yaml'.format(self.tracer))
        self.read_config(config_filename)

        self.seed = 1


    def effmag(self,m1,m2) :
        """
        returns an effective mag which is the magnitude that gives the same average flux^2
        for the mag range specified [m1,m2] assuming a flat magnitude distribution
        """
        return -0.5*2.5*np.log10( (10**(-0.8*m1)-10**(-0.8*m2))/(0.8*np.log(10.))/(m2-m1) )

    def generate_templates(self, nmodel, redshifts=None,
                           mags=None,single_mag=True):
        '''
            Dedicated wrapper for desisim.templates.GALAXY.make_templates call,
            stipulating templates in a redshift range suggested by the FDR.
            Further, assume fluxes close to the expected (within ~0.5 mags.)
            in the appropriate band.

            Class init will write ensemble stack to disk at outdir, for a given
            tracer [bgs, lrg, elg, qso], having generated nmodel templates.
            Optionally, provide redshifts and mags. to condition appropriately
            at cost of runtime.
        '''
            # Only import desisim if code is run, not at module import
        # to minimize desispec -> desisim -> desispec dependency loop
        import desisim.templates

        log = get_logger()


        # https://arxiv.org/pdf/1611.00036.pdf
        #
        normfilter_south=self.config.filter

        zrange   = (self.config.zlo, self.config.zhi)

        # Variance normalized as for psf, so we need an additional linear
        # flux loss so account for the relative factors.
        psf_loss = -self.config.psf_fiberloss / 2.5
        psf_loss = 10.**psf_loss

        rel_loss = -(self.config.wgt_fiberloss - self.config.psf_fiberloss) / 2.5
        rel_loss = 10.**rel_loss

        magrange = (self.config.med_mag, self.config.limit_mag)

        log.info('{} nmodel: {:d}'.format(self.tracer, nmodel))
        log.info('{} filter: {}'.format(self.tracer, self.config.filter))
        log.info('{} zrange: {} - {}'.format(self.tracer, zrange[0], zrange[1]))


        # NOTE THE NORMALIZATION OF MAGNITUDES DOES NOT HAVE ANY EFFECT
        # AT THE END OF THE DAY, BECAUSE BOTH TSNR2 VALUES AND EFFTIME
        # ARE RECALIBRATED TO VALUES OBTAINED EARLY IN THE SURVEY
        # (using table sv1-exposures.csv in py/desispec/data/tsnr/)
        # TO AVOID ANY ARTIFICIAL DRIFT IN THE NORMALIZATION OF THOSE
        # QUANTITIES.
        # See the scale factor applied to the flux in the routine get_ensemble
        # and the efftime normalization in the routine tsnr2_to_efftime

        # Calibration vector assumes PSF mtype.
        log.info('psf fiberloss: {:.3f}'.format(psf_loss))
        log.info('Relative fiberloss to psf morphtype: {:.3f}'.format(rel_loss))
        log.info('Generating templates ...')
        if self.tracer == 'bgs':
            # Cut on mag.
            # https://github.com/desihub/desitarget/blob/dd353c6c8dd8b8737e45771ab903ac30584db6db/py/desitarget/cuts.py#L1312
            magrange = (self.config.med_mag, self.config.limit_mag)
            if single_mag and mags is None : mags=np.repeat( self.effmag(magrange[0],magrange[1]) , nmodel)

            maker = desisim.templates.BGS(wave=self.wave, normfilter_south=normfilter_south)
            flux, wave, meta, objmeta = maker.make_templates(nmodel=nmodel, redshift=redshifts, mag=mags, south=True, zrange=zrange, magrange=magrange, seed=self.seed)

            # Additional factor rel. to psf.; TSNR put onto instrumental
            # e/A given calibration vector that includes psf-like loss.
            flux *= rel_loss

        elif self.tracer == 'lrg':
            # Cut on fib. mag. with desisim.templates setting FIBERFLUX to FLUX.
            # https://github.com/desihub/desitarget/blob/dd353c6c8dd8b8737e45771ab903ac30584db6db/py/desitarget/cuts.py#L447
            magrange = (self.config.med_fibmag, self.config.limit_fibmag)
            # consistent with tsnr on disk
            #magrange = (self.config.med_mag, self.config.limit_mag)
            if single_mag and mags is None : mags=np.repeat( self.effmag(magrange[0],magrange[1]) , nmodel)

            maker = desisim.templates.LRG(wave=self.wave, normfilter_south=normfilter_south)
            flux, wave, meta, objmeta = maker.make_templates(nmodel=nmodel, redshift=redshifts, mag=mags, south=True, zrange=zrange, magrange=magrange, seed=self.seed)

            # Take factor rel. to psf.; TSNR put onto instrumental
            # e/A given calibration vector that includes psf-like loss.
            # Note:  Oppostive to other tracers as templates normalized to fibermag.
            flux /= psf_loss
            #flux *= rel_loss

        elif self.tracer == 'elg':
            # Cut on mag.
            # https://github.com/desihub/desitarget/blob/dd353c6c8dd8b8737e45771ab903ac30584db6db/py/desitarget/cuts.py#L517
            magrange = (self.config.med_mag, self.config.limit_mag)
            if single_mag and mags is None : mags=np.repeat( self.effmag(magrange[0],magrange[1]) , nmodel)

            maker = desisim.templates.ELG(wave=self.wave, normfilter_south=normfilter_south)
            flux, wave, meta, objmeta = maker.make_templates(nmodel=nmodel, redshift=redshifts, mag=mags, south=True, zrange=zrange, magrange=magrange, seed=self.seed)

            # Additional factor rel. to psf.; TSNR put onto instrumental
            # e/A given calibration vector that includes psf-like loss.
            flux *= rel_loss

        elif self.tracer == 'qso':
            # Cut on mag.
            # https://github.com/desihub/desitarget/blob/dd353c6c8dd8b8737e45771ab903ac30584db6db/py/desitarget/cuts.py#L1422
            magrange = (self.config.med_mag, self.config.limit_mag)
            if single_mag and mags is None : mags=np.repeat( self.effmag(magrange[0],magrange[1]) , nmodel)

            maker = desisim.templates.QSO(wave=self.wave, normfilter_south=normfilter_south)
            flux, wave, meta, objmeta = maker.make_templates(nmodel=nmodel, redshift=redshifts, mag=mags, south=True, zrange=zrange, magrange=magrange, seed=self.seed)

            # Additional factor rel. to psf.; TSNR put onto instrumental
            # e/A given calibration vector that includes psf-like loss.
            flux *= rel_loss

        else:
            raise  ValueError('{} is not an available tracer.'.format(self.tracer))

        if single_mag :
            log.info('{} single effective mag: {}'.format(self.tracer, mags[0]))
        else :
            log.info('{} magrange: {} - {}'.format(self.tracer, magrange[0], magrange[1]))
        log.info("  Done generating templates")

        return  wave, flux, meta, objmeta

    def compute(self, nmodel=5, smooth=100., nz_table_filename=None, single_mag=True, convolve_to_nz=True):
        """
        Compute a template ensemble for template S/N measurements (tSNR)

        Options:
            nmodel: number of template models to generate
            smooth: smoothing scale for dF=<F - smooth(F)>
            nz_table_filename: path to ASCII file with columns zmin,zmax,n
            single_mag: generate all templates at same average magnitude to limit MC noise
            convolve_to_nz: if True, each template dF^2 is convolved to match the n(z) (redshift distribution)
        """
        log = get_logger()

        if nz_table_filename is None :
            nz_table_filename = os.environ['DESIMODEL'] + '/data/targets/nz_{}.dat'.format(self.tracer)

        _, flux, meta, objmeta         = self.generate_templates(nmodel=nmodel,single_mag=single_mag)

        # keep a copy of the templates meta data
        self.meta = meta
        for k in objmeta.dtype.names :
            if k not in self.meta.dtype.names :
                self.meta[k] = objmeta[k]

        self.ensemble_flux             = {}
        self.ensemble_dflux            = {}
        self.ensemble_meta             = meta
        self.ensemble_objmeta          = objmeta
        self.ensemble_dflux_stack      = {}
        self.smooth = smooth

        ##
        smoothing = np.ceil(smooth / self.wdelta).astype(np.int)

        log.info('Applying {:.3f} AA smoothing ({:d} pixels)'.format(smooth, smoothing))
        dflux = flux.copy()
        for i in range(flux.shape[0]):
            sflux  = convolve(flux[i], Box1DKernel(smoothing), boundary='extend')
            dflux[i] -= sflux

        log.info("Read N(z) in {}".format(nz_table_filename))
        zmin, zmax, numz = np.loadtxt(nz_table_filename, unpack=True, usecols = (0,1,2))
        # trim
        b=max(0,np.where(numz>0)[0][0]-1)
        e=min(numz.size,np.where(numz>0)[0][-1]+2)
        zmin=zmin[b:e]
        zmax=zmax[b:e]
        numz=numz[b:e]
        self.nz = Table()
        self.nz["zmin"]=zmin
        self.nz["zmax"]=zmax
        self.nz["n"]=numz
        zmid=(self.nz["zmin"]+self.nz["zmax"])/2.

        if convolve_to_nz :

            # redshifting is a simple shift in log(wave)
            # so we directy convolve with fft in a linear log(wave) grid
            # map to log scale for fast convolution with dndz
            lwave = np.log(self.wave)
            loggrid_lwave = np.linspace(lwave[0],lwave[-1],lwave.size) # linear grid of log(wave)
            loggrid_dflux = np.zeros(dflux.shape)
            loggrid_step = loggrid_lwave[1]-loggrid_lwave[0]

            loggrid_lzmin = np.log(1+zmid[0])
            number_z_bins=int((np.log(1+zmid[-1])-loggrid_lzmin)//loggrid_step)+1
            if number_z_bins%2==0 : number_z_bins += 1 # need odd number

            loggrid_lz=loggrid_lzmin+loggrid_step*np.arange(number_z_bins)
            loggrid_nz=np.interp(loggrid_lz,np.log(1+zmid),numz)

            # truncate at zrange
            loggrid_nz[(loggrid_lz<np.log(1+self.config.zlo))|(loggrid_lz>np.log(1+self.config.zhi))] = 0.

            loggrid_nz /= np.sum(loggrid_nz)
            central_lz = loggrid_lz[loggrid_lz.size//2]

            zconv_dflux = np.zeros(dflux.shape)
            for i in range(dflux.shape[0]):
                lwave_dflux = np.interp(loggrid_lwave,lwave,dflux[i])
                zi=float(self.meta['REDSHIFT'][i])
                kern = np.interp(loggrid_lz+(np.log(1+zi)-central_lz),loggrid_lz,loggrid_nz,left=0,right=0)
                if np.sum(kern)==0 : continue
                kern/=np.sum(kern)
                lwave_convolved_dflux2 = fftconvolve(lwave_dflux**2,kern,mode="same")
                zconv_dflux2 = np.interp(lwave,loggrid_lwave,lwave_convolved_dflux2,left=0,right=0)
                zconv_dflux[i] = np.sqrt(zconv_dflux2*(zconv_dflux2>0))
            dflux = zconv_dflux

        # Generate template (d)fluxes for brz bands.
        for band in ['b', 'r', 'z']:
            band_wave                 = self.wave[self.cslice[band]]
            in_band                   = np.isin(self.wave, band_wave)
            self.ensemble_flux[band]  = flux[:, in_band]
            self.ensemble_dflux[band] = dflux[:, in_band]

        zs = meta['REDSHIFT'].data

        # Stack ensemble.
        for band in ['b', 'r', 'z']:
            self.ensemble_dflux_stack[band] = np.sqrt(np.average(self.ensemble_dflux[band]**2., axis=0).reshape(1, len(self.ensemble_dflux[band].T)))

    def write(self,filename) :

        log = get_logger()

        hdr = fits.Header()
        hdr['TRACER']   = self.tracer
        hdr['FILTER']   = self.config.filter
        hdr['ZLO']      = self.config.zlo
        hdr['ZHI']      = self.config.zhi
        hdr['MEDMAG']   = self.config.med_mag
        hdr['LIMMAG']   = self.config.limit_mag
        hdr['PSFFLOSS'] = self.config.psf_fiberloss
        hdr['WGTFLOSS'] = self.config.wgt_fiberloss
        hdr['SMOOTH']   = self.smooth
        hdr['SEED']   = self.seed

        hdu_list = [fits.PrimaryHDU(header=hdr)]

        for band in ['b', 'r', 'z']:
            hdu_list.append(fits.ImageHDU(self.wave[self.cslice[band]], name='WAVE_{}'.format(band.upper())))
            hdu_list.append(fits.ImageHDU(self.ensemble_dflux_stack[band], name='DFLUX_{}'.format(band.upper())))

        hdu_list = fits.HDUList(hdu_list)

        self.meta.meta={"EXTNAME":"TEMPLATES_META"}
        hdu_list.append(fits.convenience.table_to_hdu(self.meta))

        self.nz.meta={"EXTNAME":"NZ"}
        hdu_list.append(fits.convenience.table_to_hdu(self.nz))

        hdu_list.writeto(filename, overwrite=True)

        log.info('Successfully written to '+filename)

def get_ensemble(dirpath=None, bands=["b","r","z"], smooth=0):
    '''
    Function that takes a frame object and a bitmask and
    returns ivar (and optionally mask) array(s) that have fibers with
    offending bits in fibermap['FIBERSTATUS'] set to
    0 in ivar and optionally flips a bit in mask.

    Args:
        dirpath: path to the dir. with ensemble dflux files. default is $DESIMODEL/data/tsnr
        bands:  bands to expect, typically [BRZ] - case ignored.

    Options:
        smooth:  Further convolve the residual ensemble flux.

    Returns:
        Dictionary with keys labelling each tracer (bgs, lrg, etc.) for which each value
        is a Spectra class instance with wave, flux for BRZ arms.  Note flux is the high
        frequency residual for the ensemble.  See doc. 4723.
    '''

    t0 = time.time()

    log=get_logger()
    if dirpath is None :
        dirpath = os.path.join(os.environ["DESIMODEL"],"data/tsnr")

    paths = glob.glob(dirpath + '/tsnr-ensemble-*.fits')

    wave = {}
    flux = {}
    ivar = {}
    mask = {}
    res  = {}

    ensembles  = {}

    for path in paths:
        tracer = path.split('/')[-1].split('-')[2].replace('.fits','')
        dat    = fits.open(path)

        if 'FLUXSCAL' in dat[0].header :
            scale_factor = dat[0].header['FLUXSCAL']
            log.info("for {} apply scale factor = {:4.3f}".format(path,scale_factor))
        else :
            scale_factor = 1.

        for band in bands:
            wave[band] = dat['WAVE_{}'.format(band.upper())].data
            flux[band] = scale_factor*dat['DFLUX_{}'.format(band.upper())].data
            ivar[band] = 1.e99 * np.ones_like(flux[band])

            # 125: 100. A in 0.8 pixel.
            if smooth > 0:
                flux[band] = convolve(flux[band][0,:], Box1DKernel(smooth), boundary='extend')
                flux[band] = flux[band].reshape(1, len(flux[band]))

        ensembles[tracer] = Spectra(bands, wave, flux, ivar)
        ensembles[tracer].meta = dat[0].header

    duration = time.time() - t0

    log=get_logger()
    log.info(iotime.format('read',"tsnr ensemble", duration))

    return  ensembles

def read_nea(path):
    '''
    Read a master noise equivalent area [sq. pixel] file.

    input:
        path: path to a master nea file for a given camera, e.g. b0.

    returns:
        nea: 2D split object to be evaluated at (fiber, wavelength)
        angperpix:  2D split object to be evaluated at (fiber, wavelength),
                    yielding angstrom per pixel.
    '''

    with fits.open(path, memmap=False) as fx:
        wave=fx['WAVELENGTH'].data
        angperpix=fx['ANGPERPIX'].data
        nea=fx['NEA'].data

    fiber = np.arange(len(nea))

    nea = RectBivariateSpline(fiber, wave, nea)
    angperpix = RectBivariateSpline(fiber, wave, angperpix)

    return  nea, angperpix

def fb_rdnoise(fibers, frame, psf):
    '''
    Approximate the readnoise for a given fiber (on a given camera) for the
    wavelengths present in frame. wave.

    input:
        fibers: e.g. np.arange(500) to index fiber.
        frame:  frame instance for the given camera.
        psf:  corresponding psf instance.

    returns:
        rdnoise: (nfiber x nwave) array with the estimated readnosie.  Same
                 units as OBSRDNA, e.g. ang per pix.
    '''

    ccdsizes = np.array(frame.meta['CCDSIZE'].split(',')).astype(np.float)

    xtrans = ccdsizes[0] / 2.
    ytrans = ccdsizes[1] / 2.

    rdnoise = np.zeros_like(frame.flux)

    for ifiber in fibers:
        wave_lim = psf.wavelength(ispec=ifiber, y=ytrans)
        x = psf.x(ifiber, wave_lim)

        # A | C.
        if x < xtrans:
            rdnoise[ifiber, frame.wave <  wave_lim] = frame.meta['OBSRDNA']
            rdnoise[ifiber, frame.wave >= wave_lim] = frame.meta['OBSRDNC']

        # B | D
        else:
            rdnoise[ifiber, frame.wave <  wave_lim] = frame.meta['OBSRDNB']
            rdnoise[ifiber, frame.wave >= wave_lim] = frame.meta['OBSRDND']

    return rdnoise

def var_tracer(tracer, frame, angperspecbin, fiberflat, fluxcalib, exposure_seeing_fwhm=1.1):
    '''
    Source Poisson term to the model ivar, following conventions defined at:
        https://desi.lbl.gov/trac/wiki/SurveyOps/SurveySpeed.

    See also:
        https://github.com/desihub/desispec/blob/master/py/desispec/efftime.py

    Args:
        tracer: [bgs, backup] string, defines program.
        frame: desispec.frame instance
        angperspecbin: float, angstroms per bin in spectral reductions
        fiberflat: desispec instance
        fluxcalib: desispec instance
        fiber_diameter_arcsec:

    Returns:
        nominal flux [e/specbin] corresponding to frame.wave
    '''
    log = get_logger()

    if tracer == 'bgs':
        # Nominal fiberloss dependence on seeing.  Assumes zero offset.
        fiberfrac = np.exp(0.0341 * np.log(exposure_seeing_fwhm)**3 -0.3611 * np.log(exposure_seeing_fwhm)**2 -0.7175 * np.log(exposure_seeing_fwhm) -1.5643)

        # Note: neglects transparency & EBV corrections.
        nominal = 15.8                    # r=19.5 [nanomaggie].

    elif tracer == 'backup':
        fiberfrac = np.exp(0.0989 * np.log(exposure_seeing_fwhm)**3 -0.5588 * np.log(exposure_seeing_fwhm)**2 -0.9708 * np.log(exposure_seeing_fwhm) -0.4473)
        nominal = 27.5                    # r=18.9 [nanomaggie].

    else:
        # No source poisson term otherwise.
        nominal = 0.0                     # [nanomaggie].
        fiberfrac = 1.0

        return nominal                    # [e/specbin].


    nominal *= fiberfrac

    log.info('TSNR MODEL VAR: include {} poisson source var of {:.6f} [nMg]'.format(tracer, nominal))

    nominal *= 1.e-9                      # [Mg].
    nominal /= (1.e23 / const.c.value / 1.e10 / 3631.)
    nominal /= (frame.wave)**2.           # [ergs/s/cm2/A].
    nominal *= 1.e17                      # [1.e-17 ergs/s/cm2/A].

    nominal  = fluxcalib.calib * nominal  # [e/A]

    nominal *= angperspecbin  	          # [e/specbin].
    nominal *= fiberflat.fiberflat

    log.info('TSNR MODEL VAR: include {} poisson source var of {:.6e} [e/specbin]'.format(tracer, np.median(nominal)))

    return  nominal

def var_model(rdnoise_sigma, npix_1d, angperpix, angperspecbin, fiberflat, skymodel, alpha=1.0, components=False):
    '''
    Evaluate a model for the 1D spectral flux variance, e.g. quadrature sum of readnoise and sky components.

    input:
        rdnoise_sigma:
        npix_1d:  equivalent to (1D) nea.
        angperpix:  Angstroms per pixel.
        angperspecbin: Angstroms per bin.
        fiberflat: fiberflat instance
        skymodel: Sky instance.
        alpha: empirical weighting of the rdnoise term to e.g. better fit sky fibers per exp. cam.
        components:  if True, return tuple of individual contributions to the variance.  Else return variance.

    returns:
        nfiber x nwave array of the expected variance.
    '''

    # the extraction is performed with a wavelength bin of width = angperspecbin
    # so the effective number of CCD pixels corresponding to a spectral bin width is
    npix_2d = npix_1d * (angperspecbin / angperpix)

    # then, the extracted flux per specbin is converted to an extracted flux per A, so
    # the variance has to be divided by the square of the conversion factor = angperspecbin**2

    rdnoise_variance = rdnoise_sigma**2 * npix_2d / angperspecbin**2

    # It was verified that this variance has to be increased by about 10% to match the
    # inverse variance reported in the frame files of a zero exposure (exptime=0).
    # However the correction factor (alpha) can be larger when fitted on sky fibers
    # because the precomputed effective noise equivalent number of pixels (npix_1d)
    # is valid only when the Poisson noise is negligible. It increases with the spectral flux.

    if components:
        return (alpha * rdnoise_variance, fiberflat.fiberflat * np.abs(skymodel.flux))

    else:
        return alpha * rdnoise_variance + fiberflat.fiberflat * np.abs(skymodel.flux)

def gen_mask(frame, skymodel, hw=5.):
    """
    Generate a mask for the alpha computation, masking out bright sky lines.
    Args:
        frame : uncalibrated Frame object for one camera
        skymodel : SkyModel object
        hw : (optional) float, half width of mask around sky lines in A
    Returns an array of same shape as frame, here mask=1 is good, 0 is bad
    """
    log = get_logger()

    maskfactor = np.ones_like(frame.mask, dtype=np.float)
    maskfactor[frame.mask > 0] = 0.0

    # https://github.com/desihub/desispec/blob/294cfb66428aa8be3797fd046adbd0a2267c4409/py/desispec/sky.py#L1267
    skyline=np.array([5199.4,5578.4,5656.4,5891.4,5897.4,6302.4,6308.4,6365.4,6500.4,6546.4,\
                      6555.4,6618.4,6663.4,6679.4,6690.4,6765.4,6831.4,6836.4,6865.4,6925.4,\
                      6951.4,6980.4,7242.4,7247.4,7278.4,7286.4,7305.4,7318.4,7331.4,7343.4,\
                      7360.4,7371.4,7394.4,7404.4,7440.4,7526.4,7714.4,7719.4,7752.4,7762.4,\
                      7782.4,7796.4,7810.4,7823.4,7843.4,7855.4,7862.4,7873.4,7881.4,7892.4,\
                      7915.4,7923.4,7933.4,7951.4,7966.4,7982.4,7995.4,8016.4,8028.4,8064.4,\
                      8280.4,8284.4,8290.4,8298.4,8301.4,8313.4,8346.4,8355.4,8367.4,8384.4,\
                      8401.4,8417.4,8432.4,8454.4,8467.4,8495.4,8507.4,8627.4,8630.4,8634.4,\
                      8638.4,8652.4,8657.4,8662.4,8667.4,8672.4,8677.4,8683.4,8763.4,8770.4,\
                      8780.4,8793.4,8829.4,8835.4,8838.4,8852.4,8870.4,8888.4,8905.4,8922.4,\
                      8945.4,8960.4,8990.4,9003.4,9040.4,9052.4,9105.4,9227.4,9309.4,9315.4,\
                      9320.4,9326.4,9340.4,9378.4,9389.4,9404.4,9422.4,9442.4,9461.4,9479.4,\
                      9505.4,9521.4,9555.4,9570.4,9610.4,9623.4,9671.4,9684.4,9693.4,9702.4,\
                      9714.4,9722.4,9740.4,9748.4,9793.4,9802.4,9814.4,9820.4])

    maskfactor *= (skymodel.ivar > 0.0)
    maskfactor *= (frame.ivar > 0.0)

    if hw > 0.0:
        log.info('TSNR Masking bright lines in alpha calc. (half width: {:.3f})'.format(hw))

        for line in skyline :
            if line<=frame.wave[0] or line>=frame.wave[-1]:
                continue

            ii=np.where((frame.wave>=line-hw)&(frame.wave<=line+hw))[0]

            maskfactor[:,ii]=0.0

    # Mask collimator, [4300-4500A]
    ii=np.where((frame.wave>=4300.)&(frame.wave<=4500.))[0]
    maskfactor[:,ii]=0.0

    return maskfactor

def calc_alpha(frame, fibermap, rdnoise_sigma, npix_1d, angperpix, angperspecbin, fiberflat, skymodel):
    '''
    Model Var = alpha * rdnoise component + sky.

    Calculate the best-fit alpha using the sky fibers
    available to the frame.

    input:
        frame: desispec frame instance (should be uncalibrated, i.e. e/A).
        fibermap: desispec fibermap instance.
        rdnoise_sigma:  e.g. RDNOISE value per Quadrant (float).
        npix_1d:  equivalent to 1D nea [pixels], calculated using read_nea().
        angperpix:  angstroms per pixel (float),
        fiberflat: desispec fiberflat instance.
        skymodel: desispec Sky instance.
        alpha:  nuisanve parameter to reweight rdnoise vs sky contribution to variance (float).
        components:  if True, return individual contributions to variance, else return total variance.

    returns:
       alpha:  nuisance parameter to reweight rdnoise vs sky contribution to variance (float), obtained
               as the best fit to the uncalibrated sky fibers VAR.
    '''
    log = get_logger()
    sky_indx = np.where(fibermap['OBJTYPE'] == 'SKY')[0]
    rd_var, sky_var = var_model(rdnoise_sigma, npix_1d, angperpix, angperspecbin, fiberflat, skymodel, alpha=1.0, components=True)

    maskfactor = gen_mask(frame, skymodel)
    maskfactor = maskfactor[sky_indx,:]

    def calc_alphavar(alpha):
        return alpha * rd_var[sky_indx,:] + sky_var[sky_indx,:]

    def alpha_X2(alpha):
        _var = calc_alphavar(alpha)
        _ivar =  1. / _var
        X2 = np.abs(frame.ivar[sky_indx,:] - _ivar)

        return np.sum(X2 * maskfactor)

    res = minimize(alpha_X2, x0=[1.])
    alpha = res.x[0]

    #- From JG PR #1164:
    # Noisy values of alpha can occur for observations dominated by sky noise
    # where it is not possible to calibrated the read noise. For those
    # exposures, the precise value of alpha does not impact the SNR estimation.
    if alpha < 0.8 :
        log.warning(f'tSNR forcing best fit alpha = {alpha:.4f} to 0.8')
        alpha = 0.8

    return alpha

#- Cache files from desimodel to avoid reading them N>>1 times
_camera_nea_angperpix = None
_band_ensemble = None

def calc_tsnr2_cframe(cframe):
    """
    Given cframe, calc_tsnr2 guessing frame,fiberflat,skymodel,fluxcalib to use

    Args:
        cframe: input cframe Frame object

    Returns (results, alpha) from calc_tsnr2
    """
    log = get_logger()
    dirname, filename = os.path.split(cframe.filename)
    framefile = os.path.join(dirname, filename.replace('cframe', 'frame'))
    skyfile = os.path.join(dirname, filename.replace('cframe', 'sky'))
    fluxcalibfile = os.path.join(dirname, filename.replace('cframe', 'fluxcalib'))

    for testfile in (framefile, skyfile, fluxcalibfile):
        if not os.path.exists(testfile):
            msg = 'missing {testfile}; unable to calculate TSNR2'
            log.error(msg)
            raise ValueError(msg)

    night = cframe.meta['NIGHT']
    expid = cframe.meta['EXPID']
    camera = cframe.meta['CAMERA']
    fiberflatfile = desispec.io.findfile('fiberflatnight', night, camera=camera)
    if not os.path.exists(fiberflatfile):
        ffname = os.path.basename(fiberflatfile)
        log.warning(f'{ffname} not found; using default calibs')
        fiberflatfile = findcalibfile([cframe.meta,], 'FIBERFLAT')

    frame = read_frame(framefile)
    fiberflat = read_fiberflat(fiberflatfile)
    skymodel = read_sky(skyfile)
    fluxcalib = read_flux_calibration(fluxcalibfile)

    return calc_tsnr2(frame, fiberflat, skymodel, fluxcalib)

def calc_tsnr2(frame, fiberflat, skymodel, fluxcalib, alpha_only=False, include_poisson=True) :
    '''
    Compute template SNR^2 values for a given frame

    Args:
        frame : uncalibrated Frame object for one camera
        fiberflat : FiberFlat object
        sky : SkyModel object
        fluxcalib : FluxCalib object

    returns (tsnr2, alpha):
        `tsnr2` dictionary, with keys labeling tracer (bgs,elg,etc.), of values
        holding nfiber length array of the tsnr^2 values for this camera, and
        `alpha`, the relative weighting btwn rdnoise & sky terms to model var.

    Note:  Assumes DESIMODEL is set and up to date.
    '''
    global _camera_nea_angperpix
    global _band_ensemble

    log=get_logger()

    if not (frame.meta["BUNIT"]=="count/Angstrom" or frame.meta["BUNIT"]=="electron/Angstrom" ) :
        log.error("requires an uncalibrated frame")
        raise RuntimeError("requires an uncalibrated frame")

    camera=frame.meta["CAMERA"].strip().lower()
    band=camera[0]

    psfpath=findcalibfile([frame.meta],"PSF")
    psf=GaussHermitePSF(psfpath)

    # Returns bivariate spline to be evaluated at (fiber, wave).
    if not "DESIMODEL" in os.environ :
        msg = "requires $DESIMODEL to get the NEA and the SNR templates"
        log.error(msg)
        raise RuntimeError(msg)

    if _camera_nea_angperpix is None:
        _camera_nea_angperpix = dict()

    if camera in _camera_nea_angperpix:
        nea, angperpix = _camera_nea_angperpix[camera]
    else:
        neafilename=os.path.join(os.environ["DESIMODEL"],
                                 f"data/specpsf/nea/masternea_{camera}.fits")
        log.info("read NEA file {}".format(neafilename))
        nea, angperpix = read_nea(neafilename)
        _camera_nea_angperpix[camera] = nea, angperpix

    if _band_ensemble is None:
        _band_ensemble = dict()

    if band in _band_ensemble:
        ensemble = _band_ensemble[band]
    else:
        ensembledir=os.path.join(os.environ["DESIMODEL"],"data/tsnr")
        log.info("read TSNR ensemble files in {}".format(ensembledir))
        ensemble = get_ensemble(ensembledir, bands=[band,])
        _band_ensemble[band] = ensemble

    nspec, nwave = fluxcalib.calib.shape

    fibers = np.arange(nspec)
    rdnoise = fb_rdnoise(fibers, frame, psf)

    #
    ebv = frame.fibermap['EBV']

    if np.sum(ebv!=0)>0 :
        log.info("TSNR MEDIAN EBV = {:.3f}".format(np.median(ebv[ebv!=0])))
    else :
        log.info("TSNR MEDIAN EBV = 0")

    # Evaluate.
    npix = nea(fibers, frame.wave)
    angperpix = angperpix(fibers, frame.wave)
    angperspecbin = np.mean(np.gradient(frame.wave))

    for label, x in zip(['RDNOISE', 'NEA', 'ANGPERPIX', 'ANGPERSPECBIN'], [rdnoise, npix, angperpix, angperspecbin]):
        log.info('{} \t {:.3f} +- {:.3f}'.format(label.ljust(10), np.median(x), np.std(x)))

    # Relative weighting between rdnoise & sky terms to model var.
    alpha = calc_alpha(frame, fibermap=frame.fibermap,
                rdnoise_sigma=rdnoise, npix_1d=npix,
                angperpix=angperpix, angperspecbin=angperspecbin,
                fiberflat=fiberflat, skymodel=skymodel)

    log.info(f"TSNR ALPHA = {alpha:.6f}")

    if alpha_only:
        return {}, alpha

    maskfactor = np.ones_like(frame.mask, dtype=np.float)
    maskfactor[frame.mask > 0] = 0.0
    maskfactor *= (frame.ivar > 0.0)

    tsnrs = {}

    for tracer in ensemble.keys():
        wave = ensemble[tracer].wave[band]
        dflux = ensemble[tracer].flux[band]

        if len(frame.wave) != len(wave) or not np.allclose(frame.wave, wave):
            log.warning(f'Resampling {tracer} ensemble wavelength to match input {camera} frame')
            tmp = np.zeros([dflux.shape[0], len(frame.wave)])
            for i in range(dflux.shape[0]):
                tmp[i] = np.interp(frame.wave, wave, dflux[i],
                            left=dflux[i,0], right=dflux[i,-1])
            dflux = tmp
            wave = frame.wave

        denom = var_model(rdnoise, npix, angperpix, angperspecbin, fiberflat, skymodel, alpha=alpha)

        if include_poisson:
            # TODO:  Fix default seeing-fiberfrac relation.
            denom += var_tracer(tracer, frame, angperspecbin, fiberflat, fluxcalib)

        # Work in uncalibrated flux units (electrons per angstrom); flux_calib includes exptime. tau.
        # Broadcast.
        dflux = dflux * fluxcalib.calib # [e/A]

        # Wavelength dependent fiber flat;  Multiply or divide - check with Julien.
        result = dflux * fiberflat.fiberflat

        # Apply dust transmission.
        result *= dust_transmission(frame.wave, ebv[:,None])

        result = result**2.

        result /= denom

        # Eqn. (1) of https://desi.lbl.gov/DocDB/cgi-bin/private/RetrieveFile?docid=4723;filename=sky-monitor-mc-study-v1.pdf;version=2
        tsnrs[tracer] = np.sum(result * maskfactor, axis=1)

    results=dict()
    for tracer in tsnrs.keys():
        key = 'TSNR2_{}_{}'.format(tracer.upper(), band.upper())
        results[key]=tsnrs[tracer]
        log.info('{} = {:.6f}'.format(key, np.median(tsnrs[tracer])))

    return results, alpha

def tsnr2_to_efftime(tsnr2,target_type,program="DARK") :
    """ Converts TSNR2 values to effective exposure time.
    Args:
      tsnr2: TSNR**2 values, float or numpy array
      target_type: str, "ELG","BGS","LYA", or other depending on content of  data/tsnr/tsnr-efftime.yaml
      program: str, "DARK", "BRIGHT", or other depending on content of data/tsnr/tsnr-efftime.yaml

    Returns: exptime in seconds, same type and shape if applicable as input tsnr2
    """

    #keyword="TSNR2_{}_TO_EFFTIME_{}".format(target_type,program)
    #if not keyword in efftime_config :
    #    message="no calibration for TSNR2_{} to EFFTIME_{}".format(target_type,program)
    #    raise KeyError(message)

    tracer=target_type.lower()
    tsnr_ensembles = get_ensemble()
    log = get_logger()
    if not "SNR2TIME" in tsnr_ensembles[tracer].meta.keys() :
        message = "did not find key SNR2TIME in tsnr_ensemble fits file header, the tsnr files must be deprecated, please update DESIMODEL."
        log.error(message)
        return np.zeros_like(tsnr2)

    slope = tsnr_ensembles[tracer].meta["SNR2TIME"]
    log.info("for tracer {} SNR2TIME={:f}".format(tracer,slope))

    return slope*tsnr2
