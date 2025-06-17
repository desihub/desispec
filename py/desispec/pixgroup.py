"""
desispec.pixgroup
=================

Tools to regroup spectra in individual exposures by healpix on the sky.
"""

import glob, os, sys, time, json
from collections import Counter, OrderedDict
import gzip, shutil

import numpy as np

import fitsio
from astropy.io import fits
from astropy.table import Table, vstack

from desimodel.footprint import radec2pix
from desiutil.log import get_logger
import desiutil.depend

from . import io
from .io.util import get_tempfilename, addkeys
from .maskbits import specmask
from .tsnr import calc_tsnr2_cframe

def fibermap2tilepix(fibermap, nside=64):
    """
    Maps fibermap to which healpix are covered by which petals

    Args:
        fibermap: table with columns TARGET_RA, TARGET_DEC, PETAL_LOC

    Options:
        nside (int): nested healpix nside (must be power of 2)

    Returns dict petalpix[petal] = list(healpix covered by that petal)
    """
    tilepix = dict()
    ra = fibermap['TARGET_RA']
    dec = fibermap['TARGET_DEC']
    ok = ~np.isnan(ra) & ~np.isnan(dec)
    for petal in range(10):
        ii = (fibermap['PETAL_LOC'] == petal) & ok
        healpix = np.unique(radec2pix(nside, ra[ii], dec[ii]))
        tilepix[petal] = [int(p) for p in healpix]

    return tilepix

def get_exp2healpix_map(expfile, survey=None, program=None,
        specprod_dir=None, nights=None, expids=None):
    """
    Maps exposures to healpixels using preproc/NIGHT/EXPID/tilepix*.json files

    Args:
        expfile (str): filepath to production exposure file with EXPOSURES and FRAMES HDUs

    Options:
        survey (str): filter by this survey (main, sv3, sv1, ...)
        program (str): filter by this FAPRGRM (dark, bright, backup, other)
        specprod_dir (str): override $DESI_SPECTRO_REDUX/$SPECPROD
        nights (array-like): filter to only these nights
        expids (array-like): filter to only these exposure IDs

    Returns table with columns NIGHT EXPID TILEID SURVEY PROGRAM SPECTRO HEALPIX
    """
    log = get_logger()
    if specprod_dir is None:
        specprod_dir = io.specprod_root()

    log.info(f'Reading exposures list from {expfile}')
    exps = Table.read(expfile, 'EXPOSURES')
    frames = Table.read(expfile, 'FRAMES')

    #- override FAPRGRM with what we would set it to now
    exps['FAPRGRM'] = io.meta.faflavor2program(exps['FAFLAVOR'])
    keep = exps['TILEID'] > 0
    if survey is not None:
        keep &= (exps['SURVEY'] == survey)
    if program is not None:
        keep &= (exps['FAPRGRM'] == program)
    if expids is not None:
        keep &= np.isin(exps['EXPID'], expids)
    if nights is not None:
        keep &= np.isin(exps['NIGHT'], nights)

    exptab = exps['NIGHT', 'EXPID', 'TILEID', 'SURVEY', 'FAPRGRM'][keep]

    #- From FRAMES, determine which petals actually have data
    tilepetals_with_data = dict()
    for tileid, camera in frames['TILEID', 'CAMERA']:
        if tileid not in tilepetals_with_data:
            tilepetals_with_data[tileid] = set()
        petal = int(camera[1])
        tilepetals_with_data[tileid].add(petal)

    #- read one tilepix file per TILEID to get healpix mapping
    tilepix = dict()
    for i in np.unique(exptab['TILEID'], return_index=True)[1]:
        night = exptab['NIGHT'][i]
        expid = exptab['EXPID'][i]
        tileid = exptab['TILEID'][i]
        tilepixfile = io.findfile('tilepix', night, expid, tile=tileid, readonly=True)
        with open(tilepixfile) as fp:
            tilepix.update( json.load(fp) )

    #- rows for table columns NIGHT EXPID TILEID SURVEY PROGRAM SPECTRO HEALPIX
    rows = list()

    #- Add entries for each exposure
    for night, expid, tileid, survey, program in exptab['NIGHT', 'EXPID', 'TILEID', 'SURVEY', 'FAPRGRM']:
        for petal_str in tilepix[str(tileid)]:
            petal = int(petal_str)
            if petal in tilepetals_with_data[tileid]:
                for healpix in tilepix[str(tileid)][petal_str]:
                    rows.append( (night, expid, tileid, survey, program, petal, healpix) )

    if len(rows) == 0:
        raise RuntimeError('No matching tilepix found')

    exp2pix = Table(rows=rows, names=('NIGHT', 'EXPID', 'TILEID', 'SURVEY', 'PROGRAM', 'SPECTRO', 'HEALPIX'))

    return exp2pix

#-----
class FrameLite(object):
    '''
    Lightweight Frame object for regrouping

    This is intended for I/O without the overheads of float32 -> float64
    conversion, correcting endianness, etc.
    '''
    def __init__(self, wave, flux, ivar, mask, resolution_data, fibermap, header, scores=None, model=None, redshifts=None):
        '''
        Create a new FrameLite object

        Args:
            wave: 1D array of wavlengths
            flux: 2D[nspec, nwave] fluxes
            ivar: 2D[nspec, nwave] inverse variances of flux
            mask: 2D[nspec, nwave] mask of flux; 0=good
            resolution_data 3D[nspec, ndiag, nwave] Resolution matrix diagonals
            fibermap: fibermap table
            header: FITS header

        Options:
            scores: table of QA scores
            model: 2D[nspec, nwave] spectral models
            redshifts: table of best-fit redshifts
        '''
        self.wave = wave
        self.flux = flux
        self.ivar = ivar
        self.mask = mask
        self.resolution_data = resolution_data
        self.fibermap = fibermap
        self.header = header
        self.meta = header  #- for compatibility with Frame objects
        self.scores = scores
        self.model = model
        self.redshifts = redshifts

    def __getitem__(self, index):
        '''Return a subset of the original FrameLight'''
        if not isinstance(index, slice):
            index = np.atleast_1d(index)

        if self.scores is not None:
            scores = self.scores[index]
        else:
            scores = None
            
        if self.redshifts is not None:
            redshifts = self.redshifts[index]
        else:
            redshifts = None
            
        if self.model is not None:
            model = {}
            for b in self.flux.keys():
                model[b] = self.model[b][index]
        else:
            model = None

        return FrameLite(self.wave, self.flux[index], self.ivar[index],
            self.mask[index], self.resolution_data[index], self.fibermap[index],
            self.header, scores, model, redshifts)

    @classmethod
    def read(cls, filename):
        '''
        Return FrameLite read from `filename`
        '''
        with fitsio.FITS(filename) as fx:
            header = fx[0].read_header()
            wave = fx['WAVELENGTH'].read()
            flux = fx['FLUX'].read()
            ivar = fx['IVAR'].read()
            mask = fx['MASK'].read()
            resolution_data = fx['RESOLUTION'].read()
            fibermap = fx['FIBERMAP'].read()
            fmhdr = fx['FIBERMAP'].read_header()
            if "MODEL" in fx:
                model = fx["MODEL"].read()
            else:
                model = None

            for col in ['SURVEY', 'PROGRAM']:
                if col in fmhdr and col not in header:
                    header[col] = fmhdr[col]

            if 'SCORES' in fx:
                scores = fx['SCORES'].read()
                if 'TARGETID' not in scores.dtype.names:
                    tmp = Table(scores)
                    tmp.add_column(fibermap['TARGETID'], index=0, name='TARGETID')
                    scores = np.asarray(tmp)
            else:
                scores = None

            if "REDSHIFTS" in fx:
                redshifts = Table(fx["REDSHIFTS"].read())
            else:
                redshifts = None
                
        #- Add extra fibermap columns NIGHT, EXPID, TILEID
        nspec = len(fibermap)
        night = np.tile(header['NIGHT'], nspec).astype('i4')
        expid = np.tile(header['EXPID'], nspec).astype('i4')
        tileid = np.tile(header['TILEID'], nspec).astype('i4')
        if 'MJD-OBS' in header:
            mjd = np.tile(header['MJD-OBS'], nspec).astype('f8')
        elif 'MJD' in header:
            mjd = np.tile(header['MJD'], nspec).astype('f8')
        else:
            mjd = np.zeros(nspec, dtype='f8')-1

        fibermap = Table(fibermap)
        fibermap['NIGHT'] = night
        fibermap['EXPID'] = expid
        fibermap['MJD'] = mjd
        fibermap['TILEID'] = tileid
        addkeys(fibermap.meta, fmhdr)

        fr = FrameLite(wave, flux, ivar, mask, resolution_data, fibermap, header, scores, model, redshifts)
        fr.filename = filename
        return fr

#-----
class SpectraLite(object):
    '''
    Lightweight spectra I/O object for regrouping
    '''
    def __init__(self, bands, wave, flux, ivar, mask, resolution_data,
            fibermap, exp_fibermap=None, scores=None, model=None, redshifts=None):
        '''
        Create a SpectraLite object

        Args:
            bands: list of bands, e.g. ['b', 'r', 'z']
            wave: dict of wavelengths, keyed by band
            flux: dict of fluxes, keyed by band
            ivar: dict of inverse variances, keyed by band
            mask: dict of masks, keyed by band
            resolution_data: dict of Resolution sparse diagonals, keyed by band
            fibermap: fibermap table, applies to all bands

        Options:
            exp_fibermap: per-exposure fibermap table
            scores: scores table, applies to all bands
            model: 2D[nspec, nwave] spectral models
            redshifts: table of best-fit redshifts
        '''

        self.bands = bands[:]

        #- All inputs should have the same bands
        _bands = set(bands)
        assert set(wave.keys()) == _bands
        assert set(flux.keys()) == _bands
        assert set(ivar.keys()) == _bands
        assert set(mask.keys()) == _bands
        assert set(resolution_data.keys()) == _bands

        #- All bands should have the same number of spectra
        nspec = len(fibermap)
        for band in bands:
            assert flux[band].shape[0] == nspec
            assert ivar[band].shape[0] == nspec
            assert mask[band].shape[0] == nspec
            assert resolution_data[band].shape[0] == nspec

            #- Also check wavelength dimension consistency
            nwave = len(wave[band])
            assert flux[band].shape[1] == nwave
            assert ivar[band].shape[1] == nwave
            assert mask[band].shape[1] == nwave
            assert resolution_data[band].shape[2] == nwave  #- resolution_data[band].shape[1] is ndiag

        #- scores and fibermap should be row matched with same length
        if scores is not None:
            assert len(scores) == len(fibermap)

        self.wave = wave.copy()
        self.flux = flux.copy()
        self.ivar = ivar.copy()
        self.mask = mask.copy()
        self.resolution_data = resolution_data.copy()
        self.fibermap = Table(fibermap)

        if model is not None:
            self.model = model.copy()
        else:
            self.model = None
            
        #- optional tables
        if exp_fibermap is not None:
            self.exp_fibermap = Table(exp_fibermap)
        else:
            self.exp_fibermap = None

        if scores is not None:
            self.scores = Table(scores)
        else:
            self.scores = None

        if redshifts is not None:
            self.redshifts = Table(redshifts)
        else:
            self.redshifts = None
            
        #- for compatibility with full Spectra objects
        self.meta = None
        self.extra = None
        self.extra_catalog = None

    def target_ids(self):
        """
        Return list of unique target IDs.

        The target IDs are sorted by the order that they first appear.

        Returns (array):
            an array of integer target IDs.
        """
        uniq, indices = np.unique(self.fibermap["TARGETID"], return_index=True)
        return uniq[indices.argsort()]

    def num_spectra(self):
        """
        Get the number of spectra contained in this group.

        Returns (int):
            Number of spectra contained in this group.
        """
        if self.fibermap is not None:
            return len(self.fibermap)
        else:
            return 0


    def num_targets(self):
        """
        Get the number of distinct targets.

        Returns (int):
            Number of unique targets with spectra in this object.
        """
        if self.fibermap is not None:
            return len(np.unique(self.fibermap["TARGETID"]))
        else:
            return 0

    def __add__(self, other):
        '''
        concatenate two SpectraLite objects into one
        '''
        assert self.bands == other.bands
        for band in self.bands:
            assert np.all(self.wave[band] == other.wave[band])
        if self.scores is not None:
            assert other.scores is not None

        bands = self.bands
        wave = self.wave
        flux = dict()
        ivar = dict()
        mask = dict()
        resolution_data = dict()

        if self.model is not None and other.model is not None:
            model = {}
        else:
            model = None
            
        for band in self.bands:
            flux[band] = np.vstack([self.flux[band], other.flux[band]])
            ivar[band] = np.vstack([self.ivar[band], other.ivar[band]])
            mask[band] = np.vstack([self.mask[band], other.mask[band]])
            resolution_data[band] = np.vstack([self.resolution_data[band], other.resolution_data[band]])
            if self.model is not None and other.model is not None:
                model[band] = np.vstack([self.model[band], other.model[band]])

        fibermap = vstack([self.fibermap, other.fibermap])
        if self.scores is not None:
            scores = np.hstack([self.scores, other.scores])
        else:
            scores = None

        if self.redshifts is not None and other.redshifts is not None:
            redshifts = np.hstack([self.redshifts, other.redshifts])
        else:
            redshifts = None

        if self.exp_fibermap is not None:
            exp_fibermap = np.hstack([self.exp_fibermap, other.exp_fibermap])
        else:
            exp_fibermap = None

        return SpectraLite(bands, wave, flux, ivar, mask, resolution_data,
                fibermap, exp_fibermap=exp_fibermap, scores=scores, model=model, redshifts=redshifts)

    def write(self, filename, header=None):
        '''
        Write this SpectraLite object to `filename`
        '''
        log = get_logger()
        log.warning('SpectraLite.write() is deprecated; please use desispec.io.write_spectra() instead')

        #- create directory if missing
        dirname=os.path.dirname(filename)
        if dirname != '':
            os.makedirs(dirname, exist_ok=True)

        #- Have to first write non-gzip so that we can append
        if filename.endswith('.gz'):
            tmpout = get_tempfilename(filename[-3:])
        else:
            tmpout = get_tempfilename(filename)

        #- work around c/fitsio bug that appends spaces to string column values
        #- by using astropy Table to write fibermap

        from astropy.table import Table
        fm = Table(self.fibermap)
        fm.meta['EXTNAME'] = 'FIBERMAP'

        header = io.fitsheader(header)
        desiutil.depend.add_dependencies(header)
        hdus = fits.HDUList()
        hdus.append(fits.PrimaryHDU(None, header))
        hdus.append(fits.convenience.table_to_hdu(fm))

        if self.exp_fibermap is not None:
            expfm = Table(self.exp_fibermap)
            expfm.meta['EXTNAME'] = 'EXP_FIBERMAP'
            hdus.append(fits.convenience.table_to_hdu(expfm))

        if self.scores is not None:
            scores = Table(self.scores)
            scores.meta['EXTNAME'] = 'SCORES'
            hdus.append(fits.convenience.table_to_hdu(scores))

        if self.redshifts is not None:
            redshifts = Table(self.redshifts)
            redshifts.meta['EXTNAME'] = 'REDSHIFTS'
            hdus.append(fits.convenience.table_to_hdu(redshifts))

        hdus.writeto(tmpout, overwrite=True, checksum=True)

        #- then proceed with more efficient fitsio for everything else
        #- See https://github.com/esheldon/fitsio/issues/150 for why
        #- these are written one-by-one
        ### if self.scores is not None:
        ###     fitsio.write(tmpout, self.scores, extname='SCORES')

        for band in sorted(self.bands):
            upperband = band.upper()
            fitsio.write(tmpout, self.wave[band], extname=upperband+'_WAVELENGTH',
                    header=dict(BUNIT='Angstrom'))
            fitsio.write(tmpout, self.flux[band], extname=upperband+'_FLUX',
                    header=dict(BUNIT='10**-17 erg/(s cm2 Angstrom)'))
            fitsio.write(tmpout, self.ivar[band], extname=upperband+'_IVAR',
                header=dict(BUNIT='10**+34 (s2 cm4 Angstrom2) / erg2'))
            if self.model is not None:
                fitsio.write(tmpout, self.model[band], extname=upperband+'_MODEL',
                    header=dict(BUNIT='10**-17 erg/(s cm2 Angstrom)'))
            fitsio.write(tmpout, self.mask[band], extname=upperband+'_MASK', compress='gzip')
            fitsio.write(tmpout, self.resolution_data[band], extname=upperband+'_RESOLUTION')

        #- compress if needed (via another tempfile), otherwise just rename
        if filename.endswith('.gz'):
            tmpoutgz = get_tempfilename(filename)
            with open(tmpout, 'rb') as f_in:
                with gzip.open(tmpoutgz, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.rename(tmpoutgz, filename)
            os.remove(tmpout)
        else:
            os.rename(tmpout, filename)

    @classmethod
    def read(cls, filename):
        '''
        Return a SpectraLite object read from `filename`
        '''
        with fitsio.FITS(filename) as fx:
            wave = dict()
            flux = dict()
            ivar = dict()
            mask = dict()
            resolution_data = dict()
            fibermap = fx['FIBERMAP'].read()
            model_exists = any("_MODEL" in fx)
            if model_exists:
                model = {}
            else:
                model = None
            if 'EXP_FIBERMAP' in fx:
                exp_fibermap = fx['EXP_FIBERMAP'].read()
            else:
                exp_fibermap = None

            if 'SCORES' in fx:
                scores = fx['SCORES'].read()
            else:
                scores = None

            if 'REDSHIFTS' in fx:
                redshifts = fx['REDSHIFTS'].read()
            else:
                redshifts = None

            bands = ['b', 'r', 'z']
            for band in bands:
                upperband = band.upper()
                wave[band] = fx[upperband+'_WAVELENGTH'].read()
                flux[band] = fx[upperband+'_FLUX'].read()
                ivar[band] = fx[upperband+'_IVAR'].read()
                mask[band] = fx[upperband+'_MASK'].read()
                resolution_data[band] = fx[upperband+'_RESOLUTION'].read()
                if model is not None:
                    model[band] = fx[upperband+'_MODEL'].read()

        return SpectraLite(bands, wave, flux, ivar, mask, resolution_data,
                fibermap, exp_fibermap=exp_fibermap, scores=scores, model=model, redshifts=redshifts)

def add_missing_frames(frames):
    '''
    Adds any missing frames with ivar=0 FrameLite objects with correct shape
    to match those that do exist.

    Args:
        frames: dict of FrameLite objects, keyed by (night,expid,camera)

    Modifies `frames` in-place.

    Example: if `frames` has keys (2020,1,'b0') and (2020,1,'r0') but
    not (2020,1,'z0'), this will add a blank FrameLite object for z0.

    The purpose of this is to facilitate frames2spectra, which needs
    *something* for every spectro camera for every exposure that is included.
    '''

    log = get_logger()

    #- First figure out the number of wavelengths per band
    wave = dict()
    ndiag = dict()
    for (night, expid, camera), frame in frames.items():
        band = camera[0]
        if band not in wave:
            wave[band] = frame.wave
        if band not in ndiag:
            ndiag[band] = frame.resolution_data.shape[1]

    #- Now loop through all frames, filling in any missing bands
    bands = sorted(list(wave.keys()))
    for (night, expid, camera), frame in list(frames.items()):
        frameband = camera[0]
        spectro = camera[1:]
        for band in bands:
            if band == frameband:
                continue

            bandcam = band+spectro
            if (night, expid, bandcam) in frames:
                continue

            log.warning('Creating blank data for missing frame {}'.format(
                (night, expid, bandcam)))
            nwave = len(wave[band])
            nspec = frame.flux.shape[0]
            flux = np.zeros((nspec, nwave), dtype='f4')
            ivar = np.zeros((nspec, nwave), dtype='f4')
            mask = np.zeros((nspec, nwave), dtype='u4') + specmask.NODATA
            resolution_data = np.zeros((nspec, ndiag[band], nwave), dtype='f4')

            #- Copy the header and correct the camera keyword
            if type(frame.meta) is fits.header.Header:
                header = fitsio.FITSHDR( OrderedDict(frame.meta) )
            else:
                header = fitsio.FITSHDR(frame.meta)
            header['camera'] = bandcam

            #- Make new blank scores, replacing trailing band _B/R/Z
            if frame.scores is not None:
                dtype = [ ('TARGETID', 'i8') ]
                for name in frame.scores.dtype.names:
                    if name.endswith('_'+frameband.upper()):
                        bandname = name[0:-1] + band.upper()
                        dtype.append((bandname, type(frame.scores[name][0])))

                scores = np.zeros(nspec, dtype=dtype)
                scores['TARGETID'] = frame.scores['TARGETID']
            else:
                scores = None

            #- Add the blank FrameLite object
            frames[(night,expid,bandcam)] = FrameLite(
                wave[band], flux, ivar, mask, resolution_data,
                frame.fibermap, header, scores)

def frames2spectra(frames, pix=None, nside=64, onetile=False):
    '''
    Combine a dict of FrameLite into a SpectraLite for healpix `pix`

    Args:
        frames: list or dict of FrameLight, keyed by (night, expid, camera)

    Options:
        pix: only include targets in this NESTED healpix pixel number
        nside: Healpix nside, must be power of 2
        onetile: All frames come from the same tile (impacts keyword propagation)

    Returns:
        SpectraLite object with subset of spectra from frames that are in
        the requested healpix pixel `pix`

    If `onetile` is True, propagate fibermap header keywords that apply
    to a single tile, e.g. TILEID and fiberassign keywords
    '''
    log = get_logger()

    #- support list or tuple instead of dict as input
    if isinstance(frames, (list, tuple)):
        framedict = dict()
        for fr in frames:
            night = fr.meta['NIGHT']
            expid = fr.meta['EXPID']
            camera = fr.meta['CAMERA']
            framedict[(night, expid, camera)] = fr

        frames = framedict

    #- shallow copy of frames dict in case we augment with blank frames
    frames = frames.copy()

    if pix is not None:
        log.info(f'Filtering by nside={nside} nested healpix={pix}')

    #- To support combining old+new data, recalculate TSNR2 if any
    #- frames are missing TSNR2* scores present in other frames of same bad.
    #- Assume longest list per camera is the one we want, because we also
    #- need to preserve order.  Ugh.
    frame_scores_columns = dict()
    scores_columns = dict(b=[], r=[], z=[])
    for (night,expid,cam), frame in frames.items():
        brz = cam.lower()[0]
        cols = frame.scores.dtype.names
        frame_scores_columns[(night,expid,cam)] = cols
        if len(cols) > len(scores_columns[brz]):
            scores_columns[brz] = cols

    for (night,expid,cam), frame in frames.items():
        brz = cam.lower()[0]
        if frame_scores_columns[(night,expid,cam)] != scores_columns[brz]:
            log.warning(f'Recalculating TSNR2 for ({night},{expid},{cam})')
            s = Table(frame.scores)
            tsnr2, alpha = calc_tsnr2_cframe(frame)
            for key in tsnr2:
                s[key] = tsnr2[key]

            #- standardize order
            frame.scores = np.array(s[scores_columns[brz]])

    #- Make sure that the given set of frames is complete
    #- If not, fill in missing cameras so that the stack works properly
    add_missing_frames(frames)

    #- Setup data structures to fill
    wave, flux, ivar, mask = dict(), dict(), dict(), dict()
    resolution_data, scores, fmaps = dict(), dict(), dict()

    #- Get the bands that exist in the input data
    #- identify all of the exposures for each band
    #- and instantiate some variables for the next loop
    bands, allkeys =  list(),  dict()
    for (night,expid,cam),frame in frames.items():
        band = cam[0]
        if band not in bands:
            bands.append(band)
            allkeys[band] = list()
            flux[band] = list()
            ivar[band] = list()
            mask[band] = list()
            resolution_data[band] = list()
            scores[band] = list()
            wave[band] = frame.wave
        if (cam[1],night,expid) not in fmaps.keys():
            fmaps[(cam[1],night,expid)] = dict()
        fmaps[(cam[1],night,expid)][band] = None
        allkeys[band].append((cam[1],night,expid,cam))

    bands = sorted(bands)
    for band in bands:
        assert len(allkeys[band]) != 0
        #- Select just the frames for this band using keys collected above
        #  Sort them so we are ordered by spectrograph, then night, then exposure
        band_keys = sorted(allkeys[band])

        #- Select flux, ivar, etc. for just the spectra on this healpix
        for spec,night,expid,cam in band_keys:
            bandframe = frames[(night,expid,cam)]
            if pix is not None:
                ra, dec = bandframe.fibermap['TARGET_RA'], bandframe.fibermap['TARGET_DEC']
                ok = ~np.isnan(ra) & ~np.isnan(dec)
                ra[~ok] = 0.0
                dec[~ok] = 0.0
                allpix = radec2pix(nside, ra, dec)
                ii = (allpix == pix) & ok
            else:
                ii = np.ones(bandframe.flux.shape[0]).astype(bool)

            #- Careful: very similar code below for non-filtered appending
            flux[band].append(bandframe.flux[ii])
            ivar[band].append(bandframe.ivar[ii])
            mask[band].append(bandframe.mask[ii])
            resolution_data[band].append(bandframe.resolution_data[ii])

            fmaps[(spec,night,expid)][band] = bandframe.fibermap[ii]

            if bandframe.scores is not None:
                scores[band].append(bandframe.scores[ii])


        flux[band] = np.vstack(flux[band])
        ivar[band] = np.vstack(ivar[band])
        mask[band] = np.vstack(mask[band])
        resolution_data[band] = np.vstack(resolution_data[band])

        if len(scores[band]) > 0:
            scores[band] = np.hstack(scores[band])

    #- Combine scores into a single table
    #- Why doesn't np.vstack work for this? (says invalid type promotion)
    if len(scores[bands[0]]) > 0:
        if len(bands) == 1:
            scores = scores[bands[0]]
        else:
            names = list()
            data = list()
            for band in bands[1:]:
                for colname in scores[band].dtype.names:
                    #- TARGETID appears in multiple bands; only add once
                    if colname != 'TARGETID':
                        names.append(colname)
                        data.append(scores[band][colname])
            scores = np.lib.recfunctions.append_fields(
                    scores[bands[0]], names, data)

    else:
        scores = None

    #- Go in order over all exposures, pick up fibermaps. When multiple, use the first
    #- but bitwise OR the fiberstatus.
    merged_over_cams_fmaps = list()
    sorted_keys = sorted(list(fmaps.keys()))
    for key in sorted_keys:
        banddict = fmaps[key]
        outmap = None
        for band in bands:
            if banddict[band] is None:
                continue
            if outmap is None:
                outmap = banddict[band]
            else:
                outmap['FIBERSTATUS'] |= banddict[band]['FIBERSTATUS']
        merged_over_cams_fmaps.append(outmap)

    #- Only keep fibermap keywords that apply to combined spectra, not individual exp
    general_keywords = [
            'SURVEY', 'PROGRAM', 'EXTNAME', 'LONGSTRN',
            'INSTRUME', 'OBSERVAT', 'OBS-LAT', 'OBS-LONG', 'OBS-ELEV', 'TELESCOP'
            ]

    tile_keywords = [
            'TILEID', 'TILERA', 'TILEDEC', 'FIELDROT',
            'FILEDROT', 'FA_PLAN', 'FA_HA', 'FA_RUN', 'FA_M_GFA',
            'FA_M_PET', 'FA_M_POS', 'REQRA', 'REQDEC', 'FIELDNUM',
            'FA_VER', 'FA_SURV', 'FAFLAVOR', 'FAPRGRM', 'DESIROOT',
            'GFA', 'MTL', 'SCND', 'SCND2', 'SCND3', 'SCNDMTL', 'SKY',
            'SKYSUPP', 'TARG', 'TOO', 'FAARGS', 'FAOUTDIR',
            'NOWTIME', 'RUNDATE', 'PMCORR', 'PMTIME', 'MTLTIME',
            'OBSCON', 'GOALTIME', 'GOALTYPE', 'EBVFAC', 'SBPROF',
            'MINTFRAC', 'FASCRIPT', 'SVNDM', 'SVNMTL',
            ]

    if onetile:
        keep_keywords = general_keywords + tile_keywords
    else:
        keep_keywords = general_keywords

    for fm in merged_over_cams_fmaps:
        for key in list(fm.meta.keys()):
            if key not in keep_keywords:
                del fm.meta[key]

    #- Convert to Tables to facilitate column add/drop/reorder
    for i, fm in enumerate(merged_over_cams_fmaps):
        if not isinstance(fm, Table):
            merged_over_cams_fmaps[i] = Table(fm)

    #- Standardize all fibermaps to have the same columns in same order
    if len(merged_over_cams_fmaps) > 1:
        colnames = list(merged_over_cams_fmaps[0].colnames)
        for fm in merged_over_cams_fmaps[1:]:
            for drop in set(colnames) - set(fm.colnames):
                log.warning(f"Ignoring {drop}, missing from some fibermaps")
                colnames.remove(drop)
            #- Also warn about columns that were never in original list
            for drop in set(fm.colnames) - set(colnames):
                log.warning(f"Ignoring {drop}, missing from some fibermaps")

        for i in range(len(merged_over_cams_fmaps)):
            merged_over_cams_fmaps[i] = merged_over_cams_fmaps[i][colnames]

    #- Combine all the individual fibermaps from the exposures and spectrographs
    fibermap = vstack(merged_over_cams_fmaps)

    #- assemble_fibermap now sets NaN to 0,
    #- but reset here too if combining older data
    for col in [
        'FIBER_X', 'FIBER_Y',
        'DELTA_X', 'DELTA_Y',
        'FIBER_RA', 'FIBER_DEC',
        'GAIA_PHOT_G_MEAN_MAG',
        'GAIA_PHOT_BP_MEAN_MAG',
        'GAIA_PHOT_RP_MEAN_MAG',
        ]:
        ii = np.isnan(fibermap[col])
        if np.any(ii):
            n = np.sum(ii)
            log.warning(f'Setting {n} {col} NaN to 0.0')
            fibermap[col][ii] = 0.0

    return SpectraLite(bands, wave, flux, ivar, mask, resolution_data,
            fibermap, scores=scores)

def update_frame_cache(frames, framekeys, specprod_dir=None):
    '''
    Update a cache of FrameLite objects to match requested frameskeys

    Args:
        frames: dict of FrameLite objects, keyed by (night, expid, camera)
        framekeys: list of desired (night, expid, camera)

    Updates `frames` in-place

    Notes:
        `frames` is dictionary, `framekeys` is list.
        When finished, the keys of `frames` match the entries in `framekeys`
    '''

    log = get_logger()

    #- Drop frames that we no longer need
    ndrop = 0
    for key in list(frames.keys()):
        if key not in framekeys:
            ndrop += 1
            del frames[key]

    nkeep = len(frames)

    #- Read and add the new frames that we do need
    nadd = 0
    for key in framekeys:
        if key not in frames.keys():
            night, expid, camera = key
            framefile = io.findfile('cframe', night, expid, camera,
                    specprod_dir=specprod_dir, readonly=True)
            log.debug('  Reading {}'.format(os.path.basename(framefile)))
            nadd += 1
            frames[key] = FrameLite.read(framefile)

    log.debug('Frame cache: {} kept, {} added, {} dropped, now have {}'.format(
         nkeep, nadd, ndrop, len(frames)))
