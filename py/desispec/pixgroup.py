"""
Tools to regroup spectra in individual exposures by healpix on the sky
"""

from __future__ import absolute_import, division, print_function
import glob, os, sys, time
from collections import Counter

import numpy as np

import fitsio
from astropy.io import fits
import healpy as hp

import desimodel.footprint
from desiutil.log import get_logger
import desiutil.depend

from . import io
from .maskbits import specmask

def get_exp2healpix_map(nights=None, specprod_dir=None, nside=64, comm=None):
    '''
    Returns table with columns NIGHT EXPID SPECTRO HEALPIX NTARGETS

    Options:
        nights: list of YEARMMDD to scan for exposures
        specprod_dir: override $DESI_SPECTRO_REDUX/$SPECPROD
        nside: healpix nside, must be power of 2
        comm: MPI communicator

    Note: This could be replaced by a DB query when the production DB exists.
    '''
    log = get_logger()
    if comm is None:
        rank, size = 0, 1
    else:
        rank, size = comm.rank, comm.size

    if specprod_dir is None:
        specprod_dir = io.specprod_root()

    if nights is None and rank == 0:
        nights = io.get_nights(specprod_dir=specprod_dir)

    if comm:
        nights = comm.bcast(nights, root=0)

    #-----
    #- Distribute nights over ranks, scanning their exposures to build
    #- map of exposures -> healpix

    #- Rows to add to the output table
    rows = list()

    #- for tracking exposures that we've already mapped in a different band
    night_expid_spectro = set()

    for night in nights[rank::size]:
        night = str(night)
        nightdir = os.path.join(specprod_dir, 'exposures', night)
        for expid in io.get_exposures(night, specprod_dir=specprod_dir,
                                      raw=False):
            tmpframe = io.findfile('cframe', night, expid, 'r0',
                                   specprod_dir=specprod_dir)
            expdir = os.path.split(tmpframe)[0]
            cframefiles = sorted(glob.glob(expdir + '/cframe*.fits'))
            for filename in cframefiles:
                #- parse 'path/night/expid/cframe-r0-12345678.fits'
                camera = os.path.basename(filename).split('-')[1]
                channel, spectro = camera[0], int(camera[1])

                #- skip if we already have this expid/spectrograph
                if (night, expid, spectro) in night_expid_spectro:
                    continue
                else:
                    night_expid_spectro.add((night, expid, spectro))

                log.debug('Rank {} mapping {} {}'.format(rank, night,
                    os.path.basename(filename)))
                sys.stdout.flush()

                #- Determine healpix, allowing for NaN
                columns = ['RA_TARGET', 'DEC_TARGET']
                fibermap = fitsio.read(filename, 'FIBERMAP', columns=columns)
                ra, dec = fibermap['RA_TARGET'], fibermap['DEC_TARGET']
                ok = ~np.isnan(ra) & ~np.isnan(dec)
                ra, dec = ra[ok], dec[ok]
                allpix = desimodel.footprint.radec2pix(nside, ra, dec)

                #- Add rows for final output
                for pix, ntargets in sorted(Counter(allpix).items()):
                    rows.append((night, expid, spectro, pix, ntargets))

    #- Collect rows from individual ranks back to rank 0
    if comm:
        rank_rows = comm.gather(rows, root=0)
        if rank == 0:
            rows = list()
            for r in rank_rows:
                rows.extend(r)
        else:
            rows = None

        rows = comm.bcast(rows, root=0)

    #- Create the final output table
    exp2healpix = np.array(rows, dtype=[
        ('NIGHT', 'i4'), ('EXPID', 'i8'), ('SPECTRO', 'i4'),
        ('HEALPIX', 'i8'), ('NTARGETS', 'i8')])

    return exp2healpix

#-----
class FrameLite(object):
    '''
    Lightweight Frame object for regrouping

    This is intended for I/O without the overheads of float32 -> float64
    conversion, correcting endianness, etc.
    '''
    def __init__(self, wave, flux, ivar, mask, rdat, fibermap, header, scores=None):
        '''
        Create a new FrameLite object

        Args:
            wave: 1D array of wavlengths
            flux: 2D[nspec, nwave] fluxes
            ivar: 2D[nspec, nwave] inverse variances of flux
            mask: 2D[nspec, nwave] mask of flux; 0=good
            rdat 3D[nspec, ndiag, nwave] Resolution matrix diagonals
            fibermap: fibermap table
            header: FITS header

        Options:
            scores: table of QA scores
        '''
        self.wave = wave
        self.flux = flux
        self.ivar = ivar
        self.mask = mask
        self.rdat = rdat
        self.fibermap = fibermap
        self.header = header
        self.scores = scores

    def __getitem__(self, index):
        '''Return a subset of the original FrameLight'''
        if not isinstance(index, slice):
            index = np.atleast_1d(index)

        if self.scores:
            scores = self.scores[index]
        else:
            scores = None

        return FrameLite(self.wave, self.flux[index], self.ivar[index],
            self.mask[index], self.rdat[index], self.fibermap[index],
            self.header, scores)

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
            rdat = fx['RESOLUTION'].read()
            fibermap = fx['FIBERMAP'].read()
            if 'SCORES' in fx:
                scores = fx['SCORES'].read()
            else:
                scores = None

        #- Add extra fibermap columns NIGHT, EXPID, TILEID
        nspec = len(fibermap)
        night = np.tile(header['NIGHT'], nspec).astype('i4')
        expid = np.tile(header['EXPID'], nspec).astype('i4')
        tileid = np.tile(header['TILEID'], nspec).astype('i4')
        fibermap = np.lib.recfunctions.append_fields(
            fibermap, ['NIGHT', 'EXPID', 'TILEID'], [night, expid, tileid],
            usemask=False)

        return FrameLite(wave, flux, ivar, mask, rdat, fibermap, header, scores)

#-----
class SpectraLite(object):
    '''
    Lightweight spectra I/O object for regrouping
    '''
    def __init__(self, bands, wave, flux, ivar, mask, rdat, fibermap,
            scores=None):
        '''
        Create a SpectraLite object

        Args:
            bands: list of bands, e.g. ['b', 'r', 'z']
            wave: dict of wavelengths, keyed by band
            flux: dict of fluxes, keyed by band
            ivar: dict of inverse variances, keyed by band
            mask: dict of masks, keyed by band
            rdat: dict of Resolution sparse diagonals, keyed by band
            fibermap: fibermap table, applies to all bands

        Options:
            scores: scores table, applies to all bands
        '''

        self.bands = bands[:]

        #- All inputs should have the same bands
        _bands = set(bands)
        assert set(wave.keys()) == _bands
        assert set(flux.keys()) == _bands
        assert set(ivar.keys()) == _bands
        assert set(mask.keys()) == _bands
        assert set(rdat.keys()) == _bands

        #- All bands should have the same number of spectra
        nspec = len(fibermap)
        for x in bands:
            assert flux[x].shape[0] == nspec
            assert ivar[x].shape[0] == nspec
            assert mask[x].shape[0] == nspec
            assert rdat[x].shape[0] == nspec

            #- Also check wavelength dimension consistency
            nwave = len(wave[x])
            assert flux[x].shape[1] == nwave
            assert ivar[x].shape[1] == nwave
            assert mask[x].shape[1] == nwave
            assert rdat[x].shape[2] == nwave  #- rdat[x].shape[1] is ndiag

        #- scores and fibermap should be row matched with same length
        if scores is not None:
            assert len(scores) == len(fibermap)

        self.wave = wave.copy()
        self.flux = flux.copy()
        self.ivar = ivar.copy()
        self.mask = mask.copy()
        self.rdat = rdat.copy()
        self.fibermap = fibermap
        self.scores = scores

    def __add__(self, other):
        '''
        concatenate two SpectraLite objects into one
        '''
        assert self.bands == other.bands
        for x in self.bands:
            assert np.all(self.wave[x] == other.wave[x])
        if self.scores is not None:
            assert other.scores is not None

        bands = self.bands
        wave = self.wave
        flux = dict()
        ivar = dict()
        mask = dict()
        rdat = dict()
        for x in self.bands:
            flux[x] = np.vstack([self.flux[x], other.flux[x]])
            ivar[x] = np.vstack([self.ivar[x], other.ivar[x]])
            mask[x] = np.vstack([self.mask[x], other.mask[x]])
            rdat[x] = np.vstack([self.rdat[x], other.rdat[x]])

        #- Note: tables use np.hstack not np.vstack
        fibermap = np.hstack([self.fibermap, other.fibermap])
        if self.scores is not None:
            scores = np.hstack([self.scores, other.scores])
        else:
            scores = None

        return SpectraLite(bands, wave, flux, ivar, mask, rdat, fibermap, scores)

    def write(self, filename, header=None):
        '''
        Write this SpectraLite object to `filename`
        '''

        #- create directory if missing
        dirname=os.path.dirname(filename)
        try :
            if not os.path.isdir(dirname) :
                os.makedirs(dirname)
        except FileExistsError :
            pass
        tmpout = filename + '.tmp'

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
        hdus.writeto(tmpout, overwrite=True, checksum=True)

        #- then proceed with more efficient fitsio for everything else
        #- See https://github.com/esheldon/fitsio/issues/150 for why
        #- these are written one-by-one
        if self.scores is not None:
            fitsio.write(tmpout, self.scores, extname='SCORES')
        for x in sorted(self.bands):
            X = x.upper()
            fitsio.write(tmpout, self.wave[x], extname=X+'_WAVELENGTH',
                    header=dict(BUNIT='Angstrom'))
            fitsio.write(tmpout, self.flux[x], extname=X+'_FLUX',
                    header=dict(BUNIT='10**-17 erg/(s cm2 Angstrom)'))
            fitsio.write(tmpout, self.ivar[x], extname=X+'_IVAR',
                header=dict(BUNIT='10**+34 (s2 cm4 Angstrom2) / erg2'))
            fitsio.write(tmpout, self.mask[x], extname=X+'_MASK', compress='gzip')
            fitsio.write(tmpout, self.rdat[x], extname=X+'_RESOLUTION')

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
            rdat = dict()
            fibermap = fx['FIBERMAP'].read()
            if 'SCORES' in fx:
                scores = fx['SCORES'].read()
            else:
                scores = None

            bands = ['b', 'r', 'z']
            for x in bands:
                X = x.upper()
                wave[x] = fx[X+'_WAVELENGTH'].read()
                flux[x] = fx[X+'_FLUX'].read()
                ivar[x] = fx[X+'_IVAR'].read()
                mask[x] = fx[X+'_MASK'].read()
                rdat[x] = fx[X+'_RESOLUTION'].read()

        return SpectraLite(bands, wave, flux, ivar, mask, rdat, fibermap, scores)

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
            ndiag[band] = frame.rdat.shape[1]

    #- Now loop through all frames, filling in any missing bands
    bands = sorted(list(wave.keys()))
    for (night, expid, camera), frame in list(frames.items()):
        band = camera[0]
        spectro = camera[1:]
        for x in bands:
            if x == band:
                continue

            xcam = x+spectro
            if (night, expid, xcam) in frames:
                continue

            log.warning('Creating blank data for missing frame {}'.format(
                (night, expid, xcam)))
            nwave = len(wave[x])
            nspec = frame.flux.shape[0]
            flux = np.zeros((nspec, nwave), dtype='f4')
            ivar = np.zeros((nspec, nwave), dtype='f4')
            mask = np.zeros((nspec, nwave), dtype='u4') + specmask.NODATA
            rdat = np.zeros((nspec, ndiag[x], nwave), dtype='f4')

            #- Copy the header and correct the camera keyword
            header = fitsio.FITSHDR(frame.header)
            header['camera'] = xcam

            #- Make new blank scores, replacing trailing band _B/R/Z
            dtype = list()
            if frame.scores is not None:
                for name in frame.scores.dtype.names:
                    if name.endswith('_'+band.upper()):
                        xname = name[0:-1] + x.upper()
                        dtype.append((xname, type(frame.scores[name][0])))

                scores = np.zeros(nspec, dtype=dtype)
            else:
                scores = None

            #- Add the blank FrameLite object
            frames[(night,expid,xcam)] = FrameLite(
                wave[x], flux, ivar, mask, rdat,
                frame.fibermap, header, scores)

def frames2spectra(frames, pix, nside=64):
    '''
    Combine a dict of FrameLite into a SpectraLite for healpix `pix`

    Args:
        frames: dict of FrameLight, keyed by (night, expid, camera)
        pix: NESTED healpix pixel number

    Options:
        nside: Healpix nside, must be power of 2

    Returns:
        SpectraLite object with subset of spectra from frames that are in
        the requested healpix pixel `pix`
    '''
    wave = dict()
    flux = dict()
    ivar = dict()
    mask = dict()
    rdat = dict()
    fibermap = list()
    scores = dict()

    bands = ['b', 'r', 'z']
    for x in bands:
        #- Select just the frames for this band
        keys = sorted(frames.keys())
        xframes = [frames[k] for k in keys if frames[k].header['CAMERA'].startswith(x)]
        assert len(xframes) != 0

        #- Select flux, ivar, etc. for just the spectra on this healpix
        wave[x] = xframes[0].wave
        flux[x] = list()
        ivar[x] = list()
        mask[x] = list()
        rdat[x] = list()
        scores[x] = list()
        for xf in xframes:
            ra, dec = xf.fibermap['RA_TARGET'], xf.fibermap['DEC_TARGET']
            ok = ~np.isnan(ra) & ~np.isnan(dec)
            ra[~ok] = 0.0
            dec[~ok] = 0.0
            allpix = desimodel.footprint.radec2pix(nside, ra, dec)
            ii = (allpix == pix) & ok
            flux[x].append(xf.flux[ii])
            ivar[x].append(xf.ivar[ii])
            mask[x].append(xf.mask[ii])
            rdat[x].append(xf.rdat[ii])

            if x == bands[0]:
                fibermap.append(xf.fibermap[ii])

            if xf.scores is not None:
                scores[x].append(xf.scores[ii])

        flux[x] = np.vstack(flux[x])

        ivar[x] = np.vstack(ivar[x])
        mask[x] = np.vstack(mask[x])
        rdat[x] = np.vstack(rdat[x])
        if x == bands[0]:
            fibermap = np.hstack(fibermap)

        if len(scores[x]) > 0:
            try:
                scores[x] = np.hstack(scores[x])
            except:
                import IPython; IPython.embed()

    #- Combine scores into a single table
    #- Why doesn't np.vstack work for this? (says invalid type promotion)
    if len(scores[bands[0]]) > 0:
        if len(bands) == 1:
            scores = scores(bands[0])
        else:
            names = list()
            data = list()
            for x in bands[1:]:
                names.extend(scores[x].dtype.names)
                for colname in scores[x].dtype.names:
                    data.append(scores[x][colname])

            scores = np.lib.recfunctions.append_fields(
                    scores[bands[0]], names, data)

    else:
        scores = None

    return SpectraLite(bands, wave, flux, ivar, mask, rdat, fibermap, scores)

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
                    specprod_dir=specprod_dir)
            log.debug('  Reading {}'.format(os.path.basename(framefile)))
            nadd += 1
            frames[key] = FrameLite.read(framefile)

    log.debug('Frame cache: {} kept, {} added, {} dropped, now have {}'.format(
         nkeep, nadd, ndrop, len(frames)))
