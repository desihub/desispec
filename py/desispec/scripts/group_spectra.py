"""
Regroup spectra by healpix
"""

from __future__ import absolute_import, division, print_function
import glob, os, time
from collections import Counter

import numpy as np

import fitsio
import healpy as hp

import desimodel.footprint

from desispec import io

def get_exp2healpix_map(nights=None, specprod_dir=None, nside=64, comm=None):
    '''
    Returns table NIGHT EXPID SPECTRO HEALPIX NTARGETS 
    
    This could be replaced by a DB query when the production DB exists.
    '''
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
    
    #- Loop over cframe files and build mapping
    rows = list()
    night_expid_spectro = set()
    for night in nights[rank::size]:
        nightdir = os.path.join(specprod_dir, 'exposures', night)
        for expid in io.get_exposures(night, specprod_dir=specprod_dir, raw=False):
            tmpframe = io.findfile('cframe', night, expid, 'r0', specprod_dir=specprod_dir)
            expdir = os.path.split(tmpframe)[0]
            cframefiles = sorted(glob.glob(expdir + '/cframe*.fits'))
            for filename in cframefiles:
                #- parse 'path/night/expid/cframe-r0-12345678.fits'
                camera = os.path.basename(filename).split('-')[1]
                channel, spectro = camera[0], int(camera[1])
            
                #- if we already have don't this expid/spectrograph, skip
                if (night, expid, spectro) in night_expid_spectro:
                    continue

                # print('Mapping {}'.format(os.path.basename(filename)))
                night_expid_spectro.add((night, expid, spectro))
            
                columns = ['RA_TARGET', 'DEC_TARGET']
                fibermap = fitsio.read(filename, 'FIBERMAP', columns=columns)
                ra, dec = fibermap['RA_TARGET'], fibermap['DEC_TARGET']
                ok = ~np.isnan(ra) & ~np.isnan(dec)
                ra, dec = ra[ok], dec[ok]
                allpix = desimodel.footprint.radec2pix(nside, ra, dec)
            
                for pix, ntargets in sorted(Counter(allpix).items()):
                    rows.append((night, expid, spectro, pix, ntargets))
    
    if comm:
        rank_rows = comm.gather(rows, root=0)
        rows = list()
        for r in rank_rows:
            rows.extend(r)
    
    exp2healpix = np.array(rows, dtype=[
        ('NIGHT', 'i4'), ('EXPID', 'i8'), ('SPECTRO', 'i4'),
        ('HEALPIX', 'i8'), ('NTARGETS', 'i8')])

    return exp2healpix

#-----
class FrameLite(object):
    '''Lightweight Frame object for regrouping'''
    def __init__(self, wave, flux, ivar, mask, rdat, fibermap, header, scores=None):
        """TODO: document"""
        self.wave = wave
        self.flux = flux
        self.ivar = ivar
        self.mask = mask
        self.rdat = rdat
        self.fibermap = fibermap
        self.header = header
        self.scores = scores
    
    def __getitem__(self, index):
        '''slice frame by targets...'''
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

        #- Add extra fibermap columns
        nspec = len(fibermap)
        night = np.tile(header['NIGHT'], nspec).astype('i4')
        expid = np.tile(header['EXPID'], nspec).astype('i8')
        tileid = np.tile(header['TILEID'], nspec).astype('i8')
        fibermap = np.lib.recfunctions.append_fields(
            fibermap, ['NIGHT', 'EXPID', 'TILEID'], [night, expid, tileid],
            usemask=False)

        return FrameLite(wave, flux, ivar, mask, rdat, fibermap, header, scores)

class SpectraLite(object):
    def __init__(self, bands=[], wave={}, flux={}, ivar={}, mask={}, rdat={},
                 fibermap=None, scores=None):
        self.bands = bands.copy()
        
        _bands = set(bands)
        assert set(wave.keys()) == _bands
        assert set(flux.keys()) == _bands
        assert set(ivar.keys()) == _bands
        assert set(mask.keys()) == _bands
        assert set(rdat.keys()) == _bands
        
        self.wave = wave.copy()
        self.flux = flux.copy()
        self.ivar = ivar.copy()
        self.mask = mask.copy()
        self.rdat = rdat.copy()
        self.fibermap = fibermap
        self.scores = scores
    
    def __add__(self, other):
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
        
        fibermap = np.hstack([self.fibermap, other.fibermap])
        if self.scores:
            scores = np.hstack([self.scores, other.scores])
        
        return SpectraLite(bands, wave, flux, ivar, mask, rdat, fibermap, scores)

    def write(self, filename):
        with fitsio.FITS(filename, mode='rw', clobber=True) as fx:
            fx.write(self.fibermap, extname='FIBERMAP')
            for x in sorted(self.bands):
                X = x.upper()
                fx.write(self.wave[x], extname=X+'_WAVELENGTH')
                fx.write(self.flux[x], extname=X+'_FLUX')
                fx.write(self.ivar[x], extname=X+'_IVAR')
                fx.write(self.mask[x], extname=X+'_MASK')
                fx.write(self.rdat[x], extname=X+'_RESOLUTION')


def add_missing_frames(frames):
    return frames

def frames2spectra(frames, pix, nside=64):
    '''
    frames[(night, expid, camera)] = FrameLite object for spectra
    that are in healpix `pix`
    '''
    bands = ['b', 'r', 'z']
    wave = dict()
    flux = dict()
    ivar = dict()
    mask = dict()
    rdat = dict()
    fibermap = list()
    scores = dict()

    for x in bands:
        keys = sorted(frames.keys())
        xframes = [frames[k] for k in keys if frames[k].header['CAMERA'].startswith(x)]
        assert len(xframes) != 0

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
            scores[x] = np.hstack(scores[x])

    if len(scores[bands[0]]) > 0:
        scores = hp.vstack([scores[x] for x in bands])
    else:
        scores = None

    return SpectraLite(bands, wave, flux, ivar, mask, rdat, fibermap, scores)

def update_frame_cache(frames, framekeys):
    '''
    TODO: document
    frames[(night, expid, camera)] dict of FrameLight objects
    framekeys list of (night,expid,camera) wanted
    '''

    ndrop = 0
    for key in list(frames.keys()):
        if key not in framekeys:
            ndrop += 1
            del frames[key]

    nkeep = len(frames)

    nadd = 0
    for key in framekeys:
        if key not in frames.keys():
            night, expid, camera = key
            framefile = io.findfile('cframe', night, expid, camera)
            # print('  Reading {}'.format(os.path.basename(framefile)))
            nadd += 1
            frames[key] = FrameLite.read(framefile)

    print('Frame cache: {} kept, {} added, {} dropped, now have {}'.format(
        nkeep, nadd, ndrop, len(frames)))

#-------------------------------------------------------------------------
if __name__ == '__main__':

    #- TODO: argparse

    #- Get table NIGHT EXPID SPECTRO HEALPIX NTARGETS 
    exp2pix = get_exp2healpix_map()
    assert len(exp2pix) > 0

    #- TODO: evenly distribute pixels across MPI ranks

    frames = dict()
    for pix in sorted(set(exp2pix['HEALPIX'])):
        iipix = np.where(exp2pix['HEALPIX'] == pix)[0]
        ntargets = np.sum(exp2pix['NTARGETS'][iipix])
        print('pix {} with {} targets on {} spectrograph exposures'.format(
            pix, ntargets, len(iipix)))
        framekeys = list()
        for i in iipix:
            night = exp2pix['NIGHT'][i]
            expid = exp2pix['EXPID'][i]
            spectro = exp2pix['SPECTRO'][i]
            for band in ['b', 'r', 'z']:
                camera = band + str(spectro)
                framekeys.append((night, expid, camera))

        update_frame_cache(frames, framekeys)

        #- TODO: add support for missing frames
        frames = add_missing_frames(frames)
        
        spectra = frames2spectra(frames, pix)
        specfile = io.findfile('spectra', nside=64, groupname=pix)
        spectra.write(os.path.basename(specfile))
    
        #--- DEBUG ---
        # import IPython
        # IPython.embed()
        #--- DEBUG ---
    
    
