# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
desispec.spectra
================

Class for dealing with a group of spectra from multiple bands
and the associated fibermap information.

"""
from __future__ import absolute_import, division, print_function

import os
import re
import warnings
import time
import copy
import numbers

import numpy as np
import astropy.table
from astropy.table import Table
from astropy.units import Unit

from desiutil.depend import add_dependencies
from desiutil.io import encode_table

from .maskbits import specmask
from .resolution import Resolution

class Spectra(object):
    """Represents a grouping of spectra.

    This class contains an "extended" fibermap that has information about
    the night and exposure of each spectrum.  For each band, this class has
    the wavelength grid, flux, ivar, mask, and resolution arrays.

    Parameters
    ----------
    bands : :class:`list`
        List of strings used to identify the bands.
    wave : :class:`dict`
        Dictionary of arrays specifying the wavelength grid.
    flux : :class:`dict`
        Dictionary of arrays specifying the flux for each spectrum.
    ivar : :class:`dict`
        Dictionary of arrays specifying the inverse variance.
    mask : :class:`dict`, optional
        Dictionary of arrays specifying the bitmask.
    resolution_data : :class:`dict`, optional
        Dictionary of arrays specifying the block diagonal resolution matrix.
        The object for each band must be in one of the formats supported
        by the Resolution class constructor.
    fibermap, Table-like, optional
        Extended fibermap to use. If not specified, a fake one is created.
    exp_fibermap, Table-like, optional
        Exposure-specific fibermap columns, which may not apply to a coadd.
    meta : :class:`dict`, optional
        Dictionary of arbitrary properties.
    extra : :class:`dict`, optional
        Optional dictionary of dictionaries containing extra
        floating point arrays.  The top-level is a dictionary over bands
        and each value is a dictionary containing string keys and values
        which are arrays of the same size as the flux array.
    model : :class:`dict`, optional
        Dictionary of arrays specifying the best-fit spectra model.
    single : :class:`bool`, optional
        If ``True``, store flux,ivar,resolution data in memory as single
        precision (np.float32).
    scores :
        QA scores table.
    redshifts :
        best-fit redshift table.
    scores_comments :
        dict[column] = comment to include in output file
    extra_catalog : numpy or astropy Table, optional
        optional table of metadata, rowmatched to fibermap,
        e.g. a redshift catalog for these spectra
    """
    wavelength_unit = Unit('Angstrom')
    flux_density_unit = Unit('10-17 erg cm-2 s-1 AA-1')

    def __init__(self, bands=[], wave={}, flux={}, ivar={}, mask=None,
            resolution_data=None, fibermap=None, exp_fibermap=None,
            meta=None, extra=None, model=None,
            single=False, scores=None, redshifts=None, scores_comments=None,
            extra_catalog=None):

        self._bands = bands
        self._single = single
        self._ftype = np.float64
        if single:
            self._ftype = np.float32

        #- optional "scores" measured from the spectra
        if scores is not None:
            self.scores = Table(scores)
        else:
            self.scores = None

        #- optional comments to document what each score means
        self.scores_comments = scores_comments

        self.meta = None
        if meta is None:
            self.meta = {}
        else:
            self.meta = meta.copy()

        nspec = 0

        # check consistency of input dimensions
        for b in self._bands:
            if wave[b].ndim != 1:
                raise RuntimeError("wavelength array for band {} should have dim == 1".format(b))
            if flux[b].ndim != 2:
                raise RuntimeError("flux array for band {} should have dim == 2".format(b))
            if flux[b].shape[1] != wave[b].shape[0]:
                raise RuntimeError("flux array wavelength dimension for band {} does not match wavelength grid".format(b))
            if nspec is None:
                nspec = flux[b].shape[0]
            if fibermap is not None:
                if len(fibermap) != flux[b].shape[0]:
                    raise RuntimeError("flux array number of spectra for band {} does not match fibermap".format(b))
            if ivar[b].shape != flux[b].shape:
                raise RuntimeError("ivar array dimensions do not match flux for band {}".format(b))
            if model is not None:
                if model[b].shape != flux[b].shape:
                    raise RuntimeError("model array dimensions do not match flux for band {}".format(b))
            if mask is not None:
                if mask[b].shape != flux[b].shape:
                    raise RuntimeError("mask array dimensions do not match flux for band {}".format(b))
                if mask[b].dtype not in (int, np.int64, np.int32, np.uint64, np.uint32):
                    raise RuntimeError("bad mask type {}".format(mask.dtype))
            if resolution_data is not None:
                if resolution_data[b].ndim != 3:
                    raise RuntimeError("resolution array for band {} should have dim == 3".format(b))
                if resolution_data[b].shape[0] != flux[b].shape[0]:
                    raise RuntimeError("resolution array spectrum dimension for band {} does not match flux".format(b))
                if resolution_data[b].shape[2] != wave[b].shape[0]:
                    raise RuntimeError("resolution array wavelength dimension for band {} does not match grid".format(b))
            if extra is not None:
                # if extra[band] exists, its elements should match the shape of the flux array,
                # but allow for some flexibility for missing bands or extra[band] being None.
                try:
                    for ex in extra[b].items():
                        if ex[1].shape != flux[b].shape:
                            raise RuntimeError("extra arrays must have the same shape as the flux array")
                except (KeyError, AttributeError):
                    pass

        if fibermap is not None and extra_catalog is not None:
            if len(fibermap) != len(extra_catalog):
                raise ValueError('fibermap and extra_catalog have different number of entries {} != {}'.format(
                    len(fibermap), len(extra_catalog) ))

            if ('TARGETID' in fibermap.dtype.names) and ('TARGETID' in extra_catalog.dtype.names):
                if not np.all(fibermap['TARGETID'] == extra_catalog['TARGETID']):
                    raise ValueError('TARGETID mismatch between fibermap and extra_catalog')

        # copy data

        if fibermap is not None:
            self.fibermap = fibermap.copy()
        else:
            self.fibermap = None

        if exp_fibermap is not None:
            self.exp_fibermap = exp_fibermap.copy()
        else:
            self.exp_fibermap = None

        if extra_catalog is not None:
            self.extra_catalog = extra_catalog.copy()
        else:
            self.extra_catalog = None
        if redshifts is not None:
            self.redshifts = redshifts.copy()
        else:
            self.redshifts = None

        self.wave = {}
        self.flux = {}
        self.ivar = {}
        
        if model is None:
            self.model = None
        else:
            self.model = {}

        if mask is None:
            self.mask = None
        else:
            self.mask = {}

        if resolution_data is None:
            self.resolution_data = None
            self.R = None
        else:
            self.resolution_data = {}
            self.R = {}

        if extra is None:
            self.extra = None
        else:
            self.extra = {}

        for b in self._bands:
            self.wave[b] = np.copy(wave[b])
            self.flux[b] = np.copy(flux[b].astype(self._ftype))
            self.ivar[b] = np.copy(ivar[b].astype(self._ftype))
            if model is not None:
                self.model[b] = np.copy(model[b].astype(np.float32))
            if mask is not None:
                self.mask[b] = np.copy(mask[b])
            if resolution_data is not None:
                self.resolution_data[b] = resolution_data[b].astype(self._ftype)
                self.R[b] = np.array( [ Resolution(r) for r in resolution_data[b] ] )
            if extra is not None:
                if extra[b] is not None:
                    self.extra[b] = {}
                    for ex in extra[b].items():
                        self.extra[b][ex[0]] = np.copy(ex[1].astype(self._ftype))

    @property
    def bands(self):
        """
        (list): the list of valid bands.
        """
        return self._bands

    @property
    def ftype(self):
        """
        (numpy.dtype): the data type used for floating point numbers.
        """
        return self._ftype


    def wavelength_grid(self, band):
        """
        Return the wavelength grid for a band.

        Args:
            band (str): the name of the band.

        Returns (array):
            an array containing the wavelength values.

        """
        if band not in self.bands:
            raise KeyError("{} is not a valid band".format(band))
        return self.wave[band]


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


    def _get_slice(self, index, bands=None):
        """Slice spectra by index.
        Args:
            bands (list): optional list of bands to select.

        Note: This function is intended to be private,
              to be used by __getitem__() and select().
        """
        if isinstance(index, numbers.Integral):
            index = slice(index, index+1)

        if bands is None:
            bands = copy.copy(self.bands)
        flux = dict()
        ivar = dict()
        wave = dict()
        model = dict() if self.model is not None else None
        mask = dict() if self.mask is not None else None
        rdat = dict() if self.resolution_data is not None else None
        extra = dict() if self.extra is not None else None

        for band in bands:
            flux[band] = self.flux[band][index].copy()
            ivar[band] = self.ivar[band][index].copy()
            wave[band] = self.wave[band].copy()
            if self.model is not None:
                model[band] = self.model[band][index].copy()
            if self.mask is not None:
                mask[band] = self.mask[band][index].copy()
            if self.resolution_data is not None:
                rdat[band] = self.resolution_data[band][index].copy()
            if self.extra is not None:
                extra[band] = dict()
                for col in self.extra[band]:
                    extra[band][col] = self.extra[band][col][index].copy()

        if self.fibermap is not None:
            fibermap = self.fibermap[index].copy()

            exp_fibermap = None
            if self.exp_fibermap is not None:
                j = np.isin(self.exp_fibermap['TARGETID'], fibermap['TARGETID'])
                exp_fibermap = self.exp_fibermap[j].copy()
        else:
            fibermap = None
            exp_fibermap = None

        if self.extra_catalog is not None:
            extra_catalog = self.extra_catalog[index].copy()
        else:
            extra_catalog = None

        if self.scores is not None:
            scores = Table()
            for col in self.scores.dtype.names:
                scores[col] = self.scores[col][index].copy()
        else:
            scores = None

        if self.redshifts is not None:
            redshifts = self.redshifts[index].copy()
        else:
            redshifts = None

        sp = Spectra(bands, wave, flux, ivar,
            mask=mask, resolution_data=rdat,
            fibermap=fibermap, exp_fibermap=exp_fibermap,
            meta=self.meta, extra=extra, model=model, single=self._single,
            scores=scores, redshifts=redshifts, extra_catalog=extra_catalog,
        )
        return sp


    def select(self, nights=None, exposures=None, bands=None, targets=None, fibers=None, invert=False, return_index=False):
        """
        Select a subset of the data.

        This filters the data based on a logical AND of the different
        criteria, optionally inverting that selection.

        Args:
            nights (list): optional list of nights to select.
            exposures (list): optional list of exposures to select.
            bands (list): optional list of bands to select.
            targets (list): optional list of target IDs to select.
            fibers (list): list/array of fiber indices to select.
            invert (bool): after combining all criteria, invert selection.
            return_index (bool): if True, also return the indices of selected spectra.

        Returns:
            spectra: a new Spectra object containing the selected data.
            indices (list, optional): indices of selected spectra. Only provided if return_index is True.
        """
        if bands is None:
            keep_bands = self.bands
        else:
            keep_bands = [ x for x in self.bands if x in bands ]
        if len(keep_bands) == 0:
            raise RuntimeError("no valid bands were selected!")

        keep_rows = np.ones(len(self.fibermap), bool)
        for fm_select,fm_var in zip([nights, exposures, targets, fibers],
                                    ['NIGHT', 'EXPID', 'TARGETID', 'FIBER']):
            if fm_select is not None:
                keep_selection = np.isin(self.fibermap[fm_var], fm_select)
                if sum(keep_selection) == 0:
                    raise RuntimeError("no valid "+fm_var+" were selected!")
                keep_rows = keep_rows & keep_selection

        if invert:
            keep_rows = np.invert(keep_rows)

        keep, = np.where(keep_rows)
        if len(keep) == 0:
            raise RuntimeError("selection has no spectra")

        sp = self._get_slice(keep_rows, bands=keep_bands)

        if return_index:
            return (sp, keep)

        return sp

    def __getitem__(self, index):
        """ Slice spectra by index.
            Index can be an arbitrary slice, for example:
                spectra[0]       #- single index -> slice 0:1
                spectra[0:10]    #- slice
                spectra[ [0,12,56,34] ]  #- list of indices, doesn't have to be sorted
                spectra[ spectra.fibermap['FIBER'] < 20 ]  #- boolean array

        """
        #- __getitem__ differs from _get_slice as it has a single argument
        return self._get_slice(index)

    def __len__(self):
        """Return number of spectra (not number of unique targets)"""
        return len(self.fibermap)

    def update(self, other):
        """
        Overwrite or append new data.

        Given another Spectra object, compare the fibermap information with
        the existing one.  For spectra that already exist, overwrite existing
        data with the new values.  For spectra that do not exist, append that
        data to the end of the spectral data.

        Args:
            other (Spectra): the new data to add.

        Returns:
            nothing (object updated in place).

        Note: if fibermap, scores and extra_catalog exist in the new data, they
        are appended to the existing tables. If those new tables have different columns,
        only columns with identical names will be appended. Spectra.meta is unchanged.
        """

        # Does the other Spectra object have any data?

        if other.num_spectra() == 0:
            return

        # Do we have new bands to add?

        newbands = []
        for b in other.bands:
            if b not in self.bands:
                newbands.append(b)
            else:
                if not np.allclose(self.wave[b], other.wave[b]):
                    raise RuntimeError("band {} has an incompatible wavelength grid".format(b))

        bands = list(self.bands)
        bands.extend(newbands)

        # Are we adding mask data in this update?

        add_mask = False
        if other.mask is None:
            if self.mask is not None:
                raise RuntimeError("existing spectra has a mask, cannot "
                    "update it to a spectra with no mask")
        else:
            if self.mask is None:
                add_mask = True

        # Are we adding redrock model data in this update?

        add_model = False
        if other.model is None:
            if self.model is not None:
                raise RuntimeError("existing spectra has a model, cannot "
                    "update it to a spectra with no model")
        else:
            if self.model is None:
                add_model = True

        # Are we adding resolution data in this update?

        ndiag = {}

        add_res = False
        if other.resolution_data is None:
            if self.resolution_data is not None:
                raise RuntimeError("existing spectra has resolution data, cannot "
                    "update it to a spectra with none")
        else:
            if self.resolution_data is not None:
                for b in self.bands:
                    ndiag[b] = self.resolution_data[b].shape[1]
                for b in other.bands:
                    odiag = other.resolution_data[b].shape[1]
                    if b not in self.bands:
                        ndiag[b] = odiag
                    else:
                        if odiag != ndiag[b]:
                            raise RuntimeError("Resolution matrices for a"
                                " given band must have the same dimensoins")
            else:
                add_res = True
                for b in other.bands:
                    ndiag[b] = other.resolution_data[b].shape[1]

        # Are we adding extra data in this update?

        add_extra = False
        if other.extra is None:
            if self.extra is not None:
                raise RuntimeError("existing spectra has extra data, cannot "
                    "update it to a spectra with none")
        else:
            if self.extra is None:
                add_extra = True

        # Compute which targets / exposures are new

        nother = len(other.fibermap)
        exists = np.zeros(nother, dtype=int)

        indx_original = []

        if ( (self.fibermap is not None) and
            all([x in fm.keys() for x in ['EXPID', 'FIBER']
                                for fm in [self.fibermap, other.fibermap]]) ):
            for r in range(nother):
                expid = other.fibermap[r]["EXPID"]
                fiber = other.fibermap[r]["FIBER"]
                for i, row in enumerate(self.fibermap):
                    if (expid == row["EXPID"]) and (fiber == row["FIBER"]):
                        indx_original.append(i)
                        exists[r] += 1

        if len(np.where(exists > 1)[0]) > 0:
            raise RuntimeError("found duplicate spectra (same EXPID and FIBER) in the fibermap")

        indx_exists = np.where(exists == 1)[0]
        indx_new = np.where(exists == 0)[0]

        # Make new data arrays of the correct size to hold both the old and
        # new data

        nupdate = len(indx_exists)
        nnew = len(indx_new)

        if self.fibermap is None:
            nold = 0
            newfmap = other.fibermap.copy()
        else:
            nold = len(self.fibermap)
            newfmap = encode_table(np.zeros( (nold + nnew, ),
                                   dtype=self.fibermap.dtype))
            if hasattr(self.fibermap, 'meta'):
                newfmap.meta.update(self.fibermap.meta)

        newscores = None
        if self.scores is not None:
            newscores = encode_table(np.zeros( (nold + nnew, ),
                                   dtype=self.scores.dtype))
            if hasattr(self.scores, 'meta'):
                newscores.meta.update(self.scores.meta)

        newredshifts = None
        if self.redshifts is not None:
            newredshifts = encode_table(np.zeros( (nold + nnew, ),
                                   dtype=self.redshifts.dtype))
            if hasattr(self.redshifts, 'meta'):
                newscores.meta.update(self.redshifts.meta)
                
        newextra_catalog = None
        if self.extra_catalog is not None:
            newextra_catalog = encode_table(np.zeros( (nold + nnew, ),
                                   dtype=self.extra_catalog.dtype))
            if hasattr(self.extra_catalog, 'meta'):
                newextra_catalog.meta.update(self.extra_catalog.meta)

        newwave = {}
        newflux = {}
        newivar = {}

        newmask = None
        if add_mask or self.mask is not None:
            newmask = {}

        newmodel = None
        if add_model or self.model is not None:
            newmodel = {}

        newres = None
        newR = None
        if add_res or self.resolution_data is not None:
            newres = {}
            newR = {}

        newextra = None
        if add_extra or self.extra is not None:
            newextra = {}

        for b in bands:
            nwave = None
            if b in self.bands:
                nwave = self.wave[b].shape[0]
                newwave[b] = self.wave[b]
            else:
                nwave = other.wave[b].shape[0]
                newwave[b] = other.wave[b].astype(self._ftype)
            newflux[b] = np.zeros( (nold + nnew, nwave), dtype=self._ftype)
            newivar[b] = np.zeros( (nold + nnew, nwave), dtype=self._ftype)
            if newmodel is not None:
                newmodel[b] = np.zeros( (nold + nnew, nwave), dtype=np.float32)
            if newmask is not None:
                newmask[b] = np.zeros( (nold + nnew, nwave), dtype=np.uint32)
                newmask[b][:,:] = specmask["NODATA"]
            if newres is not None:
                newres[b] = np.zeros( (nold + nnew, ndiag[b], nwave), dtype=self._ftype)
            if newextra is not None:
                newextra[b] = {}

        # Copy the old data

        if nold > 0:
            # We have some data (i.e. we are not starting with an empty Spectra)
            for newtable, original_table in zip([newfmap, newscores, newredshifts, newextra_catalog],
                                           [self.fibermap, self.scores, self.redshifts, self.extra_catalog]):
                if original_table is not None:
                    newtable[:nold] = original_table

            for b in self.bands:
                newflux[b][:nold,:] = self.flux[b]
                newivar[b][:nold,:] = self.ivar[b]
                if self.model is not None:
                    newmodel[b][:nold,:] = self.model[b]
                if self.mask is not None:
                    newmask[b][:nold,:] = self.mask[b]
                elif add_mask:
                    newmask[b][:nold,:] = 0
                if self.resolution_data is not None:
                    newres[b][:nold,:,:] = self.resolution_data[b]
                if self.extra is not None:
                    for ex in self.extra[b].items():
                        newextra[b][ex[0]] = np.zeros( newflux[b].shape,
                            dtype=self._ftype)
                        newextra[b][ex[0]][:nold,:] = ex[1]

        # Update existing spectra

        for i, s in enumerate(indx_exists):
            row = indx_original[i]
            for b in other.bands:
                newflux[b][row,:] = other.flux[b][s,:].astype(self._ftype)
                newivar[b][row,:] = other.ivar[b][s,:].astype(self._ftype)
                if other.model is not None:
                    newmodel[b][row,:] = other.model[b][s,:].astype(np.float32)
                if other.mask is not None:
                    newmask[b][row,:] = other.mask[b][s,:]
                else:
                    newmask[b][row,:] = 0
                if other.resolution_data is not None:
                    newres[b][row,:,:] = other.resolution_data[b][s,:,:].astype(self._ftype)
                if other.extra is not None:
                    for ex in other.extra[b].items():
                        if ex[0] not in newextra[b]:
                            newextra[b][ex[0]] = np.zeros(newflux[b].shape,
                                dtype=self._ftype)
                        newextra[b][ex[0]][row,:] = ex[1][s,:].astype(self._ftype)

        # Append new spectra

        if nnew > 0:
            for newtable, othertable in zip([newfmap, newscores, newredshifts, newextra_catalog],
                                           [other.fibermap, other.scores, other.redshifts, other.extra_catalog]):
                if othertable is not None:
                    if newtable.dtype == othertable.dtype:
                        newtable[nold:] = othertable[indx_new]
                    else:
                    #- if table contents do not match, still merge what we can, based on key names
                    # (possibly with numpy automatic casting)
                        for k in set(newtable.keys()).intersection(set(othertable.keys())):
                            newtable[k][nold:] = othertable[k][indx_new]

            for b in other.bands:
                newflux[b][nold:,:] = other.flux[b][indx_new].astype(self._ftype)
                newivar[b][nold:,:] = other.ivar[b][indx_new].astype(self._ftype)
                if other.model is not None:
                    newmodel[b][nold:,:] = other.model[b][indx_new].astype(np.float32)
                if other.mask is not None:
                    newmask[b][nold:,:] = other.mask[b][indx_new]
                else:
                    newmask[b][nold:,:] = 0
                if other.resolution_data is not None:
                    newres[b][nold:,:,:] = other.resolution_data[b][indx_new].astype(self._ftype)
                if other.extra is not None:
                    for ex in other.extra[b].items():
                        if ex[0] not in newextra[b]:
                            newextra[b][ex[0]] = np.zeros(newflux[b].shape,
                                dtype=self._ftype)
                        newextra[b][ex[0]][nold:,:] = ex[1][indx_new].astype(self._ftype)

        # Update all sparse resolution matrices

        for b in bands:
            if newres is not None:
                newR[b] = np.array( [ Resolution(r) for r in newres[b] ] )

        # Swap data into place

        self._bands = bands
        self.wave = newwave
        self.fibermap = newfmap
        self.flux = newflux
        self.ivar = newivar
        self.mask = newmask
        self.model = newmodel
        self.resolution_data = newres
        self.R = newR
        self.extra = newextra
        self.scores = newscores
        self.redshifts = newredshifts
        self.extra_catalog = newextra_catalog

        return

    def to_specutils(self):
        """Convert to ``specutils`` objects.

        Returns
        -------
        specutils.SpectrumList
            A list with each band represented as a specutils.Spectrum1D object.

        Raises
        ------
        NameError
            If ``specutils`` is not available in the environment.
        """
        #- only import specutils if needed for faster module import
        from specutils import SpectrumList, Spectrum1D
        from astropy.nddata import InverseVariance

        sl = SpectrumList()
        for i, band in enumerate(self.bands):
            meta = {'band': band}
            spectral_axis = self.wave[band].copy() * self.wavelength_unit
            flux = self.flux[band].copy() * self.flux_density_unit
            uncertainty = InverseVariance(self.ivar[band].copy() * (self.flux_density_unit**-2))
            mask = (self.mask[band] != 0).copy()
            meta['int_mask'] = self.mask[band].copy()
            meta['resolution_data'] = self.resolution_data[band].copy()
            try:
                meta['extra'] = self.extra[band].copy()
            except (KeyError, TypeError, AttributeError):
                pass
            if i == 0:
                #
                # Only add these to the first item in the list.
                #
                meta['bands'] = self.bands
                meta['single'] = self._single
                if self.meta:
                    meta['desi_meta'] = self.meta.copy()
                for key in ('fibermap', 'exp_fibermap',
                            'scores', 'scores_comments', 'extra_catalog'):
                    try:
                        meta[key] = getattr(self, key).copy()
                    except AttributeError:
                        pass
            sl.append(Spectrum1D(flux=flux, spectral_axis=spectral_axis,
                                 uncertainty=uncertainty, mask=mask, meta=meta))
        return sl

    @classmethod
    def from_specutils(cls, spectra):
        """Convert ``specutils`` objects to a :class:`~desispec.spectra.Spectra` object.

        Parameters
        ----------
        spectra : specutils.Spectrum1D or specutils.SpectrumList
            A ``specutils`` object.

        Returns
        -------
        :class:`~desispec.spectra.Spectra`
            The corresponding DESI-internal object.

        Raises
        ------
        NameError
            If ``specutils`` is not available in the environment.
        ValueError
            If an unknown type is found in `spectra`.
        """
        #- only import specutils if needed for faster module import
        from specutils import SpectrumList, Spectrum1D
        from astropy.nddata import InverseVariance, StdDevUncertainty

        if isinstance(spectra, SpectrumList):
            sl = spectra
            try:
                bands = sl[0].meta['bands']
            except KeyError:
                raise ValueError("Band details not supplied. Bands should be specified with, e.g.: spectra[0].meta['bands'] = ['b', 'r', 'z'].")
        elif isinstance(spectra, Spectrum1D):
            sl = [spectra]
            #
            # Assume this is a coadd across cameras.
            #
            try:
                bands = sl[0].meta['bands']
            except KeyError:
                bands = ['brz']
        else:
            raise ValueError("Unknown type input to Spectra.from_specutils!")
        #
        # Load objects that are independent of band from the first item.
        #
        single = sl[0].meta.get('single', False)
        meta = fibermap = exp_fibermap = scores = scores_comments = extra_catalog = None
        try:
            meta = sl[0].meta['desi_meta'].copy()
        except (KeyError, AttributeError):
            pass
        try:
            fibermap = sl[0].meta['fibermap'].copy()
        except (KeyError, AttributeError):
            pass
        try:
            exp_fibermap = sl[0].meta['exp_fibermap'].copy()
        except (KeyError, AttributeError):
            pass
        try:
            scores = sl[0].meta['scores'].copy()
        except (KeyError, AttributeError):
            pass
        try:
            scores_comments = sl[0].meta['scores_comments'].copy()
        except (KeyError, AttributeError):
            pass
        try:
            extra_catalog = sl[0].meta['extra_catalog'].copy()
        except (KeyError, AttributeError):
            pass
        #
        # Load band-dependent quantities.
        #
        wave = dict()
        flux = dict()
        ivar = dict()
        mask = dict()
        model = None
        resolution_data = None
        extra = None
        redshifts = None
        for i, band in enumerate(bands):
            wave[band] = sl[i].spectral_axis.to(cls.wavelength_unit).value.copy()
            flux[band] = sl[i].flux.to(cls.flux_density_unit).value.copy()
            if isinstance(sl[i].uncertainty, InverseVariance):
                ivar[band] = sl[i].uncertainty.quantity.to(cls.flux_density_unit**-2).value.copy()
            elif isinstance(sl[i].uncertainty, StdDevUncertainty):
                # Future: may need np.isfinite() here?
                ivar[band] = (sl[i].uncertainty.quantity.to(cls.flux_density_unit).value.copy())**-2
            else:
                raise ValueError("Unknown uncertainty type!")
            try:
                mask[band] = sl[i].meta['int_mask'].copy()
            except KeyError:
                try:
                    mask[band] = sl.mask.astype(np.int32).copy()
                except AttributeError:
                    mask[band] = np.zeros(flux[band].shape, dtype=np.int32)
            if 'resolution_data' in sl[i].meta:
                if resolution_data is None:
                    resolution_data = {band: sl[i].meta['resolution_data'].copy()}
                else:
                    resolution_data[band] = sl[i].meta['resolution_data'].copy()
            if 'model' in sl[i].meta:
                if model is None:
                    model = {band: sl[i].meta['model'].copy()}
                else:
                    model[band] = sl[i].meta['model'].copy()
            if 'extra' in sl[i].meta:
                if extra is None:
                    extra = {band: sl[i].meta['extra'].copy()}
                else:
                    extra[band] = sl[i].meta['extra'].copy()
        return cls(bands=bands, wave=wave, flux=flux, ivar=ivar, model=model, mask=mask,
                   resolution_data=resolution_data, fibermap=fibermap, exp_fibermap=exp_fibermap,
                   meta=meta, extra=extra, single=single, scores=scores, redshifts=redshifts,
                   scores_comments=scores_comments, extra_catalog=extra_catalog)

def _is_multitile(headers):
    """
    If headers contain more than one TILEID, return True, otherwise False

    Args:
        headers: list of dict-like objects

    Returns: True if more than one TILEID present, otherwise False

    Note: if none of the headers have TILEID, also return False
    """
    tileids = list()
    for hdr in headers:
        if 'TILEID' in hdr:
            tileids.append(hdr['TILEID'])

    if len(tileids)>0 and len(np.unique(tileids))>1:
        return True
    else:
        return False

def _remove_tile_keywords(headers):
    """
    Remove tile-specific keywords from headers

    Args:
        headers: list of dict-like objects

    Note: modified input headers in-place
    """
    tile_keywords = ['TILEID', 'TILERA', 'TILEDEC', 'FIELDROT', 'FA_RUN', 'FA_HA',
                     'NOWTIME', 'SVNDM', 'SVNMTL', 'REQRA', 'REQDEC',
                     'PMTIME', 'RUNDATE', 'FAARGS', 'MTLTIME', 'EBVFAC']

    for hdr in headers:
        for key in tile_keywords:
            if key in hdr:
                del hdr[key]

def _stack_fibermaps(fibermaps):
    """
    Stack fibermaps while handling meta keyword merging and numpy vs. Table

    Args:
        fibermaps: astropy Table or numpy structured arrays

    Returns: stacked fibermap

    Note: all fibermaps must be the same type with same columns
    """
    if isinstance(fibermaps[0], np.ndarray):
        #- note named arrays need hstack not vstack
        fibermap = np.hstack(fibermaps)
    else:
        if isinstance(fibermaps[0], astropy.table.Table):
            #- copy tables so that we can update .meta, but ok to not copy underlying data
            fibermaps = [fm.copy(copy_data=False) for fm in fibermaps]
            headers = [fm.meta for fm in fibermaps]
            if _is_multitile(headers):
                _remove_tile_keywords(headers)

            fibermap = astropy.table.vstack(fibermaps)
        else:
            raise ValueError("Can't stack fibermaps of type {}".format(
                type(fibermaps[0])))

    return fibermap

def stack(speclist):
    """
    Stack a list of spectra, return a new spectra object

    Args:
        speclist : list of Spectra objects

    returns stacked Spectra object

    Note: all input spectra must have the same bands, wavelength grid,
    and include or not the same optional elements (mask, fibermap, extra, ...).
    The returned Spectra have the meta from the first input Spectra.

    Also see Spectra.update, which is less efficient but more flexible for
    handling heterogeneous inputs
    """
    flux = dict()
    ivar = dict()
    wave = dict()
    bands = copy.copy(speclist[0].bands)
    for band in bands:
        flux[band] = np.vstack([sp.flux[band] for sp in speclist])
        ivar[band] = np.vstack([sp.ivar[band] for sp in speclist])
        wave[band] = speclist[0].wave[band].copy()
    
    if speclist[0].model is not None:
        model = dict()
        for band in bands:
            model[band] = np.vstack([sp.model[band] for sp in speclist])
    else:
        model = None

    if speclist[0].mask is not None:
        mask = dict()
        for band in bands:
            mask[band] = np.vstack([sp.mask[band] for sp in speclist])
    else:
        mask = None

    if speclist[0].resolution_data is not None:
        rdat = dict()
        for band in bands:
            rdat[band] = np.vstack([sp.resolution_data[band] for sp in speclist])
    else:
        rdat = None

    if speclist[0].fibermap is not None:
        fibermap = _stack_fibermaps([sp.fibermap for sp in speclist])
    else:
        fibermap = None

    if speclist[0].exp_fibermap is not None:
        exp_fibermap = _stack_fibermaps([sp.exp_fibermap for sp in speclist])
    else:
        exp_fibermap = None

    if speclist[0].extra_catalog is not None:
        extra_catalog = _stack_fibermaps([sp.extra_catalog for sp in speclist])
    else:
        extra_catalog = None

    if speclist[0].extra is not None:
        extra = dict()
        for band in bands:
            extra[band] = dict()
            for col in speclist[0].extra[band]:
                extra[band][col] = np.concatenate([sp.extra[band][col] for sp in speclist])
    else:
        extra = None

    if speclist[0].scores is not None:
        scores = Table()
        for col in speclist[0].scores.dtype.names:
            scores[col] = np.concatenate([sp.scores[col] for sp in speclist])
    else:
        scores = None

    if speclist[0].redshifts is not None:
        redshifts = _stack_fibermaps([sp.redshifts for sp in speclist])
    else:
        redshifts = None

    headers = [sp.meta.copy() for sp in speclist]
    if _is_multitile(headers):
        _remove_tile_keywords(headers)

    sp = Spectra(bands, wave, flux, ivar,
        mask=mask, resolution_data=rdat,
        fibermap=fibermap, exp_fibermap=exp_fibermap,
        meta=headers[0], extra=extra, model=model, scores=scores,
        redshifts=redshifts, extra_catalog=extra_catalog,
    )
    return sp
