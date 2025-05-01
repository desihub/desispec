# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
desispec.io.redrock
===================

I/O routines for reading redrock files
"""

import numpy as np
import fitsio
from astropy.table import Table, vstack

from desispec.util import argmatch
from desispec.io.util import addkeys
from desispec.io.table import read_table
from desispec.io.meta import findfile, get_lastnight
from desispec.io.spectra import determine_specgroup

def read_redrock(filename, fmcols=None):
    """
    Read REDSHIFTS table from filename, optionally including FIBERMAP columns

    Args:
        filename (str): redrock*.fits filepath

    Options:
        fmcols (list of str): columns to include from the FIBERMAP HDU

    Returns zcat Table of redshifts
    """
    with fitsio.FITS(filename, 'r') as fp:
        zcat = read_table(fp, 'REDSHIFTS')

        #- Older productions kept template name/version keywords in HDU 0
        #- instead of the REDSHIFTS HDU; read if needed
        if 'TEMNAM00' not in zcat.meta:
            hdr0 = fp[0].read_header()
            addkeys(zcat.meta, hdr0)

        #- Add FIBERMAP columns if requested
        if fmcols is not None:
            fm = read_table(fp, 'FIBERMAP', columns=fmcols)
            for col in fmcols:
                zcat[col] = fm[col]

    return zcat

def read_redrock_targetcat(tcat, fmcols=None, specprod=None):
    """
    Read and stack multiple redrock files 
    """

    #- Copy structure of tcat (but not data) in case we need to add columns
    tcat = Table(tcat, copy=False)

    #- Convert bytes -> strings for comparison operation consistency
    for col in list(tcat.colnames):
        if tcat[col].dtype.kind == 'S':  #- bytes='S', str='U' (unicode)
            tcat[col] = tcat[col].astype(str)

    #- Add PETAL_LOC = FIBER//500 if needed
    if ('FIBER' in tcat.colnames) and ('PETAL_LOC' not in tcat.colnames):
        tcat['PETAL_LOC'] = tcat['FIBER']//500

    #- Add LASTNIGHT if needed
    if ('TILEID' in tcat.colnames) and ('LASTNIGHT' not in tcat.colnames):
        for tileid in np.unique(tcat['TILEID']):
            ii = tcat['TILEID'] == tileid
            tcat['LASTNIGHT'] = get_lastnight(tileid, specprod=specprod)

    #- healpix or tiles?  Which columns determine target uniqueness?
    specgroup, keycols = determine_specgroup(tcat.colnames)

    zcat_tables = list()
    if specgroup == 'healpix':
        keycols = ['TARGETID', 'SURVEY', 'PROGRAM']
        for tt in tcat.group_by(('HEALPIX', 'SURVEY', 'PROGRAM')).groups:
            healpix = tt['HEALPIX'][0]
            survey = tt['SURVEY'][0]
            program = tt['PROGRAM'][0]
            redrockfile = findfile('redrock', healpix=healpix, survey=survey, faprogram=program, specprod=specprod)
            zcat = read_redrock(redrockfile, fmcols=fmcols)
            keep = np.isin(zcat['TARGETID'], tt['TARGETID'])
            zcat = zcat[keep]
            #- for uniqueness/order bookkeeping later
            zcat['SURVEY'] = survey
            zcat['PROGRAM'] = program
            zcat_tables.append(zcat)
    elif specgroup == 'cumulative':
        assert ('TARGETID' in tcat.colnames) or ('FIBER' in tcat.colnames), f'Target catalog must have TARGETID and/or FIBER; {tcat.colnames=}'

        #- for this use-case, we don't need LASTNIGHT or PETAL_LOC to determine uniqueness
        if 'TARGETID' in tcat.colnames:
            keycols = ['TILEID', 'TARGETID']
        else:
            keycols = ['TILEID', 'FIBER']

        for tt in tcat.group_by(('TILEID', 'LASTNIGHT', 'PETAL_LOC')).groups:
            tileid = tt['TILEID'][0]
            lastnight = tt['LASTNIGHT'][0]
            petal = tt['PETAL_LOC'][0]
            redrockfile = findfile('redrock', tile=tileid, night=lastnight, spectrograph=petal, specprod=specprod)
            zcat = read_redrock(redrockfile, fmcols=fmcols)
            zcat['TILEID'] = tileid
            if 'FIBER' in tt.colnames:
                #- tile-based Redrock files have 500 rows, corresponding to FIBER%500
                rows = tt['FIBER'] % 500
                zcat = zcat[rows]
                if 'FIBER' in zcat.colnames:
                    assert np.all(zcat['FIBER'] == tt['FIBER'])
                else:
                    zcat['FIBER'] = tt['FIBER']
            else:
                keep = np.isin(zcat['TARGETID'], tt['TARGETID'])
                zcat = zcat[keep]

            zcat_tables.append(zcat)
    else:
        raise RuntimeError(f'Unrecognized {specgroup=}')

    #- Combine tables
    zcat = vstack(zcat_tables)

    assert len(zcat) == len(tcat)

    #- Reorder zcat to match input tcat if needed
    need_to_reorder = False
    for col in keycols:
        if np.any(zcat[col] != tcat[col]):
            need_to_reorder = True
            break

    if need_to_reorder:
        ii = argmatch(zcat[keycols], tcat[keycols])
        zcat = zcat[ii]

    #- Drop any bookkeeping columns that weren't part of original request
    for extracol in ['TILEID', 'FIBER', 'SURVEY', 'PROGRAM', 'HEALPIX']:
        if (extracol in zcat.colnames) and ((fmcols is None) or (extracol not in fmcols)):
            zcat.remove_column(extracol)

    return zcat


