#!/usr/bin/env python

"""
Combine individual redrock files into a single zcatalog

Stephen Bailey
Lawrence Berkeley National Lab
Fall 2015

substantially updated Fall 2023
"""

from __future__ import absolute_import, division, print_function

import sys, os, glob, time
import itertools
import argparse
import importlib.resources
import multiprocessing as mp

import numpy as np
from numpy.lib.recfunctions import append_fields

import fitsio
from astropy.table import Table, hstack, vstack

from desiutil.log import get_logger
from desispec import io
from desispec.zcatalog import find_primary_spectra
from desispec.io.util import get_tempfilename, checkgzip, replace_prefix, write_bintable
from desispec.io.table import read_table
from desispec.util import parse_keyval
from desiutil.annotate import load_csv_units
import desiutil.depend
from desispec import validredshifts

def load_sv1_ivar_w12(hpix, targetids):
    """
    Load FLUX_IVAR_W1/W2 from sv1 target files for requested targetids

    Args:
        hpix (int): nside=8 nested healpix
        targetids (array): TARGETIDs to include

    Returns table of TARGETID, FLUX_IVAR_W1, FLUX_IVAR_W2

    Note: this is only for the special case of sv1 dark/bright and the
    FLUX_IVAR_W1/W2 columns which were not included in fiberassign for
    tiles designed before 20201212.

    Note: nside=8 nested healpix is hardcodes for simplicity because that is
    what was used for sv1 target selection and this is not trying to be a
    more generic targetid lookup function.
    """
    log = get_logger()
    #- the targets could come from any version of desitarget, so search all,
    #- but once a TARGETID is found it will be the same answer (for FLUX_IVAR*)
    #- as any other version because it is propagated from the same dr9 input
    #- Tractor files.
    targetdir = os.path.join(os.environ['DESI_TARGET'], 'catalogs', 'dr9')
    fileglob = f'{targetdir}/*/targets/sv1/resolve/*/sv1targets-*-hp-{hpix}.fits'
    sv1targetfiles = sorted(glob.glob(fileglob))
    nfiles = len(sv1targetfiles)
    ntarg = len(np.unique(targetids))
    log.info(f'Searching {nfiles} sv1 target files for {ntarg} targets in nside=8 healpix={hpix}')
    columns = ['TARGETID', 'FLUX_IVAR_W1', 'FLUX_IVAR_W2']
    targets = list()
    found_targetids = list()
    for filename in sv1targetfiles:
        tx = fitsio.read(filename, 1, columns=columns)
        keep = np.isin(tx['TARGETID'], targetids)
        keep &= ~np.isin(tx['TARGETID'], found_targetids)
        targets.append(tx[keep])
        found_targetids.extend(tx['TARGETID'][keep])

        if np.all(np.isin(targetids, found_targetids)):
            break

    targets = np.hstack(targets)

    missing = np.isin(targetids, targets['TARGETID'], invert=True)
    if np.any(missing):
        nmissing = np.sum(missing)
        log.error(f'{nmissing} TARGETIDs not found in sv1 healpix={hpix}')

    return targets

def _wrap_read_redrock(optdict):
    """read_redrock wrapper to expand dictionary of named args for multiprocessing"""
    return read_redrock(**optdict)

def read_redrock(rrfile, group=None, pertile=False, counter=None):
    """
    Read a Redrock file, combining REDSHIFTS, FIBERMAP, and TSNR2 HDUs

    Args:
        rrfile (str): full path to redrock filename

    Options:
        group (str): add group-specific columns for cumulative, pernight, healpix
        pertile (bool): input Redrock file is single tile (not healpix)
        counter (tuple): (i,n) log loading ith file out of n

    Returns (zcat, expfibermap) where zcat is a join of the redrock REDSHIFTS
    catalog and the coadded FIBERMAP
    """
    log = get_logger()
    if counter is not None:
        i, n = counter
        log.info(f'Reading {i}/{n} {rrfile}')
    else:
        log.info(f'Reading {rrfile}')

    with fitsio.FITS(rrfile) as fx:
        hdr = fx[0].read_header()
        if group is not None and 'SPGRP' in hdr and \
                hdr['SPGRP'] != group:
            log.warning("Skipping {} with SPGRP {} != group {}".format(
                rrfile, hdr['SPGRP'], group))
            return None

        redshifts = Table(fx['REDSHIFTS'].read())

        fibermap = Table(fx['FIBERMAP'].read())
        expfibermap = Table(fx['EXP_FIBERMAP'].read())

        tsnr2 = Table(fx['TSNR2'].read())

    emline_file = rrfile.replace('/redrock-', '/emline-')
    qso_mgii_file = rrfile.replace('/redrock-', '/qso_mgii-')
    qso_qn_file = rrfile.replace('/redrock-', '/qso_qn-')

    with fitsio.FITS(emline_file) as fx:
        emline = Table(fx['EMLINEFIT'].read())
    with fitsio.FITS(qso_mgii_file) as fx:
        qso_mgii = Table(fx['MGII'].read())
    with fitsio.FITS(qso_qn_file) as fx:
        qso_qn = Table(fx['QN_RR'].read())

    assert np.all(redshifts['TARGETID'] == fibermap['TARGETID'])
    assert np.all(redshifts['TARGETID'] == tsnr2['TARGETID'])
    assert np.all(redshifts['TARGETID'] == emline['TARGETID'])
    assert np.all(redshifts['TARGETID'] == qso_mgii['TARGETID'])
    assert np.all(redshifts['TARGETID'] == qso_qn['TARGETID'])

    fmcols = list(fibermap.dtype.names)
    fmcols.remove('TARGETID')

    emline_cols = ['OII_FLUX', 'OII_FLUX_IVAR']
    qso_mgii_cols = ['IS_QSO_MGII']
    qso_qn_cols = ['Z_NEW', 'ZERR_NEW', 'IS_QSO_QN_NEW_RR', 'C_LYA', 'C_CIV', 'C_CIII', 'C_MgII', 'C_Hbeta', 'C_Halpha']

    tsnr2cols = list(tsnr2.dtype.names)
    tsnr2cols.remove('TARGETID')
    data = hstack([
        redshifts,
        fibermap[fmcols],
        tsnr2[tsnr2cols],
        emline[emline_cols],
        qso_mgii[qso_mgii_cols],
        qso_qn[qso_qn_cols]
        ], join_type='exact')

    #- Add group specific columns, recognizing some some of them may
    #- have already been inherited from the fibermap.
    #- Put these columns right after TARGETID
    nrows = len(data)
    icol = 1
    if group in ('perexp', 'pernight', 'cumulative'):
        if 'TILEID' not in data.colnames:
            data.add_column(np.full(nrows, hdr['TILEID'], dtype=np.int32),
                    index=icol, name='TILEID')
            icol += 1
        if 'PETAL_LOC' not in data.colnames:
            data.add_column(np.full(nrows, hdr['PETAL'], dtype=np.int16),
                    index=icol, name='PETAL_LOC')
            icol += 1

    if group == 'perexp':
        data.add_column(np.full(nrows, hdr['NIGHT'], dtype=np.int32),
                index=icol, name='NIGHT')
        icol += 1
        data.add_column(np.full(nrows, hdr['EXPID'], dtype=np.int32),
                index=icol, name='EXPID')
    elif group == 'pernight':
        data.add_column(np.full(nrows, hdr['NIGHT'], dtype=np.int32),
                index=icol, name='NIGHT')
    elif group == 'cumulative':
        if 'LASTNIGHT' not in data.colnames:
            try:
                lastnight = int(hdr['NIGHT'])
            except KeyError:
                # Some daily reductions do not have this set, use the filename.
                log.warning(f'NIGHT keyword missing from {rrfile}!')
                lastnight = int(rrfile.split('-')[-1].split('.')[0].replace('thru', ''))
            data.add_column(np.full(nrows, lastnight, dtype=np.int32),
                    index=icol, name='LASTNIGHT')
    elif group == 'healpix':
        data.add_column(np.full(nrows, hdr['HPXPIXEL'], dtype=np.int32),
                index=icol, name='HEALPIX')

    icol += 1

    # SPGRPVAL = night for pernight, expid for perexp, subset for custom coadds
    if 'SPGRPVAL' in hdr.keys():
        val = hdr['SPGRPVAL']
        # if int, try to make int32, otherwise let numpy pick dtype
        if isinstance(val, int):
            if np.int32(val) == val:
                dtype = np.int32
            else:
                dtype = np.int64
        else:
            dtype = None
    else:
        log.warning(f'SPGRPVAL keyword missing from {rrfile}')
        if group == 'cumulative':
            val = lastnight
            dtype = np.int32
        else:
            # This is temporary. The whole section above could do with some refactoring.
            raise NotImplementedError(f'No method to reconstruct SPGRPVAL!')
    data.add_column(np.full(nrows, val, dtype=dtype),
            index=icol, name='SPGRPVAL')

    return data, expfibermap


#--------------------------------------------------------------------------

def parse(options=None):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--indir",  type=str,
            help="input directory")
    parser.add_argument("-o", "--outfile",type=str,
            help="output file")
    parser.add_argument("-t", "--tiles", type=str,
            help="ascii file with tileids to include (one per line)")

    parser.add_argument("--survey", type=str,
            help="DESI survey, e.g. sv1, sv3, main")
    parser.add_argument("--program", type=str,
            help="DESI program, e.g bright, dark")

    parser.add_argument("-g", "--group", type=str,
            help="Add columns specific to this spectral grouping "
                 "e.g. pernight adds NIGHT column from input header keyword")
    parser.add_argument("--header", type=str, nargs="*",
            help="KEYWORD=VALUE entries to add to the output header")
    parser.add_argument('--patch-missing-ivar-w12', action='store_true',
            help="Use target files to patch missing FLUX_IVAR_W1/W2 values")
    parser.add_argument('--add-units', action='store_true',
            help="Add units to output catalog from desidatamodel "
                 "column descriptions")
    parser.add_argument('--nproc', type=int, default=1,
            help="Number of multiprocessing processes to use")

    args = parser.parse_args(options)

    return args


def main(args=None):

    if not isinstance(args, argparse.Namespace):
        args = parse(options=args)

    log=get_logger()

    if args.outfile is None:
        args.outfile = io.findfile('zcatalog')

    #- If adding units, check dependencies before doing a lot of work
    if args.add_units:
        try:
            import desidatamodel
        except ImportError:
            log.critical('Unable to import desidatamodel, required to add units (try "module load desidatamodel" first)')
            sys.exit(1)

    if args.indir:
        indir = args.indir
        redrockfiles = sorted(io.iterfiles(f'{indir}', prefix='redrock', suffix='.fits'))
        pertile = (args.group != 'healpix')  # assume tile-based input unless explicitely healpix
    elif args.group == 'healpix':
        pertile = False
        survey = args.survey if args.survey is not None else "*"
        program = args.program if args.program is not None else "*"
        indir = os.path.join(io.specprod_root(), 'healpix')

        #- special case for NERSC; use read-only mount regardless of $DESI_SPECTRO_REDUX
        if indir.startswith('/global/cfs/cdirs'):
            indir = indir.replace('/global/cfs/cdirs', '/dvs_ro/cfs/cdirs')

        #- specprod/healpix/SURVEY/PROGRAM/HPIXGROUP/HPIX/redrock*.fits
        globstr = os.path.join(indir, survey, program, '*', '*', 'redrock*.fits')
        log.info(f'Looking for healpix redrock files in {globstr}')
        redrockfiles = sorted(glob.glob(globstr))
    else:
        pertile = True
        tilefile = args.tiles if args.tiles is not None else io.findfile('tiles')
        indir = os.path.join(io.specprod_root(), 'tiles', args.group)

        #- special case for NERSC; use read-only mount regardless of $DESI_SPECTRO_REDUX
        if indir.startswith('/global/cfs/cdirs'):
            indir = indir.replace('/global/cfs/cdirs', '/dvs_ro/cfs/cdirs')

        log.info(f'Loading tiles from {tilefile}')
        tiles = Table.read(tilefile)
        if args.survey is not None:
            keep = tiles['SURVEY'] == args.survey
            tiles = tiles[keep]
            if len(tiles) == 0:
                log.critical(f'No tiles kept after filtering by SURVEY={args.survey}')
                sys.exit(1)

        if args.program is not None:
            keep = tiles['PROGRAM'] == args.program
            tiles = tiles[keep]
            if len(tiles) == 0:
                log.critical(f'No tiles kept after filtering by PROGRAM={args.program}')
                sys.exit(1)

        tileids = tiles['TILEID']
        log.info(f'Searching disk for redrock*.fits files from {len(tileids)} tiles')

        redrockfiles = list()
        t0 = t1 = time.time()
        for i, tileid in enumerate(tileids):
            if i%500 == 0:
                tnow = time.time()
                dt0 = tnow - t0
                dt1 = tnow - t1
                t1 = tnow
                log.info(f'{i}/{len(tileids)}  +{dt1:.1f} sec {dt0:.1f} total elapsed')

            ### globbing is faster than iterfiles if the directory pattern level is known
            # tmp = sorted(io.iterfiles(f'{indir}/{tileid}', prefix='redrock', suffix='.fits'))
            tmp = sorted(glob.glob(f'{indir}/{tileid}/*/redrock-*.fits'))
            if len(tmp) > 0:
                redrockfiles.append(tmp)
            else:
                log.error(f'no redrock files found in {indir}/{tileid}')

        #- convert list of lists -> list
        redrockfiles = list(itertools.chain.from_iterable(redrockfiles))

    nfiles = len(redrockfiles)
    if nfiles == 0:
        msg = f'No redrock files found in {indir}'
        log.critical(msg)
        raise ValueError(msg)
    log.info(f'Reading {nfiles} redrock files')

    #- build list of args to support multiprocessing parallelism
    read_args = list()
    for ifile, rrfile in enumerate(redrockfiles):
        read_args.append(dict(rrfile=rrfile, group=args.group, pertile=pertile,
                              counter=(ifile+1, nfiles)))

    #- Read individual Redrock files
    if args.nproc>1:
        from multiprocessing import Pool
        with Pool(args.nproc) as pool:
            results = pool.map(_wrap_read_redrock, read_args)
    else:
        results = [_wrap_read_redrock(a) for a in read_args]

    #- Stack catalogs
    zcatdata = list()
    exp_fibermaps = list()
    dependencies = dict()
    for data, expfibermap in results:
        if data is not None:
            desiutil.depend.mergedep(data.meta, dependencies)
            desiutil.depend.remove_dependencies(data.meta)
            zcatdata.append(data)

        if expfibermap is not None:
            exp_fibermaps.append(expfibermap)

    log.info('Stacking zcat')
    zcat = vstack(zcatdata, join_type='exact')
    desiutil.depend.mergedep(dependencies, zcat.meta)
    log.info('Stacking exposure fibermaps')
    expfm = np.hstack(exp_fibermaps)

    #- Add FIRSTNIGHT for tile-based cumulative catalogs
    #- (LASTNIGHT was added while reading from NIGHT header keyword)
    if args.group == 'cumulative' and 'FIRSTNIGHT' not in zcat.colnames:
        log.info('Adding FIRSTNIGHT per tile')
        icol = zcat.colnames.index('LASTNIGHT')
        zcat.add_column(np.zeros(len(zcat), dtype=np.int32),
                    index=icol, name='FIRSTNIGHT')
        for tilefm in Table(expfm[['TILEID', 'NIGHT']]).group_by('TILEID').groups:
            tileid = tilefm['TILEID'][0]
            iitile = zcat['TILEID'] == tileid
            zcat['FIRSTNIGHT'][iitile] = np.min(tilefm['NIGHT'])

        #- all FIRSTNIGHT entries should be filled (no more zeros)
        bad = zcat['FIRSTNIGHT'] == 0
        if np.any(bad):
            badtiles = np.unique(zcat['TILEID'][bad])
            raise ValueError(f'FIRSTNIGHT not set for tiles {badtiles}')

    #- if TARGETIDs appear more than once, which one is best within this catalog?
    if 'TSNR2_LRG' in zcat.colnames and 'ZWARN' in zcat.colnames:
        log.info('Finding best spectrum for each target')
        nspec, primary = find_primary_spectra(zcat)
        zcat['ZCAT_NSPEC'] = nspec.astype(np.int16)
        zcat['ZCAT_PRIMARY'] = primary
    else:
        log.info('Missing TSNR2_LRG or ZWARN; not adding ZCAT_PRIMARY/_NSPEC')

    #- Used for fuji, should not be needed for later prods
    if args.patch_missing_ivar_w12:
        from desimodel.footprint import radec2pix
        missing = (zcat['FLUX_IVAR_W1'] < 0) | (zcat['FLUX_IVAR_W2'] < 0)
        missing &= zcat['OBJTYPE'] == 'TGT'
        missing &= zcat['TARGETID'] > 0

        if not np.any(missing):
            log.info('No targets missing FLUX_IVAR_W1/W2 to patch')
        else:
            #- Load targets from sv1 targeting files
            ra = zcat['TARGET_RA']
            dec = zcat['TARGET_DEC']
            nside = 8  #- use for sv1 targeting
            hpix8 = radec2pix(nside, ra, dec)
            for hpix in np.unique(hpix8[missing]):
                hpixmiss = (hpix == hpix8) & missing
                targets = load_sv1_ivar_w12(hpix, zcat['TARGETID'][hpixmiss])

                #- create dict[TARGETID] -> row number
                targetid2idx = dict(zip(targets['TARGETID'],
                                        np.arange(len(targets))))

                #- patch missing values, if they are in the targets file
                for i in np.where(hpixmiss)[0]:
                    tid = zcat['TARGETID'][i]
                    try:
                        j = targetid2idx[ tid ]
                        zcat['FLUX_IVAR_W1'][i] = targets['FLUX_IVAR_W1'][j]
                        zcat['FLUX_IVAR_W2'][i] = targets['FLUX_IVAR_W2'][j]
                    except KeyError:
                        log.warning(f'TARGETID {tid} (row {i}) not found in sv1 targets')

    # Add redshift quality flags
    zqual = validredshifts.actually_validate(zcat)
    zcat = hstack([zcat, zqual], join_type='exact')

    # columns_basic = ['TARGETID', 'TILEID', 'FIRSTNIGHT', 'LASTNIGHT', 'Z', 'ZERR', 'ZWARN', 'CHI2', 'SPECTYPE', 'SUBTYPE', 'DELTACHI2', 'PETAL_LOC', 'FIBER', 'COADD_FIBERSTATUS', 'TARGET_RA', 'TARGET_DEC', 'DESINAME', 'OBJTYPE', 'FIBERASSIGN_X', 'FIBERASSIGN_Y', 'PRIORITY', 'DESI_TARGET', 'BGS_TARGET', 'MWS_TARGET', 'SCND_TARGET', 'COADD_NUMEXP', 'COADD_EXPTIME', 'COADD_NUMNIGHT', 'COADD_NUMTILE', 'MEAN_DELTA_X', 'RMS_DELTA_X', 'MEAN_DELTA_Y', 'RMS_DELTA_Y', 'MEAN_PSF_TO_FIBER_SPECFLUX', 'MEAN_FIBER_X', 'MEAN_FIBER_Y', 'MEAN_FIBER_RA', 'STD_FIBER_RA', 'MEAN_FIBER_DEC', 'STD_FIBER_DEC', 'MIN_MJD', 'MAX_MJD', 'MEAN_MJD', 'TSNR2_BGS', 'TSNR2_ELG', 'TSNR2_LRG', 'TSNR2_LYA', 'TSNR2_QSO', 'GOOD_BGS', 'GOOD_LRG', 'GOOD_ELG', 'GOOD_QSO']  # do not split the photometric columns

    columns_basic = ['TARGETID', 'TILEID', 'HEALPIX', 'LASTNIGHT', 'Z', 'ZERR', 'ZWARN', 'CHI2', 'SPECTYPE', 'SUBTYPE', 'DELTACHI2', 'PETAL_LOC', 'FIBER', 'COADD_FIBERSTATUS', 'TARGET_RA', 'TARGET_DEC', 'DESINAME', 'OBJTYPE', 'FIBERASSIGN_X', 'FIBERASSIGN_Y', 'PRIORITY', 'DESI_TARGET', 'BGS_TARGET', 'MWS_TARGET', 'SCND_TARGET', 'CMX_TARGET', 'SV1_DESI_TARGET', 'SV1_BGS_TARGET', 'SV1_MWS_TARGET', 'SV1_SCND_TARGET', 'SV2_DESI_TARGET', 'SV2_BGS_TARGET', 'SV2_MWS_TARGET', 'SV2_SCND_TARGET', 'SV3_DESI_TARGET', 'SV3_BGS_TARGET', 'SV3_MWS_TARGET', 'SV3_SCND_TARGET' 'COADD_NUMEXP', 'COADD_EXPTIME', 'COADD_NUMNIGHT', 'COADD_NUMTILE', 'MIN_MJD', 'MAX_MJD', 'MEAN_MJD', 'TSNR2_BGS', 'TSNR2_ELG', 'TSNR2_LRG', 'TSNR2_LYA', 'TSNR2_QSO', 'GOOD_BGS', 'GOOD_LRG', 'GOOD_ELG', 'GOOD_QSO']
    columns_photom = ['PMRA', 'PMDEC', 'REF_EPOCH', 'RELEASE', 'BRICKNAME', 'BRICKID', 'BRICK_OBJID', 'MORPHTYPE', 'EBV', 'FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_W2', 'FLUX_IVAR_G', 'FLUX_IVAR_R', 'FLUX_IVAR_Z', 'FLUX_IVAR_W1', 'FLUX_IVAR_W2', 'FIBERFLUX_G', 'FIBERFLUX_R', 'FIBERFLUX_Z', 'FIBERTOTFLUX_G', 'FIBERTOTFLUX_R', 'FIBERTOTFLUX_Z', 'MASKBITS', 'SERSIC', 'SHAPE_R', 'SHAPE_E1', 'SHAPE_E2', 'REF_ID', 'REF_CAT', 'GAIA_PHOT_G_MEAN_MAG', 'GAIA_PHOT_BP_MEAN_MAG', 'GAIA_PHOT_RP_MEAN_MAG', 'PARALLAX', 'PHOTSYS']
    assert len(np.intersect1d(columns_basic, columns_photom))==0

    # Remove main-survey target bits for non-main surveys (they are not the actual main-survey target bits)
    if args.survey!='main':
        for col in ['DESI_TARGET', 'BGS_TARGET', 'MWS_TARGET', 'SCND_TARGET']:
            if col in zcat.colnames:
                zcat.remove_column(col)

    # Remove the columns that do not exist
    columns_basic = [col for col in columns_basic if col in zcat.colnames]
    columns_photom = ['TARGETID', 'TILEID'] + columns_photom
    columns_photom = [col for col in columns_photom if col in zcat.colnames]  # remove columns that do not exist
    columns_extra = ['TARGETID', 'TILEID'] + [col for col in zcat.colnames if (col not in columns_basic and col not in columns_photom)]
    columns_extra = [col for col in columns_extra if col in zcat.colnames]  # remove columns that do not exist

    zcat_basic = zcat[columns_basic].copy()
    zcat_photom = zcat[columns_photom].copy()
    zcat_extra = zcat[columns_extra].copy()

    #- we're done adding columns, convert to numpy array for fitsio
    zcat = np.array(zcat)
    zcat_basic = np.array(zcat_basic)
    zcat_photom = np.array(zcat_photom)
    zcat_extra = np.array(zcat_extra)

    #- Inherit header from first input, but remove keywords that don't apply
    #- across multiple files
    header = fitsio.read_header(redrockfiles[0], 0)
    for key in ['SPGRPVAL', 'TILEID', 'SPECTRO', 'PETAL', 'NIGHT', 'EXPID', 'HPXPIXEL',
                'NAXIS', 'BITPIX', 'SIMPLE', 'EXTEND']:
        if key in header:
            header.delete(key)

    #- Intercept previous incorrect boolean special cases
    if 'HPXNEST' in header:
        if header['HPXNEST'] == 'True':
            log.info("Correcting header HPXNEST='True' string to boolean True")
            header['HPXNEST'] = True
        elif header['HPXNEST'] == 'False':
            # False is not expected for DESI, but cover it for completeness
            log.info("Correcting header HPXNEST='False' string to boolean False")
            header['HPXNEST'] = False

    #- Add extra keywords if requested
    if args.header is not None:
        for keyval in args.header:
            key, value = parse_keyval(keyval)
            header[key] = value

    if args.survey is not None:
        header['SURVEY'] = args.survey

    if args.program is not None:
        header['PROGRAM'] = args.program

    #- Add units if requested
    if args.add_units:
        datamodeldir = str(importlib.resources.files('desidatamodel'))
        unitsfile = os.path.join(datamodeldir, 'data', 'column_descriptions.csv')
        log.info(f'Adding units from {unitsfile}')
        units, comments = load_csv_units(unitsfile)
    else:
        units = dict()
        comments = dict()

    outfile_all = os.path.join(os.path.dirname(args.outfile), 'merged', args.outfile+'.fits')
    if not os.path.isdir(os.path.dirname(outfile_all)):
        os.makedirs(os.path.dirname(outfile_all))
    log.info(f'Writing {outfile_all}')
    tmpfile = get_tempfilename(outfile_all)
    write_bintable(tmpfile, zcat_basic, header=header, extname='ZCATALOG',
                   units=units, clobber=True)
    write_bintable(tmpfile, expfm, extname='EXP_FIBERMAP', units=units)
    os.rename(tmpfile, outfile_all)
    log.info("Successfully wrote {}".format(outfile_all))

    outfile_basic = args.outfile+'-basic.fits'
    log.info(f'Writing {outfile_basic}')
    tmpfile = get_tempfilename(outfile_basic)
    write_bintable(tmpfile, zcat_basic, header=header, extname='ZCATALOG',
                   units=units, clobber=True)
    os.rename(tmpfile, outfile_basic)
    log.info("Successfully wrote {}".format(outfile_basic))

    outfile_photom = args.outfile+'-photom.fits'
    log.info(f'Writing {outfile_photom}')
    tmpfile = get_tempfilename(outfile_photom)
    write_bintable(tmpfile, zcat_photom, header=header, extname='PHOTOM',
                   units=units, clobber=True)
    os.rename(tmpfile, outfile_photom)
    log.info("Successfully wrote {}".format(outfile_photom))

    outfile_extra = args.outfile+'-extra.fits'
    log.info(f'Writing {outfile_extra}')
    tmpfile = get_tempfilename(outfile_extra)
    write_bintable(tmpfile, zcat_extra, header=header, extname='EXTRACOLS',
                   units=units, clobber=True)
    os.rename(tmpfile, outfile_extra)
    log.info("Successfully wrote {}".format(outfile_extra))

    outfile_expfm = os.path.join(os.path.dirname(args.outfile), 'exp_fibermap', args.outfile+'-expfibermap.fits')
    if not os.path.isdir(os.path.dirname(outfile_expfm)):
        os.makedirs(os.path.dirname(outfile_expfm))
    log.info(f'Writing {outfile_expfm}')
    tmpfile = get_tempfilename(outfile_expfm)
    write_bintable(tmpfile, expfm, header=header, extname='EXP_FIBERMAP',
                   units=units, clobber=True)
    os.rename(tmpfile, outfile_expfm)
    log.info("Successfully wrote {}".format(outfile_expfm))


