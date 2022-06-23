"""
desispec.io.photo
=================

Simple methods for gathering photometric and targeting catalogs.

"""
import os, pdb
from glob import glob
import numpy as np
import fitsio

from astropy.table import Table, vstack
import astropy.units as u
from astropy.coordinates import SkyCoord

from desitarget.io import desitarget_resolve_dec

from desiutil.log import get_logger, DEBUG
log = get_logger()#DEBUG)

def gather_targetdirs(tileid, fiberassign_dir=None):
    """Gather all the targeting directories used to build a given fiberassign catalog.

    Args:
        tileid (int): tile number
        fiberassign_dir (str, optional): directory to fiberassign tables

    Given a single tile, return a list of all the unique targeting directories
    used to run fiberassign to generate that tile. If there are TOOs on the
    tile, return the TOO filename itself, not just the directory.

    """
    from astropy.io import fits

    desi_root = os.environ.get('DESI_ROOT')

    if fiberassign_dir is None:
        fiberassign_dir = os.path.join(desi_root, 'target', 'fiberassign', 'tiles', 'trunk')
    
    stileid = '{:06d}'.format(tileid)
    fiberfile = os.path.join(fiberassign_dir, stileid[:3], 'fiberassign-{}.fits.gz'.format(stileid))
    if not os.path.isfile(fiberfile):
        fiberfile = fiberfile.replace('.gz', '')
        if not os.path.isfile(fiberfile):
            errmsg = 'Fiber assignment file {} not found!'.format(fiberfile)
            log.critical(errmsg)
            raise IOError(errmsg)
    log.debug('Reading {} header.'.format(fiberfile))
    # old versions of fitsio can't handle CONTINUE header cards!
    #fahdr = fitsio.read_header(fiberfile, ext=0)
    fahdr = fits.getheader(fiberfile, ext=0)

    # Gather the targeting directories.
    targetdirs = [fahdr['TARG']]
    for moretarg in ['TARG2', 'TARG3', 'TARG4']:
        if moretarg in fahdr:
            targetdirs += [fahdr[moretarg]]            

    # Any secondary targets or ToOs?
    if 'SCND' in fahdr:
        if fahdr['SCND'].strip() != '-':
            targetdirs += [fahdr['SCND']]

    if 'TOO' in fahdr:
        TOOfile = fahdr['TOO']
        # can be a KPNO directory!
        if 'DESIROOT' in TOOfile:
            TOOfile = os.path.join(desi_root, TOOfile.replace('DESIROOT/', ''))
        if TOOfile[:6] == '/data/': # fragile
            TOOfile = os.path.join(desi_root, TOOfile.replace('/data/', ''))
        if 'afternoon_planning' in TOOfile:
            TOOfile = TOOfile.replace('afternoon_planning/surveyops', 'survey/ops/surveyops') # fragile!

        if os.path.isfile(TOOfile):
            targetdirs += [TOOfile]

    cmxtargetdir = None
    for ii, targetdir in enumerate(targetdirs):
        # for secondary targets, targetdir can be a filename
        if targetdir[-4:] == 'fits': # fragile...
            targetdir = os.path.dirname(targetdir)
        if not os.path.isdir(targetdir):
            if 'DESIROOT' in targetdir:
                targetdir = os.path.join(desi_root, targetdir.replace('DESIROOT/', ''))
            if targetdir[:6] == '/data/':
                targetdir = os.path.join(desi_root, targetdir.replace('/data/', ''))
            if 'afternoon_planning' in targetdir:
                targetdir = targetdir.replace('afternoon_planning/surveyops', 'survey/ops/surveyops') # fragile!

        # special-case first-light / cmx targets
        if 'catalogs/dr9/0.47.0/targets/cmx/resolve/no-obscon/' in targetdir:
            cmxtargetdir = os.environ.get('DESI_ROOT')+'/target/catalogs/gaiadr2/0.47.0/targets/cmx/resolve/supp/'
            
        if os.path.isdir(targetdir) or os.path.isfile(targetdir):
            log.debug('Found targets directory or file {}'.format(targetdir))
            targetdirs[ii] = targetdir
        else:
            #log.warning('Targets directory or file {} not found.'.format(targetdir))
            pass

    if cmxtargetdir is not None:
        if os.path.isdir(cmxtargetdir):
            log.debug('Found targets directory or file {}'.format(cmxtargetdir))
            targetdirs = targetdirs + [cmxtargetdir]

    # Special-case an early / first-light / commissioning tile where the
    # fiberassign header is incomplete. From Adam Myers on 2022-Mar-22: "The SV1
    # target (39633154205551487) appears to be from the dr9m release that we
    # used for part of commissioning (and, maybe SV0, too?) I'm not sure how a
    # target from early SV would only appear in dr9m . It's a strange one."
    if tileid == 80736:
        dr9mdir = os.environ.get('DESI_ROOT')+'/target/catalogs/dr9m/0.44.0/targets/sv1/resolve/dark/'
        if os.path.isdir(dr9mdir):
            log.debug('Found targets directory or file {}'.format(dr9mdir))
            targetdirs = targetdirs + [dr9mdir]

    targetdirs = np.unique(np.hstack(targetdirs))
        
    # Special-case SV1 tiles where the original targeting catalogs for secondary
    # targets were missing the DR9 photometry but subsequent versions were not.
    if (tileid > 80600) and (tileid < 81000):
        newtargetdirs = []
        for ii, targetdir in enumerate(np.atleast_1d(targetdirs)):
            if 'targets/sv1/secondary/dark' in targetdir or 'targets/sv1/secondary/bright' in targetdir:
                newtargetdir = sorted(glob(targetdir.replace(targetdir.split('/')[-5], '*')))[-1] # most recent one
                #log.debug('Special-casing targetdir for tile {}: {} --> {}'.format(tileid, targetdirs[ii], newtargetdir))
                #targetdirs[ii] = newtargetdir
                log.debug('Appending targetdir for tile {}: {} --> {}'.format(tileid, targetdirs[ii], newtargetdir))
                newtargetdirs.append(newtargetdir)
        if len(newtargetdirs) > 0:
            targetdirs = np.hstack((targetdirs, newtargetdirs))

    targetdirs = np.sort(np.unique(targetdirs))
        
    return targetdirs

def targetphot_datamodel(from_file=False):
    """Initialize the targetphot data model.

    Args:
        from_file (bool, optional): read the datamodel from a file on-disk.

    Returns an `astropy.table.Table` with a consistent data model across cmx,
    sv[1-2], and main-survey observations.

    """
    if from_file:
        TARGETINGBITCOLS = [
            'CMX_TARGET',
            'DESI_TARGET', 'BGS_TARGET', 'MWS_TARGET',
            'SV1_DESI_TARGET', 'SV1_BGS_TARGET', 'SV1_MWS_TARGET',
            'SV2_DESI_TARGET', 'SV2_BGS_TARGET', 'SV2_MWS_TARGET',
            'SV3_DESI_TARGET', 'SV3_BGS_TARGET', 'SV3_MWS_TARGET',
            'SCND_TARGET',
            'SV1_SCND_TARGET', 'SV2_SCND_TARGET', 'SV3_SCND_TARGET',
            ]
        
        from pkg_resources import resource_filename
        #datamodel_file = resource_filename('desitarget.test', 't/targets.fits')
        datamodel_file = os.path.join(os.environ.get('DESI_ROOT'), 'target', 'catalogs', 'dr9', '1.1.1', 'targets',
                                      'main', 'resolve', 'dark', 'targets-dark-hp-0.fits')
        if not os.path.isfile(datamodel_file):
            errmsg = 'Unable to establish the data model using {}'.format(datamodel_file)
            log.critical(errmsg)
            raise IOError(errmsg)
        
        datamodel = Table(fitsio.read(datamodel_file, rows=0, upper=True))
        for col in datamodel.colnames:
            if '_TARGET' in col:
                datamodel.remove_column(col)
            else:
                datamodel[col] = np.zeros(datamodel[col].shape, dtype=datamodel[col].dtype)
        for col in TARGETINGBITCOLS:
            datamodel[col] = np.zeros(1, dtype=np.int64)
        
        #for col in datamodel.colnames:
        #    print("('{}', {}, '{}'),".format(col, datamodel[col].shape, datamodel[col].dtype))
    else:
        COLS = [
            ('RELEASE', (1,), '>i2'),
            ('BRICKID', (1,), '>i4'),
            ('BRICKNAME', (1,), '<U8'),
            ('BRICK_OBJID', (1,), '>i4'),
            ('MORPHTYPE', (1,), '<U4'),
            ('RA', (1,), '>f8'),
            ('RA_IVAR', (1,), '>f4'),
            ('DEC', (1,), '>f8'),
            ('DEC_IVAR', (1,), '>f4'),
            ('DCHISQ', (1, 5), '>f4'),
            ('EBV', (1,), '>f4'),
            ('FLUX_G', (1,), '>f4'),
            ('FLUX_R', (1,), '>f4'),
            ('FLUX_Z', (1,), '>f4'),
            ('FLUX_IVAR_G', (1,), '>f4'),
            ('FLUX_IVAR_R', (1,), '>f4'),
            ('FLUX_IVAR_Z', (1,), '>f4'),
            ('MW_TRANSMISSION_G', (1,), '>f4'),
            ('MW_TRANSMISSION_R', (1,), '>f4'),
            ('MW_TRANSMISSION_Z', (1,), '>f4'),
            ('FRACFLUX_G', (1,), '>f4'),
            ('FRACFLUX_R', (1,), '>f4'),
            ('FRACFLUX_Z', (1,), '>f4'),
            ('FRACMASKED_G', (1,), '>f4'),
            ('FRACMASKED_R', (1,), '>f4'),
            ('FRACMASKED_Z', (1,), '>f4'),
            ('FRACIN_G', (1,), '>f4'),
            ('FRACIN_R', (1,), '>f4'),
            ('FRACIN_Z', (1,), '>f4'),
            ('NOBS_G', (1,), '>i2'),
            ('NOBS_R', (1,), '>i2'),
            ('NOBS_Z', (1,), '>i2'),
            ('PSFDEPTH_G', (1,), '>f4'),
            ('PSFDEPTH_R', (1,), '>f4'),
            ('PSFDEPTH_Z', (1,), '>f4'),
            ('GALDEPTH_G', (1,), '>f4'),
            ('GALDEPTH_R', (1,), '>f4'),
            ('GALDEPTH_Z', (1,), '>f4'),
            ('FLUX_W1', (1,), '>f4'),
            ('FLUX_W2', (1,), '>f4'),
            ('FLUX_W3', (1,), '>f4'),
            ('FLUX_W4', (1,), '>f4'),
            ('FLUX_IVAR_W1', (1,), '>f4'),
            ('FLUX_IVAR_W2', (1,), '>f4'),
            ('FLUX_IVAR_W3', (1,), '>f4'),
            ('FLUX_IVAR_W4', (1,), '>f4'),
            ('MW_TRANSMISSION_W1', (1,), '>f4'),
            ('MW_TRANSMISSION_W2', (1,), '>f4'),
            ('MW_TRANSMISSION_W3', (1,), '>f4'),
            ('MW_TRANSMISSION_W4', (1,), '>f4'),
            ('ALLMASK_G', (1,), '>i2'),
            ('ALLMASK_R', (1,), '>i2'),
            ('ALLMASK_Z', (1,), '>i2'),
            ('FIBERFLUX_G', (1,), '>f4'),
            ('FIBERFLUX_R', (1,), '>f4'),
            ('FIBERFLUX_Z', (1,), '>f4'),
            ('FIBERTOTFLUX_G', (1,), '>f4'),
            ('FIBERTOTFLUX_R', (1,), '>f4'),
            ('FIBERTOTFLUX_Z', (1,), '>f4'),
            ('REF_EPOCH', (1,), '>f4'),
            ('WISEMASK_W1', (1,), 'uint8'),
            ('WISEMASK_W2', (1,), 'uint8'),
            ('MASKBITS', (1,), '>i2'),
            ('LC_FLUX_W1', (1, 15), '>f4'),
            ('LC_FLUX_W2', (1, 15), '>f4'),
            ('LC_FLUX_IVAR_W1', (1, 15), '>f4'),
            ('LC_FLUX_IVAR_W2', (1, 15), '>f4'),
            ('LC_NOBS_W1', (1, 15), '>i2'),
            ('LC_NOBS_W2', (1, 15), '>i2'),
            ('LC_MJD_W1', (1, 15), '>f8'),
            ('LC_MJD_W2', (1, 15), '>f8'),
            ('SHAPE_R', (1,), '>f4'),
            ('SHAPE_E1', (1,), '>f4'),
            ('SHAPE_E2', (1,), '>f4'),
            ('SHAPE_R_IVAR', (1,), '>f4'),
            ('SHAPE_E1_IVAR', (1,), '>f4'),
            ('SHAPE_E2_IVAR', (1,), '>f4'),
            ('SERSIC', (1,), '>f4'),
            ('SERSIC_IVAR', (1,), '>f4'),
            ('REF_ID', (1,), '>i8'),
            ('REF_CAT', (1,), '<U2'),
            ('GAIA_PHOT_G_MEAN_MAG', (1,), '>f4'),
            ('GAIA_PHOT_G_MEAN_FLUX_OVER_ERROR', (1,), '>f4'),
            ('GAIA_PHOT_BP_MEAN_MAG', (1,), '>f4'),
            ('GAIA_PHOT_BP_MEAN_FLUX_OVER_ERROR', (1,), '>f4'),
            ('GAIA_PHOT_RP_MEAN_MAG', (1,), '>f4'),
            ('GAIA_PHOT_RP_MEAN_FLUX_OVER_ERROR', (1,), '>f4'),
            ('GAIA_PHOT_BP_RP_EXCESS_FACTOR', (1,), '>f4'),
            ('GAIA_ASTROMETRIC_EXCESS_NOISE', (1,), '>f4'),
            ('GAIA_DUPLICATED_SOURCE', (1,), 'bool'),
            ('GAIA_ASTROMETRIC_SIGMA5D_MAX', (1,), '>f4'),
            ('GAIA_ASTROMETRIC_PARAMS_SOLVED', (1,), 'int8'),
            ('PARALLAX', (1,), '>f4'),
            ('PARALLAX_IVAR', (1,), '>f4'),
            ('PMRA', (1,), '>f4'),
            ('PMRA_IVAR', (1,), '>f4'),
            ('PMDEC', (1,), '>f4'),
            ('PMDEC_IVAR', (1,), '>f4'),
            ('PHOTSYS', (1,), '<U1'),
            ('TARGETID', (1,), '>i8'),
            ('SUBPRIORITY', (1,), '>f8'),
            ('OBSCONDITIONS', (1,), '>i8'),
            ('PRIORITY_INIT', (1,), '>i8'),
            ('NUMOBS_INIT', (1,), '>i8'),
            ('HPXPIXEL', (1,), '>i8'),
            # added columns
            ('CMX_TARGET', (1,), 'int64'),
            ('DESI_TARGET', (1,), 'int64'),
            ('BGS_TARGET', (1,), 'int64'),
            ('MWS_TARGET', (1,), 'int64'),
            ('SV1_DESI_TARGET', (1,), 'int64'),
            ('SV1_BGS_TARGET', (1,), 'int64'),
            ('SV1_MWS_TARGET', (1,), 'int64'),
            ('SV2_DESI_TARGET', (1,), 'int64'),
            ('SV2_BGS_TARGET', (1,), 'int64'),
            ('SV2_MWS_TARGET', (1,), 'int64'),
            ('SV3_DESI_TARGET', (1,), 'int64'),
            ('SV3_BGS_TARGET', (1,), 'int64'),
            ('SV3_MWS_TARGET', (1,), 'int64'),
            ('SCND_TARGET', (1,), 'int64'),
            ('SV1_SCND_TARGET', (1,), 'int64'),
            ('SV2_SCND_TARGET', (1,), 'int64'),
            ('SV3_SCND_TARGET', (1,), 'int64'),
            ]
    
        datamodel = Table()
        for col in COLS:
            datamodel[col[0]] = np.zeros(shape=col[1], dtype=col[2])

    return datamodel

def gather_targetphot(input_cat, photocache=None, tileids=None, racolumn='TARGET_RA',
                      deccolumn='TARGET_DEC', columns=None, fiberassign_dir=None):
    """Find and stack the photometric targeting information given a set of targets.

    Args:
        input_cat (astropy.table.Table): input table with the following
          (required) columns: TARGETID, RACOLUMN, DECCOLUMN and, optionally,
          TILEID.
        tileids (dict, optional): dictionary cache of targetids for large
          targeting catalogs.
    
        photocache (dict, optional): dictionary cache of targetids for large
          targeting catalogs.
        racolumn (str): name of the RA column in `input_cat` (defaults to
          RA_COLUMN)
        deccolumn (str): name of the RA column in `input_cat` (defaults to
          DEC_COLUMN)
        columns (str array): return this subset of columns
        fiberassign_dir (str, optional): top-level directory to fiberassign
          tables

    Returns a table of targeting photometry using a consistent data model across
    primary (DR9) targets, secondary targets, and targets of opportunity. The
    data model is documented in `targetphot_datamodel`.

    """
    import astropy
    from desimodel.footprint import radec2pix

    if len(input_cat) == 0:
        log.warning('No objects in input catalog.')
        return Table()

    if tileids is None:
        required_columns = ['TARGETID', racolumn, deccolumn, 'TILEID']
    else:
        required_columns = ['TARGETID', racolumn, deccolumn]

    for col in required_columns:
        if col not in input_cat.colnames:
            errmsg = 'Missing required input column {}'.format(col)
            log.critical(errmsg)
            raise ValueError(errmsg)

    if tileids is None:
        tileids = input_cat['TILEID']

    datamodel = targetphot_datamodel()
    out = Table(np.hstack(np.repeat(datamodel, len(np.atleast_1d(input_cat)))))
    out['TARGETID'] = input_cat['TARGETID']

    for tileid in np.unique(tileids):
        log.debug('Working on tile {}'.format(tileid))

        M = np.where(tileid == tileids)[0]
        out1 = out[M]
        input_cat1 = input_cat[M]

        photo, photofiles = [], []

        # Get the unique list of targetdirs
        targetdirs = gather_targetdirs(tileid, fiberassign_dir=fiberassign_dir)
        
        for targetdir in targetdirs:
            # Handle secondary targets, which have a (very!) different data model.
            if 'secondary' in targetdir:
                if 'sv1' in targetdir: # special case
                    if 'dedicated' in targetdir:
                        targetfiles = glob(os.path.join(targetdir, 'DC3R2_GAMA_priorities.fits'))
                    else:
                        targetfiles = glob(os.path.join(targetdir, '*-secondary-dr9photometry.fits')) # use??
                        #targetfiles = glob(os.path.join(targetdir, '*-secondary.fits'))
                else:
                    targetfiles = glob(os.path.join(targetdir, '*-secondary.fits'))
            elif 'ToO' in targetdir:
                targetfiles = targetdir
            else:
                alltargetfiles = glob(os.path.join(targetdir, '*-hp-*.fits'))
                filenside = fitsio.read_header(alltargetfiles[0], ext=1)['FILENSID']
                # https://github.com/desihub/desispec/issues/1711
                if np.any(np.isnan(input_cat1[racolumn])): # some SV1 targets have nan in RA,DEC
                    log.warning('Some RA, DEC are NaN in target directory {}'.format(targetdir))
                notnan = np.isfinite(input_cat1[racolumn])
                targetfiles = []
                if np.sum(notnan) > 0:
                    pixlist = radec2pix(filenside, input_cat1[racolumn][notnan], input_cat1[deccolumn][notnan])
                    for pix in set(pixlist):
                        _targetfile = alltargetfiles[0].split('hp-')[0]+'hp-{}.fits'.format(pix) # fragile
                        if os.path.isfile(_targetfile):
                            targetfiles.append(_targetfile)
    
            targetfiles = np.unique(targetfiles)
    
            if len(targetfiles) == 0:
                continue
    
            for targetfile in np.atleast_1d(targetfiles):
                # If this is a secondary target catalog or ToO, use the photocache
                # (if it exists). Also note that secondary target catalogs are
                # missing some or all of the DR9 photometry columns we need, so only
                # copy what exists.
                if photocache is not None and targetfile in photocache.keys():
                    if type(photocache[targetfile]) == astropy.table.Table:
                        I = np.where(np.isin(photocache[targetfile]['TARGETID'], input_cat1['TARGETID']))[0]
                    else:
                        photo_targetid = photocache[targetfile]
                        I = np.where(np.isin(photo_targetid, input_cat1['TARGETID']))[0]
                        
                    log.debug('Matched {} targets in {}'.format(len(I), targetfile))
                    if len(I) > 0:
                        if type(photocache[targetfile]) == astropy.table.Table:
                            cachecat = photocache[targetfile][I]
                        else:
                            cachecat = Table(fitsio.read(targetfile, rows=I))
                        
                        _photo = Table(np.hstack(np.repeat(datamodel, len(I))))
                        for col in _photo.colnames: # not all these columns will exist...
                            if col in cachecat.colnames:
                                _photo[col] = cachecat[col]
                        photofiles.append(targetfile)
                        photo.append(_photo)
                    continue
    
                if 'ToO' in targetfile:
                    photo1 = Table.read(targetfile, guess=False, format='ascii.ecsv')
                    I = np.where(np.isin(photo1['TARGETID'], input_cat1['TARGETID']))[0]
                    log.debug('Matched {} TOO targets'.format(len(I)))
                    if len(I) > 0:
                        photo1 = photo1[I]
                        _photo = Table(np.hstack(np.repeat(datamodel, len(I))))
                        for col in _photo.colnames: # not all these columns will exist...
                            if col in photo1.colnames:
                                _photo[col] = photo1[col]
                        del photo1
                        photofiles.append('TOO')
                        photo.append(_photo)
                    continue
    
                # get the correct extension name or number
                tinfo = fitsio.FITS(targetfile)
                for _tinfo in tinfo:
                    extname = _tinfo.get_extname()
                    if 'TARGETS' in extname:
                        break
                if extname == '':
                    extname = 1
                    
                # fitsio does not preserve the order of the rows but we'll sort later.
                photo_targetid = tinfo[extname].read(columns='TARGETID')
                I = np.where(np.isin(photo_targetid, input_cat1['TARGETID']))[0]
    
                log.debug('Matched {} targets in {}'.format(len(I), targetfile))
                if len(I) > 0:
                    photo1 = tinfo[extname].read(rows=I)
                    # Columns can be out of order, so sort them here based on the
                    # data model so we can stack below.
                    _photo = Table(np.hstack(np.repeat(datamodel, len(I))))
                    for col in _photo.colnames: # all these columns should exist...
                        if col in photo1.dtype.names:
                            _photo[col] = photo1[col]
                        else:
                            #log.debug('Skipping missing column {} from {}'.format(col, targetfile))
                            pass
                    del photo1
                    photofiles.append(targetfile)
                    photo.append(_photo)
    
        # backup programs have no target catalog photometry at all
        if len(photo) == 0:
            continue
            #errmsg = 'No targeting photometry found.'
            #log.critical(errmsg)
            #raise ValueError(errmsg)
    
        # np.hstack will sometimes complain even if the tables are identical...
        #photo = Table(np.hstack(photo))
        photo = vstack(photo)
    
        # make sure there are no duplicates...?
        _, uindx = np.unique(photo['TARGETID'], return_index=True)
        #if len(uindx) < len(photo):
        #    # https://stackoverflow.com/questions/30003068/how-to-get-a-list-of-all-indices-of-repeated-elements-in-a-numpy-array
        #    idx_sort = np.argsort(photo['TARGETID'], kind='mergesort')
        #    targetid_sorted = photo['TARGETID'][idx_sort].data
        #    vals, idx_start, count = np.unique(targetid_sorted, return_counts=True, return_index=True)
        #    res = np.split(idx_sort, idx_start[1:])
        #    #vals = vals[count > 1]
        #    #res = filter(lambda x: x.size > 1, res)
        
        photo = photo[uindx]
        assert(len(np.unique(photo['TARGETID'])) == len(photo))

        # sort explicitly in order to ensure order
        I = np.where(np.isin(out1['TARGETID'], photo['TARGETID']))[0]
        srt = np.hstack([np.where(tid == photo['TARGETID'])[0] for tid in out1['TARGETID'][I]])
            
        out1[I] = photo[srt]
        out[M] = out1
        del out1, photo

    if columns is not None:
        out = out[columns]
    
    return out

def tractorphot_datamodel(from_file=False):
    """Initialize the tractorphot data model for DR9 photometry.

    Args:
        from_file (bool, optional): read the datamodel from a file on-disk.

    Returns an `astropy.table.Table` which follows the Tractor catalog
    datamodel.

    """
    if from_file:
        datamodel_file = os.environ.get('DESI_ROOT')+'/external/legacysurvey/dr9/south/tractor/000/tractor-0001m002.fits'
        datamodel = Table(fitsio.read(datamodel_file, rows=0, upper=True))
        for col in datamodel.colnames:
            datamodel[col] = np.zeros(datamodel[col].shape, dtype=datamodel[col].dtype)
        
        #for col in datamodel.colnames:
        #   print("('{}', {}, '{}'),".format(col, datamodel[col].shape, datamodel[col].dtype))
    else:
        COLS = [
            ('RELEASE', (1,), '>i2'),
            ('BRICKID', (1,), '>i4'),
            ('BRICKNAME', (1,), '<U8'),
            ('OBJID', (1,), '>i4'),
            ('BRICK_PRIMARY', (1,), 'bool'),
            ('MASKBITS', (1,), '>i2'),
            ('FITBITS', (1,), '>i2'),
            ('TYPE', (1,), '<U3'),
            ('RA', (1,), '>f8'),
            ('DEC', (1,), '>f8'),
            ('RA_IVAR', (1,), '>f4'),
            ('DEC_IVAR', (1,), '>f4'),
            ('BX', (1,), '>f4'),
            ('BY', (1,), '>f4'),
            ('DCHISQ', (1, 5), '>f4'),
            ('EBV', (1,), '>f4'),
            ('MJD_MIN', (1,), '>f8'),
            ('MJD_MAX', (1,), '>f8'),
            ('REF_CAT', (1,), '<U2'),
            ('REF_ID', (1,), '>i8'),
            ('PMRA', (1,), '>f4'),
            ('PMDEC', (1,), '>f4'),
            ('PARALLAX', (1,), '>f4'),
            ('PMRA_IVAR', (1,), '>f4'),
            ('PMDEC_IVAR', (1,), '>f4'),
            ('PARALLAX_IVAR', (1,), '>f4'),
            ('REF_EPOCH', (1,), '>f4'),
            ('GAIA_PHOT_G_MEAN_MAG', (1,), '>f4'),
            ('GAIA_PHOT_G_MEAN_FLUX_OVER_ERROR', (1,), '>f4'),
            ('GAIA_PHOT_G_N_OBS', (1,), '>i2'),
            ('GAIA_PHOT_BP_MEAN_MAG', (1,), '>f4'),
            ('GAIA_PHOT_BP_MEAN_FLUX_OVER_ERROR', (1,), '>f4'),
            ('GAIA_PHOT_BP_N_OBS', (1,), '>i2'),
            ('GAIA_PHOT_RP_MEAN_MAG', (1,), '>f4'),
            ('GAIA_PHOT_RP_MEAN_FLUX_OVER_ERROR', (1,), '>f4'),
            ('GAIA_PHOT_RP_N_OBS', (1,), '>i2'),
            ('GAIA_PHOT_VARIABLE_FLAG', (1,), 'bool'),
            ('GAIA_ASTROMETRIC_EXCESS_NOISE', (1,), '>f4'),
            ('GAIA_ASTROMETRIC_EXCESS_NOISE_SIG', (1,), '>f4'),
            ('GAIA_ASTROMETRIC_N_OBS_AL', (1,), '>i2'),
            ('GAIA_ASTROMETRIC_N_GOOD_OBS_AL', (1,), '>i2'),
            ('GAIA_ASTROMETRIC_WEIGHT_AL', (1,), '>f4'),
            ('GAIA_DUPLICATED_SOURCE', (1,), 'bool'),
            ('GAIA_A_G_VAL', (1,), '>f4'),
            ('GAIA_E_BP_MIN_RP_VAL', (1,), '>f4'),
            ('GAIA_PHOT_BP_RP_EXCESS_FACTOR', (1,), '>f4'),
            ('GAIA_ASTROMETRIC_SIGMA5D_MAX', (1,), '>f4'),
            ('GAIA_ASTROMETRIC_PARAMS_SOLVED', (1,), 'uint8'),
            ('FLUX_G', (1,), '>f4'),
            ('FLUX_R', (1,), '>f4'),
            ('FLUX_Z', (1,), '>f4'),
            ('FLUX_W1', (1,), '>f4'),
            ('FLUX_W2', (1,), '>f4'),
            ('FLUX_W3', (1,), '>f4'),
            ('FLUX_W4', (1,), '>f4'),
            ('FLUX_IVAR_G', (1,), '>f4'),
            ('FLUX_IVAR_R', (1,), '>f4'),
            ('FLUX_IVAR_Z', (1,), '>f4'),
            ('FLUX_IVAR_W1', (1,), '>f4'),
            ('FLUX_IVAR_W2', (1,), '>f4'),
            ('FLUX_IVAR_W3', (1,), '>f4'),
            ('FLUX_IVAR_W4', (1,), '>f4'),
            ('FIBERFLUX_G', (1,), '>f4'),
            ('FIBERFLUX_R', (1,), '>f4'),
            ('FIBERFLUX_Z', (1,), '>f4'),
            ('FIBERTOTFLUX_G', (1,), '>f4'),
            ('FIBERTOTFLUX_R', (1,), '>f4'),
            ('FIBERTOTFLUX_Z', (1,), '>f4'),
            ('APFLUX_G', (1, 8), '>f4'),
            ('APFLUX_R', (1, 8), '>f4'),
            ('APFLUX_Z', (1, 8), '>f4'),
            ('APFLUX_RESID_G', (1, 8), '>f4'),
            ('APFLUX_RESID_R', (1, 8), '>f4'),
            ('APFLUX_RESID_Z', (1, 8), '>f4'),
            ('APFLUX_BLOBRESID_G', (1, 8), '>f4'),
            ('APFLUX_BLOBRESID_R', (1, 8), '>f4'),
            ('APFLUX_BLOBRESID_Z', (1, 8), '>f4'),
            ('APFLUX_IVAR_G', (1, 8), '>f4'),
            ('APFLUX_IVAR_R', (1, 8), '>f4'),
            ('APFLUX_IVAR_Z', (1, 8), '>f4'),
            ('APFLUX_MASKED_G', (1, 8), '>f4'),
            ('APFLUX_MASKED_R', (1, 8), '>f4'),
            ('APFLUX_MASKED_Z', (1, 8), '>f4'),
            ('APFLUX_W1', (1, 5), '>f4'),
            ('APFLUX_W2', (1, 5), '>f4'),
            ('APFLUX_W3', (1, 5), '>f4'),
            ('APFLUX_W4', (1, 5), '>f4'),
            ('APFLUX_RESID_W1', (1, 5), '>f4'),
            ('APFLUX_RESID_W2', (1, 5), '>f4'),
            ('APFLUX_RESID_W3', (1, 5), '>f4'),
            ('APFLUX_RESID_W4', (1, 5), '>f4'),
            ('APFLUX_IVAR_W1', (1, 5), '>f4'),
            ('APFLUX_IVAR_W2', (1, 5), '>f4'),
            ('APFLUX_IVAR_W3', (1, 5), '>f4'),
            ('APFLUX_IVAR_W4', (1, 5), '>f4'),
            ('MW_TRANSMISSION_G', (1,), '>f4'),
            ('MW_TRANSMISSION_R', (1,), '>f4'),
            ('MW_TRANSMISSION_Z', (1,), '>f4'),
            ('MW_TRANSMISSION_W1', (1,), '>f4'),
            ('MW_TRANSMISSION_W2', (1,), '>f4'),
            ('MW_TRANSMISSION_W3', (1,), '>f4'),
            ('MW_TRANSMISSION_W4', (1,), '>f4'),
            ('NOBS_G', (1,), '>i2'),
            ('NOBS_R', (1,), '>i2'),
            ('NOBS_Z', (1,), '>i2'),
            ('NOBS_W1', (1,), '>i2'),
            ('NOBS_W2', (1,), '>i2'),
            ('NOBS_W3', (1,), '>i2'),
            ('NOBS_W4', (1,), '>i2'),
            ('RCHISQ_G', (1,), '>f4'),
            ('RCHISQ_R', (1,), '>f4'),
            ('RCHISQ_Z', (1,), '>f4'),
            ('RCHISQ_W1', (1,), '>f4'),
            ('RCHISQ_W2', (1,), '>f4'),
            ('RCHISQ_W3', (1,), '>f4'),
            ('RCHISQ_W4', (1,), '>f4'),
            ('FRACFLUX_G', (1,), '>f4'),
            ('FRACFLUX_R', (1,), '>f4'),
            ('FRACFLUX_Z', (1,), '>f4'),
            ('FRACFLUX_W1', (1,), '>f4'),
            ('FRACFLUX_W2', (1,), '>f4'),
            ('FRACFLUX_W3', (1,), '>f4'),
            ('FRACFLUX_W4', (1,), '>f4'),
            ('FRACMASKED_G', (1,), '>f4'),
            ('FRACMASKED_R', (1,), '>f4'),
            ('FRACMASKED_Z', (1,), '>f4'),
            ('FRACIN_G', (1,), '>f4'),
            ('FRACIN_R', (1,), '>f4'),
            ('FRACIN_Z', (1,), '>f4'),
            ('ANYMASK_G', (1,), '>i2'),
            ('ANYMASK_R', (1,), '>i2'),
            ('ANYMASK_Z', (1,), '>i2'),
            ('ALLMASK_G', (1,), '>i2'),
            ('ALLMASK_R', (1,), '>i2'),
            ('ALLMASK_Z', (1,), '>i2'),
            ('WISEMASK_W1', (1,), 'uint8'),
            ('WISEMASK_W2', (1,), 'uint8'),
            ('PSFSIZE_G', (1,), '>f4'),
            ('PSFSIZE_R', (1,), '>f4'),
            ('PSFSIZE_Z', (1,), '>f4'),
            ('PSFDEPTH_G', (1,), '>f4'),
            ('PSFDEPTH_R', (1,), '>f4'),
            ('PSFDEPTH_Z', (1,), '>f4'),
            ('GALDEPTH_G', (1,), '>f4'),
            ('GALDEPTH_R', (1,), '>f4'),
            ('GALDEPTH_Z', (1,), '>f4'),
            ('NEA_G', (1,), '>f4'),
            ('NEA_R', (1,), '>f4'),
            ('NEA_Z', (1,), '>f4'),
            ('BLOB_NEA_G', (1,), '>f4'),
            ('BLOB_NEA_R', (1,), '>f4'),
            ('BLOB_NEA_Z', (1,), '>f4'),
            ('PSFDEPTH_W1', (1,), '>f4'),
            ('PSFDEPTH_W2', (1,), '>f4'),
            ('PSFDEPTH_W3', (1,), '>f4'),
            ('PSFDEPTH_W4', (1,), '>f4'),
            ('WISE_COADD_ID', (1,), '<U8'),
            ('WISE_X', (1,), '>f4'),
            ('WISE_Y', (1,), '>f4'),
            ('LC_FLUX_W1', (1, 15), '>f4'),
            ('LC_FLUX_W2', (1, 15), '>f4'),
            ('LC_FLUX_IVAR_W1', (1, 15), '>f4'),
            ('LC_FLUX_IVAR_W2', (1, 15), '>f4'),
            ('LC_NOBS_W1', (1, 15), '>i2'),
            ('LC_NOBS_W2', (1, 15), '>i2'),
            ('LC_FRACFLUX_W1', (1, 15), '>f4'),
            ('LC_FRACFLUX_W2', (1, 15), '>f4'),
            ('LC_RCHISQ_W1', (1, 15), '>f4'),
            ('LC_RCHISQ_W2', (1, 15), '>f4'),
            ('LC_MJD_W1', (1, 15), '>f8'),
            ('LC_MJD_W2', (1, 15), '>f8'),
            ('LC_EPOCH_INDEX_W1', (1, 15), '>i2'),
            ('LC_EPOCH_INDEX_W2', (1, 15), '>i2'),
            ('SERSIC', (1,), '>f4'),
            ('SERSIC_IVAR', (1,), '>f4'),
            ('SHAPE_R', (1,), '>f4'),
            ('SHAPE_R_IVAR', (1,), '>f4'),
            ('SHAPE_E1', (1,), '>f4'),
            ('SHAPE_E1_IVAR', (1,), '>f4'),
            ('SHAPE_E2', (1,), '>f4'),
            ('SHAPE_E2_IVAR', (1,), '>f4'),
            # added columns
            ('LS_ID', (1,), '>i8'),
            ('TARGETID', (1,), '>i8'),
            ]
            
        datamodel = Table()
        for col in COLS:
            datamodel[col[0]] = np.zeros(shape=col[1], dtype=col[2])

    return datamodel 

def _gather_tractorphot_onebrick(input_cat, dr9dir, radius_match, racolumn, deccolumn):
    """Support routine for gather_tractorphot."""

    assert(np.all(input_cat['BRICKNAME'] == input_cat['BRICKNAME'][0]))
    brick = input_cat['BRICKNAME'][0]
    
    idr9 = np.where((input_cat['RELEASE'] > 0) * (input_cat['BRICKID'] > 0) * (input_cat['BRICK_OBJID'] > 0))[0]
    ipos = np.delete(np.arange(len(input_cat)), idr9)

    out = Table(np.hstack(np.repeat(tractorphot_datamodel(), len(np.atleast_1d(input_cat)))))
    out['TARGETID'] = input_cat['TARGETID']
    
    # DR9 targeting photometry exists
    if len(idr9) > 0:
        assert(np.all(input_cat['PHOTSYS'][idr9] == input_cat['PHOTSYS'][idr9][0]))
    
        # find the catalog
        photsys = input_cat['PHOTSYS'][idr9][0]
    
        if photsys == 'S':
            region = 'south'
        elif photsys == 'N':
            region = 'north'
    
        #raslice = np.array(['{:06d}'.format(int(ra*1000))[:3] for ra in input_cat['RA']])
        tractorfile = os.path.join(dr9dir, region, 'tractor', brick[:3], 'tractor-{}.fits'.format(brick))
    
        if not os.path.isfile(tractorfile):
            errmsg = 'Unable to find Tractor catalog {}'.format(tractorfile)
            log.critical(errmsg)
            raise IOError(errmsg)

        # Some commissioning and SV targets can have brick_primary==False, so don't require it here.
        #<Table length=1>
        #     TARGETID     BRICKNAME BRICKID BRICK_OBJID RELEASE CMX_TARGET DESI_TARGET   SV1_DESI_TARGET   SV2_DESI_TARGET SV3_DESI_TARGET SCND_TARGET
        #      int64          str8    int32     int32     int16    int64       int64           int64             int64           int64         int64
        #----------------- --------- ------- ----------- ------- ---------- ----------- ------------------- --------------- --------------- -----------
        #39628509856927757  0352p315  503252        4109    9010          0           0 2305843009213693952               0               0           0
        #<Table length=1>
        #     TARGETID         TARGET_RA          TARGET_DEC     TILEID SURVEY PROGRAM
        #      int64            float64            float64       int32   str7    str6
        #----------------- ------------------ ------------------ ------ ------ -------
        #39628509856927757 35.333944142134406 31.496490061792002  80611    sv1  bright

        _tractor = fitsio.read(tractorfile, columns=['OBJID', 'BRICK_PRIMARY'], upper=True)
        #I = np.where(_tractor['BRICK_PRIMARY'] * np.isin(_tractor['OBJID'], input_cat['BRICK_OBJID']))[0]
        I = np.where(np.isin(_tractor['OBJID'], input_cat['BRICK_OBJID'][idr9]))[0]

        ## Some secondary programs have BRICKNAME!='' and BRICK_OBJID==0 (i.e.,
        ## not populated). However, there should always be a match here because
        ## we "repair" brick_objid in the main function.
        #if len(I) == 0: 
        #    return Table()

        tractor_dr9 = Table(fitsio.read(tractorfile, rows=I, upper=True))
    
        # sort explicitly in order to ensure order
        srt = np.hstack([np.where(objid == tractor_dr9['OBJID'])[0] for objid in input_cat['BRICK_OBJID'][idr9]])
        tractor_dr9 = tractor_dr9[srt]
        assert(np.all((tractor_dr9['BRICKID'] == input_cat['BRICKID'][idr9])*(tractor_dr9['OBJID'] == input_cat['BRICK_OBJID'][idr9])))

        tractor_dr9['LS_ID'] = np.int64(0) # will be filled in at the end
        tractor_dr9['TARGETID'] = input_cat['TARGETID'][idr9]

        out[idr9] = tractor_dr9
        del tractor_dr9
        
    # use positional matching
    if len(ipos) > 0:
        rad = radius_match * u.arcsec

        # resolve north/south
        tractorfile_north = os.path.join(dr9dir, 'north', 'tractor', brick[:3], 'tractor-{}.fits'.format(brick))
        tractorfile_south = os.path.join(dr9dir, 'south', 'tractor', brick[:3], 'tractor-{}.fits'.format(brick))
        if os.path.isfile(tractorfile_north) and not os.path.isfile(tractorfile_south):
            tractorfile = tractorfile_north
        elif not os.path.isfile(tractorfile_north) and os.path.isfile(tractorfile_south):
            tractorfile = tractorfile_south
        elif os.path.isfile(tractorfile_north) and os.path.isfile(tractorfile_south):
            if np.median(input_cat[deccolumn][ipos]) < desitarget_resolve_dec():
                tractorfile = tractorfile_south
            else:
                tractorfile = tractorfile_north
        elif not os.path.isfile(tractorfile_north) and not os.path.isfile(tractorfile_south):
            return out
                
        _tractor = fitsio.read(tractorfile, columns=['RA', 'DEC', 'BRICK_PRIMARY'], upper=True)
        iprimary = np.where(_tractor['BRICK_PRIMARY'])[0] # only primary targets
        _tractor = _tractor[iprimary] 
        coord_tractor = SkyCoord(ra=_tractor['RA']*u.deg, dec=_tractor['DEC']*u.deg)

        # Some targets can appear twice (with different targetids), so
        # to make sure we do it right, we have to loop. Example:
        #
        #     TARGETID    SURVEY PROGRAM     TARGET_RA          TARGET_DEC    OBJID BRICKID RELEASE  SKY  GAIADR    RA     DEC   GROUP BRICKNAME
        #      int64       str7    str6       float64            float64      int64  int64   int64  int64 int64  float64 float64 int64    str8
        # --------------- ------ ------- ------------------ ----------------- ----- ------- ------- ----- ------ ------- ------- ----- ---------
        # 234545047666699    sv1   other 150.31145983340912 2.587887211205909    11  345369      53     0      0     0.0     0.0     0  1503p025
        # 243341140688909    sv1   other 150.31145983340912 2.587887211205909    13  345369      55     0      0     0.0     0.0     0  1503p025

        for indx_cat, (ra, dec, targetid) in enumerate(zip(input_cat[racolumn][ipos],
                                                           input_cat[deccolumn][ipos],
                                                           input_cat['TARGETID'][ipos])):
            
            coord_cat = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
            indx_tractor, d2d, _ = coord_cat.match_to_catalog_sky(coord_tractor)
            if d2d < rad:
                _tractor = Table(fitsio.read(tractorfile, rows=iprimary[indx_tractor], upper=True))
                _tractor['LS_ID'] = np.int64(0) # will be filled in at the end
                _tractor['TARGETID'] = targetid
                out[ipos[indx_cat]] = _tractor[0]

    # Add a unique DR9 identifier.
    out['LS_ID'] = (out['RELEASE'].astype(np.int64) << 40) | (out['BRICKID'].astype(np.int64) << 16) | (out['OBJID'].astype(np.int64))

    assert(np.all(input_cat['TARGETID'] == out['TARGETID']))

    return out

def gather_tractorphot(input_cat, racolumn='TARGET_RA', deccolumn='TARGET_DEC',
                       dr9dir=None, radius_match=1.0, columns=None):
    """Retrieve the Tractor catalog for all the objects in this catalog (one brick).

    Args:
        input_cat (astropy.table.Table): input table with the following
          (required) columns: TARGETID, TARGET_RA, TARGET_DEC. Additional
          optional columns that will ensure proper matching are BRICKNAME,
          RELEASE, PHOTSYS, BRICKID, and BRICK_OBJID.
        dr9dir (str): full path to the location of the DR9 Tractor catalogs
        radius_match (float, arcsec): matching radius (default, 1 arcsec)
        columns (str array): return this subset of columns

    Returns a table of Tractor photometry. Matches are identified either using
    BRICKID and BRICK_OBJID or using positional matching (1 arcsec radius).

    """
    from desitarget.targets import decode_targetid    
    from desiutil.brick import brickname

    if len(input_cat) == 0:
        log.warning('No objects in input catalog.')
        return Table()

    for col in ['TARGETID', racolumn, deccolumn]:
        if col not in input_cat.colnames:
            errmsg = 'Missing required input column {}'.format(col)
            log.critical(errmsg)
            raise ValueError(errmsg)

    # If these columns don't exist, add them with blank entries:
    COLS = [('RELEASE', (1,), '>i2'), ('BRICKID', (1,), '>i4'), 
            ('BRICKNAME', (1,), '<U8'), ('BRICK_OBJID', (1,), '>i4'),
            ('PHOTSYS', (1,), '<U1')]
    for col in COLS:
        if col[0] not in input_cat.colnames:
            input_cat[col[0]] = np.zeros(col[1], dtype=col[2])

    if dr9dir is None:
        dr9dir = os.environ.get('DESI_ROOT')+'/external/legacysurvey/dr9'

    if not os.path.isdir(dr9dir):
        errmsg = 'DR9 directory {} not found.'.format(dr9dir)
        log.critical(errmsg)
        raise IOError(errmsg)

    ## Some secondary programs (e.g., 39632961435338613, 39632966921487347)
    ## have BRICKNAME!='' & BRICKID!=0, but BRICK_OBJID==0. Unpack those here
    ## using decode_targetid.
    #idecode = np.where((input_cat['BRICKNAME'] != '') * (input_cat['BRICK_OBJID'] == 0))[0]
    #if len(idecode) > 0:
    #    log.debug('Inferring BRICK_OBJID for {} objects using decode_targetid'.format(len(idecode)))
    #    new_objid, new_brickid, _, _, _, _ = decode_targetid(input_cat['TARGETID'][idecode])
    #    try:
    #        assert(np.all(new_brickid == input_cat['BRICKID'][idecode]))
    #    except:
    #        pdb.set_trace()
    #    input_cat['BRICK_OBJID'][idecode] = new_objid

    # BRICKNAME can sometimes be blank; fix that here. NB: this step has to come
    # *after* the decode step, above!
    inobrickname = np.where(input_cat['BRICKNAME'] == '')[0]
    if len(inobrickname) > 0:
        log.debug('Inferring brickname for {:,} objects'.format(len(inobrickname)))
        input_cat['BRICKNAME'][inobrickname] = brickname(input_cat[racolumn][inobrickname],
                                                         input_cat[deccolumn][inobrickname])

    # Split into unique brickname(s).
    bricknames = input_cat['BRICKNAME']

    out = Table(np.hstack(np.repeat(tractorphot_datamodel(), len(np.atleast_1d(input_cat)))))
    for brickname in set(bricknames):
        I = np.where(brickname == bricknames)[0]
        out[I] = _gather_tractorphot_onebrick(input_cat[I], dr9dir, radius_match, racolumn, deccolumn)

    if columns is not None:
        out = out[columns]

    return out
