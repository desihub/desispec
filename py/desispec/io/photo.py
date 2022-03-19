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

from desiutil.log import get_logger, DEBUG
log = get_logger()#DEBUG)

def get_targetdirs(tileid, fiberassign_dir=None):
    """Get all the targeting directories used to build a given fiberassign catalog.

    Args:
        tileid (int): tile number
        fiberassign_dir (str, optional): directory to fiberassign tables

    Returns a list of all the unique targeting directories.

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
            log.warning('Fiber assignment file {} not found!'.format(fiberfile))
    log.debug('Reading {} header.'.format(fiberfile))
    # old versions of fitsio can't handle CONTINUE header cards!
    #fahdr = fitsio.read_header(fiberfile, ext=0)
    fahdr = fits.getheader(fiberfile, ext=0)

    # Gather the targeting directories.
    targetdirs = [fahdr['TARG']]
    for moretarg in ['TARG2', 'TARG3', 'TARG4']:
        if moretarg in fahdr:
            if 'gaia' not in fahdr[moretarg]: # skip if we have it already
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
        if os.path.isfile(TOOfile):
            targetdirs += [TOOfile]
    
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
            
        if os.path.isdir(targetdir) or os.path.isfile(targetdir):
            log.debug('Found targets directory or file {}'.format(targetdir))
            targetdirs[ii] = targetdir
        else:
            log.warning('Targets directory or file {} not found.'.format(targetdir))
            continue

    targetdirs = np.unique(np.hstack(targetdirs))
        
    return targetdirs

def _targetphot_datamodel():
    """Initialize the targetphot data model by reading a nominal targeting catalog.

    """
    #from pkg_resources import resource_filename
    
    #datamodel_file = resource_filename('desitarget.test', 't/targets.fits')
    datamodel_file = os.path.join(os.environ.get('DESI_ROOT'), 'target', 'catalogs', 'dr9', '1.1.1', 'targets',
                                  'main', 'resolve', 'dark', 'targets-dark-hp-0.fits')
    if not os.path.isfile(datamodel_file):
        log.warning('Unable to establish the data model using {}'.format(datamodel_file))
        raise IOError

    TARGETINGBITCOLS = [
        'CMX_TARGET',
        'DESI_TARGET', 'BGS_TARGET', 'MWS_TARGET',
        'SV1_DESI_TARGET', 'SV1_BGS_TARGET', 'SV1_MWS_TARGET',
        'SV2_DESI_TARGET', 'SV2_BGS_TARGET', 'SV2_MWS_TARGET',
        'SV3_DESI_TARGET', 'SV3_BGS_TARGET', 'SV3_MWS_TARGET',
        'SCND_TARGET',
        'SV1_SCND_TARGET', 'SV2_SCND_TARGET', 'SV3_SCND_TARGET',
        ]

    datamodel = Table(fitsio.read(datamodel_file, rows=0, upper=True))
    for col in datamodel.colnames:
        if '_TARGET' in col:
            datamodel.remove_column(col)
        else:
            datamodel[col] = np.zeros(datamodel[col].shape, dtype=datamodel[col].dtype)
    for col in TARGETINGBITCOLS:
        datamodel[col] = np.zeros(1, dtype=np.int64)
        
    return datamodel

def build_targetphot(input_cat, photocache=None, racolumn='TARGET_RA', deccolumn='TARGET_DEC'):
    """Find and stack the photometric targeting information given a set of targets.

    Args:
        input_cat (astropy.table.Table): input table with the following
          (required) columns: TARGETID, TILEID, RACOLUMN, DECCOLUMN

    Returns a table of targeting photometry using a consistent data model across
    primary (DR9) targets, secondary targets, and targets of opportunity.

    """
    import astropy
    from desimodel.footprint import radec2pix

    if len(input_cat) == 0:
        log.warning('No objects in input catalog.')
        return Table()

    for col in ['TARGETID', 'TILEID', racolumn, deccolumn]:
        if col not in input_cat.colnames:
            log.warning('Missing required input column {}'.format(col))
            raise ValueError

    # Get the unique list of targetdirs
    targetdirs = np.unique(np.hstack([get_targetdirs(tileid) for tileid in set(input_cat['TILEID'])]))
    
    datamodel = _targetphot_datamodel()
    out = Table(np.hstack(np.repeat(datamodel, len(input_cat))))
    out['TARGETID'] = input_cat['TARGETID']

    photo, photofiles = [], []
    for targetdir in targetdirs:
        # Handle secondary targets, which have a (very!) different data model.
        if 'secondary' in targetdir:
            if 'sv1' in targetdir: # special case
                if 'dedicated' in targetdir:
                    targetfiles = glob(os.path.join(targetdir, 'DC3R2_GAMA_priorities.fits'))
                else:
                    targetfiles = glob(os.path.join(targetdir, '*-secondary-dr9photometry.fits'))
            else:
                targetfiles = glob(os.path.join(targetdir, '*-secondary.fits'))
        elif 'ToO' in targetdir:
            targetfiles = targetdir
        else:
            alltargetfiles = glob(os.path.join(targetdir, '*-hp-*.fits'))
            filenside = fitsio.read_header(alltargetfiles[0], ext=1)['FILENSID']
            # https://github.com/desihub/desispec/issues/1711
            if np.any(np.isnan(input_cat[racolumn])): # some SV1 targets have nan in RA,DEC
                log.warning('Some RA, DEC are NaN in target directory {}'.format(targetdir))
            notnan = np.isfinite(input_cat[racolumn])
            targetfiles = []
            if np.sum(notnan) > 0:
                pixlist = radec2pix(filenside, input_cat[racolumn][notnan], input_cat[deccolumn][notnan])
                for pix in set(pixlist):
                    # /global/cfs/cdirs/desi/target/catalogs/gaiadr2/0.48.0/targets/sv1/resolve/supp/sv1targets-supp-hp-128.fits doesn't exist...
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
            # copy what exists, e.g.,
            # /global/cfs/cdirs/desi/spectro/redux/everest/healpix/sv3/bright/153/15343/redrock-sv3-bright-15343.fits
            if photocache is not None and targetfile in photocache.keys():
                if type(photocache[targetfile]) == astropy.table.Table:
                    I = np.where(np.isin(photocache[targetfile]['TARGETID'], input_cat['TARGETID']))[0]
                else:
                    photo_targetid = photocache[targetfile]
                    I = np.where(np.isin(photo_targetid, input_cat['TARGETID']))[0]
                    
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
                I = np.where(np.isin(photo1['TARGETID'], input_cat['TARGETID']))[0]
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
            I = np.where(np.isin(photo_targetid, input_cat['TARGETID']))[0]
            
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
                        log.debug('Skipping missing column {} from {}'.format(col, targetfile))
                del photo1
                photofiles.append(targetfile)
                photo.append(_photo)

    # backup programs have no target catalog photometry at all
    if len(photo) == 0:
        log.warning('No photometry found at all!')
        photo = [out] # empty set

    # np.hstack will sometimes complain even if the tables are identical...
    #photo = Table(np.hstack(photo))
    photo = vstack(photo)

    # make sure there are no duplicates...?
    _, uindx = np.unique(photo['TARGETID'], return_index=True) 
    photo = photo[uindx]
    assert(len(np.unique(photo['TARGETID'])) == len(photo))

    # sort explicitly in order to ensure order
    I = np.where(np.isin(out['TARGETID'], photo['TARGETID']))[0]
    srt = np.hstack([np.where(tid == photo['TARGETID'])[0] for tid in out['TARGETID'][I]])
    out[I] = photo[srt]
    
    return out
