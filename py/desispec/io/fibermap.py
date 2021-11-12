"""
desispec.io.fibermap
====================

IO routines for fibermap.
"""
import os
import sys
import glob
import warnings
import time
from pkg_resources import resource_filename
import fitsio

import yaml

import numpy as np
from astropy.table import Table, Column, join
from astropy.io import fits

from desitarget.targetmask import desi_mask
from desitarget.skybricks import Skybricks
from desiutil.log import get_logger
from desiutil.depend import add_dependencies, mergedep
from desimodel.focalplane import get_tile_radius_deg

from desispec.io.util import fitsheader, write_bintable, makepath, addkeys, parse_badamps
from desispec.io.meta import rawdata_root, findfile
from . import iotime

from desispec.maskbits import fibermask

#- Subset of columns that come from original target/MTL catalog
target_columns = [
    ('TARGETID',    'i8', '', 'Unique target ID'),
    ('DESI_TARGET', 'i8', '', 'Dark survey + calibration targeting bits'),
    ('BGS_TARGET',  'i8', '', 'Bright Galaxy Survey targeting bits'),
    ('MWS_TARGET',  'i8', '', 'Milky Way Survey targeting bits'),
    ('SCND_TARGET', 'i8', '', 'Secondary program targeting bits'),
    #- TBD: COMM_TARGET, SVn_TARGET, ...
    ('TARGET_RA',   'f8', 'degree', 'Target Right Ascension [degrees]'),
    ('TARGET_DEC',  'f8', 'degree', 'Target declination [degrees]'),
    ('PLATE_RA',    'f8', 'degree', 'Right Ascension for Platemaker to use [degrees]'),
    ('PLATE_DEC',   'f8', 'degree', 'declination for Platemaker to use [degrees]'),
    # ('TARGET_RA_IVAR', 'f8', '1/degree**2', 'Inverse variance of TARGET_RA'),
    # ('TARGET_DEC_IVAR', 'f8','1/degree**2', 'Inverse variance of TARGET_DEC'),
    ('BRICKID',     'i8', '', 'Imaging Surveys brick ID'),
    ('BRICKNAME',  (str, 8), '', 'Imaging Surveys brick name'),
    ('BRICK_OBJID', 'i8', '', 'Imaging Surveys OBJID on that brick'),
    ('MORPHTYPE', (str, 4), '', 'Imaging Surveys morphological type'),
    ('PRIORITY',    'i4', '', 'Assignment priority; larger=higher priority'),
    ('SUBPRIORITY', 'f8', '', 'Assignment subpriority [0-1)'),
    ('REF_ID',      'i8', '', 'Astrometric cat refID (Gaia SOURCE_ID)'),
    ('MASKBITS',    'i2', '', 'Photometry mask bits'),
    ('PMRA',        'f4', 'marcsec/year', 'PM in +RA dir (already incl cos(dec))'),
    ('PMDEC',       'f4', 'marcsec/year', 'Proper motion in +dec direction'),
    ('PARALLAX',    'f4', 'marcsec', 'Parallax'),
    ('REF_CAT',     (str, 2), '', 'astrometry reference catalog'),
    ('REF_EPOCH',   'f4', '', 'proper motion reference epoch'),
    # ('PMRA_IVAR',   'f4', 'year**2/marcsec**2', 'Inverse variance of PMRA'),
    # ('PMDEC_IVAR',  'f4', 'year**2/marcsec**2', 'Inverse variance of PMDEC'),
    ('RELEASE',     'i2', '', 'imaging surveys release ID'),
    ('FLUX_G',      'f4', 'nanomaggies', 'g-band flux'),
    ('FLUX_R',      'f4', 'nanomaggies', 'r-band flux'),
    ('FLUX_Z',      'f4', 'nanomaggies', 'z-band flux'),
    ('FLUX_W1',     'f4', 'nanomaggies', 'WISE W1-band flux'),
    ('FLUX_W2',     'f4', 'nanomaggies', 'WISE W2-band flux'),
    ('FLUX_IVAR_G', 'f4', '1/nanomaggies**2', 'Inverse variance of FLUX_G'),
    ('FLUX_IVAR_R', 'f4', '1/nanomaggies**2', 'Inverse variance of FLUX_R'),
    ('FLUX_IVAR_Z', 'f4', '1/nanomaggies**2', 'Inverse variance of FLUX_Z'),
    ('FLUX_IVAR_W1','f4', '1/nanomaggies**2', 'Inverse variance of FLUX_W1'),
    ('FLUX_IVAR_W2','f4', '1/nanomaggies**2', 'Inverse variance of FLUX_W2'),
    ('FIBERFLUX_G', 'f4', 'nanomaggies', 'g-band model flux 1" seeing, 1.5" dia fiber'),
    ('FIBERFLUX_R', 'f4', 'nanomaggies', 'r-band model flux 1" seeing, 1.5" dia fiber'),
    ('FIBERFLUX_Z', 'f4', 'nanomaggies', 'z-band model flux 1" seeing, 1.5" dia fiber'),
    # ('FIBERFLUX_W1', 'f4', 'nanomaggies', 'W1-band model flux 1" seeing, 1.5" dia fiber'),
    # ('FIBERFLUX_W2', 'f4', 'nanomaggies', 'W2-band model flux 1" seeing, 1.5" dia fiber'),
    ('FIBERTOTFLUX_G', 'f4', 'nanomaggies', 'fiberflux model incl. all objs at this loc'),
    ('FIBERTOTFLUX_R', 'f4', 'nanomaggies', 'fiberflux model incl. all objs at this loc'),
    ('FIBERTOTFLUX_Z', 'f4', 'nanomaggies', 'fiberflux model incl. all objs at this loc'),
    # ('FIBERTOTFLUX_W1', 'f4', 'nanomaggies', 'fiberflux model incl. all objs at this loc'),
    # ('FIBERTOTFLUX_W2', 'f4', 'nanomaggies', 'fiberflux model incl. all objs at this loc'),
    ('GAIA_PHOT_G_MEAN_MAG',      'f4', 'mag', 'Gaia G band mag'),
    ('GAIA_PHOT_BP_MEAN_MAG',      'f4', 'mag', 'Gaia BP band mag'),
    ('GAIA_PHOT_RP_MEAN_MAG',      'f4', 'mag', 'Gaia RP band mag'),
    # ('MW_TRANSMISSION_G', 'f4', '', 'Milky Way dust transmission in g [0-1]'),
    # ('MW_TRANSMISSION_R', 'f4', '', 'Milky Way dust transmission in r [0-1]'),
    # ('MW_TRANSMISSION_Z', 'f4', '', 'Milky Way dust transmission in z [0-1]'),
    ('SERSIC', 'f4', '', 'Power-law index for the Sersic profile model'),
    ('SHAPE_R', 'f4', 'arcsec', 'Half-light radius of galaxy model'),
    ('SHAPE_E1', 'f4', '', 'Ellipticity component 1 for galaxy model'),
    ('SHAPE_E2', 'f4', '', 'Ellipticity component 2 for galaxy model'),
    ('EBV', 'f4', '', 'Galactic extinction E(B-V) reddening from SFD98'),
    ('PHOTSYS', (str, 1), '', 'N for BASS/MzLS, S for DECam'),
    ('OBSCONDITIONS', 'i4', '', 'bitmask of allowable observing conditions'),
    ('NUMOBS_INIT', 'i8', '', 'initial number of requested observations'),
    ('PRIORITY_INIT', 'i8', '', 'initial priority'),
    # ('NUMOBS_MORE', 'i4', '', 'current number of additional obs requested'),
    # ('HPXPIXEL', 'i8', '', 'Healpix pixel number (NESTED)')
]

### Some additional columns from targeting that I'm not including here yet
### because we don't use them in the pipeline and they may continue to evolve
# DCHISQ              f4  array[5]
# FRACFLUX_G          f4
# FRACFLUX_R          f4
# FRACFLUX_Z          f4
# FRACMASKED_G        f4
# FRACMASKED_R        f4
# FRACMASKED_Z        f4
# FRACIN_G            f4
# FRACIN_R            f4
# FRACIN_Z            f4
# NOBS_G              i2
# NOBS_R              i2
# NOBS_Z              i2
# PSFDEPTH_G          f4
# PSFDEPTH_R          f4
# PSFDEPTH_Z          f4
# GALDEPTH_G          f4
# GALDEPTH_R          f4
# GALDEPTH_Z          f4
# FLUX_W3             f4
# FLUX_W4             f4
# FLUX_IVAR_W3        f4
# FLUX_IVAR_W4        f4
# MW_TRANSMISSION_W1
#                     f4
# MW_TRANSMISSION_W2
#                     f4
# MW_TRANSMISSION_W3
#                     f4
# MW_TRANSMISSION_W4
#                     f4
# ALLMASK_G           i2
# ALLMASK_R           i2
# ALLMASK_Z           i2
# FRACDEV             f4
# FRACDEV_IVAR        f4
# SHAPEDEV_R          f4
# SHAPEDEV_E1         f4
# SHAPEDEV_E2         f4
# SHAPEDEV_R_IVAR     f4
# SHAPEDEV_E1_IVAR
#                     f4
# SHAPEDEV_E2_IVAR
#                     f4
# SHAPEEXP_R          f4
# SHAPEEXP_E1         f4
# SHAPEEXP_E2         f4
# SHAPEEXP_R_IVAR     f4
# SHAPEEXP_E1_IVAR
#                     f4
# SHAPEEXP_E2_IVAR
#                     f4
# WISEMASK_W1         u1
# WISEMASK_W2         u1
# MASKBITS            i2
# REF_ID              i8
# REF_CAT             S2
# GAIA_PHOT_G_MEAN_MAG
#                     f4
# GAIA_PHOT_G_MEAN_FLUX_OVER_ERROR
#                     f4
# GAIA_PHOT_BP_MEAN_MAG
#                     f4
# GAIA_PHOT_BP_MEAN_FLUX_OVER_ERROR
#                     f4
# GAIA_PHOT_RP_MEAN_MAG
#                     f4
# GAIA_PHOT_RP_MEAN_FLUX_OVER_ERROR
#                     f4
# GAIA_PHOT_BP_RP_EXCESS_FACTOR
#                     f4
# GAIA_ASTROMETRIC_EXCESS_NOISE
#                     f4
# GAIA_DUPLICATED_SOURCE
#                     b1
# GAIA_ASTROMETRIC_SIGMA5D_MAX
#                     f4
# GAIA_ASTROMETRIC_PARAMS_SOLVED
#                     b1
# PARALLAX            f4
# PARALLAX_IVAR       f4
# BLOBDIST            f4


#- Columns added by fiberassign
fiberassign_columns = target_columns.copy()
fiberassign_columns.extend([
    ('FIBER',       'i4', '', 'Fiber ID on the CCDs [0-4999]'),
    ('PETAL_LOC',   'i4', '', 'Petal location [0-9]'),
    ('DEVICE_LOC',  'i4', '', 'Device location on focal plane [0-523]'),
    ('LOCATION',    'i4', '', 'FP location PETAL_LOC*1000 + DEVICE_LOC'),
    ('FIBERSTATUS', 'i4', '', 'Fiber status; 0=good'),
    ('OBJTYPE', (str, 3), '', 'SKY, TGT, NON'),
    ('LAMBDA_REF',  'f4', 'Angstrom', 'Wavelength at which fiber was centered'),
    ('FIBERASSIGN_X',    'f4', 'mm', 'Expected CS5 X on focal plane'),
    ('FIBERASSIGN_Y',    'f4', 'mm', 'Expected CS5 Y on focal plane'),
    ('FA_TARGET',   'i8', '', ''),
    ('FA_TYPE',     'u1', '', 'Internal fiberassign target type'),
    # ('DESIGN_Q',    'f4', 'deg', 'Expected CS5 Q azimuthal coordinate'),
    # ('DESIGN_S',    'f4', 'mm', 'Expected CS5 S radial distance along curved focal surface'),
    # ('NUMTARGET',   'i2', '', 'Number of targets covered by positioner'),
])

#- Columns added by ICS for final fibermap
fibermap_columns = fiberassign_columns.copy()
fibermap_columns.extend([
    ('FIBER_RA',        'f8', 'degree', 'RA of actual fiber position'),
    ('FIBER_DEC',       'f8', 'degree', 'DEC of actual fiber position'),
    # ('FIBER_RA_IVAR',   'f4', '1/degree**2', 'Inverse variance of FIBER_RA [not set yet]'),
    # ('FIBER_DEC_IVAR',  'f4', '1/degree**2', 'Inverse variance of FIBER_DEC [not set yet]'),
    ('FIBER_X',    'f4', 'mm', 'CS5 X location requested by PlateMaker'),
    ('FIBER_Y',    'f4', 'mm', 'CS5 Y location requested by PlateMaker'),
    ('DELTA_X',    'f4', 'mm', 'CS5 X diff requested and actual position'),
    ('DELTA_Y',    'f4', 'mm', 'CS5 Y diff requested and actual position'),
    # ('DELTA_X_IVAR',    'f4', '1/mm**2', 'Inverse variance of DELTA_X [not set yet]'),
    # ('DELTA_Y_IVAR',    'f4', '1/mm**2', 'Inverse variance of DELTA_Y [not set yet]'),
    ('NUM_ITER',        'i4', '', 'Number of positioner iterations'),
    ('EXPTIME','f4','s','Exposure time'),
])

#- fibermap_comments[colname] = 'comment to include in FITS header'
fibermap_comments = dict([(tmp[0], tmp[3]) for tmp in fibermap_columns])
fibermap_dtype = [tmp[0:2] for tmp in fibermap_columns]

def empty_fibermap(nspec, specmin=0):
    """Return an empty fibermap Table to be filled in.

    Args:
        nspec: (int) number of fibers(spectra) to include

    Options:
        specmin: (int) starting spectrum index
    """
    import desimodel.io

    assert 0 <= nspec <= 5000, "nspec {} should be within 0-5000".format(nspec)
    fibermap = Table()
    for (name, dtype, unit, comment) in fibermap_columns:
        c = Column(name=name, dtype=dtype, unit=unit, length=nspec)
        fibermap.add_column(c)

    #- Fill in some values
    fibermap['FIBER'][:] = np.arange(specmin, specmin+nspec)
    fibers_per_spectrograph = 500
    ### fibermap['SPECTROID'][:] = fibermap['FIBER'] // fibers_per_spectrograph

    fiberpos = desimodel.io.load_focalplane()[0]
    fiberpos = fiberpos[fiberpos['DEVICE_TYPE'] == 'POS']
    fiberpos.sort('FIBER')

    ii = slice(specmin, specmin+nspec)
    fibermap['FIBERASSIGN_X'][:]   = fiberpos['OFFSET_X'][ii]
    fibermap['FIBERASSIGN_Y'][:]   = fiberpos['OFFSET_Y'][ii]
    fibermap['LOCATION'][:]   = fiberpos['LOCATION'][ii]
    fibermap['PETAL_LOC'][:]  = fiberpos['PETAL'][ii]
    fibermap['DEVICE_LOC'][:] = fiberpos['DEVICE'][ii]
    fibermap['LAMBDA_REF'][:]  = 5400.0
    fibermap['NUM_ITER'][:] = 2
    #- Set MW_TRANSMISSION_* to be slightly less than 1 to trigger dust correction code for testing
    ### fibermap['MW_TRANSMISSION_G'][:] = 0.999
    ### fibermap['MW_TRANSMISSION_R'][:] = 0.999
    ### fibermap['MW_TRANSMISSION_Z'][:] = 0.999
    fibermap['EBV'][:] = 0.001
    fibermap['PHOTSYS'][:] = 'S'

    fibermap.meta['EXTNAME'] = 'FIBERMAP'

    assert set(fibermap.keys()) == set([x[0] for x in fibermap_columns])

    return fibermap

def write_fibermap(outfile, fibermap, header=None, clobber=True, extname='FIBERMAP'):
    """Write fibermap binary table to outfile.

    Args:
        outfile (str): output filename
        fibermap: astropy Table of fibermap data
        header: header data to include in same HDU as fibermap
        clobber (bool, optional): overwrite outfile if it exists
        extname (str, optional): set the extension name.

    Returns:
        write_fibermap (str): full path to filename of fibermap file written.
    """
    log = get_logger()
    outfile = makepath(outfile)

    #- astropy.io.fits incorrectly generates warning about 2D arrays of strings
    #- Temporarily turn off warnings to avoid this; desispec.test.test_io will
    #- catch it if the arrays actually are written incorrectly.
    if header is not None:
        hdr = fitsheader(header)
    else:
        hdr = fitsheader(fibermap.meta)

    add_dependencies(hdr)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.time()
        write_bintable(outfile, fibermap, hdr, comments=fibermap_comments,
                       extname=extname, clobber=clobber)
        duration = time.time() - t0

    log.info(iotime.format('write', outfile, duration))

    return outfile


def read_fibermap(filename):
    """Reads a fibermap file and returns its data as an astropy Table

    Args:
        filename : input file name
    """
    #- Implementation note: wrapping fitsio.read() with this function allows us
    #- to update the underlying format, extension name, etc. without having
    #- to change every place that reads a fibermap.
    log = get_logger()
    t0 = time.time()

    fibermap, hdr = fitsio.read(filename, ext='FIBERMAP', header=True)
    fibermap = Table(fibermap)
    addkeys(fibermap.meta, hdr)

    duration = time.time() - t0

    #- support old simulated fiberassign files
    if 'DESIGN_X' in fibermap.colnames:
        fibermap.rename_column('DESIGN_X', 'FIBERASSIGN_X')
    if 'DESIGN_Y' in fibermap.colnames:
        fibermap.rename_column('DESIGN_Y', 'FIBERASSIGN_Y')

    log.info(iotime.format('read', filename, duration))

    return fibermap

def find_fiberassign_file(night, expid, tileid=None, nightdir=None):
    """
    Walk backwards in exposures to find matching fiberassign file

    Args:
        night (int): YEARMMDD night of observations
        expid (int): spectroscopic exposure ID

    Options:
        tileid (int): tileid to look for
        nightdir (str): base directory for raw data on that night

    Returns first fiberassign file found on or before `expid` on `night`.

    Raises FileNotFoundError if no fiberassign file is found
    """
    log = get_logger()
    if nightdir is None:
        nightdir = os.path.join(rawdata_root(), str(night))

    expdir = f'{nightdir}/{expid:08d}'

    if tileid is not None:
        faglob = nightdir+'/*/fiberassign-{:06d}.fits*'.format(tileid)
    else:
        faglob = nightdir+'/*/fiberassign*.fits*'

    fafile = None
    for filename in sorted(glob.glob(faglob)):
        if filename.endswith('.fits.gz') or filename.endswith('.fits'):
            dirname = os.path.dirname(filename)
            if dirname <= expdir:
                fafile = filename
            else:
                break
        else:
            log.debug(f'Ignoring {filename}')

    if fafile is None:
        raise FileNotFoundError(
                f'Unable to find fiberassign on {night} prior to {expid}')

    return fafile

def compare_fiberassign(fa1, fa2, compare_all=False):
    """
    Check whether two fiberassign tables agree for cols used by ICS/platemaker

    Args:
        fa1, fa2: fiberassign astropy Tables or numpy structured arrays

    Options:
        compare_all: if True, compare all columns, not just those used by ops

    Returns list of columns with mismatches; empty list if all agree

    Note: if both are NaN, it is considered a match
    """
    if compare_all:
        compare_columns = fa1.dtype.names
    else:
        compare_columns = (
                'TARGETID', 'TARGET_RA', 'TARGET_DEC',
                'PMRA', 'PMDEC', 'REF_EPOCH', 'LAMBDA_REF', 'LOCATION',
                'PETAL_LOC', 'DEVICE_LOC')

        badcol = list()
        for col in compare_columns:

            #- check for mismatches, but allow for NaN
            match = (fa1[col] == fa2[col])
            if fa1[col].dtype.kind == 'f':
                match |= (np.isnan(fa1[col]) & np.isnan(fa2[col]))

            #- negative TARGETID are allowed to mismatch if both are negative
            if col == 'TARGETID':
                match |= ((fa1[col]<0) & (fa2[col]<0))

            if np.any(~match):
                badcol.append(col)

        return badcol

def assemble_fibermap(night, expid, badamps=None, badfibers_filename=None,
                      force=False, allow_svn_override=True):
    """
    Create a fibermap for a given night and expid

    Args:
        night (int): YEARMMDD night of sunset
        expid (int): exposure ID

    Options:
        badamps (str): comma separated list of "{camera}{petal}{amp}", i.e. "[brz][0-9][ABCD]". Example: 'b7D,z8A'
        badfibers_filename (str): filename with table of bad fibers with at least two columns: FIBER and FIBERSTATUS
        force (bool): create fibermap even if missing coordinates/guide files
        allow_svn_override (bool): if True (default), allow fiberassign SVN to override raw data

    Returns:
        fibermap (HDUList): A representation of a fibermap FITS file.
    """

    log = get_logger()

    #- raw data file for header
    rawfile = findfile('raw', night, expid)
    try:
        rawheader = fits.getheader(rawfile, 'SPEC', disable_image_compression=True)
    except KeyError:
        rawheader = fits.getheader(rawfile, 'SPS', disable_image_compression=True)

    #- Find fiberassign file
    rawfafile = fafile = find_fiberassign_file(night, expid)

    #- Look for override fiberassign file in svn
    try:
        tileid = rawheader['TILEID']
    except KeyError:
        #
        # 20210114/00072405 is an example of where this happens.
        # We use the fiberassign file that find_fiberassign_file() found
        # by walking backwards through earlier exposures on the same night,
        # *even if it is actually wrong*.
        #
        log.error("TILEID not in the header from %s!", rawfile)
        rawfafile_expid = int(os.path.basename(os.path.dirname(rawfafile)))
        if rawfafile_expid != expid:
            log.error('Determining TILEID from an earlier exposure %08d/%08d!', night, rawfafile_expid)
        tileid = int(os.path.basename(rawfafile).split('-')[1].split('.')[0])
    if allow_svn_override and ('DESI_TARGET' in os.environ):
        targdir = os.getenv('DESI_TARGET')
        testfile = f'{targdir}/fiberassign/tiles/trunk/{tileid//1000:03d}/fiberassign-{tileid:06d}.fits'
        if os.path.exists(testfile+'.gz'):
            fafile = testfile+'.gz'
        elif os.path.exists(testfile):
            fafile = testfile

        if rawfafile != fafile:
            log.info(f'Overriding raw fiberassign file {rawfafile} with svn {fafile}')
        else:
            log.info(f'{testfile}[.gz] not found; sticking with raw data fiberassign file')

    #- Find coordinates file in same directory
    dirname, filename = os.path.split(rawfafile)
    globfiles = glob.glob(dirname+'/coordinates-*.fits')
    if len(globfiles) == 1:
        coordfile = globfiles[0]
    elif len(globfiles) == 0:
        message = f'No coordinates*.fits file in fiberassign dir {dirname}'
        if force:
            log.error(message + '; continuing anyway')
            coordfile = None
        else:
            raise FileNotFoundError(message)

    elif len(globfiles) > 1:
        raise RuntimeError(
            f'Multiple coordinates*.fits files in fiberassign dir {dirname}')

    #- And guide file
    dirname, filename = os.path.split(rawfafile)
    globfiles = glob.glob(dirname+'/guide-????????.fits.fz')
    if len(globfiles) == 0:
        #- try falling back to acquisition image
        globfiles = glob.glob(dirname+'/guide-????????-0000.fits.fz')

    if len(globfiles) == 1:
        guidefile = globfiles[0]
    elif len(globfiles) == 0:
        message = f'No guide-*.fits.fz file in fiberassign dir {dirname}'
        if force:
            log.error(message + '; continuing anyway')
            guidefile = None
        else:
            raise FileNotFoundError(message)

    elif len(globfiles) > 1:
        raise RuntimeError(
            f'Multiple guide-*.fits.fz files in fiberassign dir {dirname}')

    #- Read QA parameters to find max offset for POOR and BAD positioning
    #- replicates desispec.exposure_qa.get_qa_params, but that has
    #- circular import if loaded from here
    param_filename = resource_filename('desispec', 'data/qa/qa-params.yaml')
    with open(param_filename) as f:
        qa_params = yaml.safe_load(f)['exposure_qa']

    poor_offset_um = qa_params['poor_fiber_offset_mm']*1000
    bad_offset_um = qa_params['bad_fiber_offset_mm']*1000

    #- Preflight announcements
    log.info(f'Night {night} spectro expid {expid}')
    log.info(f'Raw data file {rawfile}')
    log.info(f'Fiberassign file {fafile}')
    if fafile != rawfafile:
        log.info(f'Original raw fiberassign file {rawfafile}')
    log.info(f'Platemaker coordinates file {coordfile}')
    log.info(f'Guider file {guidefile}')

    #----
    #- Read and assemble

    fa = Table.read(fafile, 'FIBERASSIGN')
    fa.sort('LOCATION')
    fa_header = fits.getheader(fafile, 'FIBERASSIGN')
    #
    # Obtain the survey name.  Reproduces the logic (though not the exact code) in
    # desitarget.targets.main_cmx_or_sv(), but can be much faster.
    # As a bonus, a freeze_iers() call can be avoided.
    #
    # survey = main_cmx_or_sv(fa)[2]
    survey = 'main'
    notmain = [name.split('_')[0] for name in fa.colnames
               if name.startswith('CMX') or name.startswith('SV')]
    if len(notmain) > 0:
        survey = notmain[0].lower()
    log.info("Fiberassign file reports survey = '%s'.", survey)

    #- if using svn fiberassign override, check consistency of columns that
    #- ICS / platemaker used for actual observations; they should never change
    if fafile != rawfafile:
        rawfa = Table.read(rawfafile, 'FIBERASSIGN')
        rawfa.sort('LOCATION')
        badcol = compare_fiberassign(rawfa, fa)

        #- special case for tile 80713 on 20210110 with PMRA,PMDEC NaN -> 0.0
        if night == 20210110 and tileid == 80713:
            for col in ['PMRA', 'PMDEC']:
                if col in badcol:
                    ii = rawfa[col] != fa[col]
                    if np.all(np.isnan(rawfa[col][ii]) & (fa[col][ii] == 0.0)):
                        log.warning(f'Ignoring {col} mismatch NaN -> 0.0 on tile {tileid} night {night}')
                        badcol.remove(col)

        if len(badcol)>0:
            msg = f'incompatible raw/svn fiberassign files for columns {badcol}'
            log.critical(msg)
            raise ValueError(msg)
        else:
            log.info('svn fiberassign columns used for obervations match raw data (good)')

    #- Tiles designed before Fall 2021 had a LOCATION:FIBER swap for fibers
    #- 3402 and 3429 at locations 6098 and 6099; check and correct if needed.
    #- see desispec #1380
    #- NOTE: this only swaps them if the incorrect combination is found
    if (6098 in fa['LOCATION']) and (6099 in fa['LOCATION']):
        iloc6098 = np.where(fa['LOCATION'] == 6098)[0][0]
        iloc6099 = np.where(fa['LOCATION'] == 6099)[0][0]
        if (fa['FIBER'][iloc6098] == 3402) and (fa['FIBER'][iloc6099] == 3429):
            log.warning(f'FIBERS 3402 and 3429 are swapped in {fafile}; correcting')
            fa['FIBER'][iloc6098] = 3429
            fa['FIBER'][iloc6099] = 3402

    #- add missing columns for data model consistency
    if 'PLATE_RA' not in fa.colnames:
        log.debug("Adding PLATE_RA column.")
        fa['PLATE_RA'] = fa['TARGET_RA']

    if 'PLATE_DEC' not in fa.colnames:
        log.debug("Adding PLATE_DEC column.")
        fa['PLATE_DEC'] = fa['TARGET_DEC']

    #- also read extra keywords from HDU 0
    fa_hdr0 = fits.getheader(fafile, 0)
    if 'OUTDIR' in fa_hdr0:
        log.debug("Rename OUTDIR -> FAOUTDIR")
        fa_hdr0.rename_keyword('OUTDIR', 'FAOUTDIR')
    longstrn = fits.Card('LONGSTRN', 'OGIP 1.0', 'The OGIP Long String Convention may be used.')
    if 'DEPNAM00' in fa_hdr0:
        log.debug("Inserting LONGSTRN keyword before DEPNAM00")
        fa_hdr0.insert('DEPNAM00', longstrn)
    else:
        log.debug("Inserting LONGSTRN keyword before TILEID")
        fa_hdr0.insert('TILEID', longstrn)

    #
    # Merge fa_hdr0 into fa_header.  unique=True means prefer cards from
    # fa_header over fa_hdr0
    #
    if 'DEPNAM00' in fa_header:
        log.debug("Merge dependencies from HDU0 into fa_header.")
        mergedep(fa_hdr0, fa_header, conflict='dst')
    fa_header.extend(fa_hdr0, unique=True)

    #- Read platemaker (pm) coordinates file; 3 formats to support:
    #  1. has FLAGS_CNT/EXP_n and DX_n, DX_n (e.g. 20201214/00067678)
    #  2. has FLAGS_CNT/EXP_n but not DX_n, DY_n (e.g. 20210402/00083144)
    #  3. doesn't have any of these (e.g. 20201220/00069029)
    # Notes:
    #  * don't use FIBER_DX/DY because some files are missing those
    #    (e.g. 20210224/00077902)
    #  * don't use FLAGS_COR_n because some files are missing that
    #    (e.g. 20210402/00083144)

    pm = None
    numiter = 0
    if coordfile is None:
        log.error('No coordinates file, thus no info on fiber positioning')
    else:
        pm = Table.read(coordfile, 'DATA')  #- PM = PlateMaker

        #- If missing columns *and* not the first in a (split) sequence,
        #- try again with the first expid in the sequence
        #- (e.g. 202010404/00083419 -> 83418)
        if 'DX_0' not in pm.colnames:
            log.error(f'Missing DX_0 in {coordfile}')
            if 'VISITIDS' in rawheader:
                firstexp = int(rawheader['VISITIDS'].split(',')[0])
                if firstexp != rawheader['EXPID']:
                    origcorrdfile = coordfile
                    coordfile = findfile('coordinates', night, firstexp)
                    log.info(f'trying again with {coordfile}')
                    pm = Table.read(coordfile, 'DATA')
                else:
                    log.error(f'no earlier coordinates file for this tile')
            else:
                log.error('Missing VISITIDS header keywords to find earlier coordinates file')

        if 'FLAGS_CNT_0' not in pm.colnames:
            log.error(f'Missing spotmatch FLAGS_CNT_0 in {coordfile}; no positioner offset info')
            pm = None
            numiter = 0
        else:
            #- Count number of iterations in file
            numiter = len([col for col in pm.colnames if col.startswith('FLAGS_CNT_')])
            log.info(f'Using FLAGS_CNT_{numiter-1} in {coordfile}')

    #- Now let's merge that platemaker coordinates table (pm) with fiberassign
    fibermap_header = fits.Header({'XTENSION': 'BINTABLE'})
    fibermap_header.extend(fa_header, unique=True)
    if pm is not None:
        pm['LOCATION'] = 1000*pm['PETAL_LOC'] + pm['DEVICE_LOC']
        keep = np.in1d(pm['LOCATION'], fa['LOCATION'])
        pm = pm[keep]
        pm.sort('LOCATION')
        log.info('%d/%d fibers in coordinates file', len(pm), len(fa))

        #- Create fibermap table to merge with fiberassign file
        fibermap = Table()
        fibermap['LOCATION'] = pm['LOCATION']
        fibermap['NUM_ITER'] = numiter

        #- Sometimes these columns are missing in the coordinates files, maybe
        #- only when numiter=1, i.e. only a blind move but not corrections?
        if f'FPA_X_{numiter-1}' in pm.colnames:
            fibermap['FIBER_X'] = pm[f'FPA_X_{numiter-1}']
            fibermap['FIBER_Y'] = pm[f'FPA_Y_{numiter-1}']
            fibermap['DELTA_X'] = pm[f'DX_{numiter-1}']
            fibermap['DELTA_Y'] = pm[f'DY_{numiter-1}']
        else:
            log.error('No FIBER_X/Y or DELTA_X/Y information from platemaker')
            fibermap['FIBER_X'] = np.zeros(len(pm))
            fibermap['FIBER_Y'] = np.zeros(len(pm))
            fibermap['DELTA_X'] = np.zeros(len(pm))
            fibermap['DELTA_Y'] = np.zeros(len(pm))

        if ('FIBER_RA' in pm.colnames) and ('FIBER_DEC' in pm.colnames):
            fibermap['FIBER_RA'] = pm['FIBER_RA']
            fibermap['FIBER_DEC'] = pm['FIBER_DEC']
        else:
            log.error('No FIBER_RA or FIBER_DEC from platemaker')
            fibermap['FIBER_RA'] = np.zeros(len(pm))
            fibermap['FIBER_DEC'] = np.zeros(len(pm))

        #- Bit definitions at https://desi.lbl.gov/trac/wiki/FPS/PositionerFlags

        #- FLAGS_EXP bit 2 is for positioners (not FIF, GIF, ...)
        #- These should match what is in fiberassign, except broken fibers
        expflags = pm[f'FLAGS_EXP_{numiter-1}']
        goodmatch = ((expflags & 4) == 4)
        if np.any(~goodmatch):
            badloc = list(pm['LOCATION'][~goodmatch])
            log.warning(f'Flagging {len(badloc)} locations without POS_POS bit set: {badloc}')

        #- Keep only matched positioners (FLAGS_CNT_n bit 0)
        cntflags = pm[f'FLAGS_CNT_{numiter-1}']
        spotmatched = ((cntflags & 1) == 1)

        num_nomatch = np.sum(goodmatch & ~spotmatched)
        if num_nomatch > 0:
            badloc = list(pm['LOCATION'][goodmatch & ~spotmatched])
            log.error(f'Flagging {num_nomatch} unmatched fiber locations: {badloc}')

        goodmatch &= spotmatched

        #- pass forward dummy column for joining with fiberassign
        fibermap['_GOODMATCH'] = goodmatch

        #- WARNING: this join can re-order the table
        fibermap = join(fa, fibermap, join_type='left')

        #- poor and bad positioning
        dr = np.sqrt(fibermap['DELTA_X']**2 + fibermap['DELTA_Y']**2) * 1000
        poorpos = ((poor_offset_um < dr) & (dr <= bad_offset_um))
        badpos = (dr > bad_offset_um) | np.isnan(dr)
        numpoor = np.count_nonzero(poorpos)
        numbad = np.count_nonzero(badpos)
        if numpoor > 0:
            log.warning(f'Flagging {numpoor} POOR positions with {poor_offset_um} < offset <= {bad_offset_um} microns')
        if numbad > 0:
            log.warning(f'Flagging {numbad} BAD positions with offset > {bad_offset_um} microns')

        #- SKY assignments on stuck positioners will get special treatment
        stucksky = (fibermap['TARGETID']<0) & (fibermap['OBJTYPE']=='SKY')

        #- Set fiber status bits
        missing = np.in1d(fibermap['LOCATION'], pm['LOCATION'], invert=True)
        missing |= ~fibermap['_GOODMATCH']
        fibermap['FIBERSTATUS'][missing] |= fibermask.MISSINGPOSITION
        fibermap['FIBERSTATUS'][poorpos] |= fibermask.POORPOSITION
        fibermap['FIBERSTATUS'][badpos & ~stucksky] |= fibermask.BADPOSITION

        #- for SKY on stuck positioners, recheck if they are on a blank sky
        #- (e.g. they might be "off" target but still ok for sky)
        log.info('Checking if SKY on stuck positioners are still on SKY locations')
        tilera = fa.meta['TILERA']
        tiledec = fa.meta['TILEDEC']
        tileradius = get_tile_radius_deg()
        skybricks = Skybricks()
        ok = skybricks.lookup_tile(tilera, tiledec, tileradius,
                fibermap['FIBER_RA'][stucksky], fibermap['FIBER_DEC'][stucksky])
        num_stucksky = len(ok)
        num_ok = np.sum(ok)
        log.info(f'Keeping {num_ok}/{num_stucksky} SKY on stuck positioners')

        fibermap['FIBERSTATUS'][stucksky][~ok] |= fibermask.BADPOSITION

        fibermap.remove_column('_GOODMATCH')

    else:
        #- No coordinates file or no positioning iterations;
        #- just use fiberassign + dummy columns
        log.error('Unable to find useful coordinates file; proceeding with fiberassign + dummy columns')
        fibermap = fa
        fibermap['NUM_ITER'] = 0
        fibermap['FIBER_X'] = 0.0
        fibermap['FIBER_Y'] = 0.0
        fibermap['DELTA_X'] = 0.0
        fibermap['DELTA_Y'] = 0.0
        fibermap['FIBER_RA'] = 0.0
        fibermap['FIBER_DEC'] = 0.0
        # Update data types to be consistent with updated value if coord file was used.
        for val in ['FIBER_X','FIBER_Y','DELTA_X','DELTA_Y']:
            old_col = fibermap[val]
            fibermap.replace_column(val,Table.Column(name=val,data=old_col.data,dtype='>f8'))
        for val	in ['LOCATION','NUM_ITER']:
            old_col = fibermap[val]
            fibermap.replace_column(val,Table.Column(name=val,data=old_col.data,dtype=np.int64))

    #- Update SKY and STD target bits to be in both CMX_TARGET and DESI_TARGET
    #- i.e. if they are set in one, also set in the other.  Ditto for SV*
    for targetcol in ['CMX_TARGET', 'SV0_TARGET', 'SV1_TARGET', 'SV2_TARGET']:
        if targetcol in fibermap.colnames:
            for mask in [
                    desi_mask.SKY, desi_mask.STD_FAINT, desi_mask.STD_BRIGHT]:
                ii  = (fibermap[targetcol] & mask) != 0
                iidesi = (fibermap['DESI_TARGET'] & mask) != 0
                fibermap[targetcol][iidesi] |= mask
                fibermap['DESI_TARGET'][ii] |= mask

    #- Add header information from rawfile
    log.debug(f'Adding header keywords from {rawfile}')

    #- Early data raw headers had bad >8 char 'FIBERASSIGN' and 'USESPLITS' keyword.
    for old, new in [('FIBERASSIGN', 'FIBASSGN'), ('USESPLITS', 'USESPLIT')]:
        if old in rawheader:
            if new in rawheader:
                if rawheader[old] == rawheader[new]:
                    log.warning("%s and %s both exist in header from %s, but they have the same value.", old, new, rawfile)
                else:
                    log.error("%s and %s both exist in header from %s, and they have the different values!", old, new, rawfile)
                log.warning("Deleting header keyword %s.", old)
                rawheader.remove(old)
            else:
                log.warning('Renaming header keyword %s -> %s.', old, new)
                rawheader.rename_keyword(old, new)

    if 'DEPNAM00' in rawheader:
        mergedep(rawheader, fibermap_header, conflict='dst')
    fibermap_header.extend(rawheader, strip=True, unique=True)
    fibermap_header.remove('EXPTIME')
    fibermap['EXPTIME'] = rawheader['EXPTIME']
    #- Add header info from guide file
    #- sometimes full header is in HDU 0, other times HDU 1...
    if guidefile is not None:
        log.debug(f'Adding header keywords from {guidefile}')
        guideheader = fits.getheader(guidefile, 0)
        if 'TILEID' not in guideheader:
            guideheader = fits.getheader(guidefile, 1)

        if fibermap_header['TILEID'] != guideheader['TILEID']:
            raise RuntimeError('fiberassign tile {} != guider tile {}'.format(
                fibermap_header['TILEID'], guideheader['TILEID']))

        if 'DEPNAM00' in guideheader:
            mergedep(guideheader, fibermap_header, conflict='dst')
        fibermap_header.extend(guideheader, strip=True, unique=True)
        fibermap_header.remove('EXPTIME')


    fibermap_header['EXTNAME'] = 'FIBERMAP'
    for key in ('ZIMAGE', 'ZSIMPLE', 'ZBITPIX', 'ZNAXIS', 'ZNAXIS1', 'ZTILE1', 'ZCMPTYPE', 'ZNAME1', 'ZVAL1', 'ZNAME2', 'ZVAL2'):
        fibermap_header.remove(key)

    #- Record input guide and coordinates files
    if guidefile is not None:
        fibermap_header['GUIDEFIL'] = os.path.basename(guidefile)
    else:
        fibermap_header['GUIDEFIL'] = 'MISSING'

    if coordfile is not None:
        fibermap_header['COORDFIL'] = os.path.basename(coordfile)
    else:
        fibermap_header['COORDFIL'] = 'MISSING'

    #- mask the fibers defined by badamps
    if badamps is not None:
        maskbits = {'b':fibermask.BADAMPB, 'r':fibermask.BADAMPR, 'z':fibermask.BADAMPZ}
        ampoffsets = {'A': 0, 'B':250, 'C':0, 'D':250}
        for (camera, petal, amplifier) in parse_badamps(badamps):
            maskbit = maskbits[camera]
            ampoffset = ampoffsets[amplifier]
            fibermin = int(petal)*500 + ampoffset
            fibermax = fibermin + 250
            ampfibs = np.arange(fibermin,fibermax)
            truefmax = fibermax - 1
            log.info(f'Masking fibers from {fibermin} to {truefmax} for camera {camera} because of badamp entry '+\
                     f'{camera}{petal}{amplifier}')
            ampfiblocs = np.in1d(fibermap['FIBER'], ampfibs)
            fibermap['FIBERSTATUS'][ampfiblocs] |= maskbit

    #- mask the fibers defined by bad fibers
    if badfibers_filename is not None:
        table=Table.read(badfibers_filename)

        # list of bad fibers that are in the fibermap
        badfibers=np.intersect1d(np.unique(table["FIBER"]),fibermap["FIBER"])

        for i,fiber in enumerate(badfibers) :
            # for each of the bad fiber, add the bad bits to the fiber status
            badfibermask = np.bitwise_or.reduce(table["FIBERSTATUS"][table["FIBER"]==fiber])
            fibermap['FIBERSTATUS'][fibermap["FIBER"]==fiber] |= badfibermask

    #- NaN are a pain; reset to dummy values
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
    #
    # Some SV1-era files had these extraneous columns, make sure they are not propagated.
    # In the case of CMX_TARGET, those bits have already been copied into DESI_TARGET above.
    #
    for col in ('NUMTARGET', 'BLOBDIST', 'FIBERFLUX_IVAR_G', 'FIBERFLUX_IVAR_R', 'FIBERFLUX_IVAR_Z', 'HPXPIXEL'):
        if col in fibermap.colnames:
            log.info("Removing column '%s' from fibermap table.", col)
            fibermap.remove_column(col)
    if survey != 'cmx' and 'CMX_TARGET' in fibermap.colnames:
        log.info("Removing column '%s' from fibermap table.", 'CMX_TARGET')
        fibermap.remove_column(col)

    #
    # Some SV1-era files did not have SV1_SCND_TARGET.
    #
    if 'SV1_DESI_TARGET' in fibermap.colnames and 'SV1_SCND_TARGET' not in fibermap.colnames:
        log.info('Adding SV1_SCND_TARGET column.')
        fibermap.add_column(np.zeros(len(fibermap), dtype=np.int64),
                            index=fibermap.index_column('SV1_MWS_TARGET') + 1,
                            name='SV1_SCND_TARGET')

    #
    # Some early files did not have FLUX_IVAR_W1, FLUX_IVAR_W2.
    #
    if 'FLUX_IVAR_W1' not in fibermap.columns and 'FLUX_IVAR_W2' not in fibermap.columns:
        log.info('Adding FLUX_IVAR_W1 and FLUX_IVAR_W2 columns.')
        fibermap.add_column(-1.0*np.ones(len(fibermap), dtype=np.float32),
                            index=fibermap.index_column('FLUX_W2') + 1,
                            name='FLUX_IVAR_W1')
        fibermap.add_column(-1.0*np.ones(len(fibermap), dtype=np.float32),
                            index=fibermap.index_column('FLUX_W2') + 2,
                            name='FLUX_IVAR_W2')
    #
    # Some SV1-era file have various columns out of order.
    #
    for column, location in (('RELEASE', 20), ('BRICKID', 21),
                             ('BRICK_OBJID', 22), ('MASKBITS', 30),
                             ('BRICKNAME', 37), ('PRIORITY_INIT', 54),
                             ('NUMOBS_INIT', 55)):
        if fibermap.index_column(column) != location:
            log.info('Reordering %s column.', column)
            column_copy = fibermap[column].copy()
            fibermap.remove_column(column)
            fibermap.add_column(column_copy, index=location, name=column)
    #
    # Some SV1-era files have RELEASE as int32.  Should be int16.
    #
    if fibermap['RELEASE'].dtype == np.dtype('>i2'):
        log.debug("RELEASE has correct type.")
    else:
        log.warning("Setting RELEASE to int16.")
        fibermap['RELEASE'] = fibermap['RELEASE'].astype(np.int16)

    #
    # Just to reduce complaints by certain FITS tools.
    #
    fibermap_header.rename_keyword('EPOCH', 'EQUINOX')

            #- if position was unknown, set FIBERSTATUS as BADPOSITION
            if col.startswith('FIBER') or col.startswith('DELTA'):
                fibermap['FIBERSTATUS'][ii] |= fibermask.BADPOSITION

    #- Some code incorrectly relies upon the fibermap being sorted by
    #- fiber number, so accomodate that before returning the table
    fibermap.sort('FIBER')

    #
    # Don't propagate fibermap.meta; we'd rather keep the comments from the
    # explicitly-constructed header.
    #
    fibermap.meta.clear()
    fibermap_hdu = fits.BinTableHDU(fibermap)
    fibermap_hdu.header.extend(fibermap_header, update=True)
    fibermap_hdulist = fits.HDUList([fits.PrimaryHDU(), fibermap_hdu])

    return fibermap_hdulist


def fibermap_new2old(fibermap):
    '''Converts new format fibermap into old format fibermap

    Args:
        fibermap: new-format fibermap table (e.g. with FLUX_G column)

    Returns:
        old format fibermap (e.g. with MAG column)

    Note: this is a transitional convenience function to allow us to
    simulate new format fibermaps while still running code that expects
    the old format.  After all code has been converted to use the new
    format, this will be removed.
    '''
    from desiutil.brick import Bricks
    from desitarget.targetmask import desi_mask

    brickmap = Bricks()
    fm = fibermap.copy()
    n = len(fm)

    isMWS = (fm['DESI_TARGET'] & desi_mask.MWS_ANY) != 0
    fm['OBJTYPE'][isMWS] = 'MWS_STAR'
    isBGS = (fm['DESI_TARGET'] & desi_mask.BGS_ANY) != 0
    fm['OBJTYPE'][isBGS] = 'BGS'

    stdmask = 0
    for name in ['STD', 'STD_FSTAR', 'STD_WD',
            'STD_FAINT', 'STD_FAINT_BEST', 'STD_BRIGHT', 'STD_BRIGHT_BEST']:
        if name in desi_mask.names():
            stdmask |= desi_mask[name]

    isSTD = (fm['DESI_TARGET'] & stdmask) != 0
    fm['OBJTYPE'][isSTD] = 'STD'

    isELG = (fm['DESI_TARGET'] & desi_mask.ELG) != 0
    fm['OBJTYPE'][isELG] = 'ELG'
    isLRG = (fm['DESI_TARGET'] & desi_mask.LRG) != 0
    fm['OBJTYPE'][isLRG] = 'LRG'
    isQSO = (fm['DESI_TARGET'] & desi_mask.QSO) != 0
    fm['OBJTYPE'][isQSO] = 'QSO'

    if ('FLAVOR' in fm.meta):
        if fm.meta['FLAVOR'] == 'arc':
            fm['OBJTYPE'] = 'ARC'
        elif fm.meta['FLAVOR'] == 'flat':
            fm['OBJTYPE'] = 'FLAT'

    fm.rename_column('TARGET_RA', 'RA_TARGET')
    fm.rename_column('TARGET_DEC', 'DEC_TARGET')

    fm['BRICKNAME'] = brickmap.brickname(fm['RA_TARGET'], fm['DEC_TARGET'])
    fm['TARGETCAT'] = np.full(n, 'UNKNOWN', dtype=(str, 20))

    fm['MAG'] = np.zeros((n,5), dtype='f4')
    fm['MAG'][:,0] = 22.5 - 2.5*np.log10(fm['FLUX_G'])
    fm['MAG'][:,1] = 22.5 - 2.5*np.log10(fm['FLUX_R'])
    fm['MAG'][:,2] = 22.5 - 2.5*np.log10(fm['FLUX_Z'])
    fm['MAG'][:,3] = 22.5 - 2.5*np.log10(fm['FLUX_W1'])
    fm['MAG'][:,4] = 22.5 - 2.5*np.log10(fm['FLUX_W2'])

    fm['FILTER'] = np.zeros((n,5), dtype=(str, 10))
    fm['FILTER'][:,0] = 'DECAM_G'
    fm['FILTER'][:,1] = 'DECAM_R'
    fm['FILTER'][:,2] = 'DECAM_Z'
    fm['FILTER'][:,3] = 'WISE_W1'
    fm['FILTER'][:,4] = 'WISE_W2'

    fm['POSITIONER'] = fm['LOCATION'].astype('i8')
    fm.rename_column('LAMBDA_REF', 'LAMBDAREF')

    fm.rename_column('FIBER_RA', 'RA_OBS')
    fm.rename_column('FIBER_DEC', 'DEC_OBS')

    if 'DESIGN_X' in fm.colnames:
        fm.rename_column('DESIGN_X', 'X_TARGET')
    if 'DESIGN_Y' in fm.colnames:
        fm.rename_column('DESIGN_Y', 'Y_TARGET')
    if 'FIBERASSIGN_X' in fm.colnames:
        fm.rename_column('FIBERASSIGN_X', 'X_TARGET')
    if 'FIBERASSIGN_Y' in fm.colnames:
        fm.rename_column('FIBERASSIGN_Y', 'Y_TARGET')

    fm['X_FVCOBS'] = fm['X_TARGET']
    fm['Y_FVCOBS'] = fm['Y_TARGET']
    fm['X_FVCERR'] = np.full(n, 1e-3, dtype='f4')
    fm['Y_FVCERR'] = np.full(n, 1e-3, dtype='f4')

    for colname in [
        'BRICKID', 'BRICK_OBJID', 'COMM_TARGET',
        'DELTA_XFPA', 'DELTA_XFPA_IVAR', 'DELTA_YFPA', 'DELTA_YFPA_IVAR',
        'DESIGN_Q', 'DESIGN_S',
        'FIBERFLUX_G', 'FIBERFLUX_R', 'FIBERFLUX_Z',
        'FIBERSTATUS',
        'FIBERTOTFLUX_G', 'FIBERTOTFLUX_R', 'FIBERTOTFLUX_Z', 'FIBER_DEC_IVAR', 'FIBER_RA_IVAR',
        'FLUX_IVAR_G', 'FLUX_IVAR_R', 'FLUX_IVAR_W1', 'FLUX_IVAR_W2', 'FLUX_IVAR_Z',
        'FLUX_G', 'FLUX_R', 'FLUX_W1', 'FLUX_W2', 'FLUX_Z',
        'MORPHTYPE', 'NUMTARGET', 'NUM_ITER',
        'PMDEC', 'PMDEC_IVAR', 'PMRA', 'PMRA_IVAR',
        'PRIORITY', 'REF_ID', 'SECONDARY_TARGET', 'SUBPRIORITY',
        'SV1_BGS_TARGET', 'SV1_DESI_TARGET', 'SV1_MWS_TARGET',
        'TARGET_DEC_IVAR', 'TARGET_RA_IVAR',
        ]:
        if colname in fm.colnames:
            fm.remove_column(colname)

    return fm
