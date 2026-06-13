"""
desispec.inventory - tools for rapidly locating targets and spectra within a
DESI spectroscopic production.

Supports both healpix-based productions (e.g. loa, which store targets by
HEALPIX pixel at nside=64) and uniqpix-based productions (e.g. matterhorn+,
which store targets by survey/program-specific UNIQPIX pixels).

Typical usage::

    from desispec.inventory import target_tiles, target_healpix, radec2targetids

    # look up which tiles observed a set of targets
    tiles = target_tiles(targetids=[12345, 67890], filename='target_inventory.h5')

    # cone search: which targets are within 30 arcsec of a position?
    targetids = radec2targetids((150.1, 2.3, 30.0), filename='target_inventory.h5')
"""

import os
import sys
import glob

import numpy as np
import h5py
from astropy.table import Table, Column, vstack
import fitsio

from desispec.io.meta import faflavor2program, findfile

def _inventory_tiledir(tiledir):
    """
    Read FIBERMAP data from the most recent night in a cumulative tile directory.

    Args:
        tiledir (str): path to a cumulative tile directory (e.g.
            ``{specprod}/tiles/cumulative/1234``); the most recent
            night subdirectory (20XXXXXX) is selected automatically.

    Returns:
        astropy.table.Table: table with columns TARGETID, FIBER,
            TARGET_RA, TARGET_DEC, TILEID, LASTNIGHT, and SURVEY and/or
            PROGRAM where available in the coadd headers.
    """
    tiledir = sorted(glob.glob(f'{tiledir}/20??????'))[-1]  #- most recent tiledir/NIGHT/
    print(f'Loading {tiledir}')
    lastnight = int(os.path.basename(tiledir))
    coaddfiles = sorted(glob.glob(f'{tiledir}/coadd*.fits*'))

    tables = list()
    fmcols = ('TARGETID', 'FIBER', 'TARGET_RA', 'TARGET_DEC')
    for filename in coaddfiles:
        with fitsio.FITS(filename) as fx:
            fibermap = fx['FIBERMAP'].read(columns=fmcols)
            header = fx['FIBERMAP'].read_header()
            hdr0 = fx[0].read_header()

        keep = np.isfinite(fibermap['TARGET_RA']) & np.isfinite(fibermap['TARGET_DEC'])
        tables.append(Table(fibermap[keep]))

    #- Stack first, then add columns that are the same for every row
    tiletable = vstack(tables)
    tiletable['TILEID'] = hdr0['TILEID']
    tiletable['LASTNIGHT'] = hdr0['NIGHT']

    # Missing in Iron, in header for Loa, and hdr0 and header for Matterhorn
    if 'SURVEY' in header:
        tiletable['SURVEY'] = header['SURVEY']
    if 'FAFLAVOR' in header:
        tiletable['PROGRAM'] = faflavor2program(header['FAFLAVOR'])

    return tiletable

def _get_unique_indices(arr):
    """
    Return a dict mapping each unique element to its indices in the array.

    Args:
        arr (numpy.ndarray): input array.

    Returns:
        dict: keys are unique values from arr; values are numpy arrays of
            integer indices where that value appears.
    """

    # Get unique values and inverse indices
    unique_values, inverse_indices = np.unique(arr, return_inverse=True)

    # Create a dictionary to store indices of each unique value
    indices_dict = {value: [] for value in unique_values}

    # Populate the dictionary with indices using the inverse indices
    for idx, value_idx in enumerate(inverse_indices):
        indices_dict[unique_values[value_idx]].append(idx)

    # Convert lists to numpy arrays for efficiency
    indices_dict = {k: np.array(v) for k, v in indices_dict.items()}

    return indices_dict

def create_inventory_zcat(zcat, outfile, ngroups=1000, hpix2upix=None):
    """
    Create an inventory HDF5 file from a zcatalog Table.

    Args:
        zcat (astropy.table.Table): zcatalog table with columns TARGETID,
            SURVEY, PROGRAM, TILEID, LASTNIGHT, FIBER, TARGET_RA, TARGET_DEC,
            and either HEALPIX or UNIQPIX.
        outfile (str): output inventory HDF5 filepath; written atomically via
            a temporary file.
        ngroups (int): number of TARGETID subgroups; targets are assigned to
            subgroup ``TARGETID % ngroups``. Default 1000.
        hpix2upix (dict or None): nested dict
            ``{survey: {program: array}}`` mapping healpix index to uniqpix
            value for each survey/program combination. Required when zcat
            contains UNIQPIX; must have ``zcat.meta['NSIDEMAX']`` set.

    Returns:
        None
    """
    print('Converting str -> bytes')
    zcat['SURVEY'] = zcat['SURVEY'].astype(bytes)
    zcat['PROGRAM'] = zcat['PROGRAM'].astype(bytes)

    if 'UNIQPIX' in zcat.colnames:
        pixtype = 'UNIQPIX'
    elif 'HEALPIX' in zcat.colnames:
        pixtype = 'HEALPIX'
    else:
        raise ValueError('Unable to find HEALPIX or UNIQPIX column in zcat')

    #- Create views with a subset of columns for different purposes
    zcat_tile = zcat.copy(copy_data=False)
    zcat_tile.keep_columns(['TARGETID', 'TILEID', 'LASTNIGHT', 'FIBER'])
    zcat_pix = zcat.copy(copy_data=False)
    zcat_pix.keep_columns(['TARGETID', 'SURVEY', 'PROGRAM', pixtype])
    zcat_radec = zcat.copy(copy_data=False)
    zcat_radec.keep_columns(['TARGETID', 'TARGET_RA', 'TARGET_DEC'])
    zcat_targetid = zcat.copy(copy_data=False)
    zcat_targetid.keep_columns(['TARGETID',])

    """
    NGROUPS = targets are subdivided by TGROUP = TARGETID % NGROUPS
    target_tiles/TGROUP
        Table TARGETID, TILEID, LASTNIGHT, FIBER
    target_(healpix|uniqpix)/TGROUP
        Table TARGETID, SURVEY, PROGRAM, HEALPIX|UNIQPIX
    (healpix|uniqpix)_targets/HEALPIX|UNIQPIX
        Table TARGETID, TARGET_RA, TARGET_DEC
    tile_targets/TILEID
        Table TARGETID
    """
    print('Grouping by TARGETID')
    target_tiles = dict()
    target_pix = dict()
    target_group = zcat['TARGETID'] % ngroups
    target_indices = _get_unique_indices(target_group)
    for tgroup, ii in target_indices.items():
        if tgroup%100 == 0:
            print(f'targetid group {tgroup}/{ngroups}')
        target_tiles[tgroup] = Table(np.unique(zcat_tile[ii]))
        target_pix[tgroup] = Table(np.unique(zcat_pix[ii]))

    print(f'Grouping by {pixtype}')
    pix_targets = dict()
    pix_indices = _get_unique_indices(zcat[pixtype])
    npix = len(pix_indices.keys())
    for pix, ii in pix_indices.items():
        if pix%1000 == 0:
            print(f'{pixtype} {pix}')
        pix_targets[pix] = Table(np.unique(zcat_radec[ii]))

    print('Grouping by TILEID')
    tile_targets = dict()
    tile_indices = _get_unique_indices(zcat['TILEID'])
    ntiles = len(tile_indices.keys())
    for tileindex, (tileid, ii) in enumerate(tile_indices.items()):
        if tileindex%100 == 0:
            print(f'tile {tileid} ({tileindex}/{ntiles})')
        tile_targets[tileid] = Table(np.unique(zcat_targetid[ii]))

    inventory = dict()
    inventory['target_tiles'] = target_tiles
    inventory[f'target_{pixtype.lower()}'] = target_pix
    inventory[f'{pixtype.lower()}_targets'] = pix_targets
    inventory['tile_targets'] = tile_targets

    nside = None
    if hpix2upix is not None:
        nside = zcat.meta['NSIDEMAX']

    write_inventory(outfile, inventory, ngroups, hpix2upix=hpix2upix, nside=nside)

def write_inventory(filename, inventory, ngroups, hpix2upix=None, nside=None):
    """
    Write an inventory dict to an HDF5 file atomically.

    Args:
        filename (str): output HDF5 filepath.
        inventory (dict): nested dict ``{group: {subgroup: Table}}`` defining
            the HDF5 group structure to write.
        ngroups (int): number of TARGETID subgroups, stored as a file-level
            HDF5 attribute.
        hpix2upix (dict or None): nested dict
            ``{survey: {program: array}}`` of healpix-to-uniqpix mappings.
            Written under ``hpix2upix/{survey}/{program}`` if provided.
        nside (int or None): healpix nside stored as an attribute on the
            ``hpix2upix`` group. Required when hpix2upix is not None.

    Returns:
        None
    """
    tmpfile = filename+'.tmp'
    with h5py.File(tmpfile, 'w') as hx:
        hx.attrs['ngroups'] = ngroups

        for group in inventory.keys():
            print(f'Writing {group}')
            hx.create_group(group)
            for subgroup in inventory[group].keys():
                hx[f'{group}/{subgroup}'] = inventory[group][subgroup]

        if hpix2upix is not None:
            hx.create_group('hpix2upix')
            hx['hpix2upix'].attrs['nside'] = nside
            for survey, programs in hpix2upix.items():
                hx.create_group(f'hpix2upix/{survey}')
                for program, hpix2upix_array in programs.items():
                    print(f'Writing hpix2upix/{survey}/{program} {len(hpix2upix_array)}')
                    hx[f'hpix2upix/{survey}/{program}'] = hpix2upix_array

    os.rename(tmpfile, filename)
    print(f'Wrote {filename}')

def create_inventory(outfile, specprod=None, ntiles=None, ngroups=1000, nproc=8, nrows=None):
    """
    Create a target inventory HDF5 file from a spectroscopic production.

    Prefers a ``zall-tilecumulative`` FITS catalog if one exists; otherwise
    reads FIBERMAP data from individual tile directories in parallel.
    Automatically detects whether the production uses healpix (nside=64) or
    uniqpix pixel indexing based on the presence of ``spectra/`` vs
    ``healpix/`` subdirectories.

    Args:
        outfile (str): output inventory HDF5 filepath.
        specprod (str or None): spectroscopic production name, overrides
            ``$SPECPROD``.
        ntiles (int or None): if set, limit to the first ntiles tiles.
            When using the zcat path this selects the first ntiles unique
            TILEIDs; when reading tile directories it takes the first ntiles
            directories alphabetically.
        ngroups (int): number of TARGETID subgroups for the inventory index.
            Default 1000.
        nproc (int): number of parallel processes when reading tile
            directories. Default 8. Ignored when a zcat file is found.
        nrows (int or None): if set, read only the first nrows rows from the
            zcat FITS file. Intended for testing; ignored when reading tile
            directories.

    Returns:
        None

    Raises:
        ValueError: if neither ``spectra/`` nor ``healpix/`` directories are
            found under the specprod root.
    """
    import healpy
    import desispec.io
    from desimodel.footprint import radec2pix
    import multiprocessing

    zcatfile = findfile('zall_tile', version='v2', specprod=specprod, readonly=True)
    specdir = desispec.io.specprod_root(specprod, readonly=True)
    if os.path.exists(zcatfile):
        print(f"Found {zcatfile}, using it to create inventory")
        columns = ('TARGETID', 'SURVEY', 'PROGRAM', 'TILEID', 'LASTNIGHT', 'FIBER', 'TARGET_RA', 'TARGET_DEC')
        rows = np.arange(nrows) if nrows is not None else None
        zcat = Table(fitsio.read(zcatfile, 'ZCATALOG', columns=columns, rows=rows))

        if ntiles is not None:
            print(f"Limiting to {ntiles} tiles")
            tileids = np.unique(zcat['TILEID'])
            keep_tiles = tileids[0:ntiles]
            zcat = zcat[np.isin(zcat['TILEID'], keep_tiles)]
    else:
        print(f"Unable to find {zcatfile}, creating inventory from tiledirs")
        tiledirs = sorted(glob.glob(f'{specdir}/tiles/cumulative/[0-9]*'))
        if ntiles is not None:
            tiledirs = tiledirs[0:ntiles]

        print(f'Loading {len(tiledirs)} tiles')
        with multiprocessing.Pool(nproc) as pool:
            results = pool.map(_inventory_tiledir, tiledirs)

        zcat = vstack(results)

    #- Support Iron and earlier, which didn't have SURVEY and PROGRAM in coadd files (!)
    #- This implementation takes ~20 seconds, but is lower memory than joins creating new tables
    if 'SURVEY' not in zcat.colnames:
        tiles = Table.read(findfile('tiles', specprod=specprod, readonly=True), 1, format='fits')
        survey_map = dict(zip(tiles['TILEID'], tiles['SURVEY']))
        program_map = dict(zip(tiles['TILEID'], tiles['PROGRAM']))
        zcat['SURVEY'] = np.array([survey_map[tid] for tid in zcat['TILEID']])
        zcat['PROGRAM'] = np.array([program_map[tid] for tid in zcat['TILEID']])

    #- fill in either UNIQPIX (matterhorn+) or HEALPIX (loa-)
    hpix2upix = None
    if os.path.isdir(os.path.join(specdir, 'spectra')):
        print('Calculating UNIQPIX')
        zcat['UNIQPIX'] = np.full(len(zcat), fill_value=-1, dtype=np.int32)
        hpix2upix = dict()
        for survey, program in np.unique(zcat['SURVEY', 'PROGRAM']):
            ii = (zcat['SURVEY'] == survey) & (zcat['PROGRAM'] == program)
            ra = zcat['TARGET_RA'][ii]
            dec = zcat['TARGET_DEC'][ii]
            hpix2upix_file = findfile('hpix2upix', specprod=specprod, survey=survey, faprogram=program, readonly=True)
            h2u, header = fitsio.read(hpix2upix_file, 'HPIX2UPIX', header=True)
            nside = header['NSIDE']
            hpix = healpy.ang2pix(nside, ra, dec, lonlat=True, nest=True)
            upix = h2u[hpix]
            zcat['UNIQPIX'][ii] = upix
            if survey not in hpix2upix:
                hpix2upix[survey] = dict()
            hpix2upix[survey][program] = h2u

        zcat.meta['NSIDEMAX'] = nside   # same for all hpix2upix files in a specprod
        assert np.all(zcat['UNIQPIX'] >= 0), 'Some UNIQPIX values are still -1'

    elif os.path.isdir(os.path.join(specdir, 'healpix')):
        print('Calculating HEALPIX')
        zcat['HEALPIX'] = radec2pix(nside=64, ra=zcat['TARGET_RA'], dec=zcat['TARGET_DEC'])
    else:
        raise ValueError(f'Unable to find spectra/ or healpix/ in {specdir}')

    #- Create the final output inventory file
    create_inventory_zcat(zcat, outfile, ngroups=ngroups, hpix2upix=hpix2upix)

def update_inventory(filename, specprod=None):
    """
    Update an existing inventory file with new data from specprod.

    Args:
        filename (str): path to an existing inventory HDF5 file.
        specprod (str or None): spectroscopic production name, overrides
            ``$SPECPROD``.

    Raises:
        NotImplementedError: always; not yet implemented.
    """
    raise NotImplementedError

def _create_header(radec, specprod):
    """
    Build a metadata dict for attaching to result table metadata.

    Args:
        radec (tuple or None): (RA, DEC, RADIUS_arcsec) if a cone search was
            performed, else None.
        specprod (str or None): spectroscopic production name; falls back to
            ``$SPECPROD`` environment variable if None.

    Returns:
        dict: keys RA, DEC, RADIUS (if radec provided) and SPECPROD.
    """
    header = dict()
    if radec is not None:
        header['RA'] = radec[0]
        header['DEC'] = radec[1]
        header['RADIUS'] = radec[2]

    if specprod is None:
        header['SPECPROD'] = os.getenv('SPECPROD', default='Unknown')
    else:
        header['SPECPROD'] = specprod

    return header

def target_tiles(targetids=None, radec=None, filename=None, inventory=None, specprod=None):
    """
    Return tile observations for one or more targets.

    Exactly one of targetids or radec must be provided.

    Args:
        targetids (int or array-like of int, optional): TARGETID(s) to look up.
        radec (tuple, optional): (RA_deg, DEC_deg, RADIUS_arcsec) cone search;
            all targets within the cone are returned.
        filename (str, optional): path to the inventory HDF5 file. Derived
            from specprod or ``$SPECPROD`` if not provided.
        inventory (dict, optional): pre-loaded inventory dict (as returned by
            loading the HDF5 file into memory). If provided, filename is not
            opened. Expected keys: ``meta`` (with ``ngroups``),
            ``target_tiles``.
        specprod (str, optional): spectroscopic production name, used to find
            the default inventory file and stored in result metadata.

    Returns:
        astropy.table.Table: columns TARGETID, TILEID, LASTNIGHT, FIBER.
            One row per (target, tile) observation. Empty table with correct
            schema if no matches found. Table metadata includes RA, DEC,
            RADIUS (if radec was given) and SPECPROD.
    """
    from desimodel.footprint import radec2pix
    assert (targetids is None) or (radec is None)
    if filename is None:
        filename = _get_default_inventory_filename(specprod)

    if radec is not None:
        targetids = radec2targetids(radec, filename)
    else:
        targetids = np.atleast_1d(targetids)

    results = list()
    if inventory is not None:
        ngroups = inventory['meta']['ngroups']
        for tid in targetids:
            subgroup = tid % ngroups
            data = inventory['target_tiles'][subgroup][:]
            results.append(Table(data[data['TARGETID'] == tid]))

    else:
        with h5py.File(filename) as hx:
            ngroups = hx.attrs['ngroups']
            for tid in targetids:
                subgroup = tid % ngroups
                data = hx[f'target_tiles/{subgroup}'][:]
                results.append(Table(data[data['TARGETID'] == tid]))

    if len(results)>0:
        result = vstack(results)
    else:
        blank = Table()
        blank.add_column(Column(name='TARGETID', dtype=int))
        blank.add_column(Column(name='TILEID', dtype='int32'))
        blank.add_column(Column(name='LASTNIGHT', dtype='int32'))
        blank.add_column(Column(name='FIBER', dtype='int32'))
        result = blank

    result.meta.update(_create_header(radec, specprod))

    #- standardize column order
    result = result['TARGETID', 'TILEID', 'LASTNIGHT', 'FIBER']

    return result

def _db_target_tiles(targetids=None, radec=None, specprod=None):
    """
    WIP: database equivalent of `target_tiles`
    """
    assert (targetids is None) or (radec is None)
    import psycopg2

    if specprod is None:
        specprod = os.environ['SPECPROD']

    if targetids is not None:
        targetstr = ', '.join([str(tid) for tid in np.atleast_1d(targetids)])
        q = f"""
SELECT z.targetid,z.lastnight,f.fiber,z.tileid
FROM {specprod}.ztile as z
JOIN {specprod}.fiberassign as f on z.tileid=f.tileid AND z.targetid=f.targetid
WHERE z.targetid IN ({targetstr})
"""
        conn = psycopg2.connect(dbname='desi', user='desi', host='specprod-db.desi.lbl.gov')
        cur = conn.cursor()
        cur.execute(q)
        rows = cur.fetchall()
    else:
        ra, dec, radius_arcsec = radec
        radius_deg = radius_arcsec/3600.0
        q = f"""
SELECT z.targetid,z.lastnight,f.fiber,z.tileid
FROM {specprod}.ztile as z
JOIN {specprod}.photometry as p on z.targetid = p.targetid
JOIN {specprod}.fiberassign as f on z.tileid = f.tileid AND z.targetid = f.targetid
WHERE q3c_radial_query(p.ra, p.dec, {ra}, {dec}, {radius_deg});
"""
        conn = psycopg2.connect(dbname='desi', user='desi', host='specprod-db.desi.lbl.gov')
        cur = conn.cursor()
        cur.execute(q)
        rows = cur.fetchall()

    result = Table(rows=rows, names=('TARGETID', 'LASTNIGHT', 'FIBER', 'TILEID'))
    result.meta.update(_create_header(radec, specprod))
    return result

def target_healpix(targetids=None, radec=None, filename=None, specprod=None):
    """
    Return the pixel and survey/program membership for one or more targets.

    Exactly one of targetids or radec must be provided.

    Args:
        targetids (int or array-like of int, optional): TARGETID(s) to look up.
        radec (tuple, optional): (RA_deg, DEC_deg, RADIUS_arcsec) cone search;
            all targets within the cone are returned.
        filename (str, optional): path to the inventory HDF5 file. Derived
            from specprod or ``$SPECPROD`` if not provided.
        specprod (str, optional): spectroscopic production name, used to find
            the default inventory file and stored in result metadata.

    Returns:
        astropy.table.Table: columns TARGETID, SURVEY, PROGRAM, and either
            HEALPIX (healpix-based inventory) or UNIQPIX (uniqpix-based
            inventory). Empty table with correct schema if no matches found.
            Table metadata includes RA, DEC, RADIUS (if radec was given)
            and SPECPROD.
    """
    from desimodel.footprint import radec2pix
    assert (targetids is None) or (radec is None)
    if filename is None:
        filename = _get_default_inventory_filename(specprod)

    if radec is not None:
        targetids = radec2targetids(radec, filename)
    else:
        targetids = np.atleast_1d(targetids)

    results = list()
    with h5py.File(filename) as hx:
        ngroups = hx.attrs['ngroups']
        pixgroup = 'target_uniqpix' if 'target_uniqpix' in hx else 'target_healpix'
        for tid in targetids:
            subgroup = tid % ngroups
            data = hx[f'{pixgroup}/{subgroup}'][:]
            results.append(Table(data[data['TARGETID'] == tid]))

    if len(results)>0:
        result = vstack(results)
    else:
        blank = Table()
        blank.add_column(Column(name='TARGETID', dtype=int))
        blank.add_column(Column(name='SURVEY', dtype='S7'))
        blank.add_column(Column(name='PROGRAM', dtype='S6'))
        pixcol = 'UNIQPIX' if pixgroup == 'target_uniqpix' else 'HEALPIX'
        blank.add_column(Column(name=pixcol, dtype=int))
        result = blank

    result.meta.update(_create_header(radec, specprod))
    return result

def _db_target_healpix(targetids=None, radec=None, specprod=None):
    """
    WIP: database equivalent of `target_healpix`
    """
    assert (targetids is None) or (radec is None)
    import psycopg2

    if specprod is None:
        specprod = os.environ['SPECPROD']

    if targetids is not None:
        targetstr = ', '.join([str(tid) for tid in np.atleast_1d(targetids)])
        q = f"SELECT targetid,survey,program,healpix FROM {specprod}.zpix WHERE targetid IN ({targetstr})"
        conn = psycopg2.connect(dbname='desi', user='desi', host='specprod-db.desi.lbl.gov')
        cur = conn.cursor()
        cur.execute(q)
        rows = cur.fetchall()
    else:
        ra, dec, radius_arcsec = radec
        radius_deg = radius_arcsec/3600.0
        q = f"""
SELECT z.targetid,z.survey,z.program,z.healpix
FROM {specprod}.photometry as p JOIN {specprod}.zpix as z on p.targetid = z.targetid
WHERE q3c_radial_query(p.ra, p.dec, {ra}, {dec}, {radius_deg});
"""
        conn = psycopg2.connect(dbname='desi', user='desi', host='specprod-db.desi.lbl.gov')
        cur = conn.cursor()
        cur.execute(q)
        rows = cur.fetchall()

    result = Table(rows=rows, names=('TARGETID', 'SURVEY', 'PROGRAM', 'HEALPIX'))
    result.meta.update(_create_header(radec, specprod))
    return result


def radec2targetids(radec, filename=None, specprod=None):
    """
    Return TARGETIDs of targets within a cone search.

    Works with both healpix-based and uniqpix-based inventory files.

    Args:
        radec (tuple): (RA_deg, DEC_deg) or (RA_deg, DEC_deg, RADIUS_arcsec).
            Default radius is 10 arcsec if not provided.
        filename (str, optional): path to the inventory HDF5 file. Derived
            from specprod or ``$SPECPROD`` if not provided.
        specprod (str, optional): spectroscopic production name, used to find
            the default inventory file.

    Returns:
        numpy.ndarray: integer array of TARGETIDs within the cone. Empty
            array if no targets found.
    """
    from healpy import ang2vec, query_disc
    from astropy.coordinates import SkyCoord
    from astropy import units as u

    if filename is None:
        filename = _get_default_inventory_filename(specprod)

    ra, dec, radius_arcsec = radec
    radius_radians = radius_arcsec * np.pi / (3600*180.)
    vec = ang2vec(ra, dec, lonlat=True)

    tables = list()
    with h5py.File(filename) as hx:
        if 'hpix2upix' in hx:
            #- uniqpix-based inventory: use hpix2upix to map healpix -> uniqpix
            nside = hx['hpix2upix'].attrs['nside']
            hpix_candidates = query_disc(nside, vec, radius_radians, nest=True, inclusive=True)
            upix_set = set()
            for survey in hx['hpix2upix'].keys():
                for program in hx[f'hpix2upix/{survey}'].keys():
                    h2u = hx[f'hpix2upix/{survey}/{program}'][:]
                    upix = h2u[hpix_candidates]
                    upix_set.update(upix[upix >= 0])
            for upix in upix_set:
                try:
                    tables.append(Table(hx[f'uniqpix_targets/{upix}'][:]))
                except KeyError:
                    pass
        else:
            #- healpix-based inventory
            nside = 64
            pixels = query_disc(nside, vec, radius_radians, nest=True, inclusive=True)
            for hpix in pixels:
                try:
                    tables.append(Table(hx[f'healpix_targets/{hpix}'][:]))
                except KeyError:
                    pass

    if len(tables) == 0:
        return np.array([], dtype=int)

    data = vstack(tables)

    cs = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
    ctgt = SkyCoord(ra=data['TARGET_RA']*u.deg, dec=data['TARGET_DEC']*u.deg, frame='icrs')
    sel = cs.separation(ctgt).to(u.arcsec).value < radius_arcsec

    return np.array(data['TARGETID'][sel])


def _get_default_inventory_filename(specprod=None):
    """
    Return the path to the default inventory file for a specprod.

    Searches in order: ``{specprod_root}/target_inventory.h5``, then
    ``$DESI_TARGET_INVENTORY_DIR/target_inventory-{specprod}.h5``.

    Args:
        specprod (str, optional): spectroscopic production name, overrides
            ``$SPECPROD``.

    Returns:
        str: path to the first inventory file found.

    Raises:
        IOError: if no inventory file is found in any of the search locations.
    """
    import desispec.io

    #- Build list of several places where the inventory files might exist
    files = list()
    files.append(os.path.join(desispec.io.specprod_root(specprod, readonly=True), 'target_inventory.h5'))
    inventory_dir = os.getenv('DESI_TARGET_INVENTORY_DIR')
    if inventory_dir is not None and os.path.isdir(inventory_dir):
        files.append(os.path.join(inventory_dir, f'target_inventory-{specprod}.h5'))

    #- Return the first inventory file found
    for filename in files:
        if os.path.exists(filename):
            return filename
    #- Raise an error if none are found
    else:
        raise IOError(f'Unable to find {specprod} inventory file')

def parse_radec(radec):
    """
    Parse a radec specification into (RA, DEC, RADIUS) floats.

    Args:
        radec (str, list, or tuple): RA and DEC in degrees, with an optional
            radius in arcsec. May be a comma-separated string
            (``"150.1,2.3"`` or ``"150.1,2.3,30.0"``), or a list/tuple with
            2 or 3 elements.

    Returns:
        tuple: (RA_deg, DEC_deg, RADIUS_arcsec) as floats. Default radius is
            10.0 arcsec if not provided.

    Raises:
        ValueError: if radec does not have 2 or 3 elements.
    """
    if isinstance(radec, str):
        radec = radec.split(',')
    if len(radec) == 2:
        ra = float(radec[0])
        dec = float(radec[1])
        radius = 10.0         # default 10 arcsec
    elif len(radec) == 3:
        ra = float(radec[0])
        dec = float(radec[1])
        radius = float(radec[2])
    else:
        raise ValueError(f'{radec=} should be "ra,dec" or "ra,dec,radius"')

    return ra, dec, radius

def main():
    import argparse

    # parser = argparse.ArgumentParser(description='Search, create, or update DESI target inventory')

    #- shared options
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument('-s', '--specprod',  help='Input specprod, overrides $SPECPROD')
    shared.add_argument('-f', '--filename', help='Inventory file')

    scriptname = os.path.basename(sys.argv[0])
    parser = argparse.ArgumentParser(description='Search, create, or update DESI target inventory',
                                     epilog=f"Run '{scriptname} <subcommand> --help' for subcommand-specific options.",
                                     parents=[shared])

    # Add subparsers
    subparsers = parser.add_subparsers(dest='subcommand', help='Sub-command help')

    #- create
    sub1 = subparsers.add_parser('create', help='Create a target inventory', parents=[shared])
    sub1.add_argument('-n', '--ntiles', type=int, help='Number of tiles to include')
    sub1.add_argument('--nproc', type=int, help='Number of parallel processes to use')

    # update
    sub2 = subparsers.add_parser('update', help='Update target inventory', parents=[shared])

    # search tiles
    sub3 = subparsers.add_parser('tiles', help='Search tiles for targetids', parents=[shared])
    sub3.add_argument('-t', '--targetids', help='comma separated TARGETIDs')
    sub3.add_argument('--radec', help='RA_DEGREES,DEC_DEGREES[,RADIUS_ARCSEC]')

    # search healpix
    sub4 = subparsers.add_parser('healpix', help='Search healpix for targetids', parents=[shared])
    sub4.add_argument('-t', '--targetids', help='comma separated TARGETIDs')
    sub4.add_argument('--radec', help='RA_DEGREES,DEC_DEGREES[,RADIUS_ARCSEC]')

    # Parse the arguments
    args = parser.parse_args()

    if args.filename is None:
        args.filename = _get_default_inventory_filename(args.specprod)

    if args.subcommand in ('tiles', 'healpix'):
        #- can't set both --targetids and --radec
        assert (args.targetids is None) or (args.radec is None)

        if args.radec is not None:
            ra_dec_radius = parse_radec(args.radec)
        else:
            ra_dec_radius = None

        if args.targetids is not None:
            targetids = np.array([int(t) for t in args.targetids.split(',')])
        else:
            targetids = None

    if args.subcommand == 'create':
        create_inventory(args.filename, specprod=args.specprod, ntiles=args.ntiles, nproc=args.nproc)
    elif args.subcommand == 'update':
        update_inventory(args.filename, specprod=args.specprod)
    elif args.subcommand == 'tiles':
        print(target_tiles(targetids=targetids, radec=ra_dec_radius, filename=args.filename))
    elif args.subcommand == 'healpix':
        print(target_healpix(targetids=targetids, radec=ra_dec_radius, filename=args.filename))
    else:
        parser.print_help()


if __name__ == '__main__':
    sys.exit(main())
