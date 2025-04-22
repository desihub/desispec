"""
desispec.inventory - TARGETID -> (SURVEY,PROGRAM,HEALPIX) and (TILEID,FIBER)
"""

import os
import sys
import glob

import numpy as np
import h5py
from astropy.table import Table, Column, vstack

def _inventory_tiledir(tiledir):
    """
    Generate tiles, healpix, and radec Tables for data in tiledir

    Args:
        tiledir (str): path to a directory of tile redshifts

    Returns tuple (tiles, healpix, radec) where

      - tiles: Table(TARGETID, TILEID, LASTNIGHT, FIBER)
      - healpix: Table(TARGETID, SURVEY, PROGRAM, HEALPIX)
      - radec: Table(TARGETID, TARGET_RA, TARGET_DEC, HEALPIX)
    """
    from desimodel.footprint import radec2pix
    import fitsio

    tiledir = sorted(glob.glob(f'{tiledir}/20??????'))[-1]  #- most recent tiledir/NIGHT/
    print(f'Loading {tiledir}')
    lastnight = int(os.path.basename(tiledir))
    coaddfiles = sorted(glob.glob(f'{tiledir}/coadd*.fits*'))
    columns = ('TARGETID', 'FIBER', 'TARGET_RA', 'TARGET_DEC')

    tiles = list()
    healpix = list()
    radec = list()
    for filename in coaddfiles:
        fibermap, header = fitsio.read(filename, 'FIBERMAP', header=True)
        fibermap = Table(fibermap)
        fibermap['HEALPIX'] = radec2pix(nside=64, ra=fibermap['TARGET_RA'], dec=fibermap['TARGET_DEC'])

        tiles.append(fibermap['TARGETID', 'FIBER'])
        healpix.append(fibermap['TARGETID', 'HEALPIX'])
        radec.append(fibermap['TARGETID', 'TARGET_RA', 'TARGET_DEC', 'HEALPIX'])

    #- Stack first, then add columns that are the same for every row
    tiles = vstack(tiles)
    tiles['TILEID'] = header['TILEID']
    tiles['LASTNIGHT'] = lastnight

    healpix = vstack(healpix)
    healpix['SURVEY'] = header['SURVEY'].lower()
    healpix['PROGRAM'] = header['PROGRAM'].lower()

    radec = vstack(radec)

    return tiles, healpix, radec

def _get_unique_indices(arr):
    """
    Return dict of indices for each unique element in an array

    Args:
        arr: numpy array

    Returns dict(unique_value) = list of indices in arr with that value

    From LBL CBorg Coder AI Model
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

def create_inventory_zcat(zcatfile, outfile, ngroups=1000):
    """
    Create an inventory file given a zall-tilecumulative redshift catalog file

    Args:
        zcatfile (str): path to zall-tilecumulative*.fits file
        outfile (str): output inventory hdf5 filepath

    Options:
        ngroups (int): number of TARGETID subgroups
    """
    from desimodel.footprint import radec2pix
    import fitsio

    #- Read the zcatalog file with all columns that we need
    print(f'Reading {zcatfile}')
    columns = ('TARGETID', 'SURVEY', 'PROGRAM', 'TILEID', 'LASTNIGHT', 'FIBER', 'TARGET_RA', 'TARGET_DEC')
    zcat = Table(fitsio.read(zcatfile, 'ZCATALOG', columns=columns))

    print('Calculating healpix')
    zcat['HEALPIX'] = radec2pix(nside=64, ra=zcat['TARGET_RA'], dec=zcat['TARGET_DEC'])

    print('Converting str -> bytes')
    zcat['SURVEY'] = zcat['SURVEY'].astype(bytes)
    zcat['PROGRAM'] = zcat['PROGRAM'].astype(bytes)

    #- Create views with a subset of columns for different purposes
    zcat_tile = zcat.copy(copy_data=False)
    zcat_tile.keep_columns(['TARGETID', 'TILEID', 'LASTNIGHT', 'FIBER'])
    zcat_healpix = zcat.copy(copy_data=False)
    zcat_healpix.keep_columns(['TARGETID', 'SURVEY', 'PROGRAM', 'HEALPIX'])
    zcat_radec = zcat.copy(copy_data=False)
    zcat_radec.keep_columns(['TARGETID', 'TARGET_RA', 'TARGET_DEC'])
    zcat_targetid = zcat.copy(copy_data=False)
    zcat_targetid.keep_columns(['TARGETID',])

    """
    NGROUPS = targets are subdivided by TGROUP = TARGETID % NGROUPS
    target_tiles
        TGROUP = Table TARGETID, TILEID, LASTNIGHT, FIBER
    target_healpix
        TGROUP = Table TARGETID, SURVEY, PROGRAM, HEALPIX
    healpix
        HEALPIX = Table TARGETID, TARGET_RA, TARGET_DEC
    tiles
        TILEID = Table TARGETID
    """
    print('Grouping by TARGETID')
    target_tiles = dict()
    target_healpix = dict()
    target_group = zcat['TARGETID'] % ngroups
    target_indices = _get_unique_indices(target_group)
    for tgroup, ii in target_indices.items():
        if tgroup%100 == 0:
            print(f'targetid group {tgroup}/{ngroups}')
        target_tiles[tgroup] = Table(np.unique(zcat_tile[ii]))
        target_healpix[tgroup] = Table(np.unique(zcat_healpix[ii]))

    print('Grouping by HEALPIX')
    healpix_targets = dict()
    healpix_indices = _get_unique_indices(zcat['HEALPIX'])
    npix = len(healpix_indices.keys())
    for hpix, ii in healpix_indices.items():
        if hpix%1000 == 0:
            print(f'healpix {hpix}')
        healpix_targets[hpix] = Table(np.unique(zcat_radec[ii]))

    print('Grouping by TILEID')
    tile_targets = dict()
    tile_indices = _get_unique_indices(zcat['TILEID'])
    ntiles = len(tile_indices.keys())
    for tileid, ii in tile_indices.items():
        if tileid%100 == 0:
            print(f'tile {tileid}/{ntiles}')
        tile_targets[tileid] = Table(np.unique(zcat_targetid[ii]))

    inventory = dict()
    inventory['target_tiles'] = target_tiles
    inventory['target_healpix'] = target_healpix
    inventory['healpix_targets'] = healpix_targets
    inventory['tile_targets'] = tile_targets

    write_inventory(outfile, inventory, ngroups)

def write_inventory(filename, inventory, ngroups):
    """
    Write inventory struction to filename; include ngroups in attr metadata
    """
    tmpfile = filename+'.tmp'
    with h5py.File(tmpfile, 'w') as hx:
        hx.attrs['ngroups'] = ngroups

        for group in inventory.keys():
            print(f'Writing {group}')
            hx.create_group(group)
            for subgroup in inventory[group].keys():
                hx[f'{group}/{subgroup}'] = inventory[group][subgroup]

    os.rename(tmpfile, filename)
    print(f'Wrote {filename}')

def create_inventory(outfile, specprod=None, ntiles=None, ngroups=1000, nproc=8):
    """
    Create target inventory from specprod that doesn't have a zall-tilecumulative file

    TODO: document; WIP
    """
    import desispec.io
    import multiprocessing

    specdir = desispec.io.specprod_root(specprod, readonly=True)
    tiledirs = sorted(glob.glob(f'{specdir}/tiles/cumulative/[0-9]*'))
    if ntiles is not None:
        tiledirs = tiledirs[0:ntiles]

    tiles = list()
    healpix = list()
    radec = list()

    print(f'Loading {len(tiledirs)} tiles')
    with multiprocessing.Pool(nproc) as pool:
        results = pool.map(_inventory_tiledir, tiledirs)
    
    for tiletable, hpixtable, radectable in results:
        tiles.append(tiletable)
        healpix.append(hpixtable)
        radec.append(radectable)

    tiles = vstack(tiles)
    healpix = vstack(healpix)
    radec = vstack(radec)

    #- convert unicode strings to bytes for hdf5
    healpix['SURVEY'] = healpix['SURVEY'].astype(bytes)
    healpix['PROGRAM'] = healpix['PROGRAM'].astype(bytes)

    tmpfile = outfile+'.tmp'
    with h5py.File(tmpfile, 'w') as hx:
        hx.attrs['ngroups'] = ngroups
       
        group = 'targetid_tiles'
        print(f'Writing {group}')
        hx.create_group(group)
        subgroups = tiles['TARGETID'] % ngroups
        indices_dict = _get_unique_indices(subgroups)
        for subgroup, ii in indices_dict.items():
            hx[f'{group}/{subgroup}'] = tiles['TARGETID', 'TILEID', 'LASTNIGHT', 'FIBER'][ii]

        group = 'targetid_healpix'
        print(f'Writing {group}')
        hx.create_group(group)
        subgroups = healpix['TARGETID'] % ngroups
        indices_dict = _get_unique_indices(subgroups)
        for subgroup, ii in indices_dict.items():
            hx[f'{group}/{subgroup}'] = healpix['TARGETID', 'SURVEY', 'PROGRAM', 'HEALPIX'][ii]

        group = 'healpix_targetid_radec'
        print(f'Writing {group}')
        hx.create_group(group)
        subgroup_indices = _get_unique_indices(radec['HEALPIX'])
        for subgroup, ii in subgroup_indices.items():
            hx[f'{group}/{subgroup}'] = radec['TARGETID', 'TARGET_RA', 'TARGET_DEC'][ii]

    os.rename(tmpfile, outfile)
    print(f'Wrote {outfile}')


def update_inventory(filename, specprod=None):
    raise NotImplementedError

def _create_header(radec, specprod):
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
    Return table of TARGETID,TILEID,LASTNIGHT,FIBER

    Args:
        targetids (int or array of int): TARGETID(s)
        radec (tuple): RA_degrees, DEC_degrees, RADIUS_arcsec

    Options:
        filename (str): inventory filename
        inventory (dict): inventory structure pre-loaded in memory
        specprod (str): spectroscopic production

    Returns Table(TARGETID,TILEID,LASTNIGHT,FIBER)

    Must input `targetids` or `radec` but not both.
    """
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
        blank.add_column(Column(name='LASTNIGHT', dtype='int32'))
        blank.add_column(Column(name='FIBER', dtype='int32'))
        blank.add_column(Column(name='TILEID', dtype='int32'))
        result = blank

    result.meta.update(_create_header(radec, specprod))
    return result

def db_target_tiles(targetids=None, radec=None, specprod=None):
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
    Return table of TARGETID,SURVEY,PROGRAM,HEALPIX

    Args:
        targetids (int or array of int): TARGETID(s)
        radec (tuple): RA_degrees, DEC_degrees, RADIUS_arcsec

    Options:
        filename (str): inventory filename
        specprod (str): spectroscopic production

    Returns Table(TARGETID,SURVEY,PROGRAM,HEALPIX)

    Must input `targetids` or `radec` but not both.
    """
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
        for tid in targetids:
            subgroup = tid % ngroups
            data = hx[f'target_healpix/{subgroup}'][:]
            results.append(Table(data[data['TARGETID'] == tid]))

    if len(results)>0:
        result = vstack(results)
    else:
        blank = Table()
        blank.add_column(Column(name='TARGETID', dtype=int))
        blank.add_column(Column(name='SURVEY', dtype='S7'))
        blank.add_column(Column(name='PROGRAM', dtype='S6'))
        blank.add_column(Column(name='HEALPIX', dtype=int))
        result = blank

    result.meta.update(_create_header(radec, specprod))
    return result

def db_target_healpix(targetids=None, radec=None, specprod=None):
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
    Return TARGETIDs in ra,dec[,radius] search

    Args:
        radec (tuple): RA,DEC or RA,DEC,RADIUS_arcsec

    Options:
        filename (str): inventory filename
        specprod (str): spectroscopic production name

    Return array of TARGETIDs
    """
    from healpy import ang2vec, query_disc
    from astropy.coordinates import SkyCoord
    from astropy import units as u

    if filename is None:
        filename = _get_default_inventory_filename(specprod)

    ra, dec, radius_arcsec = radec
    radius_radians = radius_arcsec * np.pi / (3600*180.)
    nside = 64
    vec = ang2vec(ra, dec, lonlat=True)
    pixels = query_disc(nside, vec, radius_radians, nest=True, inclusive=True)

    tables = list()
    with h5py.File(filename) as hx:
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
    Return default inventory filename

    Args:
        specprod (str): overrides $SPECPROD

    Returns filepath to inventory file (which may not yet exist)
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

def parse_radec_string(radec):
    """
    interpret radec as (RA,DEC) or (RA,DEC,RADIUS)
    """
    tmp = radec.split(',')
    if len(tmp) == 2:
        ra = float(tmp[0])
        dec = float(tmp[1])
        radius = 10.0         # default 10 arcsec
    elif len(tmp) == 3:
        ra = float(tmp[0])
        dec = float(tmp[1])
        radius = float(tmp[2])
    else:
        raise ValueError(f'{radec=} should be "ra,dec" or "ra,dec,radius"')

    return ra, dec, radius

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Search, create, or update DESI target inventory')
    parser.add_argument('-s', '--specprod',  help='Input specprod, overrides $SPECPROD')
    parser.add_argument('-f', '--filename', help='Inventory file')

    # Add subparsers
    subparsers = parser.add_subparsers(dest='subcommand', help='Sub-command help')

    #- create
    sub1 = subparsers.add_parser('create', help='Create a target inventory')
    sub1.add_argument('-n', '--ntiles', type=int, help='Number of tiles to include')
    sub1.add_argument('--nproc', type=int, help='Number of parallel processes to use')

    # update
    sub2 = subparsers.add_parser('update', help='Update target inventory')

    # search tiles
    sub3 = subparsers.add_parser('tiles', help='Search tiles for targetids')
    sub3.add_argument('-t', '--targetids', help='comma separated TARGETIDs')
    sub3.add_argument('--radec', help='RA_DEGREES,DEC_DEGREES[,RADIUS_ARCSEC]')

    # search healpix
    sub4 = subparsers.add_parser('healpix', help='Search healpix for targetids')
    sub4.add_argument('-t', '--targetids', help='comma separated TARGETIDs')
    sub4.add_argument('--radec', help='RA_DEGREES,DEC_DEGREES[,RADIUS_ARCSEC]')

    # Parse the arguments
    args = parser.parse_args()

    if args.filename is None:
        args.filename = _get_default_inventory_filename(args.specprod)

    if args.subcommand in ('tiles', 'healpix'):
        #- can't set both --targetids and --radec
        assert (args.targetids is None) or (args.ra_dec_radius is None)

        if args.radec is not None:
            ra_dec_radius = parse_radec_string(args.radec)
        else:
            ra_dec_radius = None

        if args.targetids is not None:
            targetids = np.array([int(t) for t in args.targetid.split(',')])
        else:
            targetids = None

    if args.subcommand == 'create':
        create_inventory(args.filename, specprod=args.specprod, ntiles=args.ntiles, nproc=args.nproc)
    elif args.subcommand == 'update':
        update_inventory(args.filename, specprod=args.specprod)
    elif args.subcommand == 'tiles':
        print(target_tiles(targetids=targetids, radec=ra_dec_radius, filename=args.filename))
    elif args.subcommand == 'healpix':
        print(target_healpix(targetids, radec=ra_dec_radius, filename=args.filename))
    else:
        parser.print_help()


if __name__ == '__main__':
    sys.exit(main())
