"""
desispec.specstatus
===================

Update surveyops/ops/tile-specstatus.ecsv with spectro pipeline tiles.csv.
"""

import numpy as np
from astropy.table import Table, vstack

from desiutil.log import get_logger

def update_specstatus(specstatus, tiles, update_all=False):
    """
    return new specstatus table, updated with tiles table

    Args:
        specstatus: astropy Table from surveyops/ops/tiles-specstatus.ecsv
        tiles: astropy Table from spectro/redux/daily/tiles.csv

    Returns: updated specstatus table, sorted by TILEID

    New TILEID found in tiles are added to specstatus, and any entries
    where tiles['LASTNIGHT'] > specstatus['LASTNIGHT'] (i.e. new data)
    have their non-QA columns updated.

    if update_all==True, update all non-QA related columns even if LASTNIGHT
    is the same (e.g. due to a reprocessing)

    This does not modify either of the input tables.
    """

    log = get_logger()
    specstatus = specstatus.copy()
    tiles = tiles.copy()

    #- Added for Fuji, but not in tiles-specstatus so remove
    if 'PROGRAM' in tiles.colnames:
        tiles.remove_column('PROGRAM')

    #- Confirm that they have the same columns except QA-specific ones
    tilecol = set(tiles.colnames) | set(['USER', 'QA', 'OVERRIDE', 'ZDONE', 'QANIGHT', 'ARCHIVEDATE'])
    if tilecol != set(specstatus.colnames):
        log.error('Column mismatch: {tiles.colnames} vs. {specstatus.colnames}')
        raise ValueError('Incompatible specstatus and tiles columns')

    #- even if present in tiles, specstatus trumps for these columns
    #- (i.e. never update them)
    qacols = ['USER', 'QA', 'OVERRIDE', 'ZDONE', 'QANIGHT', 'ARCHIVEDATE']

    #- Add any new tiles
    newtilerows = np.isin(tiles['TILEID'], specstatus['TILEID'], invert=True)
    num_newtiles = np.count_nonzero(newtilerows)
    if num_newtiles > 0:
        tt = list(tiles['TILEID'][newtilerows])
        log.info(f'Adding {num_newtiles} new tiles: {tt}')

        newtiles = tiles[newtilerows]
        newtiles['USER'] = np.repeat('none',num_newtiles)
        newtiles['QA'] = np.repeat('none',num_newtiles)
        newtiles['OVERRIDE'] = np.repeat(0,num_newtiles)
        newtiles['ZDONE'] = np.repeat('false',num_newtiles)
        newtiles['QANIGHT'] = np.repeat(0,num_newtiles)
        newtiles['ARCHIVEDATE'] = np.repeat(0,num_newtiles)
        newtiles = newtiles[specstatus.colnames]  #- columns in same order

        specstatus = vstack([specstatus, newtiles])
    else:
        log.info('No new tiles to add')

    #- At this point, every TILEID in tiles should be in specstatus,
    #- but ok if specstatus has TILEID not in tiles
    assert np.all(np.isin(tiles['TILEID'], specstatus['TILEID']))

    #- For rows with more recent LASTNIGHT (new data), update non-QA columns.
    #- Note: there is probably a more efficient way of doing this in bulk,
    #- but let's favor obvious over clever unless efficiency is needed
    num_updatedtiles = 0
    for i, tileid in enumerate(tiles['TILEID']):
        j = np.where(specstatus['TILEID'] == tileid)[0][0]
        if update_all or tiles['LASTNIGHT'][i] > specstatus['LASTNIGHT'][j]:
            log.info('Updating TILEID {} LASTNIGHT {} (orig LASTNIGHT {})'.format(
                tileid, tiles['LASTNIGHT'][i], specstatus['LASTNIGHT'][j]))

            num_updatedtiles += 1
            for col in specstatus.colnames:
                if col not in qacols:
                    specstatus[col][j] = tiles[col][i]

    log.info(f'Added {num_newtiles} and updated {num_updatedtiles} tiles')

    specstatus.sort('TILEID')

    return specstatus

