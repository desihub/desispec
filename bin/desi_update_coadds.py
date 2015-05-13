#!/usr/bin/env python
#
# See top-level LICENSE file for Copyright information
#
# -*- coding: utf-8 -*-
"""
Update co-adds for a single brick.

Reads the b,r,z files containing all observed spectra for a single brick and performs two steps
of coaddition: (1) create b,r,z coadd files containing the coadditions of every target observed
in each band, using the native band wavelength grid; and (2) combine the b,r,z coadds for each
object into a global coadd using linear resampling to the global wavelength grid.
"""

import argparse
import os.path

import numpy as np

import desispec.io
import desispec.coaddition
import desispec.resolution
from desispec.log import get_logger, DEBUG

def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--verbose', action = 'store_true',
        help = 'Provide verbose reporting of progress.')
    parser.add_argument('--brick', type = str, default = None, metavar = 'NAME',
        help = 'Name of brick to process')
    parser.add_argument('--target', type = int, action = 'append', metavar = 'ID', default = [ ],
        help = 'Only perform coaddition for the specified target ID (may be repeated).')
    parser.add_argument('--bands', type = str, default = 'brz',
        help = 'String listing the bands to include.')
    parser.add_argument('--specprod', type = str, default = None, metavar = 'PATH',
        help = 'Override default path ($DESI_SPECTRO_REDUX/$PRODNAME) to processed data.')
    args = parser.parse_args()
    if args.verbose:
        log = get_logger(DEBUG)
    else:
        log = get_logger()

    if args.brick is None:
        log.critical('Missing required brick argument.')
        return -1

    # Open the combined coadd file for this brick, for updating.
    coadd_all_path = desispec.io.meta.findfile('coadd_all',brickid = args.brick,specprod = args.specprod)
    coadd_all_file = desispec.io.brick.CoAddedBrick(coadd_all_path,mode = 'update')

    # Initialize dictionaries of co-added spectra for each object ID.
    coadded_spectra = { }

    # Keep track of the index we assign to each target.
    next_coadd_index = 0
    target_index = { }

    # The HDU4 table for the global coadd will go here.
    coadd_all_info = None

    # Loop over bands for this brick.
    for band in args.bands:
        # Open this band's brick file for reading.
        brick_path = desispec.io.meta.findfile('brick',brickid = args.brick,band = band,specprod = args.specprod)
        if not os.path.exists(brick_path):
            log.info('Skipping non-existent brick file {0}.'.format(brick_path))
            continue
        brick_file = desispec.io.brick.Brick(brick_path,mode = 'readonly')
        flux_in,ivar_in,wlen,resolution_in = (brick_file.hdu_list[0].data,brick_file.hdu_list[1].data,
            brick_file.hdu_list[2].data,brick_file.hdu_list[3].data)
        log.debug('Processing %s with %d exposures of %d targets...' % (
                brick_path,brick_file.get_num_spectra(),brick_file.get_num_targets()))
        if resolution_in.shape[1] != desispec.resolution.num_diagonals:
            log.error('resolution has unexpected shape (ndiag=%d != %d). Skipping this file.' % (
                resolution_in.shape[1],desispec.resolution.num_diagonals))
            brick_file.close()
            continue
        # Open this band's coadd file for updating.
        coadd_path = desispec.io.meta.findfile('coadd',brickid = args.brick,band = band,specprod = args.specprod)
        coadd_file = desispec.io.brick.CoAddedBrick(coadd_path,mode = 'update')
        # Copy the input fibermap info for each exposure into memory.
        coadd_info = np.copy(brick_file.hdu_list[4].data)
        # Also copy the first band's info to initialize the global coadd info, but remember that this
        # band might not have all targets so we could see new targets in other bands.
        if coadd_all_info is None:
            coadd_all_info = np.copy(brick_file.hdu_list[4].data)

        # Loop over objects in the input brick file.
        for index,info in enumerate(brick_file.hdu_list[4].data):
            assert index == info['INDEX'],'Index mismatch: %d != %d' % (index,info['INDEX'])
            resolution_matrix = desispec.resolution.Resolution(resolution_in[index])
            spectrum = desispec.coaddition.Spectrum(wlen,flux_in[index],ivar_in[index],resolution_matrix)
            target_id = info['TARGETID']
            # Are we only processing specified targets?
            if len(args.target) > 0 and target_id not in args.target:
                continue
            # Have we seen this target before?
            if target_id not in coadded_spectra:
                coadded_spectra[target_id] = { }
                target_index[target_id] = next_coadd_index
                next_coadd_index += 1
            # Save the coadd index to our output table.
            coadd_info['INDEX'][index] = target_index[target_id]
            # Initialize the coadd for this band and target if necessary.
            if band not in coadded_spectra[target_id]:
                coadded_spectra[target_id][band] = desispec.coaddition.Spectrum(wlen)
            # Do the coaddition.
            coadded_spectra[target_id][band] += spectrum

            # Is this exposure of this target already in our global coadd table?
            exposure = info['EXPID']
            seen = (coadd_all_info['EXPID'] == exposure) & (coadd_all_info['TARGETID'] == target_id)
            if not np.any(seen):
                log.info('Adding exposure %d of target %d to global coadd with partial band coverage.' % (
                    exposure,target_id))
                coadd_all_info.append(coadd_info[index])
            else:
                coadd_all_info['INDEX'][index] = target_index[target_id]

        # Allocate arrays for the coadded results for this band. Since we always use the same index
        # for the same target in each band, there might be some unused entries in these arrays if
        # some bands are missing for some targets.
        num_targets = 1 + np.max(coadd_info['INDEX'])
        nbins = len(wlen)
        flux_out = np.zeros((num_targets,nbins))
        ivar_out = np.zeros_like(flux_out)
        resolution_out = np.zeros((num_targets,desispec.resolution.num_diagonals,nbins))

        # Save the coadded spectra for this band.
        for target_id in coadded_spectra:
            if band not in coadded_spectra[target_id]:
                continue
            exposures = (coadd_info['TARGETID'] == target_id)
            index = target_index[target_id]
            log.debug('Saving coadd of %d exposures for target ID %d to index %d.' % (
                    np.count_nonzero(exposures),target_id,index))
            coadd = coadded_spectra[target_id][band]
            coadd.finalize()
            flux_out[index] = coadd.flux
            ivar_out[index] = coadd.ivar
            resolution_out[index] = coadd.resolution.to_fits_array()

        # Save the coadds for this band.
        coadd_file.add_objects(flux_out,ivar_out,wlen,resolution_out)
        coadd_file.hdu_list[4].data = coadd_info

        # Close files for this band.
        coadd_file.close()
        brick_file.close()

    # Allocate space for the global coadded results.
    num_targets = next_coadd_index
    nbins = len(desispec.coaddition.global_wavelength_grid)
    flux_all = np.empty((num_targets,nbins))
    ivar_all = np.empty_like(flux_all)
    resolution_all = np.empty((num_targets,desispec.resolution.num_diagonals,nbins))

    # Coadd the bands for each target ID.
    all_bands = ','.join(sorted(args.bands))
    for target_id in coadded_spectra:
        index = target_index[target_id]
        bands = ','.join(sorted(coadded_spectra[target_id].keys()))
        log.debug('Combining %s bands for target %d at index %d.' % (bands,target_id,index))
        if bands != all_bands:
            log.warning('WARNING: target %d has partial band coverage: %s' % (target_id,bands))
        coadd_all = desispec.coaddition.Spectrum(desispec.coaddition.global_wavelength_grid)
        for coadd_band in coadded_spectra[target_id].itervalues():
            coadd_all += coadd_band
        coadd_all.finalize()
        flux_all[index] = coadd_all.flux
        ivar_all[index] = coadd_all.ivar
        resolution_all[index] = coadd_all.resolution.to_fits_array()

    # Save the global coadds.
    coadd_all_file.add_objects(flux_all,ivar_all,desispec.coaddition.global_wavelength_grid,resolution_all)
    coadd_all_file.hdu_list[4].data = coadd_all_info

    # Close the combined coadd file.
    coadd_all_file.close()

if __name__ == '__main__':
    main()
