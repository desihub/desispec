#!/usr/bin/env python
#
# See top-level LICENSE file for Copyright information
#
# -*- coding: utf-8 -*-
"""
Update co-adds for a single brick.
"""

import argparse
import os.path

import numpy as np

import desispec.io
import desispec.coaddition

def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--verbose', action = 'store_true',
        help = 'Provide verbose reporting of progress.')
    parser.add_argument('--brick', type = str, default = None, metavar = 'NAME',
        help = 'Name of brick to process')
    parser.add_argument('--specprod', type = str, default = None, metavar = 'PATH',
        help = 'Override default path ($DESI_SPECTRO_REDUX/$PRODNAME) to processed data.')
    args = parser.parse_args()

    if args.brick is None:
        print 'Missing required brick argument.'
        return -1

    # Open the combined coadd file for this brick, for updating.
    coadd_all_path = desispec.io.meta.findfile('coadd_all',brickid = args.brick,specprod = args.specprod)
    coadd_all_file = desispec.io.brick.CoAddedBrick(coadd_all_path,mode = 'update')

    # Initialize dictionaries of co-added spectra for each band and object ID.
    coadded_spectra = dict(b = { },r = { },z = { })

    # Keep track of the index we assign to each target.
    target_index = { }

    # Loop over bands for this brick.
    for band in 'brz':
        # Open this band's brick file for reading.
        brick_path = desispec.io.meta.findfile('brick',brickid = args.brick,band = band,specprod = args.specprod)
        if not os.path.exists(brick_path):
            print 'Skipping non-existent brick file',brick_path
            continue
        brick_file = desispec.io.brick.Brick(brick_path,mode = 'readonly')
        num_objects = brick_file.get_num_objects()
        flux_in,ivar_in,wlen,resolution_in = (brick_file.hdu_list[0].data,brick_file.hdu_list[1].data,
            brick_file.hdu_list[2].data,brick_file.hdu_list[3].data)
        if args.verbose:
            print 'Processing %s with %d objects...' % (brick_path,num_objects)
        # Open this band's coadd file for updating.
        coadd_path = desispec.io.meta.findfile('coadd',brickid = args.brick,band = band,specprod = args.specprod)
        coadd_file = desispec.io.brick.CoAddedBrick(coadd_path,mode = 'update')
        # Copy the input fibermap info for each exposure into memory.
        coadd_info = np.copy(brick_file.hdu_list[4].data)

        # Loop over objects in the input brick file.
        next_coadd_index = 0
        for index,info in enumerate(brick_file.hdu_list[4].data):
            assert index == info['INDEX'],'Index mismatch: %d != %d' % (index,info['INDEX'])
            # Have we already added this exposure?
            resolution_matrix = desispec.io.frame.resolution_data_to_sparse_matrix(resolution_in[index])
            spectrum = desispec.coaddition.Spectrum(wlen,flux_in[index],ivar_in[index],resolution_matrix)
            target_id = info['TARGETID']
            # Add this observation to our coadd of this target.
            if target_id not in coadded_spectra[band]:
                coadded_spectra[band][target_id] = spectrum
                target_index[target_id] = next_coadd_index
                next_coadd_index += 1
            else:
                coadded_spectra[band][target_id] += spectrum
            coadd_info['INDEX'][index] = target_index[target_id]

        # Allocate arrays for the coadded results to be saved in the output FITS file.
        target_set = set(coadd_info['TARGETID'])
        num_targets = len(target_set)
        assert num_targets == next_coadd_index,'Coadd indexing error: %d != %d' % (num_targets,next_coadd_index)
        nbins = len(wlen)
        flux_out = np.empty((num_targets,nbins))
        ivar_out = np.empty_like(flux_out)
        ndiag = resolution_in.shape[1]
        resolution_out = np.empty((num_targets,ndiag,nbins))

        # Save the coadded spectra for this band.
        for target_id in coadded_spectra[band]:
            exposures = (coadd_info['TARGETID'] == target_id)
            index = coadd_info['INDEX'][exposures][0]
            if args.verbose:
                print 'Saving coadd of %d exposures for target ID %d to index %d' % (
                    np.count_nonzero(exposures),target_id,index)
            spectrum = coadded_spectra[band][target_id]
            spectrum.finalize(sparse_cutoff = ndiag//2)
            flux_out[index] = spectrum.flux
            ivar_out[index] = spectrum.ivar
            # Convert the DIA-format sparse matrix data into the canonical decreasing offset format
            # used in our FITS file.
            row_order = np.argsort(spectrum.resolution.offsets)[::-1]
            resolution_out[index] = spectrum.resolution.data[row_order]
        coadd_file.add_objects(flux_out,ivar_out,wlen,resolution_out)

        # Save the coadd info to the output file.
        coadd_file.hdu_list[4].data = coadd_info

        # Close files for this band.
        coadd_file.close()
        brick_file.close()

    # Close the combined coadd file.
    coadd_all_file.close()

if __name__ == '__main__':
    main()
