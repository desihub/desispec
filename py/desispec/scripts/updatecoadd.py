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
from astropy.io import fits

import desispec.io
import desispec.coaddition
import desispec.resolution
from desiutil.log import get_logger, DEBUG


def parse(options=None):
    parser = argparse.ArgumentParser(description="Update co-adds for a single brick.")
    parser.add_argument('--verbose', action = 'store_true',
        help = 'Provide verbose reporting of progress.')
    parser.add_argument('--brick', type = str, default = None, metavar = 'NAME',
        help = 'Name of brick to process')
    parser.add_argument('--target', type = int, metavar = 'ID', default = None,nargs="*",
        help = 'Only perform coaddition for the specified target ID(s).')
    parser.add_argument('--objtype', type = str, default = None,nargs="*",
        help = 'Only perform coaddition for the specified target type(s).')
    parser.add_argument('--bands', type = str, default = 'brz',
        help = 'String listing the bands to include.')
    parser.add_argument('--specprod', type = str, default = None, metavar = 'PATH',
        help = 'Override default path ($DESI_SPECTRO_REDUX/$SPECPROD) to processed data.')
    parser.add_argument('--fast', action="store_true", required=False,
        help = 'coadd using inverse variance weighting instead of full spectro-perfectionism')

    args = None
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args


def main(args):

    if args.verbose:
        log = get_logger(DEBUG)
    else:
        log = get_logger()

    if args.brick is None:
        log.critical('Missing required brick argument.')
        return -1

    # Open the combined coadd file for this brick, for updating.
    coadd_all_path = desispec.io.meta.findfile('coadd_all',brickname = args.brick,specprod_dir = args.specprod)
    header={"BRICKNAM":args.brick}
    coadd_all_file = desispec.io.brick.CoAddedBrick(coadd_all_path,mode = 'update',header=header)

    # Initialize dictionaries of co-added spectra for each object ID.
    coadded_spectra = { }

    # Keep track of the index we assign to each target.
    next_coadd_index = 0

    # The HDU4 table for the global coadd will go here.
    coadd_all_info = None

    ## keep track of the num_targes in each band in case they are different
    num_targets = {}

    # Loop over bands for this brick.
    for band in args.bands:
        # Open this band's brick file for reading.
        brick_path = desispec.io.meta.findfile('brick',brickname = args.brick,band = band,specprod_dir = args.specprod)
        if not os.path.exists(brick_path):
            log.info('Skipping non-existent brick file {0}.'.format(brick_path))
            continue
        brick_file = desispec.io.brick.Brick(brick_path,mode = 'readonly')
        flux_in,ivar_in,wlen,resolution_in = (brick_file.hdu_list[0].data,\
                brick_file.hdu_list[1].data,brick_file.hdu_list[2].data,\
                brick_file.hdu_list[3].data)
        log.debug('Processing %s with %d exposures of %d targets...' % (
                brick_path,brick_file.get_num_spectra(),brick_file.get_num_targets()))

        ## get the unique target id list
        utid,iutid = np.unique(brick_file.hdu_list[4].data.TARGETID,return_index=True)
        num_targets[band] = len(utid)

        # Open this band's coadd file for updating.
        coadd_path = desispec.io.meta.findfile('coadd',brickname = args.brick,\
                band = band, specprod_dir = args.specprod)
        coadd_file = desispec.io.brick.CoAddedBrick(coadd_path,mode = 'update',header=header)

        # Copy the input fibermap info for each exposure into memory.
        coadd_info = brick_file.hdu_list[4].data[iutid]
        # Also copy the first band's info to initialize the global coadd info, but remember that this
        # band might not have all targets so we could see new targets in other bands.
        if coadd_all_info is None:
            coadd_all_info = brick_file.hdu_list[4].data[iutid]

        # Loop over objects in the input brick file.
        if args.target is not None:
            w = np.in1d(coadd_info["TARGETID"],args.target)
            coadd_info = coadd_info[w]
            ## also fix the info
            w = np.in1d(coadd_all_info["TARGETID"],args.target)
            coadd_all_info = coadd_all_info[w]

        if args.objtype is not None:
            w = np.in1d(coadd_info["OBJTYPE"],args.objtype)
            coadd_info = coadd_info[w]
            ## also fix the info
            w = np.in1d(coadd_all_info["OBJTYPE"],args.objtype)
            coadd_all_info = coadd_all_info[w]

        assert len(coadd_info)>0,"no targets found with the specified target ids and object types"

        for info in coadd_info:
            target_id = info["TARGETID"]
            w = brick_file.hdu_list[4].data.TARGETID == target_id
            # Have we seen this target before?
            if target_id not in coadded_spectra:
                coadded_spectra[target_id] = { }
            log.info("Reading {} {}".format(target_id,info["EXPID"]))
            # Initialize the coadd for this band and target if necessary.
            if band not in coadded_spectra[target_id]:
                coadded_spectra[target_id][band] = desispec.coaddition.Spectrum(wlen,fast=args.fast)
            for fl,iv,re in zip(flux_in[w],ivar_in[w],resolution_in[w]):
                resolution_matrix = desispec.resolution.Resolution(re)
                spectrum = desispec.coaddition.Spectrum(wlen,fl,iv,resolution=resolution_matrix,fast=args.fast)
                # Do the coaddition.
                coadded_spectra[target_id][band] += spectrum


        # Allocate arrays for the coadded results for this band. Since we always use the same index
        # for the same target in each band, there might be some unused entries in these arrays if
        # some bands are missing for some targets.
        nbins = len(wlen)
        flux_out = np.zeros((num_targets[band],nbins))
        ivar_out = np.zeros_like(flux_out)
        resolution_out = []

        # Save the coadded spectra for this band.
        for i,info in enumerate(coadd_info):
            target_id = info["TARGETID"]
            log.info("Coadding band {} {}".format(band,target_id))
            exposures = (coadd_info['TARGETID'] == target_id)
            log.debug('Saving coadd of %d exposures for target ID %d to index %d.' % (
                    np.count_nonzero(exposures),target_id,i))
            coadd = coadded_spectra[target_id][band]
            coadd.finalize()
            flux_out[i] = coadd.flux
            ivar_out[i] = coadd.ivar
            resolution_out.append(coadd.resolution.to_fits_array())

        resolution_out = np.array(resolution_out)
        # Save the coadds for this band.
        coadd_file.add_objects(flux_out,ivar_out,wlen,resolution_out)
        coadd_file.hdu_list[4].data = coadd_info

        # Close files for this band.
        coadd_file.close()
        brick_file.close()

    # Allocate space for the global coadded results.
    global_num_targets = max(num_targets.values())

    ## check whether some bands have fewer targets
    for b in num_targets:
        if num_targets[b] != global_num_targets:
            log.warning("WARNING: band {} has fewer targets {}".format(b,global_num_targets))
    nbins = len(desispec.coaddition.global_wavelength_grid)
    flux_all = np.empty((global_num_targets,nbins))
    ivar_all = np.empty_like(flux_all)
    resolution_all = []

    # Coadd the bands for each target ID.
    all_bands = ','.join(sorted(args.bands))
    for i,info in enumerate(coadd_all_info):
        target_id = info["TARGETID"]
        bands = ','.join(sorted(coadded_spectra[target_id].keys()))
        log.debug('Combining %s bands for target %d at index %d.' % (bands,target_id,i))
        if bands != all_bands:
            log.warning('WARNING: target %d has partial band coverage: %s' % (target_id,bands))
        coadd_all = desispec.coaddition.Spectrum(desispec.coaddition.global_wavelength_grid,fast=args.fast)
        for coadd_band in coadded_spectra[target_id].values():
            coadd_all += coadd_band
        coadd_all.finalize()
        flux_all[i] = coadd_all.flux
        ivar_all[i] = coadd_all.ivar
        resolution_all.append(coadd_all.resolution.to_fits_array())

    resolution_all = np.array(resolution_all)
    # Save the global coadds.
    coadd_all_file.add_objects(flux_all,ivar_all,desispec.coaddition.global_wavelength_grid,resolution_all)
    coadd_all_file.hdu_list[4].data = coadd_all_info

    # Close the combined coadd file.
    coadd_all_file.close()
