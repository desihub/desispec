#!/usr/bin/env python
#
# See top-level LICENSE.rst file for Copyright information
#
# -*- coding: utf-8 -*-
"""
Inspect the desispec output for a single target.
"""
from __future__ import absolute_import, print_function
import argparse
import os.path

import numpy as np

import matplotlib.pyplot as plt

import astropy.table

import desispec.io
import desispec.coaddition
from desispec.log import get_logger, DEBUG

def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--verbose', action = 'store_true',
        help = 'Provide verbose reporting of progress.')
    parser.add_argument('--target', type = int, default = None, metavar = 'ID',
        help = 'Target ID number to inspect')
    parser.add_argument('--brick', type = str, default = None, metavar = 'NAME',
        help = 'Name of brick containing the requested target ID.')
    parser.add_argument('--info', action = 'store_true',
        help = 'Print tabular information from each file inspected.')
    parser.add_argument('--save-plot', type = str, default = None, metavar = 'FILE',
        help = 'File name to save the generated plot to.')
    parser.add_argument('--no-display', action = 'store_true',
        help = 'Do not display the image on screen (useful for batch processing).')
    parser.add_argument('--stride', type = int, default = 5,
        help = 'Stride to use for subsampling the spectrum data arrays.')
    parser.add_argument('--resolution-stride', type = int, default = 500,
        help = 'Stride to use for displaying the resolution.')
    parser.add_argument('--resolution-zoom', type = int, default = 100,
        help = 'Wavelength zoom to use for displaying the resolution.')
    parser.add_argument('--bands', type = str, default = 'brz',
        help = 'String listing the bands to include.')
    parser.add_argument('--specprod', type = str, default = None, metavar = 'PATH',
        help = 'Override default path ($DESI_SPECTRO_REDUX/$PRODNAME) to processed data.')
    args = parser.parse_args()
    if args.verbose:
        log = get_logger(DEBUG)
    else:
        log = get_logger()
    figure = plt.figure(figsize=(12,8))
    left_axis = plt.gca()
    figure.set_facecolor('white')
    plt.xlabel('Wavelength (Angstrom)')
    left_axis.set_ylabel('Flux (1e-17 erg/s/cm**2)')
    left_axis.set_ylim(-5,5)
    right_axis = left_axis.twinx()
    right_axis.set_ylabel('Resolution')
    right_axis.set_ylim(-0.02,1)

    colors = dict(b = 'blue', r = 'red', z = 'green')

    for band in args.bands:

        color = colors[band]
        wlen_min,wlen_max = 1e8,0.

        brick_path = desispec.io.meta.findfile('brick',brickid = args.brick,
            band = band,specprod = args.specprod)
        if not os.path.exists(brick_path):
            log.warning('No %s-band brick file found for brick %s.' % (band,args.brick))
        else:
            brick_file = desispec.io.brick.Brick(brick_path,mode = 'readonly')
            wlen = brick_file.get_wavelength_grid()
            wlen_min,wlen_max = min(wlen_min,np.min(wlen)),max(wlen_max,np.max(wlen))
            exp_flux,exp_ivar,exp_resolution,exp_info = brick_file.get_target(args.target)
            log.debug('Found %d %s-band exposures covering %.1f-%.1fA: %s' % (
                len(exp_flux),band,np.min(wlen),np.max(wlen),','.join(map(str,exp_info['EXPID']))))

            if len(exp_flux) > 0:
                if args.info:
                    exp_info = astropy.table.Table(exp_info)
                    print(exp_info)

                for flux in exp_flux:
                    plt.scatter(wlen[::args.stride],flux[::args.stride],color = color,s = 1.,alpha = 0.5)

            else:
                log.warning('No %s-band exposures recorded for target %d in brick %s' % (
                    band,args.target,args.brick))

            brick_file.close()

        coadd_path = desispec.io.meta.findfile('coadd',brickid = args.brick,
            band = band,specprod = args.specprod)
        if not os.path.exists(coadd_path):
            log.warning('No %s-band coadd file found for brick %s.' % (band,args.brick))
        else:
            coadd_file = desispec.io.brick.CoAddedBrick(coadd_path,mode = 'readonly')
            wlen = coadd_file.get_wavelength_grid()
            wlen_min,wlen_max = min(wlen_min,np.min(wlen)),max(wlen_max,np.max(wlen))
            coadd_flux,coadd_ivar,coadd_resolution,coadd_info = coadd_file.get_target(args.target)

            if len(coadd_flux) == 1:
                if args.info:
                    coadd_info = astropy.table.Table(coadd_info)
                    print(coadd_info)

                left_axis.scatter(wlen[::args.stride],coadd_flux[0,::args.stride],color = color,
                    marker = 'x',alpha = 0.5)

                R = desispec.resolution.Resolution(coadd_resolution[0]).toarray()
                ndiag = desispec.resolution.num_diagonals//2
                for index in range(0,len(R),args.resolution_stride):
                    bins = slice(index-ndiag,index+ndiag+1)
                    wlen_zoom = wlen[index] + args.resolution_zoom*(wlen[bins] - wlen[index])
                    right_axis.fill_between(wlen_zoom,R[index,bins],color = color,alpha = 0.1)

            elif len(coadd_flux) == 0:
                log.warning('No %s-band coadd available for target %d.' % (band,args.target))
            else:
                log.error('found %d (>1) coadded %d-band fluxes for target %d' % (
                    len(coadd_flux),band,args.target))

            coadd_file.close()

    coadd_all_path = desispec.io.meta.findfile('coadd_all',brickid = args.brick,specprod = args.specprod)
    if not os.path.exists(coadd_all_path):
        log.warning('No global coadd available for brick %s.' % (args.brick))
    else:
        coadd_all_file = desispec.io.brick.CoAddedBrick(coadd_all_path,mode = 'readonly')
        wlen = coadd_all_file.get_wavelength_grid()
        wlen_min,wlen_max = min(wlen_min,np.min(wlen)),max(wlen_max,np.max(wlen))
        coadd_flux,coadd_ivar,coadd_resolution,coadd_info = coadd_all_file.get_target(args.target)

        if len(coadd_flux) == 1:
            if args.info:
                coadd_info = astropy.table.Table(coadd_info)
                print(coadd_info)
            mask = (coadd_ivar[0] > 0)
            flux_error = np.zeros_like(coadd_flux[0])
            flux_error[mask] = coadd_ivar[0,mask]**-0.5
            left_axis.fill_between(wlen,coadd_flux[0]-flux_error,coadd_flux[0]+flux_error,
                color='black',alpha=0.2)
        elif len(coadd_flux) == 0:
            log.warning('No global coadd available for target %d.' % (args.target))
        else:
            log.error('found %d (>1) global coadded fluxes for target %d' % (
                len(coadd_flux),args.target))

        plt.xlim(wlen_min,wlen_max)
        coadd_all_file.close()

    if args.save_plot:
        figure.savefig(args.save_plot)
    if not args.no_display:
        plt.show()
    plt.close()

if __name__ == '__main__':
    main()
