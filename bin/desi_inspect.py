#!/usr/bin/env python
#
# See top-level LICENSE file for Copyright information
#
# -*- coding: utf-8 -*-
"""
Inspect the desispec output for a single target.
"""

import argparse
import os.path

import numpy as np

import matplotlib.pyplot as plt

import astropy.table

import desispec.io
import desispec.coaddition

def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--verbose', action = 'store_true',
        help = 'Provide verbose reporting of progress.')
    parser.add_argument('--id', type = int, default = None, metavar = 'ID',
        help = 'Target ID number to inspect')
    parser.add_argument('--brick', type = str, default = None, metavar = 'NAME',
        help = 'Name of brick containing the requested target ID.')
    parser.add_argument('--specprod', type = str, default = None, metavar = 'PATH',
        help = 'Override default path ($DESI_SPECTRO_REDUX/$PRODNAME) to processed data.')
    parser.add_argument('--stride', type = int, default = 5,
        help = 'Stride to use for subsampling the spectrum data arrays.')
    parser.add_argument('--resolution-stride', type = int, default = 500,
        help = 'Stride to use for displaying the resolution.')
    parser.add_argument('--resolution-zoom', type = int, default = 100,
        help = 'Wavelength zoom to use for displaying the resolution.')
    args = parser.parse_args()

    figure = plt.figure(figsize=(12,8))
    left_axis = plt.gca()
    figure.set_facecolor('white')
    plt.xlabel('Wavelength (Angstrom)')
    left_axis.set_ylabel('Flux (1e-17 erg/s/cm**2)')
    right_axis = left_axis.twinx()
    right_axis.set_ylabel('Resolution')
    right_axis.set_ylim(-0.02,1)

    colors = dict(b = 'blue', r = 'red', z = 'green')

    #for band in 'brz':
    for band in 'br':

        color = colors[band]

        brick_path = desispec.io.meta.findfile('brick',brickid = args.brick,band = band,specprod = args.specprod)
        if not os.path.exists(brick_path):
            print 'Target has not been bricked yet for %d-band' % band
            return -1
        brick_file = desispec.io.brick.Brick(brick_path,mode = 'readonly')
        wlen = brick_file.get_wavelength_grid()
        exp_flux,exp_ivar,exp_resolution,exp_info = brick_file.get_target(args.id)
        print 'Found %d exposures: %s' % (len(exp_flux),exp_info['EXPID'])
        exp_info = astropy.table.Table(exp_info)
        print exp_info

        for flux in exp_flux:
            plt.scatter(wlen[::args.stride],flux[::args.stride],color = color,s = 1.,alpha = 0.5)

        for resolution in exp_resolution:
            ndiag = len(resolution)//2
            R = desispec.io.frame.resolution_data_to_sparse_matrix(resolution).toarray()
            for index in range(0,len(R),args.resolution_stride):
                bins = slice(index-ndiag,index+ndiag+1)
                wlen_zoom = wlen[index] + args.resolution_zoom*(wlen[bins] - wlen[index])
                right_axis.plot(wlen_zoom,R[index,bins],color = color,ls = '-',alpha = 0.5)

        coadd_path = desispec.io.meta.findfile('coadd',brickid = args.brick,band = band,specprod = args.specprod)
        if not os.path.exists(brick_path):
            print 'Target brick has not been coadded yet.'
            continue
        coadd_file = desispec.io.brick.CoAddedBrick(coadd_path,mode = 'readonly')
        wlen = brick_file.get_wavelength_grid()
        coadd_flux,coadd_ivar,coadd_resolution,coadd_info = brick_file.get_target(args.id)
        assert len(coadd_flux) == 1,'Got more than one coadd: shape is %r' % coadd_flux.shape
        if not np.array_equal(coadd_info,exp_info):
            print 'Coadd is missing %d exposure(s).' % (len(exp_info)-len(coadd_info))

        left_axis.scatter(wlen[::args.stride],coadd_flux[0,::args.stride],color = color,s = 2.,alpha = 0.5)

    plt.show()
    plt.close()

if __name__ == '__main__':
    main()
