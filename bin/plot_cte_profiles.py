#!/usr/bin/env python

"""
Plot median of columns above/below amp boundaries
"""

import os, sys
import argparse
import numpy as np
import fitsio

import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument('-n', '--night', type=int,help='YEARMMDD night')
p.add_argument('-e', '--expid', type=int, help='Exposure ID')
p.add_argument('-c', '--camera', type=str, help='Camera')
p.add_argument('-i', '--input', help='input preproc file')
p.add_argument('--nrow', type=int, default=41,
        help='number of rows to include in median in each amp')
p.add_argument('--xminmax', nargs=2, type=int, default=(0, 0),
        help='x (column) range to plot')
p.add_argument('--debias',action='store_true')
p.add_argument('--below-in-front',action='store_true')
#p.add_argument('--psf',type=str,default=None,required=False,help='use this psf to fold the profiles')
p.add_argument('-r','--reference',type=str,default=None,required=False,help='reference preproc image to compare with (for 2 amp mode)')

args = p.parse_args()

n = args.nrow
xmin, xmax = args.xminmax

if args.input is None:
    from desispec.io import findfile
    args.input = findfile('preproc', night=args.night, expid=args.expid,
            camera=args.camera)

img = fitsio.read(args.input, 'IMAGE')

ny, nx = img.shape

if xmax == 0 :
    #xmin = nx//2 -300
    #xmax = nx//2 + 300
    xmin = 0
    xmax = nx
above = np.mean(img[ny//2:ny//2+n, xmin:xmax], axis=0)
below = np.mean(img[ny//2-n:ny//2, xmin:xmax], axis=0)

if args.debias :
    margin=100
    bias1 = np.median(img[ny//2:ny//2+n,0:margin])
    bias2 = np.median(img[ny//2:ny//2+n,nx-margin:nx])
    print("above bias = {} {}".format(bias1,bias2))
    above[:nx//2-xmin] -= bias1
    above[nx//2-xmin:] -= bias2
    bias1 = np.median(img[ny//2-n:ny//2,0:margin])
    bias2 = np.median(img[ny//2-n:ny//2,nx-margin:nx])
    print("below bias = {} {}".format(bias1,bias2))
    below[:nx//2-xmin] -= bias1
    below[nx//2-xmin:] -= bias2






xx = np.arange(xmin, xmax)

title=os.path.basename(args.input).split(".")[0]
plt.figure(title)
plt.subplot(211)
plt.title(os.path.basename(args.input))
extent = [xmin-0.5, xmax-0.5, ny//2-n-0.5, ny//2+n-0.5]
plt.imshow(img[ny//2-n:ny//2+n, xmin:xmax], vmin=-5, vmax=80, extent=extent,aspect='auto')
plt.axhline(ny//2,linestyle="--",color="white")
plt.subplot(212)
if args.below_in_front :
  plt.plot(xx, above, label='above',alpha=0.6)
  plt.plot(xx, below, label='below',alpha=0.6)
else :
    plt.plot(xx, below, label='below',alpha=0.6)
    plt.plot(xx, above, label='above',alpha=0.6)
plt.axvline(nx//2,linestyle="--",color="k")
plt.legend(loc="upper left")
plt.title(f'median of {n} rows above/below CCD amp boundary')
plt.ylim(-5,min(80,max(np.max(above),np.max(below))))
plt.xlim(xmin, xmax)
plt.xlabel('CCD column')
plt.grid(True)

if args.reference is not None :
    title="{}-vs-{}".format(os.path.basename(args.input).split(".")[0],os.path.basename(args.reference).split(".")[0])
    plt.figure(title)
    refimg = fitsio.read(args.reference, 'IMAGE')
    prof = np.median(img[ny//2-n:ny//2+n, xmin:xmax], axis=0)
    refprof = np.median(refimg[ny//2-n:ny//2+n, xmin:xmax], axis=0)
    scale = np.sum(prof*refprof)/np.sum(refprof**2)
    ok=(refprof>10)
    scale = np.median(prof[ok]/refprof[ok])
    refprof *= scale
    plt.plot(xx, refprof, label='{:0.3f} x {}'.format(scale,os.path.basename(args.reference)),alpha=0.6)
    plt.plot(xx, prof, label=os.path.basename(args.input),alpha=0.6)
    plt.title(f'median of {2*n} rows in the center of the CCD')
    plt.ylim(-5,min(80,max(np.max(above),np.max(below))))
    plt.xlim(xmin, xmax)
    plt.xlabel('CCD column')
    plt.legend(loc="upper left")
    plt.grid(True)

plt.figure()
# folding


plt.tight_layout()
plt.show()
