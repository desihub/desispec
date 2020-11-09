import argparse
import os
import fitsio
import astropy.io.fits as pyfits
from astropy.io import fits
import subprocess
import time
import numpy as np
import psycopg2
import hashlib
from os import listdir
import matplotlib.pyplot as plt

"""
###################################################
############# Usage Manual ########################
###################################################

* Pipeline for gnerating new bias+dark files
Input: night array to use

python3 desi_create_bias_dark.py -n 20200729 20200730 -r 60633 60634 60635 60636 60637 60638 60639 60640 60641 60642
"""
##########################################
############# Input ######################
##########################################
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
description="Search all bias and dark images for a series of night, Compute a master bias from a set of raw images, and combine them according to exptime",
epilog='''Cool.'''
)
parser.add_argument('-n','--nights', type = str, default = None, required = True, nargs="*",
                    help = 'nights to be used')
parser.add_argument('-r','--reject', type = int, default=[], required = False, nargs="*",
                    help = 'exposures to be rejected')
parser.add_argument('-o','--outdir', type = str, default='.', required = False,
                    help = 'output directory')
parser.add_argument('-c','--cameras', type = str, default=None, required = False, nargs="*",
                    help = 'Cameras to use (e.g. b0 r1 z9)')


args        = parser.parse_args()
night_arr=args.nights
exp_reject=args.reject

if args.cameras is None:
    tmp = list()
    for sp in range(10):
        for c in ['b', 'r', 'z']:
            tmp.append(c+str(sp))

    args.cameras = tmp

expid_all=[]
exptime_all=[]
flavor_all=[]
night_all=[]

for night in night_arr:
    expid_arr=listdir(os.getenv('DESI_SPECTRO_DATA')+'/'+night+'/')
    expid_arr.sort(reverse=False)
    for expid in expid_arr:
        if int(expid) in args.reject:
            print(f'Skipping rejected expid {expid}')
            continue

        filename=os.getenv('DESI_SPECTRO_DATA')+'/'+str(night)+'/'+str(expid).zfill(8)+'/desi-'+str(expid).zfill(8)+'.fits.fz'
        try:
            h1=fits.getheader(filename,1)
        except:
            continue

        flavor=h1['flavor'].strip().lower()
        exptime=h1['EXPTIME']
        print(night,expid,flavor,exptime)
        if flavor=='dark' or flavor=='zero':
            expid_all.append(expid)
            exptime_all.append(int(exptime))
            flavor_all.append(flavor)
            night_all.append(night)

n_exp=len(expid_all)


exptime_set_arr=np.unique(np.array(exptime_all)).tolist()

sp_arr=['0', '1','2','3','4','5','6','7','8','9']
sm_arr=['4','10','5','6','1','9','7','8','2','3']
cam_arr=['b','r','z']
##############################################################################################
############### Search all exposures with a specific exptime and compile them ################
##############################################################################################
filename_list_all={}
for exptime_set in exptime_set_arr:
    
    filename_list=''
    print('Adding exptime='+str(exptime_set))

    for i in range(n_exp):
        expid=expid_all[i]
        exptime=exptime_all[i]
        night=night_all[i]
        filename=os.getenv('DESI_SPECTRO_DATA')+'/'+str(night)+'/'+str(expid).zfill(8)+'/desi-'+str(expid).zfill(8)+'.fits.fz'

        if int(exptime) == int(exptime_set):
            filename_list=filename_list+filename+' '


    filename_list_all[exptime_set]=filename_list
    for camera in args.cameras:
        outfile = os.path.join(args.outdir, f'tmp-bias-dark-{camera}-{exptime_set}.fits')
        if not os.path.exists(outfile):
            cmd='desi_compute_bias -i '+filename_list+' -o '+outfile+' --camera '+camera
            print(cmd)
            os.system(cmd)
        else:
            print('Already done: {}'.format(os.path.basename(outfile)))

##############################################################################################
############### Compile the exposures at a specific exptime to a single file  ################
##############################################################################################
print('Now compile the exposures at a specific exptime to a single file')

for camera in args.cameras:
    hdus = fits.HDUList()
    output=os.path.join(args.outdir, 'master-bias-dark-'+camera+'.fits')
    print(camera)
    print(output)
    cutout_flux=[]
    for exptime_set in exptime_set_arr:
        filename = os.path.join(args.outdir, f'tmp-bias-dark-{camera}-{exptime_set}.fits')
        hdu_this = fits.open(filename)
        ## Store the max and min in the header ##
        name='T'+str(exptime_set)
        hdu_this[0].header['FILELIST']=filename_list_all[exptime_set]

        dataHDU = fits.ImageHDU(hdu_this[0].data, header=hdu_this[0].header, name=name)
        hdus.append(dataHDU)
 
    hdus.writeto(output, overwrite=True)


