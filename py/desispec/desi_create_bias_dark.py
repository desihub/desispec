import argparse
import os
import fitsio
import astropy.io.fits as pyfits
from astropy.io import fits
import subprocess
import pandas as pd
import time
import numpy as np
import psycopg2
import hashlib
import pdb
from os import listdir
import matplotlib.pyplot as plt

"""
###################################################
############# Usage Manual ########################
###################################################

* Pipeline for gnerating new bias+dark files
Input: night array to use

python3 desi_create_bias_dark.py -n 20200729 20200730 -r 0060633 0060634 0060635 0060636 0060637 0060638 0060639 0060640 0060641 0060642
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
parser.add_argument('-r','--reject', type = str, default = '', required = False, nargs="*",
                    help = 'exposures to be rejected')


args        = parser.parse_args()
night_arr=args.nights
exp_reject=args.reject #['0060633','0060634','0060635','0060636','0060637','0060638','0060639','0060640','0060641','0060642'] # Exposure time to reject

expid_all=[]
exptime_all=[]
flavor_all=[]
night_all=[]

for night in night_arr:
    expid_arr=listdir(os.getenv('DESI_SPECTRO_DATA')+'/'+night+'/')
    expid_arr.sort(reverse=False)
    for expid in expid_arr:
        filename=os.getenv('DESI_SPECTRO_DATA')+'/'+str(night)+'/'+str(expid).zfill(8)+'/desi-'+str(expid).zfill(8)+'.fits.fz'
        try:
            h1=fits.getheader(filename,1)
        except:
            continue
        flavor=h1['flavor'].strip()
        exptime=h1['EXPTIME']
        print(expid,flavor,exptime)
        if flavor=='dark' or flavor=='zero':
            expid_all.append(expid)
            exptime_all.append(exptime)
            flavor_all.append(flavor)
            night_all.append(night)

n_exp=len(expid_all)

exptime_set_arr=np.unique(np.array(exptime_all)).tolist()
output_dir='' # output direcotry. If current directory, use ''. Otherwise, use


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

        if((str(expid).zfill(8) not in exp_reject) and (int(exptime)==int(exptime_set))):
            filename_list=filename_list+filename+' '


    filename_list_all[exptime_set]=filename_list
    for cam in cam_arr:
        for sp in sp_arr:
            camera=cam+sp
            cmd='desi_compute_bias -i '+filename_list+' -o '+output_dir+'master_bias_dark_'+camera+'_'+str(int(exptime_set))+'.fits --camera '+camera
            print(cmd)
            os.system(cmd)

##############################################################################################
############### Compile the exposures at a specific exptime to a single file  ################
##############################################################################################
print('Now compile the exposures at a specific exptime to a single file')

for cam in cam_arr:
    for sp in sp_arr:
        camera=cam+sp
        hdus = fits.HDUList()
        output=output_dir+'master-bias-dark-'+camera+'.fits'
        print(camera)
        print(output)
        cutout_flux=[]
        try:
            for exptime_set in exptime_set_arr:
                filename=output_dir+'master_bias_dark_'+camera+'_'+str(int(exptime_set))+'.fits'
                hdu_this = fits.open(filename)
                ## Store the max and min in the header ##
                if str(exptime_set)=='0':
                    name='ZERO'
                else:
                    name='T'+str(exptime_set)
                hdu_this[0].header['FILELIST']=filename_list_all[exptime_set]
                dataHDU = fits.ImageHDU(hdu_this[0].data, header=hdu_this[0].header, name=str(exptime_set))
                hdus.append(dataHDU)
            hdus.writeto(output)
        except:
            pass


