import sys
import os
import re
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import copy
import desispec.preproc
import argparse
"""
Usage: python3 desi_bias_dark_1d_model.py
"""
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
description="Search all bias and dark images for a series of night, Compute a master bias from a set of raw images, and combine them according to exptime",
epilog='''Cool.'''
)
parser.add_argument('--mintime-linear-dark', type=int, default=600, required=False,
                    help = 'Minimum exposure time for linear evolution of bias+dark')
parser.add_argument('-p','--plot', type = bool, default = False, required = False, nargs="*",
                    help = 'If you want to output plot to check or not')

args = parser.parse_args()

camera_arr=['b0','b1','b2','b3','b4','b5','b6','b7','b8','b9','r0','r1','r2','r3','r4','r5','r6','r7','r8','r9','z0','z1','z2','z3','z4','z5','z6','z7','z8','z9']

prefix="master-bias-dark-"
plot = args.plot
exp_arr=args.exp
#exp_arr=[int(t) for t in args.exp]

def calculate_dark(exp_arr,image_arr):
    n_images=len(image_arr)
    n0=image_arr[0].shape[0]
    n1=image_arr[0].shape[1]
    # model
    dark=np.zeros((n0,n1))
    niterations=1
    for iteration in range(niterations) :
        print(f'Dark fit iteration {iteration}')

        # fit dark
        A = np.zeros((2,2))
        b0  = np.zeros((n0,n1))
        b1  = np.zeros((n0,n1))

        for image,exptime in zip(image_arr,exp_arr) :
            print(f'Adding exptime {exptime}')
            res = image
            A[0,0] += 1
            A[0,1] += exptime
            A[1,0] += exptime
            A[1,1] += exptime**2
            b0 += res
            b1 += res*exptime
        Ai = np.linalg.inv(A)
        # const + exptime * dark
        const = Ai[0,0]*b0 + Ai[0,1]*b1
        dark  = Ai[1,0]*b0 + Ai[1,1]*b1
    return dark, const

for camera in camera_arr:
    filename=prefix+camera+".fits"
    if os.path.exists(filename):
        hdu_this = fits.open(filename)
    except:
        print('Can not find file:'+filename)
        continue

    nx=len(hdu_this[0].data) #4162
    ny=len(hdu_this[0].data[0]) #4232
    #header=hdu_this[exptime].header
    #jj = desispec.preproc.parse_sec_keyword(header['DATASEC'+amp])
    indA = desispec.preproc.parse_sec_keyword('[1:'+str(int(nx/2))+',1:'+str(int(ny/2))+']')
    indB = desispec.preproc.parse_sec_keyword('['+str(int(nx/2)+1)+':'+str(int(nx))+',1:'+str(int(ny/2))+']')
    indC = desispec.preproc.parse_sec_keyword('[1:'+str(int(nx/2))+','+str(int(ny/2)+1)+':'+str(int(ny))+']')
    indD = desispec.preproc.parse_sec_keyword('['+str(int(nx/2)+1)+':'+str(int(nx))+','+str(int(ny/2)+1)+':'+str(int(ny))+']')
    image_arr=[]
    for exp in exp_arr:
        image_arr.append(hdu_this[str(exp)].data)

    dark=calculate_dark(exp_arr,image_arr)

    hdr_dark = fits.Header()
    dataHDU = fits.ImageHDU(dark,header=hdr_dark, name='dark')
    hdu_this.append(dataHDU)
    exptime_arr=[]
    image_arr=[]
    bias_image = None
    for hdu in hdu_this:
        #- EXTNAME = Tnnn
        if re.match('T\d+$', hdu.name):
            exptime_arr.append(int(hdu.name[1:]))
            image_arr.append(hdu.data)

        if hdu.name == 'T0':
            bias_image = hdu.data

    assert bias_image is not None

    exptime_arr = np.array(exptime_arr)
    image_arr = np.array(image_arr)

    #- Subtract the bias image from every image and remove the overscan
    #- since overscans shouldn't be included in the dark calculation
    for i in range(len(image_arr)):
        image_arr[i] -= bias_image

    #- Calculate the dark current using exposures long enough to
    #- establish linear dark current
    ii = (exptime_arr >= args.mintime_linear_dark)
    dark, const = calculate_dark(exptime_arr[ii], image_arr[ii])

    ny,nx = hdu_this[0].data.shape
    indLeft = (slice(0, ny), slice(0, nx//2))
    indRight = (slice(0, ny), slice(nx//2, 0))

    #- Model any remaining residuals as a 1D function of row for the
    #- left (A,C) and right (B,D) amplifiers

    for exptime, image in zip(exptime_arr, image_arr):

        #- bias image was already subtracted before calculating dark
        residual = image - exptime*dark

        ###### Pass1 subtract bias #######
        pass1=hdu_this['T'+str(exptime)].data-hdu_this['T0'].data
        pass1_1d=pass1.ravel()
        std_pass1=np.std(pass1_1d[(pass1_1d<100) & (pass1_1d>-10)])
        print('std_pass1',std_pass1)
        correction=1.0
        for i in range(1):
            ###### Pass2 Subtract dark current #######
            print('dark ',np.median(dark))
            pass2=pass1-correction*dark*float(exptime)

            ###### Pass3 subtract 1D profile #######
            profileA=np.median(pass2[indA],axis=1)
            profileB=np.median(pass2[indB],axis=1)
            profileC=np.median(pass2[indC],axis=1)
            profileD=np.median(pass2[indD],axis=1)
            profileLeft=profileA.tolist()+profileD.tolist()
            profileRight=profileB.tolist()+profileC.tolist()
            #profile_1d=np.median(pass2,axis=1) # 4162
            profile_2d_Left=np.transpose(np.tile(profileLeft,(int(ny/2),1)))
            profile_2d_Right=np.transpose(np.tile(profileRight,(int(ny/2),1)))
            profile_2d=np.concatenate((profile_2d_Left,profile_2d_Right),axis=1)
            try:
                pass3=pass2-profile_2d
            except:
                import IPython; IPython.embed()
                sys.exit(1)

            correction=1.+np.median(pass3.ravel()/(float(exptime)*dark.ravel()))
            data1d=pass3.ravel()
            std=np.std(data1d[(data1d<10) & (data1d>-10)])
            print('correction ',correction,' std ',std)
        # Store pass3 stddev
        hdu_this[exptime].header['res_std']=std
        # Store 1D profile
        hdu_this[exptime].data=np.array([profileLeft,profileRight]).astype('float32') # 
        
        if args.plot:
            plt.figure(0,figsize=(25,16))
            font = {'family' : 'sans-serif',
                    'weight' : 'normal',
                    'size'   : 10}
            plt.rc('font', **font)

            plt.subplot(241)
            plt.imshow(hdu_this['0'].data,vmin=-0.5,vmax=2)
            plt.title('Bias Image')
            plt.colorbar()

            plt.subplot(242)
            plt.imshow(pass1,vmin=-1,vmax=3)
            plt.title(camera+' '+exptime+'s After Bias Subtraction')
            plt.colorbar()

            plt.subplot(243)
            plt.imshow(dark,vmin=0,vmax=1.5/750.)
            plt.title('Dark')
            plt.colorbar()

            plt.subplot(244)
            plt.imshow(pass2,vmin=-0.5,vmax=0.5)
            plt.title('After removing dark current')
            plt.colorbar()

            plt.subplot(245)
            plt.imshow(profile_2d,vmin=-0.5,vmax=0.5)
            plt.title('2D Profile')
            plt.colorbar()

            plt.subplot(246)
            plt.imshow(pass3,vmin=-0.5,vmax=0.5)
            plt.title('After 2D Profile Subtraction')
            plt.colorbar()

            plt.subplot(247)
            plt.hist(pass3.ravel(),30,range=(-5,5),alpha=0.5)
            plt.xlabel('Residual')
            plt.title('Std='+str(std)[0:4])
            plt.show()
            print(hdu_this.info())

    hdr_dark = fits.Header()
    dataHDU = fits.ImageHDU(dark,header=hdr_dark, name='dark')
    hdu_this.append(dataHDU)

    try:
        for hdu in hdu_this:
            if hdu.name =='0':
                hdu.name='ZERO'
                hdu.data=hdu.data.astype('float32')
            elif hdu.name == 'DARK':
                hdu.data=hdu.data.astype('float32')
            else:
                hdu.name = 'T'+hdu.name
        hdu_this.writeto(prefix+camera+'-compressed.fits')
    except:
        print(prefix+camera+'-compressed.fits exists')
