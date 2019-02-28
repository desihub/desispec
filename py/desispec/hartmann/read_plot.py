from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
import numpy as np
import os
plt.style.use(astropy_mpl_style)
image_file= '/project/projectdirs/desi/spectro/teststand/rawdata/EMSpectrograph/PerformanceTestWL/RawData/2017/20170316/WINLIGHT_00004666.fits'
image_file='/global/homes/z/zhangkai/data/pix-r1-00004666.fits'
image_data = fits.getdata(image_file, ext=0)
#desi_preproc --infile WINLIGHT_00004666.fits --cameras r1 --pixfile $SCRATCH/temp/pix-r1-00004666.fits
fig= plt.figure()
a=fig.add_subplot(1,1,1)
a.set_title('4666 R1 Intensity')
plt.xlabel('X')
plt.ylabel('Y')
plt.imshow((image_data),clim=(0.,10000.),cmap='gray')
plt.colorbar()
plt.show()

image_file='/global/homes/z/zhangkai/data/pix-b1-00004666.fits'
image_data = fits.getdata(image_file, ext=0)
#desi_preproc --infile WINLIGHT_00004666.fits --cameras r1 --pixfile $SCRATCH/temp/pix-r1-00004666.fits
fig= plt.figure()
a=fig.add_subplot(1,1,1)
a.set_title('4666 B1 Intensity')
plt.xlabel('X')
plt.ylabel('Y')
plt.imshow((image_data),clim=(0.,5000.),cmap='gray')
plt.colorbar()
plt.show()

image_file='/global/homes/z/zhangkai/data/pix-z1-00004666.fits'
image_data = fits.getdata(image_file, ext=0)
#desi_preproc --infile WINLIGHT_00004666.fits --cameras r1 --pixfile $SCRATCH/temp/pix-r1-00004666.fits
fig= plt.figure()
a=fig.add_subplot(1,1,1)
a.set_title('4666 Z1 Intensity')
plt.xlabel('X')
plt.ylabel('Y')
plt.imshow((image_data),clim=(0.,10000.),cmap='gray')
plt.colorbar()
plt.show()


