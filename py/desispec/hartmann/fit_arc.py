import desispec.io
import fitsio
import numpy as np
import matplotlib.pyplot as plt
import desispec.hartmann.PSFstuff as psf_tool
from astropy.io import fits,ascii
from astropy.modeling import models, fitting
import os
from astropy.table import Table, Column
from desiutil.log import get_logger

def fit_arc(file_raw,psf_file,channel,dz,line_file='../data/arc_lines/goodlines_vacuum_hartmann.ascii',ee=0.90,display=False,file_temp='file_temp.fits'):

    log = get_logger()
    
    linelist=ascii.read(line_file)
    os.system('rm '+file_temp)
    cmd='desi_preproc -i '+file_raw+' -o '+file_temp+' --camera '+channel#[0]
    log.info(cmd)
    os.system(cmd)
    
    traceset=desispec.io.read_xytraceset(psf_file)
    wmin=traceset.wavemin
    wmax=traceset.wavemax
    nspec=traceset.nspec

    # Read in image
    HDUs = fits.open(file_temp)
    hdu=HDUs[0]
    im = np.float64(hdu.data)
    sz=im.shape
    
    wavearr=list(linelist['wave'])
    wavearr=np.array(wavearr)
    ind1=np.where(wavearr > wmin)
    ind2=np.where(wavearr < wmax)
    ind1=set(ind1[0].tolist())
    ind2=set(ind2[0].tolist())
    ind=list(ind1.intersection(ind2))
    
    n_line=len(ind)
    n=30
    pix_sz = 0.015  # pixel size in mm
    FWHM_estim = abs(dz) / 2.0 / 1.7 / pix_sz + 3.
    table0 = Table(names=('defocus','xcentroid','ycentroid','fiber','lineid','wave','Ree','FWHMx','FWHMy','Amp'),dtype=('f4','f4','f4','i4','i4','f4','f4','f4','f4','f4'))
    table = table0[:]
    
    if display:
        fig = plt.figure('Data and fit profiles', figsize=(14, 11))
    for i in range(nspec):
        if i%10 == 0 : log.info("fitting fiber {}".format(i))
        if i%10 !=0 : continue # DEBUG
        
        fiber=i
        x_psf=traceset.x_vs_wave(fiber,wavearr[ind])
        y_psf=traceset.y_vs_wave(fiber,wavearr[ind])
        x = np.linspace(0, n - 1, n)  # abcissa values for plotting x profile
        y = x  # abcissa values for plotting y profile
        
        for j in range(n_line):
            x0=x_psf[j]
            y0=y_psf[j]
            xmin = int(max(x0 - n / 2, 0.0))
            xmax = xmin + n
            if xmax > sz[1]:
                xmax = sz[1]
                xmin = xmax - n
            ymin = int(max(y0 - n / 2, 0.0))
            ymax = ymin + n
            if ymax > sz[0]:
                ymax = sz[0]
                ymin = ymax - n
            subim = im[ymin:ymax, xmin:xmax]
            #print(i,j,'x,y',xmin,xmax,ymin,ymax,np.max(subim))
            if True: # keep format
                (A, xcentroid, ycentroid, FWHMx, FWHMy,chi2) = psf_tool.PSF_Params(subim, sampling_factor=10.0, display=False, \
                           estimates={'amplitude':subim.max(),'x_mean':n/2,'y_mean':n/2,'x_stddev':FWHM_estim/2.35,'y_stddev':FWHM_estim/2.35}, \
                                                   doSkySub=False)
    
#                GFitParam = {'amplitude':A, \
#                             'x_mean':xcentroid, \
#                             'y_mean':ycentroid, \
#                             'x_stddev':FWHMx/2.0/np.sqrt(2.0*np.log(2)), \
#                             'y_stddev':FWHMy/2.0/np.sqrt(2.0*np.log(2))}
#                radii = np.linspace(0.1,n/2-2,50)
#    
#                EEvect = np.array([psf_tool.EE(subim, r, GFitParam, doSkySub=False) for r in radii])
#                maxEE = np.mean(EEvect[-5:])
#                Ree = np.interp(ee*maxEE, EEvect, radii)
                Ree = 1.
                table.add_row([dz, xmin+xcentroid, ymin+ycentroid,i,ind[j],wavearr[ind[j]], Ree, FWHMx, FWHMy, A])

    return table 
    
    
