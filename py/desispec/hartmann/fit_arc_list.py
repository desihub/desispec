import desispec.io
import numpy as np
import matplotlib.pyplot as plt
from specter.psf.gausshermite  import GaussHermitePSF
import PSFstuff as psf_tool
from astropy.io import fits,ascii
from astropy.modeling import models, fitting
import pdb
import os
from astropy.table import Table, Column

def fit_arc_list(file_raw,psf_file,channel,dz,thres=500,ee=0.90,display=False):
    file_temp='file_temp.fits'
    linelist=ascii.read('goodlines_vaccum.ascii')
    cmd='desi_preproc -i '+file_raw+' -o '+file_temp+' --camera '+channel
    print(cmd)
    os.system(cmd)
    
    traceset=desispec.io.read_xytraceset(psf_file)
    psf=GaussHermitePSF(psf_file)
    wmin=psf.wmin
    wmax=psf.wmax
    # Read in image
    HDUs = fits.open(file_temp)
    hdu=HDUs[0]
    im = np.float64(hdu.data)
    sz=im.shape
    
    wavearr=list(linelist['wave'])
    #wavearr=[3610.5080,3650.1570,3654.8390,3983.8390,4046.5720,4077.8380,4306.6720,4339.2200,4347.5060,4358.3370,4412.9890,4662.3520,4678.1490,4799.9120,4916.0700, 
    #         5085.8220,5330.7775,5341.0938,5400.5616,5460.7530,5656.6588,5675.8600,5689.8163,5719.5300,5769.5980,5790.6630,5820.1558,5852.4880,5881.8950,5944.8340, 
    #         5965.4710,6029.9970,6074.3380,6096.1630,6143.0630,6163.5940,6217.2810,6266.4950,6304.7890,6334.4280,6382.9920,6402.2460,6438.4700,6506.5280,6532.8820, 
    #         6598.9530,6717.0430,6929.4670,7032.4128,7173.9380,7245.1670, 
    #         7438.9000,7488.8710,7503.8691,7514.6518,7535.7740,7635.1060,8014.7860,8103.6930,       8300.3260,       8377.6065,       8408.2096, 
    #         8495.3598,8521.4422,8591.2583,8634.6470,8654.3831,8853.8670,8919.5007,9122.9674,       9201.7590,       9300.8530,       9313.1386, 
    #         9326.5070,9425.3790,9459.2100,9486.6800,9534.1630,9547.4000,9657.7860,9665.4240,       9784.5030,       10139.750]  SDSS arc lines
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
    for i in range(psf.nspec):
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
            print('x,y',xmin,xmax,ymin,ymax,np.max(subim))
            if np.max(subim)>thres:
                (A, xcentroid, ycentroid, FWHMx, FWHMy,chi2) = psf_tool.PSF_Params(subim, sampling_factor=10.0, display=False, \
                           estimates={'amplitude':subim.max(),'x_mean':n/2,'y_mean':n/2,'x_stddev':FWHM_estim/2.35,'y_stddev':FWHM_estim/2.35}, \
                                                   doSkySub=False)
    
                GFitParam = {'amplitude':A, \
                             'x_mean':xcentroid, \
                             'y_mean':ycentroid, \
                             'x_stddev':FWHMx/2.0/np.sqrt(2.0*np.log(2)), \
                             'y_stddev':FWHMy/2.0/np.sqrt(2.0*np.log(2))}
                radii = np.linspace(0.1,n/2-2,50)
    
                EEvect = np.array([psf_tool.EE(subim, r, GFitParam, doSkySub=False) for r in radii])
                maxEE = np.mean(EEvect[-5:])
                Ree = np.interp(ee*maxEE, EEvect, radii)
                table.add_row([dz, xmin+xcentroid, ymin+ycentroid,i,ind[j],wavearr[ind[j]], Ree, FWHMx, FWHMy, A])
    
            #DISPLAY THE FITTED PROFILES AND ENCIRCLED ENERGY
            #------------------------------------------------
            if display and np.max(subim)>thres:
                fig.clear()
                fit = models.Gaussian2D(**GFitParam)
                # higher resolution grid for plotting
                col = int(round(xcentroid))
                row = int(round(ycentroid))
                HRfactor = 8.0
                XX_highR, YY_highR = np.meshgrid(np.linspace(0, n - 1, (n - 1) * HRfactor + 1),
                                     np.linspace(0, n - 1, (n - 1) * HRfactor + 1))
                x_highR = np.linspace(0,n-1,(n - 1) * HRfactor + 1)
                y_highR = np.linspace(0,n-1,(n - 1) * HRfactor + 1)
    
                z = fit(XX_highR, YY_highR)  # the fit at higher resolution, for plotting
                col_highR = int(round(xcentroid * HRfactor))
                row_highR = int(round(ycentroid * HRfactor))
    
                ax1 = fig.add_axes([0.05,0.5,0.4,0.42], title = 'X profile', xlabel='pixels',ylabel='ADU')
                ax2 = fig.add_axes([0.55,0.5,0.4,0.42], title = 'Y profile', xlabel='pixels')
                ax3 = fig.add_axes([0.41,0.62,0.18,0.18], xticks=[], yticks=[], aspect='equal')
                ax4 = fig.add_axes([0.15,0.05,0.7,0.37], title='encircled energy', xlabel='pixels',ylabel='fraction')
                fig.suptitle('Defocus {:.3f}, source {:d}\nAnalysis type: Gaussian fit'.format(dz, j + 1))
    
                ax4.plot(radii, EEvect)
                ax4.plot([0,Ree],[ee,ee], '--', color='black')
                ax4.plot([Ree,Ree],[0,ee], '--', color='black')
                ax4.annotate('R{:02d}: {:.1f} pixels'.format(int(ee*100),Ree), (0.5,0.05), xycoords="axes fraction")
                try: 
                    profile1 = subim[row, :]
                    profile2 = z[row_highR, :]
    
                    ax1.plot(x, profile1, '--', linewidth=2, label='data')
                    ax1.plot(XX_highR[0, :], profile2, linewidth=2, label='fit')
                    m = A
                    xstart = xcentroid - FWHMx / 2.0  # exact abscissa position of the left point at half max (analytical function)
    
                    ax1.plot([xstart, xstart + FWHMx], [m / 2, m / 2], '-x', color='black', \
                                   label='FWHM')  # plot line at FWHM
                    ax1.annotate('FWHMx: {:.1f} pixels\nCHI2: {:.1f} ADU'.format(FWHMx, np.nan), \
                                (2, subim.max()), va='top', color='green')
                    ax1.set_ylim(-50, 1.05*subim.max())
                    ax1.set_xlim(0, n)
                    ax1.set_title('X profile')
                    ax1.set_xlabel('pixels')
                    ax1.set_ylabel('ADU')
                except:
                    pass

                try:    
                    profile1 = subim[:, col]
                    profile2 = z[:, col_highR]
                    ax2.plot(y, profile1, '--', linewidth=2)
                    ax2.plot(YY_highR[:, 0], profile2, linewidth=2, label='fit')
                    m = A
                    xstart = ycentroid - FWHMy / 2.0  # exact abscissa position of the left point at half max (analytical function)
                    ax2.plot([xstart, xstart + FWHMy], [m / 2, m / 2], '-x', color='black')
                    ax2.annotate('FWHMy: {:.1f} pixels\nCHI2: {:.1f} ADU'.format(FWHMy, np.nan), \
                                 (1, subim.max()), va='top', color='green')
                    ax2.set_ylim(-50, 1.05*subim.max())
                    ax2.set_xlim(0, n)
                    ax2.set_title('Y profile')
                    ax2.set_xlabel('pixels')
                except:
                    pass
    
                ax3.imshow(np.arcsinh(subim - subim.min() + 0.1), aspect='equal')
                ax3.annotate('+', (xcentroid, ycentroid), color='blue', ha='center', va='center', fontsize='xx-large',
                                                                    fontweight='light')
    
                ax1.legend(fancybox=True, shadow=True, ncol=1, fontsize=14.0, loc='best')
                plt.pause(0.00001)  # necessary to update the figure. There is a delay already...
    
    return table 
    
    
