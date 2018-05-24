"""
This includes routines to make pdf plots on the qa outputs from quicklook.

For information on QA dictionaries used here as input, visit wiki page:
https://desi.lbl.gov/trac/wiki/Pipeline/QuickLook/QuicklookQAOutputs/Science
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from desispec.qa import qalib

def plot_countspectralbins(qa_dict,outfile):
    """
    Plot count spectral bins.

    Args:
        qa_dict: dictionary of qa outputs from running qa_quicklook.CountSpectralBins
        outfile: Name of figure.
    """
    camera = qa_dict["CAMERA"]
    expid=qa_dict["EXPID"]
    paname=qa_dict["PANAME"]
    
    thrcut=qa_dict["PARAMS"]["CUTBINS"]

    fig=plt.figure()
    plt.suptitle("Fiber level check for flux after {}, Camera: {}, ExpID: {}".format(paname,camera,expid),fontsize=10,y=0.99)
    goodfib=qa_dict["METRICS"]["GOOD_FIBER"]
    ngoodfib=qa_dict["METRICS"]["NGOODFIB"]
    plt.plot(goodfib)
    plt.ylim(-0.1,1.1)
    plt.xlabel('Fiber #',fontsize=10)
    plt.text(-0.5,1,r"NGOODFIB=%i"%(ngoodfib),ha='left',
 va='top',fontsize=10,alpha=2)
    """
    gs=GridSpec(7,6)
    ax1=fig.add_subplot(gs[:,:2])
    ax2=fig.add_subplot(gs[:,2:4])
    ax3=fig.add_subplot(gs[:,4:])

    hist_med=ax1.bar(index,binslo,color='b',align='center')
    ax1.set_xlabel('Fiber #',fontsize=10)
    ax1.set_ylabel('Photon Counts > {:d}'.format(cutlo),fontsize=10)
    ax1.tick_params(axis='x',labelsize=10)
    ax1.tick_params(axis='y',labelsize=10)
    ax1.set_xlim(0)

    hist_med=ax2.bar(index,binsmed,color='r',align='center')
    ax2.set_xlabel('Fiber #',fontsize=10)
    ax2.set_ylabel('Photon Counts > {:d}'.format(cutmed),fontsize=10)
    ax2.tick_params(axis='x',labelsize=10)
    ax2.tick_params(axis='y',labelsize=10)
    ax2.set_xlim(0)

    hist_med=ax3.bar(index,binshi,color='g',align='center')
    ax3.set_xlabel('Fiber #',fontsize=10)
    ax3.set_ylabel('Photon Counts > {:d}'.format(cuthi),fontsize=10)
    ax3.tick_params(axis='x',labelsize=10)
    ax3.tick_params(axis='y',labelsize=10)
    ax3.set_xlim(0)
    """
    plt.tight_layout()
    fig.savefig(outfile)

def plot_countpix(qa_dict,outfile):
    """
    Plot pixel counts above some threshold
    
    Args:
        qa_dict: qa dictionary from countpix qa
        outfile: pdf file of the plot
    """
    expid=qa_dict["EXPID"]
    camera = qa_dict["CAMERA"]
    paname=qa_dict["PANAME"]
    #npix_amp=np.array(qa_dict["METRICS"]["NPIX_AMP"])
    litfrac=np.array(qa_dict["METRICS"]["LITFRAC_AMP"])

    cutthres=qa_dict["PARAMS"]["CUTPIX"]

    fig=plt.figure()
    plt.suptitle("Fraction of pixels lit after {}, Camera: {}, ExpID: {}".format(paname,camera,expid),fontsize=10,y=0.99)
    #ax1=fig.add_subplot(211)
    #heatmap1=ax1.pcolor(npix_amp.reshape(2,2),cmap=plt.cm.OrRd)
    ##plt.title('Total Pixels > {:d} sigma = {:f}'.format(cutthres,countlo), fontsize=10)
    #ax1.set_xlabel("# pixels > {:d} sigma (per Amp)".format(cutthres),fontsize=10)
    #ax1.tick_params(axis='x',labelsize=10,labelbottom=False)
    #ax1.tick_params(axis='y',labelsize=10,labelleft=False)
    # ax1.annotate("Amp 1\n{:f}".format(npix_amp[0]),
    #             xy=(0.4,0.4),
    #             fontsize=10
    #             )
    #ax1.annotate("Amp 2\n{:f}".format(npix_amp[1]),
    #             xy=(1.4,0.4),
    #             fontsize=10
    #             )
    #ax1.annotate("Amp 3\n{:f}".format(npix_amp[2]),
    #             xy=(0.4,1.4),
    #             fontsize=10
    #             )
    #ax1.annotate("Amp 4\n{:f}".format(npix_amp[3]),
    #             xy=(1.4,1.4),
    #             fontsize=10
    #             )
    ax2=fig.add_subplot(111)
    heatmap2=ax2.pcolor(litfrac.reshape(2,2),cmap=plt.cm.OrRd)
    ax2.set_xlabel("Fraction over {:d} sigma read noise(per Amp)".format(cutthres),fontsize=10)
    ax2.tick_params(axis='x',labelsize=10,labelbottom=False)
    ax2.tick_params(axis='y',labelsize=10,labelleft=False)
    ax2.annotate("Amp 1\n{:f}".format(litfrac[0]),
                 xy=(0.4,0.4),
                 fontsize=10
                 )
    ax2.annotate("Amp 2\n{:f}".format(litfrac[1]),
                 xy=(1.4,0.4),
                 fontsize=10
                 )
    ax2.annotate("Amp 3\n{:f}".format(litfrac[2]),
                 xy=(0.4,1.4),
                 fontsize=10
                 )
    ax2.annotate("Amp 4\n{:f}".format(litfrac[3]),
                 xy=(1.4,1.4),
                 fontsize=10
                 )
    plt.tight_layout()
    fig.savefig(outfile)

def plot_bias_overscan(qa_dict,outfile):
    """
    Map of bias from overscan from 4 regions of CCD
    
    Args:
        qa_dict: qa dictionary from bias_from_overscan qa
        outfile : pdf file of the plot
    """
    expid = qa_dict["EXPID"]
    camera = qa_dict["CAMERA"]
    paname = qa_dict["PANAME"]
    params = qa_dict["PARAMS"]
    exptime = qa_dict["EXPTIME"]
    
    bias=qa_dict["METRICS"]["BIAS"]
    bias_amp=qa_dict["METRICS"]["BIAS_AMP"]
    fig=plt.figure()
    plt.suptitle("Bias from overscan region after {}, Camera: {}, ExpID: {}".format(paname,camera,expid),fontsize=10,y=0.99)
    ax1=fig.add_subplot(111)
    heatmap1=ax1.pcolor(bias_amp.reshape(2,2),cmap=plt.cm.OrRd)
    plt.title('Bias = {:.4f}'.format(bias/exptime), fontsize=10)
    ax1.set_xlabel("Avg. bias value per Amp (photon counts)",fontsize=10)
    ax1.tick_params(axis='x',labelsize=10,labelbottom=False)
    ax1.tick_params(axis='y',labelsize=10,labelleft=False)
    ax1.annotate("Amp 1\n{:.3f}".format(bias_amp[0]/exptime),
                 xy=(0.4,0.4),
                 fontsize=10
                 )
    ax1.annotate("Amp 2\n{:.3f}".format(bias_amp[1]/exptime),
                 xy=(1.4,0.4),
                 fontsize=10
                 )
    ax1.annotate("Amp 3\n{:.3f}".format(bias_amp[2]/exptime),
                 xy=(0.4,1.4),
                 fontsize=10
                 )
    ax1.annotate("Amp 4\n{:.3f}".format(bias_amp[3]/exptime),
                 xy=(1.4,1.4),
                 fontsize=10
                 )
    fig.savefig(outfile)

def plot_XWSigma(qa_dict,outfile):
    """
    Plot XWSigma
    
    Args:
        qa_dict: qa dictionary from countpix qa
        outfile : file of the plot
    """
    camera=qa_dict["CAMERA"]
    expid=qa_dict["EXPID"]
    pa=qa_dict["PANAME"]
    xsigma=qa_dict["METRICS"]["XWSIGMA_FIB"][0]
    wsigma=qa_dict["METRICS"]["XWSIGMA_FIB"][1]
    xsigma_med=qa_dict["METRICS"]["XWSIGMA"][0]
    wsigma_med=qa_dict["METRICS"]["XWSIGMA"][1]
    xfiber=np.arange(xsigma.shape[0])
    wfiber=np.arange(wsigma.shape[0])

    fig=plt.figure()
    plt.suptitle("X & W Sigma over sky peaks, Camera: {}, ExpID: {}".format(camera,expid),fontsize=10,y=0.99)

    ax1=fig.add_subplot(221)
    hist_x=ax1.bar(xfiber,xsigma,align='center')
    ax1.set_xlabel("Fiber #",fontsize=10)
    ax1.set_ylabel("X std. dev. (# of pixels)",fontsize=10)
    ax1.tick_params(axis='x',labelsize=10)
    ax1.tick_params(axis='y',labelsize=10)
    plt.xlim(0,len(xfiber))

    ax2=fig.add_subplot(222)
    hist_w=ax2.bar(wfiber,wsigma,align='center')
    ax2.set_xlabel("Fiber #",fontsize=10)
    ax2.set_ylabel("W std. dev. (# of pixels)",fontsize=10)
    ax2.tick_params(axis='x',labelsize=10)
    ax2.tick_params(axis='y',labelsize=10)
    plt.xlim(0,len(wfiber))

    if "XWSIGMA_AMP" in qa_dict["METRICS"]:
        xsigma_amp=qa_dict["METRICS"]["XWSIGMA_AMP"][0]
        wsigma_amp=qa_dict["METRICS"]["XWSIGMA_AMP"][1]
        ax3=fig.add_subplot(223)
        heatmap3=ax3.pcolor(xsigma_amp.reshape(2,2),cmap=plt.cm.OrRd)
        plt.title('X Sigma = {:.4f}'.format(xsigma_med), fontsize=10)
        ax3.set_xlabel("X std. dev. per Amp (# of pixels)",fontsize=10)
        ax3.tick_params(axis='x',labelsize=10,labelbottom=False)
        ax3.tick_params(axis='y',labelsize=10,labelleft=False)
        ax3.annotate("Amp 1\n{:.3f}".format(xsigma_amp[0]),
                 xy=(0.4,0.4),
                 fontsize=10
                 )
        ax3.annotate("Amp 2\n{:.3f}".format(xsigma_amp[1]),
                 xy=(1.4,0.4),
                 fontsize=10
                 )
        ax3.annotate("Amp 3\n{:.3f}".format(xsigma_amp[2]),
                 xy=(0.4,1.4),
                 fontsize=10
                 )
        ax3.annotate("Amp 4\n{:.3f}".format(xsigma_amp[3]),
                 xy=(1.4,1.4),
                 fontsize=10
                 )

        ax4=fig.add_subplot(224)
        heatmap4=ax4.pcolor(wsigma_amp.reshape(2,2),cmap=plt.cm.OrRd)
        plt.title('W Sigma = {:.4f}'.format(wsigma_med), fontsize=10)
        ax4.set_xlabel("W std. dev. per Amp (# of pixels)",fontsize=10)
        ax4.tick_params(axis='x',labelsize=10,labelbottom=False)
        ax4.tick_params(axis='y',labelsize=10,labelleft=False)
        ax4.annotate("Amp 1\n{:.3f}".format(wsigma_amp[0]),
                 xy=(0.4,0.4),
                 fontsize=10
                 )
        ax4.annotate("Amp 2\n{:.3f}".format(wsigma_amp[1]),
                 xy=(1.4,0.4),
                 fontsize=10
                 )
        ax4.annotate("Amp 3\n{:.3f}".format(wsigma_amp[2]),
                 xy=(0.4,1.4),
                 fontsize=10
                 )
        ax4.annotate("Amp 4\n{:.3f}".format(wsigma_amp[3]),
                 xy=(1.4,1.4),
                 fontsize=10
                 )

    plt.tight_layout()
    fig.savefig(outfile)
    
def plot_RMS(qa_dict,outfile):
    """
    Plot RMS
    
    Args:
        qa_dict: dictionary of qa outputs from running qa_quicklook.Get_RMS
        outfile: Name of plot output file
    """
    rms=qa_dict["METRICS"]["NOISE"]
    rms_amp=qa_dict["METRICS"]["NOISE_AMP"]
    #rms_over=qa_dict["METRICS"]["NOISE_OVER"]
    rms_over_amp=qa_dict["METRICS"]["NOISE_AMP"]
    # arm=qa_dict["ARM"]
    # spectrograph=qa_dict["SPECTROGRAPH"]
    camera = qa_dict["CAMERA"]

    expid=qa_dict["EXPID"]
    pa=qa_dict["PANAME"]

    fig=plt.figure()
    plt.suptitle("NOISE image counts per amplifier, Camera: {}, ExpID: {}".format(camera,expid),fontsize=10,y=0.99)
    ax1=fig.add_subplot(211)
    heatmap1=ax1.pcolor(rms_amp.reshape(2,2),cmap=plt.cm.OrRd)
    plt.title('NOISE = {:.4f}'.format(rms), fontsize=10)
    ax1.set_xlabel("NOISE per Amp (photon counts)",fontsize=10)
    ax1.tick_params(axis='x',labelsize=10,labelbottom=False)
    ax1.tick_params(axis='y',labelsize=10,labelleft=False)
    ax1.annotate("Amp 1\n{:.3f}".format(rms_amp[0]),
                 xy=(0.4,0.4),
                 fontsize=10
                 )
    ax1.annotate("Amp 2\n{:.3f}".format(rms_amp[1]),
                 xy=(1.4,0.4),
                 fontsize=10
                 )
    ax1.annotate("Amp 3\n{:.3f}".format(rms_amp[2]),
                 xy=(0.4,1.4),
                 fontsize=10
                 )
    ax1.annotate("Amp 4\n{:.3f}".format(rms_amp[3]),
                 xy=(1.4,1.4),
                 fontsize=10
                 )
    ax2=fig.add_subplot(212)
    heatmap2=ax2.pcolor(rms_over_amp.reshape(2,2),cmap=plt.cm.OrRd)
    #plt.title('NOISE Overscan = {:.4f}'.format(rms_over), fontsize=10)
    ax2.set_xlabel("NOISE Overscan per Amp (photon counts)",fontsize=10)
    ax2.tick_params(axis='x',labelsize=10,labelbottom=False)
    ax2.tick_params(axis='y',labelsize=10,labelleft=False)
    ax2.annotate("Amp 1\n{:.3f}".format(rms_over_amp[0]),
                 xy=(0.4,0.4),
                 fontsize=10
                 )
    ax2.annotate("Amp 2\n{:.3f}".format(rms_over_amp[1]),
                 xy=(1.4,0.4),
                 fontsize=10
                 )
    ax2.annotate("Amp 3\n{:.3f}".format(rms_over_amp[2]),
                 xy=(0.4,1.4),
                 fontsize=10
                 )
    ax2.annotate("Amp 4\n{:.3f}".format(rms_over_amp[3]),
                 xy=(1.4,1.4),
                 fontsize=10
                 )
    fig.savefig(outfile)

def plot_integral(qa_dict,outfile):
    import matplotlib.ticker as ticker
    """
    Plot integral.

    Args:
        qa_dict: qa dictionary
        outfile : output plot file
    """
    expid=qa_dict["EXPID"]
    camera =qa_dict["CAMERA"]
    paname=qa_dict["PANAME"]
    integral=np.array(qa_dict["METRICS"]["FIBER_MAG"])
    std_fiberid=qa_dict["METRICS"]["STD_FIBERID"]

    fig=plt.figure()
    plt.suptitle("Total integrals of STD spectra {}, Camera: {}, ExpID: {}".format(paname,camera,expid),fontsize=10,y=0.99)
    index=np.arange(len(integral))
    ax1=fig.add_subplot(111)
    hist_med=ax1.bar(index,integral,color='b',align='center')
    ax1.set_xlabel('Fibers',fontsize=10)
    ax1.set_ylabel('Integral (photon counts)',fontsize=10)
    ax1.tick_params(axis='x',labelsize=10)
    ax1.tick_params(axis='y',labelsize=10)
    ax1.xaxis.set_major_locator(ticker.AutoLocator())
    #ax1.set_xticklabels(std_fiberid)
    
    plt.tight_layout()
    fig.savefig(outfile)

def plot_sky_continuum(qa_dict,outfile):
    """
    Plot mean sky continuum from lower and higher wavelength range for each 
    fiber and accross amps.
    
    Args:
        qa_dict: dictionary from sky continuum QA
        outfile: pdf file to save the plot
    """
    expid=qa_dict["EXPID"]
    camera = qa_dict["CAMERA"]
    paname=qa_dict["PANAME"]
    skycont_fiber=np.array(qa_dict["METRICS"]["SKYCONT_FIBER"])
    skycont=qa_dict["METRICS"]["SKYCONT"]
    index=np.arange(skycont_fiber.shape[0])
    fiberid=qa_dict["METRICS"]["SKYFIBERID"]
    fig=plt.figure()
    plt.suptitle("Mean Sky Continuum after {}, Camera: {}, ExpID: {}".format(paname,camera,expid),fontsize=10,y=0.99)
    
    ax1=fig.add_subplot(111)
    hist_med=ax1.bar(index,skycont_fiber,color='b',align='center')
    ax1.set_xlabel('SKY fiber ID',fontsize=10)
    ax1.set_ylabel('Sky Continuum (photon counts)',fontsize=10)
    ax1.tick_params(axis='x',labelsize=6)
    ax1.tick_params(axis='y',labelsize=10)
    ax1.set_xticks(index)
    ax1.set_xticklabels(fiberid)
    ax1.set_xlim(0)

    plt.tight_layout()
    fig.savefig(outfile)

def plot_sky_peaks(qa_dict,outfile):
    """
    Plot rms of sky peaks for smy fibers across amps
       
    Args:
        qa_dict: dictionary from sky peaks QA
        outfile: pdf file to save the plot
    """
    expid=qa_dict["EXPID"]
    camera=qa_dict["CAMERA"]
    paname=qa_dict["PANAME"]
    sumcount=qa_dict["METRICS"]["PEAKCOUNT"]
    fiber=np.arange(sumcount.shape[0])
    skyfiber_rms=qa_dict["METRICS"]["PEAKCOUNT_NOISE"]
    fig=plt.figure()
    plt.suptitle("Counts for Sky Fibers after {}, Camera: {}, ExpID: {}".format(paname,camera,expid),fontsize=10,y=0.99)

    ax1=fig.add_subplot(111)
    hist_x=ax1.bar(fiber,sumcount,align='center')
    ax1.set_xlabel("Fiber #",fontsize=10)
    ax1.set_ylabel("Summed counts over sky peaks (photon counts)",fontsize=10)
    ax1.tick_params(axis='x',labelsize=10)
    ax1.tick_params(axis='y',labelsize=10)
    plt.xlim(0,len(fiber))

    plt.tight_layout()
    fig.savefig(outfile)

def plot_residuals(qa_dict,outfile):
    """
    Plot histogram of sky residuals for each sky fiber
    
    Args:
        qa_dict: qa dictionary
        outfile : output plot file
    """
    expid=qa_dict["EXPID"]
    camera = qa_dict["CAMERA"]
    paname=qa_dict["PANAME"]
    med_resid_fiber=qa_dict["METRICS"]["MED_RESID_FIBER"]
    med_resid_wave=qa_dict["METRICS"]["MED_RESID_WAVE"]
    wavelength=qa_dict["METRICS"]["WAVELENGTH"]

    fig=plt.figure()

    gs=GridSpec(6,4)
    plt.suptitle("Sky Residuals after {}, Camera: {}, ExpID: {}".format(paname,camera,expid))
    
    ax0=fig.add_subplot(gs[:2,2:])
    ax0.set_axis_off()
    keys=["MED_RESID","NBAD_PCHI","NREJ","NSKY_FIB","RESID_PER"]
    skyfiberid=qa_dict["METRICS"]["SKYFIBERID"]
    
    xl=0.05
    yl=0.9
    for key in keys:
        ax0.text(xl,yl,key+': '+str(qa_dict["METRICS"][key]),transform=ax0.transAxes,ha='left',fontsize='x-small')
        yl=yl-0.1

    ax1=fig.add_subplot(gs[:2,:2])
    ax1.plot(wavelength, med_resid_wave,'b')
    ax1.set_ylabel("Med. Sky Res. (photon counts)",fontsize=10)
    ax1.set_xlabel("Wavelength(A)",fontsize=10)
    ax1.set_ylim(np.percentile(med_resid_wave,2.5),np.percentile(med_resid_wave,97.5))
    ax1.set_xlim(np.min(wavelength),np.max(wavelength))
    ax1.tick_params(axis='x',labelsize=10)
    ax1.tick_params(axis='y',labelsize=10)

    ax2=fig.add_subplot(gs[3:,:])
    index=range(med_resid_fiber.shape[0])
    hist_res=ax2.bar(index,med_resid_fiber,align='center')
    ax2.plot(index,np.zeros_like(index),'k-')
    #ax1.plot(index,med_resid_fiber,'bo')
    ax2.set_xlabel('Sky fiber ID',fontsize=10)
    ax2.set_ylabel('Med. Sky Res. (photon counts)',fontsize=10)
    ax2.tick_params(axis='x',labelsize=10)
    ax2.tick_params(axis='y',labelsize=10)
    ax2.set_xticks(index)
    ax2.set_xticklabels(skyfiberid)
    ax2.set_xlim(0)
    #plt.tight_layout()
    fig.savefig(outfile)
    
def plot_SNR(qa_dict,outfile,objlist,badfibs,fitsnr,rescut,sigmacut):
    """
    Plot SNR

    Args:
        qa_dict: dictionary of qa outputs from running qa_quicklook.Calculate_SNR
        outfile: Name of figure.
    """
    med_snr=qa_dict["METRICS"]["MEDIAN_SNR"]
    avg_med_snr=np.mean(med_snr)
    index=np.arange(med_snr.shape[0])
    resids=qa_dict["METRICS"]["SNR_RESID"]
    camera = qa_dict["CAMERA"]
    expid=qa_dict["EXPID"]
    paname=qa_dict["PANAME"]

    ra=[]
    dec=[]
    mags=[]
    snrs=[]
    o=np.arange(len(objlist))
    for t in range(len(o)):
        otype=list(objlist)[t]
        oid=np.where(np.array(list(objlist))==otype)[0][0]
        mag=qa_dict["METRICS"]["SNR_MAG_TGT"][oid][1]
        snr=qa_dict["METRICS"]["SNR_MAG_TGT"][oid][0]
        if otype == 'STD':
            fibers = qa_dict['METRICS']['STAR_FIBERID']
        else:
            fibers = qa_dict['METRICS']['%s_FIBERID'%otype]
        #- Remove invalid values for plotting
        badobj = badfibs[oid]
        if len(badobj) > 0:
            fibers = np.array(fibers)
            badfibs = np.array(badfibs)
            remove = []
            for ff in range(len(badobj)):
                rm = np.where(fibers==badobj[ff])[0]
                if len(rm) == 1:
                    remove.append(rm[0])
            badfibs=list(badfibs)
            fibers=list(fibers)
            for rr in range(len(remove)):
                fibers.remove(fibers[remove[rr]])
                mag.remove(mag[remove[rr]])
                snr.remove(snr[remove[rr]])
                for ri in range(len(remove)):
                     remove[ri]-=1
        mags.append(mag)
        snrs.append(snr)
        for c in range(len(fibers)):
            ras = qa_dict['METRICS']['RA'][fibers[c]]
            decs = qa_dict['METRICS']['DEC'][fibers[c]]
            ra.append(ras)
            dec.append(decs)

    if rescut is None and sigmacut is not None:
        range_min = np.mean(resids) - sigmacut * np.std(resids)
        range_max = np.mean(resids) + sigmacut * np.std(resids)
        for ii in range(len(resids)):
            if resids[ii] <= range_min:
                resids[ii] = range_min
            elif resids[ii] >= range_max:
                resids[ii] = range_max

    if camera[0] == 'b':
        thisfilter='DECAM_G'
    elif camera[0] == 'r':
        thisfilter='DECAM_R'
    else:
        thisfilter='DECAM_Z'

    fig=plt.figure()
    plt.suptitle("Signal/Noise after {}, Camera: {}, ExpID: {}".format(paname,camera,expid),fontsize=10,y=0.99)

    rmneg=med_snr[med_snr>=0.]
    rmind=index[med_snr>=0.]

    ax1=fig.add_subplot(221)
    hist_med=ax1.semilogy(rmind,rmneg,linewidth=1)
    ax1.set_xlabel('Fiber #',fontsize=6)
    ax1.set_ylabel('Median S/N',fontsize=8)
    ax1.tick_params(axis='x',labelsize=6)
    ax1.tick_params(axis='y',labelsize=6)
    ax1.set_xlim(0)

    ax2=fig.add_subplot(222)
    ax2.set_title('Residual SNR: (calculated SNR - fit SNR) / fit SNR',fontsize=8)
    ax2.set_xlabel('RA',fontsize=6)
    ax2.set_ylabel('DEC',fontsize=6)
    ax2.tick_params(axis='x',labelsize=6)
    ax2.tick_params(axis='y',labelsize=6)
    if rescut is not None:
        resid_plot=ax2.scatter(ra,dec,s=2,c=resids,cmap=plt.cm.bwr,vmin=-rescut,vmax=rescut)
        fig.colorbar(resid_plot,ticks=[-rescut,0.,rescut])
    else:
        resid_plot=ax2.scatter(ra,dec,s=2,c=resids,cmap=plt.cm.bwr)
        fig.colorbar(resid_plot,ticks=[np.min(resids),0,np.max(resids)])

    for i in range(len(o)):
        if i == 0:
            ax=fig.add_subplot(245)
        elif i == 1:
            ax=fig.add_subplot(246)
        elif i == 2:
            ax=fig.add_subplot(247)
        else:
            ax=fig.add_subplot(248)

        objtype=list(objlist)[i]
        objid=np.where(np.array(list(objlist))==objtype)[0][0]
        obj_mag=mags[objid]
        obj_snr=snrs[objid]
        plot_mag=sorted(obj_mag)
        plot_fit=np.array(fitsnr[objid])**2
        snr2=np.array(obj_snr)**2
        fitval=qa_dict["METRICS"]["FITCOEFF_TGT"][objid]

        if i == 0:
            ax.set_ylabel('Median S/N**2',fontsize=8)
        ax.set_xlabel('{} Mag ({})\na={:4f}, B={:4f}'.format(objtype,thisfilter,fitval[0],fitval[1]),fontsize=6)
        ax.set_xlim(16,24)
        ax.tick_params(axis='x',labelsize=6)
        ax.tick_params(axis='y',labelsize=6)
        ax.semilogy(obj_mag,snr2,'b.',markersize=1)
        ax.semilogy(plot_mag,plot_fit,'y',markersize=0.5)
    
    fig.savefig(outfile)
