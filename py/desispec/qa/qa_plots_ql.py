"""
desispec.qa.qa_plots_ql
=======================

This includes routines to make pdf plots on the qa outputs from quicklook.

For information on QA dictionaries used here as input, visit wiki page:
https://desi.lbl.gov/trac/wiki/Pipeline/QuickLook/QuicklookQAOutputs/Science
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter

from desispec.qa import qalib
from desispec.qa.qalib import s2n_funcs
from desispec.quicklook.ql_plotlib import ql_qaplot

def plot_countspectralbins(qa_dict,outfile,plotconf=None,hardplots=False):
    """
    Plot count spectral bins.

    Args:
        qa_dict: dictionary of qa outputs from running qa_quicklook.CountSpectralBins
        outfile: Name of figure.
    """
    #SE: this has become a useless plot for showing a constant number now---> prevented creating it for now until there is an actual plan for what to plot
    camera = qa_dict["CAMERA"]
    expid=qa_dict["EXPID"]
    paname=qa_dict["PANAME"]

    thrcut=qa_dict["PARAMS"]["CUTBINS"]

    fig=plt.figure()

    if plotconf:
        hardplots=ql_qaplot(fig,plotconf,qa_dict,camera,expid,outfile)

    if not hardplots:
        pass
    else:
        plt.suptitle("Fiber level check for flux after {}, Camera: {}, ExpID: {}".format(paname,camera,expid),fontsize=10,y=0.99)
        goodfib=qa_dict["METRICS"]["GOOD_FIBERS"]
        ngoodfib=qa_dict["METRICS"]["NGOODFIB"]
        plt.plot(goodfib)
        plt.ylim(-0.1,1.1)
        plt.xlabel('Fiber #',fontsize=10)
        plt.text(-0.5,1,r"NGOODFIB=%i"%(ngoodfib),ha='left',va='top',fontsize=10,alpha=2)
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

def plot_countpix(qa_dict,outfile,plotconf=None,hardplots=False):

    """
    Plot pixel counts above some threshold

    Args:
        qa_dict: qa dictionary from countpix qa
        outfile: pdf file of the plot
    """
    from desispec.util import set_backend
    _matplotlib_backend = None
    set_backend()

    expid=qa_dict["EXPID"]
    camera = qa_dict["CAMERA"]
    paname=qa_dict["PANAME"]
    #npix_amp=np.array(qa_dict["METRICS"]["NPIX_AMP"])
    litfrac=np.array(qa_dict["METRICS"]["LITFRAC_AMP"])

    cutthres=qa_dict["PARAMS"]["CUTPIX"]

    fig=plt.figure()

    if plotconf:
        hardplots=ql_qaplot(fig,plotconf,qa_dict,camera,expid,outfile)

    if not hardplots:
        pass
    else:
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

def plot_bias_overscan(qa_dict,outfile,plotconf=None,hardplots=False):

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
    if exptime == 0.:
        exptime = 1.

    fig=plt.figure()

    if plotconf:
        hardplots=ql_qaplot(fig,plotconf,qa_dict,camera,expid,outfile)

    if not hardplots:
        pass
    else:
        title="Bias from overscan region after {}, Camera: {}, ExpID: {}".format(paname,camera,expid)
        plt.suptitle(title,fontsize=10,y=0.99)
        ax1=fig.add_subplot(111)
        ax1.set_xlabel("Avg. bias value per Amp (photon counts)",fontsize=10)
        bias_amp=qa_dict["METRICS"]["BIAS_AMP"]

        heatmap1=ax1.pcolor(bias_amp.reshape(2,2),cmap=plt.cm.OrRd)
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

def plot_XWSigma(qa_dict,outfile,plotconf=None,hardplots=False):

    """
    Plot XWSigma

    Args:
        qa_dict: qa dictionary from countpix qa
        outfile : file of the plot
    """
    camera=qa_dict["CAMERA"]
    expid=qa_dict["EXPID"]
    pa=qa_dict["PANAME"]
    xsigma=np.array(qa_dict["METRICS"]["XWSIGMA_FIB"][0])
    wsigma=np.array(qa_dict["METRICS"]["XWSIGMA_FIB"][1])
    xsigma_med=qa_dict["METRICS"]["XWSIGMA"][0]
    wsigma_med=qa_dict["METRICS"]["XWSIGMA"][1]
    xfiber=np.arange(xsigma.shape[0])
    wfiber=np.arange(wsigma.shape[0])

    fig=plt.figure()

    if plotconf:
        hardplots=ql_qaplot(fig,plotconf,qa_dict,camera,expid,outfile)

    if not hardplots:
        pass
    else:
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

def plot_RMS(qa_dict,outfile,plotconf=None,hardplots=False):
    """
    Plot RMS

    Args:
        qa_dict: dictionary of qa outputs from running qa_quicklook.Get_RMS
        outfile: Name of plot output file
    """
    camera=qa_dict["CAMERA"]
    expid=qa_dict["EXPID"]
    pa=qa_dict["PANAME"]

    fig=plt.figure()

    if plotconf:
        hardplots=ql_qaplot(fig,plotconf,qa_dict,camera,expid,outfile)

    if not hardplots:
        pass
    else:
        title="NOISE image counts per amplifier, Camera: {}, ExpID: {}".format(camera,expid)
        rms_amp=qa_dict["METRICS"]["NOISE_AMP"]
        ax1=fig.add_subplot(211)

        rms_over_amp=qa_dict["METRICS"]["NOISE_OVERSCAN_AMP"]

        plt.suptitle(title,fontsize=10,y=0.99)
        heatmap1=ax1.pcolor(rms_amp.reshape(2,2),cmap=plt.cm.OrRd)
    #    ax1.set_xlabel("NOISE per Amp (photon counts)",fontsize=10)
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

def plot_integral(qa_dict,outfile,plotconf=None,hardplots=False):
    import matplotlib.ticker as ticker
    """
    Plot integral.

    Args:
        qa_dict: qa dictionary
        outfile : output plot file
    """
    expid=qa_dict["EXPID"]
    camera=qa_dict["CAMERA"]
    paname=qa_dict["PANAME"]

    fig=plt.figure()

    if plotconf:
        hardplots=ql_qaplot(fig,plotconf,qa_dict,camera,expid,outfile)

    if not hardplots:
        pass
    else:
        ax1=fig.add_subplot(111)
        integral=np.array(qa_dict["METRICS"]["SPEC_MAGS"])
        plt.suptitle("Integrated Spectral Magnitudes, Camera: {}, ExpID: {}".format(paname,camera,expid),fontsize=10,y=0.99)
        index=np.arange(len(integral))
        hist_med=ax1.bar(index,integral,color='b',align='center')
        ax1.set_xlabel('Fibers',fontsize=10)
        ax1.set_ylabel('Integral (photon counts)',fontsize=10)
        ax1.tick_params(axis='x',labelsize=10)
        ax1.tick_params(axis='y',labelsize=10)
        ax1.xaxis.set_major_locator(ticker.AutoLocator())
        #ax1.set_xticklabels(std_fiberid)

        plt.tight_layout()
        fig.savefig(outfile)

def plot_sky_continuum(qa_dict,outfile,plotconf=None,hardplots=False):

    """
    Plot mean sky continuum from lower and higher wavelength range for each
    fiber and accross amps.

    Args:
        qa_dict: dictionary from sky continuum QA
        outfile: pdf file to save the plot
    """
    expid=qa_dict["EXPID"]
    camera=qa_dict["CAMERA"]
    paname=qa_dict["PANAME"]

    fig=plt.figure()

    if plotconf:
        hardplots=ql_qaplot(fig,plotconf,qa_dict,camera,expid,outfile)

    if not hardplots:
        pass
    else:
        title="Mean Sky Continuum after {}, Camera: {}, ExpID: {}".format(paname,camera,expid)
        xtitle="SKY fiber ID"
        ytitle="Sky Continuum (photon counts)"
        skycont_fiber=np.array(qa_dict["METRICS"]["SKYCONT_FIBER"])
        fiberid=qa_dict["METRICS"]["SKYFIBERID"]
        plt.suptitle(title,fontsize=10,y=0.99)

        ax1=fig.add_subplot(111)
        index=np.arange(len(skycont_fiber))
        hist_med=ax1.bar(index,skycont_fiber,color='b',align='center')
        ax1.set_xlabel(xtitle,fontsize=10)
        ax1.set_ylabel(ytitle,fontsize=10)
        ax1.tick_params(axis='x',labelsize=6)
        ax1.tick_params(axis='y',labelsize=10)
        ax1.set_xticks(index)
        ax1.set_xticklabels(fiberid)
        ax1.set_xlim(0)

        plt.tight_layout()
        fig.savefig(outfile)

def plot_sky_peaks(qa_dict,outfile,plotconf=None,hardplots=False):

    """
    Plot rms of sky peaks for smy fibers across amps

    Args:
        qa_dict: dictionary from sky peaks QA
        outfile: pdf file to save the plot
    """


    expid=qa_dict["EXPID"]
    camera=qa_dict["CAMERA"]
    paname=qa_dict["PANAME"]
    sumcount=qa_dict["METRICS"]["PEAKCOUNT_FIB"]
    fiber=np.arange(sumcount.shape[0])
    skyfiber_rms=qa_dict["METRICS"]["PEAKCOUNT_NOISE"]

    fig=plt.figure()

    if plotconf:
        hardplots=ql_qaplot(fig,plotconf,qa_dict,camera,expid,outfile)

    if not hardplots:
        pass
    else:
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

def plot_residuals(frame,qa_dict,outfile,plotconf=None,hardplots=False):
    import random
    """
    Plot one random sky subtracted, fiber flattened spectrum per object type

    Args:
        frame: sframe object
        qa_dict: qa dictionary
        outfile : output plot file
    """

    expid=qa_dict["EXPID"]
    camera = qa_dict["CAMERA"]
    paname=qa_dict["PANAME"]
    med_resid_fiber=qa_dict["METRICS"]["MED_RESID_FIBER"]
    med_resid_wave=qa_dict["METRICS"]["MED_RESID_WAVE"]
    wavelength=qa_dict["METRICS"]["WAVELENGTH"]
    flux=frame.flux
    objects=frame.fibermap["OBJTYPE"]
    objtypes=list(set(objects))

    fig=plt.figure()

    if plotconf:
        hardplots=ql_qaplot(fig,plotconf,qa_dict,camera,expid,outfile)

    if not hardplots:
        pass
    else:
        plt.suptitle('Randomly selected sky subtracted, fiber flattenend spectra\ncamera {}, exposure, {}'.format(camera,expid),fontsize=10)

        for i in range(len(objtypes)):
            ax=fig.add_subplot('23{}'.format(i+1))

            objs=np.where(objects==objtypes[i])[0]
            obj=random.choice(objs)
            objflux=flux[obj]

            ax.set_xlabel('Wavelength (Angstroms)',fontsize=8)
            ax.set_ylabel('{} Flux (counts)'.format(objtypes[i]),fontsize=8)
            ax.tick_params(axis='x',labelsize=8)
            ax.tick_params(axis='y',labelsize=8)
            ax.plot(wavelength,objflux)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

    #    gs=GridSpec(6,4)
    #    plt.suptitle("Sky Residuals after {}, Camera: {}, ExpID: {}".format(paname,camera,expid))
    #
    #    ax0=fig.add_subplot(gs[:2,2:])
    #    ax0.set_axis_off()
    #    keys=["MED_RESID","NBAD_PCHI","NREJ","NSKY_FIB","RESID_PER"]
    #    skyfiberid=qa_dict["METRICS"]["SKYFIBERID"]
    #
    #    xl=0.05
    #    yl=0.9
    #    for key in keys:
    #        ax0.text(xl,yl,key+': '+str(qa_dict["METRICS"][key]),transform=ax0.transAxes,ha='left',fontsize='x-small')
    #        yl=yl-0.1
    #
    #    ax1=fig.add_subplot(gs[:2,:2])
    #    ax1.plot(wavelength, med_resid_wave,'b')
    #    ax1.set_ylabel("Med. Sky Res. (photon counts)",fontsize=10)
    #    ax1.set_xlabel("Wavelength(A)",fontsize=10)
    #    ax1.set_ylim(np.percentile(med_resid_wave,2.5),np.percentile(med_resid_wave,97.5))
    #    ax1.set_xlim(np.min(wavelength),np.max(wavelength))
    #    ax1.tick_params(axis='x',labelsize=10)
    #    ax1.tick_params(axis='y',labelsize=10)
    #
    #    ax2=fig.add_subplot(gs[3:,:])
    #    index=range(med_resid_fiber.shape[0])
    #    hist_res=ax2.bar(index,med_resid_fiber,align='center')
    #    ax2.plot(index,np.zeros_like(index),'k-')
    #    #ax1.plot(index,med_resid_fiber,'bo')
    #    ax2.set_xlabel('Sky fiber ID',fontsize=10)
    #    ax2.set_ylabel('Med. Sky Res. (photon counts)',fontsize=10)
    #    ax2.tick_params(axis='x',labelsize=10)
    #    ax2.tick_params(axis='y',labelsize=10)
    #    ax2.set_xticks(index)
    #    ax2.set_xticklabels(skyfiberid)
    #    ax2.set_xlim(0)
    #    #plt.tight_layout()

        fig.savefig(outfile)


def plot_SNR(qa_dict,outfile,objlist,fitsnr,rescut=0.2,sigmacut=2.,plotconf=None,hardplots=False):
    """
    Plot SNR

    Args:
        qa_dict: dictionary of qa outputs from running qa_quicklook.Calculate_SNR
        outfile: output png file
        objlist: list of objtype for log(snr**2) vs. mag plots
        badfibs: list of fibers with infs or nans to remove for plotting
        fitsnr: list of snr vs. mag fitting coefficients # JXP -- THIS IS NOT TRUE!!
        rescut: only plot residuals (+/-) less than rescut (default 0.2)
        sigmacut: only plot residuals (+/-) less than sigma cut (default 2.0)
        NOTE: rescut taken as default cut parameter
    """
    med_snr=np.array(qa_dict["METRICS"]["MEDIAN_SNR"])
    avg_med_snr=np.mean(med_snr)
    index=np.arange(med_snr.shape[0])
    resids= np.array(qa_dict["METRICS"]["SNR_RESID"])
    camera = qa_dict["CAMERA"]
    expid=qa_dict["EXPID"]
    paname=qa_dict["PANAME"]

    fig=plt.figure()

    if plotconf:
        hardplots=ql_qaplot(fig,plotconf,qa_dict,camera,expid,outfile)

    if not hardplots:
        pass
    else:
        ra=[]
        dec=[]
        mags=[]
        snrs=[]
        # Loop over object types
        for oid, otype in enumerate(objlist):
            mag=qa_dict["METRICS"]["SNR_MAG_TGT"][oid][1]
            snr=qa_dict["METRICS"]["SNR_MAG_TGT"][oid][0]
            mags.append(mag)
            snrs.append(snr)

            fibers = qa_dict['METRICS']['%s_FIBERID'%otype]
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

        for i,otype in enumerate(objlist):
            ax=fig.add_subplot('24{}'.format(i+5))

            objtype=objlist[i]
            objid=np.where(np.array(objlist)==objtype)[0][0]
            obj_mag=mags[objid]
            obj_snr=snrs[objid]
            plot_mag=sorted(obj_mag)
            #plot_fit=np.array(fitsnr[objid])**2
            snr2=np.array(obj_snr)**2
            fitval=qa_dict["METRICS"]["FITCOEFF_TGT"][objid]

            # Calculate the model
            flux = 10 ** (-0.4 * (np.array(plot_mag) - 22.5))
            funcMap = s2n_funcs(exptime=qa_dict['METRICS']['EXPTIME'])
            fitfunc = funcMap['astro']
            plot_fit = fitfunc(flux, *fitval)

            # Plot
            if i == 0:
                ax.set_ylabel('Median S/N**2',fontsize=8)
            ax.set_xlabel('{} Mag ({})\na={:.4f}, B={:.1f}'.format(objtype,thisfilter,fitval[0],fitval[1]),fontsize=6)
            if otype == 'STAR':
                ax.set_xlim(16,20)
            elif otype == 'BGS' or otype == 'MWS':
                ax.set_xlim(14,24)
            elif otype == 'QSO':
                ax.set_xlim(17,23)
            else:
                ax.set_xlim(20,25)
            ax.tick_params(axis='x',labelsize=6)
            ax.tick_params(axis='y',labelsize=6)
            ax.semilogy(obj_mag,snr2,'b.',markersize=1)
            ax.semilogy(plot_mag,plot_fit**2,'y',linewidth=1)

        fig.savefig(outfile)

def plot_lpolyhist(qa_dict,outfile,plotconf=None,hardplots=False):
    """
    Plot histogram for each legendre polynomial coefficient in WSIGMA array.

    Args:
        qa_dict: Dictionary of qa outputs from running qa_quicklook.Check_Resolution
        outfile: Name of figure.
    """
    paname = qa_dict["PANAME"]
    p0 = qa_dict["DATA"]["LPolyCoef0"]
    p1 = qa_dict["DATA"]["LPolyCoef1"]
    p2 = qa_dict["DATA"]["LPolyCoef2"]

    fig = plt.figure()

    if plotconf:
        hardplots=ql_qaplot(fig,plotconf,qa_dict,camera,expid,outfile)

    if not hardplots:
        pass
    else:
        plt.suptitle("{} QA Legendre Polynomial Coefficient Histograms".format(paname))

        # Creating subplots
        ax1 = fig.add_subplot(311)
        n1, bins1, patches1 = ax1.hist(p0, bins=20, ec='black')
        ax1.set_xticks(bins1[::3])
        ax1.xaxis.set_major_formatter(FormatStrFormatter('%0.3f'))
        ax1.set_xlabel('Zeroth Legendre Polynomial Coefficient (p0)')
        ax1.set_ylabel('Frequency')

        ax2 = fig.add_subplot(312)
        n2, bins2, patches2 = ax2.hist(p1, bins=20, ec='black')
        ax2.set_xticks(bins2[::3])
        ax2.xaxis.set_major_formatter(FormatStrFormatter('%0.3f'))
        ax2.set_xlabel('First Legendre Polynomial Coefficient (p1)')
        ax2.set_ylabel('Frequency')

        ax3 = fig.add_subplot(313)
        n3, bins3, patches3 = ax3.hist(p2, bins=20, ec='black')
        ax3.set_xticks(bins3[::3])
        ax3.xaxis.set_major_formatter(FormatStrFormatter('%0.3f'))
        ax3.set_xlabel('Second Legendre Polynomial Coefficient (p2)')
        ax3.set_ylabel('Frequency')

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)

        fig.savefig(outfile)

