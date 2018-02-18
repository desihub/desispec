"""
This includes routines to make pdf plots on the qa outputs from quicklook.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from desispec.qa import qalib

def plot_countspectralbins(qa_dict,outfile):
    """
    Plot count spectral bins.

    While reading from yaml output file, qa_dict is the value to the first top level key, 
    which is the name of that QA

    qa_dict example::

        {'CAMERA': 'r0',
         'EXPID': '00000006',
         'QATIME': '2016-08-02T14:40:03.269684',
         'PANAME': 'BOXCAR',
         'PARAMS': {'CUTLO', 100, 'CUTMED', 250, 'CUTHI', 500},
         'METRICS': {'NBINSLOW': array([ 2575.,  2611.,  2451.,  2495.,  2357.,  2452.,  
                    2528.,  2501.,  2548.,  2461.]),
                   'NBINSLOW_AMP': array([ 1249.74,     0.  ,  1198.01,     0.  ]),
                   'NBINSMED': array([ 2503.,  2539.,  2161.,  2259.,  2077.,  2163.,  2284.,  2268.,  2387.,  2210.]),
                   'NBINSMED_AMP': array([ 1149.55,     0.  ,  1095.02,     0.  ]),
                   'NBINSHIGH': array([ 2307.,  2448.,   229.,  1910.,    94.,   306.,  2056.,  1941.,  2164.,   785.]),
                   'NBINSHIGH_AMP': array([ 688.85,    0.  ,  648.75,    0.  ])
                   'NGOODFIB: 10}}}

    Args:
        qa_dict: dictionary of qa outputs from running qa_quicklook.CountSpectralBins
        outfile: Name of figure.
    """
    camera = qa_dict["CAMERA"]
    expid=qa_dict["EXPID"]
    paname=qa_dict["PANAME"]
    
    binslo=qa_dict["METRICS"]["NBINSLOW"]
    binsmed=qa_dict["METRICS"]["NBINSMED"]
    binshi=qa_dict["METRICS"]["NBINSHIGH"]

    cutlo=qa_dict["PARAMS"]["CUTLO"]
    cuthi=qa_dict["PARAMS"]["CUTHI"]
    cutmed=qa_dict["PARAMS"]["CUTMED"]

    index=np.arange(binslo.shape[0])

    fig=plt.figure()
    plt.suptitle("Count spectral bins after {}, Camera: {}, ExpID: {}".format(paname,camera,expid),fontsize=10,y=0.99)

    gs=GridSpec(7,6)
    ax1=fig.add_subplot(gs[1:4,:2])
    ax2=fig.add_subplot(gs[1:4,2:4])
    ax3=fig.add_subplot(gs[1:4,4:])
    ax4=fig.add_subplot(gs[4:,:2])
    ax5=fig.add_subplot(gs[4:,2:4])
    ax6=fig.add_subplot(gs[4:,4:])

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

    #- plotting if amp is turned on
    if "NBINSLOW_AMP" in qa_dict["METRICS"]:

        binslo_amp=qa_dict["METRICS"]["NBINSLOW_AMP"]
        binsmed_amp=qa_dict["METRICS"]["NBINSMED_AMP"]
        binshi_amp=qa_dict["METRICS"]["NBINSHIGH_AMP"]
    
        heatmap1=ax4.pcolor(binslo_amp.reshape(2,2),cmap=plt.cm.OrRd)
        ax4.set_xlabel("Avg. bins > counts: {:d} (per Amp)".format(cutlo),fontsize=8)
        ax4.tick_params(axis='x',labelsize=10,labelbottom='off')
        ax4.tick_params(axis='y',labelsize=10,labelleft='off')
        ax4.annotate("Amp 1\n{:.1f}".format(binslo_amp[0]),
                 xy=(0.4,0.4),
                 fontsize=10
                 )
        ax4.annotate("Amp 2\n{:.1f}".format(binslo_amp[1]),
                 xy=(1.4,0.4),
                 fontsize=10
                 )
        ax4.annotate("Amp 3\n{:.1f}".format(binslo_amp[2]),
                 xy=(0.4,1.4),
                 fontsize=10
                 )
        ax4.annotate("Amp 4\n{:.1f}".format(binslo_amp[3]),
                 xy=(1.4,1.4),
                 fontsize=10
                 )
        heatmap2=ax5.pcolor(binsmed_amp.reshape(2,2),cmap=plt.cm.OrRd)
        ax5.set_xlabel("Avg. bins > counts: {:d} (per Amp)".format(cutmed),fontsize=8)
        ax5.tick_params(axis='x',labelsize=10,labelbottom='off')
        ax5.tick_params(axis='y',labelsize=10,labelleft='off')
        ax5.annotate("Amp 1\n{:.1f}".format(binsmed_amp[0]),
                 xy=(0.4,0.4),
                 fontsize=10
                 )
        ax5.annotate("Amp 2\n{:.1f}".format(binsmed_amp[1]),
                 xy=(1.4,0.4),
                 fontsize=10
                 )
        ax5.annotate("Amp 3\n{:.1f}".format(binsmed_amp[2]),
                 xy=(0.4,1.4),
                 fontsize=10
                 )
        ax5.annotate("Amp 4\n{:.1f}".format(binsmed_amp[3]),
                 xy=(1.4,1.4),
                 fontsize=10
                 )

        heatmap3=ax6.pcolor(binshi_amp.reshape(2,2),cmap=plt.cm.OrRd)
        ax6.set_xlabel("Avg. bins > counts: {:d} (per Amp)".format(cuthi),fontsize=8)
        ax6.tick_params(axis='x',labelsize=10,labelbottom='off')
        ax6.tick_params(axis='y',labelsize=10,labelleft='off')
        ax6.annotate("Amp 1\n{:.1f}".format(binshi_amp[0]),
                 xy=(0.4,0.4),
                 fontsize=10
                 )
        ax6.annotate("Amp 2\n{:.1f}".format(binshi_amp[1]),
                 xy=(1.4,0.4),
                 fontsize=10
                 )
        ax6.annotate("Amp 3\n{:.1f}".format(binshi_amp[2]),
                 xy=(0.4,1.4),
                 fontsize=10
                 )
        ax6.annotate("Amp 4\n{:.1f}".format(binshi_amp[3]),
                 xy=(1.4,1.4),
                 fontsize=10
                 )
    plt.tight_layout()
    fig.savefig(outfile)

def plot_countpix(qa_dict,outfile):
    """
    Plot pixel counts above some threshold
    
    qa_dict example::
        
        {'CAMERA': 'r0',
        'EXPID': '00000006',
        'QATIME': '2016-08-02T14:39:59.157986',
        'PANAME': 'PREPROC',
        'PARAMS': {'CUTLO': 3, 'CUTHI': 10},
        'METRICS': {'NPIX_LOW': 0,
                  'NPIX_AMP': [254549, 0, 242623, 0],
                  'NPIX_HIGH': 0,
                  'NPIX_HIGH_AMP': [1566, 0, 1017, 0]}}}

    Args:
        qa_dict: qa dictionary from countpix qa
        outfile: pdf file of the plot
    """
    expid=qa_dict["EXPID"]
    camera = qa_dict["CAMERA"]
    paname=qa_dict["PANAME"]
    countlo=qa_dict["METRICS"]["NPIX_LOW"]
    countlo_amp=np.array(qa_dict["METRICS"]["NPIX_AMP"])
    counthi=qa_dict["METRICS"]["NPIX_HIGH"]
    counthi_amp=np.array(qa_dict["METRICS"]["NPIX_HIGH_AMP"])

    cutlo=qa_dict["PARAMS"]["CUTLO"]
    cuthi=qa_dict["PARAMS"]["CUTHI"]

    fig=plt.figure()
    plt.suptitle("Count pixels after {}, Camera: {}, ExpID: {}".format(paname,camera,expid),fontsize=10,y=0.99)
    ax1=fig.add_subplot(211)
    heatmap1=ax1.pcolor(countlo_amp.reshape(2,2),cmap=plt.cm.OrRd)
    plt.title('Total Pixels > {:d} sigma = {:f}'.format(cutlo,countlo), fontsize=10)
    ax1.set_xlabel("# pixels > {:d} sigma (per Amp)".format(cutlo),fontsize=10)
    ax1.tick_params(axis='x',labelsize=10,labelbottom='off')
    ax1.tick_params(axis='y',labelsize=10,labelleft='off')
    ax1.annotate("Amp 1\n{:f}".format(countlo_amp[0]),
                 xy=(0.4,0.4),
                 fontsize=10
                 )
    ax1.annotate("Amp 2\n{:f}".format(countlo_amp[1]),
                 xy=(1.4,0.4),
                 fontsize=10
                 )
    ax1.annotate("Amp 3\n{:f}".format(countlo_amp[2]),
                 xy=(0.4,1.4),
                 fontsize=10
                 )
    ax1.annotate("Amp 4\n{:f}".format(countlo_amp[3]),
                 xy=(1.4,1.4),
                 fontsize=10
                 )
    ax2=fig.add_subplot(212)
    heatmap2=ax2.pcolor(counthi_amp.reshape(2,2),cmap=plt.cm.OrRd)
    plt.title('Total Pixels > {:d} sigma = {:f}'.format(cuthi,counthi), fontsize=10)
    ax2.set_xlabel("# pixels > {:d} sigma (per Amp)".format(cuthi),fontsize=10)
    ax2.tick_params(axis='x',labelsize=10,labelbottom='off')
    ax2.tick_params(axis='y',labelsize=10,labelleft='off')
    ax2.annotate("Amp 1\n{:f}".format(counthi_amp[0]),
                 xy=(0.4,0.4),
                 fontsize=10
                 )
    ax2.annotate("Amp 2\n{:f}".format(counthi_amp[1]),
                 xy=(1.4,0.4),
                 fontsize=10
                 )
    ax2.annotate("Amp 3\n{:f}".format(counthi_amp[2]),
                 xy=(0.4,1.4),
                 fontsize=10
                 )
    ax2.annotate("Amp 4\n{:f}".format(counthi_amp[3]),
                 xy=(1.4,1.4),
                 fontsize=10
                 )
    plt.tight_layout()
    fig.savefig(outfile)

def plot_bias_overscan(qa_dict,outfile):
    """
    Map of bias from overscan from 4 regions of CCD
    
    qa_dict example::

        {'ARM': 'r',
        'EXPID': '00000006',
        'QATIME': '2016-08-02T14:39:59.773229',
        'PANAME': 'PREPROC',
        'SPECTROGRAPH': 0,
        'METRICS': {'BIAS': -0.0080487558302569373,
                  'BIAS_AMP': array([-0.01132324, -0.02867701, -0.00277266,  0.0105779 ])}}

    Args:
        qa_dict: qa dictionary from countpix qa
        outfile : pdf file of the plot
    """
    expid=qa_dict["EXPID"]
    camera =qa_dict["CAMERA"]
    paname=qa_dict["PANAME"]

    bias=qa_dict["METRICS"]["BIAS"]
    bias_amp=qa_dict["METRICS"]["BIAS_AMP"]
    fig=plt.figure()
    plt.suptitle("Bias from overscan region after {}, Camera: {}, ExpID: {}".format(paname,camera,expid),fontsize=10,y=0.99)
    ax1=fig.add_subplot(111)
    heatmap1=ax1.pcolor(bias_amp.reshape(2,2),cmap=plt.cm.OrRd)
    plt.title('Bias = {:.4f}'.format(bias), fontsize=10)
    ax1.set_xlabel("Avg. bias value per Amp (photon counts)",fontsize=10)
    ax1.tick_params(axis='x',labelsize=10,labelbottom='off')
    ax1.tick_params(axis='y',labelsize=10,labelleft='off')
    ax1.annotate("Amp 1\n{:.3f}".format(bias_amp[0]),
                 xy=(0.4,0.4),
                 fontsize=10
                 )
    ax1.annotate("Amp 2\n{:.3f}".format(bias_amp[1]),
                 xy=(1.4,0.4),
                 fontsize=10
                 )
    ax1.annotate("Amp 3\n{:.3f}".format(bias_amp[2]),
                 xy=(0.4,1.4),
                 fontsize=10
                 )
    ax1.annotate("Amp 4\n{:.3f}".format(bias_amp[3]),
                 xy=(1.4,1.4),
                 fontsize=10
                 )
    fig.savefig(outfile)

def plot_XWSigma(qa_dict,outfile):
    """
    Plot XWSigma
    
    qa_dict example::
        
        {'ARM': 'r',
         'EXPID': '00000006',
         'QATIME': '2016-07-08T06:05:34.56',
         'PANAME': 'PREPROC',
         'SPECTROGRAPH': 0,
         'METRICS': {'XSIGMA': array([ 1.9, 1.81, 1.2...]),
                   'XSIGMA_MED': 1.81,
                   'XSIGMA_AMP': array([ 1.9, 1.8, 1.7, 1.84]),
                   'WSIGMA': array([ 1.9, 1.81, 1.2...]),
                   'WSIGMA_MED': 1.81,
                   'WSIGMA_AMP': array([ 1.9, 1.8, 1.7, 1.84]),
                   'XWSIGMA': array([ 1.72, 1.72])}}

    Args:
        qa_dict: qa dictionary from countpix qa
        outfile : file of the plot
    """
    camera=qa_dict["CAMERA"]
    expid=qa_dict["EXPID"]
    pa=qa_dict["PANAME"]
    xsigma=qa_dict["METRICS"]["XSIGMA"]
    wsigma=qa_dict["METRICS"]["WSIGMA"]
    xsigma_med=qa_dict["METRICS"]["XSIGMA_MED"]
    wsigma_med=qa_dict["METRICS"]["WSIGMA_MED"]
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

    if "XSIGMA_AMP" in qa_dict["METRICS"]:
        xsigma_amp=qa_dict["METRICS"]["XSIGMA_AMP"]
        wsigma_amp=qa_dict["METRICS"]["WSIGMA_AMP"]
        ax3=fig.add_subplot(223)
        heatmap3=ax3.pcolor(xsigma_amp.reshape(2,2),cmap=plt.cm.OrRd)
        plt.title('X Sigma = {:.4f}'.format(xsigma_med), fontsize=10)
        ax3.set_xlabel("X std. dev. per Amp (# of pixels)",fontsize=10)
        ax3.tick_params(axis='x',labelsize=10,labelbottom='off')
        ax3.tick_params(axis='y',labelsize=10,labelleft='off')
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
        ax4.tick_params(axis='x',labelsize=10,labelbottom='off')
        ax4.tick_params(axis='y',labelsize=10,labelleft='off')
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
    
    qa_dict example::
        
        {'ARM': 'r',
         'EXPID': '00000006',
         'QATIME': '2016-07-08T06:05:34.56',
         'PANAME': 'PREPROC',
         'SPECTROGRAPH': 0,
         'METRICS': {'RMS': 40.218151021598679,
                   'RMS_AMP': array([ 55.16847779,   2.91397089,  55.26686528,   2.91535373])
                   'NOISE': 40.21815,
                   'NOISE_AMP': array([ 55.168,   2.913,   55.266,  2.915])
                    }
        }

    Args:
        qa_dict: dictionary of qa outputs from running qa_quicklook.Get_RMS
        outfile: Name of plot output file
    """
    rms=qa_dict["METRICS"]["RMS"]
    rms_amp=qa_dict["METRICS"]["RMS_AMP"]
    rms_over=qa_dict["METRICS"]["NOISE"]
    rms_over_amp=qa_dict["METRICS"]["NOISE_AMP"]
    # arm=qa_dict["ARM"]
    # spectrograph=qa_dict["SPECTROGRAPH"]
    camera = qa_dict["CAMERA"]

    expid=qa_dict["EXPID"]
    pa=qa_dict["PANAME"]

    fig=plt.figure()
    plt.suptitle("RMS image counts per amplifier, Camera: {}, ExpID: {}".format(camera,expid),fontsize=10,y=0.99)
    ax1=fig.add_subplot(211)
    heatmap1=ax1.pcolor(rms_amp.reshape(2,2),cmap=plt.cm.OrRd)
    plt.title('RMS = {:.4f}'.format(rms), fontsize=10)
    ax1.set_xlabel("RMS per Amp (photon counts)",fontsize=10)
    ax1.tick_params(axis='x',labelsize=10,labelbottom='off')
    ax1.tick_params(axis='y',labelsize=10,labelleft='off')
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
    plt.title('RMS Overscan = {:.4f}'.format(rms_over), fontsize=10)
    ax2.set_xlabel("RMS Overscan per Amp (photon counts)",fontsize=10)
    ax2.tick_params(axis='x',labelsize=10,labelbottom='off')
    ax2.tick_params(axis='y',labelsize=10,labelleft='off')
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
    """
    Plot integral.

    qa_dict example::
        
        {'ARM': 'r',
         'EXPID': '00000002',
         'PANAME': 'SKYSUB',
         'QATIME': '2016-08-02T15:01:26.239419',
         'SPECTROGRAPH': 0,
         'METRICS': {'INTEG': array([ 3587452.149007]),
                   'INTEG_AVG': 3587452.1490069963,
                   'INTEG_AVG_AMP': array([ 1824671.67950129,        0.        ,  1752550.23876224,
                                    0.        ])}}

    Args:
        qa_dict: qa dictionary
        outfile : output plot file
    """
    expid=qa_dict["EXPID"]
    camera =qa_dict["CAMERA"]
    paname=qa_dict["PANAME"]
    std_integral=np.array(qa_dict["METRICS"]["INTEG"])
    std_integral_avg=qa_dict["METRICS"]["INTEG_AVG"]
    std_fiberid=qa_dict["METRICS"]["STD_FIBERID"]

    fig=plt.figure()
    plt.suptitle("Total integrals of STD spectra {}, Camera: {}, ExpID: {}".format(paname,camera,expid),fontsize=10,y=0.99)
    index=np.arange(1,len(std_integral)+1)
    ax1=fig.add_subplot(211)
    hist_med=ax1.bar(index,std_integral,color='b',align='center')
    ax1.set_xlabel('STD fiber ID',fontsize=10)
    ax1.set_ylabel('Integral (photon counts)',fontsize=10)
    ax1.tick_params(axis='x',labelsize=10)
    ax1.tick_params(axis='y',labelsize=10)
    ax1.set_xticks(index)
    ax1.set_xticklabels(std_fiberid)
    
    if "INTEG_AVG_AMP" in qa_dict["METRICS"]:

        std_integral_amp=np.array(qa_dict["METRICS"]["INTEG_AVG_AMP"])
        ax2=fig.add_subplot(212)
        heatmap1=ax2.pcolor(std_integral_amp.reshape(2,2),cmap=plt.cm.OrRd)
        plt.title('Integral Average = {:.4f}'.format(std_integral_avg), fontsize=10)
        ax2.set_xlabel("Average integrals of STD spectra (photon counts)",fontsize=10)
        ax2.tick_params(axis='x',labelsize=10,labelbottom='off')
        ax2.tick_params(axis='y',labelsize=10,labelleft='off')
        ax2.annotate("Amp 1\n{:.1f}".format(std_integral_amp[0]),
                 xy=(0.4,0.4),
                 fontsize=10
                 )
        ax2.annotate("Amp 2\n{:.1f}".format(std_integral_amp[1]),
                 xy=(1.4,0.4),
                 fontsize=10
                 )
        ax2.annotate("Amp 3\n{:.1f}".format(std_integral_amp[2]),
                 xy=(0.4,1.4),
                 fontsize=10
                 )
        ax2.annotate("Amp 4\n{:.1f}".format(std_integral_amp[3]),
                 xy=(1.4,1.4),
                 fontsize=10
                 )

    plt.tight_layout()
    fig.savefig(outfile)

def plot_sky_continuum(qa_dict,outfile):
    """
    Plot mean sky continuum from lower and higher wavelength range for each 
    fiber and accross amps.
    
    Example qa_dict::
        
        {'ARM': 'r',
        'EXPID': '00000006',
        'QATIME': '2016-08-02T14:40:02.766684,
        'PANAME': 'APPLY_FIBERFLAT',
        'SPECTROGRAPH': 0,
        'METRICS': {'SKYCONT': 359.70078667259668,
                 'SKYCONT_AMP': array([ 374.19163643,    0.        ,  344.76184662,    0.        ]),
                 'SKYCONT_FIBER': [357.23814787655738,   358.14982775192709,   359.34380640332847,
                                    361.55526717275529, 360.46690568746544,   360.49561926858325,   
                                    359.08761654248656,   361.26910267767016],
                 'SKYFIBERID': [4, 19, 30, 38, 54, 55, 57, 62]}}

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
    
    ax1=fig.add_subplot(211)
    hist_med=ax1.bar(index,skycont_fiber,color='b',align='center')
    ax1.set_xlabel('SKY fiber ID',fontsize=10)
    ax1.set_ylabel('Sky Continuum (photon counts)',fontsize=10)
    ax1.tick_params(axis='x',labelsize=6)
    ax1.tick_params(axis='y',labelsize=10)
    ax1.set_xticks(index)
    ax1.set_xticklabels(fiberid)
    ax1.set_xlim(0)
    
    if "SKYCONT_AMP" in qa_dict["METRICS"]:
        skycont_amps=np.array(qa_dict["METRICS"]["SKYCONT_AMP"])
        ax2=fig.add_subplot(212)
        heatmap1=ax2.pcolor(skycont_amps.reshape(2,2),cmap=plt.cm.OrRd)
        plt.title('Sky Continuum = {:.4f}'.format(skycont), fontsize=10)
        ax2.set_xlabel("Avg. sky continuum per Amp (photon counts)",fontsize=10)
        ax2.tick_params(axis='x',labelsize=10,labelbottom='off')
        ax2.tick_params(axis='y',labelsize=10,labelleft='off')
        ax2.annotate("Amp 1\n{:.1f}".format(skycont_amps[0]),
                 xy=(0.4,0.4),
                 fontsize=10
                 )
        ax2.annotate("Amp 2\n{:.1f}".format(skycont_amps[1]),
                 xy=(1.4,0.4),
                 fontsize=10
                 )
        ax2.annotate("Amp 3\n{:.1f}".format(skycont_amps[2]),
                 xy=(0.4,1.4),
                 fontsize=10
                 )
        ax2.annotate("Amp 4\n{:.1f}".format(skycont_amps[3]),
                 xy=(1.4,1.4),
                 fontsize=10
                 )

    plt.tight_layout()
    fig.savefig(outfile)

def plot_sky_peaks(qa_dict,outfile):
    """
    Plot rms of sky peaks for smy fibers across amps
       
    Example qa_dict::

        {'ARM': 'r',
        'EXPID': '00000006',
        'QATIME': '2016-07-08T06:05:34.56',
        'PANAME': 'APPLY_FIBERFLAT', 'SPECTROGRAPH': 0,
        'METRICS': {'PEAKCOUNT': array([ 1500.0,  1400.0, ....]),
                  'PEAKCOUNT_RMS': 1445.0,
                  'PEAKCOUNT_RMS_SKY': 1455.0,
                  'PEAKCOUNT_RMS_AMP': array([ 1444.0, 1433.0, 1422.0, 1411.0])}}

    Args:
        qa_dict: dictionary from sky peaks QA
        outfile: pdf file to save the plot
    """
    expid=qa_dict["EXPID"]
    camera=qa_dict["CAMERA"]
    paname=qa_dict["PANAME"]
    sumcount=qa_dict["METRICS"]["PEAKCOUNT"]
    fiber=np.arange(sumcount.shape[0])
    skyfiber_rms=qa_dict["METRICS"]["PEAKCOUNT_RMS_SKY"]
    fig=plt.figure()
    plt.suptitle("Counts and Amp RMS for Sky Fibers after {}, Camera: {}, ExpID: {}".format(paname,camera,expid),fontsize=10,y=0.99)

    ax1=fig.add_subplot(211)
    hist_x=ax1.bar(fiber,sumcount,align='center')
    ax1.set_xlabel("Fiber #",fontsize=10)
    ax1.set_ylabel("Summed counts over sky peaks (photon counts)",fontsize=10)
    ax1.tick_params(axis='x',labelsize=10)
    ax1.tick_params(axis='y',labelsize=10)
    plt.xlim(0,len(fiber))

    if "PEAKCOUNT_RMS_AMP" in qa_dict["METRICS"]:
        sky_amp_rms=np.array(qa_dict["METRICS"]["PEAKCOUNT_RMS_AMP"])
        ax2=fig.add_subplot(212)
        heatmap2=ax2.pcolor(sky_amp_rms.reshape(2,2),cmap=plt.cm.OrRd)
        plt.title('Sky peaks for sky fibers = {:.4f}'.format(skyfiber_rms), fontsize=10)
        ax2.set_xlabel("Sky Fiber RMS for peak wavelengths per Amp (photon counts)",fontsize=10)
        ax2.tick_params(axis='x',labelsize=10,labelbottom='off')
        ax2.tick_params(axis='y',labelsize=10,labelleft='off')
        ax2.annotate("Amp 1\n{:.1f}".format(sky_amp_rms[0]),
                 xy=(0.4,0.4),
                 fontsize=10
                 )
        ax2.annotate("Amp 2\n{:.1f}".format(sky_amp_rms[1]),
                 xy=(1.4,0.4),
                 fontsize=10
                 )
        ax2.annotate("Amp 3\n{:.1f}".format(sky_amp_rms[2]),
                 xy=(0.4,1.4),
                 fontsize=10
                 )
        ax2.annotate("Amp 4\n{:.1f}".format(sky_amp_rms[3]),
                 xy=(1.4,1.4),
                 fontsize=10
                 )

    plt.tight_layout()
    fig.savefig(outfile)

def plot_residuals(qa_dict,outfile):
    """
    Plot histogram of sky residuals for each sky fiber
    
    qa_dict example::

        {'ARM': 'r',
         'EXPID': '00000002',
         'PANAME': 'SKYSUB',
         'QATIME': '2016-08-31T10:48:58.984638',
         'SPECTROGRAPH': 0,
         'METRICS': {'MED_RESID': -8.461671761494927e-06,
                   'MED_RESID_FIBER': array([ 0.29701871, -0.29444709]),
                   'NBAD_PCHI': 0,
                   'NREJ': 0,
                   'NSKY_FIB': 2,
                   'RESID_PER': [-20.15769508014313, 23.938934018349393]}}

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
    skyfiberid=qa_dict["METRICS"]["SKY_FIBERID"]
    
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
    
def plot_SNR(qa_dict,outfile,objlist,badfibs):
    """
    Plot SNR

    qa_dict example::

        {'ARM': 'r',
         'EXPID': '00000006',
         'QATIME': '2016-08-02T14:40:03.670962',
         'PANAME': 'SKYSUB',
         'SPECTROGRAPH': 0,
         'METRICS': {'ELG_FIBERID': [0, 3, 4],
                   'ELG_SNR_MAG': array([[  1.04995347,   1.75609447,   0.86920898],
                                        [ 22.40120888,  21.33947945,  23.26506996]]),
                   'LRG_FIBERID': [2, 8, 9],
                   'LRG_SNR_MAG': array([[  0.92477875,   1.45257228,   1.52262706],
                                        [ 22.75508881,  21.35451317,  21.39620209]]),
                   'MEDIAN_AMP_SNR': array([ 4.64376854,  0.        ,  5.02489801,  0.        ]),
                   'MEDIAN_SNR': array([  1.04995347,   0.47679704,   0.92477875,   1.75609447,
                                          0.86920898,   1.03979459,   0.46717453,  38.31675053,
                                          1.45257228,   1.52262706]),
                   'QSO_FIBERID': [5],
                   'QSO_SNR_MAG': array([[  1.03979459], [ 22.95341873]]),
                   'STAR_FIBERID': [7],
                   'STAR_SNR_MAG': array([[ 38.31675053], [ 17.13783646]])}}}
    
    Args:
        qa_dict: dictionary of qa outputs from running qa_quicklook.Calculate_SNR
        outfile: Name of figure.
    """

    med_snr=qa_dict["METRICS"]["MEDIAN_SNR"]
    avg_med_snr=np.mean(med_snr)
    index=np.arange(med_snr.shape[0])
    resid_snr=qa_dict["METRICS"]["SNR_RESID"]
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
        if len(badfibs) > 0:
            fibers = np.array(fibers)
            badfibs = np.array(badfibs)
            for ff in range(len(badfibs)):
                rm = np.where(fibers==badfibs[ff])[0]
                if len(rm) == 1:
                    badfibs=list(badfibs)
                    fibers=list(fibers)
                    fibers.remove(fibers[rm[0]])
                    mag.remove(mag[rm[0]])
                    snr.remove(snr[rm[0]])
                    for fi in range(len(badfibs)):
                        badfibs[fi]-=1
                        badfibs.remove(badfibs[0])
        mags.append(mag)
        snrs.append(snr)
        for c in range(len(fibers)):
            ras = qa_dict['METRICS']['RA'][fibers[c]]
            decs = qa_dict['METRICS']['DEC'][fibers[c]]
            ra.append(ras)
            dec.append(decs)

    if camera[0] == 'b':
        thisfilter='DECAM_G'
    elif camera[0] == 'r':
        thisfilter='DECAM_R'
    else:
        thisfilter='DECAM_Z'

    fig=plt.figure()
    plt.suptitle("Signal/Noise after {}, Camera: {}, ExpID: {}".format(paname,camera,expid),fontsize=10,y=0.99)

    gs=GridSpec(7,8)
    ax1=fig.add_subplot(gs[1:4,:4]) #- ax2 for amp ccd map below.

    hist_med=ax1.plot(index,med_snr,linewidth=1)
    ax1.set_xlabel('Fiber #',fontsize=10)
    ax1.set_ylabel('Median S/N',fontsize=10)
    ax1.tick_params(axis='x',labelsize=10)
    ax1.tick_params(axis='y',labelsize=10)
    ax1.set_xlim(0)

    ax2=fig.add_subplot(gs[1:4,4:])
    ax2.set_title('Residual SNR (Calculated SNR - SNR from fit)',fontsize=8)
    ax2.set_xlabel('RA',fontsize=8)
    ax2.set_ylabel('DEC',fontsize=8)
    resid_plot=ax2.scatter(ra,dec,s=2,c=resid_snr,cmap=plt.cm.bwr)
    fig.colorbar(resid_plot,ticks=[np.min(resid_snr),0,np.max(resid_snr)])
    
    #    #- plot for amp zones
    #    if "MEDIAN_AMP_SNR" in qa_dict["METRICS"]:
    #        ax2=fig.add_subplot(gs[1:4,4:])
    #        med_amp_snr=qa_dict["METRICS"]["MEDIAN_AMP_SNR"]
    #        heatmap_med=ax2.pcolor(med_amp_snr.reshape(2,2),cmap=plt.cm.OrRd)
    #        plt.title('Avg. Median S/N = {:.4f}'.format(avg_med_snr), fontsize=10)
    #        ax2.set_xlabel("Avg. Median S/N (per Amp)",fontsize=10)
    #        ax2.tick_params(axis='x',labelsize=10,labelbottom='off')
    #        ax2.tick_params(axis='y',labelsize=10,labelleft='off')
    #        ax2.annotate("Amp 1\n{:.3f}".format(med_amp_snr[0]),
    #                 xy=(0.4,0.4), #- Full scale is 2
    #                 fontsize=10
    #                 )
    #        ax2.annotate("Amp 2\n{:.3f}".format(med_amp_snr[1]),
    #                 xy=(1.4,0.4),
    #                 fontsize=10
    #                 )
    #        ax2.annotate("Amp 3\n{:.3f}".format(med_amp_snr[2]),
    #                 xy=(0.4,1.4),
    #                 fontsize=10
    #                 )
    #
    #        ax2.annotate("Amp 4\n{:.3f}".format(med_amp_snr[3]),
    #                 xy=(1.4,1.4),
    #                 fontsize=10
    #                 )

    for i in range(len(o)):
        if i == 0:
            ax=fig.add_subplot(gs[4:,:2])
        elif i == 1:
            ax=fig.add_subplot(gs[4:,2:4])
        elif i == 2:
            ax=fig.add_subplot(gs[4:,4:6])
        else:
            ax=fig.add_subplot(gs[4:,6:])

        objtype=list(objlist)[i]
        objid=np.where(np.array(list(objlist))==objtype)[0][0]
        obj_mag=mags[objid]
        obj_snr=snrs[objid]
        obj_fit_values=qa_dict["METRICS"]["FITCOEFF_TGT"][objid]

        ax.set_ylabel('Median S/N',fontsize=8)
        ax.set_xlabel('Magnitude ({})'.format(thisfilter),fontsize=8)
        ax.set_title('{}'.format(objtype), fontsize=8)
        plot_mag=np.arange(np.min(obj_mag),np.max(obj_mag),0.1)
        plot_fit=10**(obj_fit_values[0]+obj_fit_values[1]*plot_mag+obj_fit_values[2]*plot_mag**2)
        ax.set_xlim(np.min(obj_mag)-0.1,np.max(obj_mag)+0.1)
        ax.set_ylim(np.min(obj_snr)-0.1,np.max(obj_snr)+0.1)
        ax.xaxis.set_ticks(np.arange(int(np.min(obj_mag)),int(np.max(obj_mag))+1,1.0))
        ax.tick_params(axis='x',labelsize=6,labelbottom='on')
        ax.tick_params(axis='y',labelsize=6,labelleft='on')
        ax.plot(obj_mag,obj_snr,'b.')
        ax.plot(plot_mag,plot_fit,'y')
    
    plt.tight_layout()
    fig.savefig(outfile)
