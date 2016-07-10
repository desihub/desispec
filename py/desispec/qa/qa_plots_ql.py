"""
This includes routines to make pdf plots on the qa outputs from quicklook.
"""

import numpy as np
from matplotlib import pyplot as plt

def plot_countspectralbins(qa_dict,outfile):
    """Plot count spectral bins.

    While reading from yaml output file, qa_dict is the value to the first top level key, which is the name of that QA

    `qa_dict` example::

        {'ARM': 'r',
         'EXPID': '00000006',
         'MJD': 57578.78098693542,
         'PANAME': 'BOXCAR',
         'SPECTROGRAPH': 0,
         'VALUE': {'NBINS100': array([ 2575.,  2611.,  2451.,  2495.,  2357.,  2452.,  2528.,  2501.,  2548.,  2461.]),
                   'NBINS100_AMP': array([ 1249.74,     0.  ,  1198.01,     0.  ]),
                   'NBINS250': array([ 2503.,  2539.,  2161.,  2259.,  2077.,  2163.,  2284.,  2268.,  2387.,  2210.]),
                   'NBINS250_AMP': array([ 1149.55,     0.  ,  1095.02,     0.  ]),
                   'NBINS500': array([ 2307.,  2448.,   229.,  1910.,    94.,   306.,  2056.,  1941.,  2164.,   785.]),
                   'NBINS500_AMP': array([ 688.85,    0.  ,  648.75,    0.  ])}}}

    Args:
        qa_dict: dictionary of qa outputs from running qa_quicklook.CountSpectralBins
        outfile: Name of figure.
    """

    arm=qa_dict["ARM"]
    spectrograph=qa_dict["SPECTROGRAPH"]
    expid=qa_dict["EXPID"]
    paname=qa_dict["PANAME"]
    
    bins100=qa_dict["VALUE"]["NBINS100"]
    bins250=qa_dict["VALUE"]["NBINS250"]
    bins500=qa_dict["VALUE"]["NBINS500"]

    bins100_amp=qa_dict["VALUE"]["NBINS100_AMP"]
    bins250_amp=qa_dict["VALUE"]["NBINS250_AMP"]
    bins500_amp=qa_dict["VALUE"]["NBINS500_AMP"]

    index=np.arange(bins100.shape[0])

    fig=plt.figure()
    plt.suptitle("Count spectral bins after %s, Camera: %s%s, ExpID: %s"%(paname,arm,spectrograph,expid))


    ax1=fig.add_subplot(231)
    hist_med=ax1.bar(index,bins100,color='b')
    ax1.set_xlabel('Fiber #',fontsize=10)
    ax1.set_ylabel('Counts > 100',fontsize=10)
    ax1.tick_params(axis='x',labelsize=10)
    ax1.tick_params(axis='y',labelsize=10)

    ax2=fig.add_subplot(232)
    hist_med=ax2.bar(index,bins250,color='r')
    ax2.set_xlabel('Fiber #',fontsize=10)
    ax2.set_ylabel('Counts > 250',fontsize=10)
    ax2.tick_params(axis='x',labelsize=10)
    ax2.tick_params(axis='y',labelsize=10)

    ax3=fig.add_subplot(233)
    hist_med=ax3.bar(index,bins500,color='g')
    ax3.set_xlabel('Fiber #',fontsize=10)
    ax3.set_ylabel('Counts > 500',fontsize=10)
    ax3.tick_params(axis='x',labelsize=10)
    ax3.tick_params(axis='y',labelsize=10)

    ax4=fig.add_subplot(234)
    heatmap1=ax4.pcolor(bins100_amp.reshape(2,2).T,cmap=plt.cm.coolwarm)
    ax4.set_xlabel("Bins above 100 counts (per Amp)",fontsize=10)
    ax4.tick_params(axis='x',labelsize=10,labelbottom='off')
    ax4.tick_params(axis='y',labelsize=10,labelleft='off')
    ax4.annotate("Amp 1\n%.1f"%bins100_amp[0],
                 xy=(0.4,0.4),
                 fontsize=10
                 )
    ax4.annotate("Amp 2\n%.1f"%bins100_amp[1],
                 xy=(1.4,0.4),
                 fontsize=10
                 )
    ax4.annotate("Amp 3\n%.1f"%bins100_amp[2],
                 xy=(0.4,1.4),
                 fontsize=10
                 )
    ax4.annotate("Amp 4\n%.1f"%bins100_amp[3],
                 xy=(1.4,1.4),
                 fontsize=10
                 )
    ax5=fig.add_subplot(235)
    heatmap2=ax5.pcolor(bins250_amp.reshape(2,2).T,cmap=plt.cm.coolwarm)
    ax5.set_xlabel("Bins above 250 counts (per Amp)",fontsize=10)
    ax5.tick_params(axis='x',labelsize=10,labelbottom='off')
    ax5.tick_params(axis='y',labelsize=10,labelleft='off')
    ax5.annotate("Amp 1\n%.1f"%bins250_amp[0],
                 xy=(0.4,0.4),
                 fontsize=10
                 )
    ax5.annotate("Amp 2\n%.1f"%bins250_amp[1],
                 xy=(1.4,0.4),
                 fontsize=10
                 )
    ax5.annotate("Amp 3\n%.1f"%bins250_amp[2],
                 xy=(0.4,1.4),
                 fontsize=10
                 )
    ax5.annotate("Amp 4\n%.1f"%bins250_amp[3],
                 xy=(1.4,1.4),
                 fontsize=10
                 )

    ax6=fig.add_subplot(236)
    heatmap3=ax6.pcolor(bins500_amp.reshape(2,2).T,cmap=plt.cm.coolwarm)
    ax6.set_xlabel("Bins above 500 counts (per Amp)",fontsize=10)
    ax6.tick_params(axis='x',labelsize=10,labelbottom='off')
    ax6.tick_params(axis='y',labelsize=10,labelleft='off')
    ax6.annotate("Amp 1\n%.1f"%bins500_amp[0],
                 xy=(0.4,0.4),
                 fontsize=10
                 )
    ax6.annotate("Amp 2\n%.1f"%bins500_amp[1],
                 xy=(1.4,0.4),
                 fontsize=10
                 )
    ax6.annotate("Amp 3\n%.1f"%bins500_amp[2],
                 xy=(0.4,1.4),
                 fontsize=10
                 )
    ax6.annotate("Amp 4\n%.1f"%bins500_amp[3],
                 xy=(1.4,1.4),
                 fontsize=10
                 )
    plt.tight_layout()
    fig.savefig(outfile)

def plot_countpix(qa_dict,outfile):
    """
       plot pixel counts above some threshold
       qa_dict example:
           {'ARM': 'r',
            'EXPID': '00000006',
            'MJD': 57578.780697648355,
            'PANAME': 'PREPROC',
            'SPECTROGRAPH': 0,
            'VALUE': {'NPIX100': 0,
                      'NPIX100_AMP': [254549, 0, 242623, 0],
                      'NPIX3SIG': 3713,
                      'NPIX3SIG_AMP': [128158, 2949, 132594, 3713],
                      'NPIX500': 0,
                      'NPIX500_AMP': [1566, 0, 1017, 0]}}}
       args: qa_dict : qa dictionary from countpix qa
             outfile : pdf file of the plot
    """
    spectrograph=qa_dict["SPECTROGRAPH"]
    expid=qa_dict["EXPID"]
    arm=qa_dict["ARM"]
    paname=qa_dict["PANAME"]
    count3sig_amp=np.array(qa_dict["VALUE"]["NPIX3SIG_AMP"])
    count100_amp=np.array(qa_dict["VALUE"]["NPIX100_AMP"])
    count500_amp=np.array(qa_dict["VALUE"]["NPIX500_AMP"])
    fig=plt.figure()
    plt.suptitle("Count pixels after %s, Camera: %s%s, ExpID: %s"%(paname,arm,spectrograph,expid))
    ax1=fig.add_subplot(221)
    heatmap1=ax1.pcolor(count3sig_amp.reshape(2,2).T,cmap=plt.cm.coolwarm)
    ax1.set_xlabel("Counts above 3sig. (per Amp)",fontsize=10)
    ax1.tick_params(axis='x',labelsize=10,labelbottom='off')
    ax1.tick_params(axis='y',labelsize=10,labelleft='off')
    ax1.annotate("Amp 1\n%.1f"%count3sig_amp[0],
                 xy=(0.4,0.4),
                 fontsize=10
                 )
    ax1.annotate("Amp 2\n%.1f"%count3sig_amp[1],
                 xy=(1.4,0.4),
                 fontsize=10
                 )
    ax1.annotate("Amp 3\n%.1f"%count3sig_amp[2],
                 xy=(0.4,1.4),
                 fontsize=10
                 )

    ax1.annotate("Amp 4\n%.1f"%count3sig_amp[3],
                 xy=(1.4,1.4),
                 fontsize=10
                 )
    ax2=fig.add_subplot(222)
    heatmap2=ax2.pcolor(count100_amp.reshape(2,2).T,cmap=plt.cm.coolwarm)
    ax2.set_xlabel("Counts above 100 (per Amp)",fontsize=10)
    ax2.tick_params(axis='x',labelsize=10,labelbottom='off')
    ax2.tick_params(axis='y',labelsize=10,labelleft='off')
    ax2.annotate("Amp 1\n%.1f"%count100_amp[0],
                 xy=(0.4,0.4),
                 fontsize=10
                 )
    ax2.annotate("Amp 2\n%.1f"%count100_amp[1],
                 xy=(1.4,0.4),
                 fontsize=10
                 )
    ax2.annotate("Amp 3\n%.1f"%count100_amp[2],
                 xy=(0.4,1.4),
                 fontsize=10
                 )

    ax2.annotate("Amp 4\n%.1f"%count100_amp[3],
                 xy=(1.4,1.4),
                 fontsize=10
                 )
    ax3=fig.add_subplot(223)
    heatmap3=ax3.pcolor(count500_amp.reshape(2,2).T,cmap=plt.cm.coolwarm)
    ax3.set_xlabel("Counts above 500 (per Amp)",fontsize=10)
    ax3.tick_params(axis='x',labelsize=10,labelbottom='off')
    ax3.tick_params(axis='y',labelsize=10,labelleft='off')
    ax3.annotate("Amp 1\n%.1f"%count500_amp[0],
                 xy=(0.4,0.4),
                 fontsize=10
                 )
    ax3.annotate("Amp 2\n%.1f"%count500_amp[1],
                 xy=(1.4,0.4),
                 fontsize=10
                 )
    ax3.annotate("Amp 3\n%.1f"%count500_amp[2],
                 xy=(0.4,1.4),
                 fontsize=10
                 )

    ax3.annotate("Amp 4\n%.1f"%count500_amp[3],
                 xy=(1.4,1.4),
                 fontsize=10
                 )
    fig.savefig(outfile)

def plot_bias_overscan(qa_dict,outfile):
    """
       map of bias from overscan from 4 regions of CCD
       qa_dict example:
           {'ARM': 'r',
            'EXPID': '00000006',
            'MJD': 57578.780704701225,
            'PANAME': 'PREPROC',
            'SPECTROGRAPH': 0,
            'VALUE': {'BIAS': -0.0080487558302569373,
                      'BIAS_AMP': array([-0.01132324, -0.02867701, -0.00277266,  0.0105779 ])}}
       args: qa_dict : qa dictionary from countpix qa
             outfile : pdf file of the plot
    """
    spectrograph=qa_dict["SPECTROGRAPH"]
    expid=qa_dict["EXPID"]
    arm=qa_dict["ARM"]
    paname=qa_dict["PANAME"]
    bias_amp=qa_dict["VALUE"]["BIAS_AMP"]
    fig=plt.figure()
    plt.suptitle("Bias from overscan region after %s, Camera: %s%s, ExpID: %s"%(paname,arm,spectrograph,expid))
    ax1=fig.add_subplot(111)
    heatmap1=ax1.pcolor(bias_amp.reshape(2,2).T,cmap=plt.cm.coolwarm)
    ax1.set_xlabel("Avg. bias value (per Amp)",fontsize=10)
    ax1.tick_params(axis='x',labelsize=10,labelbottom='off')
    ax1.tick_params(axis='y',labelsize=10,labelleft='off')
    ax1.annotate("Amp 1\n%.3f"%bias_amp[0],
                 xy=(0.4,0.4),
                 fontsize=10
                 )
    ax1.annotate("Amp 2\n%.3f"%bias_amp[1],
                 xy=(1.4,0.4),
                 fontsize=10
                 )
    ax1.annotate("Amp 3\n%.3f"%bias_amp[2],
                 xy=(0.4,1.4),
                 fontsize=10
                 )

    ax1.annotate("Amp 4\n%.3f"%bias_amp[3],
                 xy=(1.4,1.4),
                 fontsize=10
                 )
    fig.savefig(outfile)
    
def plot_sky_continuum(qa_dict,outfile):
    """
       plot mean sky continuum from lower and higher wavelength range for each fiber and accross amps
       example qa_dict:
          {'ARM': 'r',
           'EXPID': '00000006',
           'MJD': 57578.78115332748,
           'PANAME': 'APPLY_FIBERFLAT',
           'SPECTROGRAPH': 0,
           'VALUE': {'SKY': 359.70078667259668,
                     'SKY_AMPS': array([ 374.19163643,    0.        ,  344.76184662,    0.        ]),
                     'SKY_FIBER': [357.23814787655738,  358.14982775192709,  359.34380640332847,  361.55526717275529,   360.46690568746544,  360.49561926858325,  359.08761654248656,  361.26910267767016]}}}

       args: qa_dict: dictionary from sky continuum QA
             outfile: pdf file to save the plot
    """
    spectrograph=qa_dict["SPECTROGRAPH"]
    expid=qa_dict["EXPID"]
    arm=qa_dict["ARM"]
    paname=qa_dict["PANAME"]
    sky_fiber=np.array(qa_dict["VALUE"]["SKY_FIBER"])
    sky_amps=np.array(qa_dict["VALUE"]["SKY_AMPS"])
    index=np.arange(sky_fiber.shape[0])
    fig=plt.figure()
    plt.suptitle("Mean Sky Continuum after %s, Camera: %s%s, ExpID: %s"%(paname,arm,spectrograph,expid))
    
    ax1=fig.add_subplot(211)
    hist_med=ax1.bar(index,sky_fiber,color='b')
    ax1.set_xlabel('Sky fibers',fontsize=10)
    ax1.set_ylabel('Sky Continuum',fontsize=10)
    ax1.tick_params(axis='x',labelsize=10)
    ax1.tick_params(axis='y',labelsize=10)

    ax2=fig.add_subplot(212)
    heatmap1=ax2.pcolor(sky_amps.reshape(2,2).T,cmap=plt.cm.coolwarm)
    ax2.set_xlabel("Avg. sky continuum (per Amp)",fontsize=10)
    ax2.tick_params(axis='x',labelsize=10,labelbottom='off')
    ax2.tick_params(axis='y',labelsize=10,labelleft='off')
    ax2.annotate("Amp 1\n%.1f"%sky_amps[0],
                 xy=(0.4,0.4),
                 fontsize=10
                 )
    ax2.annotate("Amp 2\n%.1f"%sky_amps[1],
                 xy=(1.4,0.4),
                 fontsize=10
                 )
    ax2.annotate("Amp 3\n%.1f"%sky_amps[2],
                 xy=(0.4,1.4),
                 fontsize=10
                 )
    ax2.annotate("Amp 4\n%.1f"%sky_amps[3],
                 xy=(1.4,1.4),
                 fontsize=10
                 )
    fig.savefig(outfile)

def plot_SNR(qa_dict,outfile):
    """Plot SNR

    `qa_dict` example::

        {'ARM': 'r',
         'EXPID': '00000006',
         'MJD': 57578.78131121235,
         'PANAME': 'SKYSUB',
         'SPECTROGRAPH': 0,
         'VALUE': {'MEDIAN_AMP_SNR': array([ 11.28466596,   0.        ,  13.18927372,   0.        ]),
                   'MEDIAN_SNR': array([ 26.29012459,  35.02498105,   3.30635973,   7.69106173,
            0.586899  ,   3.59830798,  11.75768833,   8.276959  ,  16.70907383,   4.82177165])}}}

    Args:
        qa_dict: dictionary of qa outputs from running qa_quicklook.Calculate_SNR
        outfile: Name of figure.
    """

    med_snr=qa_dict["VALUE"]["MEDIAN_SNR"]
    med_amp_snr=qa_dict["VALUE"]["MEDIAN_AMP_SNR"]
    index=np.arange(med_snr.shape[0])
    arm=qa_dict["ARM"]
    spectrograph=qa_dict["SPECTROGRAPH"]
    expid=qa_dict["EXPID"]
    paname=qa_dict["PANAME"]

    fig=plt.figure()
    plt.suptitle("Signal/Noise after %s, Camera: %s%s, ExpID: %s"%(paname,arm,spectrograph,expid))


    ax1=fig.add_subplot(211)
    hist_med=ax1.bar(index,med_snr)
    ax1.set_xlabel('Fiber #',fontsize=10)
    ax1.set_ylabel('Median S/N',fontsize=10)
    ax1.tick_params(axis='x',labelsize=10)
    ax1.tick_params(axis='y',labelsize=10)


    ax2=fig.add_subplot(212)
    heatmap_med=ax2.pcolor(med_amp_snr.reshape(2,2).T,cmap=plt.cm.coolwarm)
    ax2.set_xlabel("Avg. Median S/N (per Amp)",fontsize=10)
    ax2.tick_params(axis='x',labelsize=10,labelbottom='off')
    ax2.tick_params(axis='y',labelsize=10,labelleft='off')
    ax2.annotate("Amp 1\n%.3f"%med_amp_snr[0],
                 xy=(0.4,0.4), #- Full scale is 2
                 fontsize=10
                 )
    ax2.annotate("Amp 2\n%.3f"%med_amp_snr[1],
                 xy=(1.4,0.4),
                 fontsize=10
                 )
    ax2.annotate("Amp 3\n%.3f"%med_amp_snr[2],
                 xy=(0.4,1.4),
                 fontsize=10
                 )

    ax2.annotate("Amp 4\n%.3f"%med_amp_snr[3],
                 xy=(1.4,1.4),
                 fontsize=10
                 )

    fig.savefig(outfile)
