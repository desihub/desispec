"""
This includes routines to make pdf plots on the qa outputs from quicklook.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

def plot_countspectralbins(qa_dict,outfile):
    """Plot count spectral bins.

    While reading from yaml output file, qa_dict is the value to the first top level key, which is the name of that QA

    `qa_dict` example::

        {'ARM': 'r',
         'EXPID': '00000006',
         'QATIME': '2016-08-02T14:40:03.269684',
         'PANAME': 'BOXCAR',
         'SPECTROGRAPH': 0,
         'VALUE': {'NBINS100': array([ 2575.,  2611.,  2451.,  2495.,  2357.,  2452.,  2528.,  2501.,  2548.,  2461.]),
                   'NBINS100_AMP': array([ 1249.74,     0.  ,  1198.01,     0.  ]),
                   'NBINS250': array([ 2503.,  2539.,  2161.,  2259.,  2077.,  2163.,  2284.,  2268.,  2387.,  2210.]),
                   'NBINS250_AMP': array([ 1149.55,     0.  ,  1095.02,     0.  ]),
                   'NBINS500': array([ 2307.,  2448.,   229.,  1910.,    94.,   306.,  2056.,  1941.,  2164.,   785.]),
                   'NBINS500_AMP': array([ 688.85,    0.  ,  648.75,    0.  ])
                   'NGOODFIBERS: 10}}}

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

    gs=GridSpec(7,6)
    ax1=fig.add_subplot(gs[1:4,:2])
    ax2=fig.add_subplot(gs[1:4,2:4])
    ax3=fig.add_subplot(gs[1:4,4:])
    ax4=fig.add_subplot(gs[4:,:2])
    ax5=fig.add_subplot(gs[4:,2:4])
    ax6=fig.add_subplot(gs[4:,4:])

    hist_med=ax1.bar(index,bins100,color='b',align='center')
    ax1.set_xlabel('Fiber #',fontsize=10)
    ax1.set_ylabel('Counts > 100',fontsize=10)
    ax1.tick_params(axis='x',labelsize=10)
    ax1.tick_params(axis='y',labelsize=10)

    hist_med=ax2.bar(index,bins250,color='r',align='center')
    ax2.set_xlabel('Fiber #',fontsize=10)
    ax2.set_ylabel('Counts > 250',fontsize=10)
    ax2.tick_params(axis='x',labelsize=10)
    ax2.tick_params(axis='y',labelsize=10)

    hist_med=ax3.bar(index,bins500,color='g',align='center')
    ax3.set_xlabel('Fiber #',fontsize=10)
    ax3.set_ylabel('Counts > 500',fontsize=10)
    ax3.tick_params(axis='x',labelsize=10)
    ax3.tick_params(axis='y',labelsize=10)

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
            'QATIME': '2016-08-02T14:39:59.157986',
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
            'QATIME': '2016-08-02T14:39:59.773229',
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
    
def plot_RMS(qa_dict,outfile):
    """Plot RMS
    `qa_dict` example:
        {'ARM': 'r',
         'EXPID': '00000006',
         'MJD': 57581.91467038749,
         'PANAME': 'PREPROC',
         'SPECTROGRAPH': 0,
         'VALUE': {'RMS': 40.218151021598679,
                   'RMS_AMP': array([ 55.16847779,   2.91397089,  55.26686528,   2.91535373])}}
     Args:
        qa_dict: dictionary of qa outputs from running qa_quicklook.Get_RMS
        outfile: Name of plot output file
    """

    rms_amp=qa_dict["VALUE"]["RMS_AMP"]
    arm=qa_dict["ARM"]
    spectrograph=qa_dict["SPECTROGRAPH"]
    expid=qa_dict["EXPID"]
    mjd=qa_dict["MJD"]
    pa=qa_dict["PANAME"]

    fig=plt.figure()
    plt.suptitle("RMS image counts per amplifier, Camera: %s%s, ExpID: %s"%(arm,spectrograph,expid))
    ax1=fig.add_subplot(111)
    heatmap1=ax1.pcolor(rms_amp.reshape(2,2).T,cmap=plt.cm.coolwarm)
    ax1.set_xlabel("RMS (per Amp)",fontsize=10)
    ax1.tick_params(axis='x',labelsize=10,labelbottom='off')
    ax1.tick_params(axis='y',labelsize=10,labelleft='off')
    ax1.annotate("Amp 1\n%.3f"%rms_amp[0],
                 xy=(0.4,0.4),
                 fontsize=10
                 )
    ax1.annotate("Amp 2\n%.3f"%rms_amp[1],
                 xy=(1.4,0.4),
                 fontsize=10
                 )
    ax1.annotate("Amp 3\n%.3f"%rms_amp[2],
                 xy=(0.4,1.4),
                 fontsize=10
                 )

    ax1.annotate("Amp 4\n%.3f"%rms_amp[3],
                 xy=(1.4,1.4),
                 fontsize=10
                 )
    fig.savefig(outfile)

def plot_integral(qa_dict,outfile):
    """
    qa_dict example:
        {'ARM': 'r',
         'EXPID': '00000002',
         'PANAME': 'SKYSUB',
         'QATIME': '2016-08-02T15:01:26.239419',
         'SPECTROGRAPH': 0,
         'VALUE': {'INTEG': array([ 3587452.149007]),
                   'INTEG_AVG': 3587452.1490069963,
                   'INTEG_AVG_AMP': array([ 1824671.67950129,        0.        ,  1752550.23876224,        0.        ])}}
   """
    spectrograph=qa_dict["SPECTROGRAPH"]
    expid=qa_dict["EXPID"]
    arm=qa_dict["ARM"]
    paname=qa_dict["PANAME"]
    std_integral=np.array(qa_dict["VALUE"]["INTEG"])
    std_integral_amp=np.array(qa_dict["VALUE"]["INTEG_AVG_AMP"])

    fig=plt.figure()
    plt.suptitle("Total integrals of STD spectra %s, Camera: %s%s, ExpID: %s"%(paname,arm,spectrograph,expid))
    index=np.arange(1,len(std_integral)+1)
    ax1=fig.add_subplot(211)
    hist_med=ax1.bar(index,std_integral,color='b',align='center')
    ax1.set_xlabel('STD fibers',fontsize=10)
    ax1.set_ylabel('Integral',fontsize=10)
    ax1.tick_params(axis='x',labelsize=10)
    ax1.tick_params(axis='y',labelsize=10)
    ax1.set_xticks(index)
    ax1.set_xticklabels(index)
    
    ax2=fig.add_subplot(212)
    heatmap1=ax2.pcolor(std_integral_amp.reshape(2,2).T,cmap=plt.cm.coolwarm)
    ax2.set_xlabel("Average integrals of STD spectra",fontsize=10)
    ax2.tick_params(axis='x',labelsize=10,labelbottom='off')
    ax2.tick_params(axis='y',labelsize=10,labelleft='off')
    ax2.annotate("Amp 1\n%.1f"%std_integral_amp[0],
                 xy=(0.4,0.4),
                 fontsize=10
                 )
    ax2.annotate("Amp 2\n%.1f"%std_integral_amp[1],
                 xy=(1.4,0.4),
                 fontsize=10
                 )
    ax2.annotate("Amp 3\n%.1f"%std_integral_amp[2],
                 xy=(0.4,1.4),
                 fontsize=10
                 )
    ax2.annotate("Amp 4\n%.1f"%std_integral_amp[3],
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
           'QATIME': '2016-08-02T14:40:02.766684,
           'PANAME': 'APPLY_FIBERFLAT',
           'SPECTROGRAPH': 0,
           'VALUE': {'SKYCONT': 359.70078667259668,
                     'SKYCONT_AMP': array([ 374.19163643,    0.        ,  344.76184662,    0.        ]),
                     'SKYCONT_FIBER': [357.23814787655738,   358.14982775192709,   359.34380640332847,   361.55526717275529,
    360.46690568746544,   360.49561926858325,   359.08761654248656,   361.26910267767016],
                     'SKYFIBERID': [4, 19, 30, 38, 54, 55, 57, 62]}}

       args: qa_dict: dictionary from sky continuum QA
             outfile: pdf file to save the plot
    """
    spectrograph=qa_dict["SPECTROGRAPH"]
    expid=qa_dict["EXPID"]
    arm=qa_dict["ARM"]
    paname=qa_dict["PANAME"]
    skycont_fiber=np.array(qa_dict["VALUE"]["SKYCONT_FIBER"])
    skycont_amps=np.array(qa_dict["VALUE"]["SKYCONT_AMP"])
    index=np.arange(skycont_fiber.shape[0])
    fiberid=qa_dict["VALUE"]["SKYFIBERID"]
    fig=plt.figure()
    plt.suptitle("Mean Sky Continuum after %s, Camera: %s%s, ExpID: %s"%(paname,arm,spectrograph,expid))
    
    ax1=fig.add_subplot(211)
    hist_med=ax1.bar(index,skycont_fiber,color='b',align='center')
    ax1.set_xlabel('SKY fibers',fontsize=10)
    ax1.set_ylabel('Sky Continuum',fontsize=10)
    ax1.tick_params(axis='x',labelsize=10)
    ax1.tick_params(axis='y',labelsize=10)
    ax1.set_xticks(index)
    ax1.set_xticklabels(fiberid)
    
    ax2=fig.add_subplot(212)
    heatmap1=ax2.pcolor(skycont_amps.reshape(2,2).T,cmap=plt.cm.coolwarm)
    ax2.set_xlabel("Avg. sky continuum (per Amp)",fontsize=10)
    ax2.tick_params(axis='x',labelsize=10,labelbottom='off')
    ax2.tick_params(axis='y',labelsize=10,labelleft='off')
    ax2.annotate("Amp 1\n%.1f"%skycont_amps[0],
                 xy=(0.4,0.4),
                 fontsize=10
                 )
    ax2.annotate("Amp 2\n%.1f"%skycont_amps[1],
                 xy=(1.4,0.4),
                 fontsize=10
                 )
    ax2.annotate("Amp 3\n%.1f"%skycont_amps[2],
                 xy=(0.4,1.4),
                 fontsize=10
                 )
    ax2.annotate("Amp 4\n%.1f"%skycont_amps[3],
                 xy=(1.4,1.4),
                 fontsize=10
                 )
    fig.savefig(outfile)

def plot_SNR(qa_dict,outfile):

    """Plot SNR

    `qa_dict` example::

        {'ARM': 'r',
         'EXPID': '00000006',
         'QATIME': '2016-08-02T14:40:03.670962',
         'PANAME': 'SKYSUB',
         'SPECTROGRAPH': 0,
         'VALUE': {'ELG_FIBERID': [0, 3, 4],
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

    med_snr=qa_dict["VALUE"]["MEDIAN_SNR"]
    med_amp_snr=qa_dict["VALUE"]["MEDIAN_AMP_SNR"]
    index=np.arange(med_snr.shape[0])
    arm=qa_dict["ARM"]
    spectrograph=qa_dict["SPECTROGRAPH"]
    expid=qa_dict["EXPID"]
    paname=qa_dict["PANAME"]

    elg_snr_mag=qa_dict["VALUE"]["ELG_SNR_MAG"]
    lrg_snr_mag=qa_dict["VALUE"]["LRG_SNR_MAG"]
    qso_snr_mag=qa_dict["VALUE"]["QSO_SNR_MAG"]
    star_snr_mag=qa_dict["VALUE"]["STAR_SNR_MAG"]

    fig=plt.figure()
    plt.suptitle("Signal/Noise after %s, Camera: %s%s, ExpID: %s"%(paname,arm,spectrograph,expid))

    gs=GridSpec(7,8)
    ax1=fig.add_subplot(gs[1:4,:4])
    ax2=fig.add_subplot(gs[1:4,4:])
    ax3=fig.add_subplot(gs[4:,:2])
    ax4=fig.add_subplot(gs[4:,2:4])
    ax5=fig.add_subplot(gs[4:,4:6])
    ax6=fig.add_subplot(gs[4:,6:])

    hist_med=ax1.bar(index,med_snr,align='center')
    ax1.set_xlabel('Fiber #',fontsize=10)
    ax1.set_ylabel('Median S/N',fontsize=10)
    ax1.tick_params(axis='x',labelsize=10)
    ax1.tick_params(axis='y',labelsize=10)

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

    ax3.set_ylabel('Median S/N',fontsize=8)
    ax3.set_xlabel('Magnitude (DECAM_R)',fontsize=8)
    ax3.set_title("ELG", fontsize=8)
    ax3.set_xlim(np.min(elg_snr_mag[1])-0.1,np.max(elg_snr_mag[1])+0.1)
    ax3.set_ylim(np.min(elg_snr_mag[0])-0.1,np.max(elg_snr_mag[0])+0.1)
    ax3.xaxis.set_ticks(np.arange(int(np.min(elg_snr_mag[1])),int(np.max(elg_snr_mag[1]))+1,0.5))
    #print np.arange(int(np.min(elg_snr_mag[1]))-0.5,int(np.max(elg_snr_mag[1]))+1.0,0.5)
    ax3.tick_params(axis='x',labelsize=6,labelbottom='on')
    ax3.tick_params(axis='y',labelsize=6,labelleft='on')
    ax3.plot(elg_snr_mag[1],elg_snr_mag[0],'b.')

    ax4.set_ylabel('',fontsize=10)
    ax4.set_xlabel('Magnitude (DECAM_R)',fontsize=8)
    ax4.set_title("LRG",fontsize=8)
    ax4.set_xlim(np.min(lrg_snr_mag[1])-0.1,np.max(lrg_snr_mag[1])+0.1)
    ax4.set_ylim(np.min(lrg_snr_mag[0])-0.1,np.max(lrg_snr_mag[0])+0.1)
    ax4.xaxis.set_ticks(np.arange(int(np.min(lrg_snr_mag[1])),int(np.max(lrg_snr_mag[1]))+1,0.5))
    ax4.tick_params(axis='x',labelsize=6,labelbottom='on')
    ax4.tick_params(axis='y',labelsize=6,labelleft='on')
    ax4.plot(lrg_snr_mag[1],lrg_snr_mag[0],'r.')

    ax5.set_ylabel('',fontsize=10)
    ax5.set_xlabel('Magnitude (DECAM_R)',fontsize=8)
    ax5.set_title("QSO", fontsize=8)
    ax5.set_xlim(np.min(qso_snr_mag[1])-0.1,np.max(qso_snr_mag[1])+0.1)
    ax5.set_ylim(np.min(qso_snr_mag[0])-0.1,np.max(qso_snr_mag[0])+0.1)
    ax5.xaxis.set_ticks(np.arange(int(np.min(qso_snr_mag[1])),int(np.max(qso_snr_mag[1]))+1,0.5))
    ax5.tick_params(axis='x',labelsize=6,labelbottom='on')
    ax5.tick_params(axis='y',labelsize=6,labelleft='on')
    ax5.plot(qso_snr_mag[1],qso_snr_mag[0],'g.')

    ax6.set_ylabel('',fontsize=10)
    ax6.set_xlabel('Magnitude (DECAM_R)',fontsize=8)
    ax6.set_title("STD", fontsize=8)
    ax6.set_xlim(np.min(star_snr_mag[1])-0.1,np.max(star_snr_mag[1])+0.1)
    ax6.set_ylim(np.min(star_snr_mag[0])-0.1,np.max(star_snr_mag[0])+0.1)
    ax6.xaxis.set_ticks(np.arange(int(np.min(star_snr_mag[1])),int(np.max(star_snr_mag[1]))+1,0.5))
    ax6.tick_params(axis='x',labelsize=6,labelbottom='on')
    ax6.tick_params(axis='y',labelsize=6,labelleft='on')
    ax6.plot(star_snr_mag[1],star_snr_mag[0],'k.')

    plt.tight_layout()
    fig.savefig(outfile)
