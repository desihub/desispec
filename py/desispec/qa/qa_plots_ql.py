"""
This includes routines to make pdf plots on the qa outputs from quicklook.
"""

import numpy as np
from matplotlib import pyplot as plt

def plot_countspectralbins(qa_dict,outfile):
    """
    qa_dict: dictionary of qa outputs from running qa_quicklook.CountSpectralBins
    qa_dict example: 
                   {'EXPID': '00000002', 'SPECTROGRAPH': 0, 'THRESHOLD': 100, 'ARM': 'r', 'VALUE': {'CNTS_ABOVE_THRESH': array([ 4171.,  3492.,  3637.,  3524.,  3554.,  3595.,  3513.,  3612., 3497.,  3496.])}, 'WAVE_GRID': 0.5}
    
    While reading from yaml output file, qa_dict is the value to the first top level key, which is the name of that QA  
    """
    counts=qa_dict["VALUE"]["CNTS_ABOVE_THRESH"]
    fiberindex=np.arange(counts.shape[0])
    arm=qa_dict["ARM"]
    spectrograph=qa_dict["SPECTROGRAPH"]
    expid=qa_dict["EXPID"]    

    fig,ax=plt.subplots()
    plt.suptitle("Counts above threshold after Extraction, Camera: %s%s, ExpID: %s"%(arm,spectrograph,expid))
    ax.set_xticks(np.arange(counts.shape[0]), minor=False)

    x_label="Fiber #"
    y_label="Counts above threshold = %s"%qa_dict["THRESHOLD"]
    ax.set_xlabel(x_label,fontsize=10)
    ax.set_ylabel(y_label,fontsize=10)
    ax.tick_params(axis='x',labelsize=10)
    ax.tick_params(axis='y',labelsize=10)
    #ax.plot(counts,'bo')
    hist=ax.bar(fiberindex,counts)
    fig.savefig(outfile)

def plot_SNR(qa_dict,outfile):
    """
    qa_dict: dictionary of qa outputs from running qa_quicklook.Calculate_SNR
    qa_dict example:
    {'ARM': 'r',
  'EXPID': '00000002',
  'SPECTROGRAPH': 0,
  'VALUE': {'MED_AMP_SNR': array([ 4.06939214,  3.38182425,  0.        ,  0.        ]),
   'MED_SNR': array([ 17.11461103,   2.20929805,   2.7827068 ,   2.5686027 ,
            2.21004108,   2.71689086,   2.8345871 ,   2.93099466,
            2.17110525,   2.41165487]),
   'TOT_AMP_SNR': array([ 1116.43412277,   842.90823424,     0.        ,     0.        ]),
   'TOT_SNR': array([ 742.38974703,   99.72438062,  111.37616747,   99.19502236,
            93.49246903,  114.45615701,  104.71081645,  101.26104959,
            66.90231848,   68.12932419])}}    
    """

    med_snr=qa_dict["VALUE"]["MED_SNR"]
    tot_snr=qa_dict["VALUE"]["TOT_SNR"]
    med_amp_snr=qa_dict["VALUE"]["MED_AMP_SNR"]
    tot_amp_snr=qa_dict["VALUE"]["TOT_AMP_SNR"]
    index=np.arange(med_snr.shape[0])
    arm=qa_dict["ARM"]
    spectrograph=qa_dict["SPECTROGRAPH"]
    expid=qa_dict["EXPID"]

    fig=plt.figure()
    plt.suptitle("Signal/Noise after Sky subtraction, Camera: %s%s, ExpID: %s"%(arm,spectrograph,expid))
    

    ax1=fig.add_subplot(221)
    hist_med=ax1.bar(index,med_snr)
    ax1.set_xlabel('Fiber #',fontsize=10)
    ax1.set_ylabel('Median S/N',fontsize=10)
    ax1.tick_params(axis='x',labelsize=10)
    ax1.tick_params(axis='y',labelsize=10)

    ax2=fig.add_subplot(222)
    hist_tot=ax2.bar(index,tot_snr)
    ax2.set_xlabel('Fiber #',fontsize=10)
    ax2.set_ylabel('Total S/N',fontsize=10)
    ax2.tick_params(axis='x',labelsize=10)
    ax2.tick_params(axis='y',labelsize=10)

    ax3=fig.add_subplot(223)
    heatmap_med=ax3.pcolor(med_amp_snr.reshape(2,2).T,cmap=plt.cm.coolwarm)
    ax3.set_xlabel("Avg. Median S/N (per Amp)",fontsize=10)
    ax3.tick_params(axis='x',labelsize=10,labelbottom='off')
    ax3.tick_params(axis='y',labelsize=10,labelleft='off')
    ax3.annotate("Amp 1\n%.3f"%med_amp_snr[0], #- Name and order not sure yet.
                 xy=(0.4,1.4), #- Full scale is 2
                 fontsize=10
                 )
    ax3.annotate("Amp 2\n%.3f"%med_amp_snr[1],
                 xy=(0.4,0.4),
                 fontsize=10
                 )
    ax3.annotate("Amp 3\n%.3f"%med_amp_snr[2],
                 xy=(1.4,0.4),
                 fontsize=10
                 )
    
    ax3.annotate("Amp 4\n%.3f"%med_amp_snr[3],
                 xy=(1.4,1.4),
                 fontsize=10
                 )

    ax4=fig.add_subplot(224)
    heatmap_tot=ax4.pcolor(tot_amp_snr.reshape(2,2).T,cmap=plt.cm.coolwarm)
    #ax41=ax4.twiny()
    ax4.set_xlabel("Avg. Total S/N (per Amp)",fontsize=10)
    ax4.tick_params(axis='x',labelsize=10,labelbottom='off')
    ax4.tick_params(axis='y',labelsize=10,labelleft='off')
    ax4.annotate("Amp 1\n%.3f"%tot_amp_snr[0],
                 xy=(0.4,1.4), #- Full scale is 2
                 fontsize=10
                 )
    ax4.annotate("Amp 2\n%.3f"%tot_amp_snr[1],
                 xy=(0.4,0.4),
                 fontsize=10
                 )
    ax4.annotate("Amp 3\n%.3f"%tot_amp_snr[2],
                 xy=(1.4,0.4),
                 fontsize=10
                 )
    
    ax4.annotate("Amp 4\n%.3f"%tot_amp_snr[3],
                 xy=(1.4,1.4),
                 fontsize=10
                 )

    fig.savefig(outfile)
   
    
