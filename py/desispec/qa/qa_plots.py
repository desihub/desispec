""" Module for QA plots
"""
from __future__ import print_function, absolute_import, division

import numpy as np
from scipy import signal
import scipy
import pdb
import copy

from desiutil.log import get_logger
from desispec import fluxcalibration as dsflux
from desispec.util import set_backend
set_backend()

import matplotlib
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

from desispec import util

from desiutil.plots import plot_slices as du_pslices


def brick_zbest(outfil, zf, qabrick):
    """ QA plots for Flux calibration in a Frame

    Args:
        outfil:
        qabrick:
        zf: ZfindBase object

    Returns:
        Stuff?
    """
    sty_otype = get_sty_otype()
    # Convert types (this should become obsolete)
    param = qabrick.data['ZBEST']['PARAMS']
    zftypes = []
    for ztype in zf.spectype:
        if ztype in param['ELG_TYPES']:
            zftypes.append('ELG')
        elif ztype in param['QSO_TYPES']:
            zftypes.append('QSO')
        elif ztype in param['STAR_TYPES']:
            zftypes.append('STAR')
        else:
            zftypes.append('UNKNWN')
    zftypes = np.array(zftypes)

    # Plot
    fig = plt.figure(figsize=(8, 5.0))
    gs = gridspec.GridSpec(2,2)

    # Error vs. z
    ax0 = plt.subplot(gs[0,0])

    #
    ax0.set_ylabel(r'$\delta z / (1+z)$')
    ax0.set_ylim(0.0, 0.002)
    ax0.set_xlabel('z')

    for key in sty_otype:
        idx = np.where(zftypes == key)[0]
        if len(idx) == 0:
            continue
        ax0.scatter(zf.z[idx], zf.zerr[idx]/(1+zf.z[idx]), marker='o',
                    color=sty_otype[key]['color'], label=sty_otype[key]['lbl'])

    # Legend
    legend = ax0.legend(loc='upper left', borderpad=0.3,
                       handletextpad=0.3, fontsize='small')

    # Meta text
    ax2 = plt.subplot(gs[1,1])
    ax2.set_axis_off()
    show_meta(ax2, qabrick, 'ZBEST', outfil)


    # Finish
    plt.tight_layout(pad=0.1,h_pad=0.0,w_pad=0.0)
    plt.savefig(outfil)
    plt.close()
    print('Wrote QA ZBEST file: {:s}'.format(outfil))



def frame_skyres(outfil, frame, skymodel, qaframe):
    """
    Generate QA plots and files for sky residuals of a given frame

    Parameters
    ----------
    outfil: str
        Name of output file
    frame: Frame object
    skymodel: SkyModel object
    qaframe: QAFrame object
    """

    # Access metrics
    '''
    wavg_ivar = np.sum(res_ivar,0)
    chi2_wavg = np.sum(wavg_res**2 * wavg_ivar)
    dof_wavg = np.sum(wavg_ivar > 0.)
    pchi2_wavg = scipy.stats.chisqprob(chi2_wavg, dof_wavg)
    chi2_med = np.sum(med_res**2 * wavg_ivar)
    pchi2_med = scipy.stats.chisqprob(chi2_med, dof_wavg)
    '''
    med_res = qaframe.qa_data['SKYSUB']["METRICS"]["MED_RESID_WAVE"]
    wavg_res = qaframe.qa_data['SKYSUB']["METRICS"]["WAVG_RES_WAVE"]

    # Plot
    fig = plt.figure(figsize=(8, 10.0))
    gs = gridspec.GridSpec(4,2)
    xmin,xmax = np.min(frame.wave), np.max(frame.wave)

    # Simple residual plot
    ax0 = plt.subplot(gs[0,:])
    ax0.plot(frame.wave, med_res, label='Median Res')
    ax0.plot(frame.wave, signal.medfilt(med_res,51), color='black', label='Median**2 Res')
    ax0.plot(frame.wave, signal.medfilt(wavg_res,51), color='red', label='Med WAvgRes')

    #
    ax0.plot([xmin,xmax], [0., 0], '--', color='gray')
    ax0.plot([xmin,xmax], [0., 0], '--', color='gray')
    ax0.set_xlabel('Wavelength')
    ax0.set_ylabel('Sky Residuals (Counts)')
    ax0.set_xlim(xmin,xmax)
    ax0.set_xlabel('Wavelength')
    ax0.set_ylabel('Sky Residuals (Counts)')
    ax0.set_xlim(xmin,xmax)
    med0 = np.maximum(np.abs(np.median(med_res)), 1.)
    ax0.set_ylim(-5.*med0, 5.*med0)
    #ax0.text(0.5, 0.85, 'Sky Meanspec',
    #    transform=ax_flux.transAxes, ha='center')

    # Legend
    legend = ax0.legend(loc='upper right', borderpad=0.3,
                        handletextpad=0.3, fontsize='small')

    # Histogram of all residuals
    ax1 = plt.subplot(gs[1,0])
    xmin,xmax = -5., 5.

    # Histogram
    binsz = qaframe.qa_data['SKYSUB']["PARAMS"]["BIN_SZ"]
    hist = np.asarray(qaframe.qa_data['SKYSUB']["METRICS"]["DEVS_1D"])
    edges = np.asarray(qaframe.qa_data['SKYSUB']["METRICS"]["DEVS_EDGES"])

    xhist = (edges[1:] + edges[:-1])/2.
    ax1.hist(xhist, color='blue', bins=edges, weights=hist)#, histtype='step')
    # PDF for Gaussian
    area = binsz * np.sum(hist)

    xppf = np.linspace(scipy.stats.norm.ppf(0.0001), scipy.stats.norm.ppf(0.9999), 100)
    ax1.plot(xppf, area*scipy.stats.norm.pdf(xppf), 'r-', alpha=1.0)

    ax1.set_xlabel(r'Res/$\sigma$')
    ax1.set_ylabel('N')
    ax1.set_xlim(xmin,xmax)

    # Meta text
    #- limit the dictionary to residuals only for meta
    qaresid=copy.deepcopy(qaframe)
    resid_keys=['NREJ','NSKY_FIB','NBAD_PCHI','MED_RESID','RESID_PER']
    qaresid.qa_data['SKYSUB']['METRICS']={key:value for key,value in qaframe.qa_data['SKYSUB']
                                         ['METRICS'].items() if key in resid_keys}

    ax2 = plt.subplot(gs[1,1])
    ax2.set_axis_off()
    show_meta(ax2, qaresid, 'SKYSUB', outfil)

    #- SNR Plot
    elg_snr_mag = qaframe.qa_data['SKYSUB']["METRICS"]["ELG_SNR_MAG"]
    lrg_snr_mag = qaframe.qa_data['SKYSUB']["METRICS"]["LRG_SNR_MAG"]
    qso_snr_mag = qaframe.qa_data['SKYSUB']["METRICS"]["QSO_SNR_MAG"]
    star_snr_mag = qaframe.qa_data['SKYSUB']["METRICS"]["STAR_SNR_MAG"]

    ax3 = plt.subplot(gs[2,0])
    ax4 = plt.subplot(gs[2,1])
    ax5 = plt.subplot(gs[3,0])
    ax6 = plt.subplot(gs[3,1])

    ax3.set_ylabel(r'Median S/N')
    ax3.set_xlabel('')
    ax3.set_title(r'ELG')
    if len(elg_snr_mag[1]) > 0:  #- at least 1 elg fiber?
        select=np.where((elg_snr_mag[1] != np.array(None)) & (~np.isnan(elg_snr_mag[1])) & (np.abs(elg_snr_mag[1])!=np.inf))[0] #- Remove None, nan and inf values in mag
        if select.shape[0]>0:

            xmin=np.min(elg_snr_mag[1][select])-0.1
            xmax=np.max(elg_snr_mag[1][select])+0.1
            ax3.set_xlim(xmin,xmax)
            ax3.set_ylim(np.min(elg_snr_mag[0][select])-0.1,np.max(elg_snr_mag[0][select])+0.1)
            ax3.xaxis.set_ticks(np.arange(int(np.min(elg_snr_mag[1][select])),int(np.max(elg_snr_mag[1][select]))+1,0.5))
            ax3.tick_params(axis='x',labelsize=10,labelbottom='on')
            ax3.tick_params(axis='y',labelsize=10,labelleft='on')
            ax3.plot(elg_snr_mag[1][select],elg_snr_mag[0][select],'b.')

    ax4.set_ylabel('')
    ax4.set_xlabel('')
    ax4.set_title(r'LRG')
    if len(lrg_snr_mag[1]) > 0:  #- at least 1 lrg fiber?
        select=np.where((lrg_snr_mag[1] != np.array(None)) & (~np.isnan(lrg_snr_mag[1])) & (np.abs(lrg_snr_mag[1])!=np.inf))[0]
        if select.shape[0]>0:
            xmin=np.min(lrg_snr_mag[1][select])-0.1
            xmax=np.max(lrg_snr_mag[1][select])+0.1
            ax4.set_xlim(xmin,xmax)
            ax4.set_ylim(np.min(lrg_snr_mag[0][select])-0.1,np.max(lrg_snr_mag[0][select])+0.1)
            ax4.xaxis.set_ticks(np.arange(int(np.min(lrg_snr_mag[1][select])),int(np.max(lrg_snr_mag[1][select]))+1,0.5))
            ax4.tick_params(axis='x',labelsize=10,labelbottom='on')
            ax4.tick_params(axis='y',labelsize=10,labelleft='on')
            ax4.plot(lrg_snr_mag[1][select],lrg_snr_mag[0][select],'r.')

    ax5.set_ylabel(r'Median S/N')
    ax5.set_xlabel(r'Mag. (DECAM_R)')
    ax5.set_title(r'QSO')
    if len(qso_snr_mag[1]) > 0:  #- at least 1 qso fiber?
        select=np.where((qso_snr_mag[1] != np.array(None)) & (~np.isnan(qso_snr_mag[1])) & (np.abs(qso_snr_mag[1])!=np.inf))[0] #- Remove None, nan and inf values
        if select.shape[0]>0:

            xmin=np.min(qso_snr_mag[1][select])-0.1
            xmax=np.max(qso_snr_mag[1][select])+0.1
            ax5.set_xlim(xmin,xmax)
            ax5.set_ylim(np.min(qso_snr_mag[0][select])-0.1,np.max(qso_snr_mag[0][select])+0.1)
            ax5.xaxis.set_ticks(np.arange(int(np.min(qso_snr_mag[1][select])),int(np.max(qso_snr_mag[1][select]))+1,1.0))
            ax5.tick_params(axis='x',labelsize=10,labelbottom='on')
            ax5.tick_params(axis='y',labelsize=10,labelleft='on')
            ax5.plot(qso_snr_mag[1][select],qso_snr_mag[0][select],'g.')

    ax6.set_ylabel('')
    ax6.set_xlabel('Mag. (DECAM_R)')
    ax6.set_title(r'STD')
    if len(star_snr_mag[1]) > 0:  #- at least 1 std fiber?
        select=np.where((star_snr_mag[1] != np.array(None)) & (~np.isnan(star_snr_mag[1])) & (np.abs(star_snr_mag[1])!=np.inf))[0]
        if select.shape[0]>0:
            xmin=np.min(star_snr_mag[1][select])-0.1
            xmax=np.max(star_snr_mag[1][select])+0.1
            ax6.set_xlim(xmin,xmax)
            ax6.set_ylim(np.min(star_snr_mag[0][select])-0.1,np.max(star_snr_mag[0][select])+0.1)
            ax6.xaxis.set_ticks(np.arange(int(np.min(star_snr_mag[1][select])),int(np.max(star_snr_mag[1][select]))+1,0.5))
            ax6.tick_params(axis='x',labelsize=10,labelbottom='on')
            ax6.tick_params(axis='y',labelsize=10,labelleft='on')
            ax6.plot(star_snr_mag[1][select],star_snr_mag[0][select],'k.')

    """
    # Meta
    xlbl = 0.1
    ylbl = 0.85
    i0 = outfil.rfind('/')
    ax2.text(xlbl, ylbl, outfil[i0+1:], color='black', transform=ax2.transAxes, ha='left')
    yoff=0.15
    for key in sorted(qaframe.data['SKYSUB']['METRICS'].keys()):
        if key in ['QA_FIG']:
            continue
        # Show
        ylbl -= yoff
        ax2.text(xlbl+0.1, ylbl, key+': '+str(qaframe.data['SKYSUB']['METRICS'][key]),
            transform=ax2.transAxes, ha='left', fontsize='small')
    """


    '''
    # Residuals
    scatt_sz = 0.5
    ax_res = plt.subplot(gs[1])
    ax_res.get_xaxis().set_ticks([]) # Suppress labeling
    res = (sky_model - (true_flux*scl))/(true_flux*scl)
    rms = np.sqrt(np.sum(res**2)/len(res))
    #ax_res.set_ylim(-3.*rms, 3.*rms)
    ax_res.set_ylim(-2, 2)
    ax_res.set_ylabel('Frac Res')
    # Error
    #ax_res.plot(true_wave, 2.*ms_sig/sky_model, color='red')
    ax_res.scatter(wave,res, marker='o',s=scatt_sz)
    ax_res.plot([xmin,xmax], [0.,0], 'g-')
    ax_res.set_xlim(xmin,xmax)

    # Relative to error
    ax_sig = plt.subplot(gs[2])
    ax_sig.set_xlabel('Wavelength')
    sig_res = (sky_model - (true_flux*scl))/sky_sig
    ax_sig.scatter(wave, sig_res, marker='o',s=scatt_sz)
    ax_sig.set_ylabel(r'Res $\delta/\sigma$')
    ax_sig.set_ylim(-5., 5.)
    ax_sig.plot([xmin,xmax], [0.,0], 'g-')
    ax_sig.set_xlim(xmin,xmax)
    '''

    # Finish
    plt.tight_layout(pad=0.1,h_pad=0.0,w_pad=0.0)
    plt.savefig(outfil)
    plt.close()
    print('Wrote QA SkyRes file: {:s}'.format(outfil))



def exposure_fluxcalib(outfil, qa_data):
    """ QA plots for Flux calibration in an Exposure

    Args:
        outfil: str -- Name of PDF file
        qa_data: dict -- QA data, including that of the individual frames
    """
    # Init
    cameras = list(qa_data['frames'].keys())
    # Plot
    fig = plt.figure(figsize=(8, 5.0))
    gs = gridspec.GridSpec(2, 2)

    # Loop on channel
    clrs = dict(b='blue', r='red', z='purple')
    for qq, channel in enumerate(['b','r','z']):

        ax = plt.subplot(gs[qq % 2, qq // 2])
        allc = []
        for camera in cameras:
            if camera[0] == channel:
                allc.append(int(camera[1]))
                ax.errorbar([int(camera[1])],
                            [qa_data['frames'][camera]['FLUXCALIB']['METRICS']['ZP']],
                            yerr=[qa_data['frames'][camera]['FLUXCALIB']['METRICS']['RMS_ZP']],
                            capthick=2, fmt='o', color=clrs[channel])


    #
    #ax0.plot([xmin,xmax], [0., 0], '--', color='gray')
    #ax0.plot([xmin,xmax], [0., 0], '--', color='gray')
        ax.set_ylabel('ZP_AB')
        #import pdb; pdb.set_trace()
        ax.set_xlim(np.min(allc)-0.2, np.max(allc)+0.2)
        ax.set_xlabel('Spectrograph')
    #med0 = np.maximum(np.abs(np.median(med_res)), 1.)
    #ax0.set_ylim(-5.*med0, 5.*med0)
    #ax0.text(0.5, 0.85, 'Sky Meanspec',
    #    transform=ax_flux.transAxes, ha='center')

    # Meta text
    #ax2 = plt.subplot(gs[1,1])
    #ax2.set_axis_off()
    #show_meta(ax2, qaframe, 'FLUXCALIB', outfil)

    # Finish
    plt.tight_layout(pad=0.1,h_pad=0.0,w_pad=0.0)
    plt.savefig(outfil)
    plt.close()
    print('Wrote QA FluxCalib Exposure file: {:s}'.format(outfil))


def frame_fluxcalib(outfil, qaframe, frame, fluxcalib):
    """ QA plots for Flux calibration in a Frame

    Args:
        outfil: str, name of output file
        qaframe: dict containing QA info
        frame: frame object containing extraction of standard stars
        fluxcalib: fluxcalib object containing flux calibration

    Returns:
    """
    log = get_logger()

    # Standard stars
    #stdfibers = (frame.fibermap['OBJTYPE'] == 'STD')
    stdfibers = np.where(frame.fibermap['OBJTYPE'] == 'STD')[0]
    stdstars = frame[stdfibers]
    #nstds = np.sum(stdfibers)
    nstds = len(stdfibers)

    # Median spectrum
    medcalib = np.median(fluxcalib.calib[stdfibers],axis=0)
    ZP_AB = dsflux.ZP_from_calib(fluxcalib.wave, medcalib)


    # Plot
    fig = plt.figure(figsize=(8, 5.0))
    gs = gridspec.GridSpec(2,2)

    xmin,xmax = np.min(fluxcalib.wave), np.max(fluxcalib.wave)

    # Simple residual plot
    ax0 = plt.subplot(gs[0,:])
    #ax0.plot(frame.wave, signal.medfilt(med_res,51), color='black', label='Median**2 Res')
    #ax0.plot(frame.wave, signal.medfilt(wavg_res,51), color='red', label='Med WAvgRes')
    #ax_flux.plot(wave, sky_sig, label='Model Error')
    #ax_flux.plot(wave,true_flux*scl, label='Truth')
    #ax_flux.get_xaxis().set_ticks([]) # Suppress labeling

    #
    #ax0.plot([xmin,xmax], [0., 0], '--', color='gray')
    #ax0.plot([xmin,xmax], [0., 0], '--', color='gray')
    ax0.set_ylabel('ZP_AB')
    ax0.set_xlim(xmin, xmax)
    ax0.set_xlabel('Wavelength')
    #med0 = np.maximum(np.abs(np.median(med_res)), 1.)
    #ax0.set_ylim(-5.*med0, 5.*med0)
    #ax0.text(0.5, 0.85, 'Sky Meanspec',
    #    transform=ax_flux.transAxes, ha='center')

    # Other stars
    for ii in range(nstds):
        # Good pixels
        gdp = stdstars.ivar[ii, :] > 0.
        icalib = fluxcalib.calib[stdfibers[ii]][gdp]
        i_wave = fluxcalib.wave[gdp]
        ZP_star = dsflux.ZP_from_calib(i_wave, icalib)
        # Plot
        if ii == 0:
            lbl ='Individual stars'
        else:
            lbl = None
        ax0.plot(i_wave, ZP_star, ':', label=lbl)
    ax0.plot(fluxcalib.wave, ZP_AB, color='black', label='Median Calib')

    # Legend
    legend = ax0.legend(loc='lower left', borderpad=0.3,
                        handletextpad=0.3, fontsize='small')

    # Meta text
    ax2 = plt.subplot(gs[1,1])
    ax2.set_axis_off()
    show_meta(ax2, qaframe, 'FLUXCALIB', outfil)


    # Finish
    plt.tight_layout(pad=0.1,h_pad=0.0,w_pad=0.0)
    plt.savefig(outfil)
    plt.close()
    print('Wrote QA SkyRes file: {:s}'.format(outfil))


def frame_fiberflat(outfil, qaframe, frame, fiberflat):
    """ QA plots for fiber flat

    Args:
        outfil:
        qaframe:
        frame:
        fiberflat:

    Returns:
        Stuff?
    """
    # Setup
    fibermap = frame.fibermap
    gdp = fiberflat.mask == 0
    nfiber = len(frame.fibers)
    xfiber = np.zeros(nfiber)
    yfiber = np.zeros(nfiber)
    for ii,fiber in enumerate(frame.fibers):
        mt = np.where(fiber == fibermap['FIBER'])[0]
        xfiber[ii] = fibermap['X_TARGET'][mt]
        yfiber[ii] = fibermap['Y_TARGET'][mt]

    jet = cm = plt.get_cmap('jet')

    # Tile plot(s)
    fig = plt.figure(figsize=(8, 5.0))
    gs = gridspec.GridSpec(2,2)

    # Mean Flatfield flux in each fiber
    ax = plt.subplot(gs[0,0])
    ax.xaxis.set_major_locator(plt.MultipleLocator(100.))

    mean_flux = np.mean(frame.flux*gdp, axis=1)
    rms_mean = np.std(mean_flux)
    med_mean = np.median(mean_flux)
    #from xastropy.xutils import xdebug as xdb
    #pdb.set_trace()
    mplt = ax.scatter(xfiber, yfiber, marker='o', s=9., c=mean_flux, cmap=jet)
    mplt.set_clim(vmin=med_mean-2*rms_mean, vmax=med_mean+2*rms_mean)
    cb = fig.colorbar(mplt)
    cb.set_label('Mean Flux')

    # Mean
    ax = plt.subplot(gs[0,1])
    ax.xaxis.set_major_locator(plt.MultipleLocator(100.))
    mean_norm = np.mean(fiberflat.fiberflat*gdp,axis=1)
    m2plt = ax.scatter(xfiber, yfiber, marker='o', s=9., c=mean_norm, cmap=jet)
    m2plt.set_clim(vmin=0.98, vmax=1.02)
    cb = fig.colorbar(m2plt)
    cb.set_label('Mean of Fiberflat')

    # RMS
    ax = plt.subplot(gs[1,0])
    ax.xaxis.set_major_locator(plt.MultipleLocator(100.))
    rms = np.std(gdp*(fiberflat.fiberflat-
                      np.outer(mean_norm, np.ones(fiberflat.nwave))),axis=1)
    rplt = ax.scatter(xfiber, yfiber, marker='o', s=9., c=rms, cmap=jet)
    #rplt.set_clim(vmin=0.98, vmax=1.02)
    cb = fig.colorbar(rplt)
    cb.set_label('RMS in Fiberflat')

    # Meta text
    ax2 = plt.subplot(gs[1,1])
    ax2.set_axis_off()
    show_meta(ax2, qaframe, 'FIBERFLAT', outfil)
    """
    xlbl = 0.05
    ylbl = 0.85
    i0 = outfil.rfind('/')
    ax2.text(xlbl, ylbl, outfil[i0+1:], color='black', transform=ax2.transAxes, ha='left')
    yoff=0.10
    for key in sorted(qaframe.data['FIBERFLAT']['METRICS'].keys()):
        if key in ['QA_FIG']:
            continue
        # Show
        ylbl -= yoff
        ax2.text(xlbl+0.05, ylbl, key+': '+str(qaframe.data['FIBERFLAT']['METRICS'][key]),
            transform=ax2.transAxes, ha='left', fontsize='x-small')
    """

    # Finish
    plt.tight_layout(pad=0.1,h_pad=0.0,w_pad=0.0)
    plt.savefig(outfil)
    plt.close()
    print('Wrote QA SkyRes file: {:s}'.format(outfil))


def show_meta(ax, qaframe, qaflavor, outfil):
    """ Show meta data on the figure

    Args:
        ax: matplotlib.ax
        qaframe: QA_Frame
        qaflavor: str

    Returns:
    """
    # Meta
    xlbl = 0.05
    ylbl = 0.85
    i0 = outfil.rfind('/')
    ax.text(xlbl, ylbl, outfil[i0+1:], color='black', transform=ax.transAxes, ha='left')
    yoff=0.10
    for key in sorted(qaframe.qa_data[qaflavor]['METRICS'].keys()):
        if key in ['QA_FIG']:
            continue
        # Show
        ylbl -= yoff
        ax.text(xlbl+0.1, ylbl, key+': '+str(qaframe.qa_data[qaflavor]['METRICS'][key]),
            transform=ax.transAxes, ha='left', fontsize='x-small')


def get_sty_otype():
    """Styles for plots"""
    sty_otype = dict(ELG={'color':'green', 'lbl':'ELG'},
                     LRG={'color':'red', 'lbl':'LRG'},
                     STAR={'color':'black', 'lbl':'STAR'},
                     QSO={'color':'blue', 'lbl':'QSO'},
                     QSO_L={'color':'blue', 'lbl':'QSO z>2.1'},
                     QSO_T={'color':'cyan', 'lbl':'QSO z<2.1'})
    return sty_otype


def prod_channel_hist(qa_prod, qatype, metric, xlim=None, outfile=None, pp=None, close=True):
    """ Generate a series of histrograms (one per channel)

    Args:
        qa_prod: QA_Prod class
        qatype: str
        metric: str
        xlim: tuple, optional
        outfile: str, optional
        pp: PdfPages, optional
        close: bool, optional

    Returns:

    """
    log = get_logger()
    # Setup
    fig = plt.figure(figsize=(8, 5.0))
    gs = gridspec.GridSpec(2,2)

    # Loop on channel
    clrs = dict(b='blue', r='red', z='purple')
    for qq, channel in enumerate(['b', 'r', 'z']):
        ax = plt.subplot(gs[qq])
        #ax.xaxis.set_major_locator(plt.MultipleLocator(100.))

        # Grab QA
        qa_arr, ne_dict = qa_prod.get_qa_array(qatype, metric, channels=channel)
        # Check for nans
        isnan = np.isnan(qa_arr)
        if np.sum(isnan) > 0:
            log.error("NAN in qatype={:s}, metric={:s} for channel={:s}".format(
                qatype, metric, channel))
            qa_arr[isnan] = -999.
        # Histogram
        ax.hist(qa_arr, color=clrs[channel])
        #import pdb; pdb.set_trace()
        # Label
        ax.text(0.05, 0.85, channel, color='black', transform=ax.transAxes, ha='left')
        ax.set_xlabel('{:s} :: {:s}'.format(qatype,metric))
        if xlim is not None:
            ax.set_xlim(xlim)

    # Meta
    ax = plt.subplot(gs[3])
    ax.set_axis_off()
    xlbl = 0.05
    ylbl = 0.85
    yoff = 0.1
    ax.text(xlbl, ylbl, qa_prod.prod_name, color='black', transform=ax.transAxes, ha='left')
    nights = list(ne_dict.keys())
    #
    ylbl -= yoff
    ax.text(xlbl+0.1, ylbl, 'Nights: {}'.format(nights),
            transform=ax.transAxes, ha='left', fontsize='x-small')
    #
    ylbl -= yoff
    expids = []
    for night in nights:
        expids += ne_dict[night]
    ax.text(xlbl+0.1, ylbl, 'Exposures: {}'.format(expids),
            transform=ax.transAxes, ha='left', fontsize='x-small')

    # Finish
    plt.tight_layout(pad=0.1,h_pad=0.0,w_pad=0.0)
    if outfile is not None:
        plt.savefig(outfile)
        if close:
            plt.close()
    elif pp is not None:
        pp.savefig()
        if close:
            plt.close()
            pp.close()
    else:  # Show
        plt.show()


def skysub_resid_dual(sky_wave, sky_flux, sky_res, outfile=None, pp=None, close=True, nslices=20):
    """ Generate a plot of sky subtraction residuals
    Typically for a given channel
    Args:
        wave:
        sky_flux:
        sky_res:
        outfile:
        pp:
        close:

    Returns:

    """
    # Start the plot
    fig = plt.figure(figsize=(8, 5.0))
    gs = gridspec.GridSpec(2,1)

    # Wavelength
    ax_wave = plt.subplot(gs[0])
    du_pslices(sky_wave, sky_res, np.min(sky_wave), np.max(sky_wave),
               0., num_slices=nslices, axis=ax_wave)
    ax_wave.set_xlabel('Wavelength')
    ax_wave.set_ylabel('Residual Flux')

    # Wavelength
    ax_flux = plt.subplot(gs[1])
    du_pslices(sky_flux, sky_res, np.min(sky_flux), np.max(sky_flux),
               0., num_slices=nslices, axis=ax_flux, set_ylim_from_stats=True)
    ax_flux.set_xlabel('log10(Sky Flux)')
    ax_flux.set_ylabel('Residual Flux')
    #ax_flux.set_ylim(-600, 100)


    # Finish
    plt.tight_layout(pad=0.1,h_pad=0.0,w_pad=0.0)
    if outfile is not None:
        plt.savefig(outfile)
        if close:
            plt.close()
    elif pp is not None:
        pp.savefig()
        if close:
            plt.close()
            pp.close()
    else:  # Show
        plt.show()

def skysub_resid_series(sky_dict, xtype, outfile=None, pp=None,
                        close=True, nslices=20, dpi=1000):
    """ Generate a plot of sky subtraction residuals for a series of inputs
    Typically for a given channel
    Args:
        wave:
        sky_flux:
        sky_res:
        outfile:
        pp:
        close:

    Returns:

    """
    # Start the plot
    fig = plt.figure(figsize=(8, 5.0))
    gs = gridspec.GridSpec(sky_dict['count'],1)

    for kk in range(sky_dict['count']):
        sky_wave = sky_dict['wave'][kk]
        sky_res = sky_dict['res'][kk]
        sky_flux = sky_dict['skyflux'][kk]
        ax = plt.subplot(gs[kk])
        #ax.set_ylabel('Residual Flux')
        if xtype == 'wave': # Wavelength
            du_pslices(sky_wave, sky_res, np.min(sky_wave), np.max(sky_wave),
               0., num_slices=nslices, axis=ax)
            xlbl = 'Wavelength'
        elif xtype == 'flux': # Flux
            xlbl = 'log10(Sky Flux)'
            du_pslices(sky_flux, sky_res, np.min(sky_flux), np.max(sky_flux),
               0., num_slices=nslices, axis=ax, set_ylim_from_stats=True)
            if kk == sky_dict['count']-1:
                ax.set_xlabel('Wavelength')
            else:
                ax.get_xaxis().set_ticks([])
        if kk == sky_dict['count']-1:
            ax.set_xlabel(xlbl)
        else:
            ax.get_xaxis().set_ticks([])

    # Finish
    plt.tight_layout(pad=0.1,h_pad=0.0,w_pad=0.0)
    if outfile is not None:
        plt.savefig(outfile, dpi=dpi)
        if close:
            plt.close()
    elif pp is not None:
        pp.savefig()
        if close:
            plt.close()
            pp.close()
    else:  # Show
        plt.show()
