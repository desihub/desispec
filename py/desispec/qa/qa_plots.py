""" Module for QA plots
"""
from __future__ import print_function, absolute_import, division

import os
import numpy as np
from scipy import signal
import scipy
import scipy.stats
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
from desispec.io import makepath
from desispec.fluxcalibration import isStdStar

from desiutil import plots as desiu_p

from desispec.io import read_params
desi_params = read_params()


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



def frame_skyres(outfil, frame, skymodel, qaframe, quick_look=False):
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
    from desispec.sky import subtract_sky

    # Access metrics
    '''
    wavg_ivar = np.sum(res_ivar,0)
    chi2_wavg = np.sum(wavg_res**2 * wavg_ivar)
    dof_wavg = np.sum(wavg_ivar > 0.)
    pchi2_wavg = scipy.stats.distributions.chi2.sf(chi2_wavg, dof_wavg)
    chi2_med = np.sum(med_res**2 * wavg_ivar)
    pchi2_med = scipy.stats.distributions.chi2.sf(chi2_med, dof_wavg)
    '''
    skyfibers = np.array(qaframe.qa_data['SKYSUB']["METRICS"]["SKYFIBERID"])
    subtract_sky(frame, skymodel)
    res=frame.flux[skyfibers]
    res_ivar=frame.ivar[skyfibers]
    if quick_look:
        med_res = qaframe.qa_data['SKYSUB']["METRICS"]["MED_RESID_WAVE"]
        wavg_res = qaframe.qa_data['SKYSUB']["METRICS"]["WAVG_RES_WAVE"]
    else:
        med_res = np.median(res,axis=0)
        wavg_res = np.sum(res*res_ivar,0) / np.sum(res_ivar,0)

    # Plot
    if quick_look:
        fig = plt.figure(figsize=(8, 10.0))
        gs = gridspec.GridSpec(4,2)
    else:
        fig = plt.figure(figsize=(8, 6.0))
        gs = gridspec.GridSpec(2,2)
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
    if 'DEVS_1D' in qaframe.qa_data['SKYSUB']["METRICS"].keys(): # Online
        hist = np.asarray(qaframe.qa_data['SKYSUB']["METRICS"]["DEVS_1D"])
        edges = np.asarray(qaframe.qa_data['SKYSUB']["METRICS"]["DEVS_EDGES"])
    else: # Generate for offline
        gd_res = res_ivar > 0.
        devs = res[gd_res] * np.sqrt(res_ivar[gd_res])
        i0, i1 = int( np.min(devs) / binsz) - 1, int( np.max(devs) / binsz) + 1
        rng = tuple( binsz*np.array([i0,i1]) )
        nbin = i1-i0
        hist, edges = np.histogram(devs, range=rng, bins=nbin)

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

    if quick_look:
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
    outfile = makepath(outfil)
    plt.savefig(outfil)
    plt.close()
    print('Wrote QA SkyRes file: {:s}'.format(outfil))

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
    exptime = frame.meta['EXPTIME']
    stdfibers = np.where(isStdStar(frame.fibermap['DESI_TARGET']))[0]
    stdstars = frame[stdfibers]
    #nstds = np.sum(stdfibers)
    nstds = len(stdfibers)

    # Median spectrum
    medcalib = np.median(fluxcalib.calib[stdfibers],axis=0)
    ZP_AB = dsflux.ZP_from_calib(exptime, fluxcalib.wave, medcalib)


    # Plot
    fig = plt.figure(figsize=(8, 5.0))
    gs = gridspec.GridSpec(2,2)

    xmin,xmax = np.min(fluxcalib.wave), np.max(fluxcalib.wave)

    # Simple residual plot
    ax0 = plt.subplot(gs[0,:])

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
        ZP_star = dsflux.ZP_from_calib(exptime, i_wave, icalib)
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
    _ = makepath(outfil)
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
    _ = makepath(outfil)
    plt.savefig(outfil)
    plt.close()
    print('Wrote QA FluxCalib Exposure file: {:s}'.format(outfil))

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
    from desimodel.focalplane import fiber_area_arcsec2
    # Setup
    fibermap = frame.fibermap
    gdp = fiberflat.mask == 0
    nfiber = len(frame.fibers)
    xfiber = np.zeros(nfiber)
    yfiber = np.zeros(nfiber)
    for ii,fiber in enumerate(frame.fibers):
        mt = np.where(fiber == fibermap['FIBER'])[0]
        xfiber[ii] = fibermap['DESIGN_X'][mt]
        yfiber[ii] = fibermap['DESIGN_Y'][mt]
    area = fiber_area_arcsec2(xfiber,yfiber)
    mean_area = np.mean(area)

    jet = cm = plt.get_cmap('jet')

    # Tile plot(s)
    fig = plt.figure(figsize=(8, 5.0))
    gs = gridspec.GridSpec(2,2)

    # Mean Flatfield flux in each fiber
    ax = plt.subplot(gs[0,0])
    ax.xaxis.set_major_locator(plt.MultipleLocator(100.))

    mean_flux = np.mean(frame.flux*gdp, axis=1) / fiber_area_arcsec2(xfiber,yfiber)
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
    mean_norm = np.mean(fiberflat.fiberflat*gdp,axis=1) / (area/mean_area)
    m2plt = ax.scatter(xfiber, yfiber, marker='o', s=9., c=mean_norm, cmap=jet)
    #m2plt.set_clim(vmin=0.98, vmax=1.02)
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
    _ = makepath(outfil)
    plt.savefig(outfil)
    plt.close()
    print('Wrote QA SkyRes file: {:s}'.format(outfil))

def exposure_fiberflat(channel, expid, metric, outfile=None):
    """ Generate an Exposure level plot of a FiberFlat metric
    Args:
        channel: str, e.g. 'b', 'r', 'z'
        expid: int
        metric: str,  allowed entires are: ['meanflux']

    Returns:

    """
    from desispec.io.meta import find_exposure_night, findfile
    from desispec.io.frame import read_meta_frame, read_frame
    from desispec.io.fiberflat import read_fiberflat
    from desimodel.focalplane import fiber_area_arcsec2
    log = get_logger()
    # Find exposure
    night = find_exposure_night(expid)
    # Search for frames with the input channel
    frame0 = findfile('frame', camera=channel+'0', night=night, expid=expid)
    if not os.path.exists(frame0):
        log.fatal("No Frame 0 for channel={:s} and expid={:d}".format(channel, expid))
    # Confirm frame is a Flat
    fmeta = read_meta_frame(frame0)
    assert fmeta['FLAVOR'].strip() == 'flat'
    # Load up all the frames
    x,y,metrics = [],[],[]
    for wedge in range(10):
        # Load
        frame_file = findfile('frame', camera=channel+'{:d}'.format(wedge), night=night, expid=expid)
        fiber_file = findfile('fiberflat', camera=channel+'{:d}'.format(wedge), night=night, expid=expid)
        try:
            frame = read_frame(frame_file)
        except:
            continue
        else:
            fiberflat = read_fiberflat(fiber_file)
        fibermap = frame.fibermap
        gdp = fiberflat.mask == 0
        # X,Y
        x.append([fibermap['DESIGN_X']])
        y.append([fibermap['DESIGN_Y']])
        area = fiber_area_arcsec2(x[-1], y[-1])
        mean_area = np.mean(area)
        # Metric
        if metric == 'meanflux':
            mean_norm = np.mean(fiberflat.fiberflat*gdp,axis=1) / (area / mean_area)
            metrics.append([mean_norm])
    # Cocatenate
    x = np.concatenate(x)
    y = np.concatenate(y)
    metrics = np.concatenate(metrics)
    # Plot
    if outfile is None:
        outfile='qa_{:08d}_{:s}_fiberflat.png'.format(expid, channel)
    exposure_map(x,y,metrics, mlbl='Mean Flux',
                 title='Mean Flux for Exposure {:08d}, Channel {:s}'.format(expid, channel),
                 outfile=outfile)


def exposure_map(x,y,metric,mlbl=None, outfile=None, title=None,
                 ax=None, fig=None, psz=9., cmap=None, vmnx=None):
    """ Generic method used to generated Exposure level QA
    One channel at a time

    Args:
        x: list or ndarray
        y: list or ndarray
        metric: list or ndarray
        mlbl: str, optional
        outfile: str, optional
        title: str, optional
    """
    # Tile plot(s)
    if ax is None:
        fig = plt.figure(figsize=(8, 5.0))
        gs = gridspec.GridSpec(1,1)
        ax = plt.subplot(gs[0])
    #
    if cmap is None:
        cmap = plt.get_cmap('jet')
    if mlbl is None:
        mlbl = 'Metric'

    # Mean Flatfield flux in each fiber
    ax.set_aspect('equal', 'datalim')
    if title is not None:
        ax.set_title(title)

    mplt = ax.scatter(x,y,marker='o', s=psz, c=metric.reshape(x.shape), cmap=cmap)
    #mplt.set_clim(vmin=med_mean-2*rms_mean, vmax=med_mean+2*rms_mean)
    if fig is not None:
        cb = fig.colorbar(mplt)
        cb.set_label(mlbl)
        #
        if vmnx is not None:
            mplt.set_clim(vmin=vmnx[0], vmax=vmnx[1])

    # Finish
    plt.tight_layout(pad=0.1,h_pad=0.0,w_pad=0.0)
    if outfile is not None:
        _ = makepath(outfile)
        plt.savefig(outfile)
        print('Wrote QA SkyRes file: {:s}'.format(outfile))
        plt.close()


def exposure_s2n(qa_exp, metric, outfile='exposure_s2n.png', verbose=True,
                 mag_mnx=[18.,22.]):
    """ Generate an Exposure level plot of a S/N metric
    Args:
        qa_exp: QA_Exposure
        metric: str,  allowed entires are: ['resid']
        mag_mnx: Range of magnitudes used for residual plot

    Returns:

    """
    from desispec.io.meta import find_exposure_night, findfile
    from desispec.io.frame import read_meta_frame, read_frame
    from desispec.io.fiberflat import read_fiberflat
    from desispec.qa.qalib import s2n_funcs
    log = get_logger()

    cclrs = get_channel_clrs()

    # Find exposure
    night = find_exposure_night(qa_exp.expid)


    # Plot
    fig = plt.figure(figsize=(8, 5.0))
    gs = gridspec.GridSpec(6,2)
    cmap = plt.get_cmap('RdBu')

    # Load up all the frames
    for ss,channel in enumerate(['b','r','z']):
        # ax
        if channel != 'z':
            ax = plt.subplot(gs[0:3, ss])
        else:
            ax = plt.subplot(gs[-3:, 1])

        x,y,metrics,sv_s2n,sv_mags = [],[],[], [],[]
        for wedge in range(10):
            # Load
            camera=channel+'{:d}'.format(wedge)
            frame_file = findfile('frame', camera=camera, night=night, expid=qa_exp.expid)
            try:
                frame = read_frame(frame_file)
            except:
                continue
            fibermap = frame.fibermap
            # X,Y
            x += [fibermap['DESIGN_X'].flatten()]
            y += [fibermap['DESIGN_Y'].flatten()]
            # Metric
            if metric == 'resid':
                # Setup
                s2n_dict = qa_exp.data['frames'][camera]['S2N']
                med_snr = np.array(s2n_dict['METRICS']['MEDIAN_SNR'])
                funcMap = s2n_funcs(exptime=s2n_dict['METRICS']['EXPTIME'],
                                    r2=s2n_dict['METRICS']['r2'])
                fitfunc = funcMap['astro']
                sci_idx = s2n_dict['METRICS']['OBJLIST'].index('SCIENCE')
                coeff = s2n_dict['METRICS']['FITCOEFF_TGT'][sci_idx]
                all_mags = np.resize(np.array(s2n_dict['METRICS']['MAGNITUDES']), (500, 3))
                fidx = np.where(np.array(s2n_dict['METRICS']['FILTERS']) == s2n_dict['METRICS']['FIT_FILTER'])[0]
                mags = all_mags[:, fidx].flatten()
                gd_mag = np.isfinite(mags)

                # Need to restrict to Science
                science_ids = np.array(s2n_dict['METRICS']['SCIENCE_FIBERID'])
                gd_type = np.zeros_like(gd_mag, dtype=bool)
                gd_type[science_ids] = True

                # Second mag cut
                gd_mag2 = (mags > mag_mnx[0]) & (mags < mag_mnx[1])

                # Synthesize
                gd_resid = gd_mag & gd_type & gd_mag2

                # Residuals
                flux = 10 ** (-0.4 * (mags[gd_resid] - 22.5))
                fit_snr = fitfunc(flux, *coeff)
                resid = (med_snr[gd_resid] - fit_snr) / fit_snr

                all_resid = np.zeros_like(mags)
                all_resid[gd_resid] = resid

                # Save
                metrics += [all_resid]
                sv_mags += [mags[gd_mag]]
                sv_s2n += [med_snr[gd_mag]]
        # Concatenate
        x = np.concatenate(x)
        y = np.concatenate(y)
        metrics = np.concatenate(metrics)

        # Exposure
        exposure_map(x,y,metrics, mlbl='S/N '+metric, ax=ax, fig=fig,
                 title=None, outfile=None, psz=1., cmap=cmap, vmnx=[-0.9,0.9])
        # Label
        ax.text(0.05, 0.9, channel, color=cclrs[channel], transform=ax.transAxes, ha='left')

        # Scatter + fit
        ax_summ = plt.subplot(gs[-3+ss,0])
        ax_summ.scatter(np.concatenate(sv_mags), np.concatenate(sv_s2n), color=cclrs[channel], s=1.)
        if ss < 2:
            ax_summ.get_xaxis().set_ticks([])
        # Axes
        ax_summ.set_yscale('log', nonposy='clip')
        if ss == 1:
            ax_summ.set_ylabel('S/N')

    # Finish
    plt.tight_layout(pad=0.1,h_pad=0.0,w_pad=0.0)
    _ = makepath(outfile)
    plt.savefig(outfile)
    if verbose:
        print("Wrote: {:s}".format(outfile))
    plt.close()




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
    yoff=0.10
    i0 = outfil.rfind('/')
    ax.text(xlbl, ylbl, outfil[i0+1:], color='black', transform=ax.transAxes, ha='left')
    # Night
    ylbl -= yoff
    ax.text(xlbl+0.1, ylbl, 'Night: '+qaframe.night,
            transform=ax.transAxes, ha='left', fontsize='x-small')
    # Rest
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
    clrs = get_channel_clrs()
    for qq, channel in enumerate(['b', 'r', 'z']):
        ax = plt.subplot(gs[qq])
        #ax.xaxis.set_major_locator(plt.MultipleLocator(100.))

        # Grab QA
        qa_tbl = qa_prod.get_qa_table(qatype, metric, channels=channel)
        # Check for nans
        qa_arr = qa_tbl[metric]
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
    '''
    ax = plt.subplot(gs[3])
    ax.set_axis_off()
    xlbl = 0.05
    ylbl = 0.85
    yoff = 0.1
    ax.text(xlbl, ylbl, qa_prod.prod_name, color='black', transform=ax.transAxes, ha='left')
    nights = list(qa_tbl['NIGHT'])
    #
    ylbl -= yoff
    ax.text(xlbl+0.1, ylbl, 'Nights: {}'.format(nights),
            transform=ax.transAxes, ha='left', fontsize='x-small')
    #
    ylbl -= yoff
    expids = list(qa_tbl['EXPID'])
    ax.text(xlbl+0.1, ylbl, 'Exposures: {}'.format(expids),
            transform=ax.transAxes, ha='left', fontsize='x-small')
    '''

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

def prod_time_series(qa_prod, qatype, metric, xlim=None, outfile=None, close=True, pp=None,
                     bright_dark=0):
    """ Generate a time series plot for a production
    Args:
        qa_prod:
        qatype:
        metric:
        xlim:
        outfile:
        close:
        pp:
        bright_dark: int, optional; (flag: 0=all; 1=bright; 2=dark)

    Returns:

    """
    from astropy.time import Time

    log = get_logger()

    # Setup
    fig = plt.figure(figsize=(8, 5.0))
    gs = gridspec.GridSpec(3,1)


    # Loop on channel
    clrs = get_channel_clrs()

    # Grab QA
    all_times = []
    all_ax = []
    for cc, channel in enumerate(['b','r','z']):
        ax = plt.subplot(gs[cc])
        qa_tbl = qa_prod.get_qa_table(qatype, metric, channels=channel)
        '''
        # Check for nans
        isnan = np.isnan(qa_arr)
        if np.sum(isnan) > 0:
            log.error("NAN in qatype={:s}, metric={:s} for channel={:s}".format(
                qatype, metric, channel))
            qa_arr[isnan] = -999.
        '''
        # Convert Date to MJD
        atime = Time(qa_tbl['DATE-OBS'], format='isot', scale='utc')
        atime.format = 'mjd'
        mjd = atime.value

        # Bright dark
        if bright_dark == 0: # All
            pass
        elif bright_dark == 1: # Bright
            log.info("Using a bright/dark kludge for now")
            bright = qa_tbl['EXPTIME'] < 1200.
            qa_tbl = qa_tbl[bright]
            mjd = mjd[bright]
        elif bright_dark == 2: # Dark
            log.info("Using a bright/dark kludge for now")
            dark = qa_tbl['EXPTIME'] > 1200.
            qa_tbl = qa_tbl[dark]
            mjd = mjd[dark]

        # Scatter me
        ax.scatter(mjd, qa_tbl[metric], color=clrs[channel], s=4.)
        # Axes
        ax.set_ylabel('Metric')
        if cc < 2:
            ax.get_xaxis().set_ticks([])
        if cc ==0:
            ax.set_title('{:s} :: {:s}'.format(qatype,metric))
        all_times.append(mjd)
        all_ax.append(ax)

    # Label
    #ax.text(0.05, 0.85, channel, color='black', transform=ax.transAxes, ha='left')
    ax.set_xlabel('MJD')
    all_times = np.concatenate(all_times)
    xmin, xmax = np.min(all_times), np.max(all_times)
    for cc in range(3):
        all_ax[cc].set_xlim(xmin,xmax)


    # Finish
    plt.tight_layout(pad=0.1,h_pad=0.0,w_pad=0.0)
    if outfile is not None:
        plt.savefig(outfile)
        print("Wrote QA file: {:s}".format(outfile))
        if close:
            plt.close()
    elif pp is not None:
        pp.savefig()
        if close:
            plt.close()
            pp.close()
    else:  # Show
        plt.show()


def skyline_resid(channel, sky_wave, sky_flux, sky_res, sky_ivar, outfile=None, pp=None,
                   close=True, dpi=700):
    """ QA plot for residuals on sky lines
    ala Julien Guy
    Args:
        sky_wave:
        sky_flux:
        sky_res:
        outfile:
        pp:
        close:
        nslices:
        dpi:

    Returns:

    """
    # Grab the sky lines
    sky_peaks = desi_params['qa']['skypeaks']['PARAMS']['{:s}_PEAKS'.format(channel.upper())]
    npeaks = len(sky_peaks)

    # Collapse the sky data
    #sky_wave = np.median(sky_wave, axis=0)
    #sky_res = np.median(sky_res, axis=0)
    #sky_ivar = np.median(sky_ivar, axis=0)
    #sky_flux = np.median(sky_flux, axis=0)

    # Start the plot
    fig = plt.figure(figsize=(8, 5.0))
    gs = gridspec.GridSpec(npeaks, 1)

    wv_off = 15.

    clrs = dict(b='b', r='r', z='purple')

    # Loop on peaks
    for ss,peak in enumerate(sky_peaks):
        ax= plt.subplot(gs[ss])

        # Zoom in
        pix = np.abs(sky_wave[0,:]-peak) < wv_off

        # Calculate
        orig = np.sqrt(np.mean(sky_ivar[:,pix] * sky_res[:,pix]**2, axis=0))
        lbl=r"$\sqrt{ <flux^2/\sigma^2>}$"
        if ss > 0:
            lbl = None
        ax.plot(sky_wave[0,pix], orig, color=clrs[channel], label=lbl)

        # Sky scaling
        #lbl=r"$\sqrt{ < 1+(0.05 sky)^2/\sigma^2 > }$"

        # Labels
        ax.set_ylabel(r'$n \sigma$')
        #ax_flux.set_ylabel('Residual Flux')
        ax.set_ylim(bottom=0.)
        ax.axhline(1., color='gray', linestyle='dashed')

        if ss == 0:
            legend = ax.legend(loc='upper left', borderpad=0.3,
                        handletextpad=0.3, fontsize='small')

    # Finish
    plt.tight_layout(pad=0.1, h_pad=0.0, w_pad=0.0)
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

def skysub_resid_dual(sky_wave, sky_flux, sky_res, outfile=None, pp=None,
                      close=True, nslices=20, dpi=700):
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
    desiu_p.plot_slices(sky_wave, sky_res, np.min(sky_wave), np.max(sky_wave),
               0., num_slices=nslices, axis=ax_wave, scatter=False)
    ax_wave.set_xlabel('Wavelength')
    ax_wave.set_ylabel('Residual Flux')

    # Wavelength
    ax_flux = plt.subplot(gs[1])
    desiu_p.plot_slices(sky_flux, sky_res, np.min(sky_flux), np.max(sky_flux),
               0., num_slices=nslices, axis=ax_flux, set_ylim_from_stats=True, scatter=False)
    ax_flux.set_xlabel('log10(Sky Flux)')
    ax_flux.set_ylabel('Residual Flux')
    #ax_flux.set_ylim(-600, 100)


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

def skysub_resid_series(sky_dict, xtype, outfile=None, pp=None,
                        close=True, nslices=20, dpi=700):
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
            desiu_p.plot_slices(sky_wave, sky_res, np.min(sky_wave), np.max(sky_wave),
               0., num_slices=nslices, axis=ax, scatter=False)
            xlbl = 'Wavelength'
        elif xtype == 'flux': # Flux
            xlbl = 'log10(Sky Flux)'
            desiu_p.plot_slices(sky_flux, sky_res, np.min(sky_flux), np.max(sky_flux),
               0., num_slices=nslices, axis=ax, set_ylim_from_stats=True, scatter=False)
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

def skysub_gauss(sky_wave, sky_flux, sky_res, sky_ivar, outfile=None, pp=None,
                      close=True, binsz=0.1, dpi=700, nfbin=4):
    """ Generate a plot examining the Gaussianity of the residuals
    Typically for a given channel
    Args:
        wave:
        sky_flux:
        sky_res:
        sky_ivar:
        outfile:
        pp:
        close:

    Returns:

    """
    from scipy.stats import norm
    # Deviates
    gd_res = sky_ivar > 0.
    devs = sky_res[gd_res] * np.sqrt(sky_ivar[gd_res])

    # Start the plot
    fig = plt.figure(figsize=(8, 4.0))
    gs = gridspec.GridSpec(1,2)

    # Histogram :: Same routine as in frame_skyresid
    ax0 = plt.subplot(gs[0])
    i0, i1 = int( np.min(devs) / binsz) - 1, int( np.max(devs) / binsz) + 1
    rng = tuple(binsz*np.array([i0,i1]) )
    nbin = i1-i0
    hist, edges = np.histogram(devs, range=rng, bins=nbin)

    xhist = (edges[1:] + edges[:-1])/2.
    ax0.hist(xhist, color='blue', bins=edges, weights=hist)#, histtype='step')
    # PDF for Gaussian
    area = binsz * np.sum(hist)

    xppf = np.linspace(scipy.stats.norm.ppf(0.000001), scipy.stats.norm.ppf(0.999999), 10000)
    ax0.plot(xppf, area*scipy.stats.norm.pdf(xppf), 'r-', alpha=1.0)
    ax0.set_xlabel(r'Res/$\sigma$')
    ax0.set_ylabel('N')

    # Deviates vs. flux
    absdevs = np.abs(devs)
    asrt = np.argsort(absdevs)
    absdevs.sort()
    ndev = devs.size
    ax1 = plt.subplot(gs[1])

    # All
    xlim = (0., np.max(absdevs))
    ylim = (0.000001, 1.)
    ax1.plot(absdevs, 1-np.arange(ndev)/(ndev-1), 'k', label='All')

    # Bin by sky flux
    sflux = sky_flux[asrt]
    sky_flux.sort()
    fbins = [0.] + [sky_flux[int(ii*ndev/nfbin)] for ii in range(1,nfbin)]
    fbins += [np.max(sky_flux)]
    # Adjust last bin to be likely on sky lines
    if np.max(sky_flux) > 2000.:  # Am assuming 2000 counts is a skyline
        fbins[-2] = max(2000., fbins[-2])
    # Digitize
    f_i = np.digitize(sflux, fbins) - 1

    for kk in range(nfbin):
        lbl = 'flux = [{:d},{:d}]'.format(int(fbins[kk]),int(fbins[kk+1]))
        idx = f_i == kk
        ncut = np.sum(idx)
        ax1.plot(absdevs[idx], 1-np.arange(ncut)/(ncut-1), '--', label=lbl)

    # Gauss lines
    for kk in range(1,int(xlim[1])+1):
        ax1.plot([kk]*2, ylim, ':', color='gray')
        icl = norm.cdf(kk) - norm.cdf(-1*kk)  # Area under curve
        ax1.plot(xlim, [1-icl]*2, ':', color='gray')
        ax1.text(0.2, 1-icl, '{:d}'.format(kk)+r'$\sigma$', color='gray')

    ax1.set_xlabel(r'Res/$\sigma$')
    ax1.set_ylabel(r'Fraction greater than Res/$\sigma$')
    ax1.set_yscale("log", nonposy='clip')
    ax1.set_ylim(ylim)

    legend = ax1.legend(loc='lower left', borderpad=0.3,
                        handletextpad=0.3, fontsize='small')


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


def get_channel_clrs():
    """ Simple dict to organize styles for channels
    Returns:
        channel_dict: dict
    """
    return dict(b='blue', r='red', z='purple')
