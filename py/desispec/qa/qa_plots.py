"""
Module for QA plots
"""
from __future__ import print_function, absolute_import, division, unicode_literals

import numpy as np
from scipy import signal
import scipy
import pdb

from desispec import fluxcalibration as dsflux

import matplotlib
matplotlib.use('Agg') 
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

from desispec import util


def brick_zbest(outfil, zf, qabrick):
    """ QA plots for Flux calibration in a Frame
    Args:
        outfil:
        qabrick:
        zf: ZfindBase object

    Returns:

    """
    sty_otype = get_sty_otype()
    # Convert types (this should become obsolete)
    param = qabrick.data['ZBEST']['PARAM']
    zftypes = []
    for ztype in zf.type:
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

    for key in sty_otype.keys():
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

    Parameters:
    -------------
    outfil: str
      Name of output file
    frame: Frame object 
    skymodel: SkyModel object
    qaframe: QAFrame object
    """

    # Sky fibers
    skyfibers = np.where(frame.fibermap['OBJTYPE'] == 'SKY')[0]
    assert np.max(skyfibers) < 500  #- indices, not fiber numbers

    # Residuals
    res = frame.flux[skyfibers] - skymodel.flux[skyfibers] # Residuals
    res_ivar = util.combine_ivar(frame.ivar[skyfibers], skymodel.ivar[skyfibers]) 
    med_res = np.median(res,0)

    # Deviates
    gd_res = res_ivar > 0.
    devs = res[gd_res] * np.sqrt(res_ivar[gd_res])

    # Calculations
    wavg_res = np.sum(res*res_ivar,0) / np.sum(res_ivar,0)
    '''
    wavg_ivar = np.sum(res_ivar,0)
    chi2_wavg = np.sum(wavg_res**2 * wavg_ivar)
    dof_wavg = np.sum(wavg_ivar > 0.)
    pchi2_wavg = scipy.stats.chisqprob(chi2_wavg, dof_wavg)
    chi2_med = np.sum(med_res**2 * wavg_ivar)
    pchi2_med = scipy.stats.chisqprob(chi2_med, dof_wavg)
    '''

    # Plot
    fig = plt.figure(figsize=(8, 5.0))
    gs = gridspec.GridSpec(2,2)

    xmin,xmax = np.min(frame.wave), np.max(frame.wave)

    # Simple residual plot
    ax0 = plt.subplot(gs[0,:])
    ax0.plot(frame.wave, med_res, label='Median Res')
    ax0.plot(frame.wave, signal.medfilt(med_res,51), color='black', label='Median**2 Res')
    ax0.plot(frame.wave, signal.medfilt(wavg_res,51), color='red', label='Med WAvgRes')
    #ax_flux.plot(wave, sky_sig, label='Model Error')
    #ax_flux.plot(wave,true_flux*scl, label='Truth')
    #ax_flux.get_xaxis().set_ticks([]) # Suppress labeling

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
    binsz = 0.1
    xmin,xmax = -5., 5.
    i0, i1 = int( np.min(devs) / binsz) - 1, int( np.max(devs) / binsz) + 1
    rng = tuple( binsz*np.array([i0,i1]) )
    nbin = i1-i0
    # Histogram
    hist, edges = np.histogram(devs, range=rng, bins=nbin)
    xhist = (edges[1:] + edges[:-1])/2.
    #ax.hist(xhist, color='black', bins=edges, weights=hist)#, histtype='step')
    ax1.hist(xhist, color='blue', bins=edges, weights=hist)#, histtype='step')
    # PDF for Gaussian
    area = len(devs) * binsz
    xppf = np.linspace(scipy.stats.norm.ppf(0.0001), scipy.stats.norm.ppf(0.9999), 100)
    ax1.plot(xppf, area*scipy.stats.norm.pdf(xppf), 'r-', alpha=1.0)

    ax1.set_xlabel(r'Res/$\sigma$')
    ax1.set_ylabel('N')
    ax1.set_xlim(xmin,xmax)


    # Meta text
    ax2 = plt.subplot(gs[1,1])
    ax2.set_axis_off()
    show_meta(ax2, qaframe, 'SKYSUB', outfil)
    """
    # Meta
    xlbl = 0.1
    ylbl = 0.85
    i0 = outfil.rfind('/')
    ax2.text(xlbl, ylbl, outfil[i0+1:], color='black', transform=ax2.transAxes, ha='left')
    yoff=0.15
    for key in sorted(qaframe.data['SKYSUB']['QA'].keys()):
        if key in ['QA_FIG']:
            continue
        # Show
        ylbl -= yoff
        ax2.text(xlbl+0.1, ylbl, key+': '+str(qaframe.data['SKYSUB']['QA'][key]), 
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
    cameras = qa_data['frames'].keys()
    # Plot
    fig = plt.figure(figsize=(8, 5.0))
    gs = gridspec.GridSpec(2, 2)

    # Loop on channel
    clrs = dict(b='blue', r='red', z='purple')
    for qq, channel in enumerate(['b','r','z']):

        ax = plt.subplot(gs[qq % 2, qq // 2])
        for camera in cameras:
            if camera[0] == channel:
                ax.errorbar([int(camera[1])],
                            [qa_data['frames'][camera]['FLUXCALIB']['QA']['ZP']],
                            yerr=[qa_data['frames'][camera]['FLUXCALIB']['QA']['RMS_ZP']],
                            capthick=2, fmt='o', color=clrs[channel])


    #
    #ax0.plot([xmin,xmax], [0., 0], '--', color='gray')
    #ax0.plot([xmin,xmax], [0., 0], '--', color='gray')
        ax.set_ylabel('ZP_AB')
        #ax.set_xlim(xmin, xmax)
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


def frame_fluxcalib(outfil, qaframe, fluxcalib, indiv_stars):
    """ QA plots for Flux calibration in a Frame
    Args:
        outfil:
        qaframe:

    Returns:

    """
    # Unpack star data
    sqrtwmodel, sqrtwflux, current_ivar, chi2 = indiv_stars

    # Mean spectrum
    ZP_AB = dsflux.ZP_from_calib(fluxcalib.wave, fluxcalib.meancalib)


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
    nstars = sqrtwflux.shape[0]
    for ii in range(nstars):
        # Good pixels
        gdp = current_ivar[ii, :] > 0.
        icalib = sqrtwflux[ii, gdp] / sqrtwmodel[ii, gdp]
        i_wave = fluxcalib.wave[gdp]
        ZP_star = dsflux.ZP_from_calib(i_wave, icalib)
        # Plot
        if ii == 0:
            lbl ='Individual stars'
        else:
            lbl = None
        ax0.plot(i_wave, ZP_star, ':', label=lbl)
    ax0.plot(fluxcalib.wave, ZP_AB, color='black', label='Mean Calib')

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


def frame_fiberflat(outfil, qaframe, frame, fibermap, fiberflat):
    """ QA plots for fiber flat
    Args:
        outfil:
        qaframe:
        frame:
        fibermap:
        fiberflat:

    Returns:

    """
    # Setup
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
    for key in sorted(qaframe.data['FIBERFLAT']['QA'].keys()):
        if key in ['QA_FIG']:
            continue
        # Show
        ylbl -= yoff
        ax2.text(xlbl+0.05, ylbl, key+': '+str(qaframe.data['FIBERFLAT']['QA'][key]),
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
        ax:
        qadict:

    Returns:

    """
    # Meta
    xlbl = 0.05
    ylbl = 0.85
    i0 = outfil.rfind('/')
    ax.text(xlbl, ylbl, outfil[i0+1:], color='black', transform=ax.transAxes, ha='left')
    yoff=0.10
    for key in sorted(qaframe.data[qaflavor]['QA'].keys()):
        if key in ['QA_FIG']:
            continue
        # Show
        ylbl -= yoff
        ax.text(xlbl+0.1, ylbl, key+': '+str(qaframe.data[qaflavor]['QA'][key]),
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
