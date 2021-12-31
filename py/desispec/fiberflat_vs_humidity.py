"""
desispec.fiberflat_vs_humidity
==================

Utility functions to compute a fiber flat corrected for variations with humidity in the shack
"""
from __future__ import absolute_import, division


import numpy as np
import copy

from desiutil.log import get_logger
from desispec.fiberflat import apply_fiberflat
from desispec.fiberbitmasking import get_skysub_fiberbitmask_val

def _interpolated_fiberflat_vs_humidity(fiberflat_vs_humidity , humidity_array, humidity_point) :

    if humidity_point<=humidity_array[0] :
        i1=0
    else :
        i1=np.where(humidity_array<humidity_point)[0][-1]
    i2=i1+1
    if i2>=humidity_array.size : # return largest value
        return fiberflat_vs_humidity[-1]

    w1=(humidity_array[i2]-humidity_point)/(humidity_array[i2]-humidity_array[i1])
    w2=(humidity_point-humidity_array[i1])/(humidity_array[i2]-humidity_array[i1])
    return w1*fiberflat_vs_humidity[i1]+w2*fiberflat_vs_humidity[i2]


'''
def compute_humidity_corrected_fiberflat(calib_fiberflat, mean_fiberflat_vs_humidity , humidity_array, calib_humidity, current_humidity, frame=None) :

    log = get_logger()

    # interpolate flat for humidity during calibration exposures and for the current value
    mean_fiberflat_at_current_humidity = _interpolated_fiberflat_vs_humidity(mean_fiberflat_vs_humidity , humidity_array, current_humidity)
    mean_fiberflat_at_calib_humidity   = _interpolated_fiberflat_vs_humidity(mean_fiberflat_vs_humidity , humidity_array, calib_humidity)

    tmp_fiberflat = copy.deepcopy(calib_fiberflat)

    best_power = 1.

    if frame is not None :
        log.info("using frame")
        ivar = frame.ivar*(frame.mask==0)
        badfibermask = get_skysub_fiberbitmask_val()
        selection = (frame.fibermap["OBJTYPE"]=="SKY") & (frame.fibermap["FIBERSTATUS"] & badfibermask == 0) & (np.sum(ivar!=0,axis=1)>10)
        if np.sum(selection)>0 :
            # adjust using sky fibers
            fibers = np.where(selection)[0]
            selection = (frame.wave > 4000.) & (frame.wave < 4600)
            if np.sum(selection)>0 :

                # apply default flatfield
                tmp_frame = copy.deepcopy(frame)
                apply_fiberflat(frame,calib_fiberflat)

                waveindex = np.where(selection)[0]
                tmp_flux = tmp_frame.flux[fibers][:,waveindex]
                tmp_ivar = (tmp_frame.ivar[fibers]*(tmp_frame.mask[fibers]==0))[:,waveindex]

                humidityindex = np.argmin(np.abs(humidity_array-current_humidity))
                #b=max(0,humidityindex-10)
                #e=min(humidity_array.size,humidityindex+11)
                b=0
                e=humidity_array.size
                chi2=np.zeros(e-b)
                powers=np.ones(e-b)

                heliocor=frame.meta['HELIOCOR']
                fiberflat_wave_in_frame_system  = calib_fiberflat.wave*heliocor

                corr = np.zeros((fibers.size,waveindex.size))
                csky = np.zeros((fibers.size,waveindex.size))
                #sw   = np.sum(tmp_ivar,axis=0)
                #msky = np.sum(tmp_ivar*tmp_flux,axis=0)/(sw+(sw==0))
                msky = np.median(tmp_flux,axis=0)

                for index in range(b,e) :

                    # compute correction
                    for f,fiber in enumerate(fibers) :
                        # tune current humidity
                        #corr[f] = np.interp(frame.wave,fiberflat_wave_in_frame_system,mean_fiberflat_vs_humidity[index,fiber]/mean_fiberflat_at_calib_humidity[fiber])[waveindex]
                        # tune calib humidity
                        corr[f] = np.interp(frame.wave,fiberflat_wave_in_frame_system,mean_fiberflat_at_current_humidity[fiber]/mean_fiberflat_vs_humidity[index,fiber])[waveindex]
                    # rm median per fiber
                    # corr /= np.median(corr,axis=1)[:,None]

                    scales=np.ones(fibers.size)
                    if 1 :
                        # also compute a scale per fiber
                        aa = np.sum(tmp_ivar*msky[None,:]**2,axis=1)
                        bb = np.sum(tmp_ivar*tmp_flux*msky[None,:],axis=1)
                        scales = bb/(aa+(aa==0))
                        tmp_ivar *= np.abs(scales-1)[:,None]<0.2 # discard any fiber with scale offset larger than 0.2

                    power=1.

                    if 1 :
                        # fit a power law of the correction
                        tmp_powers=np.linspace(0.2,2.,10)

                        chi2power=np.zeros(tmp_powers.size)
                        for powindex,p in enumerate(tmp_powers) :
                            # apply correction to sky
                            csky = scales[:,None]*msky[None,:]*corr**p
                            chi2power[powindex] = np.sum(tmp_ivar*(tmp_flux-csky)**2)
                        minindex=np.argmin(chi2power)
                        bb=minindex-1
                        ee=minindex+2
                        if bb<0 :
                            bb+=1
                            ee+=1
                        if ee>=chi2power.size :
                            bb-=1
                            ee-=1
                        c=np.polyfit(tmp_powers[bb:ee],chi2power[bb:ee],2)
                        power = -c[1]/2./c[0]
                        power = max(tmp_powers[0],power)
                        power = min(tmp_powers[-1],power)

                    # apply correction to sky
                    csky = scales[:,None]*msky[None,:]*(corr**power)

                    chi2[index-b] = np.sum(tmp_ivar*(tmp_flux-csky)**2)
                    powers[index-b] = power

                    if 0 :
                        import matplotlib.pyplot as plt
                        for f,fiber in enumerate(fibers) :
                            plt.plot(frame.wave[waveindex],tmp_flux[f],"o")
                            plt.plot(frame.wave[waveindex],csky[f])
                            plt.show()

                minindex=np.argmin(chi2)
                bb=minindex-1
                ee=minindex+2
                if bb<0 :
                    bb+=1
                    ee+=1
                if ee>=chi2.size :
                    bb-=1
                    ee-=1
                c=np.polyfit(humidity_array[bb:ee],chi2[bb:ee],2)
                best_humidity = -c[1]/2./c[0]
                best_humidity = max(humidity_array[0],best_humidity)
                best_humidity = min(humidity_array[-1],best_humidity)


                best_power = powers[minindex]
                ndata=np.sum(tmp_ivar>0)

                #log.info("Consider best fit humidity = {:.2f} instead of {:.2f} with power={:.2f}".format(best_humidity,current_humidity,best_power))
                log.info("Consider best fit humidity = {:.2f} instead of {:.2f} with power={:.2f}".format(best_humidity,calib_humidity,best_power))




                if 1 :
                    import matplotlib.pyplot as plt
                    plt.subplot(2,1,1)
                    plt.plot(humidity_array[b:e],chi2/ndata,"o-")
                    plt.axvline(current_humidity,linestyle="--")
                    plt.axvline(calib_humidity,linestyle=":")
                    plt.axvline(best_humidity,linestyle="-")
                    plt.subplot(2,1,2)
                    plt.plot(humidity_array[b:e],powers,"o-")
                    plt.show()



            else :
                log.warning("no valid wavelength range for humidity flat correction using sky fibers")
        else :
            log.warning("no valid sky fiber for humidity flat correction")


    log.info(f"use {best_humidity} with {best_power}")

    # interpolate flat for humidity during calibration exposures and for the current value
    #mean_fiberflat_at_current_humidity = _interpolated_fiberflat_vs_humidity(mean_fiberflat_vs_humidity, humidity_array, best_humidity)
    mean_fiberflat_at_calib_humidity = _interpolated_fiberflat_vs_humidity(mean_fiberflat_vs_humidity , humidity_array, best_humidity)

    # apply humidity correction to current calib fiberflat
    tmp_fiberflat.fiberflat = calib_fiberflat.fiberflat * (mean_fiberflat_at_current_humidity/mean_fiberflat_at_calib_humidity)**best_power

    #import matplotlib.pyplot as plt
    #plt.plot(calib_fiberflat.wave,calib_fiberflat.fiberflat[225])
    #plt.plot(calib_fiberflat.wave,tmp_fiberflat.fiberflat[225])
    #plt.show()

    return tmp_fiberflat
'''

def _fit_flat(wavelength,flux,ivar,fibers,mean_fiberflat_vs_humidity,humidity_array) :

    log = get_logger()
    selection = (wavelength > 4000.) & (wavelength < 4600)
    if np.sum(selection)==0 :
        message="incorrect wavelength range"
        log.error(message)
        raise RuntimeError(message)
    waveindex = np.where(selection)[0]
    tmp_flux = flux[fibers][:,waveindex]
    tmp_ivar = ivar[fibers][:,waveindex]
    b=0
    e=humidity_array.size
    chi2=np.zeros(e-b)
    nfiber=tmp_flux.shape[0]
    nwave=tmp_flux.shape[1]
    corr = np.zeros((nfiber,nwave))
    csky = np.zeros((nfiber,nwave))
    msky = np.median(tmp_flux,axis=0)
    for index in range(b,e) :
        # compute correction
        corr=mean_fiberflat_vs_humidity[index,fibers][:,waveindex]
        scales=np.ones(nfiber)
        # compute a scale per fiber
        aa = np.sum(tmp_ivar*msky[None,:]**2,axis=1)
        bb = np.sum(tmp_ivar*tmp_flux*msky[None,:],axis=1)
        scales = bb/(aa+(aa==0))
        tmp_ivar *= np.abs(scales-1)[:,None]<0.2 # discard any fiber with scale offset larger than 0.2
        # apply correction to sky
        csky = scales[:,None]*msky[None,:]*corr
        chi2[index-b] = np.sum(tmp_ivar*(tmp_flux-csky)**2)

    minindex=np.argmin(chi2)



    bb=minindex-1
    ee=minindex+2
    if bb<0 :
        bb+=1
        ee+=1
    if ee>=chi2.size :
        bb-=1
        ee-=1

    '''
    # fit linear combination of 3 templates
    npar=ee-bb
    h=np.zeros((npar,nfiber,nwave))
    for i in range(npar) :
        h[i]=scales[:,None]*msky[None,:]*mean_fiberflat_vs_humidity[bb+i,fibers][:,waveindex]
    A=np.zeros((npar,npar))
    B=np.zeros(npar)
    for i in range(npar) :
        B[i]=np.sum(tmp_ivar*tmp_flux*h[i])
        for j in range(i,npar) :
            A[i,j]=np.sum(tmp_ivar*h[i]*h[j])
            if j>i : A[j,i]=A[i,j]
    Ai=np.linalg.inv(A)
    coefs=Ai.dot(B)
    log.info("coefficients = {} for humidities = {}".format(list(coefs),list(humidity_array[bb:ee])))
    flat = np.zeros((mean_fiberflat_vs_humidity.shape[1],mean_fiberflat_vs_humidity.shape[2]))
    for i,coef in enumerate(coefs) :
        flat += coef*mean_fiberflat_vs_humidity[bb+i]
    '''

    # prefered method:
    # get the chi2 minimum and interpolate
    c=np.polyfit(humidity_array[bb:ee],chi2[bb:ee],2)
    best_humidity = -c[1]/2./c[0]
    best_humidity = max(humidity_array[0],best_humidity)
    best_humidity = min(humidity_array[-1],best_humidity)
    log.info("best fit humidity = {:.2f}".format(best_humidity))
    flat = _interpolated_fiberflat_vs_humidity(mean_fiberflat_vs_humidity , humidity_array, best_humidity)

    return flat

def compute_humidity_corrected_fiberflat(calib_fiberflat, mean_fiberflat_vs_humidity , humidity_array, current_humidity, frame) :

    log = get_logger()

    best_humidity = current_humidity


    log.info("using nightly flat to fit for the best fit nightly flat humidity")
    selection = np.sum(calib_fiberflat.ivar!=0,axis=1)>10
    good_flat_fibers = np.where(selection)[0]
    flat2 = _fit_flat(calib_fiberflat.wave,calib_fiberflat.fiberflat,calib_fiberflat.ivar,good_flat_fibers,mean_fiberflat_vs_humidity,humidity_array)

    flat1 = None
    if frame is not None :
        log.info("using frame to fit for the best fit current humidity")
        ivar = frame.ivar*(frame.mask==0)
        badfibermask = get_skysub_fiberbitmask_val()
        selection = (frame.fibermap["OBJTYPE"]=="SKY") & (frame.fibermap["FIBERSTATUS"] & badfibermask == 0) & (np.sum(ivar!=0,axis=1)>10)
        if np.sum(selection)>0 :
            good_sky_fibers = np.where(selection)[0]
            heliocor=frame.meta['HELIOCOR']
            frame_wave_in_fiberflat_system  = frame.wave/heliocor
            tmp_flux = frame.flux.copy()
            tmp_ivar = ivar.copy()
            for fiber in good_sky_fibers:
                ok=(ivar[fiber]>0)
                tmp_flux[fiber] = np.interp(frame.wave,frame_wave_in_fiberflat_system[ok],frame.flux[fiber][ok])
                tmp_ivar[fiber] = np.interp(frame.wave,frame_wave_in_fiberflat_system[ok],ivar[fiber][ok])

            flat1  = _fit_flat(frame.wave,tmp_flux*flat2/calib_fiberflat.fiberflat,tmp_ivar,good_sky_fibers,mean_fiberflat_vs_humidity,humidity_array)
            #flat1  = _fit_flat(frame.wave,tmp_flux,tmp_ivar,good_sky_fibers,mean_fiberflat_vs_humidity,humidity_array)
    if flat1 is None :
        log.info("use input humidity = {:.2f}".format(current_humidity))
        flat1  = _interpolated_fiberflat_vs_humidity(mean_fiberflat_vs_humidity , humidity_array, current_humidity)

    # apply humidity correction to current calib fiberflat
    fiberflat = copy.deepcopy(calib_fiberflat)
    fiberflat.fiberflat = calib_fiberflat.fiberflat/flat2*flat1
    #fiberflat.fiberflat = flat1

    return fiberflat
