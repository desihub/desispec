"""
desispec.io.xytraceset
======================

I/O routines for XYTraceSet objects
"""

import os.path
import time
import numpy as np
from astropy.io import fits

from ..xytraceset import XYTraceSet
from desiutil.log import get_logger
from .util import makepath, iotime_message

def _traceset_from_image(wavemin,wavemax,hdu,label=None) :
    log=get_logger()
    head=hdu.header
    extname=head["EXTNAME"]
    if wavemin is not None :
        if abs(head["WAVEMIN"]-wavemin)>0.001 :
            mess="WAVEMIN not matching in hdu {} {}!={}".format(extname,head["WAVEMIN"],wavemin)
            log.error(mess)
            raise ValueError(mess)
    else :
        wavemin=head["WAVEMIN"]
    if wavemax is not None :
        if abs(head["WAVEMAX"]-wavemax)>0.001 :
            mess="WAVEMAX not matching in hdu {} {}!={}".format(extname,head["WAVEMAX"],wavemax)
            log.error(mess)
            raise ValueError(mess)
    else :
        wavemax=head["WAVEMAX"]
    if label is not None :
        log.debug("read {} from hdu {}".format(label,extname))
    else :
        log.debug("read coefficients from hdu {}".format(label,extname))
                
    return hdu.data,wavemin,wavemax 

def _traceset_from_table(wavemin,wavemax,hdu,pname) :
    log=get_logger()
    head=hdu.header
    table=hdu.data
    
    extname=head["EXTNAME"]
    i=np.where(table["PARAM"]==pname)[0][0]

    if "WAVEMIN" in table.dtype.names :
        twavemin=table["WAVEMIN"][i]
        if wavemin is not None :
            if abs(twavemin-wavemin)>0.001 :
                mess="WAVEMIN not matching in hdu {} {}!={}".format(extname,twavemin,wavemin)
                log.error(mess)
                raise ValueError(mess)
        else :
            wavemin=twavemin
    
    if "WAVEMAX" in table.dtype.names :
        twavemax=table["WAVEMAX"][i]
        if wavemax is not None :
            if abs(twavemax-wavemax)>0.001 :
                mess="WAVEMAX not matching in hdu {} {}!={}".format(extname,twavemax,wavemax)
                log.error(mess)
                raise ValueError(mess)
        else :
            wavemax=twavemax
    
    log.debug("read {} from hdu {}".format(pname,extname))
    return table["COEFF"][i],wavemin,wavemax 

def read_xytraceset(filename) :
    """
    Reads traces in PSF fits file
    
    Args:
        filename : Path to input fits file which has to contain XTRACE and YTRACE HDUs
    Returns:
         XYTraceSet object
    
    """
    #- specter import isolated within function so specter only loaded if
    #- really needed
    from specter.util.traceset import TraceSet,fit_traces

    log=get_logger()
    
    xcoef=None
    ycoef=None
    xsigcoef=None
    ysigcoef=None
    wsigmacoef=None
    wavemin=None
    wavemax=None
     
    log.info("reading traces in '%s'"%filename)

    t0 = time.time()
    fits_file = fits.open(filename)
    
    # npix_y, needed for boxcar extractions
    npix_y=0
    for hdu in [0,"XTRACE","PSF"] :
        if npix_y > 0 : break
        if hdu in fits_file : 
            head = fits_file[hdu].header
            if "NPIX_Y" in head :
                npix_y=int(head["NPIX_Y"])
    if npix_y == 0 :
        raise KeyError("Didn't find head entry NPIX_Y in hdu 0, XTRACE or PSF")
    log.debug("npix_y={}".format(npix_y))
    
    try :
        psftype=fits_file[0].header["PSFTYPE"]
    except KeyError :
        psftype=""
    
    # now read trace coefficients
    log.debug("psf is a '%s'"%psftype)
    if psftype == "bootcalib" :
        xcoef,wavemin,wavemax =_traceset_from_image(wavemin,wavemax,fits_file[0],"xcoef")
        ycoef,wavemin,wavemax =_traceset_from_image(wavemin,wavemax,fits_file[1],"ycoef")
    else :
        for k in ["XTRACE","XCOEF","XCOEFF"] :
            if k in fits_file :
                xcoef,wavemin,wavemax =_traceset_from_image(wavemin,wavemax,fits_file[k],"xcoef")
        for k in ["YTRACE","YCOEF","YCOEFF"] :
            if k in fits_file :
                ycoef,wavemin,wavemax =_traceset_from_image(wavemin,wavemax,fits_file[k],"ycoef")
        for k in ["XSIG"] :
            if k in fits_file :
                xsigcoef,wavemin,wavemax =_traceset_from_image(wavemin,wavemax,fits_file[k],"xsigcoef")
        for k in ["YSIG"] :
            if k in fits_file :
                ysigcoef,wavemin,wavemax =_traceset_from_image(wavemin,wavemax,fits_file[k],"ysigcoef")
        if "WSIGMA" in fits_file :
            wsigmacoef = fits_file["WSIGMA"].data
                
    if psftype == "GAUSS-HERMITE" : # older version where XTRACE and YTRACE are not saved in separate HDUs
        hdu=fits_file["PSF"]
        if xcoef is None    : xcoef,wavemin,wavemax =_traceset_from_table(wavemin,wavemax,hdu,"X")
        if ycoef is None    : ycoef,wavemin,wavemax =_traceset_from_table(wavemin,wavemax,hdu,"Y")
        if xsigcoef is None : xsigcoef,wavemin,wavemax =_traceset_from_table(wavemin,wavemax,hdu,"GHSIGX")
        if ysigcoef is None : ysigcoef,wavemin,wavemax =_traceset_from_table(wavemin,wavemax,hdu,"GHSIGY")
    
    log.debug("wavemin={} wavemax={}".format(wavemin,wavemax))
    
    if xcoef is None or ycoef is None :
        raise ValueError("could not find xcoef and ycoef in psf file %s"%filename)
    
    if xcoef.shape[0] != ycoef.shape[0] :
        raise ValueError("XCOEF and YCOEF don't have same number of fibers %d %d"%(xcoef.shape[0],ycoef.shape[0]))
    
    fits_file.close()
    iotime = time.time() - t0
    log.info(iotime_message('read', filename, iotime))
    
    if wsigmacoef is not None :
        log.warning("Converting deprecated WSIGMA coefficents (in Ang.) into YSIG (in CCD pixels)")
        nfiber    = wsigmacoef.shape[0]
        ncoef     = wsigmacoef.shape[1]
        nw = 100 # to get accurate dydw
        wave      = np.linspace(wavemin,wavemax,nw)
        wsig_set  = TraceSet(wsigmacoef,[wavemin,wavemax])
        y_set  = TraceSet(ycoef,[wavemin,wavemax])
        wsig_vals = np.zeros((nfiber,nw))
        for f in range(nfiber) :
            y_vals = y_set.eval(f,wave)
            dydw   = np.gradient(y_vals)/np.gradient(wave)
            wsig_vals[f]=wsig_set.eval(f,wave)*dydw
        tset = fit_traces(wave, wsig_vals, deg=ncoef-1, domain=(wavemin,wavemax))
        ysigcoef = tset._coeff
        
    return XYTraceSet(xcoef,ycoef,wavemin,wavemax,npix_y,xsigcoef=xsigcoef,ysigcoef=ysigcoef)


def write_xytraceset(outfile,xytraceset) :
    """
    Write a traceset fits file and returns path to file written.
    
    Args:
        outfile: full path to output file
        xytraceset:  desispec.xytraceset.XYTraceSet object
    
    Returns:
         full filepath of output file that was written    
    """

    log=get_logger()
    outfile = makepath(outfile, 'frame')
    hdus = fits.HDUList()
    x = fits.PrimaryHDU(xytraceset.x_vs_wave_traceset._coeff.astype('f4'))
    x.header['EXTNAME'] = "XTRACE"
    if xytraceset.meta is not None :
        for k in xytraceset.meta :
            if not k in x.header :
                x.header[k]=xytraceset.meta[k]
    hdus.append(x)
    hdus.append( fits.ImageHDU(xytraceset.y_vs_wave_traceset._coeff.astype('f4'), name="YTRACE") )
    if xytraceset.xsig_vs_wave_traceset is not None : hdus.append( fits.ImageHDU(xytraceset.xsig_vs_wave_traceset._coeff.astype('f4'), name='XSIG') )
    if xytraceset.ysig_vs_wave_traceset is not None : hdus.append( fits.ImageHDU(xytraceset.ysig_vs_wave_traceset._coeff.astype('f4'), name='YSIG') )
    for hdu in ["XTRACE","YTRACE","XSIG","YSIG"] :
        if hdu in hdus :
            hdus[hdu].header["WAVEMIN"] = xytraceset.wavemin
            hdus[hdu].header["WAVEMAX"] = xytraceset.wavemax
            hdus[hdu].header["NPIX_Y"]  = xytraceset.npix_y

    t0 = time.time()
    hdus.writeto(outfile+'.tmp', overwrite=True, checksum=True)
    os.rename(outfile+'.tmp', outfile)
    iotime = time.time() - t0
    log.info("wrote a xytraceset in {}".format(outfile))
    log.info(iotime_message('write', outfile, iotime))

    return outfile

   
   
