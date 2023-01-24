"""
desispec.scripts.inspect_dark
=============================

Please add module-level documentation.
"""

import os,sys
import numpy as np

import argparse
import fitsio
import scipy.ndimage

from astropy.table import Table

from desiutil.log import get_logger
from desispec.calibfinder import findcalibfile
from desispec.io import read_xytraceset,read_fibermap
from desispec.io.util import get_tempfilename
from desispec.maskbits import fibermask, extractmaskval

def parse(options=None):

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,description="Inspect a dark image to check for changes in the bright columns")

    parser.add_argument('-i','--infile', type = str, default = None, required = True, help = 'path to preproc fits file')
    parser.add_argument('--badfiber-table', type = str, default = None, required = False, help = 'output table with list of bad fibers (because of bad columns)')
    parser.add_argument('--badcol-table', type = str, default = None, required = False, help = 'output table with list of bad columns')
    parser.add_argument('--plot', action = 'store_true', help = 'plot the profiles')
    parser.add_argument('--threshold', type = float , default = 0.005, required = False, help = 'threshold in electrons/sec to flag columns.')
    parser.add_argument('--sigma', type = float , default = 6, required = False, help = 'required statistical significance of column detection')
    parser.add_argument('--psf', type = str , default = None, required = False, help = 'specify psf file for trace coordinates, default is automatically found.')
    parser.add_argument('--dist', type = float , default = 3., required = False, help = 'min distance in pixels between fiber trace and bad column')
    parser.add_argument('--nopsf', action = 'store_true', help = 'do not read the traces and do not associate bad columns to fibers')

    args = parser.parse_args(options)

    return args

def main(args=None):

    log=get_logger()

    if not isinstance(args, argparse.Namespace):
        args = parse(args)

    log.info("Reading "+args.infile)
    with fitsio.FITS(args.infile) as fx:
        pix  = fx['IMAGE'].read()
        head = fx['IMAGE'].read_header()
        ivar = fx['IVAR'].read()
        mask = (fx['MASK'].read() & extractmaskval) | (ivar==0.0)

    exptime = head["EXPTIME"]
    log.info("Exposure time = {:.1f} sec".format(exptime))
    threshold_electrons = args.threshold*exptime
    log.info("Threshold in electrons = {:.2f}".format(threshold_electrons))
    log.info("Threshold significance > {:.2f} sigma".format(args.sigma))

    if not args.nopsf :
        if args.psf is None :
            args.psf = findcalibfile([head],"PSF")
        log.info("Will read traceset in "+args.psf)

        tset = read_xytraceset(args.psf)
        fmap = read_fibermap(args.infile)
        fibers = fmap["FIBER"]
    else :
        tset   = None
        fibers = None

    n0=pix.shape[0]
    n1=pix.shape[1]

    # inverse variance -> variance, semi-arbitrarily clipping at 400 for ivar=0
    pixvar = 1.0/(ivar + (ivar==0)/400.0)

    profs=[]
    mprofs=[]

    col_x=[]
    col_elec_per_sec=[]
    col_sigma=[]
    col_begin=[]
    col_end=[]

    for half in range(2) :
        if half==0 :
            b=0
            e=n0//2
        else :
            b=n0//2
            e=n0

        # remove possible background before detecting bad columns
        prof = np.zeros(n1)
        mprof = np.zeros(n1)
        good=np.where(np.sum(mask[b:e]==0,axis=0)>0)[0]
        prof[good] = np.ma.median(np.ma.masked_array(data=pix[b:e,good],mask=(mask[b:e,good]!=0)),axis=0).data

        step=30 # must be more than twice as large as bad column, but not to wide to follow variations
        for lr in range(2) :
            if lr==0 :
                pb=0
                pe=n1//2
            else :
                pb=n1//2
                pe=n1
            medf = scipy.ndimage.median_filter(prof[pb:pe],step)
            mmedf = np.max(np.abs(medf))
            if mmedf> 0.5 :
                log.warning("anomalous residual background in dark of {:.2f} electrons".format(mmedf))
            prof[pb:pe] -= medf
            mprof[pb:pe] = medf

        profs.append(prof)
        mprofs.append(mprof)

        # detect cols
        aprof = np.abs(prof)
        peak = (aprof>threshold_electrons/2) # half of threshold to select columns for further inspection
        peak[1:-1] &= (aprof[1:-1]>aprof[:-2])&(aprof[1:-1]>aprof[2:])
        peak=np.where(peak)[0]

        for p in peak :

            # sum over a band of 3 pixels (to account for wide bright columns)
            pb=max(0,p-1)
            pe=min(pix.shape[1],p+2)
            tmpval=np.sum(pix[:,pb:pe],axis=1)
            tmpvar=np.sum(pixvar[:,pb:pe],axis=1)

            # median filter for column that do not cross the whole amplifier
            medsize = 400
            medval = scipy.ndimage.median_filter(tmpval, medsize)
            medsigma = scipy.ndimage.median_filter(tmpval/np.sqrt(tmpvar), medsize) * np.sqrt(medsize)/np.sqrt(np.pi/2)
            i = np.argmax(np.abs(medval))
            val = medval[i]
            significance = np.abs(medsigma[i])

            # apply threshold here
            # if np.abs(val)<threshold_electrons:
            if np.abs(val)<threshold_electrons or significance < args.sigma:
                continue

            log.info("Bad column x={} val={:.2f} electrons ({:.2f} sigma) -> {:.4f} elec/sec".format(
                p, val, significance, val/exptime))
            col_x.append(p)
            col_elec_per_sec.append( round(val/exptime, 5) )
            col_sigma.append(round(significance, 3))


    if not args.nopsf : # add fiber info
        wrange = (tset.wavemax-tset.wavemin)
        wmean  = (tset.wavemin+tset.wavemax)/2.
        wave   = np.array([wmean-wrange*0.4,wmean, wmean+wrange*0.4]) # do not test the whole range because on signal on edges

        fiberx = np.zeros((fibers.size,wave.size))
        for i in range(fibers.size) :
            fiberx[i] = tset.x_vs_wave(i,wave)

        entries={}
        entries["FIBER"]=[]
        entries["COLUMN"]=[]
        entries["ELEC_PER_SEC"]=[]
        entries["SIGMA"]=[]

        for i , x in enumerate(col_x) :
            bad = np.any(np.abs(fiberx-x)<args.dist,axis=1)
            badfibers = fibers[bad]
            log.info("COLUMN={} ELEC_PER_SEC={} FIBERS={}".format(x,col_elec_per_sec[i],list(badfibers)))
            for badfiber in badfibers :
                entries["FIBER"].append(badfiber)
                entries["COLUMN"].append(x)
                entries["ELEC_PER_SEC"].append(col_elec_per_sec[i])
                entries["SIGMA"].append(col_sigma[i])

        if args.badfiber_table is not None :
            t = Table()
            for k in entries.keys() :
                t[k]=entries[k]

            t["FIBERSTATUS"]=np.repeat(fibermask.BADCOLUMN,len(t))

            camera=head["camera"]
            t["CAMERA"]=np.repeat(camera,len(t))

            odir=os.path.dirname(os.path.abspath(args.badfiber_table))
            if not os.path.isdir(odir):
                os.makedirs(odir)

            tmpfile = get_tempfilename(args.badfiber_table)
            t.write(tmpfile, overwrite=True)
            os.rename(tmpfile, args.badfiber_table)
            log.info("wrote {}".format(args.badfiber_table))



    t = Table()
    t["CAMERA"]=np.repeat(head["camera"],len(col_x))
    t["COLUMN"]=col_x
    t["ELEC_PER_SEC"]=col_elec_per_sec
    t["SIGMA"]=col_sigma

    if len(t) > 0:
        log.info(f'Bad columns in {os.path.basename(args.infile)}:')
        print(t)
    else:
        log.info(f'No bad columns identified in {os.path.basename(args.infile)}')

    if args.badcol_table is not None :
        odir=os.path.dirname(os.path.abspath(args.badcol_table))
        if not os.path.isdir(odir):
            os.makedirs(odir)

        tmpfile = get_tempfilename(args.badcol_table)
        t.write(tmpfile, overwrite=True)
        os.rename(tmpfile, args.badcol_table)
        log.info("wrote {}".format(args.badcol_table))


    if args.plot :
        import matplotlib.pyplot as plt
        plt.figure("preproc-profiles")
        plt.subplot(111,title=os.path.basename(args.infile))
        plt.plot(mprofs[0],c="C0",alpha=0.3)
        plt.plot(mprofs[1],c="C1",alpha=0.3)
        plt.plot(profs[0],c="C0")
        plt.plot(profs[1],c="C1")
        plt.grid()
        plt.show()

    return 0

