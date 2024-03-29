"""
desispec.scripts.coadd_preproc
==============================

"""
import os,sys
import numpy as np
import fitsio

from desiutil.log import get_logger
from desispec.io import read_image,write_image,read_xytraceset
from desispec.preproc import masked_median

def parse(options=None):

    import argparse

    parser = argparse.ArgumentParser(description="coaddition of preproc images (actually a mean)")
    parser.add_argument("-i","--infile", type=str, required=True, nargs="*",help="input preproc images")
    parser.add_argument("-o","--outfile", type=str, required=True, help="output preproc image")
    parser.add_argument("--nsig", type=float, default = 4., required=False, help="nsig rejection")
    parser.add_argument("--fracerr", type=float, default = 0.1, required=False, help="fractional error used in outlier rejection")
    parser.add_argument("--weighted", action = 'store_true', help="use weighted mean instead of unweighted mean (which is the default)")

    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)

    return args

def main(args=None):

    log=get_logger()

    if args is None :
        args = parse()

    first_preproc=None
    images=[]
    ivars=[]
    masks=[]

    log.info("read inputs ...")
    for filename in args.infile:
        log.info(" read {}".format(filename))
        tmp = read_image(filename)
        if first_preproc is None :
            first_preproc = tmp
        images.append(tmp.pix.ravel())
        # make sure we don't include data with ivar=0
        mask = tmp.mask + (tmp.ivar==0).astype(int)
        masks.append(mask.ravel())
        ivars.append((tmp.ivar*(tmp.mask==0)).ravel())


    images=np.array(images)
    masks=np.array(masks)
    ivars=np.array(ivars)
    smask=np.sum(masks,axis=0)

    log.info("compute masked median image ...")
    medimage=masked_median(images,masks)

    log.info("use masked median image to discard outlier ...")
    good=((images-medimage)**2*(ivars>0)/( (medimage>0)*(medimage*args.fracerr)**2 + 1./(ivars+(ivars==0)))) < args.nsig**3
    ivars *= good.astype(float)

    if args.weighted :
        log.info("compute weighted mean ...")
        sw  = np.sum(ivars,axis=0)
        swf = np.sum(ivars*images,axis=0)
        meanimage = swf/(sw+(sw==0))
        meanivar  = sw
    else :
        log.info("compute unweighted mean ...")
        s1  = np.sum((ivars>0).astype(int),axis=0)
        sf = np.sum((ivars>0)*images,axis=0)
        meanimage = sf/(s1+(s1==0))
        meanvar   = np.sum( (ivars>0)/( ivars + (ivars==0) ))/(s1+(s1==0))**2
        meanivar  = (meanvar>0)/(meanvar+(meanvar==0))
        log.info("write nimages.fits ...")
        fitsio.write("nimages.fits",s1.reshape(first_preproc.pix.shape),clobber=True)

    log.info("compute mask ...")
    meanmask  = masks[0]
    for mask in masks[1:] :
        meanmask &= mask


    log.info("write image ...")
    preproc = first_preproc
    shape = preproc.pix.shape
    preproc.pix  = meanimage.reshape(shape)
    preproc.ivar = meanivar.reshape(shape)
    preproc.mask = meanmask.reshape(shape)

    write_image(args.outfile,preproc)
    log.info("wrote {}".format(args.outfile))
