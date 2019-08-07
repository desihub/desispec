"""
Coadd spectra
"""

from __future__ import absolute_import, division, print_function
import os, sys, time

import numpy as np

from desiutil.log import get_logger

from ..io import read_spectra,write_spectra

def parse(options=None):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--infile", type=str,  help="input spectra file")
    parser.add_argument("-o","--outfile", type=str,  help="output spectra file")
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)

    return args

def main(args=None):

    log = get_logger()

    if args is None:
        args = parse()

    spectra = read_spectra(args.infile)
    targets = np.unique(spectra.fibermap["TARGETID"])
    ntarget=targets.size
    log.debug("number of targets= {}".format(ntarget))
    for b in spectra._bands :
        log.debug("coadding band '{}'".format(b))
        nwave=spectra.wave[b].size
        tflux=np.zeros((ntarget,nwave),dtype=spectra.flux[b].dtype)
        tivar=np.zeros((ntarget,nwave),dtype=spectra.ivar[b].dtype)
        tmask=np.zeros((ntarget,nwave),dtype=spectra.mask[b].dtype)
        trdata=np.zeros((ntarget,spectra.resolution_data[b].shape[1],nwave),dtype=spectra.resolution_data[b].dtype)
        for i,tid in enumerate(targets) :
            jj=spectra.fibermap["TARGETID"]==tid
            tivar_unmasked= np.sum(spectra.ivar[b][jj],axis=0)
            tivar[i]=np.sum(spectra.ivar[b][jj]*(spectra.mask[b][jj]==0),axis=0)
            tflux[i]=np.sum(spectra.ivar[b][jj]*(spectra.mask[b][jj]==0)*spectra.flux[b][jj],axis=0)
            for r in range(spectra.resolution_data[b].shape[1]) :
                trdata[i,r]=np.sum((spectra.ivar[b][jj]*spectra.resolution_data[b][jj,r]),axis=0) # not sure applying mask is wise here
            bad=(tivar[i]==0)
            if np.sum(bad)>0 :
                tivar[i][bad] = np.sum(spectra.ivar[b][jj][bad],axis=0) # if all masked, keep original ivar
                tflux[i][bad] = np.sum(spectra.ivar[b][jj][bad]*spectra.tflux[b][jj][bad],axis=0)
            ok=(tivar[i]>0)
            if np.sum(ok)>0 :
                tflux[i][ok] /= tivar[i][ok]
            ok=(tivar_unmasked>0)
            if np.sum(ok)>0 :
                trdata[i][:,ok] /= tivar_unmasked[ok]
            tmask[i]      = np.bitwise_and.reduce(spectra.mask[b][jj],axis=0)
        spectra.flux[b] = tflux
        spectra.ivar[b] = tivar
        spectra.mask[b] = tmask
        spectra.resolution_data[b] = trdata
    jj=np.zeros(ntarget,dtype=int)
    for i,tid in enumerate(targets) :
        jj[i]=np.where(spectra.fibermap["TARGETID"]==tid)[0][0]
    spectra.fibermap=spectra.fibermap[jj]
    
    ## now deal with the fibermap
    #tfmap = spectra.fibermap[:ntarget][:]
    #for i,tid in enumerate(targets) :
    #    jj=spectra.fibermap["TARGETID"]==tid
    #    for k in spectra.fibermap.colnames :
    #        tfmap[i][k]=spectra.fibermap[jj[0]][k] # take first entry
    #spectra.fibermap=tfmap
    spectra.scores=None
    log.debug("writing {} ...".format(args.outfile))
    write_spectra(args.outfile,spectra)
