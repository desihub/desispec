#!/usr/bin/env python

"""
Run daily integration tests of the spectroscopic pipeline

Stephen Bailey
Spring 2015
"""
from __future__ import absolute_import, print_function
import sys
import os
import random
import numpy as np
from astropy.io import fits
from desispec import io

def runcmd(cmd, inputs=[], outputs=[], clobber=False):
    """
    Runs a command, checking inputs and outputs

    TODO: document this more
    """
    #- Check that inputs exist
    err = 0
    input_time = 0  #- timestamp of latest input file
    for x in inputs:
        if not os.path.exists(x):
            print("ERROR: missing input "+x)
            err = 1
        else:
            input_time = max(input_time, os.stat(x).st_mtime)

    if err > 0:
        return err

    #- Check if outputs already exist and that their timestamp is after
    #- the last input timestamp
    already_done = True
    if not clobber:
        for x in outputs:
            if not os.path.exists(x):
                already_done = False
                break
            if len(inputs)>0 and os.stat(x).st_mtime < input_time:
                already_done = False
                break

    if already_done:
        print("SKIPPING:", cmd)
        return 0

    #- Green light to go; print info
    print("RUNNING:", cmd)
    if len(inputs) > 0:
        print("  Inputs")
        for x in inputs:
            print("   ", x)
    if len(outputs) > 0:
        print("  Outputs")
        for x in outputs:
            print("   ", x)

    #- run command
    err = os.system(cmd)
    if err > 0:
        print("FAILED:", cmd)
        return err

    #- Check for outputs
    err = 0
    for x in outputs:
        if not os.path.exists(x):
            print("ERROR: missing output "+x)
            err = 2
    if err > 0:
        return err

    print("SUCCESS:", cmd)
    return 0

#-------------------------------------------------------------------------
night = '20150429'
params = dict(night=night, nspec=5)

#-----
#- Input fibermaps, spectra, and pixel-level raw data
for expid, flavor in zip([0,1,2], ['flat', 'arc', 'science']):
    cmd = "pixsim-desi --newexp {flavor} --nspec {nspec} --night {night} --expid {expid}".format(
        expid=expid, flavor=flavor, **params)
    fibermap = io.findfile('fibermap', night, expid)
    simspec = '{}/simspec-{:08d}.fits'.format(os.path.dirname(fibermap), expid)
    inputs = []
    outputs = [fibermap, simspec]
    runcmd(cmd, inputs, outputs)

    cmd = "pixsim-desi --nspec {nspec} --night {night} --expid {expid}".format(expid=expid, **params)
    inputs = [fibermap, simspec]
    outputs = list()
    for camera in ['b0', 'r0', 'z0']:
        pixfile = io.findfile('pix', night, expid, camera)
        outputs.append(pixfile)
        outputs.append(os.path.join(os.path.dirname(pixfile), os.path.basename(pixfile).replace('pix-', 'simpix-')))
    runcmd(cmd, inputs, outputs)

#-----
#- Extract

waverange = dict(
    b = "3570,5940,1.0",
    r = "5630,7740,1.0",
    z = "7440,9830,1.0",
    )
for expid in [0,1,2]:
    for channel in ['b', 'r', 'z']:
        camera = channel+'0'
        pixfile = io.findfile('pix', night, expid, camera)
        psffile = '{}/data/specpsf/psf-{}.fits'.format(os.getenv('DESIMODEL'), channel)
        framefile = io.findfile('frame', night, expid, camera)
        cmd = "exspec -i {pix} -p {psf} --specrange 0,{nspec} -w {wave} -o {frame}".format(
            pix=pixfile, psf=psffile, wave=waverange[channel], frame=framefile, **params)

        inputs = [pixfile, psffile]
        outputs = [framefile,]
        runcmd(cmd, inputs, outputs)

#-----
#- Fiber flat
expid = 0
for channel in ['b', 'r', 'z']:
    camera = channel+"0"
    framefile = io.findfile('frame', night, expid, camera)
    fiberflat = io.findfile('fiberflat', night, expid, camera)
    cmd = "desi_compute_fiberflat.py --infile {frame} --outfile {fiberflat}".format(
        frame=framefile, fiberflat=fiberflat, **params)
    inputs = [framefile,]
    outputs = [fiberflat,]
    runcmd(cmd, inputs, outputs)

#-----
#- Sky model
flat_expid = 0
expid = 2
for channel in ['b', 'r', 'z']:
    camera = channel+"0"
    framefile = io.findfile('frame', night, expid, camera)
    fibermap = io.findfile('fibermap', night, expid)
    fiberflat = io.findfile('fiberflat', night, flat_expid, camera)
    skyfile = io.findfile('sky', night, expid, camera)
    cmd="desi_compute_sky.py --infile {frame} --fibermap {fibermap} --fiberflat {fiberflat} --outfile {sky}".format(
        frame=framefile, fibermap=fibermap, fiberflat=fiberflat, sky=skyfile, **params)
    inputs = [framefile, fibermap, fiberflat]
    outputs = [skyfile, ]
    runcmd(cmd, inputs, outputs)

#-----
#- Fit standard stars
if 'STD_TEMPLATES' in os.environ:
    std_templates = os.getenv('STD_TEMPLATES')
else:
    std_templates = os.getenv('DESI_ROOT')+'/spectro/templates/stellar_templates/v1.0/stdstar_templates_v1.0.fits'

stdstarfile = io.findfile('stdstars', night, expid, spectrograph=0)
cmd = """desi_fit_stdstars.py --spectrograph 0 \
  --fibermap {fibermap} \
  --fiberflatexpid {flat_expid} \
  --models {std_templates} --outfile {stdstars}""".format(
    flat_expid=flat_expid, fibermap=fibermap, std_templates=std_templates,
    stdstars=stdstarfile)

inputs = [fibermap, std_templates]
outputs = [stdstarfile,]
runcmd(cmd, inputs, outputs)

#-----
#- Flux calibration
for channel in ['b', 'r', 'z']:
    camera = channel+"0"
    framefile = io.findfile('frame', night, expid, camera)
    fibermap  = io.findfile('fibermap', night, expid)
    fiberflat = io.findfile('fiberflat', night, flat_expid, camera)
    skyfile   = io.findfile('sky', night, expid, camera)
    calibfile = io.findfile('calib', night, expid, camera)

    #- Compute flux calibration vector
    cmd = """desi_compute_fluxcalibration.py \
      --infile {frame} --fibermap {fibermap} --fiberflat {fiberflat} --sky {sky} \
      --models {stdstars} --outfile {calib}""".format(
        frame=framefile, fibermap=fibermap, fiberflat=fiberflat, sky=skyfile,
        stdstars=stdstarfile, calib=calibfile,
        )
    inputs = [framefile, fibermap, fiberflat, skyfile, stdstarfile]
    outputs = [calibfile,]
    runcmd(cmd, inputs, outputs)

    #- Apply the flux calibration to write a cframe file
    cframefile = io.findfile('cframe', night, expid, camera)
    cmd = """desi_process_exposure.py \
      --infile {frame} --fiberflat {fiberflat} --sky {sky} --calib {calib} \
      --outfile {cframe}""".format(frame=framefile, fibermap=fibermap,
        fiberflat=fiberflat, sky=skyfile, calib=calibfile, cframe=cframefile)
    inputs = [framefile, fiberflat, skyfile, calibfile]
    outputs = [cframefile, ]
    runcmd(cmd, inputs, outputs)

#-----
#- Bricks
inputs = list()
for camera in ['b0', 'r0', 'z0']:
    inputs.append( io.findfile('cframe', night, expid, camera) )

outputs = list()
fibermap, hdr = io.read_fibermap(io.findfile('fibermap', night, expid))
bricks = set(fibermap['BRICKNAME'])
for b in bricks:
    for channel in ['b', 'r', 'z']:
        outputs.append( io.findfile('brick', brickid=b, band=channel))

cmd = "desi_make_bricks.py --night "+night
runcmd(cmd, inputs, outputs)

#-----
#- Redshifts!
for b in bricks:
    inputs = [io.findfile('brick', brickid=b, band=channel) for channel in ['b', 'r', 'z']]
    zbestfile = io.findfile('zbest', brickid=b)
    outputs = [zbestfile, ]
    cmd = "desi_zfind.py --brick {} -o {}".format(b, zbestfile)
    runcmd(cmd, inputs, outputs)

#-----
#- Did it work?
#- (this combination of fibermap, simspec, and zbest is a pain)
simdir = os.path.dirname(io.findfile('fibermap', night=night, expid=expid))
simspec = '{}/simspec-{:08d}.fits'.format(simdir, expid)
siminfo = fits.getdata(simspec, 'METADATA')


print()
print("------------------------------------------")
for b in bricks:
    zbest = io.read_zbest(io.findfile('zbest', brickid=b))
    for i in range(len(zbest.z)):
        if zbest.type[i] == 'ssp_em_galaxy':
            objtype = 'GAL'
        elif zbest.type[i] == 'spEigenStar':
            objtype = 'STAR'
        else:
            objtype = zbest.type[i]

        z, zwarn = zbest.z[i], zbest.zwarn[i]

        j = np.where(fibermap['TARGETID'] == zbest.targetid[i])[0][0]
        truetype = siminfo['OBJTYPE'][j]
        truez = siminfo['REDSHIFT'][j]
        dv = 3e5*(z-truez)/(1+truez)
        if truetype == 'SKY' and zwarn > 0:
            status = 'ok'
        elif zwarn == 0:
            if truetype == 'LRG' and objtype == 'GAL' and abs(dv) < 150:
                status = 'ok'
            elif truetype == 'ELG' and objtype == 'GAL' and abs(dv) < 150:
                status = 'ok'
            elif truetype == 'QSO' and objtype == 'QSO' and abs(dv) < 750:
                status = 'ok'
            elif truetype == 'STD' and objtype == 'STAR':
                status = 'ok'
            else:
                status = 'oops'
        else:
            status = 'oops'
        print('{0} {1:4s} {2:8.5f} {3:4d} {4:4s} {5:8.5f} - {6}'.format(b, objtype, z, zwarn, truetype, truez, status))


print("------------------------------------------")
