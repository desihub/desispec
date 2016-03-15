#
# See top-level LICENSE.rst file for Copyright information
#
# -*- coding: utf-8 -*-
"""
desispec.pipeline.plan
======================

Functions for planning and execution of pipeline operations
"""

from __future__ import absolute_import, division, print_function

import os
import time
import glob
import re

import astropy.io.fits as af

import desispec.io as io
import desispec.log as log


def find_raw(rawdir, rawnight):
    expid = io.get_exposures(rawnight, raw=True, rawdata_dir=rawdir)
    fibermap = {}
    raw = {}
    exptype = {}
    for ex in sorted(expid):
        # get the fibermap for this exposure
        fibermap[ex] = io.get_raw_files("fibermap", rawnight, ex, rawdata_dir=rawdir)
        # check the fibermap header for the "flavor"
        # keyword to see what this exposure is.
        hd = af.getheader(fibermap[ex], 1)
        exptype[ex] = hd['flavor']
        # get the raw exposures
        raw[ex] = io.get_raw_files("pix", rawnight, ex, rawdata_dir=rawdir)
    return (sorted(expid), exptype, fibermap, raw)


def find_frames(specdir, night):
    fullexpid = io.get_exposures(night, raw=False, specprod_dir=specdir)
    expid = []
    frames = {}
    exptype = {}
    for ex in sorted(fullexpid):
        found = io.get_files("frame", night, ex, specprod_dir=specdir)
        if len(found.keys()) == 0:
            continue
        expid.append(ex)
        frames[ex] = found
        # check the header for the "flavor"
        # keyword to see what this exposure is.
        testfile = frames[ex][frames[ex].keys()[0]]
        hd = af.getheader(testfile, 0)
        exptype[ex] = hd['FLAVOR']
    return (expid, exptype, frames)


def psf_newest(specdir):
    newest = {}
    psfpat = re.compile(r'psf-([brz][0-9])-([0-9]{8})\.fits')
    for root, dirs, files in os.walk(specdir, topdown=True):
        for f in files:
            psfmat = psfpat.match(f)
            if psfmat is not None:
                cam = psfmat.group(1)
                expstr = psfmat.group(2)
                expid = int(expstr)
                if cam not in newest.keys():
                    newest[cam] = expid
                else:
                    if expid > newest[cam]:
                        newest[cam] = expid
    return newest


def find_bricks(proddir):
    bricks = []
    brickpat = re.compile(r'\d{4}[pm]\d{3}')
    for root, dirs, files in os.walk(os.path.join(proddir, 'bricks'), topdown=True):
        for d in dirs:
            brickmat = brickpat.match(d)
            if brickmat is not None:
                bricks.append(d)
    return bricks


def tasks_exspec_exposure(id, raw, fibermap, wrange, psf_select):
    # These are fixed for DESI
    spec_per_bundle = 25
    nbundle = 20
    nspec = nbundle * spec_per_bundle

    tasks_extract = []
    tasks_merge = []
    tasks_clean = []

    cameras = sorted(raw.keys())
    for cam in cameras:
        if cam not in psf_select.keys():
            continue
        outbase = os.path.join("{:08d}".format(id), "frame-{}-{:08d}".format(cam, id))
        outfile = "{}.fits".format(outbase)
        psffile = os.path.join("{:08d}".format(psf_select[cam]), "psf-{}-{:08d}.fits".format(cam, psf_select[cam]))
        mergeinputs = []

        # select wavelength range based on camera
        wmin = wrange[cam][0]
        wmax = wrange[cam][1]
        dw = wrange[cam][2]

        for b in range(nbundle):
            outb = "{}-{:02d}.fits".format(outbase, b)
            mergeinputs.append(outb)
            com = ['desi_extract_spectra.py']
            com.extend(['-i', raw[cam]])
            com.extend(['-p', psffile])
            com.extend(['-f', fibermap[id]])
            com.extend(['-o', '{}.part'.format(outb)])
            com.extend(['--specmin', "{}".format(b*spec_per_bundle)])
            com.extend(['--nspec', "{}".format(spec_per_bundle)])
            com.extend(['-w', "{},{},{}".format(wmin,wmax,dw)])

            task = {}
            task['command'] = com
            task['parallelism'] = 'core'
            task['inputs'] = [raw[cam], psffile]
            task['outputs'] = [outb]

            tasks_extract.append(task)

        com = ['desi_merge_bundles.py']
        com.extend(['-o', '{}.part'.format(outfile)])
        com.extend(mergeinputs)
        task = {}
        task['command'] = com
        task['parallelism'] = 'core'
        task['inputs'] = mergeinputs
        task['outputs'] = [outfile]
        tasks_merge.append(task)

        com = ['rm', '-f']
        com.extend(mergeinputs)
        task = {}
        task['command'] = com
        task['parallelism'] = 'core'
        task['inputs'] = [outfile]
        task['outputs'] = []
        tasks_clean.append(task)

    return [tasks_extract, tasks_merge, tasks_clean]


def tasks_exspec(expid, exptype, raw, fibermap, wrange, psf_select):
    tasks_extract = []
    tasks_merge = []
    tasks_clean = []
    for ex in expid:
        if exptype[ex] == "arc":
            continue
        [exp_tasks_extract, exp_tasks_merge, exp_tasks_clean] = tasks_exspec_exposure(ex, raw[ex], fibermap, wrange, psf_select)
        tasks_extract.extend(exp_tasks_extract)
        tasks_merge.extend(exp_tasks_merge)
        tasks_clean.extend(exp_tasks_clean)
    return [tasks_extract, tasks_merge, tasks_clean]


def tasks_specex_exposure(id, raw, lamplines, bootcal=None):
    # These are fixed for DESI
    spec_per_bundle = 25
    nbundle = 20
    nspec = nbundle * spec_per_bundle

    tasks_bundle = []
    tasks_merge = []
    tasks_clean = []

    cameras = sorted(raw.keys())
    for cam in cameras:
        outbase = os.path.join("{:08d}".format(id), "psf-{}-{:08d}".format(cam, id))
        outxml = "{}.xml".format(outbase)
        outfits = "{}.fits".format(outbase)
        outspot = "{}-spots.xml".format(outbase)
        mergeinputs = []
        mergespotinputs = []
        cleanfiles = []
        for b in range(nbundle):
            outxmlb = "{}-{:02}.xml".format(outbase, b)
            outspotb = "{}-{:02}-spots.xml".format(outbase, b)
            outfitsb = "{}-{:02}.fits".format(outbase, b)
            mergeinputs.append(outxmlb)
            mergespotinputs.append(outspotb)
            cleanfiles.append(outxmlb)
            cleanfiles.append(outfitsb)
            cleanfiles.append(outspotb)
            com = ['specex_desi_psf_fit']
            com.extend(['-a', raw[cam]])
            if bootcal is not None:
                com.extend(['--xcoord-file', bootcal[cam]])
                com.extend(['--xcoord-hdu', '1'])
                com.extend(['--ycoord-file', bootcal[cam]])
                com.extend(['--ycoord-hdu', '2'])
            com.extend(['--lamplines', lamplines])
            com.extend(['--out_xml', '{}.part'.format(outxmlb)])
            com.extend(['--out_spots', '{}.part'.format(outspotb)])
            com.extend(['--out_fits', '{}.part'.format(outfitsb)])
            com.extend(['--first_bundle', "{}".format(b)])
            com.extend(['--last_bundle', "{}".format(b)])
            # For now, we use specex defaults...
            # com.extend(['--gauss_hermite_deg', '8'])
            # com.extend(['--psfmodel', 'GAUSSHERMITE'])
            # com.extend(['--half_size_x', '14'])
            # com.extend(['--half_size_y', '8'])
            # com.extend(['--fit_psf_tails'])
            # com.extend(['--fit_continuum'])
            # com.extend(['-v'])
            # com.extend(['--core'])
            # com.extend(['--no_trace_fit'])
            # com.extend(['--trace_deg_x', '6'])
            # com.extend(['--trace_deg_wave', '6'])
            # com.extend(['--legendre_deg_x', '1'])
            # com.extend(['--legendre_deg_wave', '4'])

            task = {}
            task['command'] = com
            task['parallelism'] = 'node'
            task['inputs'] = [raw[cam], bootcal[cam]]
            task['outputs'] = [outxmlb, outspotb, outfitsb]

            tasks_bundle.append(task)

        com = ['specex_merge_psf']
        com.extend(['--out-fits', '{}.part'.format(outfits)])
        com.extend(['--out-xml', '{}.part'.format(outxml)])
        com.extend(mergeinputs)
        task = {}
        task['command'] = com
        task['parallelism'] = 'core'
        task['inputs'] = mergeinputs
        task['outputs'] = [outfits, outxml]
        tasks_merge.append(task)

        com = ['specex_merge_spot']
        com.extend(['--out', '{}.part'.format(outspot)])
        com.extend(mergespotinputs)
        task = {}
        task['command'] = com
        task['parallelism'] = 'core'
        task['inputs'] = mergespotinputs
        task['outputs'] = [outspot]
        tasks_merge.append(task)

        com = ['rm', '-f']
        com.extend(cleanfiles)
        task = {}
        task['command'] = com
        task['parallelism'] = 'core'
        task['inputs'] = [outfits, outxml]
        task['outputs'] = []
        tasks_clean.append(task)

    return [tasks_bundle, tasks_merge, tasks_clean]


def tasks_specex(expid, exptype, raw, lamplines, bootcal=None):
    tasks_bundle = []
    tasks_merge = []
    tasks_clean = []
    for ex in expid:
        if exptype[ex] != "arc":
            continue
        [exp_tasks_bundle, exp_tasks_merge, exp_tasks_clean] = tasks_specex_exposure(ex, raw[ex], lamplines, bootcal=bootcal)
        tasks_bundle.extend(exp_tasks_bundle)
        tasks_merge.extend(exp_tasks_merge)
        tasks_clean.extend(exp_tasks_clean)
    return [tasks_bundle, tasks_merge, tasks_clean]


def tasks_fiberflat_exposure(id, frames, calnight):
    tasks = []
    cameras = sorted(frames.keys())
    for cam in cameras:
        infile = os.path.join("{:08d}".format(id), "frame-{}-{:08d}.fits".format(cam, id))
        outfile = os.path.join(calnight, "fiberflat-{}-{:08d}.fits".format(cam, id))

        com = ['desi_compute_fiberflat.py']
        com.extend(['--infile', infile])
        com.extend(['--outfile', '{}.part'.format(outfile)])

        task = {}
        task['command'] = com
        task['parallelism'] = 'core'
        task['inputs'] = [infile]
        task['outputs'] = [outfile]
        tasks.append(task)

    return tasks


def tasks_fiberflat(expid, exptype, frames, calnight):
    tasks = []
    for ex in expid:
        if exptype[ex] != "flat":
            continue
        exp_tasks = tasks_fiberflat_exposure(ex, frames[ex], calnight)
        tasks.extend(exp_tasks)
    return tasks


def tasks_sky_exposure(id, frames, calnight, flatexp, fibermap):
    tasks = []
    cameras = sorted(frames.keys())
    for cam in cameras:
        infile = os.path.join("{:08d}".format(id), "frame-{}-{:08d}.fits".format(cam, id))
        outfile = os.path.join("{:08d}".format(id), "sky-{}-{:08d}.fits".format(cam, id))
        flatfile = os.path.join(calnight, "fiberflat-{}-{:08d}.fits".format(cam, flatexp))
        com = ['desi_compute_sky.py']
        com.extend(['--infile', infile])
        com.extend(['--outfile', "{}.part".format(outfile)])
        com.extend(['--fibermap', fibermap[id]])
        com.extend(['--fiberflat', flatfile])

        task = {}
        task['command'] = com
        task['parallelism'] = 'core'
        task['inputs'] = [infile, fibermap[id], flatfile]
        task['outputs'] = [outfile]
        tasks.append(task)

    return tasks


def tasks_sky(expid, exptype, frames, calnight, fibermap):
    tasks = []
    for ex in expid:
        if exptype[ex] == "flat":
            continue
        if exptype[ex] == "arc":
            continue
        flatexp = None
        for fex in expid:
            if exptype[fex] == 'flat':
                if (flatexp is None) or ((fex > flatexp) and (fex < ex)):
                    flatexp = fex
        exp_tasks = tasks_sky_exposure(ex, frames[ex], calnight, flatexp, fibermap)
        tasks.extend(exp_tasks)
    return tasks


def tasks_star_exposure(id, frames, calnight, flatexp, fibermap):
    tasks = []
    cameras = sorted(frames.keys())
    for cam in cameras:
        spectrograph = cam[1]
        flatfile = os.path.join(calnight, "fiberflat-{}-{:08d}.fits".format(cam, flatexp))
        outfile = os.path.join("{:08d}".format(id), "stdstars-{}-{:08d}.fits".format(cam, id))
        com = ['desi_fit_stdstars.py']
        com.extend(['--fiberflatexpid', "{}".format(flatexp)])
        com.extend(['--outfile', "{}.part".format(outfile)])
        com.extend(['--fibermap', fibermap[id]])
        com.extend(['--models', '/project/projectdirs/desi/spectro/templates/star_templates/v1.0/stdstar_templates_v1.0.fits'])
        com.extend(['--spectrograph', spectrograph])

        task = {}
        task['command'] = com
        task['parallelism'] = 'core'
        task['inputs'] = [fibermap[id], flatfile]
        task['outputs'] = [outfile]
        tasks.append(task)

    return tasks


def tasks_star(expid, exptype, frames, calnight, fibermap):
    tasks = []
    for ex in expid:
        if exptype[ex] == "flat":
            continue
        if exptype[ex] == "arc":
            continue
        flatexp = None
        for fex in expid:
            if exptype[fex] == 'flat':
                if (flatexp is None) or ((fex > flatexp) and (fex < ex)):
                    flatexp = fex
        exp_tasks = tasks_star_exposure(ex, frames[ex], calnight, flatexp, fibermap)
        tasks.extend(exp_tasks)
    return tasks


def tasks_calcalc_exposure(id, frames, calnight, flatexp, fibermap):
    tasks = []
    cameras = sorted(frames.keys())
    for cam in cameras:
        flatfile = os.path.join(calnight, "fiberflat-{}-{:08d}.fits".format(cam, flatexp))
        skyfile = os.path.join("{:08d}".format(id), "sky-{}-{:08d}.fits".format(cam, id))
        outfile = os.path.join("{:08d}".format(id), "calib-{}-{:08d}.fits".format(cam, id))
        infile = os.path.join("{:08d}".format(id), "frame-{}-{:08d}.fits".format(cam, id))
        starfile = os.path.join("{:08d}".format(id), "stdstars-{}-{:08d}.fits".format(cam, id))
        com = ['desi_compute_fluxcalibration.py']
        com.extend(['--infile', infile])
        com.extend(['--fiberflat', flatfile])
        com.extend(['--outfile', "{}.part".format(outfile)])
        com.extend(['--fibermap', fibermap[id]])
        com.extend(['--models', starfile])
        com.extend(['--sky', skyfile])

        task = {}
        task['command'] = com
        task['parallelism'] = 'core'
        task['inputs'] = [infile, skyfile, fibermap[id], flatfile]
        task['outputs'] = [outfile]
        tasks.append(task)

    return tasks


def tasks_calcalc(expid, exptype, frames, calnight, fibermap):
    tasks = []
    for ex in expid:
        if exptype[ex] == "flat":
            continue
        if exptype[ex] == "arc":
            continue
        flatexp = None
        for fex in expid:
            if exptype[fex] == 'flat':
                if (flatexp is None) or ((fex > flatexp) and (fex < ex)):
                    flatexp = fex
        exp_tasks = tasks_calcalc_exposure(ex, frames[ex], calnight, flatexp, fibermap)
        tasks.extend(exp_tasks)
    return tasks


def tasks_calapp_exposure(id, frames, calnight, flatexp):
    tasks = []
    cameras = sorted(frames.keys())
    for cam in cameras:
        flatfile = os.path.join(calnight, "fiberflat-{}-{:08d}.fits".format(cam, flatexp))
        outfile = os.path.join("{:08d}".format(id), "cframe-{}-{:08d}.fits".format(cam, id))
        calfile = os.path.join("{:08d}".format(id), "calib-{}-{:08d}.fits".format(cam, id))
        infile = os.path.join("{:08d}".format(id), "frame-{}-{:08d}.fits".format(cam, id))
        skyfile = os.path.join("{:08d}".format(id), "sky-{}-{:08d}.fits".format(cam, id))
        com = ['desi_process_exposure.py']
        com.extend(['--infile', infile])
        com.extend(['--fiberflat', flatfile])
        com.extend(['--outfile', "{}.part".format(outfile)])
        com.extend(['--sky', skyfile])
        com.extend(['--calib', calfile])

        task = {}
        task['command'] = com
        task['parallelism'] = 'core'
        task['inputs'] = [infile, skyfile, calfile, flatfile]
        task['outputs'] = [outfile]
        tasks.append(task)

    return tasks


def tasks_calapp(expid, exptype, frames, calnight):
    tasks = []
    for ex in expid:
        if exptype[ex] == "flat":
            continue
        if exptype[ex] == "arc":
            continue
        flatexp = None
        for fex in expid:
            if exptype[fex] == 'flat':
                if (flatexp is None) or ((fex > flatexp) and (fex < ex)):
                    flatexp = fex
        exp_tasks = tasks_calapp_exposure(ex, frames[ex], calnight, flatexp)
        tasks.extend(exp_tasks)
    return tasks


def tasks_zfind(bricks, objtype, zspec):
    tasks = []
    for brk in bricks:
        brick_r = os.path.join(brk, "brick-r-{}.fits".format(brk))
        brick_b = os.path.join(brk, "brick-b-{}.fits".format(brk))
        brick_z = os.path.join(brk, "brick-z-{}.fits".format(brk))
        outfile = os.path.join(brk, "zbest-{}.fits".format(brk))

        com = ['desi_zfind.py']
        com.extend(['--brick', brk])
        com.extend(['--outfile', "{}.part".format(outfile)])
        if objtype is not None:
            com.extend(['--objtype', objtype])
        if zspec:
            com.extend(['--zspec'])

        task = {}
        task['command'] = com
        task['parallelism'] = 'core'
        task['inputs'] = [brick_r, brick_b, brick_z]
        task['outputs'] = [outfile]
        tasks.append(task)
    return tasks


def task_dist(tasklist, nworker):
    ntask = len(tasklist)
    work = {}
    for i in range(nworker):
        myn = ntask // nworker
        off = 0
        leftover = ntask % nworker
        if ( i < leftover ):
            myn = myn + 1
            off = i * myn
        else:
            off = ((myn + 1) * leftover) + (myn * (i - leftover))
        work[i] = tasklist[off:(off+myn)]
    return work




