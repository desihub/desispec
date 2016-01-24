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

import desispec.io as io
import desispec.log as log


def find_raw(rawdir, rawnight, simraw=False):
    expid = io.get_exposures(rawnight, raw=True, rawprod_dir=rawdir):
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
        if simraw:
            raw[ex] = io.get_raw_files("pix", rawnight, ex, rawdata_dir=rawdir)
        else:
            raw[ex] = io.get_raw_files("raw", rawnight, ex, rawdata_dir=rawdir)
    return (sorted(expid), exptype, fibermap, raw)


def psf_newest(specdir):
    newest = {}
    psfpat = re.compile(r'psf-([brz][0-9])-(.*).fits')
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
                    if expid > newest[cam]
                        newest[cam] = expid
        break
    return newest


def tasks_exspec_exposure(id, raw, wrange, psf_select):
    spec_per_bundle = 25
    nbundle = 20
    nspec = nbundle * spec_per_bundle
    tasks_extract = []
    tasks_merge = []

    cameras = sorted(raw.keys())
    for cam in cameras:
        outbase = os.path.join("{:08d}".format(id), "frame-{}-{:08d}".format(cam, id))
        outfile = "{}.fits".format(outbase)
        psffile = os.path.join("{:08d}".format(id), "psf-{}-{:08d}".format(cam, id))
        mergeinputs = []

        # select wavelength range based on camera
        wmin = wrange[cam][0]
        wmax = wrange[cam][1]
        dw = wrange[cam][2]

        for b in range(nbundle):
            outb = "{}-{:02d}.fits".format(outbase, b)
            mergeinputs.append(outb)
            com = ['exspec']
            com.extend(['-i', raw[cam]])
            com.extend(['-p', psffile])
            com.extend(['-o', outb])
            com.extend(['--specmin', b*spec_per_bundle])
            com.extend(['--nspec', spec_per_bundle])
            com.extend(['-w', "{},{},{}".format(wmin,wmax,dw)])

            task = {}
            task['command'] = com
            task['parallelism'] = 'core'
            task['inputs'] = [raw[cam], psffile]
            task['outputs'] = [outb]

            tasks_extract.append(task)

        com = ['merge_bundles']
        com.extend(['-o', outfile])
        com.extend(mergeinputs)

        task = {}
        task['command'] = com
        task['parallelism'] = 'core'
        task['inputs'] = mergeinputs
        task['outputs'] = [outfile]

        tasks_merge.append(task)

    return [tasks_extract, tasks_merge]


def tasks_exspec(expid, exptype, raw, wrange, psf_select):
    tasks_extract = []
    tasks_merge = []
    for ex in expid:
        if exptype[ex] == "arc":
            continue
        [exp_tasks_extract, exp_tasks_merge] = tasks_exspec_exposure(ex, raw[ex], wrange, psf_select)
        tasks_extract.extend(exp_tasks_extract)
        tasks_merge.extend(exp_tasks_merge)
    return [tasks_extract, tasks_merge]


def tasks_specex_exposure(id, raw_files, simpix=None):
    tasks_bundle = []
    tasks_merge = []
    cameras = sorted(raw.keys())
    for cam in cameras:
        outbase = os.path.join("{:08d}".format(id), "psf-{}-{:08d}".format(cam, id))
        outxml = "{}.xml".format(outbase)
        outfits = "{}.fits".format(outbase)
        mergeinputs = []
        for b in range(nbundle):
            outxmlb = "{}-{:02}.xml".format(outbase, b)
            outspotb = "{}-{:02}-spots.xml".format(outbase, b)
            outfitsb = "{}-{:02}.fits".format(outbase, b)
            mergeinputs.append(outxmlb)
            com = ['specex_desi_psf']
            com.extend(['-a', raw[cam]])
            if simpix is not None:
                com.extend(['--xcoord-file', simpix[cam]])
                com.extend(['--xcoord-hdu', '2'])
                com.extend(['--ycoord-file', simpix[cam]])
                com.extend(['--ycoord-hdu', '3'])
            com.extend(['--out_xml', outxmlb])
            com.extend(['--out_spots', outspotb])
            com.extend(['--out_fits', outfitsb])
            com.extend(['--first_bundle', b])
            com.extend(['--last_bundle', b])
            com.extend(['--gauss_hermite_deg', '8'])
            com.extend(['--psfmodel', 'GAUSSHERMITE'])
            com.extend(['--half_size_x', '14'])
            com.extend(['--half_size_y', '8'])
            com.extend(['--fit_psf_tails'])
            com.extend(['--fit_continuum'])
            com.extend(['-v'])
            com.extend(['--core'])
            com.extend(['--no_trace_fit'])
            com.extend(['--legendre_deg_x', '1'])
            com.extend(['--legendre_deg_wave', '4'])

            task = {}
            task['command'] = com
            task['parallelism'] = 'node'
            task['inputs'] = [raw[cam], simpix[cam]]
            task['outputs'] = [outxmlb, outspotb, outfitsb]

            tasks_bundle.append(task)

        com = ['specex_merge_psf']
        com.extend(['--out-fits', outfits])
        com.extend(['--out-xml', outxml])
        com.extend(mergeinputs)

        task = {}
        task['command'] = com
        task['parallelism'] = 'node'
        task['inputs'] = mergeinputs
        task['outputs'] = [outfits, outxml]

        tasks_merge.append(task)

    return [tasks_bundle, tasks_merge]


def tasks_specex(expid, exptype, raw, simpix=None):
    tasks_bundle = []
    tasks_merge = []
    for ex in expid:
        if exptype[ex] != "arc":
            continue
        exp_simpix = None
        if simpix is not None:
            exp_simpix = simpix[ex]
        [exp_tasks_bundle, exp_tasks_merge] = tasks_specex_exposure(ex, raw[ex], simpix=exp_simpix)
        tasks_bundle.extend(exp_tasks_bundle)
        tasks_merge.extend(exp_tasks_merge)
    return [tasks_bundle, tasks_merge]


def task_dist(tasklist, nworker):
    ntask = len(tasklist)
    work = {}
    for i in range(nworker):
        myn = ntask // nworker
        off = 0
        leftover = totalsize % groups
        if ( i < leftover ):
            myn = myn + 1
            off = i * myn
        else:
            off = ((myn + 1) * leftover) + (myn * (i - leftover))
        work[i] = tasklist[off:(off+myn)]
    return work




