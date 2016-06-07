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
import sys
import time
import glob
import re
import copy
import yaml

import astropy.io.fits as af

import desispec.io as io
import desispec.log as log


graph_types = [
    'night',
    'fibermap',
    'pix',
    'psfboot',
    'psf',
    'psfnight',
    'frame',
    'fiberflat',
    'sky',
    'stdstars',
    'calib',
    'cframe',
    'brick',
    'zbest'
]

_state_colors = {
    'none': '#000000',
    'done': '#00ff00',
    'fail': '#ff0000',
    'wait': '#ffff00'
}

_graph_sep = '_'


def default_options():
    allopts = {}

    opts = {}
    opts['trace-only'] = False
    opts['legendre-degree'] = 5
    opts['triplet-matching'] = True
    allopts['bootcalib'] = opts

    opts = {}
    # opts['flux-hdu'] = 1
    # opts['ivar-hdu'] = 2
    # opts['mask-hdu'] = 3
    # opts['header-hdu'] = 1
    opts['xcoord-hdu'] = 1
    opts['ycoord-hdu'] = 2
    # opts['psfmodel'] = 'GAUSSHERMITE'
    # opts['half_size_x'] = 8
    # opts['half_size_y'] = 5
    # opts['verbose'] = False
    # opts['gauss_hermite_deg'] = 6
    # opts['legendre_deg_wave'] = 4
    # opts['legendre_deg_x'] = 1
    # opts['trace_deg_wave'] = 6
    # opts['trace_deg_x'] = 6
    opts['lamplines'] = '/project/projectdirs/desi/software/edison/specex/specex-0.3.9/data/specex_linelist_boss.txt'
    allopts['specex'] = opts

    allopts['psfcombine'] = {}

    opts = {}
    opts['regularize'] = 0.0
    opts['nwavestep'] = 50
    opts['verbose'] = False
    opts['wavelength_b'] = "3579.0,5939.0,0.8"
    opts['wavelength_r'] = "5635.0,7731.0,0.8"
    opts['wavelength_z'] = "7445.0,9824.0,0.8"
    allopts['extract'] = opts

    allopts['fiberflat'] = {}

    allopts['sky'] = {}

    opts = {}
    opts['starmodels'] = '/project/projectdirs/desi/spectro/templates/star_templates/v1.1/star_templates_v1.1.fits'
    # opts['starmodels'] = '/project/projectdirs/desi/spectro/templates/basis_templates/v2.2/star_templates_v2.1.fits'
    allopts['stdstars'] = opts

    allopts['fluxcal'] = {}

    allopts['procexp'] = {}

    allopts['makebricks'] = {}

    allopts['zfind'] = {}

    return allopts


def write_options(path, opts):
    with open(path, 'w') as f:
        yaml.dump(opts, f, default_flow_style=False)
    return


def read_options(path):
    opts = None
    with open(path, 'r') as f:
        opts = yaml.load(f)
    return opts


def select_nights(allnights, nightstr):
    # Trim list of nights based on set of patterns
    nights = []
    if nightstr is not None:
        nightsel = nightstr.split(',')
        for sel in nightsel:
            pat = re.compile(sel)
            for nt in allnights:
                mat = pat.match(nt)
                if mat is not None:
                    if nt not in nights:
                        nights.append(nt)
        nights = sorted(nights)
    else:
        nights = sorted(allnights)

    return nights


def create_prod(rawdir, proddir, nightstr=None):

    expnightcount = {}

    # create main directories if they don't exist

    if not os.path.isdir(proddir):
        os.makedirs(proddir)
    
    cal2d = os.path.join(proddir, 'calib2d')
    if not os.path.isdir(cal2d):
        os.makedirs(cal2d)

    calpsf = os.path.join(cal2d, 'psf')
    if not os.path.isdir(calpsf):
        os.makedirs(calpsf)

    expdir = os.path.join(proddir, 'exposures')
    if not os.path.isdir(expdir):
        os.makedirs(expdir)

    brkdir = os.path.join(proddir, 'bricks')
    if not os.path.isdir(brkdir):
        os.makedirs(brkdir)

    plandir = os.path.join(proddir, 'plan')
    if not os.path.isdir(plandir):
        os.makedirs(plandir)

    rundir = os.path.join(proddir, 'run')
    if not os.path.isdir(rundir):
        os.makedirs(rundir)

    faildir = os.path.join(rundir, 'failed')
    if not os.path.isdir(faildir):
        os.makedirs(faildir)

    scriptdir = os.path.join(rundir, 'scripts')
    if not os.path.isdir(scriptdir):
        os.makedirs(scriptdir)

    logdir = os.path.join(rundir, 'logs')
    if not os.path.isdir(logdir):
        os.makedirs(logdir)

    optfile = os.path.join(rundir, 'options.yaml')
    if not os.path.isfile(optfile):
        opts = default_options()
        write_options(optfile, opts)

    # get list of nights

    allnights = []
    nightpat = re.compile(r'\d{8}')
    for root, dirs, files in os.walk(rawdir, topdown=True):
        for d in dirs:
            nightmat = nightpat.match(d)
            if nightmat is not None:
                allnights.append(d)
        break

    nights = select_nights(allnights, nightstr)

    # create per-night directories

    allbricks = set()

    for nt in nights:
        nexpdir = os.path.join(expdir, nt)
        if not os.path.isdir(nexpdir):
            os.makedirs(nexpdir)
        ndir = os.path.join(cal2d, nt)
        if not os.path.isdir(ndir):
            os.makedirs(ndir)
        ndir = os.path.join(calpsf, nt)
        if not os.path.isdir(ndir):
            os.makedirs(ndir)

        grph, expcount, nbricks = graph_night(rawdir, nt)
        allbricks.update(nbricks)
        expnightcount[nt] = expcount
        with open(os.path.join(plandir, "{}.dot".format(nt)), 'w') as f:
            graph_dot(grph, f)
        graph_write(os.path.join(plandir, "{}.yaml".format(nt)), grph)
        # make per-exposure dirs
        for name, node in grph.items():
            if node['type'] == 'fibermap':
                fdir = os.path.join(nexpdir, "{:08d}".format(node['id']))
                if not os.path.isdir(fdir):
                    os.makedirs(fdir)

    return expnightcount, allbricks


def graph_name(*args):
    if len(args) > 0:
        return _graph_sep.join(args)
    else:
        return ""


# Each node of the graph is a dictionary, with required keys 'type',
# 'in', and 'out'.  Where 'in' and 'out' are lists of other nodes.  
# Extra keys for each type are allowed.  Some keys (band, spec, etc)
# are technically redundant (since they could be obtained by working
# back up the graph to the raw data properties), however this is done
# for convenience.

def graph_night(rawdir, rawnight):

    grph = {}

    node = {}
    node['type'] = 'night'
    node['in'] = []
    node['out'] = []
    grph[rawnight] = node

    allbricks = set()

    expcount = {}
    expcount['flat'] = 0
    expcount['arc'] = 0
    expcount['science'] = 0

    # First, insert raw data into the graph.  We use the existence of the raw data
    # as a filter over spectrographs.  Spectrographs whose raw data do not exist
    # are excluded from the graph.

    expid = io.get_exposures(rawnight, raw=True, rawdata_dir=rawdir)
    
    campat = re.compile(r'([brz])([0-9])')

    keepspec = set()

    for ex in sorted(expid):
        # get the fibermap for this exposure
        fibermap = io.get_raw_files("fibermap", rawnight, ex, rawdata_dir=rawdir)

        # read the fibermap to get the exposure type, and while we are at it,
        # also accumulate the total list of bricks        

        fmdata, fmheader = io.read_fibermap(fibermap, header=True)
        flavor = fmheader['flavor']
        bricks = set()
        fmbricks = []
        for fmb in fmdata['BRICKNAME']:
            if len(fmb) > 0:
                fmbricks.append(fmb)
        bricks.update(fmbricks)
        allbricks.update(bricks)

        if flavor == 'arc':
            expcount['arc'] += 1
        elif flavor == 'flat':
            expcount['flat'] += 1
        else:
            expcount['science'] += 1

        node = {}
        node['type'] = 'fibermap'
        node['id'] = ex
        node['flavor'] = flavor
        node['bricks'] = bricks
        node['in'] = [rawnight]
        node['out'] = []
        name = graph_name(rawnight, "fibermap-{:08d}".format(ex))

        grph[name] = node
        grph[rawnight]['out'].append(name)

        # get the raw exposures
        raw = io.get_raw_files("pix", rawnight, ex, rawdata_dir=rawdir)

        for cam in sorted(raw.keys()):
            cammat = campat.match(cam)
            if cammat is None:
                raise RuntimeError("invalid camera string {}".format(cam))
            band = cammat.group(1)
            spec = cammat.group(2)

            keepspec.update(spec)

            node = {}
            node['type'] = 'pix'
            node['id'] = ex
            node['band'] = band
            node['spec'] = spec
            node['flavor'] = flavor
            node['in'] = [rawnight]
            node['out'] = []
            name = graph_name(rawnight, "pix-{}{}-{:08d}".format(band, spec, ex))

            grph[name] = node
            grph[rawnight]['out'].append(name)

    keep = sorted(list(keepspec))

    # Now that we have added all the raw data to the graph, we work our way
    # through the processing steps.  

    # This step is a placeholder, in case we want to combine information from
    # multiple flats or arcs before running bootcalib.  We mark these bootcalib
    # outputs as depending on all arcs and flats, but in reality we may just
    # use the first or last set.

    # Since each psfboot file takes multiple exposures as input, we first
    # create those nodes.

    for band in ['b', 'r', 'z']:
        for spec in keep:
            name = graph_name(rawnight, "psfboot-{}{}".format(band, spec))
            node = {}
            node['type'] = 'psfboot'
            node['band'] = band
            node['spec'] = spec
            node['in'] = []
            node['out'] = []
            grph[name] = node

    for name, nd in grph.items():
        if nd['type'] != 'pix':
            continue
        if (nd['flavor'] != 'flat') and (nd['flavor'] != 'arc'):
            continue
        band = nd['band']
        spec = nd['spec']
        bootname = graph_name(rawnight, "psfboot-{}{}".format(band, spec))
        grph[bootname]['in'].append(name)
        nd['out'].append(bootname)

    # Next is full PSF estimation.  Inputs are the arc image and the bootcalib
    # output file.  We also add nodes for the combined psfs.

    for band in ['b', 'r', 'z']:
        for spec in keep:
            name = graph_name(rawnight, "psfnight-{}{}".format(band, spec))
            node = {}
            node['type'] = 'psfnight'
            node['band'] = band
            node['spec'] = spec
            node['in'] = []
            node['out'] = []
            grph[name] = node

    for name, nd in grph.items():
        if nd['type'] != 'pix':
            continue
        if nd['flavor'] != 'arc':
            continue
        band = nd['band']
        spec = nd['spec']
        id = nd['id']
        bootname = graph_name(rawnight, "psfboot-{}{}".format(band, spec))
        psfname = graph_name(rawnight, "psf-{}{}-{:08d}".format(band, spec, id))
        psfnightname = graph_name(rawnight, "psfnight-{}{}".format(band, spec))
        node = {}
        node['type'] = 'psf'
        node['band'] = band
        node['spec'] = spec
        node['id'] = id
        node['in'] = [name, bootname]
        node['out'] = [psfnightname]
        grph[psfname] = node
        grph[bootname]['out'].append(psfname)
        grph[psfnightname]['in'].append(psfname)
        nd['out'].append(psfname)

    # Now we extract the flats and science frames using the nightly psf

    for name, nd in grph.items():
        if nd['type'] != 'pix':
            continue
        if nd['flavor'] == 'arc':
            continue
        band = nd['band']
        spec = nd['spec']
        id = nd['id']
        flavor = nd['flavor']
        framename = graph_name(rawnight, "frame-{}{}-{:08d}".format(band, spec, id))
        psfnightname = graph_name(rawnight, "psfnight-{}{}".format(band, spec))
        fmname = graph_name(rawnight, "fibermap-{:08d}".format(id))
        node = {}
        node['type'] = 'frame'
        node['band'] = band
        node['spec'] = spec
        node['id'] = id
        node['flavor'] = flavor
        node['in'] = [name, fmname, psfnightname]
        node['out'] = []
        grph[framename] = node
        grph[psfnightname]['out'].append(framename)
        grph[fmname]['out'].append(framename)
        nd['out'].append(framename)

    # Now build the fiberflats for each flat exposure.  We keep a list of all
    # available fiberflats while we are looping over them, since we'll need
    # that in the next step to select the "most recent" fiberflat.

    flatexpid = {}

    for name, nd in grph.items():
        if nd['type'] != 'frame':
            continue
        if nd['flavor'] != 'flat':
            continue
        band = nd['band']
        spec = nd['spec']
        id = nd['id']
        flatname = graph_name(rawnight, "fiberflat-{}{}-{:08d}".format(band, spec, id))
        node = {}
        node['type'] = 'fiberflat'
        node['band'] = band
        node['spec'] = spec
        node['id'] = id
        node['in'] = [name]
        node['out'] = []
        grph[flatname] = node
        nd['out'].append(flatname)
        cam = "{}{}".format(band, spec)
        if cam not in flatexpid.keys():
            flatexpid[cam] = []
        flatexpid[cam].append(id)

    # To compute the sky file, we use the "most recent fiberflat" that came
    # before the current exposure.

    for name, nd in grph.items():
        if nd['type'] != 'frame':
            continue
        if nd['flavor'] == 'flat':
            continue
        band = nd['band']
        spec = nd['spec']
        id = nd['id']
        cam = "{}{}".format(band, spec)
        flatid = None
        for fid in sorted(flatexpid[cam]):
            if (flatid is None):
                flatid = fid
            elif (fid > flatid) and (fid < id):
                flatid = fid
        skyname = graph_name(rawnight, "sky-{}{}-{:08d}".format(band, spec, id))
        flatname = graph_name(rawnight, "fiberflat-{}{}-{:08d}".format(band, spec, fid))
        node = {}
        node['type'] = 'sky'
        node['band'] = band
        node['spec'] = spec
        node['id'] = id
        node['in'] = [name, flatname]
        node['out'] = []
        grph[skyname] = node
        nd['out'].append(skyname)
        grph[flatname]['out'].append(skyname)

    # Construct the standard star files.  These are one per spectrograph,
    # and depend on the frames and the corresponding flats and sky files.

    stdgrph = {}

    for name, nd in grph.items():
        if nd['type'] != 'frame':
            continue
        if nd['flavor'] == 'flat':
            continue
        band = nd['band']
        spec = nd['spec']
        id = nd['id']

        starname = graph_name(rawnight, "stdstars-{}-{:08d}".format(spec, id))
        # does this spectrograph exist yet in the graph?
        if starname not in stdgrph.keys():
            fmname = graph_name(rawnight, "fibermap-{:08d}".format(id))
            grph[fmname]['out'].append(starname)
            node = {}
            node['type'] = 'stdstars'
            node['spec'] = spec
            node['id'] = id
            node['in'] = [fmname]
            node['out'] = []
            stdgrph[starname] = node

        cam = "{}{}".format(band, spec)
        flatid = None
        for fid in sorted(flatexpid[cam]):
            if (flatid is None):
                flatid = fid
            elif (fid > flatid) and (fid < id):
                flatid = fid
                
        flatname = graph_name(rawnight, "fiberflat-{}{}-{:08d}".format(band, spec, fid))
        skyname = graph_name(rawnight, "sky-{}{}-{:08d}".format(band, spec, id))

        stdgrph[starname]['in'].extend([skyname, name, flatname])

        nd['out'].append(starname)
        grph[flatname]['out'].append(starname)
        grph[skyname]['out'].append(starname)

    grph.update(stdgrph)

    # Construct calibration files

    for name, nd in grph.items():
        if nd['type'] != 'frame':
            continue
        if nd['flavor'] == 'flat':
            continue
        band = nd['band']
        spec = nd['spec']
        id = nd['id']
        cam = "{}{}".format(band, spec)
        flatid = None
        for fid in sorted(flatexpid[cam]):
            if (flatid is None):
                flatid = fid
            elif (fid > flatid) and (fid < id):
                flatid = fid
        skyname = graph_name(rawnight, "sky-{}{}-{:08d}".format(band, spec, id))
        starname = graph_name(rawnight, "stdstars-{}-{:08d}".format(spec, id))
        flatname = graph_name(rawnight, "fiberflat-{}{}-{:08d}".format(band, spec, fid))
        calname = graph_name(rawnight, "calib-{}{}-{:08d}".format(band, spec, id))
        node = {}
        node['type'] = 'calib'
        node['band'] = band
        node['spec'] = spec
        node['id'] = id
        node['in'] = [name, flatname, skyname, starname]
        node['out'] = []
        grph[calname] = node
        grph[flatname]['out'].append(calname)
        grph[skyname]['out'].append(calname)
        grph[starname]['out'].append(calname)
        nd['out'].append(calname)

    # Build cframe files

    for name, nd in grph.items():
        if nd['type'] != 'frame':
            continue
        if nd['flavor'] == 'flat':
            continue
        band = nd['band']
        spec = nd['spec']
        id = nd['id']
        cam = "{}{}".format(band, spec)
        flatid = None
        for fid in sorted(flatexpid[cam]):
            if (flatid is None):
                flatid = fid
            elif (fid > flatid) and (fid < id):
                flatid = fid
        skyname = graph_name(rawnight, "sky-{}{}-{:08d}".format(band, spec, id))
        flatname = graph_name(rawnight, "fiberflat-{}{}-{:08d}".format(band, spec, fid))
        calname = graph_name(rawnight, "calib-{}{}-{:08d}".format(band, spec, id))
        cfname = graph_name(rawnight, "cframe-{}{}-{:08d}".format(band, spec, id))
        node = {}
        node['type'] = 'cframe'
        node['band'] = band
        node['spec'] = spec
        node['id'] = id
        node['in'] = [name, flatname, skyname, calname]
        node['out'] = []
        grph[cfname] = node
        grph[flatname]['out'].append(cfname)
        grph[skyname]['out'].append(cfname)
        grph[calname]['out'].append(cfname)
        nd['out'].append(cfname)

    # Brick / Zbest dependencies

    for b in allbricks:
        zbname = "zbest-{}".format(b)
        inb = []
        for band in ['b', 'r', 'z']:
            node = {}
            node['type'] = 'brick'
            node['brick'] = b
            node['band'] = band
            node['in'] = []
            node['out'] = [zbname]
            bname = "brick-{}-{}".format(band, b)
            inb.append(bname)
            grph[bname] = node
        node = {}
        node['type'] = 'zbest'
        node['brick'] = b
        node['in'] = inb
        node['out'] = []
        grph[zbname] = node

    for name, nd in grph.items():
        if nd['type'] != 'fibermap':
            continue
        if nd['flavor'] == 'arc':
            continue
        if nd['flavor'] == 'flat':
            continue
        id = nd['id']
        bricks = nd['bricks']
        for band in ['b', 'r', 'z']:
            for spec in keep:
                cfname = graph_name(rawnight, "cframe-{}{}-{:08d}".format(band, spec, id))
                for b in bricks:
                    bname = "brick-{}-{}".format(band, b)
                    grph[bname]['in'].append(cfname)
                    grph[cfname]['out'].append(bname)

    return (grph, expcount, allbricks)


def graph_path_fibermap(rawdir, name):
    patstr = "([0-9]{{8}}){}(fibermap-[0-9]{{8}})".format(_graph_sep)
    pat = re.compile(patstr)
    mat = pat.match(name)
    if mat is None:
        raise RuntimeError("{} is not a valid fibermap name".format(name))
    night = mat.group(1)
    root = mat.group(2)
    path = os.path.join(rawdir, night, "{}.fits".format(root))
    return path


def graph_path_pix(rawdir, name):
    patstr = "([0-9]{{8}}){}(pix-[brz][0-9]-[0-9]{{8}})".format(_graph_sep)
    pat = re.compile(patstr)
    mat = pat.match(name)
    if mat is None:
        raise RuntimeError("{} is not a valid pix name".format(name))
    night = mat.group(1)
    root = mat.group(2)
    path = os.path.join(rawdir, night, "{}.fits".format(root))
    return path


def graph_path_psfboot(proddir, name):
    patstr = "([0-9]{{8}}){}(psfboot-[brz][0-9])".format(_graph_sep)
    pat = re.compile(patstr)
    mat = pat.match(name)
    if mat is None:
        raise RuntimeError("{} is not a valid psfboot name".format(name))
    night = mat.group(1)
    root = mat.group(2)
    dir = os.path.join(proddir, 'calib2d', 'psf', night)
    path = os.path.join(dir, "{}.fits".format(root))
    return path


def graph_path_psf(proddir, name):
    patstr = "([0-9]{{8}}){}psf-([brz][0-9])-([0-9]{{8}})".format(_graph_sep)
    pat = re.compile(patstr)
    mat = pat.match(name)
    if mat is None:
        raise RuntimeError("{} is not a valid psf name".format(name))
    night = mat.group(1)
    cam = mat.group(2)
    expid = mat.group(3)
    dir = os.path.join(proddir, 'exposures', night, expid)
    path = os.path.join(dir, "psf-{}-{}.fits".format(cam, expid))
    return path


def graph_path_psfnight(proddir, name):
    patstr = "([0-9]{{8}}){}(psfnight-[brz][0-9])".format(_graph_sep)
    pat = re.compile(patstr)
    mat = pat.match(name)
    if mat is None:
        raise RuntimeError("{} is not a valid psfnight name".format(name))
    night = mat.group(1)
    root = mat.group(2)
    dir = os.path.join(proddir, 'calib2d', 'psf', night)
    path = os.path.join(dir, "{}.fits".format(root))
    return path


def graph_path_frame(proddir, name):
    patstr = "([0-9]{{8}}){}frame-([brz][0-9])-([0-9]{{8}})".format(_graph_sep)
    pat = re.compile(patstr)
    mat = pat.match(name)
    if mat is None:
        raise RuntimeError("{} is not a valid frame name".format(name))
    night = mat.group(1)
    cam = mat.group(2)
    expid = mat.group(3)
    dir = os.path.join(proddir, 'exposures', night, expid)
    path = os.path.join(dir, "frame-{}-{}.fits".format(cam, expid))
    return path


def graph_path_fiberflat(proddir, name):
    patstr = "([0-9]{{8}}){}fiberflat-([brz][0-9])-([0-9]{{8}})".format(_graph_sep)
    pat = re.compile(patstr)
    mat = pat.match(name)
    if mat is None:
        raise RuntimeError("{} is not a valid fiberflat name".format(name))
    night = mat.group(1)
    cam = mat.group(2)
    expid = mat.group(3)
    dir = os.path.join(proddir, 'calib2d', night)
    path = os.path.join(dir, "fiberflat-{}-{}.fits".format(cam, expid))
    return path


def graph_path_sky(proddir, name):
    patstr = "([0-9]{{8}}){}sky-([brz][0-9])-([0-9]{{8}})".format(_graph_sep)
    pat = re.compile(patstr)
    mat = pat.match(name)
    if mat is None:
        raise RuntimeError("{} is not a valid sky name".format(name))
    night = mat.group(1)
    cam = mat.group(2)
    expid = mat.group(3)
    dir = os.path.join(proddir, 'exposures', night, expid)
    path = os.path.join(dir, "sky-{}-{}.fits".format(cam, expid))
    return path


def graph_path_stdstars(proddir, name):
    patstr = "([0-9]{{8}}){}stdstars-([0-9])-([0-9]{{8}})".format(_graph_sep)
    pat = re.compile(patstr)
    mat = pat.match(name)
    if mat is None:
        raise RuntimeError("{} is not a valid standard star name".format(name))
    night = mat.group(1)
    spec = mat.group(2)
    expid = mat.group(3)
    dir = os.path.join(proddir, 'exposures', night, expid)
    path = os.path.join(dir, "stdstars-{}-{}.fits".format(spec, expid))
    return path


def graph_path_calib(proddir, name):
    patstr = "([0-9]{{8}}){}calib-([brz][0-9])-([0-9]{{8}})".format(_graph_sep)
    pat = re.compile(patstr)
    mat = pat.match(name)
    if mat is None:
        raise RuntimeError("{} is not a valid calibration name".format(name))
    night = mat.group(1)
    cam = mat.group(2)
    expid = mat.group(3)
    dir = os.path.join(proddir, 'exposures', night, expid)
    path = os.path.join(dir, "calib-{}-{}.fits".format(cam, expid))
    return path


def graph_path_cframe(proddir, name):
    patstr = "([0-9]{{8}}){}cframe-([brz][0-9])-([0-9]{{8}})".format(_graph_sep)
    pat = re.compile(patstr)
    mat = pat.match(name)
    if mat is None:
        raise RuntimeError("{} is not a valid cframe name".format(name))
    night = mat.group(1)
    cam = mat.group(2)
    expid = mat.group(3)
    dir = os.path.join(proddir, 'exposures', night, expid)
    path = os.path.join(dir, "cframe-{}-{}.fits".format(cam, expid))
    return path


def graph_path_brick(proddir, name):
    patstr = "brick-([brz])-(.*)"
    pat = re.compile(patstr)
    mat = pat.match(name)
    if mat is None:
        raise RuntimeError("{} is not a valid brick name".format(name))
    band = mat.group(1)
    brick = mat.group(2)
    path = os.path.join(proddir, 'bricks', brick, "brick-{}-{}.fits".format(band, brick))
    return path


def graph_path_zbest(proddir, name):
    patstr = "zbest-(.*)"
    pat = re.compile(patstr)
    mat = pat.match(name)
    if mat is None:
        raise RuntimeError("{} is not a valid zbest name".format(name))
    brick = mat.group(1)
    path = os.path.join(proddir, 'bricks', brick, "zbest-{}.fits".format(brick))
    return path


def graph_path(rawdir, proddir, name, type):
    if type == 'fibermap':
        return graph_path_fibermap(rawdir, name)
    elif type == 'pix':
        return graph_path_pix(rawdir, name)
    elif type == 'psfboot':
        return graph_path_psfboot(proddir, name)
    elif type == 'psf':
        return graph_path_psf(proddir, name)
    elif type == 'psfnight':
        return graph_path_psfnight(proddir, name)
    elif type == 'frame':
        return graph_path_frame(proddir, name)
    elif type == 'fiberflat':
        return graph_path_fiberflat(proddir, name)
    elif type == 'sky':
        return graph_path_sky(proddir, name)
    elif type == 'stdstars':
        return graph_path_stdstars(proddir, name)
    elif type == 'calib':
        return graph_path_calib(proddir, name)
    elif type == 'cframe':
        return graph_path_cframe(proddir, name)
    elif type == 'brick':
        return graph_path_brick(proddir, name)
    elif type == 'zbest':
        return graph_path_zbest(proddir, name)
    elif type == 'night':
        return os.path.join(proddir, 'exposures', name)
    else:
        raise RuntimeError("unknown type {}".format(type))
    return ""


def graph_prune(grph, name, descend=False):
    # unlink from parents
    for p in grph[name]['in']:
        grph[p]['out'].remove(name)
    if descend:
        # recursively process children
        for c in grph[name]['out']:
            graph_prune(grph, c, descend=True)
    else:
        # not removing children, so only unlink
        for c in grph[name]['out']:
            grph[c]['in'].remove(name)
    del grph[name]
    return


def graph_mark(grph, name, state=None, descend=False):
    if descend:
        # recursively process children
        for c in grph[name]['out']:
            graph_mark(grph, c, state=state, descend=True)
    # set or clear state
    if state is None:
        if 'state' in grph[name].keys():
            del grph[name]['state']
    else:
        grph[name]['state'] = state
    return


def graph_slice(grph, names=None, types=None, deps=False):
    if types is None:
        types = graph_types

    newgrph = {}
    
    # First copy directly selected nodes
    for name, nd in grph.items():
        if (names is not None) and (name not in names):
            continue
        if nd['type'] not in types:
            continue
        newgrph[name] = copy.deepcopy(nd)

    # Now optionally grab all direct inputs
    if deps:
        for name, nd in newgrph.items():
            for p in nd['in']:
                if p not in newgrph.keys():
                    newgrph[p] = copy.deepcopy(grph[p])

    # Now remove links that we have pruned
    for name, nd in newgrph.items():
        newin = []
        for p in nd['in']:
            if p in newgrph.keys():
                newin.append(p)
        nd['in'] = newin
        newout = []
        for c in nd['out']:
            if c in newgrph.keys():
                newout.append(c)
        nd['out'] = newout

    return newgrph


def graph_slice_spec(grph, spectrographs=None):
    newgrph = copy.deepcopy(grph)
    if spectrographs is None:
        spectrographs = range(10)
    for name, nd in newgrph.items():
        if 'spec' in nd.keys():
            if int(nd['spec']) not in spectrographs:
                graph_prune(newgrph, name, descend=False)
    return newgrph


def graph_write(path, grph):
    with open(path, 'w') as f:
        yaml.dump(grph, f, default_flow_style=False)
    return


def graph_read(path):
    grph = None
    with open(path, 'r') as f:
        grph = yaml.load(f)
    return grph


def graph_read_prod(proddir, nightstr=None, spectrographs=None):

    plandir = os.path.join(proddir, 'plan')

    allnights = []
    planpat = re.compile(r'([0-9]{8})\.yaml')
    for root, dirs, files in os.walk(plandir, topdown=True):
        for f in files:
            planmat = planpat.match(f)
            if planmat is not None:
                night = planmat.group(1)
                allnights.append(night)
        break

    # select nights to use

    nights = None
    if nightstr is None:
        nights = allnights
    else:
        nights = select_nights(allnights, nightstr)

    # select the spectrographs to use

    spects = []
    if spectrographs is None:
        for s in range(10):
            spects.append(s)
    else:
        spc = spectrographs.split(',')
        for s in spc:
            spects.append(int(s))

    # load the graphs from selected nights and merge

    grph = {}
    for n in nights:
        nightfile = os.path.join(plandir, "{}.yaml".format(n))
        ngrph = graph_read(nightfile)
        sgrph = graph_slice_spec(ngrph, spectrographs=spects)
        grph.update(sgrph)

    return grph


def graph_dot(grph, f):
    # For visualization, we rank nodes of the same type together.

    rank = {}
    for t in graph_types:
        rank[t] = []

    for name, nd in grph.items():
        if nd['type'] not in graph_types:
            raise RuntimeError("graph node {} has invalid type {}".format(name, nd['type']))
        rank[nd['type']].append(name)

    tab = '    '
    f.write('\n// DESI Plan\n\n')
    f.write('digraph DESI {\n')
    f.write('splines=false;\n')
    f.write('overlap=false;\n')
    f.write('{}rankdir=LR\n'.format(tab))

    # organize nodes into subgraphs

    for t in graph_types:
        f.write('{}subgraph cluster{} {{\n'.format(tab, t))
        f.write('{}{}label="{}";\n'.format(tab, tab, t))
        f.write('{}{}newrank=true;\n'.format(tab, tab))
        f.write('{}{}rank=same;\n'.format(tab, tab))
        for name in sorted(rank[t]):
            nd = grph[name]
            props = "[shape=box,penwidth=3"
            if 'state' in nd.keys():
                props = "{},color=\"{}\"".format(props, _state_colors[nd['state']])
            else:
                props = "{},color=\"{}\"".format(props, _state_colors['none'])
            props = "{}]".format(props)
            f.write('{}{}"{}" {};\n'.format(tab, tab, name, props))
        f.write('{}}}\n'.format(tab))

    # write dependencies

    for t in graph_types:
        for name in sorted(rank[t]):
            for child in grph[name]['out']:
                f.write('{}"{}" -> "{}" [penwidth=1,color="#999999"];\n'.format(tab, name, child))

    # write rank grouping

    # for t in types:
    #     if (t == 'night') and len(rank[t]) == 1:
    #         continue
    #     f.write('{}{{ rank=same '.format(tab))
    #     for name in sorted(rank[t]):
    #         f.write('"{}" '.format(name))
    #     f.write(' }\n')

    f.write('}\n\n')

    return


def graph_merge_state(grph, comm=None):
    if comm is None:
        return
    elif comm.size == 1:
        return
    # check that we have the same list of nodes on all processes.  Then
    # merge the states.  "fail" overrides "None", and "done" overrides
    # them both.
    
    states = {}
    names = sorted(list(grph.keys()))
    for n in names:
        if 'state' in grph[n].keys():
            states[n] = grph[n]['state']
        else:
            states[n] = 'none'

    # print("proc {} has {} graph names".format(comm.rank, len(names)))
    # sys.stdout.flush()

    for p in range(1, comm.size):

        if comm.rank == 0:
            # print("proc {} receiving from {}".format(comm.rank, p))
            # sys.stdout.flush()
            pstates = comm.recv(source=p, tag=p)
            pnames = sorted(list(pstates.keys()))
            if pnames != names:
                raise RuntimeError("names of all objects must be the same when merging graph states")
            for n in names:
                if states[n] != 'none':
                    if states[n] != 'done':
                        if pstates[n] == 'done':
                            states[n] = 'done'
                else:
                    if pstates[n] != 'none':
                        states[n] = pstates[n]

        elif comm.rank == p:
            # print("proc {} sending to {}".format(comm.rank, 0))
            # sys.stdout.flush()
            comm.send(states, dest=0, tag=p)
        comm.barrier()

    # broadcast final merged state back to all processes.
    # print("proc {} hit bcast of states".format(comm.rank))
    # sys.stdout.flush()
    states = comm.bcast(states, root=0)

    # update process-local graph
    for n in names:
        if states[n] != 'none':
            grph[n]['state'] = states[n]

    # print("proc {} ending merge".format(comm.rank))
    # sys.stdout.flush()
    return

