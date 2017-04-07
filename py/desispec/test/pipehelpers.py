"""
Helper functions for pipeline tests.
"""

import os
import sys
import uuid
import shutil
import numpy as np

import desispec.io as io
import desispec.image as dimg

fake_env_cache = {'DESI_ROOT': None, 'DESI_SPECTRO_DATA': None,
                  'DESI_SPECTRO_REDUX': None, 'SPECPROD': None,
                  'DESIMODEL': None}

def fake_env(raw, redux, prod, model):
    global fake_env_cache
    for k in fake_env_cache:
        if k in os.environ:
            fake_env_cache[k] = os.environ[k]
    os.environ["DESI_SPECTRO_DATA"] = raw
    os.environ["DESI_ROOT"] = raw
    os.environ["DESI_SPECTRO_REDUX"] = redux
    os.environ["SPECPROD"] = prod
    os.environ["DESIMODEL"] = model
    return

def fake_env_clean():
    global fake_env_cache
    for k in fake_env_cache:
        if fake_env_cache[k] is None:
            del os.environ[k]
        else:
            os.environ[k] = fake_env_cache[k]
            fake_env_cache[k] = None
    return

def fake_redux(prod):
    dirhash = uuid.uuid4()
    reduxdir = "test_pipe_redux_{}".format(dirhash)
    if os.path.exists(reduxdir):
        shutil.rmtree(reduxdir)
    os.makedirs(reduxdir)
    proddir = os.path.join(reduxdir, prod)
    os.makedirs(proddir)
    return reduxdir


def fake_night():
    return "20170707"


def fake_raw():
    dirhash = uuid.uuid4()
    night = fake_night()
    rawdir = "test_pipe_raw_{}".format(dirhash)
    if os.path.exists(rawdir):
        shutil.rmtree(rawdir)
    os.makedirs(rawdir)

    nightdir = os.path.join(rawdir, night)

    # set up one spectrograph (500 fibers)
    nspec = 500

    # arc

    expid = "00000000"
    tileid = "0"
    flavor = "arc"
    telera = "0.0"
    teledec = "0.0"
    fmfile = os.path.join(nightdir, "fibermap-{}.fits".format(expid))

    hdr = dict(
        NIGHT = (night, 'Night of observation YEARMMDD'),
        EXPID = (expid, 'DESI exposure ID'),
        TILEID = (tileid, 'DESI tile ID'),
        FLAVOR = (flavor, 'Flavor [arc, flat, science, ...]'),
        TELRA = (telera, 'Telescope pointing RA [degrees]'),
        TELDEC = (teledec, 'Telescope pointing dec [degrees]'),
    )

    fibermap = io.empty_fibermap(nspec)
    fibermap['OBJTYPE'] = 'ARC'

    io.write_fibermap(fmfile, fibermap, header=hdr)

    for cam in ["r0", "b0", "z0"]:
        pfile = os.path.join(nightdir, "pix-{}-{}.fits".format(cam, expid))
        pix = np.random.normal(0, 3.0, size=(10,10))
        ivar = np.ones_like(pix) / 3.0**2
        mask = np.zeros(pix.shape, dtype=np.uint32)
        img = dimg.Image(pix, ivar, mask, camera=cam)
        io.write_image(pfile, img)

    # flat

    expid = "00000001"
    tileid = "1"
    flavor = "flat"
    telera = "0.0"
    teledec = "0.0"
    fmfile = os.path.join(nightdir, "fibermap-{}.fits".format(expid))

    hdr = dict(
        NIGHT = (night, 'Night of observation YEARMMDD'),
        EXPID = (expid, 'DESI exposure ID'),
        TILEID = (tileid, 'DESI tile ID'),
        FLAVOR = (flavor, 'Flavor [arc, flat, science, ...]'),
        TELRA = (telera, 'Telescope pointing RA [degrees]'),
        TELDEC = (teledec, 'Telescope pointing dec [degrees]'),
    )

    fibermap = io.empty_fibermap(nspec)
    fibermap['OBJTYPE'] = 'FLAT'

    io.write_fibermap(fmfile, fibermap, header=hdr)

    for cam in ["r0", "b0", "z0"]:
        pfile = os.path.join(nightdir, "pix-{}-{}.fits".format(cam, expid))
        pix = np.random.normal(0, 3.0, size=(10,10))
        ivar = np.ones_like(pix) / 3.0**2
        mask = np.zeros(pix.shape, dtype=np.uint32)
        img = dimg.Image(pix, ivar, mask, camera=cam)
        io.write_image(pfile, img)

    # science

    expid = "00000002"
    tileid = "2"
    flavor = "dark"
    telera = "0.0"
    teledec = "0.0"
    fmfile = os.path.join(nightdir, "fibermap-{}.fits".format(expid))

    hdr = dict(
        NIGHT = (night, 'Night of observation YEARMMDD'),
        EXPID = (expid, 'DESI exposure ID'),
        TILEID = (tileid, 'DESI tile ID'),
        FLAVOR = (flavor, 'Flavor [arc, flat, science, ...]'),
        TELRA = (telera, 'Telescope pointing RA [degrees]'),
        TELDEC = (teledec, 'Telescope pointing dec [degrees]'),
    )

    fibermap = io.empty_fibermap(nspec)
    fibermap['OBJTYPE'] = 'ELG'
    fibermap['FIBER'] = np.arange(nspec, dtype='i4')
    fibermap['TARGETID'] = np.random.randint(sys.maxsize, size=nspec)
    fibermap['BRICKNAME'] = [ '3412p195' for x in range(nspec) ]

    io.write_fibermap(fmfile, fibermap, header=hdr)

    for cam in ["r0", "b0", "z0"]:
        pfile = os.path.join(nightdir, "pix-{}-{}.fits".format(cam, expid))
        pix = np.random.normal(0, 3.0, size=(10,10))
        ivar = np.ones_like(pix) / 3.0**2
        mask = np.zeros(pix.shape, dtype=np.uint32)
        img = dimg.Image(pix, ivar, mask, camera=cam)
        io.write_image(pfile, img)

    return rawdir
