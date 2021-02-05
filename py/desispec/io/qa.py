"""
desispec.io.qa
===============

IO routines for QA
"""
from __future__ import print_function, absolute_import, division

import os, yaml
import json

from desiutil.io import yamlify

from desispec.io import findfile, read_meta_frame
from desispec.io.util import makepath
from desiutil.log import get_logger
# log=get_logger()


def qafile_from_framefile(frame_file, qaprod_dir=None, output_dir=None):
    """ Derive the QA filename from an input frame file
    Args:
        frame_file: str
        output_dir: str, optional   Over-ride default output path
        qa_dir: str, optional   Over-ride default QA

    Returns:

    """
    frame_meta = read_meta_frame(frame_file)
    night = frame_meta['NIGHT'].strip()
    camera = frame_meta['CAMERA'].strip()
    expid = int(frame_meta['EXPID'])
    if frame_meta['FLAVOR'] in ['flat', 'arc']:
        qatype = 'qa_calib'
    else:
        qatype = 'qa_data'
    # Name
    qafile = findfile(qatype, night=night, camera=camera, expid=expid,
                      outdir=output_dir, qaprod_dir=qaprod_dir)
    # Return
    return qafile, qatype


def read_qa_data(filename):
    """Read data from a QA file
    """
    # Read yaml
    with open(filename, 'r') as infile:
        qa_data = yaml.safe_load(infile)
    # Convert expid to int
    for night in qa_data.keys():
        for expid in list(qa_data[night].keys()):
            if isinstance(expid,str):
                qa_data[night][int(expid)] = qa_data[night][expid].copy()
                qa_data[night].pop(expid)
    # Return
    return qa_data


def read_qa_brick(filename):
    """Generate a QA_Brick object from a data file
    """
    from desispec.qa.qa_brick import QA_Brick
    # Read
    qa_data = read_qa_data(filename)

    # Instantiate
    qabrick = QA_Brick(in_data=qa_data)

    return qabrick


def read_qa_frame(filename):
    """Generate a QA_Frame object from a data file
    """
    from desispec.qa.qa_frame import QA_Frame
    #- check if filename is (night, expid, camera) tuple instead
    if not isinstance(filename, str):
        night, expid, camera = filename
        filename = findfile('qa', night, expid, camera)

    # Read
    qa_data = read_qa_data(filename)

    # Instantiate
    qaframe = QA_Frame(qa_data)

    return qaframe


def load_qa_frame(filename, frame_meta=None, flavor=None):
    """ Load an existing QA_Frame or generate one, as needed

    Args:
        filename: str
        frame_meta: dict like, optional
        flavor: str, optional
            Type of QA_Frame

    Returns:
        qa_frame: QA_Frame object
    """
    from desispec.qa.qa_frame import QA_Frame
    log=get_logger()
    if os.path.isfile(filename): # Read from file, if it exists
        qaframe = read_qa_frame(filename)
        log.info("Loaded QA file {:s}".format(filename))
        # Check against frame, if provided
        if frame_meta is not None:
            for key in ['camera','expid','night','flavor']:
                assert str(getattr(qaframe, key)) == str(frame_meta[key.upper()])
    else:  # Init
        if frame_meta is None:
            log.error("QA file {:s} does not exist.  Expecting frame input".format(filename))
        qaframe = QA_Frame(frame_meta)
    # Set flavor?
    if flavor is not None:
        qaframe.flavor = flavor
    # Return
    return qaframe


def load_qa_brick(filename):
    """ Load an existing QA_Brick or generate one, as needed
    Args:
        filename: str

    Returns:
    qa_brick: QA_Brick object
    """
    from desispec.qa.qa_brick import QA_Brick
    log=get_logger()
    if os.path.isfile(filename): # Read from file, if it exists
        qabrick = read_qa_brick(filename)
        log.info("Loaded QA file {:s}".format(filename))
    else:  # Init
        qabrick = QA_Brick()
    # Return
    return qabrick

def write_qa_brick(outfile, qabrick):
    """Write QA for a given exposure

    Args:
        outfile : filename
        qabrick : QA_Brick object
            _data: dict of QA info
    """
    outfile = makepath(outfile, 'qa')

    # Simple yaml
    ydict = yamlify(qabrick.data)
    with open(outfile, 'w') as yamlf:
        yamlf.write(yaml.dump(ydict))#, default_flow_style=True) )

    return outfile


def write_qa_frame(outfile, qaframe, verbose=False):
    """Write QA for a given frame

    Args:
        outfile : str
          filename
        qa_exp : QA_Frame object, with the following attributes
            qa_data: dict of QA info
    """
    log=get_logger()
    outfile = makepath(outfile, 'qa')

    # Generate the dict
    odict = {qaframe.night: {qaframe.expid: {qaframe.camera: {}, 'flavor': qaframe.flavor}}}
    odict[qaframe.night][qaframe.expid][qaframe.camera] = qaframe.qa_data
    ydict = yamlify(odict)
    # Simple yaml
    with open(outfile, 'w') as yamlf:
        yamlf.write(yaml.dump(ydict))
    if verbose:
        log.info("Wrote QA frame file: {:s}".format(outfile))

    return outfile


def write_qa_exposure(outroot, qaexp, ret_dict=False):
    """Write QA for a given exposure

    Args:
        outroot : str
          filename without format extension
        qa_exp : QA_Exposure object
        ret_dict : bool, optional
          Return dict only?  [for qa_prod, mainly]
    Returns:
        outfile or odict : str or dict
    """
    # Generate the dict
    odict = {qaexp.night: {qaexp.expid: {}}}
    odict[qaexp.night][qaexp.expid]['flavor'] = qaexp.flavor
    odict[qaexp.night][qaexp.expid]['meta'] = qaexp.meta
    cameras = list(qaexp.data['frames'].keys())
    for camera in cameras:
        odict[qaexp.night][qaexp.expid][camera] = qaexp.data['frames'][camera]
    # Return dict only?
    if ret_dict:
        return odict
    # Simple yaml
    ydict = yamlify(odict)
    outfile = outroot+'.yaml'
    outfile = makepath(outfile, 'qa')
    with open(outfile, 'w') as yamlf:
        yamlf.write( yaml.dump(ydict))#, default_flow_style=True) )

    return outfile


def load_qa_multiexp(inroot):
    """Load QA for a given production

    Args:
        inroot : str
          base filename without format extension
    Returns:
        odict : dict
    """
    log=get_logger()
    infile = inroot+'.json'
    log.info("Loading QA prod file: {:s}".format(infile))
    # Read
    if not os.path.exists(infile):
        log.info("QA prod file {:s} does not exist!".format(infile))
        log.error("You probably need to generate it with desi_qa_prod --make_frameqa=3 --slurp")
    with open(infile, 'rt') as fh:
        odict = json.load(fh)
    # Return
    return odict


def write_qa_multiexp(outroot, mdict, indent=True):
    """Write QA for a given production

    Args:
        outroot : str
          filename without format extension
        mdict : dict

    Returns:
        outfile: str
          output filename
    """
    log=get_logger()
    outfile = outroot+'.json'
    outfile = makepath(outfile, 'qa')

    ydict = yamlify(mdict)  # This works well for JSON too
    # Simple json
    with open(outfile, 'wt') as fh:
        json.dump(ydict, fh, indent=indent)
    log.info('Wrote QA Multi-Exposure file: {:s}'.format(outfile))

    return outfile


def write_qa_ql(outfile, qaresult):
    """Write QL output files

       Args:
           outfile : str
             filename to be written (yaml)
           qaresult : dict
             QAresults from run_qa()

       Returns:
           outfile : str
    """
    #import yaml
    #from desiutil.io import yamlify
    # Take in QL input and output to yaml
    #SE:  No yaml creation as of May 2018
    qadict = yamlify(qaresult)
    #f=open(outfile,"w")
    #f.write(yaml.dump(qadict))
    #f.close()
    
    g=open(outfile,"w")
    json.dump(qadict, g, sort_keys=True, indent=4)
    g.close()    
    
    return outfile


