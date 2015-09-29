"""
desispec.io.qa
===============

IO routines for QA
"""
import os, yaml

from desispec.qa.qa_exposure import QA_Frame
from desispec.io import findfile
from desispec.io.util import fitsheader, native_endian, makepath

def write_qa_frame(outfile, qaframe):
    """Write QA for a given exposure

    Args:
        outfile : filename or (night, expid) tuple
        qa_exp : QA_Exposure object, with the following attributes
            _data: dict of QA info
            expid : Exposure id
            exptype : Exposure type
    """
    outfile = makepath(outfile, 'qa')

    # Simple yaml
    with open(outfile, 'w') as yamlf:
        yamlf.write( yaml.dump(qaframe.data))#, default_flow_style=True) )

    return outfile

def read_qa_frame(filename) :
    """Read qa_exposure and return QA_Frame object with attributes
    wave, flux, ivar, mask, header.
    
    skymodel.wave is 1D common wavelength grid, the others are 2D[nspec, nwave]
    """
    #- check if filename is (night, expid, camera) tuple instead
    if not isinstance(filename, basestring):
        night, expid, camera = filename
        filename = findfile('qa', night, expid, camera) 

    # Read yaml
    with open(filename, 'r') as infile:
        qa_data = yaml.load(infile)

    # Instantiate
    qaframe = QA_Frame(qa_data['flavor'], qa_data['camera'], in_data=qa_data)

    return qaframe