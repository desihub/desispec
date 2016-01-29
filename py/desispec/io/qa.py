"""
desispec.io.qa
===============

IO routines for QA
"""
import os, yaml

from desispec.qa.qa_exposure import QA_Frame
from desispec.io import findfile
from desispec.io.util import makepath
from desispec.log import get_logger

log=get_logger()

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
    qaframe = QA_Frame(flavor=qa_data['flavor'], camera=qa_data['camera'], in_data=qa_data)

    return qaframe


def load_qa_frame(filename, frame, flavor=None):
    """ Load an existing QA_Frame or generate one, as needed
    Args:
        filename: str
        frame: Frame object
        flavor: str, optional
          Type of QA_Frame

    Returns:
    qa_frame: QA_Frame object
    """
    if os.path.isfile(filename): # Read from file, if it exists
        qaframe = read_qa_frame(filename)
        log.info("Loaded QA file {:s}".format(filename))
        # Check camera
        try:
            camera = frame.meta['CAMERA']
        except:
            pass #
        else:
            if qaframe.camera != camera:
                raise ValueError('Wrong QA file!')
    else:  # Init
        qaframe = QA_Frame(frame)
    # Set flavor?
    if flavor is not None:
        qaframe.flavor = flavor
    # Return
    return qaframe
