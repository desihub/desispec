"""
desispec.io.qa
===============

IO routines for QA
"""
import os, yaml

from desispec.qa import QA_Frame
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


def read_qa_data(filename):
    """Read data from a QA file
    """
    # Read yaml
    with open(filename, 'r') as infile:
        qa_data = yaml.load(infile)
    # Return
    return qa_data

def read_qa_frame(filename):
    """Generate a QA_Frame object from a data file
    """
    #- check if filename is (night, expid, camera) tuple instead
    if not isinstance(filename, basestring):
        night, expid, camera = filename
        filename = findfile('qa', night, expid, camera) 

    # Read
    qa_data = read_qa_data(filename)

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
        import pdb
        pdb.set_trace()
        qaframe = QA_Frame(frame)
    # Set flavor?
    if flavor is not None:
        qaframe.flavor = flavor
    # Return
    return qaframe
