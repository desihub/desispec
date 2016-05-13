"""
desispec.io.qa
===============

IO routines for QA
"""
import os, yaml

from desiutil.io import yamlify

from desispec.qa import QA_Frame
from desispec.qa import QA_Brick
from desispec.io import findfile
from desispec.io.util import makepath
from desispec.log import get_logger

log=get_logger()


def read_qa_data(filename):
    """Read data from a QA file
    """
    # Read yaml
    with open(filename, 'r') as infile:
        qa_data = yaml.load(infile)
    # Return
    return qa_data


def read_qa_brick(filename):
    """Generate a QA_Brick object from a data file
    """
    # Read
    qa_data = read_qa_data(filename)

    # Instantiate
    qabrick = QA_Brick(in_data=qa_data)

    return qabrick


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
        qaframe = QA_Frame(frame)
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
        yamlf.write( yaml.dump(ydict))#, default_flow_style=True) )

    return outfile

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
    ydict = yamlify(qaframe.data)
    with open(outfile, 'w') as yamlf:
        yamlf.write( yaml.dump(ydict))#, default_flow_style=True) )

    return outfile

