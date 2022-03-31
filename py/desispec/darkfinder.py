"""
desispec.darkfinder
====================

Reading and selecting calibration data from $DESI_SPECTRO_DARK using content of image headers
"""

import re
import os
import numpy as np
import yaml
import os.path
from desispec.util import parse_int_args, header2night
from desiutil.log import get_logger
from calibfinder import CalibFinder

def parse_date_obs(value):
    '''
    converts DATE-OBS keywork to int
    with for instance DATE-OBS=2016-12-21T18:06:21.268371-05:00
    '''
    m = re.search(r'(\d+)-(\d+)-(\d+)T', value)
    Y,M,D=tuple(map(int, m.groups()))
    dateobs=int(Y*10000+M*100+D)
    return dateobs

_sp2sm = None
_sm2sp = None
def _load_smsp():
    """
    Loads $DESI_SPECTRO_CALIB/spec/smsp.txt into global _sp2sm and _sm2sp dicts
    """
    global _sp2sm
    global _sm2sp

    tmp_sp2sm = dict()
    tmp_sm2sp = dict()

    filename = os.getenv('DESI_SPECTRO_CALIB') + "/spec/smsp.txt"
    tmp = np.loadtxt(filename, dtype=str)
    for sp, sm in tmp:
        p = int(sp[2:])
        m = int(sm[2:])
        tmp_sp2sm[str(sp)] = str(sm)
        tmp_sm2sp[str(sm)] = str(sp)
        tmp_sp2sm[p] = m
        tmp_sm2sp[m] = p

    #- Assign to global variables only after successful loading and parsing
    _sp2sm = tmp_sp2sm
    _sm2sp = tmp_sm2sp

def sp2sm(sp):
    """
    Converts spectrograph sp logical number to sm hardware number

    Args:
        sp : spectrograph int 0-9 or str sp[0-9]

    Returns "smM" if input is str "spP", or int M if input is int P

    Note: uses $DESI_SPECTRO_CALIB/spec/smsp.txt

    TODO: add support for different mappings based on night
    """
    global _sp2sm
    if _sp2sm is None:
        _load_smsp()

    return _sp2sm[sp]

def sm2sp(sm, night=None):
    """
    Converts spectrograph sm hardware number to sp logical number

    Args:
        sm : spectrograph sm number 1-10 or str sm[1-10]

    Returns "spP" if input is str "smM", or int P if input is int M

    Note: uses $DESI_SPECTRO_CALIB/spec/smsp.txt

    TODO: add support for different mappings based on night
    """
    global _sm2sp
    if _sm2sp is None:
        _load_smsp()

    return _sm2sp[sm]

def finddarkfile(headers,key,yaml_file=None) :
    """
    read and select calibration data file from $DESI_SPECTRO_DARK using the keywords found in the headers

    Args:
        headers: list of fits headers, or list of dictionnaries
        key: type of calib file, e.g. 'PSF' or 'FIBERFLAT'

    Optional:
            yaml_file: path to a specific yaml file. By default, the code will
            automatically find the yaml file from the environment variable
            DESI_SPECTRO_DARK and the CAMERA keyword in the headers

    Returns path to calibration file
    """
    cfinder = DarkFinder(headers,yaml_file)
    if cfinder.haskey(key) :
        return cfinder.findfile(key)
    else :
        return None

class DarkFinder(CalibFinder) :


    def __init__(self,headers,yaml_file=None) :
        """
        Class to read and select calibration data from $DESI_SPECTRO_CALIB using the keywords found in the headers

        Args:
            headers: list of fits headers, or list of dictionnaries

        Optional:
            yaml_file: path to a specific yaml file. By default, the code will
            automatically find the yaml file from the environment variable
            DESI_SPECTRO_CALIB and the CAMERA keyword in the headers

        """
        log = get_logger()

        # temporary backward compatibility
        if not "DESI_SPECTRO_DARK" in os.environ :
                log.error("Need environment variable DESI_SPECTRO_DARK")
                raise KeyError("Need environment variable DESI_SPECTRO_DARK")
        else :
            self.directory = os.environ["DESI_SPECTRO_DARK"]

        if len(headers)==0 :
            log.error("Need at least a header")
            raise RuntimeError("Need at least a header")

        header=dict()
        for other_header in headers :
            for k in other_header :
                if k not in header :
                    try :
                        header[k]=other_header[k]
                    except KeyError :
                        # it happens with the current version of fitsio
                        # if the value = 'None'.
                        pass

        #Maybe skip all those checks assuming that CalibFinder has been run already...
        if "CAMERA" not in header :
            log.error("no 'CAMERA' keyword in header, cannot find dark")
            log.error("header is:")
            for k in header :
                log.error("{} : {}".format(k,header[k]))
            raise KeyError("no 'CAMERA' keyword in header, cannot find dark")

        log.debug("header['CAMERA']={}".format(header['CAMERA']))
        camera=header["CAMERA"].strip().lower()

        if "SPECID" in header :
            log.debug("header['SPECID']={}".format(header['SPECID']))
            specid=int(header["SPECID"])
        else :
            specid=None

        dateobs = header2night(header)
        
        #some of those might be useful to validate the config vs info stored in DARKs?
        """ 
        detector=header["DETECTOR"].strip()
        if "CCDCFG" in header :
            ccdcfg = header["CCDCFG"].strip()
        else :
            ccdcfg = None
        if "CCDTMING" in header :
            ccdtming = header["CCDTMING"].strip()
        else :
            ccdtming = None 

        # Support simulated data even if $DESI_SPECTRO_CALIB points to
        # real data calibrations
        self.directory = os.path.normpath(self.directory)  # strip trailing /
        if detector == "SIM" and (not self.directory.endswith("sim")) :
            newdir = os.path.join(self.directory, "sim")
            if os.path.isdir(newdir) :
                self.directory = newdir"""

        if not os.path.isdir(self.directory):
            raise IOError("Dark directory {} not found".format(self.directory))

        #TODO: potentially add a checks here e.g. for files that are too early?
        
        log.debug("Use spectrograph hardware identifier SMY")
        cameraid    = "sm{}-{}".format(specid,camera[0].lower())
        if yaml_file is None :
            yaml_file = "{}/dark_config/{}_dark.yaml".format(self.directory,specid,cameraid)

        if not os.path.isfile(yaml_file) :
            log.error("Cannot read {}".format(yaml_file))
            raise IOError("Cannot read {}".format(yaml_file))


        log.debug("reading dark config data in {}".format(yaml_file))

        stream = open(yaml_file, 'r')
        data   = yaml.safe_load(stream)
        stream.close()

        #TODO: potentially add a check here e.g. for matching the config of the DARKs to the CCDCONFIG


        if not cameraid in data :
            log.error("Cannot find data for camera %s in filename %s"%(cameraid,yaml_file))
            raise KeyError("Cannot find  data for camera %s in filename %s"%(cameraid,yaml_file))

        data=data[cameraid]
        log.debug("Found %d data for camera %s in filename %s"%(len(data),cameraid,yaml_file))
        log.debug("Finding matching version ...")
        
        #TODO: decide on how to define the version exactly
        log.debug("DATE-OBS=%d"%dateobs)
        found=False
        matching_data=None
        for version in data :
            log.debug("Checking version %s"%version)
            datebegin=int(data[version]["DATE-OBS-BEGIN"])
            if dateobs < datebegin :
                log.debug("Skip version %s with DATE-OBS-BEGIN=%d > DATE-OBS=%d"%(version,datebegin,dateobs))
                continue
            if "DATE-OBS-END" in data[version] and data[version]["DATE-OBS-END"].lower() != "none" :
                dateend=int(data[version]["DATE-OBS-END"])
                if dateobs > dateend :
                    log.debug("Skip version %s with DATE-OBS-END=%d < DATE-OBS=%d"%(version,datebegin,dateobs))
                    continue

            log.debug("Found data version %s for camera %s in %s"%(version,cameraid,yaml_file))
            if found :
                log.error("But we already has a match. Please fix this ambiguity in %s"%yaml_file)
                raise KeyError("Duplicate possible calibration data. Please fix this ambiguity in %s"%yaml_file)
            found=True
            matching_data=data[version]

        if not found :
            log.error("Didn't find matching calibration data in %s"%(yaml_file))
            raise KeyError("Didn't find matching calibration data in %s"%(yaml_file))


        self.data = matching_data