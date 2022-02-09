"""
desispec.calibfinder
====================

Reading and selecting calibration data from $DESI_SPECTRO_CALIB using content of image headers
"""

import re
import os
import numpy as np
import yaml
import os.path
from desispec.util import parse_int_args, header2night
from desiutil.log import get_logger

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

def findcalibfile(headers,key,yaml_file=None) :
    """
    read and select calibration data file from $DESI_SPECTRO_CALIB using the keywords found in the headers

    Args:
        headers: list of fits headers, or list of dictionnaries
        key: type of calib file, e.g. 'PSF' or 'FIBERFLAT'

    Optional:
            yaml_file: path to a specific yaml file. By default, the code will
            automatically find the yaml file from the environment variable
            DESI_SPECTRO_CALIB and the CAMERA keyword in the headers

    Returns path to calibration file
    """
    cfinder = CalibFinder(headers,yaml_file)
    if cfinder.haskey(key) :
        return cfinder.findfile(key)
    else :
        return None

def badfibers(headers,keys=["BROKENFIBERS","BADCOLUMNFIBERS","LOWTRANSMISSIONFIBERS"],yaml_file=None) :
    """
    find list of bad fibers from $DESI_SPECTRO_CALIB using the keywords found in the headers

    Args:
        headers: list of fits headers, or list of dictionnaries

    Optional:
        keys: list of keywords, among ["BROKENFIBERS","BADCOLUMNFIBERS","LOWTRANSMISSIONFIBERS"]. Default is all of them.
        yaml_file: path to a specific yaml file. By default, the code will
        automatically find the yaml file from the environment variable
        DESI_SPECTRO_CALIB and the CAMERA keyword in the headers

    Returns List of bad fibers as a 1D array of intergers
    """
    cfinder = CalibFinder(headers,yaml_file)
    return cfinder.badfibers(keys)

class CalibFinder() :


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

        old_version = False

        # temporary backward compatibility
        if not "DESI_SPECTRO_CALIB" in os.environ :
            if "DESI_CCD_CALIBRATION_DATA" in os.environ :
                log.warning("Using deprecated DESI_CCD_CALIBRATION_DATA env. variable to find calibration data\nPlease switch to DESI_SPECTRO_CALIB a.s.a.p.")
                self.directory = os.environ["DESI_CCD_CALIBRATION_DATA"]
                old_version = True
            else :
                log.error("Need environment variable DESI_SPECTRO_CALIB")
                raise KeyError("Need environment variable DESI_SPECTRO_CALIB")
        else :
            self.directory = os.environ["DESI_SPECTRO_CALIB"]

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
        if "CAMERA" not in header :
            log.error("no 'CAMERA' keyword in header, cannot find calib")
            log.error("header is:")
            for k in header :
                log.error("{} : {}".format(k,header[k]))
            raise KeyError("no 'CAMERA' keyword in header, cannot find calib")

        log.debug("header['CAMERA']={}".format(header['CAMERA']))
        camera=header["CAMERA"].strip().lower()

        if "SPECID" in header :
            log.debug("header['SPECID']={}".format(header['SPECID']))
            specid=int(header["SPECID"])
        else :
            specid=None

        dateobs = header2night(header)

        detector=header["DETECTOR"].strip()
        if "CCDCFG" in header :
            ccdcfg = header["CCDCFG"].strip()
        else :
            ccdcfg = None
        if "CCDTMING" in header :
            ccdtming = header["CCDTMING"].strip()
        else :
            ccdtming = None

        #if "DOSVER" in header :
        #    dosver = str(header["DOSVER"]).strip()
        #else :
        #    dosver = None
        #if "FEEVER" in header :
        #    feever = str(header["FEEVER"]).strip()
        #else :
        #    feever = None

        # Support simulated data even if $DESI_SPECTRO_CALIB points to
        # real data calibrations
        self.directory = os.path.normpath(self.directory)  # strip trailing /
        if detector == "SIM" and (not self.directory.endswith("sim")) :
            newdir = os.path.join(self.directory, "sim")
            if os.path.isdir(newdir) :
                self.directory = newdir

        if not os.path.isdir(self.directory):
            raise IOError("Calibration directory {} not found".format(self.directory))


        if dateobs < 20191211 or detector == 'SIM': # old spectro identifiers
            cameraid = camera
            spectro=int(camera[-1])
            if yaml_file is None :
                if old_version :
                    yaml_file = os.path.join(self.directory,"ccd_calibration.yaml")
                else :
                    yaml_file = "{}/spec/sp{}/{}.yaml".format(self.directory,spectro,cameraid)
        else :
            if specid is None :
                log.error("dateobs = {} >= 20191211 but no SPECID keyword in header!".format(dateobs))
                raise RuntimeError("dateobs = {} >= 20191211 but no SPECID keyword in header!".format(dateobs))
            log.debug("Use spectrograph hardware identifier SMY")
            cameraid    = "sm{}-{}".format(specid,camera[0].lower())
            if yaml_file is None :
                yaml_file = "{}/spec/sm{}/{}.yaml".format(self.directory,specid,cameraid)

        if not os.path.isfile(yaml_file) :
            log.error("Cannot read {}".format(yaml_file))
            raise IOError("Cannot read {}".format(yaml_file))


        log.debug("reading calib data in {}".format(yaml_file))

        stream = open(yaml_file, 'r')
        data   = yaml.safe_load(stream)
        stream.close()


        if not cameraid in data :
            log.error("Cannot find data for camera %s in filename %s"%(cameraid,yaml_file))
            raise KeyError("Cannot find  data for camera %s in filename %s"%(cameraid,yaml_file))

        data=data[cameraid]
        log.debug("Found %d data for camera %s in filename %s"%(len(data),cameraid,yaml_file))
        log.debug("Finding matching version ...")
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
            if detector != data[version]["DETECTOR"].strip() :
                log.debug("Skip version %s with DETECTOR=%s != %s"%(version,data[version]["DETECTOR"],detector))
                continue

            if "CCDCFG" in data[version] :
                if ccdcfg is None or ccdcfg != data[version]["CCDCFG"].strip() :
                    log.debug("Skip version %s with CCDCFG=%s != %s "%(version,data[version]["CCDCFG"],ccdcfg))
                    continue

            if "CCDTMING" in data[version] :
                if ccdtming is None or ccdtming != data[version]["CCDTMING"].strip() :
                    log.debug("Skip version %s with CCDTMING=%s != %s "%(version,data[version]["CCDTMING"],ccdtming))
                    continue



            #if dosver is not None and "DOSVER" in data[version] and dosver != str(data[version]["DOSVER"]).strip() :
            #     log.debug("Skip version %s with DOSVER=%s != %s "%(version,data[version]["DOSVER"],dosver))
            #    continue
            #if feever is not None and  "FEEVER" in data[version] and feever != str(data[version]["FEEVER"]).strip() :
            #    log.debug("Skip version %s with FEEVER=%s != %s"%(version,data[version]["FEEVER"],feever))
            #   continue

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

    def haskey(self,key) :
        """
        Args:
            key: keyword, string, like 'GAINA'
        Returns:
            yes or no, boolean
        """
        return ( key in self.data )

    def value(self,key) :
        """
        Args:
            key: header keyword, string, like 'GAINA'
        Returns:
            data found in yaml file
        """
        return self.data[key]

    def findfile(self,key) :
        """
        Args:
            key: header keyword, string, like 'DARK'
        Returns:
            path to calibration file
        """
        return os.path.join(self.directory,self.data[key])

    def badfibers(self,keys=["BROKENFIBERS","BADCOLUMNFIBERS","LOWTRANSMISSIONFIBERS","BADAMPFIBERS","EXCLUDEFIBERS"]) :
        """
        Args:
            keys: ptional, list of keywords, among BROKENFIBERS,BADCOLUMNFIBERS,LOWTRANSMISSIONFIBERS,BADAMPFIBERS,EXCLUDEFIBERS. Default is all of them.

        Returns:
            List of bad fibers from yaml file as a 1D array of intergers
        """
        log = get_logger()
        fibers=[]
        badfiber_keywords=["BROKENFIBERS","BADCOLUMNFIBERS","LOWTRANSMISSIONFIBERS","BADAMPFIBERS","EXCLUDEFIBERS"]
        for key in keys :
            if key not in badfiber_keywords  :
                log.error(f"key '{key}' not in the list of valid keys for bad fibers: {validkeys}")
                continue
            if self.haskey(key) :
                val = self.value(key)
                fibers.append(parse_int_args(val))
        if len(fibers)==0 :
            return np.array([],dtype=int)
        return np.unique(np.hstack(fibers))
