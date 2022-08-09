"""
desispec.darkfinder
====================

Reading and selecting calibration data from $DESI_SPECTRO_DARK using content of image headers
"""

import re
import os
import glob
import numpy as np
from sympy import use
import yaml
import os.path
from desispec.util import parse_int_args, header2night
from desiutil.log import get_logger
from desispec.calibfinder import CalibFinder
import astropy.io.fits as fits

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
    def __init__(self,headers,yaml_file=None, use_nth_newest=1) :
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
        
        if not os.path.isdir(self.directory):
            raise IOError("Dark directory {} not found".format(self.directory))

        #TODO: potentially add a checks here e.g. for files that are too early?
        
        log.debug("Use spectrograph hardware identifier SMY")
        cameraid    = "sm{}-{}".format(specid,camera[0].lower())

        dark_filelist = glob.glob("{}/dark_frames/*{}*.fits.gz".format(self.directory,cameraid))
        dark_filelist.sort()
        dark_filelist = np.array([f for f in dark_filelist if cameraid in f])

        #this step will do the matching again that calibfinder does
        super().__init__(headers, yaml_file)

        if len(dark_filelist) == 0:
            log.warning("Didn't find matching calibration darks in $DESI_SPECTRO_DARK using from $DESI_SPECTRO_CALIB instead")
            return

        bias_filelist = np.array([f.replace('dark','bias') for f in dark_filelist])
        dark_dates = np.array([int(f.split('-')[-1].split('.')[0]) for f in dark_filelist])

        log.debug(f"Finding matching dark frames for camera {cameraid} ...")


        #loop over all dark filenames

        #TODO: decide on how to define the version exactly
        #TODO: add an option to use the default dark or second newest dark or something else for doing comparisons

        log.debug("DATE-OBS=%d"%dateobs)
        found=False
        count=0
        for datebegin in sorted(dark_dates)[::-1] :
            if dateobs >= datebegin :
                #TODO: extra checks that evaluate if selection from yaml file is matching...
                date_used=datebegin
                dark_filename=dark_filelist[dark_dates == date_used][0]
                bias_filename=dark_filelist[dark_dates == date_used][0]
                with fits.open(dark_filename,'readonly') as hdul:
                    headers2=hdul[0].header
                    
                    #those check if the already matched ver (from calibfinder) that is stored in self.data is the same as the one from the dark file
                    if headers2["DETECTOR"].strip() != self.data["DETECTOR"].strip() :
                        log.debug("Skip file %s with DETECTOR=%s != %s"%(dark_filename,headers2["DETECTOR"],self.data["DETECTOR"]))
                        continue

                    if "CCDCFG" in self.data :
                        if headers2["CCDCFG"].strip() is None or headers2["CCDCFG"].strip() != self.data["CCDCFG"].strip() :
                            log.debug("Skip file %s with CCDCFG=%s != %s "%(dark_filename,headers2["CCDCFG"],self.data["CCDCFG"]))
                            continue

                    if "CCDTMING" in self.data :
                        if headers2["CCDTMING"].strip() is None or headers2["CCDTMING"].strip() != self.data["CCDTMING"].strip() :
                            log.debug("Skip file %s with CCDTMING=%s != %s "%(dark_filename,headers2["CCDTMING"],self.data["CCDTMING"]))
                            continue
                    #This check is not done with the bias frames assuming they are always generated in pairs
                count+=1
                found=True
                log.debug(f"Found matching dark frames for camera {cameraid} created on {date_used}")
                if count >= use_nth_newest:
                    break
                   
        if found and count >= use_nth_newest:
            #TODO: maybe just overwrite the selection from yaml file in case things are matching something newer...
            self.data.update({"DARK": dark_filename,
                              "BIAS": bias_filename})
        else:
            if found:
                log.warning(f"Didn't find an {use_nth_newest}-th newest calibration with required setup, using $DESI_SPECTRO_CALIB darks instead")
            else:
                log.warning("Didn't find matching calibration darks in $DESI_SPECTRO_DARK using from $DESI_SPECTRO_CALIB instead")


        