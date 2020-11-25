#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
from astropy.table import Table
from astropy.io import fits
## Import some helper functions, you can see their definitions by uncomenting the bash shell command
from desispec.workflow.utils import define_variable_from_environment, pathjoin, give_relevant_details, get_json_dict


def get_exposure_table_column_defs(return_default_values=False):
    ## Define the column names for the exposure table and their respective datatypes, split in two
    ##     only for readability's sake
    colnames1 = ['EXPID', 'EXPTIME', 'OBSTYPE', 'SPECTROGRAPHS', 'CAMWORD', 'TILEID']
    coltypes1 = [int, float, 'S8', 'S10', 'S30', int]
    coldeflt1 = [-99, 0.0, 'unknown', '0123456789', 'a09123456789', -99]

    colnames2 = ['NIGHT', 'EXPFLAG', 'HEADERERR', 'SURVEY', 'SEQNUM', 'SEQTOT', 'PROGRAM', 'MJD-OBS']
    coltypes2 = [int, int, np.ndarray, int, int, int, 'S30', float]
    coldeflt2 = [20000101, 0, np.array([], dtype=str), 0, 1, 1, 'unknown', 50000.0]

    colnames3 = ['REQRA', 'REQDEC', 'TARGTRA', 'TARGTDEC', 'COMMENTS']
    coltypes3 = [float, float, float, float, np.ndarray]
    coldeflt3 = [-99.99, -89.99, -99.99, -89.99, np.array([], dtype=str)]

    colnames = colnames1 + colnames2 + colnames3
    coldtypes = coltypes1 + coltypes2 + coltypes3
    coldeflts = coldeflt1 + coldeflt2 + coldeflt3

    if return_default_values:
        return colnames, coldtypes, coldeflts
    else:
        return colnames, coldtypes

def default_exptypes_for_exptable():
    ## Define the science types to be included in the exposure table (case insensitive)
    return ['arc','flat','twilight','science','sci','dither','dark','bias','zero']

def get_survey_definitions():
    ## Create a rudimentary way of assigning "SURVEY keywords based on what date range a night falls into"
    survey_def = {0: (20200201, 20200315), 1: (
        20201201, 20210401)}  # 0 is CMX, 1 is SV1, 2 is SV2, ..., 99 is any testing not in these timeframes
    return survey_def

def get_surveynum(night, survey_definitions=None):
    if survey_definitions is None:
        survey_definitions = get_survey_definitions()
    for survey, (low, high) in survey_definitions.items():
        if night >= low and night <= high:
            return survey
    return 99

def night_to_month(night):
    return str(night)[:-2]




def get_exposure_table_name(night=None, extension='csv'):
    # if night is None and 'PROD_NIGHT' in os.environ:
    #     night = os.environp['PROD_NIGHT']
    return f'exposure_table_{night}.{extension}'

def get_exposure_table_path(night=None):
    # if night is None and 'PROD_NIGHT' in os.environ:
    #     night = os.environp['PROD_NIGHT']
    spec_redux = define_variable_from_environment(env_name='DESI_SPECTRO_REDUX',
                                                          var_descr="The exposure table path")
    # subdir = define_variable_from_environment(env_name='USER', var_descr="Username for unique exposure table directories")
    subdir = define_variable_from_environment(env_name='SPECPROD', var_descr="Use SPECPROD for unique exposure table directories")
    if night is None:
        return pathjoin(spec_redux,subdir,'exposure_tables')
    else:
        month = night_to_month(night)
        path = pathjoin(spec_redux,subdir,'exposure_tables',month)
        return path

def get_exposure_table_pathname(night=None, extension='csv'):#base_path,prodname
    # if night is None and 'PROD_NIGHT' in os.environ:
    #     night = os.environp['PROD_NIGHT']
    path = get_exposure_table_path(night)
    table_name = get_exposure_table_name(night, extension)
    return pathjoin(path,table_name)

def instantiate_exposure_table(rows=None):
    colnames, coldtypes = get_exposure_table_column_defs()
    outtab = Table(names=colnames,dtype=coldtypes)
    if rows is not None:
        for row in rows:
            outtab.add_row(row)
    return outtab


def summarize_exposure(raw_data_dir, night, exp, scitypes, surveynum, colnames, coldefaults, verbosely=False):
    if type(exp) is not str:
        exp = int(exp)
        exp = f'{exp:08d}'
    night = str(night)
    def give_details(verbose_output, non_verbose_output=None):
        give_relevant_details(verbose_output, non_verbose_output, verbosely=verbosely)

    if verbosely:
        print(f'\n############### {exp} ###################')
    ## Request json file is first used to quickly identify science exposures
    ## If a request file doesn't exist for an exposure, it shouldn't be an exposure we care about
    reqpath = pathjoin(raw_data_dir, night, exp, f'request-{exp}.json')
    if not os.path.isfile(reqpath):
        give_details(f'{reqpath} did not exist!', f'{exp}: skipped  -- request not found')
        return None

    ## Load the json file in as a dictionary
    req_dict = get_json_dict(reqpath)

    ## Check to see if it is a manifest file for calibrations
    if "SEQUENCE" in req_dict and req_dict["SEQUENCE"].lower() == "manifest":
        if 'PROGRAM' in req_dict:
            prog = req_dict['PROGRAM'].lower()
            if 'calib' in prog and 'done' in prog:
                if 'short' in prog:
                    return "short calib complete"
                elif 'long' in prog:
                    return "long calib complete"
                elif 'arc' in prog:
                    return 'arc calib complete'
                else:
                    pass

    ## If FLAVOR is wrong or no obstype is defines, skip it
    if 'FLAVOR' not in req_dict.keys():
        give_details(f'WARNING: {reqpath} -- flavor not given!', f'{exp}: skipped  -- flavor not given!')
        return None

    flavor = req_dict['FLAVOR'].lower()
    if flavor != 'science' and 'dark' not in scitypes and 'zero' not in scitypes:
        ## If FLAVOR is wrong
        give_details(f'ignoring: {reqpath} -- {flavor} not a flavor we care about', f'{exp}: skipped  -- not science')
        return None

    if 'OBSTYPE' not in req_dict.keys():
        ## If no obstype is defines, skip it
        give_details(f'ignoring: {reqpath} -- {flavor} flavor but obstype not defined',
                     f'{exp}: skipped  -- obstype not given')
        return None
    else:
        give_details(f'using: {reqpath}')

    ## If obstype isn't in our list of ones we care about, skip it
    obstype = req_dict['OBSTYPE'].lower()
    if obstype in scitypes:
        ## Look for the data. If it's not there, say so then move on
        datapath = pathjoin(raw_data_dir, night, exp, f'desi-{exp}.fits.fz')
        if not os.path.exists(datapath):
            give_details(f'could not find {datapath}! It had obstype={obstype}. Skipping',
                         f'{exp}: skipped  -- data not found')
            return None
        else:
            give_details(f'using: {datapath}')

        ## Raw data, so ensure it's read only and close right away just to be safe
        hdulist = fits.open(datapath, mode='readonly')
        # print(hdulist.info())

        if 'SPEC' in hdulist:
            hdu = hdulist['SPEC']
            if verbosely:
                print("SPEC found")
        elif 'SPS' in hdulist:
            hdu = hdulist['SPS']
            if verbosely:
                print("SPS found")
        else:
            print(f'{exp}: skipped  -- "SPEC" HDU not found!!')
            hdulist.close()
            return None

        header, specs = dict(hdu.header).copy(), hdu.data.copy()
        hdulist.close()
        # print(header)
        # print(specs)

        ## Define the column values for the current exposure in a dictionary
        outdict = {}
        for key,default in zip(colnames,coldefaults):
            if key in header.keys():
                val = header[key]
                if type(val) is str:
                    outdict[key] = val.lower()
                else:
                    outdict[key] = val
            else:
                outdict[key] = default
            #elif key in ['SEQTOT', 'SEQNUM']:
            #    ## If no sequence given, say it's 1 of 1
            #    outdict[key] = 1
            #elif key in ['COMMENTS', 'HEADERERR']:
            #    ## Include a comments and HEADERERR for human editing later. For now these are blank
            #    #outdict[key] = '| '
            #    outdict[key] = np.array([' '],dtype=str)
            #elif key in ['TILEID']:
            #    outdict[key] = -99

        ## For now assume that all 3 cameras were good for all operating spectrographs
        outdict['SPECTROGRAPHS'] = ''.join([str(spec) for spec in np.sort(specs)])
        outdict['CAMWORD'] = 'a' + outdict['SPECTROGRAPHS']

        ## Survey number befined in upper loop based on night
        outdict['SURVEY'] = surveynum

        ## As an example of future flag possibilites, flag science exposures are
        ##    garbage if less than 60 seconds
        if header['OBSTYPE'].lower() == 'science' and float(header['EXPTIME']) < 60:
            outdict['EXPFLAG'] = 2
        else:
            outdict['EXPFLAG'] = 0

        ## For Things defined in both request and data, if they don't match, flag in the
        ##     output file for followup/clarity
        for check in ['EXPTIME', 'OBSTYPE', 'FLAVOR']:
            rval, hval = req_dict[check], header[check]
            if rval != hval:
                give_details(f'{rval}\t{hval}')
                outdict['EXPFLAG'] = 1
                outdict['HEADERERR'] += f'req:{rval} but hdu:{hval} | '
            else:
                give_details(f'{check} checks out')

        #outdict['COMMENTS'] += '|'
        #outdict['HEADERERR'] += '|'

        #cnames,ctypes,cdefs = get_exposure_table_column_defs(return_default_values=True)
        #for nam,typ,deflt in zip(cnames,ctypes,cdefs):
        #    if nam not in outdict.keys():
        #        outdict[nam] = deflt
                
        if not verbosely:
            print(f'{exp}: done')
        return outdict

