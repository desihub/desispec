#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
from astropy.table import Table
from astropy.io import fits
## Import some helper functions, you can see their definitions by uncomenting the bash shell command
from desispec.workflow.helper_funcs import opj, listpath, get_json_dict
from desispec.workflow.helper_funcs import define_variable_from_environment, night_to_month
from desispec.workflow.helper_funcs import get_survey_definitions, get_surveynum
from desispec.workflow.helper_funcs import get_night_banner, give_relevant_details



def create_exposure_tables(nights, path_to_data=None, exp_table_path=None, science_types=None, \
                           verbose=False, overwrite_files=False):
    from desispec.workflow.helper_funcs import write_table
    ## Define where to find the data
    if path_to_data is None:
        path_to_data = define_variable_from_environment(env_name='DESI_SPECTRO_DATA',
                                                        var_descr="The data path")

    ## Define where to save the data
    if exp_table_path is None:
        exp_table_path = define_variable_from_environment(env_name='DESI_SPECTRO_REDUX',
                                                          var_descr='The exposure table path')
        exp_table_path = opj(exp_table_path, 'exposure_tables')
    if science_types is None:
        science_types = default_exptypes_for_exptable()

    ## Make the save directory exists
    os.makedirs(exp_table_path, exist_ok=True)

    ## Create an astropy table for each night. Define the columns and datatypes, but leave each with 0 rows
    # colnames, coldtypes = get_exposure_table_column_defs()
    # nightly_tabs = { night : Table(names=colnames,dtype=coldtypes) for night in nights }
    nightly_tabs = { night : create_exposure_table() for night in nights }

    ## Loop over nights
    survey_def = get_survey_definitions()
    for night in nights:
        print(get_night_banner(night))

        night_path = opj(path_to_data,str(night))

        ## Define the "Survey", for now this is just based on night
        survey_num = get_surveynum(night,survey_def)

        ## Loop through all exposures on disk
        for exp in listpath(path_to_data,str(night)):
            rowdict = summarize_exposure(night_path,night=night, exp=exp,scitypes=science_types,surveynum=survey_num,\
                                         colnames=nightly_tabs[night].colnames,verbosely=verbose)
            if rowdict is not None:
                ## Add the dictionary of column values as a new row
                nightly_tabs[night].add_row(rowdict)

        if len(nightly_tabs[night]) > 0:
            exptab_name = get_exptab_pathname(exptable_base_path, night)
            write_table(nightly_tabs[night], exptab_name, overwrite=overwrite_files)
        else:
            print('No rows to write to a file.')

        return nightly_tabs


def get_exposure_table_name(night=None, extension='csv'):
    # if night is None and 'PROD_NIGHT' in os.environ:
    #     night = os.environp['PROD_NIGHT']
    return f'exposure_table_{night}.{extension}'

def get_exposure_table_path(night=None):
    # if night is None and 'PROD_NIGHT' in os.environ:
    #     night = os.environp['PROD_NIGHT']

    month = night_to_month(night)
    path = opj(os.environ['DESI_SPECTRO_REDUX'],'exposure_tables',month)
    return path

def get_exposure_table_pathname(night=None, extension='csv'):#base_path,prodname
    # if night is None and 'PROD_NIGHT' in os.environ:
    #     night = os.environp['PROD_NIGHT']
    path = get_exposure_table_path(night)
    table_name = get_exposure_table_name(night, extension)
    return opj(path,table_name)


def get_exposure_table_column_defs():
    ## Define the column names for the exposure table and their respective datatypes, split in two
    ##     only for readability's sake
    colnames1 = ['EXPID', 'EXPTIME', 'OBSTYPE', 'SPECTROGRAPHS', 'CAMWORD', 'TILEID',  'NIGHT', 'EXPFLAG', 'HEADERERR', 'SURVEY']
    coltypes1 = [int    , float    , 'S8'     , 'S10'          , 'S30'    , int     ,  int    , int      , 'S10'      , int]

    colnames2 = ['SEQNUM', 'SEQTOT', 'PROGRAM','MJD-OBS', 'REQRA', 'REQDEC', 'TARGTRA', 'TARGTDEC', 'CALIBRATOR', 'COMMENTS']
    coltypes2 = [int     , int     , 'S30'    , float   , float  , float   , float    , float     , bool        , 'S10'     ]

    colnames = colnames1 + colnames2
    coldtypes = coltypes1 + coltypes2

    return colnames, coldtypes


def default_exptypes_for_exptable():
    ## Define the science types to be included in the exposure table (case insensitive)
    return ['arc','flat','twilight','science','sci','dither']

def create_exposure_table(rows=None):
    colnames, coldtypes = get_exposure_table_column_defs()
    outtab = Table(names=colnames,dtype=coldtypes)
    if rows is not None:
        for row in rows:
            outtab.add_row(row)
    return outtab


def summarize_exposure(raw_data_dir, night, exp, scitypes, surveynum, colnames, verbosely=False):
    def give_details(verbose_output, non_verbose_output=None):
        give_relevant_details(verbose_output, non_verbose_output, verbosely=verbosely)

    if verbosely:
        print(f'\n############### {exp} ###################')
    ## Request json file is first used to quickly identify science exposures
    ## If a request file doesn't exist for an exposure, it shouldn't be an exposure we care about
    reqpath = opj(raw_data_dir, night, exp, f'request-{exp}.json')
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
    if flavor != 'science':
        ## If FLAVOR is wrong
        give_details(f'ignoring: {reqpath} -- {flavor} not a flavor we care about', f'{exp}: skipped  -- not science')
        return None

    if 'OBSTYPE' not in req_dict.keys():
        ## If no obstype is defines, skip it
        give_details(f'ignoring: {reqpath} -- science flavor but obstype not defined',
                     f'{exp}: skipped  -- obstype not given')
        return None
    else:
        give_details(f'using: {reqpath}')

    ## If obstype isn't in our list of ones we care about, skip it
    obstype = req_dict['OBSTYPE'].lower()
    if obstype in scitypes:
        ## Look for the data. If it's not there, say so then move on
        datapath = opj(raw_data_dir, night, exp, f'desi-{exp}.fits.fz')
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
        for key in colnames:
            if key in header.keys():
                val = header[key]
                if type(val) is str:
                    outdict[key] = val.lower()
                else:
                    outdict[key] = val
            elif key in ['SEQTOT', 'SEQNUM']:
                ## If no sequence given, say it's 1 of 1
                outdict[key] = 1
            elif key in ['COMMENTS', 'HEADERERR']:
                ## Include a comments and HEADERERR for human editing later. For now these are blank
                outdict[key] = '| '

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

        outdict['COMMENTS'] += '|'
        outdict['HEADERERR'] += '|'
        if not verbosely:
            print(f'{exp}: done')
        return outdict




if __name__ == '__main__':
    overwrite_files = False
    verbose = True
    science_types = ['arc', 'flat', 'twilight', 'science', 'sci', 'dither']

    if 'DESI_SPECTRO_DATA' not in os.environ:
        os.environ['DESI_SPECTRO_DATA'] = opj(os.path.curdir, 'test_raw_data')
    if 'DESI_SPECTRO_REDUX' not in os.environ:
        os.environ['DESI_SPECTRO_REDUX'] = os.path.curdir

    ## Define where to find the data
    path_to_data = os.environ['DESI_SPECTRO_DATA']
    ## Define where to save the data
    exptable_base_path = opj(os.environ['DESI_SPECTRO_REDUX'], 'exposure_tables')

    ## Define the nights of interest
    nights = list(range(20200219, 20200230)) + list(range(20200301, 20200316))

    create_exposure_tables(nights, path_to_data=path_to_data,
                           exp_table_path=exptable_base_path, science_types=science_types,\
                           verbose = verbose, overwrite_files=overwrite_files)
