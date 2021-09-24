
import os
import sys
import numpy as np
import re
import time
from astropy.table import Table
## Import some helper functions, you can see their definitions by uncomenting the bash shell command
from desispec.workflow.exptable import get_exposure_table_path, get_exposure_table_name, \
                                       night_to_month, get_exposure_table_column_defaults
from desispec.workflow.utils import define_variable_from_environment, listpath, pathjoin
from desispec.workflow.tableio import write_table, load_table
from desispec.scripts.exposuretable import create_exposure_tables



def update_exposure_tables(nights=None, night_range=None, path_to_data=None, exp_table_path=None, obstypes=None, \
                               orig_filetype='ecsv', out_filetype='ecsv', cameras='', bad_cameras='', badamps='',
                               verbose=False, no_specprod=False, dry_run=False):
    """
    Generates processing tables for the nights requested. Requires exposure tables to exist on disk.

    Args:
        nights: str, int, or comma separated list. The night(s) to generate procesing tables for.
        night_range: str, comma separated pair of nights in form YYYYMMDD,YYYYMMDD for first_night,last_night
                          specifying the beginning and end of a range of nights to be generated.
                          last_night should be inclusive.
        path_to_data: str. The path to the raw data and request*.json and manifest* files.
        exp_table_path: str. Full path to where to exposure tables should be saved, WITHOUT the monthly directory included.
        obstypes: str or comma separated list of strings. The exposure OBSTYPE's that you want to include in the exposure table.
        orig_filetype: str. The file extension (without the '.') of the exposure tables.
        out_filetype: str. The file extension for the outputted exposure tables (without the '.').
        cameras: str. Explicitly define the cameras for which you want to reduce the data. Should be a comma separated
              list. Only numbers assumes you want to reduce r, b, and z for that camera. Otherwise specify
              separately [brz][0-9].
        bad_cameras: str. Explicitly define the cameras that you don't want to reduce the data. Should be a comma
                          separated list. Only numbers assumes you want to reduce r, b, and z for that camera.
                          Otherwise specify separately [brz][0-9].
        badamps: str. Define amplifiers that you know to be bad and should not be processed. Should be a list separated
                      by comma or semicolon. Saved list will converted to semicolons. Each entry should be of the
                      form {camera}{spectrograph}{amp}, i.e. [brz][0-9][A-D].
        verbose: boolean. Whether to give verbose output information or not. True prints more information.
        no_specprod: boolean. Create exposure table in repository location rather than the SPECPROD location


    Returns: Nothing
    """
    if nights is None and night_range is None:
        raise ValueError("Must specify either nights or night_range")
    elif nights is not None and night_range is not None:
        raise ValueError("Must only specify either nights or night_range, not both")

    if nights is None or nights=='all':
        nights = list()
        for n in listpath(os.getenv('DESI_SPECTRO_DATA')):
            #- nights are 20YYMMDD
            if re.match('^20\d{6}$', n):
                nights.append(n)
    else:
        nights = [ int(val.strip()) for val in nights.split(",") ]

    nights = np.array(nights)

    if night_range is not None:
        if ',' not in night_range:
            raise ValueError("night_range must be a comma separated pair of nights in form YYYYMMDD,YYYYMMDD")
        nightpair = night_range.split(',')
        if len(nightpair) != 2 or not nightpair[0].isnumeric() or not nightpair[1].isnumeric():
            raise ValueError("night_range must be a comma separated pair of nights in form YYYYMMDD,YYYYMMDD")
        first_night, last_night = nightpair
        nights = nights[np.where(int(first_night)<=nights.astype(int))[0]]
        nights = nights[np.where(int(last_night)>=nights.astype(int))[0]]

    print("Nights: ", nights)

    ## Define where to find the data
    if path_to_data is None:
        path_to_data = define_variable_from_environment(env_name='DESI_SPECTRO_DATA', var_descr="The data path")

    ## Define where to save the data
    usespecprod = (not no_specprod)
    if exp_table_path is None:
        exp_table_path = get_exposure_table_path(night=None,usespecprod=usespecprod)

    nights_with_data = listpath(path_to_data)
    for night in nights:
        if str(night) not in nights_with_data:
            print(f'Night: {night} not in data directory {path_to_data}. Skipping')
            continue

        month = night_to_month(night)
        exptab_path = pathjoin(exp_table_path,month)
        orig_name = get_exposure_table_name(night, extension=orig_filetype)
        orig_pathname = pathjoin(exptab_path, orig_name)
        temp_filetype = f"temp.{out_filetype}"
        temp_pathname = orig_pathname.replace(f".{orig_filetype}", f".{temp_filetype}")
        if not os.path.exists(orig_pathname):
            print(f'Night: {night} original table could not be found {orig_pathname}. Skipping this night.')
        else:
            create_exposure_tables(nights=str(night), night_range=None, path_to_data=path_to_data,
                                   exp_table_path=exp_table_path, obstypes=obstypes, exp_filetype=temp_filetype,
                                   cameras=cameras, bad_cameras=bad_cameras, badamps=badamps,
                                   verbose=verbose, no_specprod=no_specprod, overwrite_files=False)

            newtable = load_table(temp_pathname, tabletype='exptab', use_specprod=usespecprod)
            origtable = load_table(orig_pathname, tabletype='exptab', use_specprod=usespecprod)
            print(f"\n\nNumber of rows in original: {len(origtable)}, Number of rows in new: {len(newtable)}")

            if 'OBSTYPE' in origtable.colnames and set(obstypes) != set(origtable['OBSTYPE']):
                subset_rows = [obs in obstypes for obs in origtable['OBSTYPE']]
                subset_orig = origtable[subset_rows]
            else:
                subset_orig = origtable

            assert len(newtable) >= len(subset_orig), "Tables for given obstypes must be the same length"
            assert np.all([exp in newtable['EXPID'] for exp in subset_orig['EXPID']]), \
                          "All old exposures of given obstype must be present in the new table"

            mutual_colnames = [col for col in newtable.colnames if col in origtable.colnames]
            coldefs = get_exposure_table_column_defaults(asdict=True)
            for newloc,expid in enumerate(newtable['EXPID']):
                origloc = np.where(origtable['EXPID']==expid)[0]
                if len(origloc) > 1:
                    print(f"ERROR on night {night}: found more than one exposure matching expid {expid}")
                    continue
                elif len(origloc) == 1:
                    origloc = origloc[0]
                else:
                    continue
                for col in mutual_colnames:
                    origval = origtable[col][origloc]
                    newval = newtable[col][newloc]
                    if col == 'EXPFLAG'	and 'EFFTIME_ETC' in newtable.colnames and \
                            newtable['EFFTIME_ETC'][newloc] > 0. and 'aborted' in origval:
                        origorigval = origval.copy()
                        origval = origval[np.where(origval != 'aborted')]
                        print(f"Identified outdated aborted exposure flag. Removing that. Original set: " + \
                              f"{origorigval}, Updated origset: {origval}")
                    elif col == 'COMMENTS' and 'EFFTIME_ETC' in newtable.colnames and newtable['EFFTIME_ETC'][newloc] > 0. and \
                            'EXPFLAG' in origtable.colnames and 'aborted' in origtable['EXPFLAG'][origloc]:
                        origorigval = origval.copy()
                        valcheck = np.array([('For EXPTIME:' not in val) for val in origval])
                        origval = origval[valcheck]
                        print(f"Identified outdated aborted exptime COMMENT. Removing that. Original set: " + \
                              f"{origorigval}, Updated origset: {origval}")
                    if np.isscalar(origtable[col][origloc]):
                        if origval != coldefs[col] and newval != origval:
                            print(f"Difference detected for Night {night}, exp {expid}, " + \
                                  f"col {col}: orig={origval}, new={newval}. Taking the original value. ")
                            newtable[col][newloc] = origval
                    else:
                        if not np.array_equal(origval, coldefs[col]) and \
                           not np.array_equal(newval, origval):
                            print(f"Difference detected for Night {night}, exp {expid}, " + \
                                  f"col {col}: orig={origval}, new={newval}. Appending the two arrays.")
                            combined_val = newval[newval != '']
                            for val in origval:
                                if val != '' and val not in newval:
                                    combined_val = np.append(combined_val,[val])
                            newtable[col][newloc] = combined_val

            if dry_run:
                print("\n\nOutput file would have been:")
                newtable.pprint_all()

                names = [col for col in newtable.colnames if col not in ['HEADERERR','EXPFLAG','COMMENTS']]
                t1 = newtable[names]
                t2 = load_table(temp_pathname,tabletype='exptab',use_specprod=usespecprod)[names]
                t1.values_equal(t2).pprint_all()
            else:
                ftime = time.strftime("%Y%m%d_%Hh%Mm")
                replaced_pathname = orig_pathname.replace(f".{orig_filetype}", f".replaced-{ftime}.{orig_filetype}")
                print(f"Moving original file from {orig_pathname} to {replaced_pathname}")
                os.rename(orig_pathname,replaced_pathname)
                time.sleep(0.1)
                out_pathname = orig_pathname.replace(f".{orig_filetype}", f".{out_filetype}")
                write_table(newtable, out_pathname)
                print(f"Updated file save to {out_pathname}. Original archived as {replaced_pathname}")

            os.remove(temp_pathname)
            print(f"Removed the temporary file {temp_pathname}")
            print("\n\n")

        print("Exposure table regenerations complete")

        ## Flush the outputs
        sys.stdout.flush()
        sys.stderr.flush()
