
import os
import sys
import numpy as np
import re
import time

from desispec.workflow.exptable import get_exposure_table_path, \
                                       get_exposure_table_name, \
                                       default_obstypes_for_exptable,\
                                       night_to_month, \
                                       get_exposure_table_column_defaults
from desispec.workflow.utils import define_variable_from_environment, listpath, \
                                    pathjoin
from desispec.workflow.tableio import write_table, load_table
from desispec.scripts.exposuretable import create_exposure_tables



def update_exposure_tables(nights=None, night_range=None, path_to_data=None,
                           exp_table_path=None, obstypes=None, orig_filetype='csv',
                           out_filetype='csv',  verbose=False, no_specprod=False,
                           dry_run=False):
    """
    Generates updated exposure tables for the nights requested. Requires
    exposure tables to exist on disk.

    Args:
        nights: str, int, or comma separated list. The night(s) to generate
                                                   procesing tables for.
        night_range: str. comma separated pair of nights in form
                          YYYYMMDD,YYYYMMDD for first_night,last_night
                          specifying the beginning and end of a range of
                          nights to be generated. last_night should be
                          inclusive.
        path_to_data: str. The path to the raw data and request*.json and
                           manifest* files.
        exp_table_path: str. Full path to where to exposure tables should be
                             saved, WITHOUT the monthly directory included.
        obstypes: str. The exposure OBSTYPE's that you want to include in the
                       exposure table. Can be a comma separated list.
        orig_filetype: str. The file extension (without the '.') of the exposure
                            tables.
        out_filetype: str. The file extension for the outputted exposure tables
                           (without the '.').
        verbose: boolean. Whether to give verbose output information or not.
                          True prints more information.
        no_specprod: boolean. Create exposure table in repository location
                              rather than the SPECPROD location

    Returns:
        Nothing
    """
    ## Make sure user specified what nights to run on
    if nights is None and night_range is None:
        raise ValueError("Must specify either nights or night_range."
                         +" To process all nights give nights=all")

    ## Define where to find the data
    if path_to_data is None:
        path_to_data = define_variable_from_environment(env_name='DESI_SPECTRO_DATA',
                                                        var_descr="The data path")

    ## Get all nights in 2020's with data
    nights_with_data = list()
    for n in listpath(path_to_data):
        # - nights are 20YYMMDD
        if re.match('^202\d{5}$', n):
            nights_with_data.append(n)

    ## If unpecified or given "all", set nights to all nights with data
    check_night = False
    if nights is None or nights == 'all':
        nights = [int(night) for night in nights_with_data]
        ## No need to check nights since derived from disk
    else:
        nights = [int(val.strip()) for val in nights.split(",")]
        ## If nights are specified, make sure we check that there is actually data
        check_night = True
    nights = np.array(nights)

    ## If user specified a night range, cut nights to that range of dates
    if night_range is not None:
        if ',' not in night_range:
            raise ValueError("night_range must be a comma separated pair of "
                             + "nights in form YYYYMMDD,YYYYMMDD")
        nightpair = night_range.split(',')
        if len(nightpair) != 2 or not nightpair[0].isnumeric() \
                or not nightpair[1].isnumeric():
            raise ValueError("night_range must be a comma separated pair of "
                             + "nights in form YYYYMMDD,YYYYMMDD")
        first_night, last_night = nightpair
        nights = nights[np.where(int(first_night)<=nights.astype(int))[0]]
        nights = nights[np.where(int(last_night)>=nights.astype(int))[0]]

    ## Parse the obstypes of the input
    if obstypes is not None:
        obstypes = [ val.strip('\t ') for val in obstypes.split(",") ]
    else:
        obstypes = default_obstypes_for_exptable()

    ## Define where to save the data
    usespecprod = (not no_specprod)
    if exp_table_path is None:
        exp_table_path = get_exposure_table_path(night=None,
                                                 usespecprod=usespecprod)

    ## Tell user the final list of nights and starting looping over them
    print("Nights: ", nights)
    for night in nights:
        if check_night and str(night) not in nights_with_data:
            print(f'Night {night} not in data directory: {path_to_data}. Skipping')
            continue

        ## Define where we should be looking for the exposure tables
        month = night_to_month(night)
        exptab_path = pathjoin(exp_table_path,month)
        orig_name = get_exposure_table_name(night, extension=orig_filetype)
        orig_pathname = pathjoin(exptab_path, orig_name)

        ## If the exposure table doesn't exist, skip, since we are updating
        ## not generating.
        if not os.path.exists(orig_pathname):
            print(f'Could not find exposure table for night={night} at:'
                  + f' {orig_pathname}. Skipping this night.')
            continue

        ## Create a temporary file pathname
        temp_filetype = f"updatetemp.{out_filetype}"
        temp_pathname = orig_pathname.replace(f".{orig_filetype}",
                                              f".{temp_filetype}")

        ## Create a fresh version of the exposure table using the current
        ## code and save it to the temporary pathname
        obstypes_str = ','.join(obstypes)
        create_exposure_tables(nights=str(night), night_range=None,
                               path_to_data=path_to_data,
                               exp_table_path=exp_table_path,
                               obstypes=obstypes_str, exp_filetype=temp_filetype,
                               cameras=None, bad_cameras=None,
                               badamps=None, verbose=verbose,
                               no_specprod=no_specprod, overwrite_files=False)

        ## Load the old and new tables to compare
        newtable = load_table(temp_pathname, tabletype='exptab',
                              use_specprod=usespecprod)
        origtable = load_table(orig_pathname, tabletype='exptab',
                               use_specprod=usespecprod)

        ## Print some useful information and do some sanity checks that
        ## The new table has as much or more data than the old
        print(f"\n\nNumber of rows in original: {len(origtable)}"
              + f", Number of rows in new: {len(newtable)}")

        if 'OBSTYPE' in origtable.colnames \
                and not set(origtable['OBSTYPE']).issubset(set(obstypes)):
            subset_rows = np.array([obs in obstypes for obs in origtable['OBSTYPE']])
            subset_orig = origtable[subset_rows]
        else:
            subset_orig = origtable

        assert len(newtable) >= len(subset_orig), \
               "Tables for given obstypes must greater or equal length"
        assert np.all([exp in newtable['EXPID'] for exp in subset_orig['EXPID']]), \
               "All old exposures of given obstype must be present in the new table"

        ## Go through exposure by exposure and check each columns value
        ## in the new vs the original
        mutual_colnames = [col for col in newtable.colnames if col in origtable.colnames]
        coldefs = get_exposure_table_column_defaults(asdict=True)
        for newloc,expid in enumerate(newtable['EXPID']):
            ## Match to the row in the original table
            origloc = np.where(origtable['EXPID']==expid)[0]
            if len(origloc) > 1:
                print(f"ERROR on night {night}: found more than one exposure"
                      + f"matching expid {expid}")
                continue
            elif len(origloc) == 1:
                origloc = origloc[0]
            else:
                print(f"New exposure identified: {newtable[newloc]}")
                continue
            ## For colnames that the two columns share, compare values.
            for col in mutual_colnames:
                origval = origtable[col][origloc]
                newval = newtable[col][newloc]
                ## Clean up three special cases of bad flags/comments in early data
                if col == 'EXPFLAG'	and 'EFFTIME_ETC' in newtable.colnames and \
                        newtable['EFFTIME_ETC'][newloc] > 0. and 'aborted' in origval:
                    origorigval = origval.copy()
                    origval = origval[np.where(origval != 'aborted')]
                    print("Identified outdated aborted exposure flag. "
                          + "Removing that. Original set: "
                          + f"{origorigval}, Updated origset: {origval}")
                if col == 'COMMENTS' and 'EFFTIME_ETC' in newtable.colnames \
                        and newtable['EFFTIME_ETC'][newloc] > 0. and \
                        'EXPFLAG' in origtable.colnames \
                        and 'aborted' in origtable['EXPFLAG'][origloc]:
                    origorigval = origval.copy()
                    valcheck = np.array([('For EXPTIME:' not in val) for val in origval])
                    origval = origval[valcheck]
                    print(f"Identified outdated aborted exptime COMMENT."
                          + "Removing that. Original set: "
                          + f"{origorigval}, Updated origset: {origval}")
                if col == 'HEADERERR' and 'PURPOSE:->' in origval:
                    origorigval = origval.copy()
                    valcheck = (np.array(origval) != 'PURPOSE:->')
                    origval = origval[valcheck]
                    print(f"Identified outdated PURPOSE null->null HEADERERR."
                          + " Removing that. Original set: "
                          + f"{origorigval}, Updated origset: {origval}")
                ## If columns differ and original isn't a default value,
                ## then take the original user-defined value
                if np.isscalar(origtable[col][origloc]):
                    if origval != coldefs[col] and newval != origval:
                        print(f"Difference detected for Night {night}, exp {expid}, "
                              + f"col {col}: orig={origval}, new={newval}. "
                              + "Taking the original value. ")
                        newtable[col][newloc] = origval
                else:
                    if not np.array_equal(origval, coldefs[col]) and \
                       not np.array_equal(newval, origval):
                        print(f"Difference detected for Night {night}, exp {expid}, "
                              + f"col {col}: orig={origval}, new={newval}. "
                              + "Taking union of the two arrays.")
                        combined_val = newval[newval != '']
                        for val in origval:
                            if val != '' and val not in newval:
                                combined_val = np.append(combined_val,[val])
                        newtable[col][newloc] = combined_val

        ## If just testing, print the table and a cell-by-cell equality test
        ## for the scalar columns
        ## If not testing, move the original table to an archived filename
        ## and save the updated table to the official exptable pathname
        if dry_run:
            print("\n\nOutput file would have been:")
            newtable.pprint_all()

            names = [col for col in newtable.colnames if col not in ['HEADERERR','EXPFLAG','COMMENTS']]
            t1 = newtable[names]
            t2 = load_table(temp_pathname, tabletype='exptab',
                                           use_specprod=usespecprod)[names]
            t1.values_equal(t2).pprint_all()
        else:
            ftime = time.strftime("%Y%m%d_%Hh%Mm")
            replaced_pathname = orig_pathname.replace(f".{orig_filetype}",
                                                      f".replaced-{ftime}.{orig_filetype}")
            print(f"Moving original file from {orig_pathname} to {replaced_pathname}")
            os.rename(orig_pathname,replaced_pathname)
            time.sleep(0.1)
            out_pathname = orig_pathname.replace(f".{orig_filetype}", f".{out_filetype}")
            write_table(newtable, out_pathname)
            print(f"Updated file saved to {out_pathname}. Original archived as {replaced_pathname}")

        ## Cleanup the temporary table created with the fresh version of the
        ## create_exposure_table script
        os.remove(temp_pathname)
        print(f"Removed the temporary file {temp_pathname}")
        print("\n\n")

        ## Flush the outputs
        sys.stdout.flush()
        sys.stderr.flush()
    print("Exposure table regenerations complete")
