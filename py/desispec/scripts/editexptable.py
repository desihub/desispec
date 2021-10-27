
from desispec.workflow.exptable import get_exposure_table_name,get_exposure_table_path, \
                                       get_exposure_flags, get_last_step_options, get_exposure_table_column_defs, \
                                       keyval_change_reporting, deconstruct_keyval_reporting
from desispec.workflow.tableio import load_table, write_table
from desispec.workflow.utils import pathjoin
from desispec.io.util import parse_cameras, decode_camword, create_camword, parse_badamps, validate_badamps

import os
import numpy as np
from astropy.table import Table

def process_int_range_inclusive(input_string):
    """
    Given a str indicating a range of integers, this auto-detects the symbol used and returns that range as an INCLUSIVE
    numpy array of ints. Symbol can be ':', '-', or '..'.

    Args:
        input_string, str. String with integer range with the upper value being included in the output. E.g. 100:102
                           returns 100,101,102.

    Returns:
        np.array. Array of ints for the range specified in the input_string.
    """
    for symbol in [':','-','..']:
        if symbol in input_string:
            first,last = input_string.split(symbol)
            return np.arange(int(first),int(last)+1)

def parse_int_list_term(input_string, allints=None):
    """
    Given a str this determines what integer values it represents. Whether that be "all" indicating all ints in the
    table column, a range of integers specified with ':', '-', or '..', or a single integer. This should not be a list.

    Args:
        input_string, str. String with either integer range, single integer, or 'all'. 'all' requires allints
        allints, np.array. One dimensional array of all integers. Returns if 'all' is specified.

    Returns:
        out_array, np.array. Array of ints for the string specified.
    """
    if input_string.lower() == 'all' and allints is not None:
        out_array = np.asarray(allints)
    elif input_string.isnumeric():
        out_array = np.atleast_1d(int(input_string))
    elif np.any([symb in input_string for symb in [':','-','..']]):
        out_array = process_int_range_inclusive(input_string)
    else:
        raise ValueError(f"Couldn't understand input {input_string}")
    return out_array

def parse_int_list(input_string, allints=None, only_unique=True):
    """
    Given a str this determines what integer values it represents. Whether that be "all" indicating all ints in the
    table column, a range of integers specified with ':', '-', or '..', a single integer, or an indeterminant number of
    them in a comma separated list.

    Args:
        input_string, str. String with either integer range, single integer, or 'all'. It can have a combination of
                           multiple of these separated by a comma. 'all' requires allints.
        allints, np.array. One dimensional array of all integers. Returns if 'all' is specified.
        only_unique, bool. True if you want a unique set returned. Otherwise repeated entries in the input string
                           are kept.

    Returns:
        out_array, np.array. Array of ints for the string specified.
    """
    input_string = input_string.strip(' \t,')
    out_array = np.atleast_1d()
    for substr in input_string.split(","):
        out_array = np.append(out_array, parse_int_list_term(substr, allints=allints))
    if only_unique:
        out_array = np.unique(out_array)
    return out_array.astype(int)

def columns_not_to_report():
    """
    Returns list of column names that shouldn't have reporting information saved because they are user-defined values.
    """
    return ['COMMENTS', 'HEADERERR', 'BADCAMWORD', 'BADAMPS', 'LASTSTEP', 'EXPFLAG']

def columns_not_to_edit():
    """
    Defines column names that shouldn't be edited.
    """
    ## Occasionally unchanging things like NIGHT or TILEID have been missing in the headers, so we won't restrict
    ## that even though it typically shouldn't be edited if the data is there
    return ['EXPID', 'CAMWORD', 'OBSTYPE']

def columns_not_to_append():
    """
    Defines column names that shouldn't be edited.
    """
    return ['LASTSTEP', 'SURVEY', 'FA_SURV', 'FAPRGRM', 'GOALTYPE']

def validate_value(colname, value, joinsymb):
    """
    Checks that the value provided matches the syntax of the colname given. If the syntax is incorrect
    an error is raised.

    Warning: may change the value of "value" and returns it.

    Args:
        colname, str. The name of the column that is being edited.
        value, any scalar type. The value that the column's current value should be changed to.

    Returns:
        value, any scalar type. The value that the column's current value should be changed to. This is verified to
                                have the proper syntax for the colname given.

    """
    ## Match data type and convert where necessary
    if colname == 'EXPFLAG':
        ## Make sure the exposure flag is a valid one
        expflags = get_exposure_flags()
        value = value.lower().replace(' ','_')
        if value not in expflags:
            raise ValueError(f"Couldn't understand exposure flag: '{value}'. Available options are: {expflags}.")
    elif colname == 'BADAMPS':
        ## Make sure we can decode the badamp value (or easily correct it so we can decode it)
        ## This raises an error if it can't be converted to a viable list
        value = validate_badamps(value, joinsymb=joinsymb)
    elif colname == 'BADCAMWORD':
        ## Make sure we can understand the cameras given
        ## This raises an error if it cant be parsed
        value = parse_cameras(value)
    elif colname == 'LASTSTEP':
        options = get_last_step_options()
        value = value.lower()
        if value not in options:
            raise ValueError(f"Couldn't understand laststep: '{value}'. Available options are: {options}.")
    elif joinsymb in value:
        print(f"WARNING: For colname {colname} you provided a value '{value}' that contains the default"+
              f" joinsymbol='{joinsymb}'. This is allowed, but use at your own caution. Continuing...")
    elif '|' in value:
        print(f"WARNING: For colname {colname} you provided a value '{value}' that contains the default"+
              " indicator of an array string in the exposure tables (the 'pipe' i.e. '|'."+
              " This is allowed, but use at your own caution. Continuing...")
    else:
        ## Otherwise we don't have a strict syntax, so pass it
        pass
    return value

def document_in_comments(tablerow,colname,value,comment_col='HEADERERR'):
    """
    Places "reporting" string in the appropriate comment column of the exposure table to document the edits being
    made.

    Note: This alters and returns the input tablerow. How astropy handles this may vary. As of Jan 2021, I believe a copy
    is made in memory upon altering of a tablerow object. So the output here should be returned and assigned to
    overwrite the old value in the input table.

    Args:
        tablerow, astropy.table.Row. A table row with columns colname and comment_col. Comment_col must be a numpy array.
        colname, str. The name of the column that is being edited.
        value, any scalar type. The value that the column's current value should be changed to.
        comment_col, str. The name of the comment column where the change reporting should be placed. Default is HEADERERR.

    Returns:
        tablerow, astropy.table.Row. A table row with columns colname and comment_col. Comment_col is a numpy array
                                     with the new reporting string included.

    """
    colname = colname.upper()
    if colname in columns_not_to_report():
        return tablerow

    existing_entries = [colname in term for term in tablerow[comment_col]]
    if np.any(existing_entries):
        loc = np.where(existing_entries)[0][0]
        entry = tablerow[comment_col][loc]
        key, origval, oldval = deconstruct_keyval_reporting(entry)
        if key != colname:
            print("Key didn't match colname in document_in_comments")
            raise
        new_entry = keyval_change_reporting(colname, origval, value)
        tablerow[comment_col][loc] = new_entry
    else:
        reporting = keyval_change_reporting(colname, tablerow[colname], value)
        tablerow[comment_col] = np.append(tablerow[comment_col], reporting)
    return tablerow

def change_exposure_table_rows(exptable, exp_str, colname, value, include_comment='', append_string=False,
                               overwrite_value=False, joinsymb=','):
    """
    Changes the column named colname to value of value for rows of exposure table in exptable that correspond to the
    exposures defined in exp_str.

    Note: This edits and returns the exptable given in the inputs.

    Args:
        exptable, astropy.table.Table. An exposure table defined in desispec.workflow.exptable. Each column is an exposure.
        exp_str, str. A string representing the exposure ID's for which you want to edit the column to a new value.
                      The string can be any combination of integer ranges, single integers, or 'all'. Each range or integer
                      is separated by a comma. 'all' implies all exposures. Ranges can be given using ':', '-', or '..'.
                      All ranges are assumed to be inclusive.
        colname, str. The column name in the exptable where you want to change values.
        value, any scalar type. The value you want to change the column value of each exp_str exposure row to.
        include_comment, str. A user specified comment to be added to the COMMENTS column after setting colname to
                              value for the given exp_str exposures.
        append_string, bool. True if you want to append your input value to the end of an existing string.
        overwrite_value, bool. Default is False. Must be set to True if you want to overwrite a non-default value.
                               If current value is a default value for that column for that row,
                               this doesn't need to be set.
        joinsymb, str. The symbol used to separate string elements that are being appended. Shouldn't be '|'.
                       Default is ','.

    Returns:
        exptable, astropy.table.Table. The exposure table given in the input, with edits made to the column colname
                                       for the rows corresponding to the exposure ID's in exp_str.

    """
    ## Make sure colname exists before proceeding
    ## Don't edit fixed columns
    colname = colname.upper()
    if colname in columns_not_to_edit():
        raise ValueError(f"Not allowed to edit colname={colname}.")
    if colname not in exptable.colnames:
        raise ValueError(f"Colname {colname} not in exposure table")
    if append_string and colname in columns_not_to_append():
        raise ValueError(f"Cannot append_string to {colname}")
    if append_string and overwrite_value:
        raise ValueError("Cannot append_string and overwrite_value.")

    ## Parse the exposure numbers
    exposure_list = parse_int_list(exp_str, allints=exptable['EXPID'].data, only_unique=True)

    ## Match exposures to row numbers
    row_numbers = []
    for exp in exposure_list:
        rownum = np.where(exptable['EXPID'] == exp)[0]
        if rownum.size > 0:
            row_numbers.append(rownum[0])
    row_numbers = np.asarray(row_numbers)

    ## Make sure the value will work
    ## (returns as is if fine, corrects syntax if it can, or raises an error if it can't)
    value = validate_value(colname, value, joinsymb)

    ## If appending camwords, let's convert to camera list only once to save computation
    if colname == 'BADCAMWORD' and append_string:
        value_as_camlist = decode_camword(value)

    ## Inform user on whether reporting will be done
    if colname in columns_not_to_report():
        print("Won't do comment reporting for user defined column.")

    ## Get column names and definitions
    colnames,coldtypes,coldeflts = get_exposure_table_column_defs(return_default_values=True)
    colnames,coldtypes,coldeflts = np.array(colnames),np.array(coldtypes),np.array(coldeflts,dtype=object)
    cur_dtype = coldtypes[colnames==colname][0]
    cur_default = coldeflts[colnames==colname][0]

    if include_comment != '' and 'COMMENTS' not in colnames:
        print("Given a comment to append to the exposure tables, but COMMENTS isn't in column names. "+
              "Not including comment")

    ## Assign new value
    isstr = (cur_dtype in [str, np.str, np.str_] or type(cur_dtype) is str)
    isarr = (cur_dtype in [list, np.array, np.ndarray])
    appendable = (colname not in columns_not_to_append())

    if append_string and not isstr:
        raise ValueError(f"Told to append_string but {colname} isn't a string: {cur_dtype}")
    elif overwrite_value:
        print(f"Overwriting values in column: {colname} to '{value}' for exposures: {exposure_list}.")
    elif append_string:
        print(f"Appending '{value}' to existing entries in column: {colname} for exposures: {exposure_list}.")
    elif isarr:
        print(f"Appending {value} to arrays in column: {colname} for exposures: {exposure_list}.")
    else:
        print(f"Changing default values in column: {colname} to '{value}' for exposures: {exposure_list}.")

    orig_exptable = exptable.copy()[row_numbers]
    for rownum in row_numbers:
        if colname == 'BADCAMWORD' and exptable[colname][rownum] != cur_default and append_string:
            curcams = decode_camword(exptable[colname][rownum])
            if len(set(value_as_camlist).difference(set(curcams))) == 0:
                print(f"For exposure: {exp}. Asked to append '{value}' to '{exptable[colname][rownum]}'" +
                       " but all bad cameras are already present. Skipping and not commenting.")
                continue
            else:
                curcams.extend(value_as_camlist)
                combinedcams = list(set(curcams))
                exptable[colname][rownum] = create_camword(combinedcams)
        elif colname == 'BADAMPS' and append_string and exptable[colname][rownum] != cur_default:
            curamps = exptable[colname][rownum].split(joinsymb)
            value_as_amplist = value.split(joinsymb)
            newvals = list(set(value_as_amplist).difference(set(curamps)))
            if len(newvals) == 0:
                print(f"For exposure: {exp}. Asked to append '{value}' to '{exptable[colname][rownum]}'"+
                        " but all badamps are already present. Skipping and not commenting.")
                continue
            else:
                curamps.extend(newvals)
                exptable[colname][rownum] = joinsymb.join(curamps)
        elif isstr and append_string and exptable[colname][rownum] != cur_default:
            exptable[colname][rownum] += f'{joinsymb}{value}'
        elif isarr:
            if overwrite_value and len(exptable[colname][rownum])>0:
                exptable[rownum] = document_in_comments(exptable[rownum],colname,value)
                exptable[colname][rownum] = np.append(cur_default, value)
            else:
                exptable[colname][rownum] = np.append(exptable[colname][rownum], value)
        else:
            if overwrite_value or exptable[colname][rownum] == cur_default:
                exptable[rownum] = document_in_comments(exptable[rownum],colname,value)
                exptable[colname][rownum] = value
            else:
                exp = exptable[rownum]['EXPID']
                err = f"In exposure {exp} for column {colname}: asked to fill non-default " + \
                      f"entry '{exptable[colname][rownum]}' with '{value}'.\n" + \
                      f"\t\tTo overwrite, use --overwrite-value.\n"
                if appendable:
                    err += "\t\tTo append to the existing, use --append-string.\n"
                err += f"\t\tOriginal column entries for requested exposures were:\n"
                for exp,val in zip(list(orig_exptable['EXPID']), list(orig_exptable[colname])):
                    err += f"\t\t\t{exp}: {val}\n"
                err += "\n\t\tExiting."
                raise ValueError (err)

        if include_comment != '' and 'COMMENTS' in colnames:
            exptable['COMMENTS'][rownum] = np.append(exptable['COMMENTS'][rownum], include_comment)
            meaningful_comments = (exptable['COMMENTS'][rownum] != '')
            exptable['COMMENTS'][rownum] = exptable['COMMENTS'][rownum][meaningful_comments]

    return exptable

def edit_exposure_table(exp_str, colname, value, night=None, include_comment='', tablepath=None,
                        append_string=False, overwrite_value=False, use_spec_prod=True,
                        read_user_version=False, write_user_version=False, overwrite_file=True, joinsymb=','):
    """
    Edits the exposure table on disk to change the column named colname to value of value for rows of exposure table
    that correspond to the exposures defined in exp_str. The table on disk can be defined using night given directly
    with tablepath.

    Note: This overwrites an exposure table file on disk by default.

    Args:
        exp_str, str. A string representing the exposure ID's for which you want to edit the column to a new value.
                      The string can be any combination of integer ranges, single integers, or 'all'. Each range or integer
                      is separated by a comma. 'all' implies all exposures. Ranges can be given using ':', '-', or '..'.
                      All ranges are assumed to be inclusive.
        colname, str. The column name in the exptable where you want to change values.
        value, any scalar type. The value you want to change the column value of each exp_str exposure row to.
        night, str or int. The night the exposures were acquired on. This uniquely defines the exposure table.
        include_comment, str. A user specified comment to be added to the COMMENTS column after setting colname to
                              value for the given exp_str exposures.
        tablepath, str. A relative or absolute path to the exposure table file, if named differently from the default
                        in desispec.workflow.exptable.
        append_string, bool. True if you want to append your input value to the end of an existing string.
        overwrite_value, bool. Default is False. Must be set to True if you want to overwrite a non-default value.
                               If current value is a default value for that column for that row,
                               this doesn't need to be set.
        use_spec_prod, bool. True if you want to read in the exposure table defined by night from the currently
                             defined SPECPROD as opposed to the exposure table repository location. Default is True.
        read_user_version, bool. True if you want to read in an exposure table saved including the current user's
                                 USER name. Meant for test editing of a file multiple times. If the file doesn't exist,
                                 the non-user value is loaded. Default is False.
        write_user_version, bool. True if you want to write in an exposure table saved including the current user's
                                 USER name. Meant for test editing of a file without overwriting the true exposure table.
                                 Default is False.
        overwrite_file, bool. True if you want to overwrite the file on disk. Default is True.
        joinsymb, str. The symbol used to separate string elements that are being appended. Shouldn't be '|'.
                       Default is ','.
    """
    ## Don't edit fixed columns
    colname = colname.upper()
    if tablepath is None and night is None:
        raise ValueError("Must specify night or the path to the table.")
    if colname in columns_not_to_edit():
        raise ValueError(f"Not allowed to edit colname={colname}.")
    if append_string and colname in ['LASTSTEP', 'SURVEY', 'FA_SURV', 'FAPRGRM', 'GOALTYPE']:
        raise ValueError(f"Cannot append_string to {colname}")
    if append_string and overwrite_value:
        raise ValueError("Cannot append_string and overwrite_value.")

    ## Get the file locations
    if tablepath is not None:
        path, name = os.path.split(tablepath)
    else:
        path = get_exposure_table_path(night=night, usespecprod=use_spec_prod)
        name = get_exposure_table_name(night=night)#, extension='.csv')

    pathname = pathjoin(path, name)
    user_pathname = os.path.join(path, name.replace('.csv', '_' + str(os.environ['USER']) + '.csv'))

    ## Read in the table
    if read_user_version:
        if os.path.isfile(user_pathname):
            exptable = load_table(tablename=user_pathname, tabletype='exptable')
        else:
            print("Couldn't locate a user version of the exposure table, loading the default version of the table.")
            exptable = load_table(tablename=pathname, tabletype='exptable')
    else:
        exptable = load_table(tablename=pathname, tabletype='exptable')

    if exptable is None:
        print("There was a problem loading the exposure table... Exiting.")
        return

    ## Do the modification
    outtable = change_exposure_table_rows(exptable, exp_str, colname, value, include_comment,
                                          append_string, overwrite_value, joinsymb)

    ## Write out the table
    if write_user_version:
        write_table(outtable, tablename=user_pathname, tabletype='exptable', overwrite=overwrite_file)
        print(f"Wrote edited table to: {user_pathname}")
    else:
        write_table(outtable, tablename=pathname, tabletype='exptable', overwrite=overwrite_file)
        print(f"Wrote edited table to: {pathname}")
