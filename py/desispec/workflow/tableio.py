import os
import numpy as np
from astropy.table import Table


###################################################
################  Table Functions #################
###################################################
from desispec.workflow.utils import pathjoin
from desiutil.log import get_logger

def ensure_scalar(val, joinsymb='|',comma_replacement=';'):
    """
    Ensures that the object in val is a scalar that can be save to a Table cell (i.e. row of a column or
    column of a row). If the it is an array or list, it uses joinsymb to turn them into a single string.

    Args:
        val, a scalar datatype, list, or array. The value to be converted to a scalar quantity (returning the val if
                                                it is already a scalar).
        joinsymb, str. A string symbol *other than comma* that will be used to join the multiple values of a list
                       or array.
        comma_replacement, str. A string symbol that should be used to replace any existing commas in the data, such
                                that the value can be saved in a csv format.

    Returns:
        val or outstr, any scalar type or string. The output string which is a scalar quantity capable of being
                                                  written to a single table cell (in a csv or fits file, for example).
    """
    if type(val) in [str, np.str, np.str_]:
        if ',' in val:
            val = val.replace(',', comma_replacement)
        return val
    elif val is None or type(val) is np.ma.core.MaskedConstant or np.isscalar(val):
        return val
    else:
        val = np.atleast_1d(val).astype(str)
        outstr = joinsymb.join(val) + joinsymb
        if ',' in outstr:
            outstr = outstr.replace(',', comma_replacement)
        return outstr

def split_str(val, joinsymb='|',comma_replacement=';'):
    """
    Attempts to intelligently interpret an input scalar. If it is a string it looks to see if it was a list or array
    objects that was joined to be a single string using joinsymb. If it identifies that, it will split that into the
    original list/array. Otherwise it will return the val as-is.

    Args:
        val, any datatype. The input to be checked to see if it is in fact a list/array that was joined into a string
                           for saving in a Table.
        joinsymb, str. The symbol used to join values in a list/array when saving. Should not be a comma.
        comma_replacement, str. Replace instances of this symbol with commas when loading ONLY scalar columns in a table,
                                as e.g. BADAMPS is used in the pipeline and symbols like ';' are problematic
                                on the command line. Comment arrays do not need to be converted back and forth.

    Returns:
        val or split_list, any datatype or np.array.
    """
    if type(val) in [str, np.str, np.str_]:
        if val.isnumeric():
            if '.' in val:
                return float(val)
            else:
                return int(val)
        elif joinsymb not in val:
            if val.lower() == 'true':
                return True
            elif val.lower() == 'false':
                return False
            else:
                if comma_replacement in val:
                    val = val.replace(comma_replacement, ',')
                return val
        else:
            val = val.strip(joinsymb)
            split_list = np.array(val.split(joinsymb))
            if '.' in split_list[0] and split_list[0].isnumeric():
                return split_list.astype(float)
            elif split_list[0].isnumeric():
                return split_list.astype(int)
            else:
                split_list = np.array([val.strip('\t ') for val in split_list.astype(str)]).astype(str)
                return split_list
    else:
        return val


def write_table(origtable, tablename=None, tabletype=None, joinsymb='|', overwrite=True, verbose=False,
                write_empty=False, use_specprod=True):
    """
    Workflow function to write exposure, processing, and unprocessed tables. It allows for multi-valued table cells, which are
    reduced to strings using the joinsymb. It writes to a temp file before moving the fully written file to the
    name given by tablename (or the default for table of type tabletype).

    Args:
        origtable, Table. Either exposure table or processing table.
        tablename, str. Full pathname of where the table should be saved, including the extension. Originally save to
                        *.temp.{ext} and then moved to *.{ext}. If None, it looks up the default for typetable.
        tabletype, str. Used if tablename is None to get the default name for the type of table.
        joinsymb, str. The symbol used to join values in a list/array when saving. Should not be a comma.
        overwrite, bool. Whether to overwrite the file on disk if it already exists. Default is currently True.
        verbose, bool. Whether to give verbose amounts of information (True) or succinct/no outputs (False). Default is False.
        write_empty, bool. Whether to write an empty table to disk. The default is False. Warning: code is less robust
                           to column datatypes on read/write if the table is empty. May cause issues if this is set to True.
        use_specprod, bool. If True and tablename not specified and tabletype is exposure table, this looks for the
                            table in the SPECPROD rather than the exptab repository. Default is True.
    Returns:
        Nothing.
    """
    log = get_logger()
    if tablename is None and tabletype is None:
        log.error("Pathname or type of table is required to save the table")
        return

    if tabletype is not None:
        tabletype = standardize_tabletype(tabletype)

    if tablename is None:
        tablename = translate_type_to_pathname(tabletype, use_specprod=use_specprod)

    if not write_empty and len(origtable) == 0:
        log.warning(f'NOT writing zero length table to {tablename}')
        return
        
    if verbose:
        log.info("In write table", tablename,'\n', tabletype)
        log.info(origtable[0:2])
    basename, ext = os.path.splitext(tablename)

    temp_name = f'{basename}.temp{ext}'
    if verbose:
        log.info(ext ,temp_name)
    table = origtable.copy()

    if ext in ['.csv', '.ecsv']:
        if verbose:
            log.info("Given table: ", table.info)
        # replace_cols = {}

        for nam in table.colnames:
            ndim = table[nam].ndim
            if ndim > 1 or type(table[nam][0]) in [list, np.ndarray, np.array] or table[nam].dtype is object:
                if verbose:
                    log.info(f'{nam} is {ndim} dimensions, changing to string')
                col = [ensure_scalar(row, joinsymb=joinsymb) for row in table[nam]]
                # replace_cols[nam] = Table.Column(name=nam,data=col)
                if type(table[nam]) is Table.MaskedColumn:
                    col = Table.MaskedColumn(name=nam, data=col)
                else:
                    col = Table.Column(name=nam, data=col)
                table.replace_column(nam, col)

        # for nam, col in replace_cols.items():
        #     t.replace_column(nam,col)

        if np.any([c.ndim > 1 or type(table[nam][0]) in [list, np.ndarray, np.array] for c in
                   table.itercols()]) and verbose:
            log.warning("A column was still more than one dimensional")
            log.info(table.info())

        table.write(temp_name, format=f'ascii{ext}', overwrite=overwrite)
    else:
        table.write(temp_name, overwrite=True)

    os.rename(temp_name, tablename)
    if verbose:
        log.info("Written table: ", table.info)

def standardize_tabletype(tabletype):
    """
    Given the user defined type of table it returns the proper 'tabletype' expected by the pipeline

    Args:
        tabletype, str. Allows for a flexible number of input options, but should refer to either the 'exposure',
                         'processing', or 'unprocessed' table types.

    Returns:
         tabletype, str. Standardized tabletype values. Either "exptable", "proctable", "unproctable".
    """
    if tabletype.lower() in ['exp', 'exposure', 'etable', 'exptable', 'exptab', 'exposuretable', 'exposure_table']:
        tabletype = 'exptable'
    elif tabletype.lower() in ['proc', 'processing', 'proctable', 'proctab', 'int', 'ptable', 'internal']:
        tabletype = 'proctable'
    elif tabletype.lower() in ['unproc', 'unproctable', 'unproctab', 'unprocessed', 'unprocessing', 'unproc_table']:
        tabletype = 'unproctable'
    return tabletype

def translate_type_to_pathname(tabletype, use_specprod=True):
    """
    Given the type of table it returns the proper file pathname

    Args:
        tabletype, str. Allows for a flexible number of input options, but should refer to either the 'exposure',
                         'processing', or 'unprocessed' table types.
        use_specprod, bool. If True and tablename not specified and tabletype is exposure table, this looks for the
                            table in the SPECPROD rather than the exptab repository. Default is True.

    Returns:
         tablename, str. Full pathname including extension of the table type. Uses environment variables to determine
                         the location.
    """
    from desispec.workflow.exptable import get_exposure_table_path, get_exposure_table_pathname, get_exposure_table_name
    from desispec.workflow.proctable import get_processing_table_path, get_processing_table_pathname, get_processing_table_name
    tabletype = standardize_tabletype(tabletype)
    if tabletype == 'exptable':
        tablename = get_exposure_table_pathname(night=None,usespecprod=use_specprod)
    elif tabletype == 'proctable':
        tablename = get_processing_table_pathname()
    elif tabletype == 'unproctable':
        tablepath = get_processing_table_path()
        tablename = get_processing_table_name().replace("processing", 'unprocessed')
        tablename = pathjoin(tablepath, tablename)
    return tablename

def load_table(tablename=None, tabletype=None, joinsymb='|', verbose=False, process_mixins=True, use_specprod=True):
    """
    Workflow function to read in exposure, processing, and unprocessed tables. It allows for multi-valued table cells, which are
    generated from strings using the joinsymb. It reads from the file given by tablename (or the default for table of
    type tabletype).

    Args:
        tablename, str. Full pathname of where the table should be saved, including the extension. Originally save to
                        *.temp.{ext} and then moved to *.{ext}. If None, it looks up the default for typetable. If
                        tabletype is None it uses this to try and identify the tabletype and uses that to get the
                        default column names and types.
        tabletype, str. Used if tablename is None to get the default name for the type of table. Also used to get the
                        column datatypes and defaults.
        joinsymb, str. The symbol used to join values in a list/array when saving. Should not be a comma.
        verbose, bool. Whether to give verbose amounts of information (True) or succinct/no outputs (False). Default is False.
        process_mixins, bool. Whether to look for and try to split strings into lists/arrays. The default is True.
                              Warning: The exposure and processing tables have default data types which are multi-value.
                              If this is set to False, the default data types will be incorrect and issues are likely
                              to arise.
        use_specprod, bool. If True and tablename not specified and tabletype is exposure table, this looks for the
                            table in the SPECPROD rather than the exptab repository. Default is True.

    Returns:
        table, Table. Either exposure table or processing table that was loaded from tablename (or from default name
                      based on tabletype). Returns None if the file doesn't exist.
    """
    from desispec.workflow.exptable import instantiate_exposure_table, get_exposure_table_column_defs
    from desispec.workflow.proctable import instantiate_processing_table, get_processing_table_column_defs
    log = get_logger()

    if tabletype is not None:
        tabletype = standardize_tabletype(tabletype)

    if tablename is None:
        if tabletype is None:
            log.error("Must specify either tablename or tabletype in load_table()")
            return None
        else:
            tablename = translate_type_to_pathname(tabletype, use_specprod=use_specprod)
    else:
        if tabletype is None:
            log.info("tabletype not given in load_table(), trying to guess based on filename")
            filename = os.path.split(tablename)[-1]
            if 'exp' in filename or 'etable' in filename:
                tabletype = 'exptable'
            elif 'unproc' in filename:
                tabletype = 'unproctable'
            elif 'proc' in filename or 'ptable' in filename:
                tabletype = 'proctable'

            if tabletype is None:
                log.warning(f"Couldn't identify type based on filename {filename}")
            else:
                log.info(f"Based on filename {filename}, identified type as {tabletype}")

    if os.path.isfile(tablename):
        log.info(f"Found table: {tablename}")
    elif tabletype is not None:
        log.info(f'Table {tablename} not found, creating new table of type {tabletype}')
        if tabletype == 'exptable':
            return instantiate_exposure_table()
        elif tabletype == 'unproctable':
            return instantiate_exposure_table()
        elif tabletype == 'proctable':
            return instantiate_processing_table()
        else:
            log.warning(f"Couldn't create type {tabletype}, unknown table type")
            return None
    else:
        log.error(f"In load_table:\n\tCouldn't find: {tabletype} and tabletype not specified, returning None")
        return None

    basename, ext = os.path.splitext(tablename)
    if ext in ['.csv', '.ecsv']:
        table = Table.read(tablename, format=f'ascii{ext}')

        if verbose:
            log.info("Raw loaded table: ", table.info)

        if tabletype in ['exptable', 'unproctable']:
            colnames, coltypes, coldefaults = get_exposure_table_column_defs(return_default_values=True)
        elif tabletype == 'proctable':
            colnames, coltypes, coldefaults = get_processing_table_column_defs(return_default_values=True)
        else:
            colnames = table.colnames
            coltypes = [table[nam].dtype for nam in colnames]
            coldefaults = [guess_default_by_dtype(typ) for typ in coltypes]
        colnames, coltypes = np.array(colnames), np.array(coltypes)

        if len(table ) >0:
            outcolumns = []
            for nam, typ, default in zip(colnames ,coltypes, coldefaults):
                if type(table[nam]) is Table.MaskedColumn:
                    data, mask = table[nam].data, table[nam].mask
                else:
                    data, mask = table[nam].data, None
                col, dtyp = process_column(data, typ, mask=mask, default=default, joinsymb=joinsymb, \
                                           process_mixins=process_mixins, verbose=verbose)

                newcol = Table.Column(name=nam ,data=col ,dtype=dtyp)
                if dtyp in [list, np.array, np.ndarray]:
                    newcol.shape = (len(col),)
                    for ii in range(len(col)):
                        try:
                            newcol[ii] = np.atleast_1d(newcol[ii])
                        except:
                            import pdb
                            pdb.set_trace()
                outcolumns.append(newcol)
            table = Table(outcolumns)
        else:
            table = Table(names=colnames, dtype=coltypes)
    else:
        table = Table.read(tablename)

    if verbose:
        log.info("Expanded table: ", table.info)
    return table

def guess_default_by_dtype(typ):
    """
    Returns a default value given a data type. To be used in filling a table if no default is given.

    Args:
        typ, DataType. The datatype of the element you want a default value for.

    Returns:
        default value for that type. Can be int, float, str, list, or array. If it can't guess, it returns the
        integer -99 .
    """
    if typ in [int ,np.int8, np.int16, np.int32, np.int32]:
        return -99
    elif typ in [float, np.float32, np.float64]:
        return -99.0
    elif typ in [str, np.str, np.str_]:
        return 'unknown'
    elif typ == list:
        return []
    elif typ in [np.array, np.ndarray]:
        return np.array([])
    else:
        return -99

def process_column(data, typ, mask=None, default=None, joinsymb='|', process_mixins=True ,verbose=False):
    """
    Used with load_table to process a Table.Column after being read in. It fills in masked values with defaults,
    and identifies and splits mixin columns (columns that should be a list/array) back into their list/array from
    their string representation.

    Args:
        data, Table.Column or Table.MaskedColumn. Column of data to be checked for masked rows (to be filled with
                                                  default) and string-ed versions of lists/arrays that need to be
                                                  expanded out.
        typ, DataType. The expected datatype of the data in data. May differ from the type of the input data, in which
                       case the data will be transformed.
        mask, np.array. A mask array with True in row elements of the input data array that are masked and False in
                        row elements that are not masked.
        default, any type. The default value to be used for masked rows.
        joinsymb, str. The symbol used to join values in a list/array when saving. Should not be a comma.
        process_mixins, bool. Whether to look for and try to split strings into lists/arrays. The default is True.
                              Warning: The exposure and processing tables have default data types which are multi-value.
                              If this is set to False, the default data types will be incorrect and issues are likely
                              to arise.
        verbose, bool. Whether to give verbose amounts of information (True) or succinct/no outputs (False). Default is False.


    Returns:
        col, list or np.array. A new data vector similar to input 'data' except with masked values filled in and
                               mixin strings expanded back into np.array's.
        dtyp, DataType. The data type of a row element in the return col.
    """
    log = get_logger()

    if default is None:
        default = guess_default_by_dtype(typ)

    if mask is not None and np.sum(np.bitwise_not(mask)) == 0:
        return [default ] *len(data), typ

    if mask is None:
        mask = np.zeros(len(data)).astype(bool)

    array_like = (typ in [list, np.array, np.ndarray])
    dtyp = typ

    if mask is not None:
        first = data[np.bitwise_not(mask)][0]
    else:
        first = data[0]
    firsttype = type(first)

    if verbose:
        log.debug(first, firsttype, firsttype in [str, np.str, np.str_])
    if process_mixins and firsttype in [str, np.str, np.str_] and joinsymb in first:
        do_split_str = True
        if typ not in [list, np.array, np.ndarray]:
            log.warning("Found mixin column with scalar datatype:")
            log.info("\tcolname={nam}, first={first}, typefirst={firsttyp}, dtype={typ}")
            log.info("\tchanging to np.array datatype")
            dtyp = np.array
    else:
        do_split_str = False

    col = []
    for rowdat ,rowmsk in zip(data ,mask):
        if rowmsk:
            col.append(default)
        elif do_split_str:
            col.append(split_str(rowdat, joinsymb=joinsymb))
        elif array_like:
            col.append(np.array([rowdat]))
        else:
            col.append(rowdat)

    if verbose:
        log.info(col)

    return col, dtyp

def write_tables(tables, tablenames=None, tabletypes=None, write_empty=False, verbose=False, overwrite=True):
    """
    Workflow function to write multiple exposure, processing, and unprocessed tables. It allows for multi-valued
    table cells, which are reduced to strings. It writes to a temp file before moving the fully
    written file to the name given by tablenames (or the default for table of types tabletypes).

    Args:
        tables, list/array of Table's. List or array of exposure tables, unprocessed tables, and/or processing table.
        tablenames, list/array of str's. List or array of the full pathnames to where the tables should be saved,
                                         including the extension. If None, it looks up the default for each of tabletypes.
        tabletype, list/array of str's. List or array of table types to be used if tablenames is None to get the
                                        default name for each type of table.
        write_empty, bool. Whether to write an empty table to disk. The default is False. Warning: code is less robust
                           to column datatypes on read/write if the table is empty. May cause issues if this is set to True.
        overwrite, bool. Whether to overwrite the file on disk if it already exists. Default is currently True.
        verbose, bool. Whether to give verbose amounts of information (True) or succinct/no outputs (False). Default is False.

    Returns:
        Nothing.
    """
    log = get_logger()
    if tablenames is None and tabletypes is None:
        log.error("Need to define either tablenames or the table types in write_tables")
    elif tablenames is None:
        for tabl, tabltyp in zip(tables, tabletypes):
            if write_empty or len(tabl) > 0:
                write_table(tabl, tabletype=tabltyp, verbose=verbose, overwrite=overwrite, write_empty=write_empty)
    else:
        for tabl, tablname in zip(tables, tablenames):
            if write_empty or len(tabl) > 0:
                write_table(tabl, tablename=tablname, verbose=verbose, overwrite=overwrite, write_empty=write_empty)


def load_tables(tablenames=None, tabletypes=None, verbose=False):
    """
    Workflow function to read in multiple exposure, processing, and unprocessed tables. It allows for multi-valued
    table cells, which are generated from strings using the joinsymb. It reads from the files given by
    tablenames (or the default for tables of types in tabletypes).

    Args:
        tablename, list/array of str's. List or array of the full pathnames of where the tables should be saved,
                                        including the extension.
        tabletype, list/array of str's. List or array of the table types, which are used if tablenames is None to get
                                        the default name for the type of table. They are also used to get the
                                        column datatypes and defaults.
        verbose, bool. Whether to give verbose amounts of information (True) or succinct/no outputs (False). Default is False.

    Returns:
        tabs, list of Table's. Either exposure table or processing table that was loaded from tablename (or from default name
                      based on tabletype). Returns None if the file doesn't exist.
    """
    tabs = []
    if tablenames is None and tabletypes is None:
        pass
    elif tablenames is None:
        for tabltyp in tabletypes:
            tabs.append(load_table(tabletype=tabltyp, verbose=verbose))
    elif tabletypes is None:
        for tablname in tablenames:
            tabs.append(load_table(tablename=tablname, verbose=verbose))
    else:
        for tablname ,tabltyp in zip(tablenames , tabletypes):
            tabs.append(load_table(tablename=tablname, tabletype=tabltyp, verbose=verbose))
    return tabs
