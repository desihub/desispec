import os
import numpy as np
from astropy.table import Table


###################################################
################  Table Functions #################
###################################################
from desispec.workflow.utils import pathjoin


def ensure_scalar(val, joinsymb='|'):
    if val is None or type(val) in [str, np.str, np.str_, np.ma.core.MaskedConstant] or np.isscalar(val):
        return val
    else:
        val = np.atleast_1d(val).astype(str)
        outstr = joinsymb.join(val) + joinsymb
        return outstr

def split_str(val, joinsymb='|'):
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


def write_table(origtable, tablename=None, table_type=None, joinsymb='|', overwrite=True, verbose=False, write_empty=False):
    if tablename is None and table_type is None:
        print("Pathname or type of table is required to save the table")
        return

    if tablename is None:
        tablename = translate_type_to_pathname(table_type)

    if not write_empty and len(origtable) == 0:
        print(f'NOT writing zero length table to {tablename}')
        return
        
    if verbose:
        print("In write table" ,tablename ,'\n' ,table_type)
        print(origtable[0:2])
    basename, ext = os.path.splitext(tablename)

    temp_name = f'{basename}.temp{ext}'
    if verbose:
        print(ext ,temp_name)
    table = origtable.copy()

    if ext in ['.csv', '.ecsv']:
        if verbose:
            print("Given table: ", table.info)
        # replace_cols = {}

        for nam in table.colnames:
            ndim = table[nam].ndim
            if ndim > 1 or type(table[nam][0]) in [list, np.ndarray, np.array] or table[nam].dtype is object:
                if verbose:
                    print(f'{nam} is {ndim} dimensions, changing to string')
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
            print("A column was still more than one dimensional")
            print(table.info())

        table.write(temp_name, format=f'ascii{ext}', overwrite=overwrite)
    else:
        table.write(temp_name, overwrite=True)

    os.rename(temp_name, tablename)
    if verbose:
        print("Written table: ", table.info)


def translate_type_to_pathname(table_type):
    from desispec.workflow.exptable import get_exposure_table_path, get_exposure_table_pathname, get_exposure_table_name
    from desispec.workflow.proctable import get_processing_table_path, get_processing_table_pathname, get_processing_table_name
    if table_type.lower() in ['exp', 'exposure', 'etable']:
        tablename = get_exposure_table_pathname()
    elif table_type.lower() in ['proc', 'processing', 'int', 'ptable', 'interal']:
        tablename = get_processing_table_pathname()
    elif table_type.lower() in ['unproc', 'unproc_table', 'unprocessed', 'unprocessing']:
        tablepath = get_processing_table_path()
        tablename = get_processing_table_name().replace("processing", 'unprocessed')
        tablename = pathjoin(tablepath, tablename)
    return tablename


def load_table(tablename=None, table_type=None, joinsymb='|', verbose=False, process_mixins=True):
    from desispec.workflow.exptable import instantiate_exposure_table, get_exposure_table_column_defs
    from desispec.workflow.proctable import instantiate_processing_table, get_processing_table_column_defs

    if tablename is None:
        if table_type is None:
            print("Must specify either tablename or table_type in load_table()")
            return None
        else:
            tablename = translate_type_to_pathname(table_type)
    else:
        if table_type is None:
            print("table_type not given in load_table(), trying to guess based on filename")
            filename = os.path.split(tablename)[1]
            if 'exposure' in filename:
                table_type = 'exposure'
            elif 'unprocessed' in filename or 'unproc' in filename:
                table_type = 'unproc'
            elif 'processing' in filename:
                table_type = 'processing'
            elif 'etable' in filename or 'exp' in filename:
                table_type = 'exposure'
            elif 'ptable' in filename or 'proc' in filename:
                table_type = 'processing'

            if table_type is None:
                print(f"Couldn't identify type based on filename {filename}")
            else:
                print(f"Based on filename {filename}, identified type as {table_type}")

    if os.path.isfile(tablename):
        print(f"Found table: {tablename}")
    elif table_type is not None:
        print(f'Table {tablename} not found, creating new table of type {table_type}')
        if table_type.lower() in ['exp', 'exposure', 'etable']:
            return instantiate_exposure_table()
        elif table_type.lower() in ['unproc', 'unproc_table', 'unprocessed', 'unprocessing']:
            return instantiate_exposure_table()
        elif table_type.lower() in ['proc', 'processing', 'int', 'ptable', 'interal']:
            return instantiate_processing_table()
        else:
            print(f"Couldn't create type {table_type}, unknown table type")
            return None
    else:
        print(f"In load_table:\n\tCouldn't find: {table_type} and table_type not specified, returning None")
        return None

    basename, ext = os.path.splitext(tablename)
    if ext in ['.csv', '.ecsv']:
        table = Table.read(tablename, format=f'ascii{ext}')

        if verbose:
            print("Raw loaded table: ", table.info)

        if table_type.lower() in ['exp', 'exposure', 'etable', 'unproc', 'unproc_table', 'unprocessed', 'unprocessing']:
            colnames, coltypes, coldefaults = get_exposure_table_column_defs(return_default_values=True)
        elif table_type.lower() in ['proc', 'processing', 'int', 'ptable', 'interal']:
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
        print("Expanded table: ", table.info)
    return table

def guess_default_by_dtype(typ):
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
        print(first, firsttype, firsttype in [str, np.str, np.str_])
    if process_mixins and firsttype in [str, np.str, np.str_] and joinsymb in first:
        do_split_str = True
        if typ not in [list, np.array, np.ndarray]:
            print("Found mixin column with scalar datatype:")
            print("\tcolname={nam}, first={first}, typefirst={firsttyp}, dtype={typ}")
            print("\tchanging to np.array datatype")
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
        print(col)

    return col, dtyp


# def backup_tables(tables, fullpathnames=None, table_types=None):
#     return write_tables(tables, fullpathnames, table_types)

def write_tables(tables, fullpathnames=None, table_types=None, write_empty=False, verbose=False, overwrite=True):
    if fullpathnames is None and table_types is None:
        print("Need to define either fullpathnames or the table types in write_tables")
    elif fullpathnames is None:
        for tabl, tabltyp in zip(tables, table_types):
            if write_empty or len(tabl) > 0:
                write_table(tabl, table_type=tabltyp, verbose=verbose, overwrite=overwrite, write_empty=write_empty)
    else:
        for tabl, tablname in zip(tables, fullpathnames):
            if write_empty or len(tabl) > 0:
                write_table(tabl, tablename=tablname, verbose=verbose, overwrite=overwrite, write_empty=write_empty)


def load_tables(fullpathnames=None, tabtypes=None):
    tabs = []
    if fullpathnames is None and tabtypes is None:
        pass
    elif fullpathnames is None:
        for tabltyp in tabtypes:
            tabs.append(load_table(table_type=tabltyp))
    elif tabtypes is None:
        for tablname in fullpathnames:
            tabs.append(load_table(tablename=tablname))
    else:
        for tablname ,tabltyp in zip(fullpathnames ,tabtypes):
            tabs.append(load_table(tablename=tablname, table_type=tabltyp))
    return tabs
