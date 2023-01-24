"""
desispec.workflow.proc_dashboard_funcs
======================================

Please add module-level documentation.
"""
import os,glob
import json
import sys
import re
import time,datetime
import numpy as np
from os import listdir
from astropy.table import Table
from astropy.io import fits

########################
### Helper Functions ###
########################
from desispec.io import rawdata_root, specprod_root
from desispec.io.util import camword_to_spectros, decode_camword, \
    difference_camwords, create_camword, parse_badamps
from desispec.workflow.exptable import get_exposure_table_column_types, \
    default_obstypes_for_exptable, get_exposure_table_column_defaults, \
    get_exposure_table_pathname
from desispec.workflow.proctable import get_processing_table_pathname
from desispec.workflow.tableio import load_table


def get_output_dir(desi_spectro_redux, specprod, output_dir, makedir=True):
    if 'DESI_SPECTRO_DATA' not in os.environ.keys():
        os.environ['DESI_SPECTRO_DATA'] = '/global/cfs/cdirs/desi/spectro/data/'

    if specprod is None:
        if 'SPECPROD' not in os.environ.keys():
            os.environ['SPECPROD'] = 'daily'
        specprod = os.environ['SPECPROD']
    else:
        os.environ['SPECPROD'] = specprod

    if desi_spectro_redux is None:
        if 'DESI_SPECTRO_REDUX' not in os.environ.keys():  # these are not set by default in cronjob mode.
            os.environ['DESI_SPECTRO_REDUX'] = \
                '/global/cfs/cdirs/desi/spectro/redux/'
        desi_spectro_redux = os.environ['DESI_SPECTRO_REDUX']
    else:
        os.environ['DESI_SPECTRO_REDUX'] = desi_spectro_redux

    ## Verify the production directory exists
    prod_dir = os.path.join(desi_spectro_redux, specprod)
    if not os.path.exists(prod_dir):
        raise ValueError(
            f"Path {prod_dir} doesn't exist for production directory.")

    ## Define output_dir if not defined
    if output_dir is None:
        if 'DESI_DASHBOARD' not in os.environ.keys():
            os.environ['DESI_DASHBOARD'] = os.path.join(prod_dir,
                                                        'run', 'dashboard')
        output_dir = os.environ["DESI_DASHBOARD"]
    else:
        os.environ['DESI_DASHBOARD'] = output_dir

    ## Ensure we have directories to output to
    if makedir:
        os.makedirs(output_dir, exist_ok=True)

    return output_dir, prod_dir

def get_nights_dict(nights_arg, start_night, end_night, prod_dir):
    if nights_arg is None or nights_arg == 'all' \
            or (',' not in nights_arg and int(nights_arg) < 20000000):
        nights = list()
        for n in listdir(
                os.path.join(prod_dir, 'run', 'scripts', 'night')):
            # - nights are 20YYMMDD
            if re.match('^20\d{6}$', n):
                nights.append(n)
    else:
        nights = [nigh.strip(' \t') for nigh in nights_arg.split(',')]

    # tonight=what_night_is_it()   # Disabled per Anthony's request
    # if str(tonight) not in nights:
    #    nights.append(str(tonight))
    nights.sort(reverse=True)

    nights = np.array(nights)

    if start_night is not None:
        nights = nights[
            np.where(int(start_night) <= nights.astype(int))[0]]
    if end_night is not None:
        nights = nights[np.where(int(end_night) >= nights.astype(int))[0]]

    if nights_arg is not None and nights_arg.isnumeric() and len(
            nights) >= int(nights_arg):
        if end_night is None or start_night is not None:
            print(f"Only showing the most recent {int(nights_arg)} days")
            nights = nights[:int(nights_arg)]
        else:
            nights = nights[-1 * int(nights_arg):]

    nights_dict = dict()
    for night in nights:
        month = night[:6]
        if month not in nights_dict.keys():
            nights_dict[month] = [night]
        else:
            nights_dict[month].append(night)

    return nights_dict, nights

def get_tables(night, check_on_disk=False, exptab_colnames=None):
    if exptab_colnames is None:
        exptab_colnames = ['EXPID', 'FA_SURV', 'FAPRGRM', 'CAMWORD', 'BADCAMWORD',
                           'BADAMPS', 'EXPTIME', 'OBSTYPE', 'TILEID', 'COMMENTS',
                           'LASTSTEP']

    file_exptable = get_exposure_table_pathname(night)
    file_processing = get_processing_table_pathname(specprod=None,
                                                    prodmod=night)
    # procpath,procname = os.path.split(file_processing)
    # file_unprocessed = os.path.join(procpath,procname.replace('processing','unprocessed'))
    edefs = get_exposure_table_column_defaults(asdict=True)
    for col in exptab_colnames:
        if col not in edefs.keys():
            ValueError(f"requested dashboard exposure table column {col} not" +
                       f" in the exposure table columns: {edefs.keys()}.")

    try:  # Try reading tables first. Switch to counting files if failed.
        d_exp = load_table(file_exptable, tabletype='exptable')
        for col in exptab_colnames:
            if col not in d_exp.colnames:
                d_exp[col] = edefs[col]
    except:
        print(
            f'WARNING: Error reading exptable for {night}. Changing check_on_disk to True and scanning files on disk.')
        etypes = get_exposure_table_column_types(asdict=True)
        exptab_dtypes = [etypes[col] for col in exptab_colnames]
        d_exp = Table(names=exptab_colnames, dtype=exptab_dtypes)
        check_on_disk = True

    unaccounted_for_expids, unaccounted_for_tileids = [], []
    if check_on_disk:
        rawdatatemplate = os.path.join(rawdata_root(), night, '{zexpid}',
                                       'desi-{zexpid}.fits.fz')
        rawdata_fileglob = rawdatatemplate.format(zexpid='*')
        known_exposures = set(list(d_exp['EXPID']))
        newexpids = list(find_new_exps(rawdata_fileglob, known_exposures))
        newexpids.sort(reverse=True)
        default_obstypes = default_obstypes_for_exptable()
        for expid in newexpids:
            zfild_expid = str(expid).zfill(8)
            filename = rawdatatemplate.format(zexpid=zfild_expid)
            h1 = fits.getheader(filename, 1)
            header_info = {keyword: 'unknown' for keyword in
                           ['SPCGRPHS', 'EXPTIME',
                            'FA_SURV', 'FAPRGRM'
                                       'OBSTYPE', 'TILEID']}
            for keyword in header_info.keys():
                if keyword in h1.keys():
                    header_info[keyword] = h1[keyword]

            if header_info['OBSTYPE'] in default_obstypes:
                header_info['EXPID'] = expid
                header_info['LASTSTEP'] = 'all'
                header_info['COMMENTS'] = []
                if header_info['SPCGRPHS'] != 'unknown':
                    specs = str(header_info['SPCGRPHS']).replace(' ', '').replace(',', '')
                    header_info['CAMWORD'] = f'a{specs}'
                else:
                    header_info['CAMWORD'] = header_info['SPCGRPHS']
                header_info.pop('SPCGRPHS')
                d_exp.add_row(header_info)
                unaccounted_for_expids.append(expid)
                unaccounted_for_tileids.append(header_info['TILEID'])

    try:
        d_processing = load_table(file_processing, tabletype='proctable')
    except:
        d_processing = None
        print('WARNING: Error reading proctable. Only exposures in preproc'
              + ' directory will be marked as processing.')

    return d_exp, d_processing, np.array(unaccounted_for_expids), \
           np.unique(unaccounted_for_tileids)

def get_terminal_steps(expected_by_type):
    ## Determine the last filetype that is expected for each obstype
    terminal_steps = dict()
    for obstype, expected in expected_by_type.items():
        terminal_steps[obstype] = None
        keys = list(expected.keys())
        for key in reversed(keys):
            if expected[key] > 0:
                terminal_steps[obstype] = key
                break
    return terminal_steps

def get_file_list(filename, doaction=True):
    if doaction and filename is not None and os.path.exists(filename):
        output = np.atleast_1d(np.loadtxt(filename, dtype=int)).tolist()
    else:
        output = []
    return output

def get_skipped_ids(expid_filename, skip_ids=True):
    return get_file_list(filename=expid_filename, doaction=skip_ids)

def what_night_is_it():
    """
    Return the current night
    """
    d = datetime.datetime.utcnow() - datetime.timedelta(7 / 24 + 0.5)
    tonight = int(d.strftime('%Y%m%d'))
    return tonight

def find_new_exps(fileglob, known_exposures):
    """
    Check the path given for new exposures
    """
    datafiles = sorted(glob.glob(fileglob))
    newexp = list()
    for filepath in datafiles:
        expid = int(os.path.basename(os.path.dirname(filepath)))
        if expid not in known_exposures:
            newexp.append(expid)

    return set(newexp)

def check_running(proc_name= 'desi_dailyproc',suppress_outputs=False):
    """
    Check if the desi_dailyproc process is running
    """
    import psutil
    running = False
    mypid = os.getpid()
    for p in psutil.process_iter():
        if p.pid != mypid and proc_name in ' '.join(p.cmdline()):
            if not suppress_outputs:
                print('ERROR: {} already running as PID {}:'.format(proc_name,p.pid))
                print('  ' + ' '.join(p.cmdline()))
            running = True
            break
    return running


#################################
### HTML Generating Functions ###
#################################
def return_color_profile():
    color_profile = dict()
    color_profile['DEFAULT'] = {'font':'#000000' ,'background':'#ccd1d1'} # gray
    color_profile['NULL'] = {'font': '#34495e', 'background': '#ccd1d1'}  # gray on gray
    color_profile['BAD'] = {'font':'#000000' ,'background':'#d98880'}  #  red
    color_profile['INCOMPLETE'] = {'font': '#000000','background':'#f39c12'}  #  orange
    color_profile['GOOD'] = {'font':'#000000' ,'background':'#7fb3d5'}   #  blue
    color_profile['OVERFULL'] = {'font': '#000000','background':'#c39bd3'}   # purple
    return color_profile


def make_html_page(monthly_tables, outfile, titlefill='Processing',
                   show_null=False, color_profile=None):
    if color_profile is None:
        color_profile = return_color_profile()

    html_page = _initialize_page(color_profile, titlefill=titlefill)

    for month, nightly_tables in monthly_tables.items():
        if len(nightly_tables) == 0:
            continue
        print(
            "Month: {}, nights: {}".format(month, list(nightly_tables.keys())))
        nightly_table_htmls, statuses = list(), list()
        for night, night_info in nightly_tables.items():
            if len(night_info) == 0:
                continue
            ####################################
            ### Table for individual night ####
            ####################################
            nightly_table_html, status = \
                generate_nightly_table_html(night_info, night, show_null)
            nightly_table_htmls.append(nightly_table_html)
            statuses.append(status)
        html_page += generate_monthly_table_html(nightly_table_htmls,
                                                 statuses, month)

    # html_page += js_import_str(os.environ['DESI_DASHBOARD'])
    html_page += js_str()
    html_page += _closing_str()
    with open(outfile, 'w') as hs:
        hs.write(html_page)
        print(f"Write to {outfile} complete.")

    if 'NERSC_HOST' in os.environ and outfile.startswith(
            '/global/cfs/cdirs/desi'):
        url = outfile.replace('/global/cfs/cdirs/desi',
                              'https://data.desi.lbl.gov/desi')
        print(f"This can be found via webserver at: {url}")

def generate_monthly_table_html(tables, statuses, month):
    """
    Add a collapsible and extendable table to the html file for a specific month
    Input
    tables: list of tables generated by 'nightly_table'
    month: string of YYYYMM, e.g. 202001
    output: The string to be added to the html file
    """
    month_dict = {'01':'January','02':'February','03':'March','04':'April','05':'May','06':'June',
                  '07':'July','08':'August','09':'September','10':'October','11':'November','12':'December'}

    heading = f"{month_dict[month[4:]]} {month[:4]} ({month})"
    month_table_str = '\n<!--Begin {}-->\n'.format(month)

    statuses = np.array(statuses)
    monthlystatus = 'DEFAULT'
    if np.all(statuses == 'GOOD'):
        monthlystatus = 'GOOD'
    elif np.any(statuses == 'BAD'):
        monthlystatus = 'BAD'
    elif np.any(statuses == 'INCOMPLETE'):
        monthlystatus = 'INCOMPLETE'
    elif np.any(statuses == 'OVERFULL'):
        monthlystatus = 'OVERFULL'
    month_table_str += f'<button class="collapsible" id="{monthlystatus}">' \
                       + heading + '</button>'

    month_table_str += '<div class="content" style="display:inline-block;min-height:0%;">\n'
    #month_table_str += "<table id='c'>"
    for table_str in tables:
        month_table_str += table_str

    #month_table_str += "</table></div>\n"
    month_table_str += "</div>\n"
    month_table_str += '<!--End {}-->\n\n'.format(month)

    return month_table_str

def generate_nightly_table_html(night_info, night, show_null):
    if len(night_info) == 0:
        return '', 'NULL'

    ngood, ninter, nbad, nnull, nover, n_notnull, noprocess, norecord = \
        0, 0, 0, 0, 0, 0, 0, 0

    main_body = ""
    for key, row_info in reversed(night_info.items()):
        table_row = _table_row(row_info)
        if not show_null and 'NULL' in table_row:
            continue
        main_body += ("\t" + table_row + "\n")
        status = str(row_info["STATUS"]).lower()
        if status == 'processing':
            if 'COLOR' in row_info:
                color = row_info['COLOR']
            else:
                color = table_row.split('">')[0].split('id="')[1]
            if color == 'GOOD':
                ngood += 1
                n_notnull += 1
            elif color == 'BAD':
                nbad += 1
                n_notnull += 1
            elif color == 'INCOMPLETE':
                ninter += 1
                n_notnull += 1
            elif color == 'OVERFULL':
                nover += 1
                n_notnull += 1
            else:
                nnull += 1
        elif status == 'unprocessed':
            noprocess += 1
        elif status == 'unrecorded':
            norecord += 1
        else:
            nnull += 1

    # Night dropdown table
    htmltab = r'&nbsp;&nbsp;&nbsp;&nbsp;'
    heading = (f"Night {night}{htmltab}"
               + f"Complete: {ngood}/{n_notnull}{htmltab}"
               + f"Incomplete: {ninter}/{n_notnull}{htmltab}"
               + f"Failed: {nbad}/{n_notnull}{htmltab}"
               + f"Overfull: {nover}/{n_notnull}{htmltab}"
               + f"Unprocessed: {noprocess}{htmltab}"
               + f"NoTabEntry: {norecord}{htmltab}"
               + f"Other: {nnull}"
               )

    night_status = 'DEFAULT'
    if ngood == n_notnull:
        night_status = "GOOD"
    elif ninter > 0:
        night_status = "INCOMPLETE"
    elif nbad > 0:
        night_status = "BAD"
    elif nover > 0:
        night_status = "OVERFULL"
    nightly_table_str = '<!--Begin {}-->\n'.format(night)
    nightly_table_str += f'<button class="collapsible" id="{night_status}">{heading}</button>'
    nightly_table_str += '<div class="content" style="display:inline-block;min-height:0%;">\n'
    # table header
    nightly_table_str += "<table id='c' class='nightTable'><tbody>\n\t<tr>"
    for col in list(row_info.keys()):
        colname = str(col).upper()
        if colname != 'COLOR':
            nightly_table_str += f"<th>{colname}</th>"
    nightly_table_str += "</tr>\n"

    # Add body
    nightly_table_str += main_body

    # End table
    nightly_table_str += "</tbody></table></div>\n"
    nightly_table_str += '<!--End {}-->\n\n'.format(night)
    return nightly_table_str, night_status


def read_json(filename_json):
    night_json_info = None
    if os.path.exists(filename_json):
        with open(filename_json) as json_file:
            try:
                night_json_info = json.load(json_file)
            except:
                print(f"Error trying to load {filename_json}, "
                      + "continuing without that information.")
    return night_json_info


def write_json(output_data, filename_json):
    ## write out the night_info to json file
    with open(filename_json, 'w') as json_file:
        try:
            json.dump(output_data, json_file)
        except:
            print(f"Error trying to dump {filename_json}, "
                  + "not saving that information.")


def _initialize_page(color_profile, titlefill='Processing'):
    """
    Initialize the html file for showing the statistics, giving all the headers and CSS setups.
    """
    # strTable="<html><style> table {font-family: arial, sans-serif;border-collapse: collapse;width: 100%;}"
    # strTable=strTable+"td, th {border: 1px solid #dddddd;text-align: left;padding: 8px;}"
    # strTable=strTable+"tr:nth-child(even) {background-color: #dddddd;}</style>"
    html_page = """<html><head>
<meta http-equiv="content-type" content="text/html; charset=UTF-8"><style>
    h1 {font-family: 'sans-serif';font-size:50px;color:#4CAF50}
    #c {font-family: 'Trebuchet MS', Arial, Helvetica, sans-serif;border-collapse: collapse;width: 100%;}
    #c td, #c th {border: 1px solid #ddd;padding: 8px;}
    /* #c tr:nth-child(even){background-color: #f2f2f2;} */
    #c tr:hover {background-color: #ddd;}
    #c th {padding-top: 12px;  padding-bottom: 12px;  text-align: left;  background-color: #34495e;  color: white;}
    .collapsible {background-color: #eee;color: #444;cursor: pointer;padding: 18px;width: 100%;border: none;text-align: left;outline: none;font-size: 25px;}
    .regular {background-color: #eee;color: #444;  cursor: pointer;  padding: 18px;  width: 25%;  border: 18px;  text-align: left;  outline: none;  font-size: 25px;}
    .active, .collapsible:hover { background-color: #ccc;}
    .content {padding: 0 18px;display: table;overflow: hidden;background-color: #f1f1f1;maxHeight:0px;}
    /* The Modal (background) */
    .modal {
    display: none;        /* Hidden by default */
    position: fixed;      /* Stay in place */
    z-index: 1;           /* Sit on top */
    padding-top: 100px;  /* Location of the box */
    left: 0;
    top: 0;
    width: 100%;         /* Full width */
    height: 90%;        /* Full height */
    overflow: auto;     /* Enable scroll if needed */
    background-color: rgb(0,0,0);      /* Fallback color */
    background-color: rgba(0,0,0,0.4); /* Black w/ opacity */
    }

    /* Modal Content */
    .modal-content {
    background-color: #fefefe;
    margin: auto;
    padding: 20px;
    border: 1px solid #888;
    width: 80%;
    }


   /* The Close Button */
   .close {
    color: #aaaaaa;
    float: right;
    font-size: 28px;
    font-weight: bold;
    }
    .close:hover,
    .close:focus {
         color: #000;
         text-decoration: none;
         cursor: pointer;
     }

    #obstypelist {

          background-position: 10px 10px;
          background-repeat: no-repeat;
          width: 10%;
          font-size: 16px;
          padding: 12px 20px 12px 40px;
          border: 1px solid #ddd;
          margin-bottom: 12px;
    }
    #exptimelist {

          background-position: 10px 10px;
          background-repeat: no-repeat;
          width: 10%;
          font-size: 16px;
          padding: 12px 20px 12px 40px;
          border: 1px solid #ddd;
          margin-bottom: 12px;
    }

    """

    for ctype,cdict in color_profile.items():
        background = cdict['background']
        html_page += f'\t#{ctype} ' + '{background-color:' + f'{background}' + ';}\n'

    html_page + "\n"
    ## Table rows shouldn't do the default background because of cell coloring
    for ctype,cdict in color_profile.items():
        font = cdict['font']
        background = '#eee'#cdict['background'] # no background for a whole table after implementing color codes for processing columns
        html_page += f'\ttable tr#{ctype} '+'{background-color:'+f'{background}; color:{font}'+';}\n'

    ## Finally there is a class of table element that is null in a good way
    ## Label as such
    html_page += 'table td#GOODNULL {background-color:#7fb3d5;color:gray}\n'

    html_page += '</style>\n\n'
    html_page += f"</head><body><h1>DESI '{os.environ['SPECPROD']}' "
    html_page += f'{titlefill} Status Monitor</h1>\n'

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    # running='No'
    # if check_running(proc_name='desi_dailyproc',suppress_outputs=True):
    #     running='Yes'
    #     strTable=strTable+"<div style='color:#00FF00'>{} {} running: {}</div>\n".format(timestamp,'desi_dailyproc',running)
    script = os.path.basename(sys.argv[0])
    html_page += f'<div style="color:#00FF00;margin-bottom:20px"> {script} running at: {timestamp}</div>\n'

    html_page += 'Color Legend:\n'
    html_page += '<table style="margin-bottom:20px;margin-left:20px"><tr>\n'

    for ctype in color_profile.keys():
        html_page += f' <td id="{ctype}">{ctype}</td>\n'
    html_page += f'</tr></table>\n'

    html_page += '\n\n'
    html_page += """Filter By Status:
    <select style="margin-bottom:10px" id="statuslist" onchange="filterByStatus()" class='form-control'>
    <option>processing</option>
    <option>unprocessed</option>
    <option>unaccounted</option>
    <option>ALL</option>
    </select>
    """
    # The following codes are for filtering rows by obstype and exptime. Not in use for now, but basically can be enabled anytime.
    # html_page +="""Filter By OBSTYPE:
    # <select id="obstypelist" onchange="filterByObstype()" class='form-control'>
    # <option>ALL</option>
    # <option>SCIENCE</option>
    # <option>FLAT</option>
    # <option>ARC</option>
    # <option>DARK</option>
    # </select>
    # Exptime Limit:
    # <select id="exptimelist" onchange="filterByExptime()" class='form-control'>
    # <option>ALL</option>
    # <option>5</option>
    # <option>30</option>
    # <option>120</option>
    # <option>900</option>
    # <option>1200</option>
    # </select>
    # """
    return html_page

def _closing_str():
    closing = """<div class="crt-wrapper"></div>
                 <div class="aadvantage-wrapper"></div>
                 </body></html>"""
    return closing

def _table_row(dictionary):
    idlabel = dictionary.pop('COLOR')
    color_profile = return_color_profile()
    if dictionary["STATUS"] != 'processing':
        style_str = 'display:none;'
    else:
        style_str = ''

    if idlabel is None:
        row_str = '<tr style="{}">'.format(style_str)
    else:
        row_str = '<tr style="'+style_str+'" id="'+str(idlabel)+'">'

    for elem in dictionary.values():
        chars = str(elem).split('/')
        if len(chars)==2: # m/n
            if chars[0]=='0' and chars[1]=='0':
                row_str += _table_element_id(elem, 'GOODNULL')
            elif chars[0]=='0' and chars[1]!='0':
                row_str += _table_element_id(elem, 'BAD')
            elif chars[0]!='0' and int(chars[0])<int(chars[1]):
                row_str += _table_element_id(elem, 'INCOMPLETE')
            elif chars[0]!='0' and int(chars[0])==int(chars[1]):
                row_str += _table_element_id(elem, 'GOOD')
            else:
                row_str += _table_element_id(elem, 'OVERFULL')

        else:
            row_str += _table_element(elem)
    row_str += '</tr>'#\n'
    return row_str

def _table_element(elem):
    return '<td>{}</td>'.format(elem)

def _table_element_style(elem,style):
    return f'<td style="{style}">{elem}</td>'

def _table_element_id(elem,id):
    return f'<td id="{id}">{elem}</td>'

def _hyperlink(rel_path,displayname):
    hlink = f'<a href="{rel_path}" target="_blank"' \
            + f' rel="noopener noreferrer">{displayname}</a>'
    return hlink

def _str_frac(numerator,denominator):
    frac = f'{numerator}/{denominator}'
    return frac

def _js_path(output_dir):
    return os.path.join(output_dir,'js','open_nightly_table.js')

def js_import_str(output_dir):  # Not used
    output_path = _js_path(output_dir)
    if not os.path.exists(os.path.join(output_dir,'js')):
        os.makedirs(os.path.join(output_dir,'js'))
    if not os.path.exists(output_path):
        _write_js_script(output_path)
    return f'<script type="text/javascript" src="{output_path}"></script>'

def _write_js_script(output_path):
    """
    Return the javascript script to be added to the html file
    """
    s="""
        var coll = document.getElementsByClassName('collapsible');
        var i;
        for (i = 0; i < coll.length; i++) {
            coll[i].nextElementSibling.style.maxHeight='0px';
            coll[i].addEventListener('click', function() {
                this.classList.toggle('active');
                var content = this.nextElementSibling;
                if (content.style.maxHeight){
                   content.style.maxHeight = null;
                } else {
                  content.style.maxHeight = '0px';
                        }
                });
         };
         var b1 = document.getElementById('b1');
         b1.addEventListener('click',function() {
             for (i = 0; i < coll.length; i++) {
                 coll[i].nextElementSibling.style.maxHeight=null;
                                               }});
         var b2 = document.getElementById('b2');
         b2.addEventListener('click',function() {
             for (i = 0; i < coll.length; i++) {
                 coll[i].nextElementSibling.style.maxHeight='0px'
                         }});
        """
    with open(output_path,'w') as outjs:
        outjs.write(s)

def js_str(): # Used
    """
        Return the javascript script to be added to the html file
    """
    s="""
        <script >
            var coll = document.getElementsByClassName('collapsible');
            var i;
            for (i = 0; i < coll.length; i++) {
                coll[i].nextElementSibling.style.maxHeight='0px';
                coll[i].addEventListener('click', function() {
                    this.classList.toggle('active');
                    var content = this.nextElementSibling;
                    if (content.style.maxHeight){
                       content.style.maxHeight = null;
                    } else {
                      content.style.maxHeight = '0px';
                            }
                    });
             };
             var b1 = document.getElementById('b1');
             b1.addEventListener('click',function() {
                 for (i = 0; i < coll.length; i++) {
                     coll[i].nextElementSibling.style.maxHeight=null;
                                                   }});
             var b2 = document.getElementById('b2');
             b2.addEventListener('click',function() {
                 for (i = 0; i < coll.length; i++) {
                     coll[i].nextElementSibling.style.maxHeight='0px'
                             }});
           function filterByStatus() {
                var input, filter, table, tr, td, i;
                input = document.getElementById("statuslist");
                filter = input.value.toUpperCase();
                tables = document.getElementsByClassName("nightTable")
                for (j = 0; j < tables.length; j++){
                 table = tables[j]
                 tr = table.getElementsByTagName("tr");
                 for (i = 0; i < tr.length; i++) {
                   td = tr[i].getElementsByTagName("td")[15];
                   console.log(td)
                   if (td) {
                       if (td.innerHTML.toUpperCase().indexOf(filter) > -1 || filter==='ALL') {
                           tr[i].style.display = "";
                       } else {
                          tr[i].style.display = "none";
                              }
                       }
                                                                            }
                             }}

           function filterByObstype() {
                var input, filter, table, tr, td, i;
                input = document.getElementById("obstypelist");
                filter = input.value.toUpperCase();
                tables = document.getElementsByClassName("nightTable")
                for (j = 0; j < tables.length; j++){
                 table = tables[j]
                 tr = table.getElementsByTagName("tr");
                 for (i = 0; i < tr.length; i++) {
                   td = tr[i].getElementsByTagName("td")[2];
                   if (td) {
                       if (td.innerHTML.toUpperCase().indexOf(filter) > -1 || filter==='ALL') {
                           tr[i].style.display = "";
                       } else {
                          tr[i].style.display = "none";
                              }
                       }
                                                                            }
                             }}

            function filterByExptime() {
                var input, filter, table, tr, td, i;
                input = document.getElementById("exptimelist");
                filter = input.value.toUpperCase();
                tables = document.getElementsByClassName("nightTable")
                for (j = 0; j < tables.length; j++){
                 table = tables[j]
                 tr = table.getElementsByTagName("tr");
                 for (i = 0; i < tr.length; i++) {
                   td = tr[i].getElementsByTagName("td")[3];
                   if (td) {
                       if (filter==='ALL') {
                           tr[i].style.display = "";
                       } else if (parseInt(td.innerHTML) <= parseInt(filter)){
                           tr[i].style.display = ""; }
                       else {
                          tr[i].style.display = "none";
                              }
                       }
                                                                            }
                             }}


       </script>
        """
    return s
