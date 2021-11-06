import argparse
import os,glob
import re
from astropy.io import fits,ascii
from astropy.table import Table, vstack
import time,datetime
import numpy as np
import psutil
from os import listdir
import json

# import desispec.io.util
from desispec.workflow.exptable import get_exposure_table_pathname, default_obstypes_for_exptable
from desispec.workflow.proctable import get_processing_table_pathname
from desispec.workflow.tableio import load_table
from desispec.io.meta import specprod_root, rawdata_root
from desispec.io.util import decode_camword, camword_to_spectros, difference_camwords, parse_badamps, create_camword


########################
### Helper Functions ###
########################

def return_color_profile():
    color_profile = dict()
    color_profile['NULL'] = {'font':'#34495e' ,'background':'#ccd1d1'} # gray
    color_profile['BAD'] = {'font':'#000000' ,'background':'#d98880'}  #  red
    color_profile['INCOMPLETE'] = {'font': '#000000','background':'#f39c12'}  #  orange
    color_profile['GOOD'] = {'font':'#000000' ,'background':'#7fb3d5'}   #  blue
    color_profile['OVERFUL'] = {'font': '#000000','background':'#c39bd3'}   # purple
    return color_profile

def get_file_list(filename, doaction=True):
    if doaction and filename is not None and os.path.exists(filename):
        output = np.atleast_1d(np.loadtxt(filename, dtype=int)).tolist()
    else:
        output = []
    return output

def get_skipped_expids(expid_filename, skip_expids=True):
    return get_file_list(filename=expid_filename, doaction=skip_expids)

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

def parse(options):
    """
    Initialize the parser to read input
    """
    # Initialize
    parser = argparse.ArgumentParser(description="Search the filesystem and summarize the existance of files output from "+
                                     "the daily processing pipeline. Can specify specific nights, give a number of past nights,"+
                                     " or use --all to get all past nights.")

    # File I/O
    parser.add_argument('--redux-dir', type=str, help="Product directory, point to $DESI_SPECTRO_REDUX by default ")
    parser.add_argument('--output-dir', type=str, help="output portal directory for the html pages, which defaults to your home directory ")
    parser.add_argument('--output-name', type=str, default='dashboard.html', help="name of the html page (to be placed in --output-dir).")
    parser.add_argument('--specprod',type=str, help="overwrite the environment keyword for $SPECPROD")
    parser.add_argument("-e", "--skip-expid-file", type=str, required=False,
                        help="Relative pathname for file containing expid's to skip. "+\
                             "Automatically. They are assumed to be in a column"+\
                             "format, one per row. Stored internally as integers, so zero padding is "+\
                             "accepted but not required.")
    #parser.add_argument("--skip-null", type=str, required=False,
    #                    help="Relative pathname for file containing expid's to skip. "+\
    #                         "Automatically. They are assumed to be in a column"+\
    #                         "format, one per row. Stored internally as integers, so zero padding is "+\
    #                         "accepted but not required.")
    # Specify Nights of Interest
    parser.add_argument('-n','--nights', type=str, default = None, required = False,
                        help="nights to monitor. Can be 'all', a comma separated list of YYYYMMDD, or a number "+
                             "specifying the previous n nights to show (counting in reverse chronological order).")
    parser.add_argument('--start-night', type=str, default = None, required = False,
                        help="This specifies the first night to include in the dashboard. "+
                             "Default is the earliest night available.")
    parser.add_argument('--end-night', type=str, default = None, required = False,
                        help="This specifies the last night (inclusive) to include in the dashboard. Default is today.")
    parser.add_argument('--check-on-disk',  action="store_true",
                        help="Check raw data directory for additional unaccounted for exposures on disk "+
                             "beyond the exposure table.")
    parser.add_argument('--ignore-json-archive',  action="store_true",
                        help="Ignore the existing json archive of good exposure rows, regenerate all rows from "+
                             "information on disk. As always, this will write out a new json archive," +
                             " overwriting the existing one.")
    # Read in command line and return
    args = None
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)

    return args



######################
### Main Functions ###
######################
def main(args):
    """ Code to generate a webpage for monitoring of desi_dailyproc production status
    Usage:
    -n can be 'all' or series of nights separated by comma or blank like 20200101,20200102 or 20200101 20200102
    Normal Mode:
    desi_proc_dashboard -n 3  --output-dir /global/cfs/cdirs/desi/www/collab/dailyproc/
    desi_proc_dashboard -n 20200101,20200102 --output-dir /global/cfs/cdirs/desi/www/collab/dailyproc/
    """
    args.show_null = True
    if 'DESI_SPECTRO_DATA' not in os.environ.keys():
        os.environ['DESI_SPECTRO_DATA'] = '/global/cfs/cdirs/desi/spectro/data/'

    if 'SPECPROD' not in os.environ.keys() and args.specprod is None:
        os.environ['SPECPROD']='daily'
    elif args.specprod is None:
        args.specprod = os.environ["SPECPROD"]
    else:
        os.environ['SPECPROD'] = args.specprod

    if args.redux_dir is None:
        if 'DESI_SPECTRO_REDUX' not in os.environ.keys(): # these are not set by default in cronjob mode.
            os.environ['DESI_SPECTRO_REDUX'] = '/global/cfs/cdirs/desi/spectro/redux/'
        args.redux_dir = os.environ['DESI_SPECTRO_REDUX']
    else:
        os.environ['DESI_SPECTRO_REDUX'] = args.redux_dir

    if args.output_dir is None:
        if 'DESI_DASHBOARD' not in os.environ.keys():
            os.environ['DESI_DASHBOARD']=os.environ["HOME"]
        args.output_dir = os.environ["DESI_DASHBOARD"]
    else:
        os.environ['DESI_DASHBOARD'] = args.output_dir

    ## Verify the production directory exists
    args.prod_dir = os.path.join(args.redux_dir,args.specprod)
    if not os.path.exists(args.prod_dir):
        raise ValueError(f"Path {args.prod_dir} doesn't exist for production directory.")

    ## Ensure we have directories to output to
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir,'files'), exist_ok=True)

    ############
    ## Input ###
    ############
    if args.skip_expid_file is not None:
        skipd_expids = set(get_skipped_expids(args.skip_expid_file, skip_expids=True))
    else:
        skipd_expids = set()
        
    if args.nights is None or args.nights=='all' or ',' not in args.nights:
        nights = list()
        for n in listdir( os.path.join(args.prod_dir,'run','scripts','night') ):
            #- nights are 20YYMMDD
            if re.match('^20\d{6}$', n):
                nights.append(n)
    else:
        nights = [nigh.strip(' \t') for nigh in args.nights.split(',')]

    #tonight=what_night_is_it()   # Disabled per Anthony's request
    #if str(tonight) not in nights:
    #    nights.append(str(tonight))
    nights.sort(reverse=True)

    nights = np.array(nights)

    if args.start_night is not None:
        nights = nights[np.where(int(args.start_night)<=nights.astype(int))[0]]
    if args.end_night is not None:
        nights = nights[np.where(int(args.end_night)>=nights.astype(int))[0]]

    if args.nights is not None and args.nights.isnumeric() and len(nights) >= int(args.nights):
        if args.end_night is None or args.start_night is not None:
            print("Only showing the most recent {} days".format(int(args.nights)))
            nights = nights[:int(args.nights)]
        else:
            nights = nights[-1*int(args.nights):]
            
    nights_dict = dict()
    for night in nights:
        month = night[:6]
        if month not in nights_dict.keys():
            nights_dict[month] = [night]
        else:
            nights_dict[month].append(night)


    print('Searching '+args.prod_dir+' for ',nights)

    ######################################################################
    ## sub directories. Should not change if generated by the same code ##
    ## that follows the same directory strucure ##
    ######################################################################
    color_profile = return_color_profile()
    strTable = _initialize_page(color_profile)

    timestamp=time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())
    # running='No'
    # if check_running(proc_name='desi_dailyproc',suppress_outputs=True):
    #     running='Yes'
    #     strTable=strTable+"<div style='color:#00FF00'>{} {} running: {}</div>\n".format(timestamp,'desi_dailyproc',running)
    strTable +=  f"<div style='color:#00FF00'> desi_dailyproc running at: {timestamp}</div>\n"

    for ctype,cdict in color_profile.items():
        background = cdict['background']
        strTable += "\t<div style='color:{}'>{}</div>".format(background,ctype)
        
    strTable += '\n\n'
    strTable +="""Filter By Status:
    <select id="statuslist" onchange="filterByStatus()" class='form-control'>
    <option>processing</option>
    <option>unprocessed</option>
    <option>unaccounted</option>
    <option>ALL</option>
    </select>
    """
    # The following codes are for filtering rows by obstype and exptime. Not in use for now, but basically can be enabled anytime. 
    #strTable +="""Filter By OBSTYPE:
    #<select id="obstypelist" onchange="filterByObstype()" class='form-control'>
    #<option>ALL</option>
    #<option>SCIENCE</option>
    #<option>FLAT</option>
    #<option>ARC</option>
    #<option>DARK</option>
    #</select>
    #Exptime Limit:
    #<select id="exptimelist" onchange="filterByExptime()" class='form-control'>
    #<option>ALL</option>
    #<option>5</option>
    #<option>30</option>
    #<option>120</option>
    #<option>900</option>
    #<option>1200</option>
    #</select>
    #"""
    for month, nights_in_month in nights_dict.items():
        print("Month: {}, nights: {}".format(month,nights_in_month))
        nightly_tables = []
        for night in nights_in_month:
            ####################################
            ### Table for individual night ####
            ####################################
            nightly_tables.append(nightly_table(night, args.output_dir, skipd_expids, show_null=args.show_null,
                                                check_on_disk=args.check_on_disk,
                                                ignore_json_archive=args.ignore_json_archive))
        strTable += monthly_table(nightly_tables,month)

    #strTable += js_import_str(os.environ['DESI_DASHBOARD'])
    strTable += js_str()
    strTable += _closing_str()
    with open(os.path.join(os.environ['DESI_DASHBOARD'],args.output_name),'w') as hs:
        hs.write(strTable)

def monthly_table(tables,month):
    """
    Add a collapsible and extendable table to the html file for a specific month
    Input
    table: the table generated by 'nightly_table'
    night: like 20200131
    output: The string to be added to the html file
    """
    month_dict = {'01':'January','02':'February','03':'March','04':'April','05':'May','06':'June',
                  '07':'July','08':'August','09':'September','10':'October','11':'November','12':'December'}

    heading="{} {} ({})".format(month_dict[month[4:]],month[:4],month)
    month_table_str = '\n<!--Begin {}-->\n'.format(month)
    month_table_str += '<button class="collapsible">'+heading+'</button><div class="content" style="display:inline-block;min-height:0%;">\n'
    #month_table_str += "<table id='c'>"
    for table_str in tables:
        month_table_str += table_str

    #month_table_str += "</table></div>\n"
    month_table_str += "</div>\n"
    month_table_str += '<!--End {}-->\n\n'.format(month)

    return month_table_str

def nightly_table(night,output_dir,skipd_expids=set(),show_null=True,check_on_disk=False,ignore_json_archive=False):
    """
    Add a collapsible and extendable table to the html file for one specific night
    Input
    night: like 20200131
    output: The string to be added to the html file
    """
    filename_json = os.path.join(output_dir,'files','night_info_'+os.environ['SPECPROD']+'_'+night+'.json')
    if not ignore_json_archive and os.path.exists(filename_json):
        with open(filename_json) as json_file:
            try:
                night_info_pre=json.load(json_file)
                night_info = calculate_one_night_use_file(night,check_on_disk,night_info_pre=night_info_pre)
            except:
                night_info = calculate_one_night_use_file(night,check_on_disk)
    else:
        night_info = calculate_one_night_use_file(night,check_on_disk)

    with open(filename_json,'w') as json_file:
        json.dump(night_info,json_file)

    ngood,ninter,nbad,nnull,nover,n_notnull,noprocess,norecord = 0,0,0,0,0,0,0,0
    main_body = ""
    for expid in reversed(night_info.keys()):
        if int(expid) in skipd_expids:
            continue
        row_info = night_info[expid]
        table_row = _table_row(row_info[1:],idlabel=row_info[0])

        if not show_null and 'NULL' in table_row:
            continue
        
        main_body += ("\t" + table_row + "\n")
        status = str(row_info[-1]).lower()
        if status == 'processing':
            if 'GOOD' in table_row:
                ngood += 1
                n_notnull += 1
            elif 'BAD' in table_row:
                nbad += 1
                n_notnull += 1
            elif 'INCOMPLETE' in table_row:
                ninter += 1
                n_notnull += 1
            elif 'OVERFUL' in table_row:
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
                + f"Unprocessed: {noprocess}{htmltab}"
                + f"NoTabEntry: {norecord}{htmltab}"
                + f"Other: {nnull}"
               )

    nightly_table_str= '<!--Begin {}-->\n'.format(night)
    nightly_table_str += '<button class="collapsible">' + heading + \
                         '</button><div class="content" style="display:inline-block;min-height:0%;">\n'
    # table header
    nightly_table_str += ("<table id='c' class='nightTable'><tbody>\n"
                          + "\t<tr>"
                          + "<th>EXPID</th>"
                          + "<th>TILE ID</th>"
                          + "<th>OBSTYPE</th>"
                          + "<th>FA SURV</th>"
                          + "<th>FA PRGRM</th>"
                          + "<th>LAST STEP</th>"
                          + "<th>EXP TIME</th>"
                          + "<th>PROC CAMWORD</th>"
                          + "<th>PSF File</th>"
                          + "<th>frame file</th>"
                          + "<th>FFlat file</th>"
                          + "<th>sframe file</th>"
                          + "<th>sky file</th>"
                          + "<th>std star</th>"
                          + "<th>cframe file</th>"
                          + "<th>slurm file</th>"
                          + "<th>log file</th>"
                          + "<th>COMMENTS</th>"
                          + "<th>status</th>"
                          + "</tr>\n"
                          )
    # Add body
    nightly_table_str += main_body

    # End table
    nightly_table_str += "</tbody></table></div>\n"
    nightly_table_str += '<!--End {}-->\n\n'.format(night)
    return nightly_table_str


def calculate_one_night_use_file(night, check_on_disk=False, night_info_pre=None):
    """
    For a given night, return the file counts and other other information for each exposure taken on that night
    input: night
    output: a dictionary containing the statistics with expid as key name
    FLAVOR: FLAVOR of this exposure
    OBSTYPE: OBSTYPE of this exposure
    EXPTIME: Exposure time
    SPECTROGRAPHS: a list of spectrographs used
    n_spectrographs: number of spectrographs
    n_psf: number of PSF files
    n_ff:  number of fiberflat files
    n_frame: number of frame files
    n_sframe: number of sframe files
    n_cframe: number of cframe files
    n_sky: number of sky files
    """
    ## Note that the following list should be in order of processing. I.e. the first filetype given should be the
    ## first file type generated. This is assumed for the automated "terminal step" determination that follows
    expected_by_type = dict()
    expected_by_type['arc'] =     {'psf': 1, 'frame': 0, 'ff': 0, 'sframe': 0, 'std': 0, 'cframe': 0}
    expected_by_type['flat'] =    {'psf': 1, 'frame': 1, 'ff': 1, 'sframe': 0, 'std': 0, 'cframe': 0}
    expected_by_type['science'] = {'psf': 1, 'frame': 1, 'ff': 0, 'sframe': 1, 'std': 1, 'cframe': 1}
    expected_by_type['twilight'] ={'psf': 1, 'frame': 1, 'ff': 0, 'sframe': 0, 'std': 0, 'cframe': 0}
    expected_by_type['zero'] =    {'psf': 0, 'frame': 0, 'ff': 0, 'sframe': 0, 'std': 0, 'cframe': 0}
    expected_by_type['dark'] = expected_by_type['zero']
    expected_by_type['sky']  = expected_by_type['science']
    expected_by_type['null'] = expected_by_type['zero']

    ## Determine the last filetype that is expected for each obstype
    terminal_steps = dict()
    for obstype, expected in expected_by_type.items():
        terminal_steps[obstype] = None
        keys = list(expected.keys())
        for key in reversed(keys):
            if expected[key] > 0:
                terminal_steps[obstype] = key
                break

    file_exptable = get_exposure_table_pathname(night)
    file_processing = get_processing_table_pathname(specprod=None, prodmod=night)
    # procpath,procname = os.path.split(file_processing)
    # file_unprocessed = os.path.join(procpath,procname.replace('processing','unprocessed'))

    specproddir = specprod_root()
    webpage = os.environ['DESI_DASHBOARD']
    logpath = os.path.join(specproddir, 'run', 'scripts', 'night', night)

    exptab_colnames = ['EXPID', 'FA_SURV', 'FAPRGRM', 'CAMWORD', 'BADCAMWORD',
                       'BADAMPS', 'EXPTIME', 'OBSTYPE', 'TILEID', 'COMMENTS',
                       'LASTSTEP']
    exptab_dtypes = [int, 'S20', 'S20', 'S40', float, 'S10', int, np.ndarray, 'S10']
    try: # Try reading tables first. Switch to counting files if failed.
        d_exp = load_table(file_exptable, tabletype='exptable')
        if 'LASTSTEP' in d_exp.colnames:
            d_exp = d_exp[exptab_colnames]
        else:
            d_exp = d_exp[exptab_colnames[:-1]]
            d_exp['LASTSTEP'] = 'all'
    except:
        print(f'WARNING: Error reading exptable for {night}. Changing check_on_disk to True and scanning files on disk.')
        d_exp = Table(names=exptab_colnames,dtype=exptab_dtypes)
        check_on_disk = True

    unaccounted_for_expids = []
    if check_on_disk:
        rawdatatemplate = os.path.join(rawdata_root(), night, '{zexpid}', 'desi-{zexpid}.fits.fz')
        rawdata_fileglob = rawdatatemplate.format(zexpid='*')
        known_exposures = set(list(d_exp['EXPID']))
        newexpids = list(find_new_exps(rawdata_fileglob, known_exposures))
        newexpids.sort(reverse=True)
        default_obstypes = default_obstypes_for_exptable()
        for expid in newexpids:
            zfild_expid = str(expid).zfill(8)
            filename = rawdatatemplate.format(zexpid=zfild_expid)
            h1 = fits.getheader(filename, 1)
            header_info = {keyword: 'unknown' for keyword in ['SPCGRPHS', 'EXPTIME',
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
                    header_info['CAMWORD'] = 'a' + str(header_info['SPCGRPHS']).replace(' ', '').replace(',', '')
                else:
                    header_info['CAMWORD'] = header_info['SPCGRPHS']
                header_info.pop('SPCGRPHS')
                d_exp.add_row(header_info)
                unaccounted_for_expids.append(expid)

    try:
        d_processing = load_table(file_processing, tabletype='proctable')
    except:
        d_processing = None
        print('WARNING: Error reading proctable. Only exposures in preproc'
              + ' directory will be marked as processing.')

    preproc_glob = os.path.join(os.environ['DESI_SPECTRO_REDUX'],
                                os.environ['SPECPROD'],
                                'preproc', str(night), '[0-9]*[0-9]')
    expid_processing = set([int(os.path.basename(fil)) for fil in glob.glob(preproc_glob)])

    if d_processing is not None and len(d_processing)>0:
        new_proc_expids = set(np.concatenate(d_processing['EXPID']).astype(int))
        expid_processing.update(new_proc_expids)

    logfiletemplate = os.path.join(logpath,'{pre}-{night}-{zexpid}-{specs}{jobid}.{ext}')
    fileglob_template = os.path.join(specproddir, 'exposures', str(night),
                                     '{zexpid}', '{ftype}-{cam}[0-9]-{zexpid}.{ext}')
    def count_num_files(ftype, expid):
        zfild_expid = str(expid).zfill(8)
        if ftype == 'stdstars':
            cam = ''
        else:
            cam = '[brz]'
        if ftype == 'fit-psf':
            ext = 'fits*'
        elif ftype == 'badcolumns':
            ext = 'csv'
        elif ftype == 'biasnight':
            ext = 'fits.gz'
        else:
            ext = 'fits'
        fileglob = fileglob_template.format(ftype=ftype, zexpid=zfild_expid,
                                            cam=cam, ext=ext)
        return len(glob.glob(fileglob))

    output = dict()
    d_exp.sort('EXPID')
    lasttile, first_exp_of_tile = None, None
    for row in d_exp:
        expid = int(row['EXPID'])
        ## For those already marked as GOOD or NULL in cached rows, take that and move on
        if night_info_pre is not None and str(expid) in night_info_pre and night_info_pre[str(expid)][0] in ['GOOD','NULL']:
            output[str(expid)] = night_info_pre[str(expid)]
            continue

        zfild_expid = str(expid).zfill(8)
        obstype = str(row['OBSTYPE']).lower().strip()
        tileid = str(row['TILEID'])
        if obstype == 'science':
            tileid_str = '<a href="'+'https://data.desi.lbl.gov/desi/target/fiberassign/tiles/trunk/' + \
                         tileid.zfill(6)[0:3]+'/fiberassign-'+tileid.zfill(6)+'.png'+'">'+tileid+'</a>'
            if lasttile != tileid:
                first_exp_of_tile = zfild_expid
                lasttile = tileid
        elif obstype == 'zero': # or obstype == 'other':
            continue
        else:
            tileid_str = '----'

        exptime = np.round(row['EXPTIME'],decimals=1)
        # if expid in proccamwords_by_expid.keys():
        #     proccamword = proccamwords_by_expid[expid]
        # else:
        #     proccamword = row['CAMWORD']
        proccamword = row['CAMWORD']
        if 'BADCAMWORD' in d_exp.colnames:
            proccamword = difference_camwords(proccamword,row['BADCAMWORD'])
        if obstype != 'science' and 'BADAMPS' in d_exp.colnames and row['BADAMPS'] != '':
            badcams = []
            for (camera, petal, amplifier) in parse_badamps(row['BADAMPS']):
                badcams.append(f'{camera}{petal}')
            badampcamword = create_camword(list(set(badcams)))
            proccamword = difference_camwords(proccamword, badampcamword)

        laststep = str(row['LASTSTEP'])
        ## temporary hack to remove annoying "aborted exposure" comments that happened on every exposure in SV3
        comments = list(row['COMMENTS'])
        bad_ind = None
        for ii,comment in enumerate(comments):
            if 'For EXPTIME: req=' in comment:
                bad_ind = ii
        if bad_ind is not None:
            comments.pop(bad_ind)
        comments = ', '.join(comments)

        if 'FA_SURV' in row.colnames and row['FA_SURV'] != 'unknown':
            fasurv = row['FA_SURV']
        else:
            fasurv = 'unkwn'
        if 'FAPRGRM' in row.colnames and row['FAPRGRM'] != 'unknown':
            faprog = row['FAPRGRM']
        else:
            faprog = 'unkwn'
        if obstype not in ['science', 'twilight']:
            if fasurv == 'unkwn':
                fasurv = '----'
            if faprog == 'unkwn':
                faprog = '----'

        if obstype in expected_by_type.keys():
            expected = expected_by_type[obstype].copy()
            terminal_step = terminal_steps[obstype]
        else:
            expected = expected_by_type['null'].copy()
            terminal_step = None

        if laststep == 'ignore':
            expected = expected_by_type['null'].copy()
            terminal_step = None
        elif laststep != 'all' and obstype == 'science':
            if laststep == 'skysub':
                expected['std'] = 0
                expected['cframe'] = 0
                terminal_step = 'sframe'
            elif laststep == 'fluxcal':
                pass
            else:
                print(f"WARNING: didn't understand science exposure expid={expid} of night {night}: laststep={laststep}")
        elif laststep != 'all' and obstype != 'science':
            print(f"WARNING: didn't understand non-science exposure expid={expid} of night {night}: laststep={laststep}")

        cameras = decode_camword(proccamword)
        nspecs = len(camword_to_spectros(proccamword, full_spectros_only=False))
        ncams = len(cameras)

        nfiles = dict()
        if obstype == 'arc':
            nfiles['psf'] = count_num_files(ftype='fit-psf', expid=expid)
        else:
            nfiles['psf'] = count_num_files(ftype='psf', expid=expid)
        nfiles['frame'] = count_num_files(ftype='frame', expid=expid)
        nfiles['ff'] = count_num_files(ftype='fiberflat', expid=expid)
        nfiles['sky'] = count_num_files(ftype='sky', expid=expid)
        nfiles['sframe'] = count_num_files(ftype='sframe', expid=expid)
        nfiles['std'] = count_num_files(ftype='stdstars', expid=expid)
        nfiles['cframe'] = count_num_files(ftype='cframe', expid=expid)

        if terminal_step == 'std':
            nexpected = nspecs
        else:
            nexpected = ncams

        if terminal_step is None:
            row_color = 'NULL'
        elif expected[terminal_step] == 0:
            row_color = 'NULL'
        elif nfiles[terminal_step] == 0:
            row_color = 'BAD'
        elif nfiles[terminal_step] < nexpected:
            row_color = 'INCOMPLETE'
        elif nfiles[terminal_step] == nexpected:
            row_color = 'GOOD'
        else:
            row_color = 'OVERFUL'

        if expid in expid_processing:
            status = 'processing'
        elif expid in unaccounted_for_expids:
            status = 'unaccounted'
        else:
            status = 'unprocessed'

        slurm_hlink, log_hlink = '----', '----'
        if row_color not in ['GOOD','NULL'] and obstype.lower() in ['arc','flat','science']:
            file_head = obstype.lower()
            lognames = glob.glob(logfiletemplate.format(pre=file_head, night=night,
                                       zexpid=zfild_expid, specs='*', jobid='', ext='log'))
            ## If no unified science script, identify which log to point to
            if obstype.lower() == 'science' and len(lognames)==0:
                ## First chronologically is the prestdstar
                lognames = glob.glob(logfiletemplate.format(pre='prestdstar',
                                                            night=night, zexpid=zfild_expid,
                                                            specs='*', jobid='', ext='log'))
                file_head = 'prestdstar'
                lognames_std = glob.glob(logfiletemplate.format(pre='stdstarfit',
                                           night=night, zexpid=first_exp_of_tile,
                                           specs='*', jobid='', ext='log'))
                ## If stdstar logs exist and we have all files for prestdstar
                ## link to stdstar
                if nfiles['sframe'] == ncams and len(lognames_std)>0:
                    lognames = lognames_std
                    file_head = 'stdstarfit'
                    lognames_post = glob.glob(logfiletemplate.format(pre='poststdstar',
                                               night=night, zexpid=zfild_expid,
                                               specs='*', jobid='', ext='log'))
                    ## If poststdstar logs exist and we have all files for stdstar
                    ## link to poststdstar
                    if nfiles['std'] == nspecs and len(lognames_post)>0:
                        lognames = lognames_post
                        file_head = 'poststdstar'

            newest_jobid = '00000000'
            spectrographs = ''

            for log in lognames:
                jobid = log[-12:-4]
                if int(jobid) > int(newest_jobid):
                    newest_jobid = jobid
                    spectrographs = log.split('-')[-2]
            if newest_jobid != '00000000' and len(spectrographs)!=0:
                logname = logfiletemplate.format(pre=file_head, night=night, zexpid=zfild_expid,
                                                 specs=spectrographs,  jobid='-'+newest_jobid, ext='log')
                slurmname = logfiletemplate.format(pre=file_head, night=night, zexpid=zfild_expid,
                                                   specs=spectrographs, jobid='', ext='slurm')

                slurm_hlink = _hyperlink( os.path.relpath(slurmname, webpage), 'Slurm')
                log_hlink   = _hyperlink( os.path.relpath(logname, webpage),   'Log'  )

        output[str(expid)] = [ row_color,
                               str(expid),
                               tileid_str,
                               obstype,
                               fasurv,
                               faprog,
                               laststep,
                               str(exptime),
                               proccamword,
                               _str_frac( nfiles['psf'],    ncams * expected['psf'] ),
                               _str_frac( nfiles['frame'],  ncams * expected['frame'] ),
                               _str_frac( nfiles['ff'],     ncams * expected['ff'] ),
                               _str_frac( nfiles['sframe'], ncams * expected['sframe'] ),
                               _str_frac( nfiles['sky'],    ncams * expected['sframe'] ),
                               _str_frac( nfiles['std'],    nspecs * expected['std'] ),
                               _str_frac( nfiles['cframe'], ncams * expected['cframe'] ),
                               slurm_hlink,
                               log_hlink,
                               comments,
                               status            ]
    return output


def _initialize_page(color_profile):
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
        font = cdict['font']
        background = ''#cdict['background'] # no background for a whole table after implementing color codes for processing columns
        html_page += 'table tr#'+str(ctype)+'  {background-color:'+str(background)+'; color:'+str(font)+';}\n'

    html_page += '</style>\n'
    html_page += '</head><body><h1>DESI '+os.environ["SPECPROD"]+' Processing Status Monitor</h1>\n'

    return html_page

def _closing_str():
    closing = """<div class="crt-wrapper"></div>
                 <div class="aadvantage-wrapper"></div>
                 </body></html>"""
    return closing

def _table_row(elements,idlabel=None):
    color_profile = return_color_profile()
    if elements[-1]!='processing':
        style_str='display:none;'
    else:
        style_str=''

    if idlabel is None:
        row_str = '<tr style="{}">'.format(style_str)
    else:
        row_str = '<tr style="'+style_str+'" id="'+str(idlabel)+'">'

    for elem in elements:
        chars = str(elem).split('/')
        if len(chars)==2: # m/n
            if chars[0]=='0' and chars[1]=='0':
                row_str += _table_element_style(elem,'background-color:'+color_profile['GOOD']['background']+';color:gray')
            elif chars[0]=='0' and chars[1]!='0':
                row_str += _table_element_style(elem,'background-color:'+color_profile['BAD']['background'])
            elif chars[0]!='0' and int(chars[0])<int(chars[1]):
                row_str += _table_element_style(elem,'background-color:'+color_profile['INCOMPLETE']['background'])
            elif chars[0]!='0' and int(chars[0])==int(chars[1]):
                row_str += _table_element_style(elem,'background-color:'+color_profile['GOOD']['background']) # Medium Aqua Green
            else:
                row_str += _table_element_style(elem, 'background-color:' + color_profile['OVERFUL']['background'])  # Medium Aqua Green

        else:
            row_str += _table_element(elem)
    row_str += '</tr>'#\n'
    return row_str

def _table_element(elem):
    return '<td>{}</td>'.format(elem)

def _table_element_style(elem,style):
    return '<td style="{}">{}</td>'.format(style,elem)

def _hyperlink(rel_path,displayname):
    hlink =  '<a href="{}" target="_blank" rel="noopener noreferrer">{}</a>'.format(rel_path,displayname)
    return hlink

def _str_frac(numerator,denominator):
    frac = '{}/{}'.format(numerator,denominator)
    return frac

def _js_path(output_dir):
    return os.path.join(output_dir,'js','open_nightly_table.js')

def js_import_str(output_dir):  # Not used
    output_path = _js_path(output_dir)
    if not os.path.exists(os.path.join(output_dir,'js')):
        os.makedirs(os.path.join(output_dir,'js'))
    if not os.path.exists(output_path):
        _write_js_script(output_path)
    return '<script type="text/javascript" src="{}"></script>'.format(output_path)

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



        
if __name__=="__main__":
    args = parse(options=None)
    main(args)
