import argparse
import os,glob
import re
from astropy.io import fits,ascii
import time,datetime
import numpy as np
import psutil
from os import listdir
from collections import OrderedDict
from desispec.workflow.exptable import get_exposure_table_pathname
from desispec.workflow.proctable import get_processing_table_pathname, get_processing_table_name
from desispec.workflow.tableio import load_table

########################
### Helper Functions ###
########################

def return_color_profile():
    color_profile = {}
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


def find_newexp(night, fileglob, known_exposures):
    """
    Check the path given for new exposures
    """
    datafiles = sorted(glob.glob(fileglob))
    newexp = list()
    for filepath in datafiles:
        expid = int(os.path.basename(os.path.dirname(filepath)))
        if (night, expid) not in known_exposures:
            newexp.append((night, expid))

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
    parser.add_argument('-n','--nights', type=str, default = None, required = False, help="nights to monitor. Can be 'all', a "+
                                                                                          "comma separated list of YYYYMMDD, or"+
                                                                                          "a number specifying the previous n nights to show"+
                                                                                          " (counting in reverse chronological order).")
    parser.add_argument('--start-night', type=str, default = None, required = False, help="This specifies the first night" + \
                                                                                           " to include in the dashboard. Default is the earliest night available.")
    parser.add_argument('--end-night', type=str, default = None, required = False, help="This specifies the last night (inclusive)"+\
                                                                                           " to include in the dashboard. Default is today.")
    parser.add_argument('--include-short-scis',  action="store_true",  help="Include sci exps < 60s rather than treating as  null/zeros")
    
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
    if 'DESI_SPECTRO_REDUX' not in os.environ.keys(): # these are not set by default in cronjob mode.
        os.environ['DESI_SPECTRO_REDUX']='/global/cfs/cdirs/desi/spectro/redux/'
        os.environ['DESI_SPECTRO_DATA']='/global/cfs/cdirs/desi/spectro/data/'

    if 'SPECPROD' not in os.environ.keys() and args.specprod is None:
        os.environ['SPECPROD']='daily'
    elif args.specprod is None:
        args.specprod = os.getenv("SPECPROD")
    else:
        os.environ['SPECPROD'] = args.specprod

    if args.redux_dir is None:
        args.redux_dir = os.getenv('DESI_SPECTRO_REDUX')
    else:
        os.environ['DESI_SPECTRO_REDUX'] = args.redux_dir


    if 'DESI_DASHBOARD' not in os.environ.keys() and args.output_dir is None:
        os.environ['DESI_DASHBOARD']=os.getenv("HOME")
        args.output_dir = os.getenv("DESI_DASHBOARD")
    elif args.output_dir is None:
        args.output_dir = os.getenv("DESI_DASHBOARD")
    else:
        os.environ['DESI_DASHBOARD'] = args.output_dir
        
    args.prod_dir = os.path.join(args.redux_dir,args.specprod)
    
    ############
    ## Input ###
    ############
    if args.skip_expid_file is not None:
        skipd_expids = set(get_skipped_expids(args.skip_expid_file, skip_expids=True))
    else:
        skipd_expids = set()
        
    if args.nights is None or args.nights=='all' or ',' not in args.nights:
        nights = list()
        for n in listdir(os.getenv('DESI_SPECTRO_DATA')):
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
            
    nights_dict = OrderedDict()
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
    strTable=_initialize_page(color_profile)

    timestamp=time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())
    running='No'
    if check_running(proc_name='desi_dailyproc',suppress_outputs=True):
        running='Yes'
        strTable=strTable+"<div style='color:#00FF00'>{} {} running: {}</div>\n".format(timestamp,'desi_dailyproc',running)
    for ctype,cdict in color_profile.items():
        background = cdict['background']
        strTable += "\t<div style='color:{}'>{}</div>".format(background,ctype)
        
    strTable += '\n\n'
    strTable +="""Filter By Status:
    <select id="statuslist" onchange="filterByStatus()" class='form-control'>
    <option>processing</option>
    <option>unprocessed</option>
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
        webpage = os.path.join(os.getenv('DESI_DASHBOARD'), 'links', month)
        if not os.path.exists(webpage):
            os.makedirs(webpage)
        cmd = "fix_permissions.sh -a {}".format(webpage)
        os.system(cmd)
        nightly_tables = []
        for night in nights_in_month:
            ####################################
            ### Table for individual night ####
            ####################################
            nightly_tables.append(nightly_table(night,skipd_expids,show_null=args.show_null,use_short_sci=args.include_short_scis))
        strTable += monthly_table(nightly_tables,month)

    #strTable += js_import_str(os.getenv('DESI_DASHBOARD'))
    strTable += js_str()
    strTable += _closing_str()
    with open(os.path.join(os.getenv('DESI_DASHBOARD'),args.output_name),'w') as hs:
        hs.write(strTable)
    ##########################
    #### Fix Permission ######
    ##########################
    cmd="fix_permissions.sh -a {}".format(os.getenv('DESI_DASHBOARD'))
    os.system(cmd)


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

def nightly_table(night,skipd_expids=set(),show_null=True,use_short_sci=False):
    """
    Add a collapsible and extendable table to the html file for one specific night
    Input
    night: like 20200131
    output: The string to be added to the html file
    """
    night_info = calculate_one_night_use_file(night,use_short_sci)

    ngood,ninter,nbad,nnull,nover,n_notnull = 0,0,0,0,0,0
    main_body = ""
    for expid,row_info in night_info.items():
        if int(expid) in skipd_expids:
            continue

        table_row = _table_row(row_info[1:],idlabel=row_info[0])

        if not show_null and 'NULL' in table_row:
            continue
        
        main_body += table_row
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
        

    heading="Night {night}   Complete: {ngood}/{nnotnull}    Some: {ninter}/{nnotnull}    Bad: {nbad}/{nnotnull}".format(
                                                                                                         night=night,\
                                                                                                         ngood=ngood,\
                                                                                                         nnotnull=n_notnull,\
                                                                                                         ninter=ninter,\
                                                                                                         nbad=nbad)        
    nightly_table_str= '<!--Begin {}-->\n'.format(night)
    nightly_table_str += '<button class="collapsible">'+heading+'</button><div class="content" style="display:inline-block;min-height:0%;">\n'
    # Add table
    nightly_table_str += "<table id='c' class='nightTable'><tbody><tr><th>Expid</th><th>FLAVOR</th><th>OBSTYPE</th><th>EXPTIME</th><th>SPECTROGRAPHS</th><th>TILEID</th>"
    nightly_table_str += "<th>PSF File</th><th>frame file</th><th>FFlat file</th><th>sframe file</th><th>sky file</th>"
    nightly_table_str += "<th>std star</th><th>cframe file</th><th>slurm file</th><th>log file</th><th>status</th></tr>"

    nightly_table_str += main_body
    nightly_table_str += "</tbody></table></div>\n"
    nightly_table_str += '<!--End {}-->\n\n'.format(night)
    return nightly_table_str


def calculate_one_night_use_file(night, use_short_sci=False):
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
    cams_per_spgrph = 3

    totals_by_type = {}
    totals_by_type['ARC'] =    {'psf': cams_per_spgrph, 'ff': 0,               'frame': 0,               'sframe': 0}
    totals_by_type['FLAT'] =   {'psf': cams_per_spgrph, 'ff': cams_per_spgrph, 'frame': cams_per_spgrph, 'sframe': 0}
    totals_by_type['SKY'] =    {'psf': cams_per_spgrph, 'ff': 0,               'frame': cams_per_spgrph, 'sframe': cams_per_spgrph}
    totals_by_type['TWILIGHT']={'psf': cams_per_spgrph, 'ff': 0,               'frame': cams_per_spgrph, 'sframe': 0}
    totals_by_type['ZERO'] =   {'psf': 0,               'ff': 0,               'frame': 0,               'sframe': 0}
    totals_by_type['SCIENCE'], totals_by_type['NONE'] = totals_by_type['SKY'], totals_by_type['SKY']

    table_dir =  os.path.join(os.getenv('DESI_SPECTRO_REDUX'), os.getenv('SPECPROD'),'processing_tables')
    file_processing = '{}/{}{}-{}.csv'.format(table_dir,'processing_table_',os.getenv('SPECPROD'),night)
    #file_unprocessed = '{}/{}{}-{}.csv'.format(table_dir,'unprocessed_table_',os.getenv('SPECPROD'),night)
    file_exptable=get_exposure_table_pathname(night)
    try: # Try reading tables first. Switch to counting files if not failed.
        d_exp =  ascii.read(file_exptable, data_start=2, delimiter=',')
        d_processing = load_table(file_processing) # commented out temporarily, might used later
        #d_unprocessed = load_table(file_unprocessed)
    except:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Warning ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Error reading exptable or processing table, using brutal force method of scaning files.')
        print('All exposures will be marked as unprocessed.')
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        return calculate_one_night(night, use_short_sci=use_short_sci)
    expid_processing=[]

    for expid_list in d_processing['EXPID']:
        expid_processing += expid_list.tolist()
    expid_processing = set(expid_processing)
    expids = list(d_exp['EXPID'])
    expids.sort(reverse=True)

    fileglob = os.path.join(os.getenv('DESI_SPECTRO_REDUX'), os.getenv('SPECPROD'), 'exposures', str(night), '{}', '{}')

    logpath = os.path.join(os.getenv('DESI_SPECTRO_REDUX'), os.getenv("SPECPROD"), 'run', 'scripts', 'night', night)
    #if os.path.exists(logpath):
    #    cmd="fix_permissions.sh -a {}".format(logpath)
    #    os.system(cmd)

    webpage = os.getenv('DESI_DASHBOARD')
    logfileglob = os.path.join(logpath,'{}-{}-{}-*.{}') # search for the log file 
    logfiletemplate = os.path.join(logpath,'{}-{}-{}-{}{}.{}')

    output = OrderedDict()
    d_exp.add_index('EXPID')

    for expid in expids:
        zfild_expid = str(expid).zfill(8)
        """
        filename = os.path.join(os.getenv('DESI_SPECTRO_DATA'), str(night), zfild_expid,
                                'desi-' + str(expid).zfill(8) + '.fits.fz')
        h1 = fits.getheader(filename, 1)

        header_info = {keyword: 'Unknown' for keyword in ['FLAVOR', 'SPCGRPHS', 'EXPTIME', 'OBSTYPE']}
        for keyword in header_info.keys():
            if keyword in h1.keys():
                header_info[keyword] = h1[keyword]
        """
        flavor='NULL'
        obstype = str(d_exp.loc[expid]['OBSTYPE']).upper().strip()
        exptime = d_exp.loc[expid]['EXPTIME']
        spcgrphs = str(d_exp.loc[expid]['CAMWORD'])[1:] #spcgrphs = str(d_exp.loc[expid]['SPECTROGRAPHS'])
        tileid = str(d_exp.loc[expid]['TILEID'])
        if obstype == 'SCIENCE':
            tileid_str = '<a href="'+'https://data.desi.lbl.gov/desi/target/fiberassign/tiles/trunk/'+tileid.zfill(6)[0:3]+'/fiberassign-'+tileid.zfill(6)+'.png'+'">'+tileid+'</a>'
        elif obstype == 'OTHER' or obstype == 'ZERO':
            continue
        else:
            tileid_str =  tileid
        file_psf = glob.glob(fileglob.format(zfild_expid, 'psf-[brz]?-????????.fits'))
        file_fit_psf = glob.glob(fileglob.format(zfild_expid, 'fit-psf-[brz]?-????????.fits'))
        file_fiberflat = glob.glob(fileglob.format(zfild_expid, 'fiberflat-[brz]?-????????.fits'))
        file_frame = glob.glob(fileglob.format(zfild_expid, 'frame-??-????????.fits'))
        file_sframe = glob.glob(fileglob.format(zfild_expid, 'sframe-??-????????.fits'))
        file_cframe = glob.glob(fileglob.format(zfild_expid, 'cframe-??-????????.fits'))
        file_sky = glob.glob(fileglob.format(zfild_expid, 'sky*.fits'))
        file_stdstar = glob.glob(fileglob.format(zfild_expid, 'stdstars-?-????????.fits'))
        if obstype in totals_by_type.keys():
            n_tots = totals_by_type[obstype]
        else:
            n_tots = totals_by_type['NONE']

        n_spgrph = int(len(list(spcgrphs)))
        row_color = "NULL"
        npsfs = len(file_psf) + len(file_fit_psf)
        nframes = len(file_frame)
        ncframes = len(file_cframe)
        nfiberflat = len(file_fiberflat)
        nsframe = len(file_sframe)
        nsky = len(file_sky)
        nstdstar = len(file_stdstar)
        
        if obstype.lower() == 'arc':
            nfiles = npsfs
            n_tot_spgrphs = n_spgrph * n_tots['psf']
            nstdstar_expected = 0
        elif obstype.lower() == 'flat':
            nfiles = nframes
            n_tot_spgrphs = n_spgrph * n_tots['frame']
            nstdstar_expected = 0
        elif obstype.lower() == 'science':
            nfiles = ncframes
            n_tot_spgrphs = n_spgrph * n_tots['sframe']
            nstdstar_expected = n_spgrph
        else:
            nfiles = 0
            n_tot_spgrphs = 0
            nstdstar_expected = 0

        if n_tots['psf'] == 0:
            row_color = 'NULL'
        elif not use_short_sci and obstype.upper() == 'SCIENCE' and float(exptime)< 60.:
            row_color = 'NULL'
        elif nfiles == 0:
            row_color = 'BAD'
        elif nfiles < n_tot_spgrphs:
            row_color = 'INCOMPLETE'
        elif nfiles == n_tot_spgrphs:
            row_color = 'GOOD'
        else:
            row_color = 'OVERFUL'

        hlink1 = '----'
        hlink2 = '----'
        if expid in expid_processing:
            status = 'processing'
        else:
            status = 'unprocessed'

        if row_color not in ['GOOD','NULL'] and obstype.lower() in ['arc','flat','science']:
            file_head = obstype.lower()
            lognames = glob.glob(logfileglob.format(file_head, night,zfild_expid,'log'))
            if obstype.lower() == 'science': 
                lognames_pre = glob.glob(logfileglob.format('prestdstar',night,zfild_expid,'log'))
                lognames_post = glob.glob(logfileglob.format('poststdstar',night,zfild_expid,'log'))
                if len(lognames_post)>0:
                    lognames = lognames_post
                    file_head = 'poststdstar'
                else:
                    lognames = lognames_pre
                    file_head = 'prestdstar'
            newest_jobid = '00000000'
            spectrographs = ''

            for log in lognames:
                jobid = log[-12:-4]
                if int(jobid) > int(newest_jobid):
                    newest_jobid = jobid
                    spectrographs = log.split('-')[-2]
            if newest_jobid != '00000000' and len(spectrographs)!=0:
                logname = logfiletemplate.format(file_head, night,zfild_expid,spectrographs,'-'+newest_jobid,'log')
                logname_only = logname.split('/')[-1]

                slurmname = logfiletemplate.format(file_head, night,zfild_expid,spectrographs,'','slurm')
                slurmname_only = slurmname.split('/')[-1]
                relpath_log = os.path.join('links',night[:-2],logname_only)
                relpath_slurm = os.path.join('links',night[:-2],slurmname_only)

                if not os.path.exists(os.path.join(webpage,relpath_log)):
                    cmd = "ln -s {} {}".format(logname,os.path.join(webpage,relpath_log))
                    os.system(cmd)
                if not os.path.exists(os.path.join(webpage, relpath_slurm)):
                    cmd = "ln -s {} {}".format(slurmname, os.path.join(webpage, relpath_slurm))
                    os.system(cmd)

                hlink1 = _hyperlink(relpath_slurm, 'Slurm')
                hlink2 = _hyperlink(relpath_log, 'Log')

        output[str(expid)] = [row_color, \
                              expid, \
                              flavor,\
                              obstype,\
                              exptime, \
                              'SP: '+spcgrphs.replace('SP',''), \
                              tileid_str, \
                              _str_frac( npsfs,               n_spgrph * n_tots['psf']), \
                              _str_frac( nframes,             n_spgrph * n_tots['frame']), \
                              _str_frac( nfiberflat, n_spgrph * n_tots['ff']), \
                              _str_frac( nsframe,    n_spgrph * n_tots['sframe']), \
                              _str_frac( nsky,       n_spgrph * n_tots['sframe']), \
                              _str_frac( nstdstar,   nstdstar_expected), \
                              _str_frac( ncframes,            n_spgrph * n_tots['sframe']), \
                              hlink1, \
                              hlink2, status     ]
    return output


def calculate_one_night(night, use_short_sci=False):
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
    cams_per_spgrph = 3

    totals_by_type = {}
    totals_by_type['ARC'] =    {'psf': cams_per_spgrph, 'ff': 0,               'frame': 0,               'sframe': 0}
    totals_by_type['FLAT'] =   {'psf': cams_per_spgrph, 'ff': cams_per_spgrph, 'frame': cams_per_spgrph, 'sframe': 0}
    totals_by_type['SKY'] =    {'psf': cams_per_spgrph, 'ff': 0,               'frame': cams_per_spgrph, 'sframe': cams_per_spgrph}
    totals_by_type['TWILIGHT']={'psf': cams_per_spgrph, 'ff': 0,               'frame': cams_per_spgrph, 'sframe': 0}
    totals_by_type['ZERO'] =   {'psf': 0,               'ff': 0,               'frame': 0,               'sframe': 0}
    totals_by_type['SCIENCE'], totals_by_type['NONE'] = totals_by_type['SKY'], totals_by_type['SKY']

    rawdata_fileglob = '{}/{}/*/desi-*.fits.fz'.format(os.getenv('DESI_SPECTRO_DATA'), night)
    known_exposures = set()
    newexp = find_newexp(night, rawdata_fileglob, known_exposures)
    expids = [t[1] for t in newexp]
    expids.sort(reverse=True)

    fileglob = os.path.join(os.getenv('DESI_SPECTRO_REDUX'), os.getenv('SPECPROD'), 'exposures', str(night), '{}', '{}')

    logpath = os.path.join(os.getenv('DESI_SPECTRO_REDUX'), os.getenv("SPECPROD"), 'run', 'scripts', 'night', night)
    if os.path.exists(logpath):
        cmd="fix_permissions.sh -a {}".format(logpath)
        os.system(cmd)

    webpage = os.getenv('DESI_DASHBOARD')
    logfileglob = os.path.join(logpath,'{}-{}-{}-*.{}')
    logfiletemplate = os.path.join(logpath,'{}-{}-{}-{}{}.{}')

    output = OrderedDict()
    for expid in expids:
        zfild_expid = str(expid).zfill(8)
        # Check the redux folder for reduced files
        filename = os.path.join(os.getenv('DESI_SPECTRO_DATA'), str(night), zfild_expid,
                                'desi-' + str(expid).zfill(8) + '.fits.fz')
        h1 = fits.getheader(filename, 1)


        header_info = {keyword: 'Unknown' for keyword in ['FLAVOR', 'SPCGRPHS', 'EXPTIME', 'OBSTYPE','TILEID']}
        for keyword in header_info.keys():
            if keyword in h1.keys():
                header_info[keyword] = h1[keyword]

        file_psf = glob.glob(fileglob.format(zfild_expid, 'psf-[brz]?-????????.fits'))
        file_fit_psf = glob.glob(fileglob.format(zfild_expid, 'fit-psf-[brz]?-????????.fits'))
        file_fiberflat = glob.glob(fileglob.format(zfild_expid, 'fiberflat-[brz]?-????????.fits'))
        file_frame = glob.glob(fileglob.format(zfild_expid, 'frame-??-????????.fits'))
        file_sframe = glob.glob(fileglob.format(zfild_expid, 'sframe-??-????????.fits'))
        file_cframe = glob.glob(fileglob.format(zfild_expid, 'cframe-??-????????.fits'))
        file_sky = glob.glob(fileglob.format(zfild_expid, 'sky*.fits'))
        file_stdstar = glob.glob(fileglob.format(zfild_expid, 'stdstars-?-????????.fits'))

        obstype = str(header_info['OBSTYPE']).upper().strip()
        if obstype in totals_by_type.keys():
            n_tots = totals_by_type[obstype]
        else:
            n_tots = totals_by_type['NONE']

        n_spgrph = int(len(header_info['SPCGRPHS'].split(',')))
        tileid = str(header_info['TILEID'])
        if obstype == 'SCIENCE':
            tileid_str = '<a href="'+'https://data.desi.lbl.gov/desi/target/fiberassign/tiles/trunk/'+tileid.zfill(6)[0:3]+'/fiberassign-'+tileid.zfill(6)+'.png'+'">'+tileid+'</a>'
        elif obstype == 'OTHER' or obstype == 'ZERO':
            continue
        else:
            tileid_str =  tileid

        row_color = "NULL"
        npsfs = len(file_psf) + len(file_fit_psf)        
        nframes = len(file_frame)
        ncframes = len(file_cframe)
        nstdstar = len(file_stdstar)
        if obstype.lower() == 'arc':
            nfiles = npsfs
            n_tot_spgrphs = n_spgrph * n_tots['psf']
            nstdstar_expected = 0
        elif obstype.lower() == 'flat':
            nfiles = nframes
            n_tot_spgrphs = n_spgrph * n_tots['frame']
            nstdstar_expected = 0
        elif obstype.lower() == 'science':
            nfiles = ncframes
            n_tot_spgrphs = n_spgrph * n_tots['sframe']
            nstdstar_expected = n_spgrph
        else:
            nfiles = 0
            n_tot_spgrphs = 0
            nstdstar_expected = 0
            
        if n_tots['psf'] == 0:
            row_color = 'NULL'
        elif not use_short_sci and obstype.upper() == 'SCIENCE' and float(header_info['EXPTIME'])< 60.:
            row_color = 'NULL'
        elif nfiles == 0:
            row_color = 'BAD'
        elif nfiles < n_tot_spgrphs:
            row_color = 'INCOMPLETE'
        elif nfiles == n_tot_spgrphs:
            row_color = 'GOOD'
        else:
            row_color = 'OVERFUL'

        hlink1 = '----'
        hlink2 = '----'
        if row_color not in ['GOOD','NULL'] and obstype.lower() in ['arc','flat','science']:
            file_head = obstype.lower()
            lognames = glob.glob(logfileglob.format(file_head, night,zfild_expid,'log'))
            if obstype.lower() == 'science':
                lognames_pre = glob.glob(logfileglob.format('prestdstar',night,zfild_expid,'log'))
                lognames_post = glob.glob(logfileglob.format('poststdstar',night,zfild_expid,'log'))
                if len(lognames_post)>0:
                    lognames = lognames_post
                    file_head = 'poststdstar'
                else:
                    lognames = lognames_pre
                    file_head = 'prestdstar'
            newest_jobid = '00000000'
            spectrographs = ''

            for log in lognames:
                jobid = log[-12:-4]
                if int(jobid) > int(newest_jobid):
                    newest_jobid = jobid
                    spectrographs = log.split('-')[-2]
            if newest_jobid != '00000000' and len(spectrographs)!=0:
                logname = logfiletemplate.format(file_head, night,zfild_expid,spectrographs,'-'+newest_jobid,'log')
                logname_only = logname.split('/')[-1]

                slurmname = logfiletemplate.format(file_head, night,zfild_expid,spectrographs,'','slurm')
                slurmname_only = slurmname.split('/')[-1]
                relpath_log = os.path.join('links',night[:-2],logname_only)
                relpath_slurm = os.path.join('links',night[:-2],slurmname_only)
                
                if not os.path.exists(os.path.join(webpage,relpath_log)):
                    cmd = "ln -s {} {}".format(logname,os.path.join(webpage,relpath_log))
                    os.system(cmd)
                if not os.path.exists(os.path.join(webpage, relpath_slurm)):
                    cmd = "ln -s {} {}".format(slurmname, os.path.join(webpage, relpath_slurm))
                    os.system(cmd)

                hlink1 = _hyperlink(relpath_slurm, 'Slurm')
                hlink2 = _hyperlink(relpath_log, 'Log')
        status = 'unprocessed'
        output[str(expid)] = [row_color, \
                              expid, \
                              header_info['FLAVOR'],\
                              obstype,\
                              header_info['EXPTIME'], \
                              'SP: '+header_info['SPCGRPHS'].replace('SP',''), \
                              tileid_str, \
                              _str_frac( npsfs,               n_spgrph * n_tots['psf']), \
                              _str_frac( nframes,             n_spgrph * n_tots['frame']), \
                              _str_frac( len(file_fiberflat), n_spgrph * n_tots['ff']), \
                              _str_frac( len(file_sframe),    n_spgrph * n_tots['sframe']), \
                              _str_frac( len(file_sky),       n_spgrph * n_tots['sframe']), \
                              _str_frac( nstdstar,            nstdstar_expected), \
                              _str_frac( ncframes,            n_spgrph * n_tots['sframe']), \
                              hlink1, \
                              hlink2, status         ]
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
    html_page += '</head><body><h1>DESI '+os.getenv("SPECPROD").upper()+' Processing Status Monitor</h1>\n'

    return html_page

def _closing_str():
    closing = """<div class="crt-wrapper"></div>
                 <div class="aadvantage-wrapper"></div>
                 </body></html>"""
    return closing

def _table_row(elements,idlabel=None):
    color_profile = return_color_profile()
    if elements[15]=='unprocessed':
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
