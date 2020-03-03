import argparse
import os,glob
import re
from astropy.io import fits
import time,datetime
import numpy as np
import psutil
from os import listdir
from collections import OrderedDict

#HEX #005AB5 R 0  G 90  B 181
#HEX #DC3220 R 220  G 50  B

#import desispec.io as desi_io
########################
### Helper Functions ###
########################
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


def get_catchup_nights(catchup_filename, docatchup=True):
    if docatchup and catchup_filename is not None and os.path.exists(catchup_filename):
        catchup = np.atleast_1d(np.loadtxt(catchup_filename, dtype=int)).tolist()
    else:
        catchup = []

    return catchup

def check_running(proc_name= 'desi_dailyproc'):
    """
    Check if the desi_dailyproc process is running
    """
    running = False
    mypid = os.getpid()
    for p in psutil.process_iter():
        if p.pid != mypid and proc_name in ' '.join(p.cmdline()):
            print('ERROR: {}} already running as PID {}:'.format(proc_name,p.pid))
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
    parser.add_argument('--prod-dir', type=str, help="Product directory, point to $DESI_SPECTRO_REDUX/$SPECPROD by default ")
    parser.add_argument('--output-dir', type=str, help="output portal directory for the html pages ")
    parser.add_argument('--output-name', type=str, help="name of the html page (to be placed in --output-dir, which defaults to your home directory).")

    # Specify Nights of Interest
    parser.add_argument('-n','--nights', type=str, default = None, required = False, help="nights to monitor. Can be a "+
                                                                                          "comma separated list of YYYYMMDD or"+
                                                                                          "a number specifying the previous n nights to show"+
                                                                                          " (counting in reverse chronological order).")
    parser.add_argument('--all', action="store_true",
                        help="run all nights. If set, ignores --nights (i.e. this supersedes it).")

    # Data Pruning
    parser.add_argument("--cameras", type=str, required=False,
                        help="Explicitly define the spectrographs for which you want"+
                             " summary statistics. Should be a comma separated list."+
                             " Just a number assumes you want to reduce R, B, and Z "+
                             "for that camera. Otherwise specify separately [BRZ|brz][0-9].")

    # Additional Options / Code Flags
    parser.add_argument("--ignore-instances", action="store_true",
                        help="Allow script to run even if another instance is "+
                             "running. Use with care.")
    parser.add_argument("--ignore-cori-node", action="store_true",
                        help="Allow script to run on nodes other than cori21")

    # Read in command line and return
    args = None
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)

    if args.catchup_file is not None:
        args.catchup = True

    return args



######################
### Main Functions ###
######################
def main(args):
    """ Code to generate a webpage for monitoring of desi_dailyproc production status
    Usage:
    -n can be 'all' or series of nights separated by comma or blank like 20200101,20200102 or 20200101 20200102
    Normal Mode:
    desi_proc_dashboard -n all --n_nights 3  --output_dir /global/cfs/cdirs/desi/www/users/zhangkai/desi_proc_dashboard/
    desi_proc_dashboard -n 20200101,20200102 --n_nights 3  --output_dir /global/cfs/cdirs/desi/www/users/zhangkai/desi_proc_dashboard/
    desi_proc_dashboard -n 20200101 20200102 --n_nights 3  --output_dir /global/cfs/cdirs/desi/www/users/zhangkai/desi_proc_dashboard/
    Cron job script:
        */30 * * * * /global/common/software/desi/cori/desiconda/20190804-1.3.0-spec/conda/bin/python3 \
            /global/cfs/cdirs/desi/users/zhangkai/desi/code/desispec/py/desispec/desi_proc_dashboard.py -n all \
            --n_nights 30 --output_dir /global/cfs/cdirs/desi/www/users/zhangkai/desi_proc_dashboard/ \
            >/global/cfs/cdirs/desi/users/zhangkai/desi_proc_dashboard.log \
            2>/global/cfs/cdirs/desi/users/zhangkai/desi_proc_dashboard.err & \
            output_url https://portal.nersc.gov/project/desi/users/zhangkai/desi_proc_dashboard/
    """
    if 'DESI_SPECTRO_REDUX' not in os.environ.keys(): # these are not set by default in cronjob mode.
        os.environ['DESI_SPECTRO_REDUX']='/global/cfs/cdirs/desi/spectro/redux/'
        os.environ['DESI_SPECTRO_DATA']='/global/cfs/cdirs/desi/spectro/data/'
        os.environ['SPECPROD']='daily'
    if args.prod_dir is None:
        args.prod_dir = os.path.join(os.environ['DESI_SPECTRO_REDUX'], os.environ['SPECPROD'])
    ############
    ## Input ###
    ############

    if args.nights[0]=='all':
        nights = list()
        for n in listdir(os.getenv('DESI_SPECTRO_DATA')):
            #- nights are 20YYMMDD
            if re.match('^20\d{6}$', n):
                nights.append(int(n))
    else:
        try:
            print(args.nights)
            if len(args.nights)==1: # list separted by , or a single night
                nights=[int(night) for night in args.nights[0].split(',')]
            else:
                nights=[int(night) for night in args.nights]
            print('Get nights',nights)
        except:
            nights=[]

    tonight=what_night_is_it()
    if not tonight in nights:
        nights.append(tonight)
    nights.sort(reverse=True)

    if int(args.n_nights)<=len(nights):
        nights=nights[0:int(args.n_nights)]


    nights_dict = OrderedDict()
    for night in nights:
        month = night[4:6]
        if month not in nights_dict.keys():
            nights_dict[month] = [night]
        else:
            nights_dict[month].append(night)


    print('Searching '+args.prod_dir+' for ',nights)

    ######################################################################
    ## sub directories. Should not change if generated by the same code ##
    ## that follows the same directory strucure ##
    ######################################################################

    strTable=_initialize_page()

    timestamp=time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())
    running='No'
    if check_running(proc_name='desi_dailyproc'):
        running='Yes'
    strTable=strTable+"<div style='color:#00FF00'>{} {} running: {}</div>".format(timestamp,'desi_dailyproc',running)

    for month, nights in nights_dict.items():
        nightly_tables = []
        for night in nights:
            ####################################
            ### Table for individual night ####
            ####################################
            nightly_tables.append(nightly_table(night))
        strTable += monthly_table(nightly_tables,month)

    strTable=strTable+js_import_str(args.output_dir)
    strTable=strTable+"</html>"
    with open(os.path.join(args.output_dir,"desi_proc_dashboard.html"),'w') as hs:
        hs.write(strTable)

    ##########################
    #### Fix Permission ######
    ##########################
    cmd="fix_permissions.sh -a {}".format(args.output_dir)
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

    heading="{} {}, ({})".format(month_dict[month[4:]],month[:4],month)
    month_table_str = '<!--Begin {}-->\n'.format(month)
    month_table_str += "<button class='collapsible'>"+heading+"</button><div class='content' style='display:inline-block;min-height:0%;'>\n"
    month_table_str += "<table id='c'>"
    for table_str in tables:
        month_table_str += table_str

    month_table_str += "</table></div>\n"
    month_table_str += '<!--End {}-->\n\n'.format(month)

    return month_table_str

def nightly_table(night):
    """
    Add a collapsible and extendable table to the html file for one specific night
    Input
    night: like 20200131
    output: The string to be added to the html file
    """
    heading="Night {}".format(night)
    nightly_table_str= '<!--Begin {}-->\n'.format(night)
    nightly_table_str += "<button class='collapsible'>"+heading+"</button><div class='content' style='display:inline-block;min-height:0%;'>\n"
    nightly_table_str += "<table id='c'><tr><th>Expid</th><th>FLAVOR</th><th>OBSTYPE</th><th>EXPTIME</th><th>SPECTROGRAGHS</th>"
    nightly_table_str += "<th>PSF File</th><th>FFlat file</th><th>frame file</th><th>sframe file</th><th>sky file</th>"
    nightly_table_str += "<th>cframe file</th><th>slurm file</th><th>log file</th></tr>"

    night_info = calculate_one_night(night)

    for expid,row_info in night_info.items():
        nightly_table_str += _table_row(row_info)

    nightly_table_str += "</table></div>\n"
    nightly_table_str += '<!--End {}-->\n\n'.format(night)
    return nightly_table_str


def calculate_one_night(night):
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
    output = OrderedDict()
    for expid in expids:
        zfild_expid = str(expid).zfill(8)
        # Check the redux folder for reduced files
        filename = os.path.join(os.getenv('DESI_SPECTRO_DATA'), str(night), zfild_expid,
                                'desi-' + str(expid).zfill(8) + '.fits.fz')
        h1 = fits.getheader(filename, 1)

        header_info = {keyword: 'Unknown' for keyword in ['FLAVOR', 'SPCGRPHS', 'EXPTIME', 'OBSTYPE']}
        for keyword in header_info.keys():
            if keyword in h1.keys():
                header_info[keyword] = h1[keyword].strip()

        file_psf = glob.glob(fileglob.format(zfild_expid, 'psf*.fits'))
        file_fit_psf = glob.glob(fileglob.format(zfild_expid, 'fit-psf*.fits'))
        file_fiberflat = glob.glob(fileglob.format(zfild_expid, 'fiberflat*.fits'))
        file_frame = glob.glob(fileglob.format(zfild_expid, 'frame*.fits'))
        file_sframe = glob.glob(fileglob.format(zfild_expid, 'sframe*.fits'))
        file_cframe = glob.glob(fileglob.format(zfild_expid, 'cframe*.fits'))
        file_sky = glob.glob(fileglob.format(zfild_expid, 'sky*.fits'))

        obstype = str(header_info['OBSTYPE']).upper().strip()
        if obstype in totals_by_type.keys():
            n_tots = totals_by_type[obstype]
        else:
            n_tots = totals_by_type['NONE']

        n_spgrph = int(len(header_info['SPCGRPHS'].split(',')))

        output[str(expid)] = [expid, \
                              header_info['FLAVOR'],\
                              header_info['OBSTYPE'],\
                              header_info['EXPTIME'], \
                              header_info['SPCGRPHS'], \
                              _str_frac( len(file_psf) + len(file_fit_psf), n_spgrph * n_tots['psf']), \
                              _str_frac( len(file_fiberflat),               n_spgrph * n_tots['ff']), \
                              _str_frac( len(file_frame),                   n_spgrph * n_tots['frame']), \
                              _str_frac( len(file_sframe),                  n_spgrph * n_tots['sframe']), \
                              _str_frac( len(file_sky),                     n_spgrph * n_tots['sframe']), \
                              _str_frac( len(file_cframe),                  n_spgrph * n_tots['sframe']), \
                              _hyperlink('./here.txt', 'Slurm'), \
                              _hyperlink('./there.txt', 'Log')         ]
    return output


def _initialize_page(color_profile):
    """
    Initialize the html file for showing the statistics, giving all the headers and CSS setups.
    """
    # strTable="<html><style> table {font-family: arial, sans-serif;border-collapse: collapse;width: 100%;}"
    # strTable=strTable+"td, th {border: 1px solid #dddddd;text-align: left;padding: 8px;}"
    # strTable=strTable+"tr:nth-child(even) {background-color: #dddddd;}</style>"
    html_page = """<html><style>
    h1 {font-family: 'sans-serif';font-size:50px;color:#4CAF50}
    #c {font-family: 'Trebuchet MS', Arial, Helvetica, sans-serif;border-collapse: collapse;width: 100%;}
    #c td, #c th {border: 1px solid #ddd;padding: 8px;}
    #c tr:nth-child(even){background-color: #f2f2f2;}
    #c tr:hover {background-color: #ddd;}
    #c th {padding-top: 12px;  padding-bottom: 12px;  text-align: left;  background-color: #4CAF50;  color: white;}
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
    </style>
    <h1>DESI Daily Processing Status Monitor</h1>"""

    return html_page

def _table_row(elements):
    row_str = '<tr>'
    for elem in elements:
        row_str += _table_element(elem)
    row_str += '</tr>\n'
    return row_str

def _table_element(elem):
    return '<td>{}</td>'.format(elem)

def _hyperlink(rel_path,displayname):
    hlink =  '<a href="{}">{}</a>'.format(rel_path,displayname)
    return hlink

def _str_frac(numerator,denominator):
    frac = '{}/{}'.format(numerator,denominator)
    return frac

def _js_path(output_dir):
    return os.path.join(output_dir,'js','open_nightly_table.js')

def js_import_str(output_dir):
    output_path = _js_path(output_dir)
    if not os.path.exists(os.path.join(output_dir,'js')):
        os.makedirs(os.path.join(output_dir,'js'))
    if not os.path.exists(output_path):
        _write_js_script(output_path)
    return '<script type = "text/javascript" src="{}"></script>'.format(output_path)

def _write_js_script(output_path):
    """
    Return the javascript script to be added to the html file
    """
    s="""<script>
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
        </script>"""
    with open(_js_path(output_path),'w') as outjs:
        outjs.write(s)






        
if __name__=="__main__":
    args = parse(options=None)
    main(args)
