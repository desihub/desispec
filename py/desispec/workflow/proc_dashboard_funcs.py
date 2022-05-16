import argparse
import os,glob
import re
import time,datetime
import numpy as np
from os import listdir
import json


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

# def parse(options):
#     """
#     Initialize the parser to read input
#     """
#     # Initialize
#     parser = argparse.ArgumentParser(description="Search the filesystem and summarize the existance of files output from "+
#                                      "the daily processing pipeline. Can specify specific nights, give a number of past nights,"+
#                                      " or use --all to get all past nights.")
#
#     # File I/O
#     parser.add_argument('--redux-dir', type=str, help="Product directory, point to $DESI_SPECTRO_REDUX by default ")
#     parser.add_argument('--output-dir', type=str, help="output portal directory for the html pages, which defaults to your home directory ")
#     parser.add_argument('--output-name', type=str, default='dashboard.html', help="name of the html page (to be placed in --output-dir).")
#     parser.add_argument('--specprod',type=str, help="overwrite the environment keyword for $SPECPROD")
#     parser.add_argument("-e", "--skip-expid-file", type=str, required=False,
#                         help="Relative pathname for file containing expid's to skip. "+\
#                              "Automatically. They are assumed to be in a column"+\
#                              "format, one per row. Stored internally as integers, so zero padding is "+\
#                              "accepted but not required.")
#     #parser.add_argument("--skip-null", type=str, required=False,
#     #                    help="Relative pathname for file containing expid's to skip. "+\
#     #                         "Automatically. They are assumed to be in a column"+\
#     #                         "format, one per row. Stored internally as integers, so zero padding is "+\
#     #                         "accepted but not required.")
#     # Specify Nights of Interest
#     parser.add_argument('-n','--nights', type=str, default = None, required = False,
#                         help="nights to monitor. Can be 'all', a comma separated list of YYYYMMDD, or a number "+
#                              "specifying the previous n nights to show (counting in reverse chronological order).")
#     parser.add_argument('--start-night', type=str, default = None, required = False,
#                         help="This specifies the first night to include in the dashboard. "+
#                              "Default is the earliest night available.")
#     parser.add_argument('--end-night', type=str, default = None, required = False,
#                         help="This specifies the last night (inclusive) to include in the dashboard. Default is today.")
#     parser.add_argument('--check-on-disk',  action="store_true",
#                         help="Check raw data directory for additional unaccounted for exposures on disk "+
#                              "beyond the exposure table.")
#     parser.add_argument('--ignore-json-archive',  action="store_true",
#                         help="Ignore the existing json archive of good exposure rows, regenerate all rows from "+
#                              "information on disk. As always, this will write out a new json archive," +
#                              " overwriting the existing one.")
#     # Read in command line and return
#     args = parser.parse_args(options)
#
#     return args


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
        font = cdict['font']
        background = ''#cdict['background'] # no background for a whole table after implementing color codes for processing columns
        html_page += 'table tr#'+str(ctype)+'  {background-color:'+str(background)+'; color:'+str(font)+';}\n'

    html_page += '</style>\n'
    html_page += '</head><body><h1>DESI '+os.environ["SPECPROD"]+f' {titlefill} Status Monitor</h1>\n'

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
