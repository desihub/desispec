""" Module for generating QA HTML
"""
from __future__ import print_function, absolute_import, division

import os
import numpy as np
import glob

from desispec.io import meta, get_nights, get_exposures
from desispec.io.util import makepath

def header(title):
    """
    Parameters
    ----------
    title : str, optional

    Returns
    -------

    """
    head = '<?xml version="1.0" encoding="UTF-8"?>\n'
    head += '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">\n'


    head += '<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">\n'
    head += '\n'
    head += '<head>\n'
    head += '\n'
    head += '<meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />\n'
    head += '<title>{:s}</title>\n'.format(title)
    head += '<meta name="keywords" content="" />\n'
    head += '<meta name="description" content="" />\n'
    head += '<script type="text/javascript" src="jquery/jquery-1.4.2.min.js"></script>\n'
    head += '<script type="text/javascript" src="jquery/jquery.slidertron-0.1.js"></script>\n'
    head += '<link href="style.css" rel="stylesheet" type="text/css" media="screen" />\n'
    head += '\n'
    head += '</head>\n'

    # Begin the Body
    head += '<body>\n'
    head += '<h1>{:s}</h1>\n'.format(title)
    head += '<hr>\n'

    return head


def finish(f, body, links=None):
    """ Fill in the HTML file and end it
    Parameters
    ----------
    f : file
    body : str
    links : str, optional
    """
    # Write links
    if links is not None:
        f.write(links)
        f.write('</ul>\n')
        f.write('<hr>\n')
    # Write body
    f.write(body)
    # Finish
    end = '</body>\n'
    end += '</html>\n'
    f.write(end)

    return end


def init(f, title):
    head = header(title)
    f.write(head)
    # Init links
    links = '<h2>Quick Links</h2>\n'
    links += '<ul>\n'
    return links


def calib(qaprod_dir=None, specprod_dir=None):
    """ Generate HTML to orgainze calib HTML
    """
    # Organized HTML
    html_file = meta.findfile('qa_calib_html', qaprod_dir=qaprod_dir)
    html_path,_ = os.path.split(html_file)
    makepath(html_file)
    # Open
    f = open(html_file, 'w')
    init(f, 'Calibration QA')

    # Loop on Nights
    nights = get_nights(sub_folder='calibnight', specprod_dir=specprod_dir)
    nights.sort()
    links = ''
    body = ''
    for night in nights:
        all_png = glob.glob(html_path+'/'+night+'/qa*.png')
        if len(all_png) == 0:
            continue
        # Find expid
        expids = []
        for png in all_png:
            expids.append(int(png[-12:-4])) # A bit risky
        expids = np.unique(expids)
        expids.sort()
        f.write('<h2> Night -- {:s} </h2>\n'.format(night))
        f.write('<h3><ul>\n')
        for expid in expids:
            # Link
            f.write('<li><a href="{:s}/qa-{:08d}.html">Exposure {:08d}</a></li>\n'.format(night, expid, expid))
            # Generate Exposure html
            calib_exp(night, expid, qaprod_dir=qaprod_dir)
        f.write('</ul></h3>\n')

    # Finish
    finish(f,body)

    # Return
    return links, body


def calib_exp(night, expid, qaprod_dir=None):
    """ Geneate HTML for calib exposure PNGs
    Args:
        night:
        expid:

    Returns:

    """
    # File name
    html_file = meta.findfile('qa_calib_exp_html', night=night, expid=expid, qaprod_dir=qaprod_dir)
    html_path,_ = os.path.split(html_file)
    f = open(html_file, 'w')
    init(f, 'Calibration Exposure QA')

    # Loop on Nights
    for ctype in ['flat']:
        links = ''
        body = ''
        #
        all_png = glob.glob(html_path+'/qa-{:s}-*-{:08d}.png'.format(ctype,expid))
        all_png.sort()
        if len(all_png) == 0:
            continue
        # Type
        links +='<h2> {:s} Calib</h2>\n'.format(ctype)
        for png in all_png:
            _,png_file = os.path.split(png)
            # Image
            href="{:s}".format(png_file[:-4])
            links += '<li><a class="reference internal" href="#{:s}">{:s}</a></li>\n'.format(href, href)
            body += '<div class="section" id="{:s}">\n'.format(href)
            body += '<img class ="research" src="{:s}" width="100%" height="auto"/>\n'.format(png_file)
            #f.write('<li><a href="{:s}/qa-{:08d}.html">Exposure {:08d}</a></li>\n'.format(night, expid, expid))
    f.write('<ul>\n')
    f.write(links)
    f.write('</ul>\n')
    f.write(body)

    # Finish
    finish(f,'')

    # Return
    return links, body


def make_exposures(qaprod_dir=None):
    """ Generate HTML to organize exposure HTML

    Parameters
    ----------

    Returns
    -------
    links : str
    body : str

    """
    # Organized HTML
    html_file = meta.findfile('qa_exposures_html', qaprod_dir=qaprod_dir)
    html_path,_ = os.path.split(html_file)
    f = open(html_file, 'w')
    init(f, 'Exposures QA')

    # Loop on Nights
    nights = get_nights(specprod_dir=qaprod_dir)  # Scans for nights in QA
    nights.sort()
    links = ''
    body = ''
    for night in nights:
        # HTML
        f.write('<h2> Night -- {:s} </h2>\n'.format(night))
        f.write('<h3><ul>\n')
        # Loop on exposures
        for expid in get_exposures(night, specprod_dir=qaprod_dir):
            if not os.path.exists(html_path+'/'+night+'/{:08d}'.format(expid)):
                continue
            # Link
            f.write('<li><a href="{:s}/{:08d}/qa-{:08d}.html">Exposure {:08d}</a></li>\n'.format(night, expid, expid, expid))
            # Generate Exposure html
            make_exposure(night, expid, qaprod_dir=qaprod_dir)
        f.write('</ul></h3>\n')

    # Finish
    finish(f,body)

def make_exposure(night, expid, qaprod_dir=None):
    """ Generate HTML for exposure PNGs

    Parameters
    ----------
    setup : str
    cbset : str
    det : int

    Returns
    -------
    links : str
    body : str

    """
    # File name
    html_file = meta.findfile('qa_exposure_html', night=night, expid=expid, qaprod_dir=qaprod_dir)
    html_path,_ = os.path.split(html_file)
    f = open(html_file, 'w')
    init(f, 'Exposure QA')

    links = ''
    body = ''
    # Loop on Nights
    for ctype in ['sky', 'flux']:
        #
        all_png = glob.glob(html_path+'/qa-{:s}-*-{:08d}.png'.format(ctype,expid))
        all_png.sort()
        if len(all_png) == 0:
            continue
        # Type
        links += '<h2> {:s} Calib</h2>\n'.format(ctype)
        for png in all_png:
            _,png_file = os.path.split(png)
            # Image
            href="{:s}".format(png_file[:-4])
            links += '<li><a class="reference internal" href="#{:s}">{:s}</a></li>\n'.format(href, href)
            body += '<div class="section" id="{:s}">\n'.format(href)
            body += '<img class ="research" src="{:s}" width="100%" height="auto"/>\n'.format(png_file)
            #f.write('<li><a href="{:s}/qa-{:08d}.html">Exposure {:08d}</a></li>\n'.format(night, expid, expid))
    f.write('<ul>\n')
    f.write(links)
    f.write('</ul>\n')
    f.write(body)

    # Finish
    finish(f,'')

    # Return
    return links, body



def toplevel(qaprod_dir=None):
    """ Generate HTML to top level QA
    Mainly generates the highest level HTML file
    which has links to the Exposure and Calib QA.

    This also slurps any .png files in the top-level

    Parameters
    ----------
    setup : str
    cbset : str
    det : int

    Returns
    -------
    links : str
    body : str

    """
    # Organized HTML
    html_file = meta.findfile('qa_toplevel_html', qaprod_dir=qaprod_dir)
    html_path,_ = os.path.split(html_file)
    f = open(html_file, 'w')
    init(f, 'Top Level QA')

    # Calib?
    calib2d_file = meta.findfile('qa_calib_html', qaprod_dir=qaprod_dir)
    if os.path.exists(calib2d_file):
        # Truncate the path
        c2d_path, fname = os.path.split(calib2d_file)
        last_slash = c2d_path.rfind('/')
        f.write('<h2><a href="{:s}">Calibration QA</a></h2>\n'.format(c2d_path[last_slash+1:]+'/'+fname))
        # Full path
        #f.write('<h2><a href="{:s}">Calibration QA</a></h2>\n'.format(calib2d_file))
    # Exposures?
    exposures_file = meta.findfile('qa_exposures_html', qaprod_dir=qaprod_dir)
    if os.path.exists(exposures_file):
        # Truncated path
        exp_path, fname = os.path.split(exposures_file)
        last_slash = exp_path.rfind('/')
        f.write('<h2><a href="{:s}">Exposures QA</a></h2>\n'.format(exp_path[last_slash+1:]+'/'+fname))
        # Full path
        #f.write('<h2><a href="{:s}">Exposures QA</a></h2>\n'.format(exposures_file))

    # Existing PNGs
    f.write('<hr>\n')
    f.write('<h2>PNGs</h2>\n')
    all_png = glob.glob(html_path+'/*.png')
    all_png.sort()
    # Type
    links = ''
    body = ''
    for png in all_png:
        _, png_file = os.path.split(png)
        # Image
        href="{:s}".format(png_file[:-4])
        links += '<li><a class="reference internal" href="#{:s}">{:s}</a></li>\n'.format(href, href)
        body += '<div class="section" id="{:s}">\n'.format(href)
        body += '<img class ="research" src="{:s}" width="100%" height="auto"/>\n'.format(png_file)
    f.write('<h3><ul>\n')
    f.write(links)
    f.write('</ul></h3>\n')
    f.write(body)

    # Finish
    finish(f,'')

    # Return
    return
