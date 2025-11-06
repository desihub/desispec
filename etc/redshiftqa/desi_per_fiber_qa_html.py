#!/usr/bin/env python

# Create the redshift QA HTML files
# Usage:
# ./desi_per_fiber_qa_html.py --stats per_fiber_qa_stats.fits -o /global/cfs/cdirs/desicollab/users/rongpu/redshift_qa/new/loa

from __future__ import division, print_function
import sys, os, glob, time, warnings, gc
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table, vstack, hstack
import fitsio
from astropy.io import fits

import argparse

pvalue_threshold = 1e-4

parser = argparse.ArgumentParser()
parser.add_argument('--stats', type=str, help="per-fiber stats path", required=True)
parser.add_argument('-o', '--output', type=str, help="output directory path", required=True)
args = parser.parse_args()

stats_fn = args.stats
output_dir = args.output
html_dir = os.path.join(output_dir, 'html')

stats = Table(fitsio.read(stats_fn))

if not os.path.isdir(html_dir):
    os.makedirs(html_dir)

f = open(os.path.join(output_dir, 'fiber_directory.html'), "w")
f.write('<html>\n')
f.write('<table>\n')
for fiber in np.arange(5000):
    f.write('<td><a href=html/fiber_{}.html>FIBER_{}</a></td>\n'.format(fiber, fiber))
    f.write('</tr>\n')
f.write('</table>\n')
f.close()

for fiber in np.arange(5000):

    mask = stats['FIBER']==fiber
    index = np.where(mask)[0][0]

    f = open(os.path.join(html_dir, 'fiber_{}.html'.format(fiber)), "w")
    f.write('<html>\n')

    f.write('<style>\ntable, th, td {\n  border:1px solid black;\n}\n</style>\n')

    f.write('<p>')
    if fiber==0:
        f.write('<a href="fiber_1.html">Next Fiber</a>')
    elif fiber==4999:
        f.write('<a href="fiber_0.html">Previous Fiber</a>')
    else:
        f.write('<a href="fiber_{}.html">Previous Fiber</a> &nbsp; <a href="fiber_{}.html">Next Fiber</a>'.format(fiber-1, fiber+1))
    f.write('</p>')

    ################################## summary table ##################################
    f.write('<table>\n')

    f.write('<tr>\n')
    f.write('<th></th>\n')
    for tracer in ['BGS_BRIGHT', 'LRG', 'ELG_LOP', 'QSO', 'ELG_VLO', 'BGS_FAINT']:
        for apply_good_z_cut in [False, True]:
            if apply_good_z_cut is False:
                f.write('<th>{} (all)</th>\n'.format(tracer))
            else:
                f.write('<th>{} (good z)</th>\n'.format(tracer))
    f.write('</tr>\n')

    f.write('<tr>\n')
    f.write('<th>N(z) p-value</th>\n')
    for tracer in ['BGS_BRIGHT', 'LRG', 'ELG_LOP', 'QSO', 'ELG_VLO', 'BGS_FAINT']:
        for apply_good_z_cut in [False, True]:
            if apply_good_z_cut is False:
                pvalue_col = tracer.lower()+'_ks_pvalue_allz'
            else:
                pvalue_col = tracer.lower()+'_ks_pvalue_goodz'
            pvalue = stats[pvalue_col][index]
            if pvalue==-99:
                f.write('<td>N/A</td>\n')
            elif pvalue<pvalue_threshold:
                f.write('<th><p style="color:red;">{:.4g}</p></th>\n'.format(pvalue))
            else:
                f.write('<td>{:.4g}</td>\n'.format(pvalue))
    f.write('</tr>\n')

    f.write('<tr>\n')
    f.write('<th>fiber failure rate</th>\n')
    for tracer in ['BGS_BRIGHT', 'LRG', 'ELG_LOP', 'QSO', 'ELG_VLO', 'BGS_FAINT']:
        n_tot = stats[tracer.lower()+'_n_tot'][index]
        n_fail = stats[tracer.lower()+'_n_fail'][index]
        frac_fail = stats[tracer.lower()+'_frac_fail'][index]
        frac_fail_err = stats[tracer.lower()+'_frac_fail_err'][index]
        if n_tot==-99:
            f.write('<td colspan="2">N/A</td>\n')
        else:
            f.write('<td colspan="2">{:.1f} &pm; {:.1f}% ({}/{})</td>\n'.format(frac_fail*100, frac_fail_err*100, n_fail, n_tot))
    f.write('</tr>\n')

    f.write('<tr>\n')
    f.write('<th>overall failure rate</th>\n')
    for tracer in ['BGS_BRIGHT', 'LRG', 'ELG_LOP', 'QSO', 'ELG_VLO', 'BGS_FAINT']:
        mask = stats[tracer.lower()+'_n_tot']!=-99
        if np.sum(mask)==0:
            n_tot = -99
        else:
            n_tot = np.sum(stats[tracer.lower()+'_n_tot'][mask])
            n_fail = np.sum(stats[tracer.lower()+'_n_fail'][mask])
            frac_fail = n_fail/n_tot
        if n_tot==-99:
            f.write('<td colspan="2">N/A</td>\n')
        else:
            f.write('<td colspan="2">{:.1f}% ({}/{})</td>\n'.format(frac_fail*100, n_fail, n_tot))
    f.write('</tr>\n')

    f.write('</table>\n')

    ################################## QA plots ##################################

    f.write('<table>\n')
    for tracer in ['BGS_BRIGHT', 'LRG', 'ELG_LOP', 'QSO', 'ELG_VLO', 'BGS_FAINT']:

        f.write('<th></th>\n')
        f.write('<th>{}</th>\n'.format(tracer))
        f.write('<tr>\n')

        for apply_good_z_cut in [False, True]:

            if apply_good_z_cut:
                f.write('<td>Good z</td>\n')
                png_dir = '{}_goodz'.format(tracer.lower())
            else:
                f.write('<td>All</td>\n')
                png_dir = '{}_allz'.format(tracer.lower())

            image_fn = os.path.join(png_dir, 'fiber_{}_{}.png'.format(fiber, os.path.basename(png_dir)))

            f.write('<td><a href=\'../png/{}\'><img src=\'../png/{}\' width=\'1200\'></a></td>\n'.format(image_fn, image_fn))
            f.write('</tr>\n')

    f.write('</table>\n')
    f.close()
