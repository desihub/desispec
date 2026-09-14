#!/usr/bin/env python

# This is step 3 of the redshift QA generation procedure
# Create the redshift QA HTML files

# Usage:
# export QA_DIR=/global/cfs/cdirs/desicollab/users/rongpu/redshift_qa/new/loa
# ./desi_per_fiber_qa_html.py --stats per_fiber_qa_stats.fits -o $QA_DIR

from __future__ import division, print_function
import sys, os, glob, time, warnings, gc
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table, vstack, hstack
import fitsio
from astropy.io import fits

import argparse


def main():

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

    # Track fibers with problematic p-values
    fibers_with_elg_lop_allz_issues = set()
    fibers_with_other_issues = set()

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
                    # Track which fibers have issues
                    if tracer == 'ELG_LOP' and apply_good_z_cut is False:
                        fibers_with_elg_lop_allz_issues.add(fiber)
                    else:
                        fibers_with_other_issues.add(fiber)
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

    ################################## Fiber directory ##################################

    f = open(os.path.join(output_dir, 'fiber_directory.html'), "w")
    f.write('<html>\n')
    f.write('<head><style>\n')
    f.write('body { font-family: Arial, sans-serif; margin: 20px; }\n')
    f.write('h2 { margin-top: 30px; }\n')
    f.write('table { border-collapse: collapse; }\n')
    f.write('td { padding: 5px 10px; }\n')
    f.write('a { text-decoration: none; }\n')
    f.write('a:hover { text-decoration: underline; }\n')
    f.write('</style></head>\n')
    f.write('<body>\n')
    f.write('<h1>DESI QA Fiber Directory</h1>\n')

    # List fibers with non-ELG_LOP issues at the top
    if len(fibers_with_other_issues) > 0:
        f.write('<h2>Fibers with Issues (excluding ELG_LOP all-only)</h2>\n')
        f.write('<p style="color:red; font-weight:bold;">Total: {} fibers</p>\n'.format(len(fibers_with_other_issues)))
        f.write('<p>')
        sorted_problem_fibers = sorted(fibers_with_other_issues)
        for i, fiber in enumerate(sorted_problem_fibers):
            if i > 0:
                f.write(', ')
            f.write('<a href=html/fiber_{}.html style="color:red; font-weight:bold;">{}</a>'.format(fiber, fiber))
        f.write('</p>\n')
    else:
        f.write('<h2>No fibers with issues (excluding ELG_LOP all-only)</h2>\n')

    f.write('<h2>All Fibers</h2>\n')

    # Group fibers in ranges of 500
    for group_start in range(0, 5000, 500):
        group_end = group_start + 499
        f.write('<h2></h2>\n')
        f.write('<table>\n')

        # Write fibers in rows of 10
        for row_start in range(group_start, group_start + 500, 10):
            f.write('<tr>\n')
            for fiber in range(row_start, min(row_start + 10, 5000)):
                if fiber in fibers_with_other_issues:
                    # Red and bold for fibers with issues other than ELG_LOP (all)
                    f.write('<td><a href=html/fiber_{}.html style="color:red; font-weight:bold;">{}</a></td>\n'.format(fiber, fiber))
                elif fiber in fibers_with_elg_lop_allz_issues:
                    # Red only for fibers with only ELG_LOP (all) issues
                    f.write('<td><a href=html/fiber_{}.html style="color:red;">{}</a></td>\n'.format(fiber, fiber))
                else:
                    f.write('<td><a href=html/fiber_{}.html>{}</a></td>\n'.format(fiber, fiber))
            f.write('</tr>\n')

        f.write('</table>\n')

    f.write('</body>\n')
    f.write('</html>\n')
    f.close()

if __name__ == '__main__':
    main()
