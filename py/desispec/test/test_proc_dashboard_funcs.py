# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""Test desispec.workflow.proc_dashboard_funcs page building
"""

import os
import shutil
import tempfile
import unittest
from collections import OrderedDict

from desispec.workflow.proc_dashboard_funcs import make_html_page, \
    year_page_pathname


def _night_info(expid):
    """One dashboard row, of the shape populate_exp_night_info returns"""
    return {f'science_{expid}': {'COLOR': 'GOOD', 'EXPID': str(expid),
                                 'OBSTYPE': 'science', 'STATUS': 'COMPLETED'}}


class TestDashboardPages(unittest.TestCase):
    """The dashboard is split into one page per year plus a master page"""

    def setUp(self):
        self.outdir = tempfile.mkdtemp()
        self.outfile = os.path.join(self.outdir, 'dashboard.html')
        self.origspecprod = os.environ.get('SPECPROD')
        os.environ['SPECPROD'] = 'test'
        ## reverse chronological, as populate_monthly_tables produces
        self.monthly_tables = OrderedDict([
            ('202601', {20260115: _night_info(3001)}),
            ('202512', {20251210: _night_info(2001)}),
            ('202511', {20251105: _night_info(1001)}),
            ])

    def tearDown(self):
        shutil.rmtree(self.outdir, ignore_errors=True)
        if self.origspecprod is None:
            del os.environ['SPECPROD']
        else:
            os.environ['SPECPROD'] = self.origspecprod

    def test_one_page_per_year_plus_a_master(self):
        """Each year gets its own page and the master keeps the old name"""
        make_html_page(self.monthly_tables, self.outfile, show_null=True)

        pages = sorted(os.listdir(self.outdir))
        self.assertEqual(pages, ['dashboard-2025.html', 'dashboard-2026.html',
                                 'dashboard.html'])

        ## every month is on exactly one year page, and none on the master
        master = open(self.outfile).read()
        for month, year in [('202601', '2026'), ('202512', '2025'),
                            ('202511', '2025')]:
            marker = f'<!--Begin {month}-->'
            page = open(year_page_pathname(self.outfile, year)).read()
            self.assertIn(marker, page)
            self.assertNotIn(marker, master)
            other = '2025' if year == '2026' else '2026'
            self.assertNotIn(marker,
                             open(year_page_pathname(self.outfile, other)).read())

    def test_master_frames_the_years_newest_first(self):
        """The master is small and offers a link per year, newest first"""
        make_html_page(self.monthly_tables, self.outfile, show_null=True)
        master = open(self.outfile).read()

        self.assertIn('<iframe id="yearframe"', master)
        self.assertIn('var years = ["2026", "2025"]', master)
        ## the newest year is framed in the markup, so the page shows something
        ## without javascript, and data-year matches so showYear() doesn't
        ## refetch the same file on load
        self.assertIn('data-year="2026" src="dashboard-2026.html"', master)
        for year in ['2025', '2026']:
            self.assertIn(f'<a href="#{year}" id="yearlink-{year}">', master)
            ## relative, so the pages move together
            self.assertIn(f'"dashboard-{year}.html"', master)
            self.assertNotIn(self.outdir, master)

        ## the point of the split is that the master is cheap to load
        self.assertLess(len(master), 20000)

    def test_year_pages_stand_alone(self):
        """A year page is a complete dashboard, openable on its own"""
        make_html_page(self.monthly_tables, self.outfile, show_null=True)
        page = open(year_page_pathname(self.outfile, '2025')).read()

        self.assertTrue(page.startswith('<html>'))
        self.assertIn('</body></html>', page.replace('\n', '').replace(' ', ''))
        self.assertIn('Filter By Status', page)
        self.assertIn('Color Legend', page)
        self.assertIn('collapsible', page)
        ## the buttons this handler wants belong to desi_dashboard.py, so
        ## without a guard the whole script would throw on load
        self.assertIn('if (b1)', page)

    def test_year_pages_hide_what_the_master_already_shows(self):
        """Framed, a year page would otherwise repeat the title and legend"""
        make_html_page(self.monthly_tables, self.outfile, show_null=True)
        page = open(year_page_pathname(self.outfile, '2025')).read()

        ## kept for opening the page directly, but hidden inside the frame
        self.assertIn('<div id="pageheader">', page)
        self.assertIn('.framed #pageheader {display: none;}', page)
        ## set in the head so the hidden header never flashes before paint
        self.assertIn('window.self !== window.top', page)
        self.assertLess(page.index('window.self !== window.top'),
                        page.index('<body>'))

        ## the master shows them unconditionally, it is never framed itself
        master = open(self.outfile).read()
        self.assertIn('Status Monitor</h1>', master)
        self.assertIn('Color Legend', master)
        self.assertNotIn('<div id="pageheader">', master)

    def test_years_no_longer_covered_are_removed(self):
        """A rerun over fewer nights shouldn't leave pages behind"""
        make_html_page(self.monthly_tables, self.outfile, show_null=True)
        self.assertTrue(os.path.exists(year_page_pathname(self.outfile, '2025')))

        smaller = OrderedDict([('202601', {20260115: _night_info(3001)})])
        make_html_page(smaller, self.outfile, show_null=True)

        self.assertFalse(os.path.exists(year_page_pathname(self.outfile, '2025')))
        self.assertTrue(os.path.exists(year_page_pathname(self.outfile, '2026')))
        self.assertNotIn('#2025', open(self.outfile).read())

    def test_other_dashboards_in_the_directory_are_left_alone(self):
        """The two dashboards share an output directory"""
        zoutfile = os.path.join(self.outdir, 'zdashboard.html')
        make_html_page(self.monthly_tables, zoutfile, show_null=True)
        make_html_page(OrderedDict([('202601', {20260115: _night_info(3001)})]),
                       self.outfile, show_null=True)

        ## the exposure dashboard dropping 2025 must not delete the z one's
        self.assertTrue(os.path.exists(year_page_pathname(zoutfile, '2025')))
        self.assertFalse(os.path.exists(year_page_pathname(self.outfile, '2025')))

    def test_empty_months_do_not_make_a_year(self):
        """A month with no nights shouldn't put an empty year in the nav"""
        tables = OrderedDict([('202601', {20260115: _night_info(3001)}),
                              ('202412', {})])
        make_html_page(tables, self.outfile, show_null=True)

        self.assertFalse(os.path.exists(year_page_pathname(self.outfile, '2024')))
        self.assertNotIn('#2024', open(self.outfile).read())
