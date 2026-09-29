# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""Test desispec.workflow.proc_dashboard_funcs page building
"""

import os
import shutil
import tempfile
import unittest
from collections import OrderedDict

from desispec.workflow.queue import get_non_final_states
from desispec.workflow.proc_dashboard_funcs import make_html_page, \
    year_page_pathname, generate_nightly_table_html, \
    generate_monthly_table_html, _combine_banner_statuses


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
            self.assertIn(f'<a href="#{year}" id="yearlink-{year}" class="GOOD">',
                          master)
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

    def test_year_links_are_colored_by_their_months(self):
        """A year link combines its months as a month banner combines nights"""
        def night(expid, color, status):
            return {f'science_{expid}': _row(expid, color, status)}

        tables = OrderedDict([
            ## a failure outranks work still pending
            ('202602', {20260201: night(4001, 'PENDING', 'PENDING')}),
            ('202601', {20260115: night(3001, 'BAD', 'FAILED')}),
            ## pending outranks done
            ('202512', {20251210: night(2001, 'PENDING', 'RUNNING')}),
            ('202511', {20251105: night(1001, 'GOOD', 'COMPLETED')}),
            ## all done
            ('202412', {20241210: night(501, 'GOOD', 'COMPLETED')}),
            ])
        make_html_page(tables, self.outfile, show_null=True)
        master = open(self.outfile).read()

        for year, status in [('2026', 'BAD'), ('2025', 'PENDING'),
                             ('2024', 'GOOD')]:
            with self.subTest(year=year):
                self.assertIn(f'id="yearlink-{year}" class="{status}">', master)
        ## every status a link can take has a color
        for status in ['BAD', 'PENDING', 'GOOD', 'DEFAULT']:
            self.assertIn(f'.yearnav a.{status} {{', master)

    def test_selecting_a_year_keeps_its_color(self):
        """Marking the active year must not replace the class that colors it"""
        make_html_page(self.monthly_tables, self.outfile, show_null=True)
        master = open(self.outfile).read()
        self.assertIn("classList.toggle('active'", master)
        self.assertNotIn('link.className =', master)

    def test_header_is_compact(self):
        """The run time and the legend share a single line"""
        make_html_page(self.monthly_tables, self.outfile, show_null=True)
        for page in [open(self.outfile).read(),
                     open(year_page_pathname(self.outfile, '2026')).read()]:
            self.assertNotIn('running at:', page)
            ## both inside the one row, the run time first
            row = page.split('<div class="pageinfo">', 1)[1].split('</div>', 1)[0]
            self.assertIn('<span class="runtime">Run at: ', row)
            self.assertIn('<span class="legend">Color Legend:', row)
            self.assertLess(row.index('Run at:'), row.index('Color Legend:'))
            self.assertIn(' <span id="GOOD">GOOD</span>', row)
            self.assertIn('.pageinfo {display: flex; flex-wrap: wrap;', page)
            self.assertEqual(page.count('Color Legend:'), 1)

    def test_year_links_sit_above_the_frame(self):
        """The year links come after the header, directly above the year"""
        make_html_page(self.monthly_tables, self.outfile, show_null=True)
        master = open(self.outfile).read()
        self.assertLess(master.index('<div class="pageinfo">'),
                        master.index('<nav class="yearnav">'))
        self.assertLess(master.index('</nav>'),
                        master.index('<iframe id="yearframe"'))

    def test_empty_months_do_not_make_a_year(self):
        """A month with no nights shouldn't put an empty year in the nav"""
        tables = OrderedDict([('202601', {20260115: _night_info(3001)}),
                              ('202412', {})])
        make_html_page(tables, self.outfile, show_null=True)

        self.assertFalse(os.path.exists(year_page_pathname(self.outfile, '2024')))
        self.assertNotIn('#2024', open(self.outfile).read())


def _row(expid, color, status):
    """One dashboard row with a given color and status"""
    return {'COLOR': color, 'EXPID': str(expid), 'OBSTYPE': 'science',
            'STATUS': status}


class TestBannerStatuses(unittest.TestCase):
    """Night and month banners combine what they hold, with problems first"""

    def test_combine_precedence(self):
        """A problem outranks pending work, which outranks being done"""
        for statuses, expected in [
                (['GOOD', 'GOOD'], 'GOOD'),
                ([], 'GOOD'),
                (['GOOD', 'PENDING'], 'PENDING'),
                (['PENDING', 'BAD'], 'BAD'),
                (['PENDING', 'INCOMPLETE'], 'INCOMPLETE'),
                (['PENDING', 'OVERFULL'], 'OVERFULL'),
                (['INCOMPLETE', 'BAD'], 'BAD'),
                (['OVERFULL', 'INCOMPLETE'], 'INCOMPLETE'),
                (['GOOD', 'DEFAULT'], 'DEFAULT'),
                (['DEFAULT', 'PENDING'], 'PENDING')]:
            with self.subTest(statuses=statuses):
                self.assertEqual(_combine_banner_statuses(statuses), expected)

    def test_night_with_pending_jobs_is_pending(self):
        """Unfinished jobs on an otherwise good night read as pending"""
        for color in ['PENDING', 'RUNNING']:
            with self.subTest(color=color):
                night_info = {'a': _row(1, 'GOOD', 'COMPLETED'),
                              'b': _row(2, color, color)}
                _, status = generate_nightly_table_html(night_info, 20260115,
                                                        show_null=True)
                self.assertEqual(status, 'PENDING')

    def test_every_unfinished_queue_state_is_pending(self):
        """Rows colored by any non-final queue state keep the night pending"""
        for state in get_non_final_states():
            with self.subTest(state=state):
                night_info = {'a': _row(1, 'GOOD', 'COMPLETED'),
                              'b': _row(2, state, state)}
                html, status = generate_nightly_table_html(night_info, 20260115,
                                                           show_null=True)
                self.assertEqual(status, 'PENDING')
                ## RUNNING keeps a count of its own
                if state == 'RUNNING':
                    self.assertIn('Pending: 0/2', html)
                    self.assertIn('Running: 1/2', html)
                else:
                    self.assertIn('Pending: 1/2', html)
                    self.assertIn('Running: 0/2', html)

    def test_night_with_bad_and_incomplete_jobs_is_bad(self):
        """Nights rank a failure above a partial result, as months do"""
        night_info = {'a': _row(1, 'BAD', 'FAILED'),
                      'b': _row(2, 'INCOMPLETE', 'COMPLETED')}
        _, status = generate_nightly_table_html(night_info, 20260115,
                                                show_null=True)
        self.assertEqual(status, 'BAD')

    def test_night_with_bad_and_pending_jobs_is_bad(self):
        """A failure isn't hidden behind jobs that haven't finished"""
        night_info = {'a': _row(1, 'BAD', 'FAILED'),
                      'b': _row(2, 'PENDING', 'PENDING')}
        _, status = generate_nightly_table_html(night_info, 20260115,
                                                show_null=True)
        self.assertEqual(status, 'BAD')

    def test_month_with_a_pending_night_is_pending(self):
        """The month banner follows its nights"""
        html = generate_monthly_table_html(['', ''], ['GOOD', 'PENDING'],
                                           '202601')
        self.assertIn('<button class="collapsible monthbanner" id="PENDING">', html)
        html = generate_monthly_table_html(['', ''], ['BAD', 'PENDING'],
                                           '202601')
        self.assertIn('<button class="collapsible monthbanner" id="BAD">', html)


class TestStickyBanners(unittest.TestCase):
    """Month and night banners stay in view only while their rows are"""

    def setUp(self):
        self.outdir = tempfile.mkdtemp()
        self.outfile = os.path.join(self.outdir, 'dashboard.html')
        self.origspecprod = os.environ.get('SPECPROD')
        os.environ['SPECPROD'] = 'test'
        tables = OrderedDict([
            ('202602', {20260202: _night_info(4002),
                        20260201: _night_info(4001)}),
            ('202601', {20260115: _night_info(3001)}),
            ])
        make_html_page(tables, self.outfile, show_null=True)
        self.page = open(year_page_pathname(self.outfile, '2026')).read()

    def tearDown(self):
        shutil.rmtree(self.outdir, ignore_errors=True)
        if self.origspecprod is None:
            del os.environ['SPECPROD']
        else:
            os.environ['SPECPROD'] = self.origspecprod

    def test_banners_are_bounded_by_their_sections(self):
        """Each banner opens its own section, which closes after its rows"""
        page = self.page
        self.assertEqual(page.count('<section class="month">'), 2)
        self.assertEqual(page.count('<section class="night">'), 3)
        self.assertEqual(page.count('</section>'), 5)
        self.assertEqual(page.count('<section class="month">\n<button class="collapsible monthbanner"'), 2)
        self.assertEqual(page.count('<section class="night">\n<button class="collapsible nightbanner"'), 3)
        ## every night of a month is inside that month's section
        for month, nights in [('202602', [20260202, 20260201]),
                              ('202601', [20260115])]:
            start = page.index(f'<!--Begin {month}-->')
            end = page.index(f'<!--End {month}-->')
            month_html = page[start:end]
            self.assertTrue(month_html.rstrip().endswith('</section>'))
            for night in nights:
                self.assertIn(f'<!--Begin {night}-->', month_html)
                self.assertIn(f'<!--End {night}-->', month_html)

    def test_banners_are_followed_by_their_content(self):
        """The collapse handler toggles the banner's next sibling"""
        for banner in ['monthbanner', 'nightbanner']:
            pieces = self.page.split(f'<button class="collapsible {banner}"')[1:]
            self.assertGreater(len(pieces), 0)
            for piece in pieces:
                after = piece.split('</button>', 1)[1]
                self.assertTrue(after.startswith('<div class="content"'), banner)

    def test_css_lets_banners_stick(self):
        """Banners are sticky and nothing around them is a scroll container"""
        self.assertIn('.monthbanner {position: sticky; top: 0;', self.page)
        self.assertIn('.nightbanner {position: sticky; '
                      + 'top: var(--month-banner-height', self.page)
        ## hidden is only a fallback for browsers that don't know clip, so
        ## clip has to come after it to win where it is understood
        content_rule = self.page.split('.content {', 1)[1].split('}', 1)[0]
        self.assertIn('overflow: clip', content_rule)
        if 'overflow: hidden' in content_rule:
            self.assertLess(content_rule.index('overflow: hidden'),
                            content_rule.index('overflow: clip'))
        ## the column names stick below both banners
        self.assertIn('.nightTable th {position: sticky;', self.page)
        self.assertIn('top: calc(var(--month-banner-height, 66px) '
                      + '+ var(--night-banner-height, 66px))', self.page)
        ## each banner is measured, not just the first of its kind
        self.assertIn("['.monthbanner', '--month-banner-height']", self.page)
        self.assertIn("['.nightbanner', '--night-banner-height']", self.page)
        self.assertIn('querySelectorAll(kinds[j][0])', self.page)
