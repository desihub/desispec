"""
Test desispec.scripts.proddag, which builds the navigable HTML view of a
production's job dependency graph.
"""

import json
import os
import re
import shutil
import tempfile
import unittest

from astropy.table import Table

from desispec.scripts.proddag import (
    find_proctables,
    read_production_dag,
    write_prod_dag_html,
    _elapsed_seconds,
    _first_expid_and_count,
    _parse_dep_ids,
)


def _proctable_rows(night, rows):
    """
    Build a processing table in the format Table.read produces, i.e. with
    multi-valued columns as '|' separated strings.

    Args:
        night (int): The night these rows belong to.
        rows (list of dict): One dict per job with keys seq, jobdesc, deps,
            and optionally tile, expids, qid, status, camword.

    Returns:
        Table: A processing table with the columns the reader requires.
    """
    out = {'NIGHT': [], 'INTID': [], 'JOBDESC': [], 'TILEID': [],
           'EXPID': [], 'PROCCAMWORD': [], 'LATEST_QID': [], 'STATUS': [],
           'INT_DEP_IDS': []}
    for row in rows:
        intid = (night - 20000000) * 1000 + row['seq']
        expids = row.get('expids', [])
        deps = row.get('deps', [])
        out['NIGHT'].append(night)
        out['INTID'].append(intid)
        out['JOBDESC'].append(row['jobdesc'])
        out['TILEID'].append(row.get('tile', -99))
        out['EXPID'].append(''.join(f'{e}|' for e in expids))
        out['PROCCAMWORD'].append(row.get('camword', 'a0123456789'))
        out['LATEST_QID'].append(row.get('qid', 1))
        out['STATUS'].append(row.get('status', 'COMPLETED'))
        out['INT_DEP_IDS'].append(''.join(f'{d}|' for d in deps))
    return Table(out)


class TestProdDag(unittest.TestCase):
    """Tests for reading a production DAG and writing its HTML view"""

    @classmethod
    def setUpClass(cls):
        cls.reduxdir = tempfile.mkdtemp()
        cls.specprod = 'dagtest'
        cls.proddir = os.path.join(cls.reduxdir, cls.specprod)
        cls.procdir = os.path.join(cls.proddir, 'processing_tables')
        os.makedirs(cls.procdir)

        cls.origenv = os.environ.copy()
        os.environ['DESI_SPECTRO_REDUX'] = cls.reduxdir
        os.environ['SPECPROD'] = cls.specprod

        ## night 1: a bias that a later night links from, plus a downstream job
        n1, n2 = 20240301, 20240302
        cls.night1, cls.night2 = n1, n2
        t1 = _proctable_rows(n1, [
            dict(seq=0, jobdesc='biasnight', expids=[100], qid=5001),
            dict(seq=1, jobdesc='ccdcalib', expids=[101], qid=5002,
                 deps=[(n1 - 20000000) * 1000 + 0]),
        ])
        ## night 2 links from night 1, giving one cross-night edge, and has a
        ## job that was never submitted (qid 1)
        t2 = _proctable_rows(n2, [
            dict(seq=0, jobdesc='linkcal', qid=5003,
                 deps=[(n1 - 20000000) * 1000 + 0]),
            dict(seq=1, jobdesc='tilenight', tile=1234, expids=[200, 201],
                 qid=5004, deps=[(n2 - 20000000) * 1000 + 0]),
            dict(seq=2, jobdesc='cumulative', tile=1234, expids=[200],
                 qid=1, status='UNKNOWN',
                 deps=[(n2 - 20000000) * 1000 + 1]),
        ])
        t1.write(os.path.join(cls.procdir,
                              f'processing_table_{cls.specprod}-{n1}.csv'))
        t2.write(os.path.join(cls.procdir,
                              f'processing_table_{cls.specprod}-{n2}.csv'))
        ## an unprocessed table in the same directory must be ignored
        t1.write(os.path.join(cls.procdir,
                              f'unprocessed_table_{cls.specprod}-{n1}.csv'))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.reduxdir)
        for key in ('DESI_SPECTRO_REDUX', 'SPECPROD'):
            if key in cls.origenv:
                os.environ[key] = cls.origenv[key]
            elif key in os.environ:
                del os.environ[key]

    # ------------------------------------------------------------------
    # small parsing helpers
    # ------------------------------------------------------------------

    def test_first_expid_and_count(self):
        """EXPID cells parse into (first, count), empty giving (-1, 0)"""
        self.assertEqual(_first_expid_and_count('100|101|102|'), (100, 3))
        self.assertEqual(_first_expid_and_count('100|'), (100, 1))
        self.assertEqual(_first_expid_and_count('|'), (-1, 0))
        self.assertEqual(_first_expid_and_count(''), (-1, 0))

    def test_parse_dep_ids(self):
        """INT_DEP_IDS cells parse into a list of ints"""
        self.assertEqual(_parse_dep_ids('240301000|240301001|'),
                         [240301000, 240301001])
        self.assertEqual(_parse_dep_ids('|'), [])
        self.assertEqual(_parse_dep_ids(''), [])

    def test_elapsed_seconds(self):
        """Slurm ELAPSED strings parse to seconds"""
        self.assertEqual(_elapsed_seconds('00:02:48'), 168)
        self.assertEqual(_elapsed_seconds('01:00:00'), 3600)
        self.assertEqual(_elapsed_seconds('2-03:00:00'), 2 * 86400 + 3 * 3600)
        self.assertEqual(_elapsed_seconds('--'), -1)
        self.assertEqual(_elapsed_seconds('nonsense'), -1)

    # ------------------------------------------------------------------
    # reading the graph
    # ------------------------------------------------------------------

    def test_read_production_dag(self):
        """All jobs and dependencies are read, with cross-night edges counted"""
        data = read_production_dag(self.proddir)
        self.assertEqual(data['njobs'], 5)
        self.assertEqual(data['nedges'], 4)
        ## only the linkcal -> night1 bias edge crosses a night boundary
        self.assertEqual(data['xnight'], 1)
        self.assertEqual(data['ndangling'], 0)
        self.assertEqual(data['nights'], [self.night1, self.night2])
        self.assertEqual(len(data['cols']['ni']), 5)

    def test_unprocessed_tables_ignored(self):
        """An unprocessed_table in the same directory is not read"""
        data = read_production_dag(self.proddir)
        ## reading the unprocessed copy too would double night 1's two jobs
        self.assertEqual(data['njobs'], 5)

    def test_intid_reconstruction(self):
        """The stored night plus sequence reproduces the original INTID"""
        data = read_production_dag(self.proddir)
        cols = data['cols']
        recovered = set()
        for i in range(data['njobs']):
            night = data['nights'][cols['ni'][i]]
            seq = cols['seq'][i]
            intid = -seq if seq < 0 else (night - 20000000) * 1000 + seq
            recovered.add(intid)
        expected = {(self.night1 - 20000000) * 1000 + 0,
                    (self.night1 - 20000000) * 1000 + 1,
                    (self.night2 - 20000000) * 1000 + 0,
                    (self.night2 - 20000000) * 1000 + 1,
                    (self.night2 - 20000000) * 1000 + 2}
        self.assertEqual(recovered, expected)

    def test_reads_the_requested_production_not_the_environment(self):
        """
        specprod_dir must decide which production is read.

        Resolving the processing table directory from $SPECPROD instead would
        silently return the environment's production while labelling the result
        as the requested one.
        """
        ## a second, deliberately different production alongside the first
        other_specprod = 'dagtest_other'
        other_proddir = os.path.join(self.reduxdir, other_specprod)
        other_procdir = os.path.join(other_proddir, 'processing_tables')
        os.makedirs(other_procdir, exist_ok=True)
        night = 20240401
        table = _proctable_rows(night, [
            dict(seq=0, jobdesc='biasnight', expids=[900], qid=7001),
        ])
        table.write(os.path.join(
            other_procdir, f'processing_table_{other_specprod}-{night}.csv'))

        ## $SPECPROD still points at the first production
        self.assertEqual(os.environ['SPECPROD'], self.specprod)

        data = read_production_dag(other_proddir)
        self.assertEqual(data['njobs'], 1)
        self.assertEqual(data['nights'], [night])
        self.assertEqual(data['specprod'], other_specprod)
        ## and the original production is unaffected
        self.assertEqual(read_production_dag(self.proddir)['njobs'], 5)

        shutil.rmtree(other_proddir)

    def test_find_proctables_honours_its_argument(self):
        """find_proctables() must glob under the directory it is given"""
        found = find_proctables(self.proddir)
        self.assertEqual(len(found), 2)
        for path in found:
            self.assertTrue(path.startswith(self.proddir))
        ## a directory with no processing tables yields nothing, rather than
        ## falling back to the environment's production
        empty = os.path.join(self.reduxdir, 'dagtest_empty')
        os.makedirs(os.path.join(empty, 'processing_tables'), exist_ok=True)
        self.assertEqual(find_proctables(empty), [])
        shutil.rmtree(empty)

    def test_nights_restriction(self):
        """Passing nights restricts which processing tables are read"""
        data = read_production_dag(self.proddir, nights=[self.night2])
        self.assertEqual(data['njobs'], 3)
        self.assertEqual(data['nights'], [self.night2])
        ## the cross-night dependency now points outside what was read
        self.assertEqual(data['ndangling'], 1)
        self.assertEqual(data['xnight'], 0)

    def test_multivalued_expid_count(self):
        """A job with several exposures records the first and the count"""
        data = read_production_dag(self.proddir)
        cols = data['cols']
        tilenight = data['jobdescs'].index('tilenight')
        idx = [i for i in range(data['njobs']) if cols['jd'][i] == tilenight]
        self.assertEqual(len(idx), 1)
        self.assertEqual(cols['exp'][idx[0]], 200)
        self.assertEqual(cols['nexp'][idx[0]], 2)

    # ------------------------------------------------------------------
    # writing the page
    # ------------------------------------------------------------------

    def test_write_prod_dag_html(self):
        """The page embeds the graph as parseable JSON with no placeholders"""
        data = read_production_dag(self.proddir)
        outdir = os.path.join(self.reduxdir, 'out')
        outfile = os.path.join(outdir, 'dag.html')
        write_prod_dag_html(data, outfile)
        self.assertTrue(os.path.exists(outfile))

        html = open(outfile).read()
        for placeholder in ('__DATA__', '__TITLE__', '__HEADING__'):
            self.assertNotIn(placeholder, html)

        match = re.search(
            r'<script id="dagdata" type="application/json">(.*?)</script>',
            html, re.S)
        self.assertIsNotNone(match)
        payload = json.loads(match.group(1).replace('<\\/', '</'))
        self.assertEqual(payload['njobs'], data['njobs'])
        self.assertEqual(payload['nedges'], data['nedges'])
        ## relprefix must point from the page back to the production
        self.assertTrue(payload['relprefix'].endswith('/'))
        self.assertTrue(os.path.isdir(
            os.path.join(outdir, payload['relprefix'])))

    def test_embedded_json_has_no_script_terminator(self):
        """A literal </ inside the payload would close the script block early"""
        data = read_production_dag(self.proddir)
        ## a camword-ish string containing the dangerous sequence
        data['camwords'].append('</script>')
        outfile = os.path.join(self.reduxdir, 'out2', 'dag.html')
        write_prod_dag_html(data, outfile)
        html = open(outfile).read()
        body = html.split('<script id="dagdata" type="application/json">')[1]
        payload_text = body.split('</script>')[0]
        payload = json.loads(payload_text.replace('<\\/', '</'))
        self.assertIn('</script>', payload['camwords'])


if __name__ == '__main__':
    unittest.main()
