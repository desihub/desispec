"""
desispec.scripts.proddag
========================

Build a navigable HTML view of the job dependency graph of an entire
production.

A production is a single connected DAG rather than a set of independent
per-night graphs: cross-night dependencies (linked calibrations, cumulative
redshifts spanning nights, darknight stacks) tie the nights together. A real
production has of order 100,000 jobs, which is far too many to draw at once,
so the generated page is a browsable object rather than one static diagram. It
opens on a per-night overview and lets you drill into a single night's graph or
walk the ancestors and descendants of any one job across night boundaries.

See also desi_job_graph, which renders a single night as a static mermaid
diagram.
"""

import os
import glob
import json
import time
from importlib import resources

import numpy as np
from astropy.table import Table

from desiutil.log import get_logger

from desispec.io.meta import specprod_root
from desispec.workflow.proctable import get_processing_table_path


## Columns the page needs from a processing table
REQUIRED_COLUMNS = ('NIGHT', 'INTID', 'JOBDESC', 'TILEID', 'EXPID',
                    'PROCCAMWORD', 'LATEST_QID', 'INT_DEP_IDS')

## Job states that are worth colouring distinctly. Anything else is bucketed
## as UNKNOWN by the page.
STATE_COLORS = {
    'COMPLETED': '#7fbf7b',
    'RUNNING': '#8bbee8',
    'PENDING': '#c6c6c6',
    'FAILED': '#d95f0e',
    'OUT_OF_MEMORY': '#c1492b',
    'TIMEOUT': '#b3562a',
    'CANCELLED': '#fed98e',
    'DEP_NOT_SUBD': '#e7a6c8',
    'MAX_RESUB': '#b07aa1',
    'NOTSUBMITTED': '#fcae1e',
    'UNKNOWN': '#ffffcc',
}


def _as_int(value, default=-1):
    """Coerce a possibly-masked table value to int, with a fallback"""
    try:
        if value is None or value is np.ma.masked:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _first_expid_and_count(raw):
    """
    Parse a processing table EXPID cell into (first_expid, n_expids).

    Processing tables store multi-valued columns as '|'-separated strings when
    read with Table.read, e.g. '334725|334726|'. Returns (-1, 0) when empty.
    """
    text = str(raw).strip()
    parts = [p for p in text.split('|') if p not in ('', '--')]
    if len(parts) == 0:
        return -1, 0
    try:
        return int(parts[0]), len(parts)
    except ValueError:
        return -1, len(parts)


def _parse_dep_ids(raw):
    """Parse a processing table INT_DEP_IDS cell into a list of ints"""
    text = str(raw).strip()
    out = []
    for part in text.split('|'):
        if part in ('', '--'):
            continue
        try:
            out.append(int(part))
        except ValueError:
            continue
    return out


def find_proctables(specprod_dir):
    """
    Return the sorted processing table pathnames for a production.

    Args:
        specprod_dir (str): Full path to the production directory.

    Returns:
        list: Sorted pathnames of the per-night processing tables.
    """
    ## get_processing_table_path reads the environment, which the caller has
    ## already pointed at specprod_dir
    procdir = get_processing_table_path(None)
    pattern = os.path.join(procdir, 'processing_table_*.csv')
    ## exclude the unprocessed tables, which live in the same directory
    return sorted(p for p in glob.glob(pattern)
                  if os.path.basename(p).startswith('processing_table_'))


def read_production_dag(specprod_dir, nights=None):
    """
    Read every processing table of a production into a compact graph.

    Args:
        specprod_dir (str): Full path to the production directory.
        nights (list of int, optional): Only read these nights.

    Returns:
        dict: A JSON-serialisable columnar representation of the DAG. String
            valued fields are interned into lookup tables and referenced by
            index so that the payload stays small enough to embed in one page.
    """
    log = get_logger()

    proctables = find_proctables(specprod_dir)
    if nights is not None:
        wanted = set(int(n) for n in nights)
        proctables = [p for p in proctables
                      if _night_from_proctable_name(p) in wanted]
    log.info(f'Found {len(proctables)} processing tables in {specprod_dir}')

    ## interning tables
    jobdescs, jobdesc_idx = [], {}
    statuses, status_idx = [], {}
    camwords, camword_idx = [], {}

    def intern(value, values, index):
        value = str(value)
        if value not in index:
            index[value] = len(values)
            values.append(value)
        return index[value]

    ## per-job columns
    col_night, col_seq, col_jd, col_st = [], [], [], []
    col_tile, col_qid, col_exp, col_nexp, col_cw = [], [], [], [], []

    intid_to_node = {}
    ## dependencies are resolved after all nights are read, since they cross
    ## night boundaries and may point either backwards or forwards in time
    pending_deps = []
    skipped = []

    for path in proctables:
        try:
            table = Table.read(path)
        except Exception as err:      # noqa: BLE001 - report and carry on
            skipped.append((os.path.basename(path), f'unreadable: {err}'))
            continue

        absent = [c for c in REQUIRED_COLUMNS if c not in table.colnames]
        if len(absent) > 0:
            skipped.append((os.path.basename(path),
                            'missing columns ' + ','.join(absent)))
            continue

        has_status = 'STATUS' in table.colnames
        for row in table:
            night = _as_int(row['NIGHT'])
            intid = _as_int(row['INTID'])
            if night < 0 or intid < 0:
                continue

            node = len(col_night)
            intid_to_node[intid] = node

            ## INTID is night-encoded as (night - 20000000) * 1000 + sequence,
            ## so only the sequence has to be stored
            seq = intid - (night - 20000000) * 1000
            if not 0 <= seq < 1000:
                ## unexpected encoding; fall back to storing the raw intid
                seq = -intid

            expid, nexp = _first_expid_and_count(row['EXPID'])
            status = str(row['STATUS']) if has_status else 'UNKNOWN'
            if status in ('', '--'):
                status = 'UNKNOWN'

            col_night.append(night)
            col_seq.append(seq)
            col_jd.append(intern(row['JOBDESC'], jobdescs, jobdesc_idx))
            col_st.append(intern(status, statuses, status_idx))
            col_tile.append(_as_int(row['TILEID']))
            col_qid.append(_as_int(row['LATEST_QID']))
            col_exp.append(expid)
            col_nexp.append(nexp)
            col_cw.append(intern(row['PROCCAMWORD'], camwords, camword_idx))

            for dep in _parse_dep_ids(row['INT_DEP_IDS']):
                pending_deps.append((dep, node))

    ## resolve dependency INTIDs to node indices
    edge_src, edge_dst, dangling = [], [], 0
    for dep_intid, dst in pending_deps:
        src = intid_to_node.get(dep_intid)
        if src is None:
            dangling += 1
            continue
        edge_src.append(src)
        edge_dst.append(dst)

    nights_present = sorted(set(col_night))
    night_pos = {n: i for i, n in enumerate(nights_present)}

    ## count how many dependencies cross a night boundary, which is what makes
    ## a production a single graph rather than one graph per night
    n_xnight = sum(1 for s, d in zip(edge_src, edge_dst)
                   if col_night[s] != col_night[d])

    log.info(f'Read {len(col_night)} jobs and {len(edge_src)} dependencies '
             f'across {len(nights_present)} nights '
             f'({n_xnight} dependencies cross a night boundary)')
    if dangling > 0:
        log.warning(f'{dangling} dependencies point at jobs that are not in '
                    'any processing table that was read; they are omitted')
    if len(skipped) > 0:
        log.warning(f'Skipped {len(skipped)} processing tables:')
        for name, why in skipped[:10]:
            log.warning(f'    {name}: {why}')
        if len(skipped) > 10:
            log.warning(f'    ... and {len(skipped) - 10} more')

    return {
        'specprod': os.path.basename(specprod_dir.rstrip('/')),
        'specproddir': specprod_dir,
        'generated': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'nights': nights_present,
        'jobdescs': jobdescs,
        'statuses': statuses,
        'camwords': camwords,
        'stateColors': STATE_COLORS,
        'njobs': len(col_night),
        'nedges': len(edge_src),
        'xnight': n_xnight,
        'ndangling': dangling,
        'nskipped': len(skipped),
        'cols': {
            'ni': [night_pos[n] for n in col_night],
            'seq': col_seq,
            'jd': col_jd,
            'st': col_st,
            'tile': col_tile,
            'qid': col_qid,
            'exp': col_exp,
            'nexp': col_nexp,
            'cw': col_cw,
        },
        'esrc': edge_src,
        'edst': edge_dst,
    }


def _elapsed_seconds(text):
    """Parse a slurm ELAPSED string ([D-]HH:MM:SS) into seconds, or -1"""
    text = str(text).strip()
    if text in ('', '--', 'INVALID'):
        return -1
    days = 0
    if '-' in text:
        dstr, text = text.split('-', 1)
        try:
            days = int(dstr)
        except ValueError:
            return -1
    bits = text.split(':')
    try:
        bits = [int(b) for b in bits]
    except ValueError:
        return -1
    while len(bits) < 3:
        bits.insert(0, 0)
    return days * 86400 + bits[0] * 3600 + bits[1] * 60 + bits[2]


def update_states_from_queue(data, chunk=400, dry_run_level=0):
    """
    Replace the processing table job states with live states from Slurm.

    The STATUS column of a processing table is only refreshed when proc_night
    runs against that night, so it goes stale. This queries sacct for the
    actual state of every job that has a real queue id. sacct cannot be handed
    ~100,000 job ids at once, so the query is chunked; on a full production
    this takes a while, which is why it is opt-in.

    Args:
        data (dict): Output of read_production_dag(), modified in place.
        chunk (int): Number of queue ids per sacct call.
        dry_run_level (int): Passed through to queue_info_from_qids.

    Returns:
        dict: The same dict, with updated 'statuses'/'cols' and an added
            'cols.el' column of elapsed seconds (-1 where unknown).
    """
    from desispec.workflow.queue import queue_info_from_qids

    log = get_logger()
    qids = data['cols']['qid']
    unique = sorted({q for q in qids if q > 1})
    log.info(f'Querying Slurm for {len(unique)} job ids in chunks of {chunk}')

    state_by_qid, elapsed_by_qid = {}, {}
    for start in range(0, len(unique), chunk):
        batch = unique[start:start + chunk]
        try:
            qinfo = queue_info_from_qids(batch, dry_run_level=dry_run_level,
                                         loglevel='warning')
        except Exception as err:      # noqa: BLE001 - keep whatever we have
            log.warning(f'sacct query failed for a chunk of {len(batch)} '
                        f'job ids, leaving those states as-is: {err}')
            continue
        if qinfo is None:
            continue
        for row in qinfo:
            ## sacct can return sub-steps like 12345.batch; keep the parent
            jobid = str(row['JOBID']).split('.')[0]
            try:
                jobid = int(jobid)
            except ValueError:
                continue
            state_by_qid[jobid] = str(row['STATE']).split()[0]
            if 'ELAPSED' in qinfo.colnames:
                elapsed_by_qid[jobid] = _elapsed_seconds(row['ELAPSED'])
        if (start // chunk) % 25 == 0:
            log.info(f'  ...{min(start + chunk, len(unique))}/{len(unique)}')

    ## re-intern statuses, since sacct can report states the tables never had
    statuses, status_idx = [], {}
    def intern(value):
        if value not in status_idx:
            status_idx[value] = len(statuses)
            statuses.append(value)
        return status_idx[value]

    old_statuses = data['statuses']
    new_st, new_el = [], []
    n_updated = 0
    for i, qid in enumerate(qids):
        if qid <= 1:
            state = 'NOTSUBMITTED'
        elif qid in state_by_qid:
            state = state_by_qid[qid]
            n_updated += 1
        else:
            state = old_statuses[data['cols']['st'][i]]
        new_st.append(intern(state))
        new_el.append(elapsed_by_qid.get(qid, -1))

    data['statuses'] = statuses
    data['cols']['st'] = new_st
    data['cols']['el'] = new_el
    log.info(f'Updated {n_updated} job states from Slurm; '
             f'states present: {statuses}')
    return data


def write_prod_dag_html(data, outfile, relprefix=None):
    """
    Write the navigable DAG page, with the graph embedded as JSON.

    Args:
        data (dict): Output of read_production_dag().
        outfile (str): Pathname of the HTML file to write.
        relprefix (str, optional): Path from the directory holding outfile to
            the production directory, used to build links to slurm logs. If
            None it is derived from the two pathnames.

    Returns:
        str: The pathname written.
    """
    log = get_logger()

    outdir = os.path.dirname(os.path.abspath(outfile))
    if relprefix is None:
        relprefix = os.path.relpath(data['specproddir'], outdir)
    if not relprefix.endswith('/'):
        relprefix += '/'
    data = dict(data, relprefix=relprefix)

    template = resources.files('desispec').joinpath(
            'data/proddag_template.html').read_text()

    ## the payload is embedded in a <script type="application/json"> block, so
    ## the only sequence that has to be neutralised is a literal '</'
    payload = json.dumps(data, separators=(',', ':')).replace('</', '<\\/')
    title = f"{data['specprod']} job DAG"
    heading = (f"{data['specprod']} &mdash; production job graph")

    html = (template
            .replace('__TITLE__', title)
            .replace('__HEADING__', heading)
            .replace('__DATA__', payload))

    os.makedirs(outdir, exist_ok=True)
    with open(outfile, 'w') as fx:
        fx.write(html)
    log.info(f'Wrote {outfile} ({len(html)/1e6:.2f} MB)')
    return outfile


def main(args=None):
    """
    Entry point for desi_prod_dag.

    Args:
        args (argparse.Namespace, optional): Parsed arguments. If None they are
            parsed from the command line.

    Returns:
        int: 0 on success.
    """
    if args is None:
        args = parse(None)

    ## resolve the production directory, allowing either a name or a full path
    if args.specprod is not None and os.path.isdir(args.specprod):
        specprod_dir = os.path.abspath(args.specprod)
    else:
        specprod_dir = specprod_root(args.specprod)
    if not os.path.isdir(specprod_dir):
        raise IOError(f'Production directory not found: {specprod_dir}')

    ## point the workflow path helpers at this production
    os.environ['DESI_SPECTRO_REDUX'] = os.path.dirname(specprod_dir)
    os.environ['SPECPROD'] = os.path.basename(specprod_dir)

    outfile = args.output
    if outfile is None:
        outfile = os.path.join(specprod_dir, 'run', 'jobgraph',
                               f'proddag-{os.path.basename(specprod_dir)}.html')

    nights = None
    if args.nights is not None:
        nights = [int(n) for n in str(args.nights).replace(',', ' ').split()]

    data = read_production_dag(specprod_dir, nights=nights)
    if data['njobs'] == 0:
        raise RuntimeError(f'No jobs found in any processing table under '
                           f'{specprod_dir}')
    if args.update_from_queue:
        update_states_from_queue(data)
    write_prod_dag_html(data, outfile)

    print(f"Wrote {outfile}")
    print(f"  {data['njobs']} jobs, {data['nedges']} dependencies, "
          f"{len(data['nights'])} nights, "
          f"{data['xnight']} cross-night dependencies")
    desi_root = os.environ.get('DESI_ROOT')
    if desi_root and os.path.abspath(outfile).startswith(desi_root):
        url = os.path.abspath(outfile).replace(
                desi_root, 'https://data.desi.lbl.gov/desi')
        print(f"  {url}")
    return 0


def parse(options=None):
    """
    Parse command line arguments for desi_prod_dag.

    Args:
        options (list, optional): Arguments to parse instead of sys.argv.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    import argparse
    p = argparse.ArgumentParser(
        description='Build a navigable HTML view of a production job DAG.')
    p.add_argument('-s', '--specprod', type=str, required=False,
                   help='override $SPECPROD, or a full path to a production '
                        'directory')
    p.add_argument('-o', '--output', type=str, required=False,
                   help='output HTML file (default '
                        'run/jobgraph/proddag-SPECPROD.html in the production)')
    p.add_argument('-n', '--nights', type=str, required=False,
                   help='restrict to these nights, comma or space separated')
    p.add_argument('--update-from-queue', action='store_true',
                   help='query Slurm for live job states instead of trusting '
                        'the STATUS column of the processing tables, which is '
                        'only refreshed when proc_night runs. Slow on a full '
                        'production; pair it with --nights')
    return p.parse_args(options)


def _night_from_proctable_name(path):
    """Extract the night from a processing table pathname, or -1"""
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    tail = stem.replace('processing_table_', '')
    ## name is processing_table_<specprod>-<night>
    if '-' in tail:
        tail = tail.rsplit('-', 1)[1]
    try:
        return int(tail)
    except ValueError:
        return -1
