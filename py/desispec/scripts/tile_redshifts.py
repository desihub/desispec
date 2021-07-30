
import sys, os, glob
import re
import subprocess
import numpy as np
from astropy.table import Table, vstack

from desiutil.log import get_logger

import desispec.io
from desispec.workflow.exptable import get_exposure_table_pathname
from desispec.workflow import batch


def parse(options=None):
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("-n", "--night", type=int, nargs='+', help="YEARMMDD nights")
    p.add_argument("-t", "--tileid", type=int, help="Tile ID")
    p.add_argument("-e", "--expid", type=int, nargs='+', help="exposure IDs")
    p.add_argument("-g", "--group", type=str, required=True,
                   help="cumulative, pernight, perexp, or a custom name")
    p.add_argument("--run_zmtl", action="store_true",
                   help="also run make_zmtl_files")
    p.add_argument("--explist", type=str,
                   help="file with columns TILE NIGHT EXPID to use")
    p.add_argument("--nosubmit", action="store_true",
                   help="generate scripts but don't submit batch jobs")
    p.add_argument("--noafterburners", action="store_true",
                   help="Do not run afterburners (like QSO fits)")
    p.add_argument("--batch-queue", type=str, default='realtime',
                   help="batch queue name")
    p.add_argument("--batch-reservation", type=str,
                   help="batch reservation name")
    p.add_argument("--batch-dependency", type=str,
                   help="job dependencies passed to sbatch --dependency")
    p.add_argument("--system-name", type=str,
                   help="batch system name, e.g. cori-haswell, cori-knl, perlmutter-gpu")

    # TODO
    # p.add_argument("--outdir", type=str, help="output directory")
    # p.add_argument("--scriptdir", type=str, help="script directory")
    # p.add_argument("--per-exposure", action="store_true",
    #         help="fit redshifts per exposure instead of grouping")
    if options is None:
        args = p.parse_args()
    else:
        args = p.parse_args(options)

    return args

def main(args):
    batch_scripts, failed_jobs = generate_tile_redshift_scripts(**args.__dict__)
    num_error = len(failed_jobs)
    sys.exit(num_error)

def get_tile_redshift_relpath(tileid,group,night=None,expid=None):
    """
    Determine the relative output directory of the tile redshift batch script for spectra+coadd+redshifts for a tile

    Args:
        tileid (int): Tile ID
        group (str): cumulative, pernight, perexp, or a custom name
        night (int): Night
        expid (int): Exposure ID

    Returns:
        outdir (str): the relative path of output directory of the batch script from the specprod/run/scripts
    """
    log = get_logger()
    # - output directory relative to reduxdir
    if group == 'cumulative':
        outdir = f'tiles/{group}/{tileid}/{night}'
    elif group == 'pernight':
        outdir = f'tiles/{group}/{tileid}/{night}'
    elif group == 'perexp':
        outdir = f'tiles/{group}/{tileid}/{expid:08d}'
    elif group == 'pernight-v0':
        outdir = f'tiles/{tileid}/{night}'
    else:
        outdir = f'tiles/{group}/{tileid}'
        log.warning(f'Non-standard tile group={group}; writing outputs to {outdir}/*')
    return outdir

def get_tile_redshift_script_pathname(tileid,group,night=None,expid=None):
    """
    Generate the pathname of the tile redshift batch script for spectra+coadd+redshifts for a tile

    Args:
        tileid (int): Tile ID
        group (str): cumulative, pernight, perexp, or a custom name
        night (int): Night
        expid (int): Exposure ID

    Returns:
        (str): the pathname of the tile redshift batch script
    """
    reduxdir = desispec.io.specprod_root()
    outdir = get_tile_redshift_relpath(tileid,group,night=night,expid=expid)
    scriptdir = f'{reduxdir}/run/scripts/{outdir}'
    suffix = get_tile_redshift_script_suffix(tileid,group,night=night,expid=expid)
    batchscript = f'coadd-redshifts-{suffix}.slurm'
    return os.path.join(scriptdir, batchscript)

def get_tile_redshift_script_suffix(tileid,group,night=None,expid=None):
    """
    Generate the suffix of the tile redshift batch script for spectra+coadd+redshifts for a tile

    Args:
        tileid (int): Tile ID
        group (str): cumulative, pernight, perexp, or a custom name
        night (int): Night
        expid (int): Exposure ID

    Returns:
        suffix (str): the suffix of the batch script
    """
    log = get_logger()
    if group == 'cumulative':
        suffix = f'{tileid}-thru{night}'
    elif group == 'pernight':
        suffix = f'{tileid}-{night}'
    elif group == 'perexp':
        suffix = f'{tileid}-exp{expid:08d}'
    elif group == 'pernight-v0':
        suffix = f'{tileid}-{night}'
    else:
        suffix = f'{tileid}-{group}'
        log.warning(f'Non-standard tile group={group}; writing outputs to {suffix}.*')
    return suffix

def batch_tile_redshifts(tileid, exptable, group, spectrographs=None,
                         submit=False, queue='realtime', reservation=None,
                         dependency=None, system_name=None, run_zmtl=False,
                         noafterburners=False):
    """
    Generate batch script for spectra+coadd+redshifts for a tile

    Args:
        tileid (int): Tile ID
        exptable (Table): has columns NIGHT EXPID to use; ignores other columns.
            Doesn't need to be full pipeline exposures table (but could be)
        group (str): cumulative, pernight, perexp, or a custom name

    Options:
        spectrographs (list of int): spectrographs to include
        submit (bool): also submit batch script to queue
        queue (str): batch queue name
        reservation (str): batch reservation name
        dependency (str): passed to sbatch --dependency upon submit
        system_name (str): batch system name, e.g. cori-haswell, perlmutter-gpu
        run_zmtl (bool): if True, also run make_zmtl_files
        noafterburners (bool): if True, do not run QSO afterburners

    Returns tuple (scriptpath, error):
        scriptpath (str): full path to generated script
        err (int): return code from submitting job (0 if submit=False)

    By default this generates the script but don't submit it
    """
    log = get_logger()
    if spectrographs is None:
        spectrographs = (0,1,2,3,4,5,6,7,8,9)

    if (group == 'perexp') and len(exptable)>1:
        msg = f'group=perexp requires 1 exptable row, not {len(exptable)}'
        log.error(msg)
        raise ValueError(msg)

    nights = np.unique(np.asarray(exptable['NIGHT']))
    if (group in ['pernight', 'pernight-v0']) and len(nights)>1:
        msg = f'group=pernight requires all exptable rows to be same night, not {nights}'
        log.error(msg)
        raise ValueError(msg)

    tileids = np.unique(np.asarray(exptable['TILEID']))
    if len(tileids)>1:
        msg = f'batch_tile_redshifts requires all exptable rows to be same tileid, not {tileids}'
        log.error(msg)
        raise ValueError(msg)
    elif len(tileids) == 1 and tileids[0] != tileid:
        msg = f'Specified tileid={tileid} didnt match tileid given in exptable, {tileids}'
        log.error(msg)
        raise ValueError(msg)

    spectro_string = ' '.join([str(sp) for sp in spectrographs])
    num_nodes = len(spectrographs)

    frame_glob = list()
    for night, expid in zip(exptable['NIGHT'], exptable['EXPID']):
        frame_glob.append(f'exposures/{night}/{expid:08d}/cframe-[brz]$SPECTRO-{expid:08d}.fits')

    #- Be explicit about naming. Night should be the most recent Night.
    #- Expid only used for labeling perexp, for which there is only one row here anyway
    night = np.max(exptable['NIGHT'])
    expid = np.min(exptable['EXPID'])

    frame_glob = ' '.join(frame_glob)

    batchscript = get_tile_redshift_script_pathname(
            tileid, group, night=night, expid=expid)
    batchlog = batchscript.replace('.slurm', r'-%j.log')

    scriptdir = os.path.split(batchscript)[0]
    os.makedirs(scriptdir, exist_ok=True)

    outdir = get_tile_redshift_relpath(tileid, group, night=night, expid=expid)
    suffix = get_tile_redshift_script_suffix(
            tileid, group, night=night, expid=expid)
    jobname = f'redrock-{suffix}'

    write_redshift_script(batchscript, outdir,
            jobname=jobname,
            num_nodes=num_nodes,
            group=group,
            spectro_string=spectro_string, suffix=suffix,
            frame_glob=frame_glob,
            queue=queue, system_name=system_name,
            onetile=True, tileid=tileid, night=night,
            run_zmtl=run_zmtl, noafterburners=noafterburners)

    err = 0
    if submit:
        cmd = ['sbatch' ,]
        if reservation:
            cmd.extend(['--reservation', reservation])
        if dependency:
            cmd.extend(['--dependency', dependency])

        # - sbatch requires the script to be last, after all options
        cmd.append(batchscript)

        err = subprocess.call(cmd)
        basename = os.path.basename(batchscript)
        if err == 0:
            log.info(f'submitted {basename}')
        else:
            log.error(f'Error {err} submitting {basename}')

    return batchscript, err

def write_redshift_script(batchscript, outdir,
        jobname, num_nodes,
        group, spectro_string, suffix, frame_glob,
        queue='regular', system_name=None,
        onetile=True, tileid=None, night=None,
        run_zmtl=False, noafterburners=False,
        redrock_nodes=1, redrock_cores_per_rank=1,
        ):
    """
    Write a batch script for running coadds, redshifts, and afterburners

    Args:
        batchscript (str): filepath to batch script to write
        outdir (str): output directory to write data
        jobname (str): slurm job name
        num_nodes (int): number of nodes to allocate
        group (str): used for tile redshifts, e.g. 'cumulative'
        spectro_string (str): e.g. '0 1 2 3' spectrographs to run
        suffix (str): filename suffix (e.g. TILEID-thruNIGHT)
        frame_glob (str): glob for finding input cframes

    Options:
        queue (str): queue name
        system_name (str): e.g. cori-haswell, cori-knl, perlmutter-gpu
        onetile (bool): coadd assuming input is for a single tile?
        tileid (int): tileid to process; only needed for group='cumulative'
        night (int): process through night YEARMMDD; only needed for group='cumulative'
        run_zmtl (bool): if True, also run zmtl
        noafterburners (bool): if True, skip QSO afterburners
        redrock_nodes (int): number of nodes for each redrock call
        redrock_cores_per_rank (int): number of cores/rank to use for redrock

    Note: some of these options are hacked to also be used by healpix_redshifts,
    e.g. by providing spectro_string='sv3' instead of list of spectrographs.

    Note: Use redrock_cores_per_rank > 1 to reserve extra memory per rank
    for large input coadd files (e.g. sv3 healpix).
    """
    log = get_logger()

    if redrock_nodes > num_nodes:
        msg = f'redrock_nodes ({redrock_nodes}) cannot be larger than job num_nodes ({num_nodes})'
        log.error(msg)
        raise ValueError(msg)

    batch_config = batch.get_config(system_name)

    batchlog = batchscript.replace('.slurm', r'-%j.log')

    #- tileid and night are required for cumulative redshifts but not others
    #- (frameglob captures the info for other cases)
    if group == 'cumulative':
        err = False
        if tileid is None:
            log.error("group='cumulative' requires tileid to be set")
            err = True
        if night is None:
            log.error("group='cumulative' requires night to be set")
            err = True
        if err:
            raise ValueError("group='cumulative' missing tileid and/or night")

    if onetile:
        onetileopt = '--onetile'
    else:
        onetileopt = ''

    #- system specific options, e.g. "--constraint=haswell"
    batch_opts = list()
    if 'batch_opts' in batch_config:
        for opt in batch_config['batch_opts']:
            batch_opts.append(f'#SBATCH {opt}')
    batch_opts = '\n'.join(batch_opts)

    runtime = 10 + int(10 * batch_config['timefactor'] * redrock_cores_per_rank)
    runtime_hh = runtime // 60
    runtime_mm = runtime % 60

    cores_per_node = batch_config['cores_per_node']
    threads_per_core = batch_config['threads_per_core']
    threads_per_node = cores_per_node * threads_per_core

    logdir = os.path.join(outdir, 'logs')

    with open(batchscript, 'w') as fx:
        fx.write(f"""#!/bin/bash

#SBATCH -N {num_nodes}
#SBATCH --account desi
#SBATCH --qos {queue}
#SBATCH --job-name {jobname}
#SBATCH --output {batchlog}
#SBATCH --time={runtime_hh:02d}:{runtime_mm:02d}:00
#SBATCH --exclusive
{batch_opts}

echo --- Starting at $(date)
START_TIME=$SECONDS

pushd $DESI_SPECTRO_REDUX/$SPECPROD
mkdir -p {outdir}
mkdir -p {logdir}

echo
echo --- Generating files in $(pwd)/{outdir}
echo""")

        if frame_glob is not None:
            fx.write(f"""
echo --- Grouping frames to spectra at $(date)
for SPECTRO in {spectro_string}; do
    spectra={outdir}/spectra-$SPECTRO-{suffix}.fits
    splog={logdir}/spectra-$SPECTRO-{suffix}.log

    if [ -f $spectra ]; then
        echo $(basename $spectra) already exists, skipping grouping
    else
        # Check if any input frames exist
        CFRAMES=$(ls {frame_glob})
        MISSING_CFRAMES=$?
        NUM_CFRAMES=$(echo $CFRAMES | wc -w)
        if [ $MISSING_CFRAMES -ne 0 ] && [ $NUM_CFRAMES -gt 0 ]; then
            echo ERROR: some expected cframes missing for spectrograph $SPECTRO but proceeding anyway
        fi
        if [ $NUM_CFRAMES -gt 0 ]; then
            echo Grouping $NUM_CFRAMES cframes into $(basename $spectra), see $splog
            cmd="srun -N 1 -n 1 -c {threads_per_node} desi_group_spectra --inframes $CFRAMES --outfile $spectra"
            echo RUNNING $cmd &> $splog
            $cmd &>> $splog &
            sleep 0.5
        else
            echo ERROR: no input cframes for spectrograph $SPECTRO, skipping
        fi
    fi
done
echo Waiting for desi_group_spectra to finish at $(date)
wait
""")

        fx.write(f"""
echo
echo --- Coadding spectra at $(date)
for SPECTRO in {spectro_string}; do
    spectra={outdir}/spectra-$SPECTRO-{suffix}.fits
    coadd={outdir}/coadd-$SPECTRO-{suffix}.fits
    colog={logdir}/coadd-$SPECTRO-{suffix}.log

    if [ -f $coadd ]; then
        echo $(basename $coadd) already exists, skipping coadd
    elif [ -f $spectra ]; then
        echo Coadding $(basename $spectra) into $(basename $coadd), see $colog
        cmd="srun -N 1 -n 1 -c {threads_per_node} desi_coadd_spectra {onetileopt} --nproc 16 -i $spectra -o $coadd"
        echo RUNNING $cmd &> $colog
        $cmd &>> $colog &
        sleep 0.5
    else
        echo ERROR: missing $(basename $spectra), skipping coadd
    fi
done
echo Waiting for desi_coadd_spectra to finish at $(date)
wait
""")

        fx.write(f"""
echo
echo --- Running redrock at $(date)
echo Using {redrock_nodes} nodes per redrock call
echo Using {redrock_cores_per_rank} cores per rank for redrock
for SPECTRO in {spectro_string}; do
    coadd={outdir}/coadd-$SPECTRO-{suffix}.fits
    redrock={outdir}/redrock-$SPECTRO-{suffix}.fits
    rrdetails={outdir}/rrdetails-$SPECTRO-{suffix}.h5
    rrlog={logdir}/redrock-$SPECTRO-{suffix}.log

    if [ -f $redrock ]; then
        echo $(basename $redrock) already exists, skipping redshifts
    elif [ -f $coadd ]; then
        echo Running redrock on $(basename $coadd), see $rrlog
        cmd="srun -N {redrock_nodes} -n {cores_per_node*redrock_nodes//redrock_cores_per_rank} -c {threads_per_core*redrock_cores_per_rank} rrdesi_mpi -i $coadd -o $redrock -d $rrdetails"
        echo RUNNING $cmd &> $rrlog
        $cmd &>> $rrlog &
        sleep 0.5
    else
        echo ERROR: missing $(basename $coadd), skipping redshifts
    fi
done
echo Waiting for redrock to finish at $(date)
wait
""")

        if group == 'cumulative':
            fx.write(f"""
echo
tileqa={outdir}/tile-qa-{suffix}.fits
if [ -f $tileqa ]; then
    echo --- $(basename $tileqa) already exists, skipping desi_tile_qa
else
    echo --- Running desi_tile_qa
    tile_qa_log={logdir}/tile-qa-{tileid}-thru{night}.log
    desi_tile_qa -n {night} -t {tileid} &> $tile_qa_log
fi
""")

        if run_zmtl:
            fx.write(f"""
# These run fast; use a single node for all 10 petals without srun overhead
echo
echo --- Running make_zmtl_files at $(date)
for SPECTRO in {spectro_string}; do
    coadd={outdir}/coadd-$SPECTRO-{suffix}.fits
    redrock={outdir}/redrock-$SPECTRO-{suffix}.fits
    zmtl={outdir}/zmtl-$SPECTRO-{suffix}.fits
    zmtllog={logdir}/zmtl-$SPECTRO-{suffix}.log

    if [ -f $zmtl ]; then
        echo $(basename $zmtl) already exists, skipping make_zmtl_files
    elif [[ -f $coadd && -f $redrock ]]; then
        echo Running make_zmtl_files on $(basename $redrock), see $zmtllog
        cmd="make_zmtl_files -in $redrock -out $zmtl"
        echo RUNNING $cmd &> $zmtllog
        $cmd &>> $zmtllog &
    else
        echo ERROR: missing $(basename $redrock) or $(basename $coadd), skipping zmtl
    fi
done
echo Waiting for zmtl to finish at $(date)
wait
""")

        if not noafterburners:
            fx.write(f"""
echo
echo --- Running QSO afterburners at $(date)
for SPECTRO in {spectro_string}; do
    coadd={outdir}/coadd-$SPECTRO-{suffix}.fits
    redrock={outdir}/redrock-$SPECTRO-{suffix}.fits
    qsomgii={outdir}/qso_mgii-$SPECTRO-{suffix}.fits
    qsoqn={outdir}/qso_qn-$SPECTRO-{suffix}.fits
    qsomgiilog={logdir}/qso_mgii-$SPECTRO-{suffix}.log
    qsoqnlog={logdir}/qso_qn-$SPECTRO-{suffix}.log

    # QSO MgII afterburner
    if [ -f $qsomgii ]; then
        echo $(basename $qsomgii) already exists, skipping QSO MgII afterburner
    elif [ -f $redrock ]; then
        echo Running QSO MgII afterburner, see $qsomgiilog
        cmd="srun -N 1 -n 1 -c {threads_per_node} desi_qso_mgii_afterburner --coadd $coadd --redrock $redrock --output $qsomgii --target_selection qso --save_target all"
        echo RUNNING $cmd &> $qsomgiilog
        $cmd &>> $qsomgiilog &
        sleep 0.5
    else
        echo ERROR: missing $(basename $redrock), skipping QSO MgII afterburner
    fi

    # QSO QuasarNet (QN) afterburner
    if [ -f $qsoqn ]; then
        echo $(basename $qsoqn) already exists, skipping QSO QuasarNet afterburner
    elif [ -f $redrock ]; then
        echo Running QSO QuasarNet afterburner, see $qsoqnlog
        cmd="srun -N 1 -n 1 -c {threads_per_node} desi_qso_qn_afterburner --coadd $coadd --redrock $redrock --output $qsoqn --target_selection qso --save_target all"
        echo RUNNING $cmd &> $qsoqnlog
        $cmd &>> $qsoqnlog &
        sleep 0.5
    else
        echo ERROR: missing $(basename $redrock), skipping QSO QN afterburner
    fi

done
echo Waiting for QSO afterburners to finish at $(date)
wait
""")

        fx.write(f"""
echo
echo --- Files in {outdir}:
for prefix in spectra coadd redrock zmtl qso_qn qso_mgii tile-qa; do
    echo  "   " $(ls {outdir}/$prefix*.fits | wc -l) $prefix
done

popd &> /dev/null

END_TIME=$SECONDS
DURATION_MINUTES=$(( ($END_TIME - $START_TIME)/60 ))
DURATION_SECONDS=$(( ($END_TIME - $START_TIME)%60 ))

echo
echo --- Done at $(date) in ${{DURATION_MINUTES}}m${{DURATION_SECONDS}}s
        """)

    log.info(f'Wrote {batchscript}')


def _read_minimal_exptables(nights=None):
    """
    Read exposure tables while handling evolving formats

    Args:
        nights (list of int): nights to include (default all nights found)

    Returns exptable with just columns TILEID, NIGHT, EXPID filtered by science
        exposures with LASTSTEP='all' and TILEID>=0

    Note: the returned table is *not* the full pipeline exposures table because
        the format of that changed during SV1 and thus can't be stacked without
        trimming down the columns.  This trims to just the minimal columns
        needed by desi_tile_redshifts.
    """
    log = get_logger()
    if nights is None:
        reduxdir = desispec.io.specprod_root()
        etab_files = glob.glob(f'{reduxdir}/exposure_tables/202???/exposure_table_202?????.csv')
    else:
        etab_files = list()
        for night in nights:
            etab_file = get_exposure_table_pathname(night)
            if os.path.exists(etab_file):
                etab_files.append(etab_file)
            elif night >= 20201201:
                log.error(f"Exposure table missing for night {night}")
            else:
                # - these are expected for the daily run, ok
                log.debug(f"Exposure table missing for night {night}")

    etab_files = sorted(etab_files)
    exptables = list()
    for etab_file in etab_files:
        t = Table.read(etab_file)
        keep = (t['OBSTYPE'] == 'science') & (t['TILEID'] >= 0)
        if 'LASTSTEP' in t.colnames:
            keep &= (t['LASTSTEP'] == 'all')
        t = t[keep]
        exptables.append(t['TILEID', 'NIGHT', 'EXPID'])

    return vstack(exptables)


def generate_tile_redshift_scripts(group, night=None, tileid=None, expid=None, explist=None,
                                   run_zmtl=False, noafterburners=False,
                                   batch_queue='realtime', batch_reservation=None,
                                   batch_dependency=None, system_name=None, nosubmit=False):
    """
    Creates a slurm script to run redshifts per tile. By default it also submits the job to Slurm. If nosubmit
    is True, the script is created but not submitted to Slurm.

    Args:
        group (str): Type of coadd redshifts to run. Options are cumulative, pernight, perexp, or a custom name.
        night (int, or list or np.array of int's): YEARMMDD nights.
        tileid (int): Tile ID.
        expid (int, or list or np.array of int's): Exposure IDs.
        explist (str): File with columns TILE NIGHT EXPID to use
        run_zmtl (bool): If True, also run make_zmtl_files
        noafterburners (bool): If True, do not run QSO afterburners
        batch_queue (str): Batch queue name. Default is 'realtime'.
        batch_reservation (str): Batch reservation name.
        batch_dependency (str): Job dependencies passed to sbatch --dependency .
        system_name (str): Batch system name, e.g. cori-haswell, cori-knl, perlmutter-gpu.
        nosubmit (bool): Generate scripts but don't submit batch jobs. Default is False.

    Returns:
        batch_scripts (list of str): The path names of the scripts created during the function call
                                     that returned a null batcherr.
        failed_jobs (list of str): The path names of the scripts created during the function call
                                   that returned a batcherr.
    """
    log = get_logger()
    
    # - If --tileid, --night, and --expid are all given, create exptable
    if ((tileid is not None) and (night is not None) and
            (len(night) == 1) and (expid is not None)):
        log.info('Creating exposure table from --tileid --night --expid options')
        exptable = Table()
        exptable['EXPID'] = expid
        exptable['NIGHT'] = night[0]
        exptable['TILEID'] = tileid
    
        if explist is not None:
            log.warning('Ignoring --explist, using --tileid --night --expid')
    
    # - otherwise load exposure tables for those nights
    elif explist is None:
        if night is not None:
            log.info(f'Loading production exposure tables for nights {night}')
        else:
            log.info(f'Loading production exposure tables for all nights')
    
        exptable = _read_minimal_exptables(night)
    
    else:
        log.info(f'Loading exposure list from {explist}')
        if explist.endswith('.fits'):
            exptable = Table.read(explist, format='fits')
        elif explist.endswith('.csv'):
            exptable = Table.read(explist, format='ascii.csv')
        elif explist.endswith('.ecsv'):
            exptable = Table.read(explist, format='ascii.ecsv')
        else:
            exptable = Table.read(explist, format='ascii')
    
        if night is not None:
            keep = np.in1d(exptable['NIGHT'], night)
            exptable = exptable[keep]
    
    # - Filter exposure tables by exposure IDs or by tileid
    # - Note: If exptable was created from --expid --night --tileid these should
    # - have no effect, but are left in for code flow simplicity
    if expid is not None:
        keep = np.in1d(exptable['EXPID'], expid)
        exptable = exptable[keep]
        #expids = np.array(exptable['EXPID'])
        tileids = np.unique(np.array(exptable['TILEID']))
    
        # - if provided, tileid should be redundant with the tiles in those exps
        if tileid is not None:
            if not np.all(exptable['TILEID'] == tileid):
                log.critical(f'Exposure TILEIDs={tileids} != --tileid={tileid}')
                sys.exit(1)
    
    elif tileid is not None:
        keep = (exptable['TILEID'] == tileid)
        exptable = exptable[keep]
        #expids = np.array(exptable['EXPID'])
        tileids = np.array([tileid, ])
    
    else:
        tileids = np.unique(np.array(exptable['TILEID']))
    
    # - anything left?
    if len(exptable) == 0:
        log.critical(f'No exposures left after filtering by tileid/night/expid')
        sys.exit(1)
    
    # - If cumulative, find all prior exposures that also observed these tiles
    # - NOTE: depending upon options, this might re-read all the exptables again
    # - NOTE: this may not scale well several years into the survey
    if group == 'cumulative':
        log.info(f'{len(tileids)} tiles; searching for exposures on prior nights')
        allexp = _read_minimal_exptables()
        keep = np.in1d(allexp['TILEID'], tileids)
        exptable = allexp[keep]
        ## Ensure we only include data for nights up to and including specified nights
        if (night is not None):
            lastnight = int(np.max(night))
            exptable = exptable[exptable['NIGHT'] <= lastnight]
        #expids = np.array(exptable['EXPID'])
        tileids = np.unique(np.array(exptable['TILEID']))

    # - Generate the scripts and optionally submit them
    failed_jobs, batch_scripts = list(), list()

    for tileid in tileids:
        tilerows = (exptable['TILEID'] == tileid)
        nights = np.unique(np.array(exptable['NIGHT'][tilerows]))
        expids = np.unique(np.array(exptable['EXPID'][tilerows]))
        log.info(f'Tile {tileid} nights={nights} expids={expids}')
        submit = (not nosubmit)
        opts = dict(
                submit=submit,
                run_zmtl=run_zmtl,
                noafterburners=noafterburners,
                queue=batch_queue,
                reservation=batch_reservation,
                dependency=batch_dependency,
                system_name=system_name,
            )
        if group == 'perexp':
            for i in range(len(exptable[tilerows])):
                batchscript, batcherr = batch_tile_redshifts(
                    tileid, exptable[tilerows][i:i + 1], group, **opts)
        elif group in ['pernight', 'pernight-v0']:
            for night in nights:
                thisnight = exptable['NIGHT'] == night
                batchscript, batcherr = batch_tile_redshifts(
                    tileid, exptable[tilerows & thisnight], group, **opts)
        else:
            batchscript, batcherr = batch_tile_redshifts(
                tileid, exptable[tilerows], group, **opts)
        if batcherr != 0:
            failed_jobs.append(batchscript)
        else:
            batch_scripts.append(batchscript)

    #- Report num_error but don't sys.exit for pipeline workflow needs, do that at script level
    num_error = len(failed_jobs)
    if num_error > 0:
        tmp = [os.path.basename(filename) for filename in failed_jobs]
        log.error(f'problem submitting {num_error} scripts: {tmp}')

    #- Return batch_scripts for use in pipeline and failed_jobs for explicit exit code in script
    return batch_scripts, failed_jobs
