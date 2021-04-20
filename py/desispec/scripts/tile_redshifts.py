
import sys, os, glob
import subprocess
import numpy as np
from astropy.table import Table, vstack

from desiutil.log import get_logger

import desispec.io
from desispec.workflow.tableio import load_table
from desispec.workflow.exptable import get_exposure_table_pathname
from desispec.workflow import batch

def get_tile_redshift_relpath(tileid,group,night=None,expid=None):
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
    reduxdir = desispec.io.specprod_root()
    # - output directory relative to reduxdir
    outdir = get_tile_redshift_relpath(tileid,group,night=night,expid=expid)
    scriptdir = f'{reduxdir}/run/scripts/{outdir}'
    os.makedirs(scriptdir, exist_ok=True)

    suffix = get_tile_redshift_script_suffix(tileid,group,night=night,expid=expid)
    batchscript = f'coadd-redshifts-{suffix}.slurm'
    return os.path.join(scriptdir, batchscript)

def get_tile_redshift_script_suffix(tileid,group,night=None,expid=None):
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
                         submit=False, queue='realtime', reservation=None, dependency=None,
                         system_name=None):
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

    Returns tuple (scriptpath, error):
        scriptpath (str): full path to generated script
        err (int): return code from submitting job (0 if submit=False)

    By default this generates the script but don't submit it
    """
    log = get_logger()
    if spectrographs is None:
        spectrographs = (0 ,1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9)

    if (group == 'perexp') and len(exptable ) >1:
        msg = f'group=perexp requires 1 exptable row, not {len(exptable)}'
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

    batch_config = batch.get_config(system_name)

    outdir = get_tile_redshift_relpath(tileid, group, night=night, expid=expid)
    batchscript = get_tile_redshift_script_pathname(tileid, group, night=night, expid=expid)
    suffix = get_tile_redshift_script_suffix(tileid,group,night=night,expid=expid)
    jobname = f'redrock-{suffix}'
    batchlog = batchscript.replace('.slurm', r'-%j.log')

    # - system specific options, e.g. "--constraint=haswell"
    batch_opts = list()
    if 'batch_opts' in batch_config:
        for opt in batch_config['batch_opts']:
            batch_opts.append(f'#SBATCH {opt}')
    batch_opts = '\n'.join(batch_opts)

    runtime = 10 + int(10 * batch_config['timefactor'])
    runtime_hh = runtime // 60
    runtime_mm = runtime % 60

    cores_per_node = batch_config['cores_per_node']
    threads_per_core = batch_config['threads_per_core']
    threads_per_node = cores_per_node * threads_per_core

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

echo Starting at $(date)

cd $DESI_SPECTRO_REDUX/$SPECPROD
mkdir -p {outdir}
echo Generating files in $(pwd)/tiles/{tileid}/{group}
for SPECTRO in {spectro_string}; do
    spectra={outdir}/spectra-$SPECTRO-{suffix}.fits
    splog={outdir}/spectra-$SPECTRO-{suffix}.log

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
            sleep 1
        else
            echo ERROR: no input cframes for spectrograph $SPECTRO, skipping
        fi
    fi
done
echo Waiting for desi_group_spectra to finish at $(date)
wait

echo Coadding spectra at $(date)
for SPECTRO in {spectro_string}; do
    spectra={outdir}/spectra-$SPECTRO-{suffix}.fits
    coadd={outdir}/coadd-$SPECTRO-{suffix}.fits
    colog={outdir}/coadd-$SPECTRO-{suffix}.log

    if [ -f $coadd ]; then
        echo $(basename $coadd) already exists, skipping coadd
    elif [ -f $spectra ]; then
        echo Coadding $(basename $spectra) into $(basename $coadd), see $colog
        cmd="srun -N 1 -n 1 -c {threads_per_node} desi_coadd_spectra --nproc 16 -i $spectra -o $coadd"
        echo RUNNING $cmd &> $colog
        $cmd &>> $colog &
        sleep 1
    else
        echo ERROR: missing $(basename $spectra), skipping coadd
    fi
done
echo Waiting for desi_coadd_spectra to finish at $(date)
wait

echo Running redrock at $(date)
for SPECTRO in {spectro_string}; do
    spectra={outdir}/spectra-$SPECTRO-{suffix}.fits
    zbest={outdir}/zbest-$SPECTRO-{suffix}.fits
    redrock={outdir}/redrock-$SPECTRO-{suffix}.h5
    rrlog={outdir}/redrock-$SPECTRO-{suffix}.log

    if [ -f $zbest ]; then
        echo $(basename $zbest) already exists, skipping redshifts
    elif [ -f $spectra ]; then
        echo Running redrock on $(basename $spectra), see $rrlog
        cmd="srun -N 1 -n {cores_per_node} -c {threads_per_core} rrdesi_mpi $spectra -o $redrock -z $zbest"
        echo RUNNING $cmd &> $rrlog
        $cmd &>> $rrlog &
        sleep 1
    else
        echo ERROR: missing $(basename $spectra), skipping redshifts
    fi
done
echo Waiting for redrock to finish at $(date)
wait
echo Done at $(date)
""")

    log.info(f'Wrote {batchscript}')

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
        nights = list()
        reduxdir = desispec.io.specprod_root()
        for nightdir in sorted(glob.glob(f'{reduxdir}/exposures/20??????')):
            try:
                n = int(os.path.basename(nightdir))
                nights.append(n)
            except ValueError:
                pass

    exptables = list()
    for night in nights:
        expfile = get_exposure_table_pathname(night)
        if os.path.exists(expfile):
            t = Table.read(expfile)
            keep = (t['OBSTYPE'] == 'science') & (t['TILEID'] >= 0)
            if 'LASTSTEP' in t.colnames:
                keep &= (t['LASTSTEP'] == 'all')

            t = t[keep]
            exptables.append(t['TILEID', 'NIGHT', 'EXPID'])
        elif night >= 20201201:
            log.error(f"Exposure table missing for night {night}")
        else:
            # - these are expected for the daily run, ok
            log.debug(f"Exposure table missing for night {night}")

    return vstack(exptables)


def create_and_submit_tile_redshifts(group, night=None, tileid=None, expid=None, explist=None,
                        batch_queue='realtime', batch_reservation=None,
                        batch_dependency=None, system_name=None, nosubmit=False):
    """
    Creates a slurm script to run redshifts per tile. By default it also submits the job to Slurm. If nosubmit
    is True, the script is created but not submitted to Slurm.

    Args:
        night (int, or list or np.array of int's): YEARMMDD nights.
        tileid (int): Tile ID.
        expid (int, or list or np.array of int's): Exposure IDs.
        group (str): Type of coadd redshifts to run. Options are cumulative, pernight, perexp, or a custom name.
        explist (str): File with columns TILE NIGHT EXPID to use
        nosubmit (bool): Generate scripts but don't submit batch jobs. Default is False.
        batch_queue (str): Batch queue name. Default is 'realtime'.
        batch_reservation (str): Batch reservation name.
        batch_dependency (str): Job dependencies passed to sbatch --dependency .
        system_name (str): Batch system name, e.g. cori-haswell, cori-knl, perlmutter-gpu.

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
        #expids = np.array(exptable['EXPID'])
        tileids = np.unique(np.array(exptable['TILEID']))
    
    # - Generate the scripts and optionally submit them
    failed_jobs, batch_scripts = list(), list()
    if (night is not None):
        lastnight = int(np.max(night))
    for tileid in tileids:
        tilerows = ((exptable['TILEID'] == tileid) & (exptable['NIGHT'] <= lastnight))
        nights = np.unique(np.array(exptable['NIGHT'][tilerows]))
        expids = np.unique(np.array(exptable['EXPID'][tilerows]))
        log.info(f'Tile {tileid} nights={nights} expids={expids}')
        submit = (not nosubmit)
        if group != 'perexp':
            batchscript, batcherr = batch_tile_redshifts(
                tileid, exptable[tilerows], group, submit=submit,
                queue=batch_queue, reservation=batch_reservation,
                dependency=batch_dependency, system_name=system_name
            )
        else:
            for i in range(len(exptable[tilerows])):
                batchscript, batcherr = batch_tile_redshifts(
                    tileid, exptable[tilerows][i:i + 1], group, submit=submit,
                    queue=batch_queue, reservation=batch_reservation,
                    dependency=batch_dependency, system_name=system_name
                )
    
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