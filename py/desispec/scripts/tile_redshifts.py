
import sys, os, glob
import subprocess
import numpy as np
from astropy.table import Table, vstack

from desiutil.log import get_logger

import desispec.io
from desispec.workflow import batch


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

    night = np.max(exptable['NIGHT'])

    frame_glob = ' '.join(frame_glob)

    # - output directory relative to reduxdir
    if group == 'cumulative':
        outdir = f'tiles/{group}/{tileid}/{night}'
        suffix = f'{tileid}-thru{night}'
    elif group == 'pernight':
        outdir = f'tiles/{group}/{tileid}/{night}'
        suffix = f'{tileid}-{night}'
    elif group == 'perexp':
        outdir = f'tiles/{group}/{tileid}/{expid:08d}'
        suffix = f'{tileid}-exp{expid:08d}'
    elif group == 'pernight-v0':
        outdir = f'tiles/{tileid}/{night}'
        suffix = f'{tileid}-{night}'
    else:
        outdir = f'tiles/{group}/{tileid}'
        suffix = f'{tileid}-{group}'
        log.warning(f'Non-standard tile group={group}; writing outputs to {outdir}/PREFIX-{suffix}.*')

    reduxdir = desispec.io.specprod_root()
    scriptdir = f'{reduxdir}/run/scripts/{outdir}'
    os.makedirs(scriptdir, exist_ok=True)

    batch_config = batch.get_config(system_name)

    jobname = f'redrock-{suffix}'
    batchscript = f'{scriptdir}/coadd-redshifts-{suffix}.slurm'
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