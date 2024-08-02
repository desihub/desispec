"""
desispec.scripts.tile_redshifts
===============================

"""
import sys, os, glob
import re
import subprocess
import argparse
import numpy as np
from astropy.table import Table, vstack

from desispec.io.util import parse_cameras
from desispec.workflow.redshifts import create_desi_zproc_batch_script
from desispec.workflow.exptable import read_minimal_science_exptab_cols
from desiutil.log import get_logger

from desispec.workflow import batch

def parse(options=None):

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--nights", type=int, nargs='+', help="YEARMMDD nights")
    parser.add_argument("-t", "--tileid", type=int, help="Tile ID")
    parser.add_argument("-e", "--expids", type=int, nargs='+', help="exposure IDs")
    #parser.add_argument("-s", "--spectrographs", type=str,
    #        help="spectrographs to include, e.g. 0-4,9; includes final number in range")
    parser.add_argument("-g", "--group", type=str, required=True,
                   help="cumulative, pernight, perexp, or a custom name")
    parser.add_argument("--run_zmtl", action="store_true",
                   help="also run make_zmtl_files")
    parser.add_argument("--explist", type=str,
                   help="file with columns TILE NIGHT EXPID to use")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Don't use gpus")
    parser.add_argument("--max-gpuprocs", type=int, default=4,
                        help="Number of GPU prcocesses per node")
    parser.add_argument("--nosubmit", action="store_true",
                   help="generate scripts but don't submit batch jobs")
    parser.add_argument("--no-afterburners", action="store_true",
                   help="Do not run afterburners (like QSO fits)")
    parser.add_argument("--batch-queue", type=str, default='regular',
                   help="batch queue name")
    parser.add_argument("--batch-reservation", type=str,
                   help="batch reservation name")
    parser.add_argument("--batch-dependency", type=str,
                   help="job dependencies passed to sbatch --dependency")
    parser.add_argument("--system-name", type=str, default=batch.default_system(),
                   help="batch system name, e.g. cori-haswell, cori-knl, perlmutter-gpu")

    args = parser.parse_args(options)

    return args

def main(args=None):
    if not isinstance(args, argparse.Namespace):
        args = parse(args)

    batch_scripts, failed_jobs = generate_tile_redshift_scripts(**args.__dict__)
    num_error = len(failed_jobs)
    sys.exit(num_error)

def generate_tile_redshift_scripts(group, nights=None, tileid=None, expids=None, explist=None,
                                   camword=None, max_gpuprocs=None, no_gpu=False,
                                   run_zmtl=False, no_afterburners=False,
                                   batch_queue='regular', batch_reservation=None,
                                   batch_dependency=None, system_name=None, nosubmit=False):
    """
    Creates a slurm script to run redshifts per tile. By default it also submits the job to Slurm. If nosubmit
    is True, the script is created but not submitted to Slurm.

    Args:
        group (str): Type of coadd redshifts to run. Options are cumulative, pernight, perexp, or a custom name.
        nights (int, or list or np.array of int's): YEARMMDD nights.
        tileid (int): Tile ID.
        expids (int, or list or np.array of int's): Exposure IDs.
        explist (str): File with columns TILE NIGHT EXPID to use
        camword (str): camword of cameras to include
        max_gpuprocs (int): Number of gpu processes
        no_gpu (bool): Default false. If true it doesn't use GPU's even if available.
        run_zmtl (bool): If True, also run make_zmtl_files
        no_afterburners (bool): If True, do not run QSO afterburners
        batch_queue (str): Batch queue name. Default is 'regular'.
        batch_reservation (str): Batch reservation name.
        batch_dependency (str): Job dependencies passed to sbatch --dependency .
        system_name (str): Batch system name, e.g. cori-haswell, cori-knl, perlmutter-gpu.
        nosubmit (bool): Generate scripts but don't submit batch jobs. Default is False.

    Returns:
        batch_scripts (list of str): The path names of the scripts created during the function call
                                     that returned a null batcherr.
        failed_jobs (list of str): The path names of the scripts created during the function call
                                   that returned a batcherr.

    Note: specify ``spectrographs`` or ``camword`` but not both
    """
    log = get_logger()

    # - If --tileid, --nights, and --expids are all given, create exptable
    if ((tileid is not None) and (nights is not None) and
            (len(nights) == 1) and (expids is not None)):
        log.info('Creating exposure table from --tileid --nights --expids options')
        exptable = Table()
        exptable['EXPID'] = expids
        exptable['NIGHT'] = nights[0]
        exptable['TILEID'] = tileid

        if explist is not None:
            log.warning('Ignoring --explist, using --tileid --nights --expids')

    # - otherwise load exposure tables for those nights
    elif explist is None:
        if nights is not None:
            log.info(f'Loading production exposure tables for nights {nights}')
        else:
            log.info(f'Loading production exposure tables for all nights')

        exptable = read_minimal_science_exptab_cols(nights)

    else:
        log.info(f'Loading exposure list from {explist}')
        if explist.endswith( ('.fits', '.fits.gz') ):
            exptable = Table.read(explist, format='fits')
        elif explist.endswith('.csv'):
            exptable = Table.read(explist, format='ascii.csv')
        elif explist.endswith('.ecsv'):
            exptable = Table.read(explist, format='ascii.ecsv')
        else:
            exptable = Table.read(explist, format='ascii')

        if nights is not None:
            keep = np.in1d(exptable['NIGHT'], nights)
            exptable = exptable[keep]

    # - Filter exposure tables by exposure IDs or by tileid
    # - Note: If exptable was created from --expids --nights --tileid these should
    # - have no effect, but are left in for code flow simplicity
    if expids is not None:
        keep = np.in1d(exptable['EXPID'], expids)
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
        log.critical(f'No exposures left after filtering by tileid/nights/expids')
        sys.exit(1)

    #if (spectrographs is not None) and (camword is not None):
    #    msg = f'Give {spectrographs=} OR {camword=} but not both'
    #    log.error(msg)
    #    raise ValueError(msg)

    #if spectrographs is not None:
    #    camword = spectro_to_camword(spectrographs)

    if camword is not None:
        if isinstance(camword, str):
            camword = parse_cameras(camword)
    else:
        camword = 'a0123456789'

    # - If cumulative, find all prior exposures that also observed these tiles
    # - NOTE: depending upon options, this might re-read all the exptables again
    # - NOTE: this may not scale well several years into the survey
    if group == 'cumulative':
        if nights is not None:
            lastnight = int(np.max(nights))
        elif exptable is not None:
            lastnight = int(np.max(exptable['NIGHT']))
        else:
            lastnight = None
        log.info(f'{len(tileids)} tiles; searching for exposures on prior nights')
        log.info(f'Reading all exposure_tables from all nights')
        newexptable = read_minimal_science_exptab_cols(tileids=tileids)
        newexptable = newexptable[['EXPID', 'NIGHT', 'TILEID']]

        if exptable is not None:
            expids = exptable['EXPID']
            missing_exps = np.in1d(expids, newexptable['EXPID'], invert=True)
            if np.any(missing_exps):
                log.warning(f'Identified {np.sum(missing_exps)} missing exposures '
                            + f'in the exposure cache. Resetting the cache to acquire'
                            + f' them from all nights')
                ## reset_cache will remove cache but it won't be repopulated
                ## unless we request all nights. So let's request all nights
                ## then subselect to the nights we want
                latest_exptable = read_minimal_science_exptab_cols(tileids=tileids,
                                                                   reset_cache=True)
                latest_exptable = latest_exptable[['EXPID', 'NIGHT', 'TILEID']]
                missing_exps = np.in1d(expids, newexptable['EXPID'], invert=True)
                if np.any(missing_exps):
                    log.error(f'Identified {np.sum(missing_exps)} missing exposures '
                        + f'in the exposure cache even after updating. Using the '
                        + f'appending the user provided exposures but this may '
                        + f'indicate a problem.')
                    newexptable = vstack([latest_exptable, exptable[missing_exps]])
                else:
                    newexptable = latest_exptable

        newexptable.sort(['EXPID'])
        exptable = newexptable

        ## Ensure we only include data for nights up to and including specified nights
        if lastnight is not None:
            log.info(f'Selecting only those exposures on nights before or '
                     + f'during the latest night provided: {lastnight}')
            exptable = exptable[exptable['NIGHT'] <= lastnight]

        #expids = np.array(exptable['EXPID'])
        tileids = np.unique(np.array(exptable['TILEID']))

    # - Generate the scripts and optionally submit them
    failed_jobs, batch_scripts = list(), list()

    for tileid in tileids:
        tilerows = (exptable['TILEID'] == tileid)
        inights = np.unique(np.array(exptable['NIGHT'][tilerows]))
        iexpids = np.unique(np.array(exptable['EXPID'][tilerows]))
        log.info(f'Tile {tileid} nights={inights} expids={iexpids}')
        submit = (not nosubmit)
        opts = dict(
                camword=camword,
                submit=submit,
                max_gpuprocs=max_gpuprocs,
                no_gpu=no_gpu,
                run_zmtl=run_zmtl,
                no_afterburners=no_afterburners,
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
            for night in inights:
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


def batch_tile_redshifts(tileid, exptable, group, camword=None,
                         submit=False, queue='regular', reservation=None,
                         max_gpuprocs=None, no_gpu=False,
                         dependency=None, system_name=None, run_zmtl=False,
                         no_afterburners=False):
    """
    Generate batch script for spectra+coadd+redshifts for a tile

    Args:
        tileid (int): Tile ID
        exptable (Table): has columns NIGHT EXPID to use; ignores other columns.
            Doesn't need to be full pipeline exposures table (but could be)
        group (str): cumulative, pernight, perexp, or a custom name

    Options:
        camword (str): camword of cameras to include
        submit (bool): also submit batch script to queue
        queue (str): batch queue name
        reservation (str): batch reservation name
        max_gpuprocs (int): Number of gpu processes
        no_gpu (bool): Default false. If true it doesn't use GPU's even if available.
        dependency (str): passed to sbatch --dependency upon submit
        system_name (str): batch system name, e.g. cori-haswell, perlmutter-gpu
        run_zmtl (bool): if True, also run make_zmtl_files
        no_afterburners (bool): if True, do not run QSO afterburners

    Returns tuple (scriptpath, error):
        scriptpath (str): full path to generated script
        err (int): return code from submitting job (0 if submit=False)

    By default this generates the script but don't submit it
    """
    log = get_logger()
    if camword is None:
        camword = 'a0123456789'

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

    #- Be explicit about naming. Night should be the most recent Night.
    #- Expid only used for labeling perexp, for which there is only one row here anyway
    expids = np.unique(np.asarray(exptable['EXPID']))

    cmdline = ['desi_zproc',
               '-t', str(tileid),
               '-g', group,
               '-n', ' '.join(nights.astype(str)),
               '-e', ' '.join(expids.astype(str)),
               '-c', camword,
               '--mpi']

    if run_zmtl:
        cmdline.append('--run-zmtl')
    if no_afterburners:
        cmdline.append('--no-afterburners')

    scriptfile = create_desi_zproc_batch_script(group=group, tileid=tileid, cameras=camword,
                                                nights=nights, expids=expids,
                                                queue=queue,
                                                cmdline=cmdline,
                                                system_name=system_name,
                                                max_gpuprocs=max_gpuprocs,
                                                no_gpu=no_gpu)

    err = 0
    if submit:
        cmd = ['sbatch' ,]
        if reservation:
            cmd.extend(['--reservation', reservation])
        if dependency:
            cmd.extend(['--dependency', dependency])

        # - sbatch requires the script to be last, after all options
        cmd.append(scriptfile)

        err = subprocess.call(cmd)
        basename = os.path.basename(scriptfile)
        if err == 0:
            log.info(f'Submitted {basename}')
        else:
            log.error(f'Error {err} submitting {basename}')

    return scriptfile, err
