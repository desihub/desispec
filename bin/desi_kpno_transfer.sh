#!/bin/bash
#
# This script is for transferring data that are *not* critical for
# DESI pipeline operations from KPNO to NERSC.  We reserve the term "dts"
# for the script(s) that *do* transfer the critical pipeline data.
#
# Configuration
#
log=${HOME}/desi_kpno_transfer.log
[[ -f ${log} ]] && /bin/touch ${log}
#
# Source, staging and destination should be in 1-1-1 correspondence.
#
source_directories=(/data/dts/exposures/raw)
# staging_directories=($(/bin/realpath ${DESI_ROOT}/spectro/staging/raw))
destination_directories=($(/bin/realpath ${DESI_SPECTRO_DATA}))
n_source=${#source_directories[@]}
# The existence of this file will shut down data transfers.
kill_switch=${HOME}/stop_dts
# Wait this long before checking for new data.
sleep=10m
#
# Functions
#
function sprun {
    echo "$@" >> ${log}
    "$@" >> ${log} 2>&1
    return $?
}
#
# Endless loop!
#
while /bin/true; do
    /bin/date +'%Y-%m-%dT%H:%M:%S%z' >> ${log}
    if [[ -f ${kill_switch} ]]; then
        echo "${kill_switch} detected, shutting down transfer script." >> ${log}
        exit 0
    fi
    #
    # Find symlinks at KPNO.
    #
    for (( k=0; k < ${n_source}; k++ )); do
        src=${source_directories[$k]}
        # staging=${staging_directories[$k]}
        dest=${destination_directories[$k]}
        sprun /bin/rsync --verbose --no-motd \
            --recursive --copy-dirlinks --times --omit-dir-times \
            dts:${src}/ ${dest}/
        status=$?
        #
        # Transfer complete.
        #
        if [[ "${status}" == "0" ]]; then
            #
            # Check permissions.
            #
            sprun /bin/chmod 2750 ${dest}/
            for f in ${dest}/*; do
                sprun /bin/chmod 0440 ${f}
            done
            #
            # Verify checksums.
            #
            if [[ -f ${dest}/checksum.sha256sum ]]; then
                (cd ${dest}/ && /bin/sha256sum --quiet --check checksum.sha256sum) &>> ${log}
                # TODO: Add error handling.
            else
                echo "WARNING: no checksum file for ${dest}." >> ${log}
            fi
        elif [[ "${status}" == "done" ]]; then
            #
            # Do nothing, successfully.
            #
            :
        else
            echo "ERROR: rsync problem detected!" >> ${log}
        fi
    done
    /bin/sleep ${sleep}
done
