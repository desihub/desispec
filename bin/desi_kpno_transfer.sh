#!/bin/bash
#
# This script is for transferring data that are *not* critical for
# DESI pipeline operations from KPNO to NERSC.  We reserve the term "dts"
# for the script(s) that *do* transfer the critical pipeline data.
#
# Configuration
#
# Source, staging and destination should be in 1-1-1 correspondence.
#
source_directories=(/exposures/desi/sps)
# staging_directories=($(/bin/realpath ${DESI_ROOT}/spectro/staging/raw))
destination_directories=($(/bin/realpath ${DESI_ROOT}/engineering/spectrograph/sps))
n_source=${#source_directories[@]}
# The existence of this file will shut down data transfers.
kill_switch=${HOME}/stop_dts
#
# Functions
#
function sprun {
    echo "$@" >> ${log}
    "$@" >> ${log} 2>&1
    return $?
}
#
#
#
function usage {
    local execName=$(basename $0)
    (
    echo "${execName} [-d] [-h] [-s TIME]"
    echo ""
    echo "Transfer non-critical DESI data from KPNO to NERSC."
    echo ""
    echo "    -d      = Run in daemon mode.  If not specificed, the script will run once and exit."
    echo "    -h      = Print this message and exit."
    echo "    -s TIME = Sleep for TIME between transfers. Only relevant in daemon mode."
    ) >&2
}
#
# Options
#
# Run once and exit.
daemon=/bin/false
# Wait this long before checking for new data.
sleep=24h
while getopts dhs: argname; do
    case ${argname} in
        d) daemon=/bin/true ;;
        h) usage; exit 0 ;;
        s) sleep=${OPTARG} ;;
        *) usage; exit 1 ;;
    esac
done
shift $((OPTIND-1))
#
# Endless loop!
#
while /bin/true; do
    if [[ -f ${kill_switch} ]]; then
        echo "${kill_switch} detected, shutting down transfer script."
        exit 0
    fi
    #
    # Find symlinks at KPNO.
    #
    for (( k=0; k < ${n_source}; k++ )); do
        src=${source_directories[$k]}
        # staging=${staging_directories[$k]}
        dest=${destination_directories[$k]}
        log=${dest}.log
        [[ -f ${log} ]] || /bin/touch ${log}
        /bin/date +'%Y-%m-%dT%H:%M:%S%z' >> ${log}
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
            sprun find ${dest} -type d -exec /bin/chmod 2750 \{\} \;
            sprun find ${dest} -type f -exec /bin/chmod 0440 \{\} \;
            #
            # Verify checksums.
            #
            # if [[ -f ${dest}/checksum.sha256sum ]]; then
            #     (cd ${dest}/ && /bin/sha256sum --quiet --check checksum.sha256sum) &>> ${log}
            #     # TODO: Add error handling.
            # else
            #     echo "WARNING: no checksum file for ${dest}." >> ${log}
            # fi
        # elif [[ "${status}" == "done" ]]; then
            #
            # Do nothing, successfully.
            #
            # :
        else
            echo "ERROR: rsync problem detected!" >> ${log}
        fi
    done
    if ${daemon}; then
        /bin/sleep ${sleep}
    else
        exit 0
    fi
done
