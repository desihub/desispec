#!/bin/bash
#
# Configuration
#
log=${HOME}/desi_dts.log
[[ -f ${log} ]] && /bin/touch ${log}
#
# Source, staging, destination and hpss should be in 1-1-1-1 correspondence.
#
source_directories=(/data/dts/exposures/raw)
# source_directories=(/data/dts/exposures/test)
staging_directories=($(/bin/realpath ${DESI_ROOT}/spectro/staging/raw))
# staging_directories=($(/bin/realpath ${CSCRATCH}/desi/spectro/staging/raw))
destination_directories=($(/bin/realpath ${DESI_SPECTRO_DATA}))
# destination_directories=($(/bin/realpath ${CSCRATCH}/desi/spectro/data))
hpss_directories=(desi/spectro/data)
n_source=${#source_directories[@]}
# Enable activation of the DESI pipeline.  If this is /bin/false, only
# transfer files.
run_pipeline=/bin/true
pipeline_host=edison
# The existence of this file will shut down data transfers.
kill_switch=${HOME}/stop_dts
# Call this executable on the pipeline host.
desi_night=$(/bin/realpath ${HOME}/bin/wrap_desi_night.sh)
ssh="/bin/ssh -q ${pipeline_host}"
# Wait this long before checking for new data.
sleep=10m
# sleep=1m
# UTC time in hours to trigger HPSS backups.
backup_time=20
#
# Functions
#
function sprun {
    echo "DEBUG: $@" >> ${log}
    "$@" >> ${log} 2>&1
    return $?
}
#
# Endless loop!
#
while /bin/true; do
    echo "INFO:" $(/bin/date +'%Y-%m-%dT%H:%M:%S%z') >> ${log}
    if [[ -f ${kill_switch} ]]; then
        echo "INFO: ${kill_switch} detected, shutting down transfer daemon." >> ${log}
        exit 0
    fi
    #
    # Find symlinks at KPNO.
    #
    for (( k=0; k < ${n_source}; k++ )); do
        src=${source_directories[$k]}
        staging=${staging_directories[$k]}
        dest=${destination_directories[$k]}
        status_dir=$(/bin/dirname ${staging})/status
        links=$(/bin/ssh -q dts /bin/find ${src} -type l 2>/dev/null | sort)
        if [[ -n "${links}" ]]; then
            for l in ${links}; do
                exposure=$(/bin/basename ${l})
                night=$(/bin/basename $(/bin/dirname ${l}))
                #
                # New night detected?
                #
                [[ ! -d ${staging}/${night} ]] && \
                    sprun /bin/mkdir -p ${staging}/${night}
                #
                # Has exposure already been transferred?
                #
                if [[ ! -d ${staging}/${night}/${exposure} && \
                    ! -d ${dest}/${night}/${exposure} ]]; then
                    sprun /bin/rsync --verbose --no-motd \
                        --recursive --copy-dirlinks --times --omit-dir-times \
                        dts:${src}/${night}/${exposure}/ ${staging}/${night}/${exposure}/
                    status=$?
                else
                    echo "INFO: ${staging}/${night}/${exposure} already transferred." >> ${log}
                    status="done"
                fi
                #
                # Transfer complete.
                #
                if [[ "${status}" == "0" ]]; then
                    #
                    # Check permissions.
                    #
                    sprun /bin/chmod 2750 ${staging}/${night}/${exposure}
                    for f in ${staging}/${night}/${exposure}/*; do
                        sprun /bin/chmod 0440 ${f}
                    done
                    #
                    # Verify checksums.
                    #
                    if [[ -f ${staging}/${night}/${exposure}/checksum-${night}-${exposure}.sha256sum ]]; then
                        (cd ${staging}/${night}/${exposure} && /bin/sha256sum --quiet --check checksum-${night}-${exposure}.sha256sum) &>> ${log}
                        checksum_status=$?
                    else
                        echo "WARNING: no checksum file for ${night}/${exposure}." >> ${log}
                        checksum_status=0
                    fi
                    #
                    # Did we pass checksums?
                    #
                    if [[ "${checksum_status}" == "0" ]]; then
                        #
                        # Set up DESI_SPECTRO_DATA.
                        #
                        [[ ! -d ${dest}/${night} ]] && \
                            sprun /bin/mkdir -p ${dest}/${night}
                        #
                        # Move data into DESI_SPECTRO_DATA.
                        #
                        [[ ! -d ${dest}/${night}/${exposure} ]] && \
                            sprun /bin/mv ${staging}/${night}/${exposure} ${dest}/${night}
                        #
                        # Is this a "realistic" exposure?
                        #
                        if ${run_pipeline} && \
                            [[ -f ${dest}/${night}/${exposure}/desi-${exposure}.fits.fz && \
                               -f ${dest}/${night}/${exposure}/fibermap-${exposure}.fits ]]; then
                            #
                            # Run update
                            #
                            sprun ${ssh} ${desi_night} update \
                                --night ${night} --expid ${exposure} \
                                --nersc ${pipeline_host} --nersc_queue realtime \
                                --nersc_maxnodes 25
                            #
                            # if (flat|arc) done, run flat|arc update.
                            #
                            if [[ -f ${dest}/${night}/${exposure}/flats-${night}-${exposure}.done ]]; then
                                sprun ${ssh} ${desi_night} flats \
                                    --night ${night} \
                                    --nersc ${pipeline_host} --nersc_queue realtime \
                                    --nersc_maxnodes 25
                                sprun desi_dts_status --directory ${status_dir} --last flats ${night} ${exposure}
                            elif [[ -f ${dest}/${night}/${exposure}/arcs-${night}-${exposure}.done ]]; then
                                sprun ${ssh} ${desi_night} arcs \
                                    --night ${night} \
                                    --nersc ${pipeline_host} --nersc_queue realtime \
                                    --nersc_maxnodes 25
                                sprun desi_dts_status --directory ${status_dir} --last arcs ${night} ${exposure}
                            #
                            # if night done run redshifts
                            #
                            elif [[ -f ${dest}/${night}/${exposure}/science-${night}-${exposure}.done ]]; then
                                sprun ${ssh} ${desi_night} redshifts \
                                    --night ${night} \
                                    --nersc ${pipeline_host} --nersc_queue realtime \
                                    --nersc_maxnodes 25
                                sprun desi_dts_status --directory ${status_dir} --last science ${night} ${exposure}
                            else
                                sprun desi_dts_status --directory ${status_dir} ${night} ${exposure}
                            fi
                        else
                            echo "INFO: ${night}/${exposure} appears to be test data.  Skipping pipeline activation." >> ${log}
                        fi
                    else
                        echo "ERROR: checksum problem detected for ${night}/${exposure}!" >> ${log}
                        sprun desi_dts_status --directory ${status_dir} --failure ${night} ${exposure}
                    fi
                elif [[ "${status}" == "done" ]]; then
                    #
                    # Do nothing, successfully.
                    #
                    :
                else
                    echo "ERROR: rsync problem detected!" >> ${log}
                    sprun desi_dts_status --directory ${status_dir} --failure ${night} ${exposure}
                fi
            done
        else
            echo "WARNING: No links found, check connection." >> ${log}
        fi
        #
        # Are any nights eligible for backup?
        # 12:00 MST = 19:00 UTC.
        # Plus one hour just to be safe, so after 20:00 UTC.
        #
        yesterday=$(/bin/date --date="@$(($(/bin/date +%s) - 86400))" +'%Y%m%d')
        now=$(/bin/date -u +'%H')
        hpss_file=$(echo ${hpss_directories[$k]} | tr '/' '_')
        ls_file=${CSCRATCH}/${hpss_file}.txt
        if (( now >= backup_time )); then
            if [[ -d ${dest}/${yesterday} ]]; then
                sprun /bin/rm -f ${ls_file}
                sprun /usr/common/mss/bin/hsi -O ${ls_file} ls -l ${hpss_directories[$k]}
                #
                # Both a .tar and a .tar.idx file should be present.
                #
                if [[ $(/usr/bin/grep ${yesterday} ${ls_file} | /usr/bin/wc -l) != 2 ]]; then
                    (cd ${dest} && \
                        /usr/common/mss/bin/htar -cvhf \
                            ${hpss_directories[$k]}/${hpss_file}_${yesterday}.tar \
                            -H crc:verify=all ${yesterday}) &>> ${log}
                fi
            else
                echo "WARNING: No data from ${yesterday} detected, skipping HPSS backup." >> ${log}
            fi
        fi
    done
    /bin/sleep ${sleep}
done
