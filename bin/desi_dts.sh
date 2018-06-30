#!/bin/bash
#
# Configuration
#
log=${HOME}/desi_dts.log
[[ -f ${log} ]] && /bin/touch ${log}
src=/data/dts/exposures/raw
staging=$(/bin/realpath ${DESI_ROOT}/spectro/staging/raw)
dest=$(/bin/realpath ${DESI_SPECTRO_DATA})
run_pipeline=/bin/false
pipeline_host=edison
ssh="/bin/ssh -q ${pipeline_host}"
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
    #
    # Find symlinks at KPNO.
    #
    links=$(ssh -q dts /bin/find ${src} -type l 2>/dev/null)
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
            if [[ ! -d ${staging}/${night}/${exposure} ]]; then
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
                    # TODO: Add error handling.
                else
                    echo "WARNING: no checksum file for ${night}/${exposure}." >> ${log}
                fi
                #
                # Set up DESI_SPECTRO_DATA.
                #
                [[ ! -d ${dest}/${night} ]] && \
                    sprun /bin/mkdir -p ${dest}/${night}
                #
                # "Copy" data into DESI_SPECTRO_DATA.
                #
                [[ ! -L ${dest}/${night}/${exposure} ]] && \
                    sprun /bin/ln -s ${staging}/${night}/${exposure} ${dest}/${night}/${exposure}
                #
                # Is this a "realistic" exposure?
                #
                if [[ ${run_pipeline} && \
                      -f ${dest}/${night}/${exposure}/desi-${exposure}.fits.fz && \
                      -f ${dest}/${night}/${exposure}/fibermap-${exposure}.fits ]]; then
                    #
                    # Run update
                    #
                    sprun ${ssh} desi_night update \
                        --night ${night} --expid ${exposure} \
                        --nersc ${pipeline_host} --nersc_queue realtime \
                        --nersc_maxnodes 25
                    #
                    # if (flat|arc) done, run flat|arc update.
                    #
                    [[ -f ${dest}/${night}/${exposure}/flats-${night}-${exposure}.done ]] && \
                        sprun ${ssh} desi_night flats \
                            --night ${night} \
                            --nersc ${pipeline_host} --nersc_queue realtime \
                            --nersc_maxnodes 25
                    [[ -f ${dest}/${night}/${exposure}/arcs-${night}-${exposure}.done ]] && \
                        sprun ${ssh} desi_night arcs \
                            --night ${night} \
                            --nersc ${pipeline_host} --nersc_queue realtime \
                            --nersc_maxnodes 25
                    #
                    # if night done run redshifts
                    #
                    [[ -f ${dest}/${night}/${exposure}/science-${night}-${exposure}.done ]] && \
                        sprun ${ssh} desi_night redshifts \
                            --night ${night} \
                            --nersc ${pipeline_host} --nersc_queue realtime \
                            --nersc_maxnodes 25
                else
                    echo "INFO: ${night}/${exposure} appears to be test data.  Skipping pipeline activation." >> ${log}
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
    else
        echo "WARNING: No links found, check connection." >> ${log}
    fi
    /bin/sleep ${sleep}
done
