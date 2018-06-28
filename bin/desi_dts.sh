#!/bin/bash
#
# Configuration
#
log=${HOME}/desi_dts.log
[[ -f ${log} ]] && /bin/touch ${log}
src=/data/dts/exposures/raw
staging=$(/bin/realpath ${DESI_ROOT}/spectro/staging/raw)
dest=$(/bin/realpath ${DESI_SPECTRO_DATA})
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
    links=$(ssh -q dts /bin/find ${src} -type l 2>/dev/null)
    if [[ -n "${links}" ]]; then
        for l in ${links}; do
            exposure=$(/bin/basename ${l})
            night=$(/bin/basename $(/bin/dirname ${l}))
            [[ ! -d ${staging}/${night} ]] && \
                sprun /bin/mkdir -p ${staging}/${night}
            if [[ ! -d ${staging}/${night}/${exposure} ]]; then
                sprun /bin/rsync --verbose --no-motd \
                    --recursive --copy-dirlinks --times --omit-dir-times \
                    dts:${src}/${night}/${exposure}/ ${staging}/${night}/${exposure}/
                status=$?
            fi
            if [[ "${status}" == "0" ]]; then
                [[ ! -d ${dest}/${night} ]] && \
                    sprun /bin/mkdir -p ${dest}/${night}
                [[ ! -L ${dest}/${night}/${exposure} ]] && \
                    sprun /bin/ln -s ${staging}/${night}/${exposure} ${dest}/${night}/${exposure}
            else
                echo "ERROR: rsync problem detected!" >> ${log}
            fi
        done
    else
        echo "WARNING: No links found, check connection." >> ${log}
    fi
    /bin/sleep 10m
done
#
# TODO
#
# Check file permissions
# Checksums
# Pipeline notifications.
#
