#!/bin/bash
#
# Copy files to a directory with a cadence that approximates
# DESI data taking.
#
function usage {
    local execName=$(basename $0)
    (
    echo "usage ${execName} [-h] [-D DIR] [-n NIGHT] [-p SECONDS] [-S DIR]"
    echo "        DATA - Type of data to transfer."
    echo "          -D - Destination directory."
    echo "          -h - Print usage information and exit."
    echo "          -n - Set the night value, if necessary."
    echo "          -p - Sleep for SECONDS between copies."
    echo "          -S - Source directory."
    ) >&2
}
#
# Parse arguments
#
src=/global/project/projectdirs/desi/spectro/sim/dts/20160310
dst=/global/project/projectdirs/desi/spectro/sim/dts/inbox
night=$(basename ${src})
# Three times an hour.
naptime=1200
# Gap between delivery of individual files.
latency=10
deliveryOptions='-p "module load desimodules" -p "module switch desispec/my-master" -n edisongrid'
while getopts D:hn:p:S: argname; do
    case ${argname} in
        D)
            dst=${OPTARG}
            ;;
        h)
            usage
            exit 0
            ;;
        n)
            night=${OPTARG}
            ;;
        p)
            naptime=${OPTARG}
            ;;
        S)
            src=${OPTARG}
            ;;
        *)
            echo "Unknown option!"
            usage
            exit 1
    esac
done
shift $((OPTIND - 1))
#
# Check inputs
#
if [[ -d ${dst} ]]; then
    /bin/rm -f ${dst}/*
else
    /bin/mkdir -p ${dst}
fi
if [[ ! -d ${src} ]]; then
    echo "Can't find directory: ${src}"
    exit 1
fi
#
# Get a list of exposures.
#
exposures=$(/bin/ls -1 ${src}/fibermap-* | /usr/bin/tr '.-' ' ' | /usr/bin/awk '{print $2}')
#
# Loop over exposures.
#
stage=start
for e in ${exposures}; do
    # This hack removes leading zeros.
    expid=$((e + 0))
    /bin/cp -v ${src}/fibermap-${e}.fits ${dst}
    desi_dts_delivery ${deliveryOptions} ${dst}/fibermap-${e}.fits ${expid} ${night} ${stage}
    /bin/sleep ${latency}
    /bin/cp -v ${src}/guider-${e}.fits.fz ${dst}
    desi_dts_delivery ${deliveryOptions} ${dst}/fibermap-${e}.fits ${expid} ${night} update
    /bin/sleep ${latency}
    /bin/cp -v ${src}/desi-${e}.fits.fz ${dst}
    desi_dts_delivery ${deliveryOptions} ${dst}/fibermap-${e}.fits ${expid} ${night} update
    /bin/sleep ${naptime}
    stage=update
done
#
# Dummy file to signal end-of-night
#
/bin/cp -v ${src}/weather* ${dst}
desi_dts_delivery ${deliveryOptions} ${dst}/weather-${night}.fits ${expid} ${night} end
