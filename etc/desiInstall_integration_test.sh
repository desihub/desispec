#!/bin/bash
#
# Test installing and running DESI pipeline from a "bare" account.
#
# For a truly bare-bones test, this script should be invoked with
#     /usr/bin/env -i HOME=/global/homes/d/desi SHELL=/bin/bash USER=desi GROUP=desi bash -l desiInstall_integration_test.sh
# replace "desi" with the account name, as needed.
#
# Exit if anything goes wrong.
#
# set -e
echo `date` Running desiInstall_integration_test on `hostname`
#
# Set up software directories.
#
userDir=${SCRATCH}/test_desiInstall
[[ -d ${userDir} ]] && /bin/rm -rf ${userDir}
/bin/mkdir -p ${userDir}
export DESI_PRODUCT_ROOT=${userDir}/test_desiInstall/software/${NERSC_HOST}
moduleDir=${userDir}/software/modules/${NERSC_HOST}
/bin/mkdir -p ${DESI_PRODUCT_ROOT}
/bin/mkdir -p ${moduleDir}
#
# desiInstall needs requests, so have to set up hpcports first.
#
source /project/projectdirs/cmb/modules/hpcports_NERSC.sh
if [[ "${NERSC_HOST}" == "edison" || "${NERSC_HOST}" == "cori" ]]; then
    hpcports gnu
else
    hpcports
fi
module load astropy-hpcp
module load scipy-hpcp
module load matplotlib-hpcp
module load ipython-hpcp
module load yaml-hpcp
module load fitsio-hpcp
module load requests-hpcp
module load subversion-hpcp
module list
#
# Install desiutil.
#
[[ -f test_desiInstall.ini ]] && /bin/rm test_desiInstall.ini
echo '[Module Processing]' > test_desiInstall.ini
echo "nersc_module_dir = ${userDir}/software/modules" >> test_desiInstall.ini
[[ -f desiBootstrap.sh ]] && /bin/rm desiBootstrap.sh
wget https://raw.githubusercontent.com/desihub/desiutil/master/bin/desiBootstrap.sh
/bin/chmod ug+x desiBootstrap.sh
echo ./desiBootstrap.sh -v -c test_desiInstall.ini -p $(which python)
./desiBootstrap.sh -v -c test_desiInstall.ini -p $(which python)
module use ${moduleDir}
module load desiutil
module list
#
# Install DESI infrastructure packages.
#
desiInstall -d -v -c test_desiInstall.ini desitree 0.2.0
desiInstall -v -c test_desiInstall.ini desimodel 0.4.3
desiInstall -d -v -c test_desiInstall.ini desimodules 0.9.3
module load desimodules
module list
#
# Install DESI pipeline packages.
#
desiInstall -d -v -c test_desiInstall.ini speclite v0.4
desiInstall -d -v -c test_desiInstall.ini specter 0.5.0
desiInstall -d -v -c test_desiInstall.ini specsim v0.4
desiInstall -d -v -c test_desiInstall.ini desimodel 0.4.4
desiInstall -d -v -c test_desiInstall.ini desitarget 0.4.0
desiInstall -d -v -c test_desiInstall.ini desispec 0.6.0
desiInstall -d -v -c test_desiInstall.ini desisim 0.11.0
module switch desimodel/0.4.4
module load speclite
module load specter
module load specsim
module load desitarget
module load desispec
module load desisim
module list
#
# Install redmonster.
#
REDMONSTER_VERSION=1.1.0
wget --no-verbose --output-document=${DESI_PRODUCT_ROOT}/redmonster-${REDMONSTER_VERSION}.tar.gz https://github.com/desihub/redmonster/archive/1.1.0.tar.gz
export DESI_PRODUCT_ROOT=${userDir}
/bin/mkdir -p ${DESI_PRODUCT_ROOT}/redmonster
tar -x -z -C ${DESI_PRODUCT_ROOT} -f ${DESI_PRODUCT_ROOT}/redmonster-${REDMONSTER_VERSION}.tar.gz
/bin/mv -v ${DESI_PRODUCT_ROOT}/redmonster-${REDMONSTER_VERSION} ${DESI_PRODUCT_ROOT}/redmonster/${REDMONSTER_VERSION}
export REDMONSTER=${DESI_PRODUCT_ROOT}/redmonster/${REDMONSTER_VERSION}
if [[ -z "${PYTHONPATH}" ]]; then
    export PYTHONPATH=${REDMONSTER}/python
else
    export PYTHONPATH=${REDMONSTER}/python:${PYTHONPATH}
fi
export REDMONSTER_TEMPLATES_DIR=${REDMONSTER}/templates
#
# Set environment variables.
#
export DESI_ROOT=/project/projectdirs/desi
export DESI_BASIS_TEMPLATES=${DESI_ROOT}/spectro/templates/basis_templates/v2.2
#
# Reproduce the daily integration test.
#
if [[ -z "${SCRATCH}" ]]; then
    echo "ERROR: need to set SCRATCH environment variable"
    exit 1
fi
#
# Where should output go?
#
export DAILYTEST_ROOT=${SCRATCH}/desi
export PIXPROD=dailytest
export DESI_SPECTRO_DATA=${DAILYTEST_ROOT}/spectro/sim/${PIXPROD}
export DESI_SPECTRO_SIM=${DAILYTEST_ROOT}/spectro/sim
export PRODNAME=dailytest
export SPECPROD=dailytest
export DESI_SPECTRO_REDUX=${DAILYTEST_ROOT}/spectro/redux
#
# Cleanup from previous tests
#
simDir=${DESI_SPECTRO_SIM}/${PIXPROD}
outDir=${DESI_SPECTRO_REDUX}/${SPECPROD}
/bin/rm -rf ${simDir}
/bin/rm -rf ${outDir}
#
# Run the test
#
export DESI_LOGLEVEL=DEBUG
/bin/mkdir -p ${simDir}
/bin/mkdir -p ${outDir}
python -m desispec.test.integration_test > ${outDir}/dailytest.log

echo
echo "[...]"
echo

tail -10 ${outDir}/dailytest.log

echo `date` done with desiInstall_integration_test
