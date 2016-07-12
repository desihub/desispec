#!/bin/bash
#
# Test installing and running DESI pipeline from a "bare" account.
#
# For a truly bare-bones test, this script should be invoked with
#     /usr/bin/env -i HOME=/global/homes/d/desi SHELL=/bin/bash USER=desi GROUP=desi bash -l anaconda_integration_test.sh
# replace "desi" with the account name, as needed.
#
# Exit if anything goes wrong.
#
# set -e
echo `date` Running anaconda_integration_test on `hostname`
#
# Set up software directories.
#
[[ -d ${HOME}/.conda ]] && /bin/rm -rf ${HOME}/.conda
module load python/2.7-anaconda
conda config --set always_yes yes --set changeps1 no
userDir=${SCRATCH}/test_anaconda
[[ -d ${userDir} ]] && /bin/rm -rf ${userDir}
/bin/mkdir -p ${userDir}
condaDir=${userDir}/anaconda
conda create -q -p ${condaDir} numpy
# conda create -q -p ${condaDir} numpy=1.10.4
source activate ${condaDir}
#
# Install base packages.
#
# conda install -q astropy scipy matplotlib ipython pyyaml requests
conda install -q astropy=1.1.1 scipy matplotlib ipython pyyaml requests
pip install fitsio
#
# Install desiutil.
#
pip install git+https://github.com/desihub/desiutil.git@1.4.0#egg=desiutil
pip install git+https://github.com/desihub/desimodel.git@0.4.4#egg=desimodel
export DESIMODEL=${userDir}/desimodel/0.4.4
/bin/mkdir -p ${DESIMODEL}
install_desimodel_data -D 0.4.4
#
# Install DESI pipeline packages.
#
pip install git+https://github.com/dkirkby/speclite.git@v0.4#egg=speclite
pip install git+https://github.com/desihub/specter.git@0.4.1#egg=specter
pip install git+https://github.com/desihub/specsim.git@v0.4#egg=specsim
pip install git+https://github.com/desihub/desitarget.git@0.3.3#egg=desitarget
pip install git+https://github.com/desihub/desispec.git@0.5.0#egg=desispec
pip install git+https://github.com/desihub/desisim.git#egg=desisim
#
# Install redmonster.
#
export DESI_PRODUCT_ROOT=${userDir}
wget --no-verbose --output-document=${DESI_PRODUCT_ROOT}/redmonster-v0.3.0.tar.gz https://bitbucket.org/redmonster/redmonster/get/v0.3.0.tar.gz
/bin/mkdir -p ${DESI_PRODUCT_ROOT}/redmonster
tar -x -z -C ${DESI_PRODUCT_ROOT} -f ${DESI_PRODUCT_ROOT}/redmonster-v0.3.0.tar.gz
/bin/mv -v ${DESI_PRODUCT_ROOT}/redmonster-redmonster-67e35df2b697 ${DESI_PRODUCT_ROOT}/redmonster/0.3.0
export REDMONSTER=${DESI_PRODUCT_ROOT}/redmonster/0.3.0
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
export DESI_BASIS_TEMPLATES=${DESI_ROOT}/spectro/templates/basis_templates/v2.0
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
export DESI_SPECTRO_REDUX=${DAILYTEST_ROOT}/spectro/redux
#
# Cleanup from previous tests
#
simDir=${DESI_SPECTRO_SIM}/${PIXPROD}
outDir=${DESI_SPECTRO_REDUX}/${PRODNAME}
/bin/rm -rf ${simDir}
/bin/rm -rf ${outDir}
#
# Run the test
#
export DESI_LOGLEVEL=DEBUG
/bin/mkdir -p ${simDir}
/bin/mkdir -p ${outDir}
# newexp-desi --flavor flat --nspec 5 --night 20160510 --expid 0
# pixsim-desi --nspec 5 --night 20160510 --expid 0
# newexp-desi --flavor arc --nspec 5 --night 20160510 --expid 1
# pixsim-desi --nspec 5 --night 20160510 --expid 1
newexp-desi --flavor dark --nspec 5 --night 20160510 --expid 2
# pixsim-desi --nspec 5 --night 20160510 --expid 2
# python -m desispec.test.integration_test
