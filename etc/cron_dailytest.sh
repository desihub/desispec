#!/bin/bash
#- Cronjob to run daily integration tests on edison.nersc.gov

#- Recreate a normal login environment
# export NERSC_HOST=edison
# export MODULEPATH=$MODULEPATH:/usr/common/usg/Modules/modulefiles:/usr/syscom/nsg/modulefiles:/usr/syscom/nsg/opt/modulefiles:/usr/common/das/Modules/modulefiles:/usr/common/graphics/Modules/modulefiles:/usr/common/tig/Modules/modulefiles
# source ~/.bashrc
# source ~/.bash_profile

#- Load our code
module load desispec/master
module load desisim/master
module load specter/master
module load redmonster/master
module switch desimodel/trunk

#- Update software packages
cd $DESISPEC; git pull
cd $DESISIM; git pull
cd $SPECTER_DIR; git pull
cd $DESIMODEL; svn update
#- TODO: Requires password
### cd $REDMONSTER; git pull

#- Environment variables necessary for production
export DESI_TEMPLATE_ROOT=$DESI_ROOT/datachallenge/dc2/templates
export DESI_ELG_TEMPLATES=$DESI_TEMPLATE_ROOT/elg_templates.fits
export DESI_LRG_TEMPLATES=$DESI_TEMPLATE_ROOT/lrg_templates.fits
export DESI_STD_TEMPLATES=$DESI_TEMPLATE_ROOT/std_templates.fits
export DESI_QSO_TEMPLATES=$DESI_TEMPLATE_ROOT/qso_templates_v1.1.fits

export PIXPROD=dailytest
export DESI_SPECTRO_DATA=$DESI_ROOT/spectro/sim/$PIXPROD
export DESI_SPECTRO_SIM=$DESI_ROOT/spectro/sim

export PRODNAME=dailytest
export DESI_SPECTRO_REDUX=$DESI_ROOT/spectro/redux

#- Run the test
python -m desispec.test.integration_test

