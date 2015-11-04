#!/bin/bash
#PBS -q debug
#PBS -l walltime=00:30:00
#PBS -l mppwidth=24
#PBS -A desi
#PBS -j oe

#- Cron/batch job to run daily integration tests on edison.nersc.gov

set -e
echo `date` Running dailytest on `hostname`

#- Load our code
module load desispec/master
module load desisim/master
module load specter/master
module load redmonster/master
module switch desimodel/trunk

#- Update software packages
echo 'updating desispec'; cd $DESISPEC; git pull; fix_permissions.sh .
echo 'updating desisim'; cd $DESISIM; git pull; fix_permissions.sh .
echo 'updating specter'; cd $SPECTER_DIR; git pull; fix_permissions.sh .
echo 'updating desimodel'; cd $DESIMODEL; svn update; fix_permissions.sh .
echo 'updating redmonster'; cd $REDMONSTER; git pull; fix_permissions.sh .

exit 0

#- Environment variables necessary for production
export DESI_TEMPLATE_ROOT=$DESI_ROOT/datachallenge/dc2/templates
export DESI_ELG_TEMPLATES=$DESI_TEMPLATE_ROOT/elg_templates.fits
export DESI_LRG_TEMPLATES=$DESI_TEMPLATE_ROOT/lrg_templates.fits
export DESI_STD_TEMPLATES=$DESI_TEMPLATE_ROOT/std_templates.fits
export DESI_QSO_TEMPLATES=$DESI_TEMPLATE_ROOT/qso_templates_v1.1.fits

#- Where should output go?
export DAILYTEST_ROOT=$SCRATCH/desi
## export DAILYTEST_DIR=$DESI_ROOT

export PIXPROD=dailytest
export DESI_SPECTRO_DATA=$DAILYTEST_ROOT/spectro/sim/$PIXPROD
export DESI_SPECTRO_SIM=$DAILYTEST_ROOT/spectro/sim

export PRODNAME=dailytest
export DESI_SPECTRO_REDUX=$DAILYTEST_ROOT/spectro/redux

#- Run the test
outdir=$DESI_SPECTRO_REDUX/$PRODNAME
mkdir -p $outdir
python -m desispec.test.integration_test > $outdir/dailytest.log

echo
echo "[...]"
echo

tail -10 $outdir/dailytest.log

echo `date` done with dailytest
