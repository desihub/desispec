#!/bin/bash
#PBS -q debug
#PBS -l walltime=00:30:00
#PBS -l mppwidth=24
#PBS -A desi
#PBS -j oe

#- Cron/batch job to run daily integration tests on edison.nersc.gov

set -e
echo `date` Running dailytest on `hostname`

#- Configure desi environment if needed
if [ -z "$DESIROOT" ]; then
    source /project/projectdirs/desi/software/modules/desi_environment.sh
fi

#- Load our code
module load speclite/master
module load desispec/master
module load desisim/master
module load specter/master
module load desitarget/master
module load redmonster/master
module switch desimodel/master

#- Update software packages
echo 'updating speclite'; cd $SPECLITE; git pull; fix_permissions.sh .
echo 'updating desispec'; cd $DESISPEC; git pull; fix_permissions.sh .
echo 'updating desisim'; cd $DESISIM; git pull; fix_permissions.sh .
echo 'updating specter'; cd $SPECTER_DIR; git pull; fix_permissions.sh .
echo 'updating desitarget'; cd $DESITARGET; git pull; fix_permissions.sh .
echo 'updating redmonster'; cd $REDMONSTER; git pull; fix_permissions.sh .
echo 'updating desimodel'; cd $DESIMODEL; git pull; fix_permissions.sh .

#- Also update desimodel data from svn trunk
svn update $DESIMODEL/data/

#- Ensure that $SCRATCH is defined so that we don't accidentally clobber stuff
if [ -z "$SCRATCH" ]; then
    echo "ERROR: need to set SCRATCH environment variable"
    exit 1
fi

#- Where should output go?
export DAILYTEST_ROOT=$SCRATCH/desi

export PIXPROD=dailytest
export DESI_SPECTRO_DATA=$DAILYTEST_ROOT/spectro/sim/$PIXPROD
export DESI_SPECTRO_SIM=$DAILYTEST_ROOT/spectro/sim

export PRODNAME=dailytest
export DESI_SPECTRO_REDUX=$DAILYTEST_ROOT/spectro/redux

#- Cleanup from previous tests
simdir=$DESI_SPECTRO_SIM/$PIXPROD
outdir=$DESI_SPECTRO_REDUX/$PRODNAME
rm -rf $simdir
rm -rf $outdir

#- Run the test
mkdir -p $simdir
mkdir -p $outdir
python -m desispec.test.integration_test > $outdir/dailytest.log

echo
echo "[...]"
echo

tail -10 $outdir/dailytest.log

echo `date` done with dailytest
