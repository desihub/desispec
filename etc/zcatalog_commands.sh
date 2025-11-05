# NOTE: after this script was developed, we re-discovered bin/desi_zcatalog_wrapper,
# which does the same within python, plus generates the zall files.

# Example commands to generate all of the zcatalog files, e.g.
#
#   salloc -N 1 -C cpu -t 04:00:00 -q interactive
#   module load desidatamodel
#   export SPECPROD=loa
#   cd $DESI_ROOT/spectro/redux/$SPECPROD/zcatalog/v2
#   source $DESISPEC/etc/zcatalog_commands.sh

# NOTE: desi_zcatalog will skip over any SURVEY/PROGRAM combinations already done,
# so it is safe to re-run this script until all files are generated.

# requires "module load desidatamodel" first to be able to add units
python -c "import desidatamodel"
if [ $? -ne 0 ]; then
    echo 'Please run "module load desidatamodel" first'
    return 1  # note: return not exit so that if this is sourced, it won't exit parent shell
fi

# create output log directory
mkdir -p logs

# SURVEY PROGRAM combinations to process
survey_program=(
    "cmx other"
    "sv1 dark"
    "sv1 bright"
    "sv1 backup"
    "sv1 other"
    "sv2 dark"
    "sv2 bright"
    "sv2 backup"
    "sv3 dark"
    "sv3 bright"
    "sv3 backup"
    "main dark"
    "main bright"
    "main backup"
    "special dark"
    "special bright"
    "special backup"
)

# I/O saturates so don't use all the cores for parallel reads
NPROC=64

for pair in "${survey_program[@]}"; do
    # Split the pair into separate SURVEY PROGRAM variables
    read -r SURVEY PROGRAM <<< "$pair"

    # Generate the zpix and ztile catalogs
    echo $(date '+%Y-%m-%d %T') "Generating zpix  for $SURVEY $PROGRAM"
    desi_zcatalog --survey $SURVEY --program $PROGRAM --group healpix    --nproc $NPROC -o $SURVEY/zpix-$SURVEY-$PROGRAM             >> logs/zpix-$SURVEY-$PROGRAM.log
    echo $(date '+%Y-%m-%d %T') "Generating ztile for $SURVEY $PROGRAM"
    desi_zcatalog --survey $SURVEY --program $PROGRAM --group cumulative --nproc $NPROC -o $SURVEY/ztile-$SURVEY-$PROGRAM-cumulative >> logs/ztile-$SURVEY-$PROGRAM-cumulative.log
done

echo $(date '+%Y-%m-%d %T') All done



