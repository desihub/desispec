# NOTE: after this script was developed, we re-discovered bin/desi_zcatalog_wrapper,
# which does the same within python, plus generates the zall files.

# Example commands to generate all of the zcatalog files, e.g.
#
#   salloc -N 1 -C cpu -t 04:00:00 -q interactive
#   module load desidatamodel
#   export SPECPROD=loa
#   cd $DESI_ROOT/spectro/redux/$SPECPROD/zcatalog/v2
#   source $DESISPEC/etc/zcatalog_commands.sh --ztile
#   # (uniqpix reduction requires the ztile zcatalogs)
#   source $DESISPEC/etc/zcatalog_commands.sh --zpix
#
# Use --zpix to generate zpix catalogs only.
# Use --ztile to generate ztile catalogs only.
# One of --zpix or --ztile is required (both may be specified to run both).
#
# Use --dry-run to print commands without executing them:
#   source $DESISPEC/etc/zcatalog_commands.sh --zpix --dry-run
#
# Use --healpix to use --group healpix instead of the default --group uniqpix:
#   source $DESISPEC/etc/zcatalog_commands.sh --zpix --healpix

# NOTE: desi_zcatalog will skip over any SURVEY/PROGRAM combinations already done,
# so it is safe to re-run this script until all files are generated.

# Parse options
_DRY_RUN=false
_PIX_GROUP=uniqpix
_RUN_ZPIX=false
_RUN_ZTILE=false
for _arg in "$@"; do
    [ "$_arg" = "--dry-run" ] && _DRY_RUN=true
    [ "$_arg" = "--healpix" ] && _PIX_GROUP=healpix
    [ "$_arg" = "--zpix"    ] && _RUN_ZPIX=true
    [ "$_arg" = "--ztile"   ] && _RUN_ZTILE=true
done
unset _arg

if ! $_RUN_ZPIX && ! $_RUN_ZTILE; then
    echo 'Please specify --zpix or --ztile'
    unset _DRY_RUN _PIX_GROUP _RUN_ZPIX _RUN_ZTILE
    return 1
fi

# requires "module load desidatamodel" first to be able to add units
python -c "import desidatamodel"
if [ $? -ne 0 ]; then
    echo 'Please run "module load desidatamodel" first'
    unset _DRY_RUN _PIX_GROUP _RUN_ZPIX _RUN_ZTILE
    return 1  # note: return not exit so that if this is sourced, it won't exit parent shell
fi

# create output log directory
if $_DRY_RUN; then echo "mkdir -p logs"; else mkdir -p logs; fi

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
    "special other"
)

# I/O saturates so don't use all the cores for parallel reads
NPROC=64

for pair in "${survey_program[@]}"; do
    # Split the pair into separate SURVEY PROGRAM variables
    read -r SURVEY PROGRAM <<< "$pair"

    # Generate the zpix and/or ztile catalogs
    if $_RUN_ZPIX; then
        echo $(date '+%Y-%m-%d %T') "Generating zpix  for $SURVEY $PROGRAM"
        if $_DRY_RUN; then
            echo "desi_zcatalog --survey $SURVEY --program $PROGRAM --group $_PIX_GROUP --nproc $NPROC -o $SURVEY/zpix-$SURVEY-$PROGRAM             >> logs/zpix-$SURVEY-$PROGRAM.log"
        else
            desi_zcatalog --survey $SURVEY --program $PROGRAM --group $_PIX_GROUP --nproc $NPROC -o $SURVEY/zpix-$SURVEY-$PROGRAM             >> logs/zpix-$SURVEY-$PROGRAM.log
        fi
    fi
    if $_RUN_ZTILE; then
        echo $(date '+%Y-%m-%d %T') "Generating ztile for $SURVEY $PROGRAM"
        if $_DRY_RUN; then
            echo "desi_zcatalog --survey $SURVEY --program $PROGRAM --group cumulative --nproc $NPROC -o $SURVEY/ztile-$SURVEY-$PROGRAM-cumulative >> logs/ztile-$SURVEY-$PROGRAM-cumulative.log"
        else
            desi_zcatalog --survey $SURVEY --program $PROGRAM --group cumulative --nproc $NPROC -o $SURVEY/ztile-$SURVEY-$PROGRAM-cumulative >> logs/ztile-$SURVEY-$PROGRAM-cumulative.log
        fi
    fi
done

echo $(date '+%Y-%m-%d %T') All done

unset _DRY_RUN _PIX_GROUP _RUN_ZPIX _RUN_ZTILE
