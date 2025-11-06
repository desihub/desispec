# Compute the per-fiber redshift success rates and save them to disk
salloc -N 1 -C cpu -t 04:00:00 -q interactive ./desi_per_fiber_qa_stats.py -i /global/cfs/cdirs/desicollab/users/rongpu/tmp/zcatalog/v0.4/main/ -o per_fiber_qa_stats.fits

# Make per-fiber per-tracer redshift QA plots
salloc -N 1 -C cpu -t 04:00:00 -q interactive ./desi_per_fiber_qa_plots.py -i /global/cfs/cdirs/desicollab/users/rongpu/tmp/zcatalog/v0.4/main/ --tracer LRG --stats per_fiber_qa_stats.fits -o /global/cfs/cdirs/desicollab/users/rongpu/redshift_qa/new/loa
salloc -N 1 -C cpu -t 04:00:00 -q interactive ./desi_per_fiber_qa_plots.py -i /global/cfs/cdirs/desicollab/users/rongpu/tmp/zcatalog/v0.4/main/ --tracer ELG_LOP --stats per_fiber_qa_stats.fits -o /global/cfs/cdirs/desicollab/users/rongpu/redshift_qa/new/loa
salloc -N 1 -C cpu -t 04:00:00 -q interactive ./desi_per_fiber_qa_plots.py -i /global/cfs/cdirs/desicollab/users/rongpu/tmp/zcatalog/v0.4/main/ --tracer ELG_VLO --stats per_fiber_qa_stats.fits -o /global/cfs/cdirs/desicollab/users/rongpu/redshift_qa/new/loa
salloc -N 1 -C cpu -t 04:00:00 -q interactive ./desi_per_fiber_qa_plots.py -i /global/cfs/cdirs/desicollab/users/rongpu/tmp/zcatalog/v0.4/main/ --tracer QSO --stats per_fiber_qa_stats.fits -o /global/cfs/cdirs/desicollab/users/rongpu/redshift_qa/new/loa
salloc -N 1 -C cpu -t 04:00:00 -q interactive ./desi_per_fiber_qa_plots.py -i /global/cfs/cdirs/desicollab/users/rongpu/tmp/zcatalog/v0.4/main/ --tracer BGS_BRIGHT --stats per_fiber_qa_stats.fits -o /global/cfs/cdirs/desicollab/users/rongpu/redshift_qa/new/loa
salloc -N 1 -C cpu -t 04:00:00 -q interactive ./desi_per_fiber_qa_plots.py -i /global/cfs/cdirs/desicollab/users/rongpu/tmp/zcatalog/v0.4/main/ --tracer BGS_FAINT --stats per_fiber_qa_stats.fits -o /global/cfs/cdirs/desicollab/users/rongpu/redshift_qa/new/loa

# Create the redshift QA HTML files
./desi_per_fiber_qa_html.py --stats per_fiber_qa_stats.fits -o /global/cfs/cdirs/desicollab/users/rongpu/redshift_qa/new/loa

