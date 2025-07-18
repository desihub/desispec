# salloc -N 1 -C cpu -t 04:00:00 -q interactive

cd /global/cfs/cdirs/desi/public/dr2/spectro/redux/loa/zcatalog/v2

desi_zcatalog --survey cmx --program other --group healpix --nproc 128 -o cmx/zpix-cmx-other > logs/zpix-cmx-other.log
desi_zcatalog --survey cmx --program other --group cumulative --nproc 128 -o cmx/ztile-cmx-other-cumulative > logs/ztile-cmx-other-cumulative.log

desi_zcatalog --survey sv1 --program backup --group healpix --nproc 128 -o sv1/zpix-sv1-backup > logs/zpix-sv1-backup.log
desi_zcatalog --survey sv1 --program bright --group healpix --nproc 128 -o sv1/zpix-sv1-bright > logs/zpix-sv1-bright.log
desi_zcatalog --survey sv1 --program dark --group healpix --nproc 128 -o sv1/zpix-sv1-dark > logs/zpix-sv1-dark.log
desi_zcatalog --survey sv1 --program other --group healpix --nproc 128 -o sv1/zpix-sv1-other > logs/zpix-sv1-other.log
desi_zcatalog --survey sv1 --program backup --group cumulative --nproc 128 -o sv1/ztile-sv1-backup-cumulative > logs/ztile-sv1-backup-cumulative.log
desi_zcatalog --survey sv1 --program bright --group cumulative --nproc 128 -o sv1/ztile-sv1-bright-cumulative > logs/ztile-sv1-bright-cumulative.log
desi_zcatalog --survey sv1 --program dark --group cumulative --nproc 128 -o sv1/ztile-sv1-dark-cumulative > logs/ztile-sv1-dark-cumulative.log
desi_zcatalog --survey sv1 --program other --group cumulative --nproc 128 -o sv1/ztile-sv1-other-cumulative > logs/ztile-sv1-other-cumulative.log

desi_zcatalog --survey sv2 --program backup --group healpix --nproc 128 -o sv2/zpix-sv2-backup > logs/zpix-sv2-backup.log
desi_zcatalog --survey sv2 --program bright --group healpix --nproc 128 -o sv2/zpix-sv2-bright > logs/zpix-sv2-bright.log
desi_zcatalog --survey sv2 --program dark --group healpix --nproc 128 -o sv2/zpix-sv2-dark > logs/zpix-sv2-dark.log
desi_zcatalog --survey sv2 --program backup --group cumulative --nproc 128 -o sv2/ztile-sv2-backup-cumulative > logs/ztile-sv2-backup-cumulative.log
desi_zcatalog --survey sv2 --program bright --group cumulative --nproc 128 -o sv2/ztile-sv2-bright-cumulative > logs/ztile-sv2-bright-cumulative.log
desi_zcatalog --survey sv2 --program dark --group cumulative --nproc 128 -o sv2/ztile-sv2-dark-cumulative > logs/ztile-sv2-dark-cumulative.log

desi_zcatalog --survey sv3 --program backup --group healpix --nproc 128 -o sv3/zpix-sv3-backup > logs/zpix-sv3-backup.log
desi_zcatalog --survey sv3 --program bright --group healpix --nproc 128 -o sv3/zpix-sv3-bright > logs/zpix-sv3-bright.log
desi_zcatalog --survey sv3 --program dark --group healpix --nproc 128 -o sv3/zpix-sv3-dark > logs/zpix-sv3-dark.log
desi_zcatalog --survey sv3 --program backup --group cumulative --nproc 128 -o sv3/ztile-sv3-backup-cumulative > logs/ztile-sv3-backup-cumulative.log
desi_zcatalog --survey sv3 --program bright --group cumulative --nproc 128 -o sv3/ztile-sv3-bright-cumulative > logs/ztile-sv3-bright-cumulative.log
desi_zcatalog --survey sv3 --program dark --group cumulative --nproc 128 -o sv3/ztile-sv3-dark-cumulative > logs/ztile-sv3-dark-cumulative.log

desi_zcatalog --survey main --program backup --group healpix --nproc 128 -o main/zpix-main-backup > logs/zpix-main-backup.log
desi_zcatalog --survey main --program bright --group healpix --nproc 128 -o main/zpix-main-bright > logs/zpix-main-bright.log
desi_zcatalog --survey main --program dark --group healpix --nproc 128 -o main/zpix-main-dark > logs/zpix-main-dark.log
desi_zcatalog --survey main --program backup --group cumulative --nproc 128 -o main/ztile-main-backup-cumulative > logs/ztile-main-backup-cumulative.log
desi_zcatalog --survey main --program bright --group cumulative --nproc 128 -o main/ztile-main-bright-cumulative > logs/ztile-main-bright-cumulative.log
desi_zcatalog --survey main --program dark --group cumulative --nproc 128 -o main/ztile-main-dark-cumulative > logs/ztile-main-dark-cumulative.log

desi_zcatalog --survey special --program backup --group healpix --nproc 128 -o special/zpix-special-backup > logs/zpix-special-backup.log
desi_zcatalog --survey special --program bright --group healpix --nproc 128 -o special/zpix-special-bright > logs/zpix-special-bright.log
desi_zcatalog --survey special --program dark --group healpix --nproc 128 -o special/zpix-special-dark > logs/zpix-special-dark.log
desi_zcatalog --survey special --program backup --group cumulative --nproc 128 -o special/ztile-special-backup-cumulative > logs/ztile-special-backup-cumulative.log
desi_zcatalog --survey special --program bright --group cumulative --nproc 128 -o special/ztile-special-bright-cumulative > logs/ztile-special-bright-cumulative.log
desi_zcatalog --survey special --program dark --group cumulative --nproc 128 -o special/ztile-special-dark-cumulative > logs/ztile-special-dark-cumulative.log
