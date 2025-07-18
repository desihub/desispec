# salloc -N 1 -C cpu -t 04:00:00 -q interactive

desi_zcatalog --survey cmx --program other --group healpix --nproc 128 -o cmx/zpix-cmx-other
desi_zcatalog --survey cmx --program other --group cumulative --nproc 128 -o cmx/ztile-cmx-other-cumulative

desi_zcatalog --survey sv1 --program backup --group healpix --nproc 128 -o sv1/zpix-sv1-backup
desi_zcatalog --survey sv1 --program bright --group healpix --nproc 128 -o sv1/zpix-sv1-bright
desi_zcatalog --survey sv1 --program dark --group healpix --nproc 128 -o sv1/zpix-sv1-dark
desi_zcatalog --survey sv1 --program other --group healpix --nproc 128 -o sv1/zpix-sv1-other
desi_zcatalog --survey sv1 --program backup --group cumulative --nproc 128 -o sv1/ztile-sv1-backup-cumulative
desi_zcatalog --survey sv1 --program bright --group cumulative --nproc 128 -o sv1/ztile-sv1-bright-cumulative
desi_zcatalog --survey sv1 --program dark --group cumulative --nproc 128 -o sv1/ztile-sv1-dark-cumulative
desi_zcatalog --survey sv1 --program other --group cumulative --nproc 128 -o sv1/ztile-sv1-other-cumulative

desi_zcatalog --survey sv2 --program backup --group healpix --nproc 128 -o sv2/zpix-sv2-backup
desi_zcatalog --survey sv2 --program bright --group healpix --nproc 128 -o sv2/zpix-sv2-bright
desi_zcatalog --survey sv2 --program dark --group healpix --nproc 128 -o sv2/zpix-sv2-dark
desi_zcatalog --survey sv2 --program backup --group cumulative --nproc 128 -o sv2/ztile-sv2-backup-cumulative
desi_zcatalog --survey sv2 --program bright --group cumulative --nproc 128 -o sv2/ztile-sv2-bright-cumulative
desi_zcatalog --survey sv2 --program dark --group cumulative --nproc 128 -o sv2/ztile-sv2-dark-cumulative

desi_zcatalog --survey sv3 --program backup --group healpix --nproc 128 -o sv3/zpix-sv3-backup
desi_zcatalog --survey sv3 --program bright --group healpix --nproc 128 -o sv3/zpix-sv3-bright
desi_zcatalog --survey sv3 --program dark --group healpix --nproc 128 -o sv3/zpix-sv3-dark
desi_zcatalog --survey sv3 --program backup --group cumulative --nproc 128 -o sv3/ztile-sv3-backup-cumulative
desi_zcatalog --survey sv3 --program bright --group cumulative --nproc 128 -o sv3/ztile-sv3-bright-cumulative
desi_zcatalog --survey sv3 --program dark --group cumulative --nproc 128 -o sv3/ztile-sv3-dark-cumulative

desi_zcatalog --survey main --program backup --group healpix --nproc 128 -o main/zpix-main-backup
desi_zcatalog --survey main --program bright --group healpix --nproc 128 -o main/zpix-main-bright
desi_zcatalog --survey main --program dark --group healpix --nproc 128 -o main/zpix-main-dark
desi_zcatalog --survey main --program backup --group cumulative --nproc 128 -o main/ztile-main-backup-cumulative
desi_zcatalog --survey main --program bright --group cumulative --nproc 128 -o main/ztile-main-bright-cumulative
desi_zcatalog --survey main --program dark --group cumulative --nproc 128 -o main/ztile-main-dark-cumulative

desi_zcatalog --survey special --program backup --group healpix --nproc 128 -o special/zpix-special-backup
desi_zcatalog --survey special --program bright --group healpix --nproc 128 -o special/zpix-special-bright
desi_zcatalog --survey special --program dark --group healpix --nproc 128 -o special/zpix-special-dark
desi_zcatalog --survey special --program backup --group cumulative --nproc 128 -o special/ztile-special-backup-cumulative
desi_zcatalog --survey special --program bright --group cumulative --nproc 128 -o special/ztile-special-bright-cumulative
desi_zcatalog --survey special --program dark --group cumulative --nproc 128 -o special/ztile-special-dark-cumulative
