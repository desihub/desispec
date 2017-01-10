.. code-block:: console

    usage: desi_pipe [-h] [--raw RAW] [--redux REDUX] [--prod PROD]
                     [--nights NIGHTS]
                     [--nersc_host NERSC_HOST] 
                     [--nersc_max_nodes NERSC_MAX_NODES]                    
                     [--shell_mpi_run SHELL_MPI_RUN]                        
                     [--shell_max_cores SHELL_MAX_CORES] [--fakeboot]
                     [--fakepsf] [--spectrographs SPECTROGRAPHS] 
                     [--debug]                   

    Set up pipeline runs for a production.

    optional arguments:
      -h, --help            show this help message and exit
      --data DATA           override DESI_SPECTRO_DATA
      --redux REDUX         override DESI_SPECTRO_REDUX
      --prod PROD           override SPECPROD
      --model MODEL         override DESIMODEL
      --nights NIGHTS       comma separated (YYYYMMDD) or regex 
                            pattern
      --nersc_host NERSC_HOST
                            NERSC slurm scripts host name 
                            (edison|cori)
      --nersc_max_nodes NERSC_MAX_NODES
                            NERSC slurm scripts max nodes to use. 
                            Default is size of debug queue max.
      --shell_mpi_run SHELL_MPI_RUN
                            bash scripts command to launch MPI 
                            pipeline steps. If --shell_max_cores is 
                            1, this is ignored.
      --shell_max_cores SHELL_MAX_CORES
                            bash scripts max cores to use.
      --fakeboot            bypass bootcalib
      --fakepsf             bypass specex
      --spectrographs SPECTROGRAPHS
                            process only this comma-separated list of
                            spectrographs
      --debug               in setup script, set log level to DEBUG
      --shifter             shifter image to use in alternate slurm 
                            scripts
