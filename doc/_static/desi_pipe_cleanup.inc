.. code-block:: console

    $> desi_pipe cleanup --help
    
    usage: desi_pipe cleanup [options] (use --help for details)

    Clean up stale task states in the DB

    optional arguments:
      -h, --help            show this help message and exit
      --failed              Also clear failed states
      --submitted           Also clear submitted flag
      --tasktypes TASKTYPES
                            comma separated list of task types to clean (fibermap,
                            rawdata,preproc,psf,psfnight,traceshift,extract,fiberf
                            lat,fiberflatnight,sky,starfit,fluxcalib,cframe,qadata
                            ,spectra,redshift)
      --expid EXPID         Only clean tasks for this exposure ID.