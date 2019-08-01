.. code-block:: console

    $> desi_pipe update --help

    usage: desi_pipe update [options] (use --help for details)

    Update a production

    optional arguments:
      -h, --help       show this help message and exit
      --nights NIGHTS  comma separated (YYYYMMDD) or regex pattern- only nights
                       matching these patterns will be examined.
      --nside NSIDE    HEALPix nside value to use for spectral grouping.
      --expid EXPID    Only update the production for a single exposure ID.
