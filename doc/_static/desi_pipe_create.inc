.. code-block:: console

    $> desi_pipe create --help

    usage: desi_pipe create [options] (use --help for details)

    Create a new production

    optional arguments:
      -h, --help            show this help message and exit
      --root ROOT           value to use for DESI_ROOT
      --data DATA           value to use for DESI_SPECTRO_DATA
      --redux REDUX         value to use for DESI_SPECTRO_REDUX
      --prod PROD           value to use for SPECPROD
      --force               force DB creation even if prod exists on disk (useful
                            for simulations
      --basis BASIS         value to use for DESI_BASIS_TEMPLATES
      --calib CALIB         value to use for DESI_SPECTRO_CALIB
      --db-sqlite           Use SQLite database backend.
      --db-sqlite-path DB_SQLITE_PATH
                            Override path to SQLite DB
      --db-postgres         Use PostgreSQL database backend. You must correctly
                            configure your ~/.pgpass file!
      --db-postgres-host DB_POSTGRES_HOST
                            Set PostgreSQL hostname
      --db-postgres-port DB_POSTGRES_PORT
                            Set PostgreSQL port number
      --db-postgres-name DB_POSTGRES_NAME
                            Set PostgreSQL DB name
      --db-postgres-user DB_POSTGRES_USER
                            Set PostgreSQL user name
      --db-postgres-authorized DB_POSTGRES_AUTHORIZED
                            Additional PostgreSQL users / roles to authorize
      --nside NSIDE         HEALPix nside value to use for spectral grouping.
