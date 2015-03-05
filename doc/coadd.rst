Coaddition Notes
================

DESI Environment
----------------

Ssh to edison.nersc.gov (remember to use `ssh -A` to propagate your keys for github access) and::

	source /project/projectdirs/desi/software/modules/desi_environment.sh

Installation
------------

Clone the git package and select the development branch::

	git clone git@github.com:desihub/desispec.git
	cd desispec
	git checkout \#3

Per-Login Setup
---------------

Manually set paths for using this installation (assuming `bash`)::

	cd desispec
	export PATH=$PWD/bin:$PATH
	export PYTHONPATH=$PWD/py:$PYTHONPATH

Set pipeline paths::

	export DESI_SPECTRO_REDUX=$DESI_ROOT/spectro/redux
	export PRODNAME=sjb/cedar2a
	export DESI_SPECTRO_SIM=$DESI_ROOT/spectro/sim
	export DESI_SPECTRO_DATA=$DESI_SPECTRO_SIM/alpha-5

Tests
-----

Convert mocks using::

	rm -rf $DESI_SPECTRO_REDUX/bricks
	desi_make_bricks.py --night 20150211 --verbose

Note that the code is not yet smart enough to do the right thing for exposures that have already been added to brick files, hence the `rm` command above.
