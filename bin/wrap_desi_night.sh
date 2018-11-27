#!/bin/bash

#- This script is called by the data transfer daemon via ssh
#- to launch new jobs for each exposure as they arrive.

#- Add module locations that are not included by default when spawning
#- a command via ssh
module use /usr/common/software/modulefiles

#- define the software environment to use
source /project/projectdirs/desi/software/desi_environment.sh 18.11

#- define the production specific environment variables
source /global/cscratch1/sd/desi/desi/spectro/redux/nightly/setup.sh

#- call desi_night with whatever args were passed in
echo RUNNING desi_night $@
# echo '  (not really)'
desi_night $@

#- debugging; maybe leave on to show what happens?
echo 'Production database tasks:'
desi_pipe top --once
echo 'Batch jobs in the queue:'
squeue -u $USER

