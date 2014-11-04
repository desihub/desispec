#
# See top-level LICENSE file for Copyright information
#
# -*- coding: utf-8 -*-


import os


def data_root ():
    dir = os.environ[ 'DESIDATA' ]
    if dir == None:
        raise RuntimeError('DESIDATA environment variable not set')
    return dir


