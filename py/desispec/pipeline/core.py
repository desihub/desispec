#
# See top-level LICENSE file for Copyright information
#
# -*- coding: utf-8 -*-


class Module( object ):
    """
    This class represents a step of the pipeline.  Usually it represents a
    single small part of the processing, but one could create Modules that
    perform several steps simultaneously.
    """

    def __init__( self ):

    def parallelism( self ):
        raise NotImplementedError('Pipeline Module parallelism() method not implemented')


