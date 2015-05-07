========
desispec
========

Introduction
------------

This package contains scripts and packages for building and running DESI spectroscopic analyses.


Installation
------------

You can install these tools in a variety of ways.  Here are several that may be of interest:

1.  Manually running from the git checkout.  Add the "bin" directory to your $PATH environment variable and add the "py" directory to your $PYTHONPATH environment variable.
2.  Install (and uninstall) a symlink to your live git checkout::

        $>  python setup.py develop --prefix=/path/to/somewhere
        $>  python setup.py develop --prefix=/path/to/somewhere --uninstall

3.  Install a fixed version of the tools::

        $>  python setup.py install --prefix=/path/to/somewhere


Versioning
----------

If you have tagged a version and wish to set the package version based on your current git location::

    $>  python setup.py version

And then install as usual

Status
------

.. image:: https://travis-ci.org/desihub/desispec.png
    :target: https://travis-ci.org/desihub/desispec
    :alt: Travis Build Status

License
-------

Please see the ``LICENSE`` file.
