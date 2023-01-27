#
# See top-level LICENSE.rst file for Copyright information
#
# -*- coding: utf-8 -*-
"""
desispec.pipeline.tasks
=======================

Classes that describe pipeline tasks.
"""

from __future__ import absolute_import, division, print_function

# On first import, load all the classes that we have, based on the files in
# this directory.  Every file should be named after the type of task (psf,
# frame, etc), and every file should define a class named "TaskBlah" where
# "Blah" can be whatever string you want.

# We could use class (rather than instance) methods in all these task classes
# (since we generally only have one instance in this dictionary below).
# However, by using instances here we leave open the possibility to pass in
# configuration information in the constructors in the future.

from . import base

if base.task_classes is None:
    import sys
    import re
    import pkgutil
    import inspect
    base.task_classes = dict()
    tasknamepat = re.compile(r".*\.(.*)")
    taskclasspat = re.compile(r"Task.*")
    __path__ = pkgutil.extend_path(__path__, __name__)
    for importer, modname, ispkg in pkgutil.walk_packages(path=__path__,
        prefix=__name__+'.'):
        # "modname" is now the name relative to this package (e.g. tasks.foo).
        # Split out the "foo" part, since that is the name of the task type
        # we are adding.
        tasknamemat = tasknamepat.match(modname)
        if tasknamemat is None:
            raise RuntimeError("task submodule name error")
        taskname = tasknamemat.group(1)
        if taskname=="base": continue

        # import the module
        __import__(modname)
        # search the classes in the module for the Task class.
        taskclass = None
        is_class_member = lambda member: inspect.isclass(member) and \
            member.__module__ == modname
        classmembers = inspect.getmembers(sys.modules[modname],
            is_class_member)
        for classname, classobj in classmembers:
            taskclassmat = taskclasspat.match(classname)
            if taskclassmat is not None:
                taskclass = classobj
                break
        if (taskclass is None) and (taskname != "base"):
            raise RuntimeError("No Task class found for task {}"\
                .format(taskname))
        # add the class to the dictionary.
        base.task_classes[taskname] = taskclass()
    base.default_task_chain = ["preproc", "psf", "psfnight", "traceshift",
        "extract", "fiberflat", "fiberflatnight", "sky", "starfit",
        "fluxcalib", "cframe", "spectra", "redshift"]
