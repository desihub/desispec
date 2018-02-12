#
# See top-level LICENSE.rst file for Copyright information
#
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from collections import OrderedDict

from ..defs import (task_name_sep, task_state_to_int, task_int_to_state)

from ...util import option_list

from ...io import findfile

from .base import BaseTask


# NOTE: only one class in this file should have a name that starts with "Task".

class TaskFibermap(BaseTask):
    """Class containing the properties of one fibermap.

    Since fibermaps have no dependencies and are not created by the pipeline,
    this class is just used to specify names, etc.

    """
    def __init__(self):
        self._cols = [
            "night",
            "expid",
            "flavor",
            "state"
        ]
        self._coltypes = [
            "integer",
            "integer",
            "text",
            "integer"
        ]
        super(TaskFibermap, self).__init__()


    def _name_split(self, name):
        """See BaseTask.name_split.
        """
        fields = name.split(task_name_sep)
        if (len(fields) != 3) or (fields[1] != "fibermap"):
            raise RuntimeError("name \"{}\" not valid for a psf".format(name))
        ret = dict()
        ret["night"] = int(fields[0])
        ret["expid"] = int(fields[2])
        return ret


    def _name_join(self, props):
        """See BaseTask.name_join.
        """
        return "{:08d}{}fibermap{}{:08d}".format(props["night"],
            task_name_sep, task_name_sep, props["expid"])


    def _paths(self, name):
        """See BaseTask.paths.
        """
        props = self.name_split(name)
        return [ findfile("fibermap", night=props["night"],
            expid=props["expid"]) ]


    def _create(self, db):
        """See BaseTask.create.
        """
        with db.conn as con:
            createstr = "create table fibermap (name text unique"
            for col in zip(self._cols, self._coltypes):
                createstr = "{}, {} {}".format(createstr, col[0], col[1])
            createstr = "{})".format(createstr)
            con.execute(createstr)
        return


    def _insert(self, db, props):
        """See BaseTask.insert.
        """
        name = self.name_join(props)
        with db.conn as con:
            cur = con.cursor()
            cur.execute('insert or replace into fibermap values ("{}", {}, '
            '{}, "{}", {})'.format(name, props["night"], props["expid"],
            props["flavor"], task_state_to_int["waiting"]))
            con.commit()
        return


    def _retrieve(self, db, name):
        """See BaseTask.retrieve.
        """
        ret = dict()
        with db.conn as con:
            cur = con.cursor()
            cur.execute(\
                'select * from fibermap where name = "{}"'.format(name))
            row = cur.fetchone()
            if row is None:
                raise RuntimeError("task {} not in database".format(name))
            ret["name"] = name
            ret["night"] = row[1]
            ret["expid"] = row[2]
            ret["flavor"] = row[3]
            ret["state"] = task_int_to_state(row[4])
        return ret


    def _state_set(self, db, name, state):
        """See BaseTask.state_set.
        """
        with db.conn as con:
            cur = con.cursor()
            cur.execute('update fibermap set state = {} where name = "{}"'\
                .format(task_state_to_int(state), name))
            con.commit()
        return


    def _state_get(self, db, name):
        """See BaseTask.state_get.
        """
        st = None
        with db.conn as con:
            cur = con.cursor()
            cur.execute(\
                'select state from fibermap where name = "{}"'.format(name))
            row = cur.fetchone()
            if row is None:
                raise RuntimeError("task {} not in database".format(name))
            st = task_int_to_state(row[0])
        return st


    def _deps(self, name):
        """See BaseTask.deps.
        """
        return list()


    def _run_max_procs(self, procs_per_node):
        """See BaseTask.run_max_procs.
        """
        return 1


    def _run_time(self, name, procs_per_node, db=None):
        """See BaseTask.run_time.
        """
        return 0


    def _run_defaults(self):
        """See BaseTask.run_defaults.
        """
        return dict()


    def _run_cli(self, name, opts, procs):
        """See BaseTask.run_cli.
        """
        return ""


    def _run(self, name, opts, comm):
        """See BaseTask.run.
        """
        return
