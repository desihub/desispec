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

class TaskPix(BaseTask):
    """Class containing the properties of one preprocessed pixel file.
    """
    def __init__(self):
        self._cols = [
            "night",
            "band",
            "spec",
            "expid",
            "flavor",
            "state"
        ]
        self._coltypes = [
            "integer",
            "text",
            "integer",
            "integer",
            "text",
            "integer"
        ]
        super(TaskPix, self).__init__()


    def _name_split(self, name):
        """See BaseTask.name_split.
        """
        fields = name.split(task_name_sep)
        if (len(fields) != 5) or (fields[1] != "pix"):
            raise RuntimeError("name \"{}\" not valid for a pix".format(name))
        ret = dict()
        ret["night"] = int(fields[0])
        ret["band"] = fields[2]
        ret["spec"] = int(fields[3])
        ret["expid"] = int(fields[4])
        return ret


    def _name_join(self, props):
        """See BaseTask.name_join.
        """
        return "{:08d}{}pix{}{:s}{}{:d}{}{:08d}".format(props["night"],
            task_name_sep, task_name_sep, props["band"], task_name_sep,
            props["spec"], task_name_sep, props["expid"])


    def _paths(self, name):
        """See BaseTask.paths.
        """
        props = self.name_split(name)
        camera = "{}{}".format(props["band"], props["spec"])
        return [ findfile("pix", night=props["night"], expid=props["expid"],
            camera=camera, groupname=None, nside=None, band=props["band"],
            spectrograph=props["spec"]) ]


    def _create(self, db):
        """See BaseTask.create.
        """
        with db.conn as con:
            createstr = "create table pix (name text unique"
            for col in zip(self._cols, self._coltypes):
                createstr = "{}, {} {}".format(createstr, col[0], col[1])
            createstr = "{})".format(createstr)
            con.execute(createstr)
        return


    def _insert(self, db, props):
        """See BaseTask.insert.
        """
        name = self.name_join(props)
        db.conn.execute('insert or replace into pix values ("{}", {}, '
            '"{}", {}, {}, "{}", {})'.format(name, props["night"],
            props["band"], props["spec"], props["expid"], props["flavor"],
            task_state_to_int["waiting"]))
        return


    def _retrieve(self, db, name):
        """See BaseTask.retrieve.
        """
        ret = dict()
        with db.conn as con:
            cur = con.cursor()
            cur.execute(\
                'select * from pix where name = "{}"'.format(name))
            row = cur.fetchone()
            if row is None:
                raise RuntimeError("task {} not in database".format(name))
            ret["name"] = name
            ret["night"] = row[1]
            ret["band"] = row[2]
            ret["spec"] = row[3]
            ret["expid"] = row[4]
            ret["flavor"] = row[5]
            ret["state"] = task_int_to_state(row[6])
        return ret


    def _state_set(self, db, name, state):
        """See BaseTask.state_set.
        """
        with db.conn as con:
            cur = con.cursor()
            cur.execute('update pix set state = {} where name = "{}"'\
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
                'select state from pix where name = "{}"'.format(name))
            row = cur.fetchone()
            if row is None:
                raise RuntimeError("task {} not in database".format(name))
            st = task_int_to_state(row[0])
        return st


    def _deps(self, name):
        """See BaseTask.deps.
        """
        from .base import task_classes
        props = self.name_split(name)
        fmap = task_classes["fibermap"].name_join(props)

        # FIXME: add raw data file here eventually.
        deptasks = [ fmap ]
        return deptasks


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
