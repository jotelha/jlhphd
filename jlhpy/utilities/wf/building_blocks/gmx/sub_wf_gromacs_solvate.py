# -*- coding: utf-8 -*-
"""Indenter bounding sphere sub workflow."""

import datetime

from fireworks import Firework
from fireworks.user_objects.firetasks.filepad_tasks import GetFilesByQueryTask
from fireworks.user_objects.firetasks.filepad_tasks import AddFilesTask
from imteksimfw.fireworks.user_objects.firetasks.cmd_tasks import CmdTask

from jlhpy.utilities.wf.workflow_generator import (
    WorkflowGenerator, ProcessAnalyzeAndVisualize)
from jlhpy.utilities.wf.mixin.mixin_wf_storage import (
   DefaultPullMixin, DefaultPushMixin)

import jlhpy.utilities.wf.file_config as file_config


class GromacsSolvateMain(WorkflowGenerator):
    """
    Solvate in water via GROMACS.

    dynamic infiles:
    - data_file:     default.gro
    - topology_file: default.top

    outfiles:
    - data_file:     default.gro
    - topology_file: default.top
    """

    def main(self, fws_root=[]):
        fw_list = []

        # GMX solvate
        # ----------
        step_label = self.get_step_label('gmx_solvate')

        files_in = {
            'data_file':       'default.gro',
            'topology_file':   'default.top',
        }
        files_out = {
            'data_file':       'solvate.gro',
            'topology_file':   'default.top',  # modified by gmx
        }

        #  gmx solvate -cp pull.gro -cs spc216.gro -o solvated.gro -p sys.top
        fts_gmx_solvate = [CmdTask(
            cmd='gmx',
            opt=['solvate',
                 '-cp', 'default.gro',  # input coordinates file
                 '-cs', 'spc216.gro',  # water coordinates
                 '-p', 'default.top',  # in- and output topology file
                 '-o', 'solvate.gro',  # output coordinates file
                 ],
            env='python',
            stderr_file='std.err',
            stdout_file='std.out',
            stdlog_file='std.log',
            store_stdout=True,
            store_stderr=True,
            store_stdlog=False,
            fizzle_bad_rc=True)]

        fw_gmx_solvate = self.build_fw(
            fts_gmx_solvate, step_label,
            parents=fws_root,
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_noqueue_category'])

        fw_list.append(fw_gmx_solvate)

        return fw_list, [fw_gmx_solvate], [fw_gmx_solvate]


class GromacsSolvate(
        DefaultPullMixin, DefaultPushMixin,
        ProcessAnalyzeAndVisualize,
        ):
    def __init__(self, *args, **kwargs):
        ProcessAnalyzeAndVisualize.__init__(self,
            main_sub_wf=GromacsSolvateMain, *args, **kwargs)
