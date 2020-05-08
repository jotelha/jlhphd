# -*- coding: utf-8 -*-
"""Indenter bounding sphere sub workflow."""

import datetime
import glob
import os
import pymongo

from fireworks import Firework
from fireworks.user_objects.firetasks.filepad_tasks import GetFilesByQueryTask
from fireworks.user_objects.firetasks.filepad_tasks import AddFilesTask
from imteksimfw.fireworks.user_objects.firetasks.cmd_tasks import CmdTask

from jlhpy.utilities.wf.workflow_generator import (
    SubWorkflowGenerator, ProcessAnalyzeAndVisualizeSubWorkflowGenerator)
from jlhpy.utilities.wf.mixin.mixin_wf_storage import (
   DefaultPullMixin, DefaultPushMixin)

import jlhpy.utilities.wf.file_config as file_config


class GromacsSolvateMain(SubWorkflowGenerator):
    """
    Solvate in water via GROMACS.

    dynamic infiles:
        only queried in pull stub, otherwise expected through data flow

    - data_file:     default.gro
        queried by { 'metadata->type': 'pull_gro' }
    - topology_file: default.top
        queried by { 'metadata->type': 'pull_top' }

    outfiles:
    - data_file:       default.gro
        tagged as {'metadata->type': 'solvate_gro'}
    - topology_file: default.top
        tagged as {'metadata->type': 'solvate_top'}
    """

    def __init__(self, *args, **kwargs):
        sub_wf_name = 'GromacsSolvateMain'
        if 'wf_name_prefix' not in kwargs:
            kwargs['wf_name_prefix'] = sub_wf_name
        else:
            kwargs['wf_name_prefix'] = ':'.join((kwargs['wf_name_prefix'], sub_wf_name))
        super().__init__(*args, **kwargs)


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

        fw_gmx_solvate = Firework(fts_gmx_solvate,
            name=self.get_fw_label(step_label),
            spec={
                '_category': self.hpc_specs['fw_noqueue_category'],
                '_files_in':  files_in,
                '_files_out': files_out,
                'metadata': {
                    'project': self.project_id,
                    'datetime': str(datetime.datetime.now()),
                    'step':    step_label,
                    **self.kwargs
                }
            },
            parents=[*fws_root])

        fw_list.append(fw_gmx_solvate)

        return fw_list, [fw_gmx_solvate], [fw_gmx_solvate]


class GromacsSolvateSubWorkflowGenerator(
        DefaultPullMixin, DefaultPushMixin,
        ProcessAnalyzeAndVisualizeSubWorkflowGenerator,
        ):
    def __init__(self, *args, **kwargs):
        sub_wf_name = ' GromacsSolvate'
        if 'wf_name_prefix' not in kwargs:
            kwargs['wf_name_prefix'] = sub_wf_name
        else:
            kwargs['wf_name_prefix'] = ':'.join((kwargs['wf_name_prefix'], sub_wf_name))
        ProcessAnalyzeAndVisualizeSubWorkflowGenerator.__init__(self,
            main_sub_wf=GromacsSolvateMain(*args, **kwargs),
            *args, **kwargs)
