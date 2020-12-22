# -*- coding: utf-8 -*-
"""Substrate fixed box minimization sub workflow."""

import datetime
import glob
import os
import pymongo
import warnings

from fireworks.user_objects.firetasks.filepad_tasks import GetFilesByQueryTask
from fireworks.user_objects.firetasks.templatewriter_task import TemplateWriterTask
from imteksimfw.fireworks.user_objects.firetasks.cmd_tasks import CmdTask

from jlhpy.utilities.wf.workflow_generator import (
    WorkflowGenerator, ProcessAnalyzeAndVisualize)
from jlhpy.utilities.wf.mixin.mixin_wf_storage import (
   DefaultPullMixin, DefaultPushMixin)
from jlhpy.utilities.wf.building_blocks.sub_wf_lammps_analysis import LAMMPSTrajectoryAnalysis
import jlhpy.utilities.wf.file_config as file_config
import jlhpy.utilities.wf.phys_config as phys_config


class FilterNetCDF(WorkflowGenerator):
    """
    Analysis of probe forces.

    inputs:
    - metadata->step_specific->filter_netcdf->group

    dynamic infiles:
    - trajectory_file: default.nc
    - index_file:      default.ndx

    outfiles:
    - trajectory_file: filtered.nc
    """

    def main(self, fws_root=[]):

        fw_list = []

        # filter nectdf
        # -------------
        step_label = self.get_step_label('filter')

        files_in = {
            'trajectory_file': 'default.nc',
            'index_file': 'default.ndx',
        }
        files_out = {
            'trajectory_file': 'filtered.nc',
        }

        fts_filter = [CmdTask(
            cmd='ncfilter',
            opt=['--debug', '--log',
                 'ncfilter.log', 'default.nc', 'filtered.nc',
                 {'key': 'metadata->step_specific->filter_netcdf->group'}],
            env='python',
            stderr_file='std.err',
            stdout_file='std.out',
            stdlog_file='std.log',
            store_stdout=True,
            store_stderr=True,
            fizzle_bad_rc=True)]

        fw_filter = self.build_fw(
            fts_filter, step_label,
            parents=fws_root,
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_queue_category'],
            queueadapter=self.hpc_specs['single_node_job_queueadapter_defaults']
        )

        fw_list.append(fw_filter)

        return fw_list, [fw_filter], [fw_filter]


class LAMMPSMinimization(
        DefaultPullMixin, DefaultPushMixin,
        ProcessAnalyzeAndVisualize,
        ):
    def __init__(self, *args, **kwargs):
        super().__init__(
            main_sub_wf=LAMMPSMinimizationMain,
            analysis_sub_wf=LAMMPSTrajectoryAnalysis,
            *args, **kwargs)