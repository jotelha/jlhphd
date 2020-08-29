# -*- coding: utf-8 -*-
import datetime
import glob
import os
import pymongo

from fireworks import Firework
from imteksimfw.fireworks.user_objects.firetasks.cmd_tasks import PickledPyEnvTask

from jlhpy.utilities.geometry.bounding_box import get_bounding_box_via_ase
from jlhpy.utilities.vis.plot_side_views_with_boxes import plot_side_views_with_boxes_via_ase

from imteksimfw.fireworks.utilities.serialize import serialize_module_obj
from jlhpy.utilities.wf.workflow_generator import (
    WorkflowGenerator, ProcessAnalyzeAndVisualizeWorkflowGenerator)
from jlhpy.utilities.wf.mixin.mixin_wf_storage import (
   DefaultPullMixin, DefaultPushMixin)


class FlatSubstrateMeasuresMain(WorkflowGenerator):
    """Flat substrate measures sub workflow.

    dynamic infiles:
    - data_file: default.pdb

    outfiles:
    - data_file: default.pdb (unchanged)

    outputs:
        - metadata->system->substrate->bounding_box ([[float]])
    """
    def main(self, fws_root=[]):
        fw_list = []

        # Bounding box Fireworks
        # -------------------------
        step_label = self.get_step_label('bounding_box')

        files_in = {
            'data_file':      'default.pdb',
        }
        files_out = {
            'data_file':      'default.pdb',
        }

        func_str = serialize_module_obj(get_bounding_box_via_ase)

        fts_bounding_box = [PickledPyEnvTask(
            func=func_str,
            args=['default.pdb'],
            outputs=[
                'metadata->system->substrate->bounding_box',
            ],
            env='imteksimpy',
            stderr_file='std.err',
            stdout_file='std.out',
            stdlog_file='std.log',
            store_stdout=True,
            store_stderr=True,
            store_stdlog=True,
            propagate=True,
        )]

        fw_bounding_box = Firework(fts_bounding_box,
            name=self.get_fw_label(step_label),
            spec={
                '_category': self.hpc_specs['fw_noqueue_category'],
                '_files_in':  files_in,
                '_files_out': files_out,
                'metadata': {
                    'project':  self.project_id,
                    'datetime': str(datetime.datetime.now()),
                    'step':     step_label,
                     **self.kwargs
                }
            },
            parents=fws_root)

        fw_list.append(fw_bounding_box)

        return fw_list, [fw_bounding_box], [fw_bounding_box]


class FlatSubstrateMeasuresVis(
        WorkflowGenerator):
    """Flat substrate measures visualization sub workflow.

    inputs:
    - metadata->system->substrate->bounding_box ([[float]])

    dynamic infiles:
    - data_file: default.pdb

    outfiles:
    - png_file:     default.png
    """
    def main(self, fws_root=[]):
        fw_list = []
        # Plot sideviews
        # --------------
        step_label = self.get_step_label('vis')

        files_in = {
            'data_file': 'default.pdb',
        }
        files_out = {
            'png_file': 'default.png'
        }

        func_str = serialize_module_obj(plot_side_views_with_boxes_via_ase)

        fts_vis = [PickledPyEnvTask(
            func=func_str,
            args=['default.pdb', 'default.png'],
            inputs=[
                'metadata->system->substrate->bounding_box',
            ],  # inputs appended to args
            env='imteksimpy',
            stderr_file='std.err',
            stdout_file='std.out',
            store_stdout=True,
            store_stderr=True,
            propagate=True,
        )]

        fw_vis = Firework(fts_vis,
            name=self.get_fw_label(step_label),
            spec={
                '_category': self.hpc_specs['fw_queue_category'],
                '_queueadapter': {
                    **self.hpc_specs['single_core_job_queueadapter_defaults'],
                },
                '_files_in':  files_in,
                '_files_out': files_out,
                'metadata': {
                    'project':  self.project_id,
                    'datetime': str(datetime.datetime.now()),
                    'step':     step_label,
                     **self.kwargs
                }
            },
            parents=[*fws_root])

        fw_list.append(fw_vis)
        return fw_list, [fw_vis], [fw_vis]


class FlatSubstrateMeasuresWorkflowGenerator(
        DefaultPullMixin, DefaultPushMixin,
        ProcessAnalyzeAndVisualizeWorkflowGenerator,
        ):
    def __init__(self, *args, **kwargs):
        sub_wf_name = 'FlatSubstrateMeasures'
        if 'wf_name_prefix' not in kwargs:
            kwargs['wf_name_prefix'] = sub_wf_name
        else:
            kwargs['wf_name_prefix'] = ':'.join((kwargs['wf_name_prefix'], sub_wf_name))
        ProcessAnalyzeAndVisualizeWorkflowGenerator.__init__(self,
            main_sub_wf=FlatSubstrateMeasuresMain(*args, **kwargs),
            vis_sub_wf=FlatSubstrateMeasuresVis(*args, **kwargs),
            *args, **kwargs)
