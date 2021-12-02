# -*- coding: utf-8 -*-
"""Probe on substrate normal approach."""

import datetime
import dill
import glob
import logging
import os
import pymongo
import warnings

from fireworks import Firework, Workflow, FWAction
from fireworks.user_objects.firetasks.dataflow_tasks import JoinDictTask, ForeachTask
from fireworks.user_objects.firetasks.fileio_tasks import FileTransferTask
from fireworks.user_objects.firetasks.filepad_tasks import GetFilesByQueryTask
from fireworks.user_objects.firetasks.script_task import PyTask
from fireworks.user_objects.firetasks.templatewriter_task import TemplateWriterTask
from imteksimfw.fireworks.user_objects.firetasks.cmd_tasks import CmdTask, EvalPyEnvTask, PickledPyEnvTask
from imteksimfw.fireworks.user_objects.firetasks.recover_tasks import RecoverTask

from jlhpy.utilities.wf.workflow_generator import (
    WorkflowGenerator, ProcessAnalyzeAndVisualize)
from jlhpy.utilities.wf.mixin.mixin_wf_storage import (
   DefaultPullMixin, DefaultPushMixin)


import jlhpy.utilities.wf.file_config as file_config
import jlhpy.utilities.wf.phys_config as phys_config

from ..mixin.mixin_wf_storage import DefaultPushMixin


class ForeachPushStub(DefaultPushMixin, WorkflowGenerator):

    def main(self, fws_root=[]):
        fw_list = []

        step_label = self.get_step_label('empty_fw')

        files_in = {
        }
        files_out = {
            'data_file': 'default.lammps'
        }

        fts_empty = [
            PyTask(func='eval', args=['pass'])
        ]
        fw_empty = self.build_fw(
            fts_empty, step_label,
            parents=[*fws_root],
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_noqueue_category'],
        )
        fw_list.append(fw_empty)

        return fw_list, [fw_empty], [fw_empty]


def detour_fw_action_from_wf_dict(wf_dict):
    wf = Workflow.from_dict(wf_dict)
    return FWAction(detours=[wf], propagate=True)


class LAMMPSTrajectoryFrameExtractionMain(WorkflowGenerator):
    """
    Extract specific frames from NetCDF trajectory and convert to LAMMPS data files.

    inputs:
    - metadata->step_specific->frame_extraction->first_frame_to_extract
    - metadata->step_specific->frame_extraction->last_frame_to_extract
    - metadata->step_specific->frame_extraction->every_nth_frame_to_extract

    dynamic infiles:
        only queried in pull stub, otherwise expected through data flow

    - data_file:       default.lammps
    - trajectory_file: default.nc

    outfiles:
    # - data_file:       default.lammps, passed through unchanged
    # - trajectory_file: default.nc, passed through unchanged

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._foreach_push_stub = ForeachPushStub(*args, **kwargs)

    def main(self, fws_root=[]):
        fw_list = []

        # extract frames
        # --------------
        step_label = self.get_step_label('extract_frames')

        files_in = {
            'data_file':        'default.lammps',
            'trajectory_file':  'default.nc',
        }
        files_out = {}

        # glob pattern and regex to match output of netcdf2data.py
        local_glob_pattern = 'frame_*.lammps'
        frame_index_regex = '(?<=frame_)([0-9]+)(?=\\.lammps)'

        fts_extract_frames = [
            # format netcdf2data command line parameter
            EvalPyEnvTask(
                func='lambda a, b, c: "-".join((a,b,b))',
                inputs=[
                    'metadata->step_specific->frame_extraction->first_frame_to_extract',
                    'metadata->step_specific->frame_extraction->last_frame_to_extract',
                    'metadata->step_specific->frame_extraction->every_nth_frame_to_extract',
                ],
                outputs=[
                    'metadata->step_specific->frame_extraction->netcdf2data_frames_parameter'
                ],
            ),

            # netcdf2data.py writes file named frame_0.lammps ... frame_n.lammps
            CmdTask(
                cmd='netcdf2data',
                opt=['--verbose', '--frames',
                     {'key': 'metadata->step_specific->frame_extraction->netcdf2data_frames_parameter'},
                     'default.lammps', 'default.nc'],
                env='python',
                stderr_file='std.err',
                stdout_file='std.out',
                stdlog_file='std.log',
                store_stdout=True,
                store_stderr=True,
                fizzle_bad_rc=True),

            # get current working directory
            PyTask(
                func='os.getcwd',
                outputs=['cwd'],
            ),

            # put together absolute glob pattern
            PyTask(
                func='os.path.join',
                inputs=['cwd', 'local_glob_pattern'],
                outpus=['absolut_glob_pattern'],
            ),

            # create list of all output files, probably unsorted [ frame_n.lammps ... frame_m.lammps ]
            PyTask(
                func='glob.glob',
                inputs=['absolute_glob_pattern'],
                outpus=['unsorted_frame_file_list'],
            ),

            # ugly utilization of eval: eval(expression,globals,locals) has empty globals {}
            # and the content of "nested_frame_file_list", i.e. {"frame_file_list": [ frame_0.lammps ... frame_n.lammps ] }
            # handed as 2nd and 3rd positional argument. Knowledge about the internal PyTask function call is necessary here.

            # create list of unsorted frame indices, extracted from file names, [ n ... m ]
            PyTask(
                func='eval',
                args=['[ int(f[ f.rfind("_")+1:f.rfind(".") ]) for f in unsorted_frame_file_list ]', {}],
                inputs=['nested_unsorted_frame_file_list'],
                output='unsorted_frame_index_list',
            ),

            # sort list of  frame indices, [ 1 ... n ]
            PyTask(
                func='sorted',
                inputs=['unsorted_frame_file_list'],
                outputs=['sorted_frame_index_list'],
            ),

            # nest list of frame indices and list of file into
            # { "unsorted_frame_index_list": [ n ... m ],
            #   "unsorted_frame_file_list": [ frame_n.lammps ... frame_m.lammps ] }
            JoinDictTask(
                inputs=['unsorted_frame_index_list', 'unsorted_frame_file_list'],
                output=['joint_unsorted_frame_index_file_list'],
            ),

            # create nested indexed representation of
            # { 'indexed_frame_file_dict' : { '1': { type: data, value: frame_1.lammps }, ..., 'n': { type: data, value: frame_n.lammps, frame: n } } }
            PyTask(
                func='eval',
                args=['{ "indexed_frame_file_dict" : { str(i): {"type": "data", "value": f } for i,f in zip(unsorted_frame_index_list,unsorted_frame_file_list) } }', {}],
                inputs=['joint_unsorted_frame_index_file_list'],
                outputs=['nested_indexed_frame_file_dict'],
            ),

            # create list of nested dicts of all output files
            # [ { type: data, value: frame_0.lammps } ... { type: data, value: frame_n.lammps } ]
            PyTask(
                func='eval',
                args=['[ v for k,v in sorted(indexed_frame_file_dict.items()) ]', {}],
                inputs=['nested_indexed_frame_file_dict'],
                outputs=['sorted_frame_file_dict_list'],
            ),

            # create sorted list of nested dicts of frame indices
            # [ { type: data, value: 1 } } ... [ { type: data, value: frame_n.lammps, frame: n } } ]
            PyTask(
                func='eval',
                args=['[ { "type": "data", "value": k} for k in sorted(indexed_frame_file_dict.keys()) ]', {}],
                inputs=['nested_indexed_frame_file_dict'],
                outputs=['sorted_frame_index_dict_list'],
            )
        ]

        fw_extract_frames = self.build_fw(
            fts_extract_frames, step_label,
            parents=[*fws_root],
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_noqueue_category'],
            fw_spec={
                'local_glob_pattern': local_glob_pattern,
                'frame_index_regex': frame_index_regex,
            }
        )

        fw_list.append(fw_extract_frames)

        # restore frames
        # --------------

        # def append_storage_detour():
        push_wf = self._foreach_push_stub.build_wf()
        push_wf_dict = push_wf.as_dict()
        # push_fw_action = FWAction(detours=push_wf)

        func_str = dill.dumps(detour_fw_action_from_wf_dict)

        step_label = self.get_step_label('restore_frames')

        files_in = {}
        files_out = {}

        # Maybe need propagate
        fts_restore_frames = [
            ForeachTask(
                split=['sorted_frame_index_dict_list', 'sorted_frame_file_dict_list'],
                # store frame index in metadata and push to specs in order to preserve
                # processing order of frames for subsequent fireworks
                # TODO: replace with PyEnvTask
                task=[
                    PickledPyEnvTask(
                        func=func_str,
                        args=[push_wf_dict]
                    )
                    # PickledPyEnvTask(
                    #    func=func_str,
                    #    args=['''__import__("fireworks").core.firework.FWAction(
                    #                mod_spec={
                    #                    "_set": {"metadata->frame_index": value},
                    #                    "_push": {"processed_frame_index_list": value}
                    #                }
                    #            )''', {}],
                    #    inputs=['sorted_frame_index_dict_list']),

                    # compute position from metadata and update metadata
                    # PyTask(
                    #    func='eval',
                    #    args=['''__import__("fireworks").core.firework.FWAction(
                    #                mod_spec={
                    #                    "_set": {
                    #                        "metadata->initial_sb_in_dist": sb_in_dist,
                    #                        "metadata->sb_in_dist": round(
                    #                            float(sb_in_dist) + float(frame_index) * float(netcdf_frequency) * float(time_step) * float(
                    #                                constant_indenter_velocity), 6),
                    #                        "metadata->ellapsed_time_steps": int(frame_index) * int(netcdf_frequency),
                    #                        "metadata->ellapsed_time": round(float(frame_index) * float(netcdf_frequency) * float(time_step), 2)
                    #                    }
                    #                }
                    #            )''', {}],
                    #    inputs=['metadata']
                    # ),
                ]
            )
        ]

        fw_restore_frames = self.build_fw(
            fts_extract_frames, step_label,
            parents=[fw_extract_frames],
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_noqueue_category'],
            fw_spec={
                'local_glob_pattern': local_glob_pattern,
                'frame_index_regex': frame_index_regex,
            }
        )
        fw_list.append(fw_restore_frames)

        return fw_list, [fw_restore_frames], [fw_extract_frames]


class LAMMPSTrajectoryFrameExtraction(
        DefaultPullMixin,
        LAMMPSTrajectoryFrameExtractionMain,
        ):
    pass