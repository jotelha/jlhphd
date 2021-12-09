# -*- coding: utf-8 -*-
"""Probe on substrate normal approach."""
import logging

from fireworks.user_objects.firetasks.dataflow_tasks import CommandLineTask, JoinDictTask
from fireworks.user_objects.firetasks.script_task import PyTask

from imteksimfw.fireworks.user_objects.firetasks.cmd_tasks import (
     CmdTask, EvalPyEnvTask, PickledPyEnvTask)
from imteksimfw.fireworks.user_objects.firetasks.dataflow_tasks import BranchWorkflowTask
from imteksimfw.utils.serialize import serialize_module_obj

from jlhpy.utilities.wf.workflow_generator import WorkflowGenerator
from jlhpy.utilities.wf.mixin.mixin_wf_storage import DefaultPullMixin

from ..mixin.mixin_wf_storage import DefaultPushMixin

from jlhpy.utilities.analysis.map_distances_and_frames import (
    get_frame_range_from_distance_range, compute_distance_from_frame_number)


class ForeachPushStub(DefaultPushMixin, WorkflowGenerator):
    """"Stores a data file after branching via ForEach task."""
    def main(self, fws_root=[]):
        fw_list = []

        step_label = self.get_step_label('retrieve_metadata')

        files_in = {}
        files_out = {}

        func_str = serialize_module_obj(compute_distance_from_frame_number)

        fts_retrieve_metadata = [
            # Remove encapsulating list from sorted_frame_file_dict_list
            EvalPyEnvTask(
                func='lambda l: l[0]',
                inputs=['sorted_frame_file_dict_list'],
                outputs=['sorted_frame_file_dict'],
            ),

            EvalPyEnvTask(
                func='lambda f: int(f[ f.rfind("_")+1:f.rfind(".")])',
                inputs=['sorted_frame_file_dict->value'],
                outputs=['metadata->step_specific->frame_extraction->frame_number'],
                propagate=True,
            ),

            # compute distance between substrate and probe in extracted frame
            PickledPyEnvTask(
                func=func_str,
                inputs=[
                    'metadata->step_specific->frame_extraction->frame_number',
                    'metadata->step_specific->merge->z_dist',
                    'metadata->step_specific->probe_normal_approach->constant_indenter_velocity',
                    'metadata->step_specific->probe_normal_approach->netcdf_frequency',
                    'metadata->step_specific->frame_extraction->time_step',  # this should come from somewhere else
                ],
                outputs=[
                    'metadata->step_specific->frame_extraction->distance',
                ],
                propagate=True,
                env='imteksimpy',
            ),
        ]
        fw_retrieve_metadata = self.build_fw(
            fts_retrieve_metadata, step_label,
            parents=[*fws_root],
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_noqueue_category'],
        )
        fw_list.append(fw_retrieve_metadata)

        # retrieve frames
        #----------------
        step_label = self.get_step_label('retrieve_frames')

        files_in = {}
        files_out = {
            'data_file': 'default.lammps'
        }

        fts_retrieve_frames = [
            # TODO: Replace with FileTransferTask
            CommandLineTask(
                command_spec={
                    "command": ["cp"],
                    "sorted_frame_file_dict": {
                        "source": "sorted_frame_file_dict",
                    },
                    "restored_frame_file_dict": {
                        "target": {
                            "type": "path", "value": "default.lammps"
                        },
                    },
                },
                inputs=["sorted_frame_file_dict"],
                outputs=["restored_frame_file_dict"],
            ),
        ]

        fw_retrieve_frames = self.build_fw(
            fts_retrieve_frames, step_label,
            parents=[fw_retrieve_metadata],
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_noqueue_category'],
        )
        fw_list.append(fw_retrieve_frames)

        return fw_list, [fw_retrieve_frames], [fw_retrieve_metadata]


class LAMMPSTrajectoryFrameExtractionMain(WorkflowGenerator):
    """
    Extract specific frames from NetCDF trajectory and convert to LAMMPS data files.

    inputs:
    - metadata->step_specific->frame_extraction->first_distance_to_extract
    - metadata->step_specific->frame_extraction->last_distance_to_extract
    - metadata->step_specific->frame_extraction->distance_interval

    - metadata->step_specific->merge->z_dist # Ang
    - metadata->step_specific->probe_normal_approach->steps
    - metadata->step_specific->probe_normal_approach->netcdf_frequency
    - metadata->step_specific->probe_normal_approach->constant_indenter_velocity # Ang / fs

    outputs:
    - metadata->step_specific->frame_extraction->distance

    dynamic infiles:
        only queried in pull stub, otherwise expected through data flow

    - data_file:       default.lammps
    - trajectory_file: default.nc

    outfiles:
    - data_file:       default.lammps, extracted frame, one per branch
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

        func_str = serialize_module_obj(get_frame_range_from_distance_range)

        fts_extract_frames = [
            # compute desired frames from approach run parameters
            PickledPyEnvTask(
                func=func_str,
                inputs=[
                    'metadata->step_specific->frame_extraction->first_distance_to_extract',
                    'metadata->step_specific->frame_extraction->last_distance_to_extract',
                    'metadata->step_specific->frame_extraction->distance_interval',
                    'metadata->step_specific->merge->z_dist',
                    'metadata->step_specific->probe_normal_approach->constant_indenter_velocity',
                    'metadata->step_specific->probe_normal_approach->netcdf_frequency',
                    'metadata->step_specific->frame_extraction->time_step', # this should come from somewhere else
                ],
                outputs=[
                    'metadata->step_specific->frame_extraction->first_frame_to_extract',
                    'metadata->step_specific->frame_extraction->last_frame_to_extract',
                    'metadata->step_specific->frame_extraction->every_nth_frame_to_extract',
                ],
                stderr_file='std.err',
                stdout_file='std.out',
                store_stdout=True,
                store_stderr=True,
                env='imteksimpy',
            ),

            # format netcdf2data command line parameter, end frame is exclusive, hence +1
            EvalPyEnvTask(
                func='lambda a, b, c: "-".join((str(a),str(int(b)+1),str(c)))',
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
                outputs=['absolute_glob_pattern'],
            ),

            # create list of all output files, probably unsorted [ frame_n.lammps ... frame_m.lammps ]
            PyTask(
                func='glob.glob',
                inputs=['absolute_glob_pattern'],
                outputs=['unsorted_frame_file_list'],
            ),

            # nest list of all output files into {"frame_file_list": [ frame_n.lammps ... frame_m.lammps ] }
            JoinDictTask(
                inputs=['unsorted_frame_file_list'],
                output='nested_unsorted_frame_file_list',
            ),

            # ugly utilization of eval: eval(expression,globals,locals) has empty globals {}
            # and the content of "nested_frame_file_list", i.e. {"frame_file_list": [ frame_0.lammps ... frame_n.lammps ] }
            # handed as 2nd and 3rd positional argument. Knowledge about the internal PyTask function call is necessary here.

            # create list of unsorted frame indices, extracted from file names, [ n ... m ]
            PyTask(
                func='eval',
                args=['[ int(f[ f.rfind("_")+1:f.rfind(".") ]) for f in unsorted_frame_file_list ]', {}],
                inputs=['nested_unsorted_frame_file_list'],
                outputs=['unsorted_frame_index_list',]
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
                output='joint_unsorted_frame_index_file_list',
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
            # [ { type: data, value: 1 }, ...,  { type: data, value: n } ]
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

        push_wf = self._foreach_push_stub.build_wf()

        step_label = self.get_step_label('branch_frames')

        files_in = {}
        files_out = {}

        # Maybe need propagate
        fts_branch_frames = [
            BranchWorkflowTask(
                # split=['sorted_frame_index_dict_list', 'sorted_frame_file_dict_list'],
                split='sorted_frame_file_dict_list',
                addition_wf=push_wf,
                # store frame index in metadata and push to specs in order to preserve
                # processing order of frames for subsequent fireworks
                # TODO: replace with PyEnvTask
                superpose_addition_on_my_fw_spec=True,
                stdlog_file='std.log',
                store_stdlog=True,
                loglevel=logging.DEBUG,
            )
        ]

        fw_branch_frames = self.build_fw(
            fts_branch_frames, step_label,
            parents=[fw_extract_frames],
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_noqueue_category'],
            fw_spec={
                'local_glob_pattern': local_glob_pattern,
                'frame_index_regex': frame_index_regex,
            }
        )
        fw_list.append(fw_branch_frames)

        return fw_list, [fw_branch_frames], [fw_extract_frames]


class LAMMPSTrajectoryFrameExtraction(
        DefaultPullMixin,
        LAMMPSTrajectoryFrameExtractionMain,
        ):
    pass