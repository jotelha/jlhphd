from fireworks.user_objects.firetasks.dataflow_tasks import CommandLineTask

from imteksimfw.utils.serialize import serialize_module_obj
from imteksimfw.fireworks.user_objects.firetasks.cmd_tasks import (
    EvalPyEnvTask, PickledPyEnvTask)

from jlhpy.utilities.wf.workflow_generator import WorkflowGenerator
from jlhpy.utilities.wf.mixin.mixin_wf_storage import DefaultPushMixin

from jlhpy.utilities.analysis.map_distances_and_frames import compute_distance_from_frame_number


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
                fork=True,
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