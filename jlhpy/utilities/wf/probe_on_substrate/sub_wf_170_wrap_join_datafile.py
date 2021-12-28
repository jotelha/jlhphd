# -*- coding: utf-8 -*-
import glob
import os
import pymongo
from fireworks.user_objects.firetasks.filepad_tasks import GetFilesByQueryTask
from fireworks.user_objects.firetasks.templatewriter_task import TemplateWriterTask
from imteksimfw.fireworks.user_objects.firetasks.cmd_tasks import CmdTask, PickledPyEnvTask

from jlhpy.utilities.wf.workflow_generator import WorkflowGenerator
from jlhpy.utilities.wf.mixin.mixin_wf_storage import (
   DefaultPullMixin, DefaultPushMixin)

from jlhpy.utilities.geometry.bounding_box import dump_cell_from_lammps_data_file_to_yaml_file
from jlhpy.utilities.prep.merge import merge_lammps_datafiles

import jlhpy.utilities.wf.file_config as file_config
from imteksimfw.utils.serialize import serialize_module_obj

# TODO: replace by something genreic. However, only needed for rendering.
# from jlhpy.utilities.wf.mappings import sds_lammps_type_atom_name_mapping


class WrapJoinDataFileMain(WorkflowGenerator):
    """Make molecules in LAMMPS datafile whole again and reset image flags.

    inputs:
    - metadata->system->counterion->name
    - metadata->system->counterion->resname
    - metadata->system->solvent->resname
    - metadata->system->substrate->name
    - metadata->system->substrate->resname
    - metadata->system->surfactant->resname
    - metadata->step_specific->wrap_join->type_name_mapping

    dynamic infiles:
        only queried in pull stub, otherwise expected through data flow

    - data_file: default.lammps
    - index_file: groups.ndx, only passed through

    static infiles:
        always queried within main trunk

    - vmd_jlhvmd: jlhvmd.tcl, utility functions used in input script
        queried by {'metadata->name': file_config.VMD_JLHVMD}

    - vmd_input_file_template: default.tcl.template,
        queried by {'metadata->name': file_config.VMD_WRAP_JOIN_TEMPLATE}

    outfiles:
    - data_file: default.lammps
    """
    def push_infiles(self, fp):

        # static infiles for main
        # -----------------------
        step_label = self.get_step_label('push_infiles')

        fp_files = []

        # input file template
        infiles = sorted(glob.glob(os.path.join(
            self.infile_prefix,
            file_config.VMD_SUBDIR,
            file_config.VMD_WRAP_JOIN_TEMPLATE)))

        files = {os.path.basename(f): f for f in infiles}

        # metadata common to all these files
        metadata = {
            'project': self.project_id,
            'type': 'input',
            'name': file_config.VMD_WRAP_JOIN_TEMPLATE,
            'step': step_label,
        }

        # insert these input files into data base
        for name, file_path in files.items():
            identifier = '/'.join((self.project_id, name))
            fp_files.append(
                fp.add_file(
                    file_path,
                    identifier=identifier,
                    metadata=metadata))

        # jlhvmd
        infiles = sorted(glob.glob(os.path.join(
            self.infile_prefix,
            file_config.VMD_SUBDIR,
            file_config.VMD_JLHVMD)))

        files = {os.path.basename(f): f for f in infiles}

        metadata = {
            'project': self.project_id,
            'type': 'input',
            'name': file_config.VMD_JLHVMD,
            'step': step_label,
        }

        # insert these input files into data base
        for name, file_path in files.items():
            identifier = '/'.join((self.project_id, name))
            fp_files.append(
                fp.add_file(
                    file_path,
                    identifier=identifier,
                    metadata=metadata))

        return fp_files

    def main(self, fws_root=[]):
        fw_list = []

        # -----------------
        step_label = self.get_step_label('input_files_pull')

        files_in = {}
        files_out = {
            'template_file': 'default.tcl.template',
            'library_file': 'jlhvmd.tcl',
        }

        fts_pull = [
            GetFilesByQueryTask(
                query={
                    'metadata->project': self.project_id,
                    'metadata->name': file_config.VMD_WRAP_JOIN_TEMPLATE,
                },
                sort_key='metadata.datetime',
                sort_direction=pymongo.DESCENDING,
                limit=1,
                new_file_names=['default.tcl.template']),
            GetFilesByQueryTask(
                query={
                    'metadata->project': self.project_id,
                    'metadata->name': file_config.VMD_JLHVMD,
                },
                sort_key='metadata.datetime',
                sort_direction=pymongo.DESCENDING,
                limit=1,
                new_file_names=['jlhvmd.tcl'])
        ]

        fw_pull = self.build_fw(
            fts_pull, step_label,
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_noqueue_category'])

        fw_list.append(fw_pull)

        # VMD input script template
        # -----------------------------
        step_label = self.get_step_label('vmd_wrap_join_template')

        files_in = {
            'template_file': 'default.tcl.template'
        }
        files_out = {
            'input_file': 'default.tcl'
        }

        # Jinja2 context:
        static_context = {
            'data_file': 'default.lammps',
            'bb_yaml': 'bb.yaml'
        }

        dynamic_context = {
            'counterion_name': 'metadata->system->counterion->name',
            'counterion_residue_name': 'metadata->system->counterion->resname',
            'solvent_residue_name': 'metadata->system->solvent->resname',
            'substrate_name': 'metadata->system->substrate->name',
            'substrate_residue_name': 'metadata->system->substrate->resname',
            'surfactant_residue_name': 'metadata->system->surfactant->resname',

            'type_name_mapping': 'metadata->step_specific->wrap_join->type_name_mapping',
            # TODO remove type entries from template
            # H2O_H_type, H2O_O_type
        }

        fts_template = [
            TemplateWriterTask({
                'context': static_context,
                'context_inputs': dynamic_context,
                'template_file': 'default.tcl.template',
                'template_dir': '.',
                'output_file': 'default.tcl'})
        ]

        fw_template = self.build_fw(
            fts_template, step_label,
            parents=[fw_pull, *fws_root],
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_noqueue_category'])

        fw_list.append(fw_template)

        # VMD run
        # -------
        step_label = self.get_step_label('vmd_run')

        files_in = {
            'index_file': 'groups.ndx',
            'input_file': 'default.tcl',
            'data_file': 'default.lammps',
            'library_file': 'jlhvmd.tcl',
        }
        files_out = {
            'bb_file': 'bb.yaml',
            'index_file': 'groups.ndx', # only passed through
            'input_file': 'default.tcl',
            'reference_file': 'default.lammps',
            'wrap_joint_file': 'wrap-joint.lammps',
            'library_file': 'jlhvmd.tcl',
            'unprapped_snapshot': 'unwrapped.png',
            'wrapped_snapshot': 'wrapped.png',
            'wrap_joint_snapshot': 'wrap-joint.png',
        }

        func_str = serialize_module_obj(dump_cell_from_lammps_data_file_to_yaml_file)

        fts_vmd_run = [
            # Extract BB
            PickledPyEnvTask(
                func=func_str,
                args=['default.lammps', 'bb.yaml'],
                env='imteksimpy',
            ),
            CmdTask(
                cmd='vmd',
                opt=['-eofexit', '-e', 'default.tcl'],
                env='python',
                stderr_file='std.err',
                stdout_file='std.out',
                stdlog_file='std.log',
                store_stdout=True,
                store_stderr=True,
                fizzle_bad_rc=True),
            # All renderings steps switched off for VMD in text-only mode
            # convert tga snapshots to png
            # CmdTask(
            #     cmd='convert',
            #     opt=['unwrapped.tga', 'unwrapped.png'],
            #     env='python',
            #     stderr_file='convert_unwrapped.err',
            #     stdout_file='convert_unwrapped.out',
            #     stdlog_file='convert_unwrapped.log',
            #     store_stdout=True,
            #     store_stderr=True,
            #     fizzle_bad_rc=True),
            # CmdTask(
            #     cmd='convert',
            #     opt=['wrapped.tga', 'wrapped.png'],
            #     env='python',
            #     stderr_file='convert_wrapped.err',
            #     stdout_file='convert_wrapped.out',
            #     stdlog_file='convert_wrapped.log',
            #     store_stdout=True,
            #     store_stderr=True,
            #     fizzle_bad_rc=True),
            # CmdTask(
            #     cmd='convert',
            #     opt=['wrap-joint.tga', 'wrap-joint.png'],
            #     env='python',
            #     stderr_file='convert_wrap_joint.err',
            #     stdout_file='convert_wrap_joint.out',
            #     stdlog_file='convert_wrap_joint.log',
            #     store_stdout=True,
            #     store_stderr=True,
            #     fizzle_bad_rc=True),
        ]

        fw_vmd_run = self.build_fw(
            fts_vmd_run, step_label,
            parents=[fw_template, fw_pull, *fws_root],
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_queue_category'],
            queueadapter=self.hpc_specs['single_core_job_queueadapter_defaults'])

        fw_list.append(fw_vmd_run)

        # merge with reference
        # --------------------
        # vmd & topotools mess up the order of types and drop some datafile
        # content, i.e. velocities. Compare against original file and fix
        # introduced errors
        step_label = self.get_step_label('merge')

        files_in = {
            'reference_file': 'reference.lammps',
            'wrap_joint_file': 'wrap-joint.lammps',
        }
        files_out = {
            'data_file': 'default.lammps',
        }

        func_str = serialize_module_obj(merge_lammps_datafiles)

        fts_merge = [
            # Extract BB
            PickledPyEnvTask(
                func=func_str,
                args=['wrap-joint.lammps', 'reference.lammps', 'default.lammps'],
                env='imteksimpy',
            ),
        ]

        fw_merge = self.build_fw(
            fts_merge, step_label,
            parents=[fw_vmd_run],
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_noqueue_category'])

        fw_list.append(fw_merge)

        return fw_list, [fw_vmd_run, fw_merge], [fw_vmd_run, fw_template]


class WrapJoinDataFile(DefaultPullMixin, DefaultPushMixin, WrapJoinDataFileMain):
    pass
