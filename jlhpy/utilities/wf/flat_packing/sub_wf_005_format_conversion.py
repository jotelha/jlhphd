# -*- coding: utf-8 -*-
import datetime
import glob
import os
import pymongo

from imteksimfw.fireworks.user_objects.firetasks.cmd_tasks import PickledPyEnvTask

from jlhpy.utilities.prep.convert import convert_lammps_data_to_pdb
from jlhpy.utilities.prep.unwrap import unwrap_lammps_data

from imteksimfw.utils.serialize import serialize_module_obj
from jlhpy.utilities.wf.workflow_generator import WorkflowGenerator
from jlhpy.utilities.wf.mixin.mixin_wf_storage import DefaultPullMixin, DefaultPushMixin


class FormatConversionMain(WorkflowGenerator):
    """Convert substrate file format.

    inputs:
    - metadata->step_specific->conversion->lmp_type_to_element_mapping
    - metadata->step_specific->conversion->element_to_pdb_atom_name_mapping
    - metadata->step_specific->conversion->element_to_pdb_residue_name_mapping

    dynamic infiles:
    - data_file: default.lammps

    outfiles:
    - data_file: default.pdb

    outputs:

    """
    def main(self, fws_root=[]):
        fw_list = []

        # unwrap
        # ------

        # wrapped LAMMPS data files are undesired as image flag information
        # is lost in conversion and may lead to single atoms jumping across boundaries
        # possibly breaking smooth surfaces or molecules

        step_label = self.get_step_label('unwrap')

        files_in = {
            'data_file': 'in.lammps',
        }
        files_out = {
            'data_file': 'out.lammps',
        }

        func_str = serialize_module_obj(unwrap_lammps_data)

        fts_unwrap = [PickledPyEnvTask(
            func=func_str,
            args=['in.lammps', 'out.lammps'],
            kwargs={
                'atom_style': 'full',
            },
            env='imteksimpy',
            stderr_file='std.err',
            stdout_file='std.out',
            stdlog_file='std.log',
            store_stdout=True,
            store_stderr=True,
            store_stdlog=True,
            propagate=False,
        )]

        fw_unwrap = self.build_fw(
            fts_unwrap, step_label,
            parents=fws_root,
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_queue_category'],
            queueadapter=self.hpc_specs['quick_single_core_job_queueadapter_defaults'])

        fw_list.append(fw_unwrap)


        # convert
        # -------------------------
        step_label = self.get_step_label('convert')

        files_in = {
            'data_file': 'default.lammps',
        }
        files_out = {
            'data_file': 'default.pdb',
        }

        func_str = serialize_module_obj(convert_lammps_data_to_pdb)

        fts_conversion = [PickledPyEnvTask(
            func=func_str,
            args=['default.lammps', 'default.pdb'],
            kwargs={
                'lammps_style': 'full',
                'lammps_units': 'real',
            },
            kwargs_inputs={
                'lmp_ase_type_mapping': 'metadata->step_specific->conversion->lmp_type_to_element_mapping',
                'ase_pmd_type_mapping': 'metadata->step_specific->conversion->element_to_pdb_atom_name_mapping',
                'ase_pmd_residue_mapping': 'metadata->step_specific->conversion->element_to_pdb_residue_name_mapping',
            },
            env='imteksimpy',
            stderr_file='std.err',
            stdout_file='std.out',
            stdlog_file='std.log',
            store_stdout=True,
            store_stderr=True,
            store_stdlog=True,
            propagate=False,
        )]

        fw_conversion = self.build_fw(
            fts_conversion, step_label,
            parents=[fw_unwrap],
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_queue_category'],
            queueadapter=self.hpc_specs['quick_single_core_job_queueadapter_defaults'])

        fw_list.append(fw_conversion)

        return fw_list, [fw_conversion], [fw_unwrap]


class FormatConversion(
        DefaultPullMixin, DefaultPushMixin,
        FormatConversionMain):
    pass
