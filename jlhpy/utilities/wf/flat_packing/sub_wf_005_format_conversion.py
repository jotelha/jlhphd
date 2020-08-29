# -*- coding: utf-8 -*-
import datetime
import glob
import os
import pymongo

from fireworks import Firework
from imteksimfw.fireworks.user_objects.firetasks.cmd_tasks import PickledPyEnvTask

from jlhpy.utilities.prep.convert import convert_lammps_data_to_pdb

from imteksimfw.fireworks.utilities.serialize import serialize_module_obj
from jlhpy.utilities.wf.workflow_generator import SubWorkflowGenerator
from jlhpy.utilities.wf.mixin.mixin_wf_storage import DefaultPullMixin, DefaultPushMixin


class FormatConversionMain(SubWorkflowGenerator):
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

        fw_conversion = Firework(fts_conversion,
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

        fw_list.append(fw_conversion)

        return fw_list, [fw_conversion], [fw_conversion]


class FormatConversionSubWorkflowGenerator(
        DefaultPullMixin, DefaultPushMixin,
        FormatConversionMain,
        ):
