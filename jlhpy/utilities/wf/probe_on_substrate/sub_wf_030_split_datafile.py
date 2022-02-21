# -*- coding: utf-8 -*-
import datetime
import glob
import os
import pymongo

from fireworks.user_objects.firetasks.filepad_tasks import GetFilesByQueryTask
from fireworks.user_objects.firetasks.templatewriter_task import TemplateWriterTask
from imteksimfw.fireworks.user_objects.firetasks.cmd_tasks import EvalPyEnvTask, CmdTask

from jlhpy.utilities.wf.mixin.mixin_wf_storage import DefaultPullMixin, DefaultPushMixin
from jlhpy.utilities.wf.workflow_generator import WorkflowGenerator

import jlhpy.utilities.wf.file_config as file_config


class SplitDatafileMain(WorkflowGenerator):
    """Split LAMMPS datafile into coordinates and topology and into force field components.


    static infiles:
    - template_file: default.input.template
        identified by 'metadata->name': file_config.LMP_INPUT_TEMPLATE

    dynamic infiles:
    - data_file: datafile.lammps

    outfiles:
    - data_file: default.lammps
    - coeff_file: coeff.input
    - index_file: groups.ndx
    """

    def push_infiles(self, fp):
        fp_files = []

        step_label = self.get_step_label('push_infiles')

        # gmx2pdb template file
        infiles = sorted(glob.glob(os.path.join(
            self.infile_prefix,
            file_config.LMP_INPUT_TEMPLATE_SUBDIR,
            file_config.LMP_INPUT_TEMPLATE)))

        metadata = {
            'project': self.project_id,
            'type': 'template',
            'step': step_label,
            'name': file_config.LMP_INPUT_TEMPLATE
        }

        files = {os.path.basename(f): f for f in infiles}

        for name, file_path in files.items():
            identifier = '/'.join((self.project_id, name))  # identifier is like a path on a file system
            fp_files.append(
                fp.add_file(
                    file_path,
                    identifier=identifier,
                    metadata=metadata))

        return fp_files

    def main(self, fws_root=[]):
        fw_list = []

        ### GMX2PDB

        # pull lmp input file
        # -------------------

        step_label = self.get_step_label('input_files_pull')

        files_in = {}
        files_out = {
            'template_file': 'default.input.template',
        }

        fts_pull_template = [
            GetFilesByQueryTask(
                query={
                    'metadata->project': self.project_id,
                    'metadata->name': file_config.LMP_INPUT_TEMPLATE,
                },
                sort_key='metadata.datetime',
                sort_direction=pymongo.DESCENDING,
                limit=1,
                new_file_names=['default.input.template'])]

        fw_pull_template = self.build_fw(
            fts_pull_template, step_label,
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_noqueue_category'])

        fw_list.append(fw_pull_template)

        # fill lmp input script template
        # ------------------------------

        step_label = self.get_step_label('fill_template')

        files_in = {'template_file': 'default.template'}
        files_out = {'input_file': 'default.input'}

        # Jinja2 context:
        static_context = {
            'header': ', '.join((
                self.project_id,
                self.get_fw_label(step_label),
                str(datetime.datetime.now()))),
            'mode': 'split_datafile',
            'base_name': 'default',  # determines output file name default.lammps
            'coeff_outfile': 'coeff.input',
            'data_file': 'datafile.lammps',
            'freeze_substrate': False,
            'has_indenter': True,
            'manual_indenter_region': True,
            'ndx_file': 'default.ndx',
            'shift_system_with_box': True,
            'use_eam': False,
            'use_ewald': False,
            'write_coeff_to_datafile': False,  # don't write force field parameters to datafile
            'write_coeff': True,  # write force field paraemeters to seperate file
            'write_groups_to_file': True,
        }

        # parameters for selecting the  indenter atoms
        dynamic_context = {
            'box_shift_z': 'run->split_datafile->shift_z',  # shift in z direction to have surface zero-aligned
            'indenter_substrate_dist': 'metadata->step_specific->merge->z_dist',  # distance between substrate and indenter
            'substrate_thickness': 'metadata->system->substrate->height',
            'indenter_height': 'run->split_datafile->indenter_height',
            'region_tolerance': 'metadata->step_specific->split_datafile->region_tolerance',
        }

        fts_fill_template = [
            EvalPyEnvTask(  # region defining indenter
                func='lambda h: 2*h',
                inputs=['metadata->system->indenter->bounding_sphere->radius'],
                outputs=['run->split_datafile->indenter_height'],
                propagate=False,
            ),
            EvalPyEnvTask(  # shift whole system to zero-align substrate-solvent interface
                func='lambda z, tol: z + tol',
                inputs=['metadata->system->substrate->height', 'metadata->step_specific->split_datafile->shift_tolerance'],
                outputs=['run->split_datafile->shift_z'],
                propagate=False,
            ),
            TemplateWriterTask({
                'context': static_context,
                'context_inputs': dynamic_context,
                'template_file': 'default.template',
                'template_dir': '.',
                'output_file': 'default.input'}
            )]

        fw_fill_template = self.build_fw(
            fts_fill_template, step_label,
            parents=[fw_pull_template, *fws_root],
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_noqueue_category'])

        fw_list.append(fw_fill_template)

        # LAMMPS run
        # ----------

        step_label = self.get_step_label('lmp_run')

        files_in = {
            'data_file': 'datafile.lammps',
            'input_file': 'default.input',
        }
        files_out = {
            'conserved_coeff_file': 'coeff.input',
            'data_file': 'default.lammps',
            'index_file': 'default.ndx',
            'conserved_input_file': 'default.input',  # untouched
            'log_file': 'log.lammps',
        }
        fts_lmp_run = [CmdTask(
            cmd='lmp',
            opt=['-in', 'default.input'],
            env='python',
            fork=True,
            stderr_file='std.err',
            stdout_file='std.out',
            stdlog_file='std.log',
            store_stdout=True,
            store_stderr=True,
            fizzle_bad_rc=True)]

        fw_lmp_run = self.build_fw(
            fts_lmp_run, step_label,
            parents=[fw_fill_template, *fws_root],
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_noqueue_category'])

        fw_list.append(fw_lmp_run)

        return fw_list, [fw_lmp_run], [fw_lmp_run, fw_fill_template]


class SplitDatafile(
        DefaultPullMixin, DefaultPushMixin,
        SplitDatafileMain):
    pass
