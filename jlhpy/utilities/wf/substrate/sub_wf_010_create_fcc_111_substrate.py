# -*- coding: utf-8 -*-
"""Create FCC 111 substrate block."""

import datetime
import glob
import os
import pymongo


from fireworks import Firework
from fireworks.user_objects.firetasks.filepad_tasks import GetFilesByQueryTask
from fireworks.user_objects.firetasks.templatewriter_task import TemplateWriterTask

from imteksimfw.fireworks.user_objects.firetasks.cmd_tasks import CmdTask
from imteksimfw.fireworks.user_objects.firetasks.cmd_tasks import PickledPyEnvTask
from imteksimfw.utils.serialize import serialize_module_obj

from jlhpy.utilities.wf.workflow_generator import WorkflowGenerator, ProcessAnalyzeAndVisualize
from jlhpy.utilities.wf.mixin.mixin_wf_storage import DefaultPullMixin, DefaultPushMixin
from jlhpy.utilities.prep.create_fcc_111 import create_fcc_111_data_file

import jlhpy.utilities.wf.file_config as file_config

class CreateSubstrateMain(WorkflowGenerator):
    """
    Create FCC 111 substrate block.

    static infiles:
    - template_file: default.input.template,
        queried by {'metadata->name': file_config.LMP_CONVERT_XYZ_INPUT_TEMPLATE}
    - mass_file: mass.input,
        queried by {'metadata->name': file_config.LMP_MASS_INPUT}

    inputs:
    - metadata->system->substrate->approximate_measures ([float])
    - metadata->system->substrate->element (str): Element name for ASE (i.e. 'Au')
    - metadata->system->substrate->lattice_constant ([float])
    - metadata->system->substrate->lmp->type (int): Type number in LAMMPS

    outfiles:
    - data_file:       default.lammps
        tagged as {'metadata->type': 'initial_config'}

    outputs:
    - metadata->system->substrate->measures ([float])
    """
    def push_infiles(self, fp):
        step_label = self.get_step_label('push_infiles')
        fp_files = []

        # convert xyz input template files
        infiles = sorted(glob.glob(os.path.join(
            self.infile_prefix,
            file_config.LMP_INPUT_TEMPLATE_SUBDIR,
            file_config.LMP_CONVERT_XYZ_INPUT_TEMPLATE)))

        files = {os.path.basename(f): f for f in infiles}

        # metadata common to all these files
        metadata = {
            'project': self.project_id,
            'type': 'template',
            'name': file_config.LMP_CONVERT_XYZ_INPUT_TEMPLATE,
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

        # mass input file
        infiles = sorted(glob.glob(os.path.join(
            self.infile_prefix,
            file_config.LMP_FF_SUBDIR,
            file_config.LMP_MASS_INPUT)))

        files = {os.path.basename(f): f for f in infiles}

        metadata = {
            'project': self.project_id,
            'type': 'input',
            'name': file_config.LMP_MASS_INPUT,
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

        # query input files
        # -----------------
        step_label = self.get_step_label('input_files_pull')

        files_in = {}
        files_out = {
            'template_file':  'default.input.template',
            'mass_file':      'mass.input'
        }

        fts_pull = [
            GetFilesByQueryTask(
                query={
                    'metadata->project': self.project_id,
                    'metadata->name':    file_config.LMP_CONVERT_XYZ_INPUT_TEMPLATE,
                },
                sort_key='metadata.datetime',
                sort_direction=pymongo.DESCENDING,
                limit=1,
                new_file_names=['default.input.template']),
            GetFilesByQueryTask(
                query={
                    'metadata->project': self.project_id,
                    'metadata->name':    file_config.LMP_MASS_INPUT,
                },
                sort_key='metadata.datetime',
                sort_direction=pymongo.DESCENDING,
                limit=1,
                new_file_names=['mass.input'])]

        fw_pull = Firework(fts_pull,
            name=self.get_fw_label(step_label),
            spec={
                '_category': self.hpc_specs['fw_noqueue_category'],
                '_files_in': files_in,
                '_files_out': files_out,
                'metadata': {
                    'project': self.project_id,
                    'datetime': str(datetime.datetime.now()),
                    'step':    step_label,
                    **self.kwargs
                }
            },
            parents=None)

        fw_list.append(fw_pull)

        # Create substrate
        # ----------------
        step_label = self.get_step_label('create_substrate')

        files_in = {}
        files_out = {
            'data_file': 'default.xyz',
        }

        func_str = serialize_module_obj(create_fcc_111_data_file)

        fts_create_substrate = [PickledPyEnvTask(
            func=func_str,
            args=['default.xyz'],
            kwargs_inputs={
                'approximate_measures': 'metadata->system->substrate->approximate_measures',
                'lattice_constant': 'metadata->system->substrate->lattice_constant',
                'element': 'metadata->system->substrate->element',
            },
            outputs=[
                'metadata->system->substrate->measures',
            ],
            env='imteksimpy',
            stderr_file='std.err',
            stdout_file='std.out',
            store_stdout=True,
            store_stderr=True,
            propagate=True,
        )]

        fw_create_substrate = Firework(fts_create_substrate,
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
            parents=[*fws_root])

        fw_list.append(fw_create_substrate)

        # fill convert xyz template
        # --------------------
        step_label = self.get_step_label('fill_template')

        files_in = {
            'template_file': 'default.input.template',
        }
        files_out = {
            'input_file': 'default.input',
        }

        # Jinja2 context:
        # static_template_context = {}

        dynamic_template_context = {
            'type':     'metadata->system->substrate->lmp->type',
            'measures': 'metadata->system->substrate->measures',
        }

        fts_template = [TemplateWriterTask({
            'context_inputs': dynamic_template_context,
            # 'context': static_template_context,
            'template_file': 'default.input.template',
            'template_dir': '.',
            'output_file': 'default.input'})]

        fw_template = Firework(fts_template,
            name=self.get_fw_label(step_label),
            spec={
                '_category': self.hpc_specs['fw_noqueue_category'],
                '_files_in': files_in,
                '_files_out': files_out,
                'metadata': {
                    'project': self.project_id,
                    'datetime': str(datetime.datetime.now()),
                    'step':    step_label,
                     **self.kwargs
                }
            },
            parents=[fw_pull, fw_create_substrate])

        fw_list.append(fw_template)

        # Convert to LAMMPS data
        # ----------------------
        step_label = self.get_step_label('lmp_convert_xyz')

        files_in = {
            'data_file':  'default.xyz',
            'input_file': 'default.input',
            'mass_file':  'mass.input',
        }
        files_out = {
            'data_file':  'default.lammps',
            'input_file': 'default.input',  # untouched
            'mass_file':  'mass.input',  # untouched
        }
        fts_lmp_convert_xyz = [CmdTask(
            cmd='lmp',
            opt=['-in', 'default.input'],
            env='python',
            stderr_file='std.err',
            stdout_file='std.out',
            stdlog_file='std.log',
            store_stdout=True,
            store_stderr=True,
            fizzle_bad_rc=True)]

        fw_lmp_convert_xyz = Firework(fts_lmp_convert_xyz,
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
            parents=[fw_create_substrate, fw_template, fw_pull])

        fw_list.append(fw_lmp_convert_xyz)

        return fw_list, [fw_lmp_convert_xyz], [fw_create_substrate]


class CreateSubstrateWorkflowGenerator(
        DefaultPushMixin,
        ProcessAnalyzeAndVisualize,
        ):
    def __init__(self, *args, **kwargs):
        super().__init__(main_sub_wf=CreateSubstrateMain, *args, **kwargs)
