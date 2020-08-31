# -*- coding: utf-8 -*-
"""Indenter bounding sphere sub workflow."""

import datetime
import glob
import os
import pymongo

from fireworks.user_objects.firetasks.templatewriter_task import TemplateWriterTask
from fireworks.user_objects.firetasks.filepad_tasks import GetFilesByQueryTask
from imteksimfw.fireworks.user_objects.firetasks.cmd_tasks import CmdTask

from jlhpy.utilities.wf.workflow_generator import WorkflowGenerator, ProcessAnalyzeAndVisualize
from jlhpy.utilities.wf.mixin.mixin_wf_storage import DefaultPullMixin, DefaultPushMixin

import jlhpy.utilities.wf.file_config as file_config


class GromacsPrepMain(WorkflowGenerator):
    """Prepare system for processing with GROMACS.

    inputs:
    - metadata->system->box->length (float)
    - metadata->system->box->width (float)
    - metadata->system->box->height (float)

    dynamic infiles:
    - data_file:     in.pdb
    - topology_file: default.top

    outfiles:
    - data_file:       default.gro
    - topology_file:   default.top
    - restraint_file:  default.posre.itp
    """

    def push_infiles(self, fp):
        step_label = self.get_step_label('push_infiles')

        # top template files
        infiles = sorted(glob.glob(os.path.join(
            self.infile_prefix,
            file_config.GMX_TOP_SUBDIR,
            file_config.GMX_PULL_TOP_TEMPLATE)))

        files = {os.path.basename(f): f for f in infiles}

        # metadata common to all these files
        metadata = {
            'project': self.project_id,
            'type': 'input',
            'name': file_config.GMX_PULL_TOP_TEMPLATE,
            'step': step_label,
        }

        fp_files = []
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
            'template_file':  'sys.top.template',
        }

        fts_pull = [
            GetFilesByQueryTask(
                query={
                    'metadata->project': self.project_id,
                    'metadata->name':    file_config.GMX_TOP_TEMPLATE,
                },
                sort_key='metadata.datetime',
                sort_direction=pymongo.DESCENDING,
                limit=1,
                new_file_names=['sys.top.template'])]

        fw_pull = self.build_fw(
            fts_pull, step_label,
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_noqueue_category'])

        fw_list.append(fw_pull)

        # top template
        # ------------
        step_label = self.get_step_label('gmx_top_template')

        files_in = {'template_file': 'sys.top.template'}
        files_out = {'topology_file': 'sys.top'}

        # Jinja2 context:
        static_template_context = {
            'system_name':  'default',
            'header':       ', '.join((
                self.project_id,
                self.get_fw_label(step_label),
                str(datetime.datetime.now()))),
        }

        dynamic_template_context = {
            'nsurfactant': 'metadata->system->surfactant->nmolecules',
            'surfactant':  'metadata->system->surfactant->name',
            'ncounterion': 'metadata->system->counterion->nmolecules',  # make system symmetric
            'counterion':  'metadata->system->counterion->name',
            'nsubstrate':  'metadata->system->substrate->natoms',
            'substrate':   'metadata->system->substrate->name',
        }

        fts_template = [TemplateWriterTask({
            'context': static_template_context,
            'context_inputs': dynamic_template_context,
            'template_file': 'sys.top.template',
            'template_dir': '.',
            'output_file': 'sys.top'})]

        fw_template = self.build_fw(
            fts_template, step_label,
            parents=[fw_pull, *fws_root],
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_noqueue_category'])

        fw_list.append(fw_template)

        # GMX pdb2gro
        # -----------
        step_label = self.get_step_label('gmx_pdb2gro')

        files_in = {'data_file': 'in.pdb'}
        files_out = {'data_file': 'default.gro'}
        # 'topology_file':   'default.top',
        # 'restraint_file':  'default.posre.itp'}

        fts_gmx_pdb2gro = [CmdTask(
            cmd='gmx',
            opt=['pdb2gmx',
                 '-f', 'in.pdb',
                 '-o', 'default.gro',
                 '-p', 'default.top',
                 '-i', 'default.posre.itp',
                 '-ff', 'charmm36',
                 '-water', 'tip3p'],
            env='python',
            stderr_file='std.err',
            stdout_file='std.out',
            store_stdout=True,
            store_stderr=True,
            fizzle_bad_rc=True)]

        fw_gmx_pdb2gro = self.build_fw(
            fts_gmx_pdb2gro, step_label,
            parents=fws_root,
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_noqueue_category'])

        fw_list.append(fw_gmx_pdb2gro)

        # GMX editconf
        # ------------
        step_label = self.get_step_label('gmx_editconf')

        files_in = {
            'data_file': 'in.gro',
            'topology_file':   'default.top',
            'restraint_file':  'default.posre.itp'}
        files_out = {
            'data_file': 'default.gro',
            'topology_file':   'default.top'}
            #'restraint_file':  'default.posre.itp'

        fts_gmx_editconf = [CmdTask(
            cmd='gmx',
            opt=['editconf',
                 '-f', 'in.gro',
                 '-o', 'default.gro',
                 '-box',
                 {'key': 'metadata->system->box->length'},
                 {'key': 'metadata->system->box->width'},
                 {'key': 'metadata->system->box->height'},
                 '-bt', 'triclinic',  # box type
                 ],
            env='python',
            stderr_file='std.err',
            stdout_file='std.out',
            store_stdout=True,
            store_stderr=True,
            fizzle_bad_rc=True)]

        fw_gmx_editconf = self.build_fw(
            fts_gmx_editconf, step_label,
            parents=[fw_gmx_pdb2gro, fw_template],
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_noqueue_category'])

        fw_list.append(fw_gmx_editconf)

        return fw_list, [fw_gmx_editconf], [fw_gmx_pdb2gro, fw_template]


class GromacsPrep(DefaultPullMixin, DefaultPushMixin, ProcessAnalyzeAndVisualize):
    def __init__(self, *args, **kwargs):
        super().__init__(main_sub_wf=GromacsPrepMain, *args, **kwargs)
