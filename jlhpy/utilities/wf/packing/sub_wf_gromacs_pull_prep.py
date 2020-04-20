# -*- coding: utf-8 -*-
"""Indenter bounding sphere sub workflow."""

import datetime
import glob
import os
import pymongo

from fireworks import Firework
from fireworks.user_objects.firetasks.filepad_tasks import GetFilesByQueryTask
from fireworks.user_objects.firetasks.filepad_tasks import AddFilesTask
from fireworks.user_objects.firetasks.templatewriter_task import TemplateWriterTask

from imteksimfw.fireworks.user_objects.firetasks.cmd_tasks import CmdTask

from jlhpy.utilities.wf.workflow_generator import SubWorkflowGenerator

import jlhpy.utilities.wf.file_config as file_config


class GromacsPullPrepSubWorkflowGenerator(SubWorkflowGenerator):
    """

    Inputs:
        - metadata->system->surfactant->nmolecules
        - metadata->system->surfactant->name
        - metadata->system->counterion->nmolecules
        - metadata->system->counterion->name
        - metadata->system->substrate->natoms
        - metadata->system->substrate->name
    """
    def __init__(self, *args, **kwargs):
        if 'wf_name_prefix' not in kwargs:
            kwargs['wf_name_prefix'] = 'GROMACS pulling prep sub-workflow'
        super().__init__(*args, **kwargs)

    def push_infiles(self, fp):
        step_label = self.get_step_label('push_infiles')

        infiles = sorted(glob.glob(os.path.join(
            self.infile_prefix,
            file_config.GMX_MDP_SUBDIR,
            file_config.GMX_PULL_MDP_TEMPLATE)))

        files = {os.path.basename(f): f for f in infiles}

        # metadata common to all these files
        metadata = {
            'project': self.project_id,
            'type': 'input',
            'name': file_config.GMX_PULL_MDP_TEMPLATE,
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

        # insert these input files into data base
        for name, file_path in files.items():
            identifier = '/'.join((self.project_id, name))
            fp_files.append(
                fp.add_file(
                    file_path,
                    identifier=identifier,
                    metadata=metadata))

        return fp_files

    def pull(self, fws_root=[]):
        fw_list = []

        step_label = self.get_step_label('pull')

        files_in = {}
        files_in = {}
        files_out = {
            'data_file': 'default.gro',
        }

        fts_pull = [
            GetFilesByQueryTask(
                query={
                    'metadata->project':    self.source_project_id,
                    'metadata->type':       'initial_config_gro',
                    **self.parameter_dict
                },
                sort_key='metadata.datetime',
                sort_direction=pymongo.DESCENDING,
                limit=1,
                new_file_names=['default.gro'])]


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
            parents=fws_root)

        fw_list.append(fw_pull)

        return fw_list, [fw_pull], [fw_pull]

    def main(self, fws_root=[]):
        fw_list = []

        # query input files
        # -----------------
        step_label = self.get_step_label('input_files_pull')

        files_in = {}
        files_out = {
            'template_file':  'sys.top.template',
            'parameter_file': 'pull.mdp.template',
        }

        fts_pull = [
            GetFilesByQueryTask(
                query={
                    'metadata->project': self.project_id,
                    'metadata->name':    file_config.GMX_PULL_TOP_TEMPLATE,
                },
                sort_key='metadata.datetime',
                sort_direction=pymongo.DESCENDING,
                limit=1,
                new_file_names=['sys.top.template']),
            GetFilesByQueryTask(
                query={
                    'metadata->project': self.project_id,
                    'metadata->name':   'pull.mdp.template',
                },
                sort_key='metadata.datetime',
                sort_direction=pymongo.DESCENDING,
                limit=1,
                new_file_names=['pull.mdp.template'])]

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

        # top template
        # ------------
        step_label = self.get_step_label('gmx top template')

        files_in =  { 'template_file': 'sys.top.template' }
        files_out = { 'topology_file': 'sys.top' }

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
            'ncounterion': 'metadata->system->counterion->nmolecules',
            'counterion':  'metadata->system->counterion->name',
            'nsubstrate':  'metadata->system->substrate->natoms',
            'substrate':   'metadata->system->substrate->name',
        }

        fts_template = [ TemplateWriterTask( {
            'context': static_template_context,
            'context_inputs': dynamic_template_context,
            'template_file': 'sys.top.template',
            'template_dir': '.',
            'output_file': 'sys.top'} ) ]

        fw_template = Firework(fts_template,
            name=self.get_fw_label(step_label),
            spec = {
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
            parents=[fw_pull,*fws_root])

        fw_list.append(fw_template)

        # GMX index file
        # --------------
        step_label = self.get_step_label('gmx gmx_make_ndx')

        files_in = {'data_file': 'default.gro'}
        files_out = {'index_file': 'default.ndx'}

        fts_gmx_make_ndx = [
            CmdTask(
                cmd='gmx',
                opt=['make_ndx',
                     '-f', 'default.gro',
                     '-o', 'default.ndx',
                  ],
                env = 'python',
                stdin_key    = 'stdin',
                stderr_file  = 'std.err',
                stdout_file  = 'std.out',
                stdlog_file  = 'std.log',
                store_stdout = True,
                store_stderr = True,
                store_stdlog = True,
                fizzle_bad_rc= True) ]

        fw_gmx_make_ndx = Firework(fts_gmx_make_ndx,
            name=self.get_fw_label(step_label),
            spec = {
                '_category': self.hpc_specs['fw_noqueue_category'],
                '_files_in':  files_in,
                '_files_out': files_out,
                'stdin':    'q\n',
                'metadata': {
                    'project': self.project_id,
                    'datetime': str(datetime.datetime.now()),
                    'step':    step_label,
                    **self.kwargs
                }
            },
            parents=fws_root )

        fw_list.append(fw_gmx_make_ndx)


        # GMX pulling groups
        # ------------------
        step_label = self.get_step_label('gmx pulling groups')

        files_in = {
            'data_file':      'default.gro',
            'topology_file':  'default.top',
            'index_file':     'in.ndx',
            'parameter_file': 'in.mdp',
        }
        files_out = {
            'data_file':      'default.gro', # pass through unmodified
            'topology_file':  'default.top', # pass unmodified
            'index_file':     'out.ndx',
            'input_file':     'out.mdp',
        }

        # TODO:
        fts_make_pull_groups = [ CmdTask(
            cmd='gmx_tools',
            opt=['--verbose', '--log', 'default.log',
                'make','pull_groups',
                '--topology-file', 'default.top',
                '--coordinates-file', 'default.gro',
                '--residue-name', 'SDS',  # TODO
                '--atom-name', 'C12',  # TODO
                '--reference-group-name', 'Substrate',
                 '-k', 1000,
                 '--rate', 0.1, '--',
                'in.ndx', 'out.ndx', 'in.mdp', 'out.mdp'],
            env = 'python',
            stderr_file  = 'std.err',
            stdout_file  = 'std.out',
            stdlog_file  = 'std.log',
            store_stdout = True,
            store_stderr = True,
            store_stdlog = True,
            fizzle_bad_rc= True) ]

        fw_make_pull_groups = Firework(fts_make_pull_groups,
            name=self.get_fw_label(step_label),
            spec = {
                '_category': self.hpc_specs['fw_noqueue_category'],
                '_files_in':  files_in,
                '_files_out': files_out,
                'metadata': {
                    'project':  self.project_id,
                    'datetime': str(datetime.datetime.now()),
                    'step':    step_label,
                     **self.kwargs
                }
            },
            parents=[*fws_root, fw_pull, fw_template, fw_gmx_make_ndx] )

        fw_list.append(fw_make_pull_groups)

        return fw_list, [fw_make_pull_groups], [fw_template, fw_gmx_make_ndx, fw_make_pull_groups]

    def push(self, fws_root=[]):
        fw_list = []

        step_label = self.get_step_label('push')

        files_in = {
            'topology_file':  'default.top',
            'index_file':     'default.ndx',
            'input_file':     'default.mdp',
        }

        fts_push = [
            AddFilesTask({
                'compress': True,
                'paths': "default.top",
                'metadata': {
                    'project': self.project_id,
                    'datetime': str(datetime.datetime.now()),
                    'type':    'top_pull',
                }
            }),
            AddFilesTask({
                'compress': True,
                'paths': "default.ndx",
                'metadata': {
                    'project': self.project_id,
                    'datetime': str(datetime.datetime.now()),
                    'type':    'ndx_pull',
                }
            }),
            AddFilesTask({
                'compress': True,
                'paths': "default.mdp",
                'metadata': {
                    'project': self.project_id,
                    'datetime': str(datetime.datetime.now()),
                    'type':    'mdp_pull',
                }
            })]

        fw_push = Firework(fts_push,
            name=self.get_fw_label(step_label),
            spec={
                '_category': self.hpc_specs['fw_noqueue_category'],
                '_files_in': files_in,
                'metadata': {
                    'project': self.project_id,
                    'datetime': str(datetime.datetime.now()),
                    'step':    step_label,
                    **self.kwargs
                }
            },
            parents=fws_root)

        fw_list.append(fw_push)

        return fw_list, [fw_push], [fw_push]
