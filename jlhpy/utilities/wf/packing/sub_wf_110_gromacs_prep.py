# -*- coding: utf-8 -*-
"""Indenter bounding sphere sub workflow."""

import datetime
import pymongo

from fireworks import Firework
from fireworks.user_objects.firetasks.filepad_tasks import GetFilesByQueryTask
from fireworks.user_objects.firetasks.filepad_tasks import AddFilesTask
from imteksimfw.fireworks.user_objects.firetasks.cmd_tasks import CmdTask

from jlhpy.utilities.wf.workflow_generator import (
    SubWorkflowGenerator, ProcessAnalyzeAndVisualizeSubWorkflowGenerator)
from jlhpy.utilities.wf.mixin.mixin_wf_storage import DefaultStorageMixin

class GromacsPrepMain(SubWorkflowGenerator):
    def __init__(self, *args, **kwargs):
        sub_wf_name = 'GromacsPrepMain'
        if 'wf_name_prefix' not in kwargs:
            kwargs['wf_name_prefix'] = sub_wf_name
        else:
            kwargs['wf_name_prefix'] = ':'.join((kwargs['wf_name_prefix'], sub_wf_name))
        super().__init__(*args, **kwargs)

    def pull(self, fws_root=[]):
        fw_list = []

        step_label = self.get_step_label('pull')

        files_in = {}
        files_out = {'data_file': 'in.pdb'}

        fts_pull = [GetFilesByQueryTask(
                query={
                    'metadata->project':    self.source_project_id,
                    'metadata->type':       'initial_config',
                    **self.parameter_dict
                },
                sort_key='metadata.datetime',
                sort_direction=pymongo.DESCENDING,
                limit=1,
                new_file_names=['in.pdb'])]

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

        # PDB chain
        # ---------
        step_label = self.get_step_label('pdb_chain')

        files_in =  {'data_file': 'in.pdb' }
        files_out = {'data_file': 'out.pdb'}

        fts_pdb_chain = [CmdTask(
            cmd='pdb_chain',
            env='python',
            stdin_file='in.pdb',
            stdout_file='out.pdb',
            store_stdout=False,
            store_stderr=False,
            fizzle_bad_rc=True)]

        fw_pdb_chain = Firework(fts_pdb_chain,
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
            parents=fws_root)

        fw_list.append(fw_pdb_chain)

        # PDB tidy
        # --------
        step_label = self.get_step_label('pdb_tidy')

        files_in =  {'data_file': 'in.pdb' }
        files_out = {'data_file': 'out.pdb'}

        fts_pdb_tidy = [CmdTask(
            cmd='pdb_tidy',
            env='python',
            stdin_file='in.pdb',
            stdout_file='out.pdb',
            store_stdout=False,
            store_stderr=False,
            fizzle_bad_rc=True)]

        fw_pdb_tidy = Firework(fts_pdb_tidy,
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
            parents=[fw_pdb_chain])

        fw_list.append(fw_pdb_tidy)

        # GMX pdb2gro
        # -----------
        step_label = self.get_step_label('gmx gmx2gro')

        files_in =  {'data_file': 'in.pdb'}
        files_out = {
            'data_file': 'default.gro',
            'topology_file':   'default.top',
            'restraint_file':  'default.posre.itp'}

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

        fw_gmx_pdb2gro = Firework(fts_gmx_pdb2gro,
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
            parents=[fw_pdb_tidy])

        fw_list.append(fw_gmx_pdb2gro)


        # GMX editconf
        # ------------
        step_label = self.get_step_label('gmx editconf')

        files_in = {
            'data_file': 'in.gro',
            'topology_file':   'default.top',
            'restraint_file':  'default.posre.itp'}
        files_out = {
            'data_file': 'default.gro',
            'topology_file':   'default.top',
            'restraint_file':  'default.posre.itp'}

        fts_gmx_editconf = [CmdTask(
            cmd='gmx',
            opt=['editconf',
                 '-f', 'in.gro',
                 '-o', 'default.gro',
                 '-d', 2.0,  # distance between content and box boundary in nm
                 '-bt', 'cubic',  # box type
                ],
            env='python',
            stderr_file='std.err',
            stdout_file='std.out',
            store_stdout=True,
            store_stderr=True,
            fizzle_bad_rc=True)]

        fw_gmx_editconf = Firework(fts_gmx_editconf,
            name=self.get_fw_label(step_label),
            spec={
                '_category': self.hpc_specs['fw_noqueue_category'],
                '_files_in':  files_in,
                '_files_out': files_out,
                'metadata': {
                    'project': self.project_id,
                    'datetime': str(datetime.datetime.now()),
                    'step':    step_label,
                    **self.kwargs,
                }
            },
            parents=[fw_gmx_pdb2gro])

        fw_list.append(fw_gmx_editconf)

        return fw_list, [fw_gmx_editconf], [fw_pdb_chain]

    def push(self, fws_root=[]):
        fw_list = []

        step_label = self.get_step_label('push')

        files_in = {
            'data_file':      'default.gro',
            'topology_file':  'default.top',
            'restraint_file': 'default.posre.itp'}

        fts_push = [
            AddFilesTask({
                'compress': True,
                'paths': "default.gro",
                'metadata': {
                    'project': self.project_id,
                    'datetime': str(datetime.datetime.now()),
                    'type':    'initial_config_gro'}
            }),
            AddFilesTask({
                'compress': True,
                'paths': "default.top",
                'metadata': {
                    'project': self.project_id,
                    'datetime': str(datetime.datetime.now()),
                    'type':    'initial_config_top'}
            }),
            AddFilesTask({
                'compress': True ,
                'paths': "default.posre.itp",
                'metadata': {
                    'project': self.project_id,
                    'datetime': str(datetime.datetime.now()),
                    'type':    'initial_config_posre_itp'}
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


class GromacsPrepSubWorkflowGenerator(
        DefaultStorageMixin,
        ProcessAnalyzeAndVisualizeSubWorkflowGenerator,
        ):
    def __init__(self, *args, **kwargs):
        sub_wf_name = 'GromacsPrep'
        if 'wf_name_prefix' not in kwargs:
            kwargs['wf_name_prefix'] = sub_wf_name
        else:
            kwargs['wf_name_prefix'] = ':'.join((kwargs['wf_name_prefix'], sub_wf_name))
        ProcessAnalyzeAndVisualizeSubWorkflowGenerator.__init__(self,
            main_sub_wf=GromacsPrepMain(*args, **kwargs),
            *args, **kwargs)
