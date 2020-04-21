# -*- coding: utf-8 -*-
"""Indenter bounding sphere sub workflow."""

import datetime
import glob
import os
import pymongo

from fireworks import Firework
from fireworks.user_objects.firetasks.filepad_tasks import GetFilesByQueryTask
from fireworks.user_objects.firetasks.filepad_tasks import AddFilesTask
from imteksimfw.fireworks.user_objects.firetasks.cmd_tasks import CmdTask

from jlhpy.utilities.wf.workflow_generator import SubWorkflowGenerator

import jlhpy.utilities.wf.file_config as file_config


class GromacsEnergyMinimizationAfterSolvationSubWorkflowGenerator(SubWorkflowGenerator):
    """
    Energy minimization with GROMACS.

    dynamic infiles:
        only queried in pull stub, otherwise expected through data flow

    - data_file:     default.gro
        queried by { 'metadata->type': 'solvate_gro' }
    - topology_file: default.top
        queried by { 'metadata->type': 'solvate_top' }

    static infiles:
        always queried within main trunk

    - parameter_file: default.mdp,
        queried by {'metadata->name': file_config.GMX_EM_SOLVATED_MDP}

    outfiles:
    - log_file:        em.log
        tagged as {'metadata->type': 'em_solvated_log'}
    - energy_file:     em.edr
        tagged as {'metadata->type': 'em_solvated_edr'}
    - trajectory_file: em.trr
        tagged as {'metadata->type': 'em_solvated_trr'}
    - data_file:       em.gro
        tagged as {'metadata->type': 'em_solvated_gro'}
    """

    def __init__(self, *args, **kwargs):
        if 'wf_name_prefix' not in kwargs:
            kwargs['wf_name_prefix'] = 'GROMACS energy minimization after solvation sub-workflow'
        super().__init__(*args, **kwargs)

    def push_infiles(self, fp):
        step_label = self.get_step_label('push_infiles')

        infiles = sorted(glob.glob(os.path.join(
            self.infile_prefix,
            file_config.GMX_MDP_SUBDIR,
            file_config.GMX_EM_SOLVATED_MDP)))

        files = {os.path.basename(f): f for f in infiles}

        # metadata common to all these files
        metadata = {
            'project': self.project_id,
            'type': 'input',
            'name': file_config.GMX_EM_SOLVATED_MDP,
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

    def pull(self, fws_root=[]):
        fw_list = []

        step_label = self.get_step_label('pull')

        files_in = {}
        files_out = {
            'data_file':      'default.gro',
            'topology_file':  'default.top',
        }

        fts_pull = [
            GetFilesByQueryTask(
                query={
                    'metadata->project':    self.source_project_id,
                    'metadata->type':       'solvate_gro',
                    **self.parameter_dict
                },
                sort_key='metadata.datetime',
                sort_direction=pymongo.DESCENDING,
                limit=1,
                new_file_names=['default.gro']),
            GetFilesByQueryTask(
                query={
                    'metadata->project':    self.source_project_id,
                    'metadata->type':       'solvate_top',
                    **self.parameter_dict
                },
                sort_key='metadata.datetime',
                sort_direction=pymongo.DESCENDING,
                limit=1,
                new_file_names=['default.top']),
            # GetFilesByQueryTask(
            #     query={
            #         'metadata->project':    self.source_project_id,
            #         'metadata->type':       'initial_config_posre_itp',
            #         **self.parameter_dict
            #     },
            #     sort_key='metadata.datetime',
            #     sort_direction=pymongo.DESCENDING,
            #     limit=1,
            #     new_file_names=['default.posre.itp'])
            ]

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
        files_out = {'input_file': 'default.mdp'}

        fts_pull = [
            GetFilesByQueryTask(
                query={
                    'metadata->project': self.project_id,
                    'metadata->name':    file_config.GMX_EM_SOLVATED_MDP,
                },
                sort_key='metadata.datetime',
                sort_direction=pymongo.DESCENDING,
                limit=1,
                new_file_names=['default.mdp'])]

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

        # GMX grompp
        # ----------
        step_label = self.get_step_label('gmx grompp')

        files_in = {
            'input_file':      'default.mdp',
            'data_file':       'default.gro',
            'topology_file':   'default.top',
        }
        files_out = {
            'input_file':     'default.tpr',
            'parameter_file': 'mdout.mdp',
        }

        fts_gmx_grompp = [CmdTask(
            cmd='gmx',
            opt=['grompp',
                 '-f', 'default.mdp',
                 '-c', 'default.gro',
                 '-r', 'default.gro',
                 '-o', 'default.tpr',
                 '-p', 'default.top',
                ],
            env='python',
            stderr_file='std.err',
            stdout_file='std.out',
            stdlog_file='std.log',
            store_stdout=True,
            store_stderr=True,
            fizzle_bad_rc=True)]

        fw_gmx_grompp = Firework(fts_gmx_grompp,
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
            parents=[*fws_root, fw_pull])

        fw_list.append(fw_gmx_grompp)


        # GMX mdrun
        # ---------
        step_label = self.get_step_label('gmx mdrun')

        files_in = {'input_file':   'default.tpr'}
        files_out = {
            'log_file':        'default.log',
            'energy_file':     'default.edr',
            'trajectory_file': 'default.trr',
            'data_file':       'default.gro'
        }

        fts_gmx_mdrun = [CmdTask(
            cmd='gmx',
            opt=['mdrun',
                 '-deffnm', 'default', '-v'],
            env='python',
            stderr_file='std.err',
            stdout_file='std.out',
            stdlog_file='std.log',
            store_stdout=True,
            store_stderr=True,
            fizzle_bad_rc=True)]

        fw_gmx_mdrun = Firework(fts_gmx_mdrun,
            name=self.get_fw_label(step_label),
            spec={
                '_category': self.hpc_specs['fw_queue_category'],
                '_queueadapter': {
                    'queue':    self.hpc_specs['queue'],
                    'walltime': self.hpc_specs['walltime'],
                    'ntasks':   self.hpc_specs['logical_cores_per_node'],  # get 1 node
                },
                '_files_in':  files_in,
                '_files_out': files_out,
                'metadata': {
                    'project': self.project_id,
                    'datetime': str(datetime.datetime.now()),
                    'step':    step_label,
                    **self.kwargs,
                }
            },
            parents=[fw_gmx_grompp])

        fw_list.append(fw_gmx_mdrun)

        return fw_list, [fw_gmx_mdrun], [fw_gmx_grompp]

    def push(self, fws_root=[]):
        fw_list = []

        step_label = self.get_step_label('push')

        files_out = {}
        files_in = {
            'log_file':        'default.log',
            'energy_file':     'default.edr',
            'trajectory_file': 'default.trr',
            'data_file':       'default.gro'}

        fts_push = [
            AddFilesTask({
                'compress': True,
                'paths': "default.log",
                'metadata': {
                    'project': self.project_id,
                    'datetime': str(datetime.datetime.now()),
                    'type':    'em_solvated_log'}
            }),
            AddFilesTask({
                'compress': True,
                'paths': "default.edr",
                'metadata': {
                    'project': self.project_id,
                    'datetime': str(datetime.datetime.now()),
                    'type':    'em_solvated_edr'}
            }),
            AddFilesTask({
                'compress': True,
                'paths': "default.trr",
                'metadata': {
                    'project': self.project_id,
                    'datetime': str(datetime.datetime.now()),
                    'type':    'em_solvated_trr'}
            }),
            AddFilesTask({
                'compress': True,
                'paths': "default.gro",
                'metadata': {
                    'project': self.project_id,
                    'datetime': str(datetime.datetime.now()),
                    'type':    'em_solvated_gro'}
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
