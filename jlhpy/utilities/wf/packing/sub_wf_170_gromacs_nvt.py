# -*- coding: utf-8 -*-
import datetime
import glob
import os
import pymongo

from fireworks import Firework
from fireworks.user_objects.firetasks.filepad_tasks import GetFilesByQueryTask
from fireworks.user_objects.firetasks.filepad_tasks import AddFilesTask
from fireworks.user_objects.firetasks.templatewriter_task import TemplateWriterTask
from imteksimfw.fireworks.user_objects.firetasks.cmd_tasks import CmdTask

from jlhpy.utilities.wf.workflow_generator import (
    WorkflowGenerator, ProcessAnalyzeAndVisualize)
from jlhpy.utilities.wf.mixin.mixin_wf_storage import (
   DefaultPullMixin, DefaultPushMixin)

from jlhpy.utilities.wf.building_blocks.sub_wf_gromacs_analysis import GromacsVacuumTrajectoryAnalysis
from jlhpy.utilities.wf.building_blocks.sub_wf_gromacs_vis import GromacsTrajectoryVisualization

import jlhpy.utilities.wf.file_config as file_config


class GromacsNVTEquilibrationMain(WorkflowGenerator):
    """
    NVT equilibration with GROMACS.

    dynamic infiles:
        only queried in pull stub, otherwise expected through data flow

    - data_file:     default.gro
        queried by { 'metadata->type': 'em_solvated_gro' }
    - topology_file: default.top
        queried by { 'metadata->type': 'solvate_top' }

    static infiles:
        always queried within main trunk

    - parameter_file: default.mdp,
        queried by {'metadata->name': file_config.GMX_NVT_MDP}


    outfiles:
    - log_file:        default.log
        tagged as {'metadata->type': 'nvt_log'}
    - energy_file:     default.edr
        tagged as {'metadata->type': 'nvt_edr'}
    - trajectory_file: default.xtc
        tagged as {'metadata->type': 'nvt_xtc'}
    - data_file:       default.gro
        tagged as {'metadata->type': 'nvt_gro'}
    - index_file:      default.ndx
        tagged as {'metadata->type': 'nvt_ndx'}

    - topology_file:   default.top
        pass through untouched
    """
    def push_infiles(self, fp):

        # static infiles for main
        # -----------------------
        step_label = self.get_step_label('push_infiles')

        infiles = sorted(glob.glob(os.path.join(
            self.infile_prefix,
            file_config.GMX_MDP_SUBDIR,
            file_config.GMX_NVT_MDP)))

        files = {os.path.basename(f): f for f in infiles}

        # metadata common to all these files
        metadata = {
            'project': self.project_id,
            'type': 'input',
            'name': file_config.GMX_NVT_MDP,
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
        files_out = {'input_file': 'default.mdp'}

        fts_pull_mdp = [
            GetFilesByQueryTask(
                query={
                    'metadata->project': self.project_id,
                    'metadata->name':    file_config.GMX_NVT_MDP,
                },
                sort_key='metadata.datetime',
                sort_direction=pymongo.DESCENDING,
                limit=1,
                new_file_names=['default.mdp'])]

        fw_pull_mdp = Firework(fts_pull_mdp,
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

        fw_list.append(fw_pull_mdp)

        # GMX index file
        # --------------
        step_label = self.get_step_label('gmx_make_ndx')

        files_in = {'data_file': 'default.gro'}
        files_out = {'index_file': 'default.ndx'}

        fts_gmx_make_ndx = [
            CmdTask(
                cmd='gmx',
                opt=['make_ndx',
                     '-f', 'default.gro',
                     '-o', 'default.ndx',
                  ],
                env='python',
                stdin_key='stdin',
                stderr_file='std.err',
                stdout_file='std.out',
                stdlog_file='std.log',
                store_stdout=True,
                store_stderr=True,
                store_stdlog=True,
                fizzle_bad_rc=True) ]

        fw_gmx_make_ndx = Firework(fts_gmx_make_ndx,
            name=self.get_fw_label(step_label),
            spec = {
                '_category': self.hpc_specs['fw_noqueue_category'],
                '_files_in':  files_in,
                '_files_out': files_out,
                'stdin':    'q\n',  # do nothing
                'metadata': {
                    'project': self.project_id,
                    'datetime': str(datetime.datetime.now()),
                    'step':    step_label,
                    **self.kwargs
                }
            },
            parents=fws_root)

        fw_list.append(fw_gmx_make_ndx)

        # GMX non-Substrate group
        # -----------------------
        step_label = self.get_step_label('gmx_tools_make_group')

        files_in = {
            'data_file':      'default.gro',
            'topology_file':  'default.top',
            'index_file':     'in.ndx',
            #'parameter_file': 'in.mdp',
        }
        files_out = {
            'data_file':      'default.gro', # pass through unmodified
            'topology_file':  'default.top', # pass unmodified
            'index_file':     'out.ndx',
            #'input_file':     'out.mdp',
        }

        fts_make_group = [CmdTask(
            cmd='gmx_tools',
            opt=['--verbose', '--log', 'default.log',
                'ndx', 'invert',
                '--topology-file', 'default.top',
                '--coordinates-file', 'default.gro',
                '--residue-name', {'key': 'metadata->system->substrate->name'},
                '--group-name', 'non-Substrate',
                '--',
                'in.ndx', 'out.ndx'],
            env='python',
            stderr_file='std.err',
            stdout_file='std.out',
            stdlog_file='std.log',
            store_stdout=True,
            store_stderr=True,
            store_stdlog=False,
            fizzle_bad_rc=True)]

        fw_make_group = Firework(fts_make_group,
            name=self.get_fw_label(step_label),
            spec={
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
            parents=[*fws_root, fw_gmx_make_ndx] )

        fw_list.append(fw_make_group)


        # GMX grompp
        # ----------
        step_label = self.get_step_label('gmx_grompp')

        files_in = {
            'index_file':      'default.ndx',
            'input_file':      'default.mdp',
            'data_file':       'default.gro',
            'topology_file':   'default.top',
        }
        files_out = {
            'input_file':     'default.tpr',
            'parameter_file': 'mdout.mdp',
            'topology_file':  'default.top',  # pass through untouched
            'index_file':     'default.ndx',  # pass through untouched
        }

        # gmx grompp -f nvt.mdp -n nvt.ndx -c em_solvated.gro -r em_solvated.gro -o nvt.tpr -p sys.top
        fts_gmx_grompp = [CmdTask(
            cmd='gmx',
            opt=['grompp',
                 '-f', 'default.mdp',
                 '-n', 'default.ndx',
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
            parents=[*fws_root, fw_pull_mdp, fw_make_group])

        fw_list.append(fw_gmx_grompp)


        # GMX mdrun
        # ---------
        step_label = self.get_step_label('gmx_mdrun')

        files_in = {
            'input_file':    'default.tpr',
            'topology_file': 'default.top',  # pass through untouched
            'index_file':    'default.ndx',  # pass through untouched
        }
        files_out = {
            'log_file':        'default.log',
            'energy_file':     'default.edr',
            'trajectory_file': 'default.xtc',
            'data_file':       'default.gro',
            'topology_file':   'default.top',  # pass through untouched
            'index_file':      'default.ndx',  # pass through untouched
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
                    **self.hpc_specs['single_node_job_queueadapter_defaults'],  # get 1 node
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


    # @property
    # def files_out_list(self):
    #     return [
    #         {
    #             'file_label': 'log_file',
    #             'file_name':  'default.log',
    #             'type_label': 'nvt_log',
    #         },
    #         {
    #             'file_label': 'energy_file',
    #             'file_name':  'default.edr',
    #             'type_label': 'nvt_edr',
    #         },
    #         {
    #             'file_label': 'trajectory_file',
    #             'file_name':  'default.xtc',
    #             'type_label': 'nvt_xtc',
    #         },
    #         {
    #             'file_label': 'data_file',
    #             'file_name':  'default.gro',
    #             'type_label': 'nvt_gro',
    #         },
    #         {
    #             'file_label': 'index_file',
    #             'file_name':  'default.ndx',
    #             'type_label': 'nvt_ndx',
    #         },
    #     ]


class GromacsNVTEquilibration(
        DefaultPullMixin, DefaultPushMixin,
        ProcessAnalyzeAndVisualize,
        ):
    def __init__(self, *args, **kwargs):
        ProcessAnalyzeAndVisualize.__init__(self,
            main_sub_wf=GromacsNVTEquilibrationMain,
            analysis_sub_wf=GromacsVacuumTrajectoryAnalysis,
            vis_sub_wf=GromacsTrajectoryVisualization,
            *args, **kwargs)
