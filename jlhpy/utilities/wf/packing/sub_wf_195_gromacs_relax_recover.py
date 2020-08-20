# -*- coding: utf-8 -*-
"""GROMACS relaxation recovery sub workflow."""

import datetime
import logging

from fireworks import Firework, Workflow

from imteksimfw.fireworks.user_objects.firetasks.cmd_tasks import CmdTask
from imteksimfw.fireworks.user_objects.firetasks.recover_tasks import RecoverTask

from jlhpy.utilities.wf.workflow_generator import SubWorkflowGenerator

# class GromacsRestartMain(SubWorkflowGenerator):
#
#     def main(self, fws_root=[]):
#         fw_list = []
#
#         # GMX mdrun
#         # ---------
#         step_label = self.get_step_label('gmx_mdrun')
#
#         files_in = {
#             'checkpoint_file': 'default.cpt',
#             'log_file':        'default.log',
#             'energy_file':     'default.edr',
#             'trajectory_file': 'default.trr',
#             # 'compressed_traj_file': 'default.xtc',
#             'data_file':       'default.gro',
#             'input_file':      'default.tpr',
#             'topology_file':   'default.top',
#             'index_file':      'default.ndx',
#         }
#         files_out = {
#             'log_file':        'default.log',
#             'energy_file':     'default.edr',
#             'trajectory_file': 'default.trr',
#             # 'compressed_traj_file': 'default.xtc',
#             'data_file':       'default.gro',
#             'topology_file':   'default.top',  # pass through untouched
#             'index_file':      'default.ndx',  # pass through untouched
#         }
#
#         fts_gmx_mdrun = [CmdTask(
#             cmd='gmx',
#             opt=['mdrun', '-cpi', 'default'],
#             env='python',
#             stderr_file='std.err',
#             stdout_file='std.out',
#             stdlog_file='std.log',
#             store_stdout=True,
#             store_stderr=True,
#             fizzle_bad_rc=True)]
#
#         fw_gmx_mdrun = Firework(fts_gmx_mdrun,
#             name=self.get_fw_label(step_label),
#             spec={
#                 '_category': self.hpc_specs['fw_queue_category'],
#                 '_queueadapter': {
#                     **self.hpc_specs['single_node_job_queueadapter_defaults'],  # get 1 node
#                 },
#                 '_files_in':  files_in,
#                 '_files_out': files_out,
#                 'metadata': {
#                     'project': self.project_id,
#                     'datetime': str(datetime.datetime.now()),
#                     'step':    step_label,
#                     **self.kwargs,
#                 }
#             },
#             parents=fws_root)
#
#         fw_list.append(fw_gmx_mdrun)
#
#         return fw_list, [fw_gmx_mdrun], [fw_gmx_mdrun]


# class GromacsRestartSubWorkflowGenerator(
#         DefaultPullMixin, DefaultPushMixin,
#         ProcessAnalyzeAndVisualizeSubWorkflowGenerator,
#         ):
#     def __init__(self, *args, **kwargs):
#         sub_wf_name = 'GromacsRestart'
#         if 'wf_name_prefix' not in kwargs:
#             kwargs['wf_name_prefix'] = sub_wf_name
#         else:
#             kwargs['wf_name_prefix'] = ':'.join((kwargs['wf_name_prefix'], sub_wf_name))
#         ProcessAnalyzeAndVisualizeSubWorkflowGenerator.__init__(self,
#             main_sub_wf=GromacsRestartMain(*args, **kwargs),
#             analysis_sub_wf=GromacsVacuumTrajectoryAnalysisSubWorkflowGenerator(*args, **kwargs),
#             vis_sub_wf=GromacsTrajectoryVisualizationSubWorkflowGenerator(*args, **kwargs),
#             *args, **kwargs)


class GromacsRelaxationRecoverMain(SubWorkflowGenerator):
    """
    NPT relaxation with GROMACS without restraints on ions.

    dynamic infiles:
    - log_file:        default.log
        tagged as {'metadata->type': 'relax_log'}
    - energy_file:     default.edr
        tagged as {'metadata->type': 'relax_edr'}
    - trajectory_file: default.trr
        tagged as {'metadata->type': 'relax_trr'}
    - data_file:       default.gro
        tagged as {'metadata->type': 'relax_gro'}

    - index_file:      default.ndx
        pass through untouched
    - topology_file:   default.top
        pass through untouched

    outfiles:
    - log_file:        default.log
        tagged as {'metadata->type': 'relax_log'}
    - energy_file:     default.edr
        tagged as {'metadata->type': 'relax_edr'}
    - trajectory_file: default.trr
        tagged as {'metadata->type': 'relax_trr'}
    - data_file:       default.gro
        tagged as {'metadata->type': 'relax_gro'}

    - index_file:      default.ndx
        pass through untouched
    - topology_file:   default.top
        pass through untouched

    """

    def __init__(self, *args, **kwargs):
        sub_wf_name = 'GromacsRecoverMain'
        if 'wf_name_prefix' not in kwargs:
            kwargs['wf_name_prefix'] = sub_wf_name
        else:
            kwargs['wf_name_prefix'] = ':'.join((kwargs['wf_name_prefix'], sub_wf_name))
        super().__init__(*args, **kwargs)

    def main(self, fws_root=[]):
        fw_list = []

        # GMX mdrun restart
        # -----------------
        files_in = {
            'checkpoint_file': 'default.cpt',
            'log_file':        'default.log',
            'energy_file':     'default.edr',
            'trajectory_file': 'default.trr',
            'compressed_traj_file': 'default.xtc',
            'data_file':       'default.gro',
            'input_file':      'default.tpr',
            'topology_file':   'default.top',
            'index_file':      'default.ndx',
        }
        files_out = {
            'log_file':        'default.log',
            'energy_file':     'default.edr',
            'trajectory_file': 'default.trr',
            'compressed_traj_file': 'default.xtc',
            'data_file':       'default.gro',
            'topology_file':   'default.top',  # pass through untouched
            'index_file':      'default.ndx',  # pass through untouched
        }

        step_label = self.get_step_label('gmx_mdrun')

        fts_gmx_mdrun = [CmdTask(
            cmd='gmx',
            opt=['mdrun', '-v', '-deffnm', 'default', '-cpi', 'default'],
            env='python',
            stderr_file='std.err',
            stdout_file='std.out',
            stdlog_file='std.log',
            store_stdout=True,
            store_stderr=True,
            fizzle_bad_rc=True)]

        # as many spec as possible derived from fizzled parent
        fw_gmx_mdrun = Firework(fts_gmx_mdrun,
                                name=self.get_fw_label(step_label),
                                spec={
                                    '_files_in':  files_in,
                                    '_files_out': files_out,
                                })

        restart_wf = Workflow([fw_gmx_mdrun])

        # recovery
        # --------

        files_in = {
            'log_file':        'default.log',
            'energy_file':     'default.edr',
            'trajectory_file': 'default.trr',
            'compressed_traj_file': 'default.xtc',
            'data_file':       'default.gro',
            'topology_file':   'default.top',
            'index_file':      'default.ndx',
        }

        files_out = {
            'checkpoint_file': 'default.cpt',
            'log_file':        'default.log',
            'energy_file':     'default.edr',
            'trajectory_file': 'default.trr',
            'compressed_traj_file': 'default.xtc',
            'data_file':       'default.gro',
            'topology_file':   'default.top',
            'input_file':      'default.tpr',
            'index_file':      'default.ndx',
        }

        step_label = self.get_step_label('gmx_recovery')

        fts_gmx_recovery = [RecoverTask(
            restart_wf=restart_wf.as_dict(),
            superpose_restart_on_parent_fw_spec=True,
            repeated_recover_fw_name=step_label,
            max_restarts=2,
            fizzle_on_no_restart_file=False,
            restart_file_glob_patterns='default.cpt',
            other_glob_patterns=[
                "default.gro",
                "default.edr",
                "default.ndx",
                "default.tpr",
                "default.xtc",
                "default.log",
                "default.top",
                "default.trr",
            ],
            restart_counter='metadata->step_specific->gmx_relaxation->restart_count',
            store_stdlog=True,
            stdlog_file='std.log',
            loglevel=logging.DEBUG)]

        fw_gmx_recovery = Firework(
            fts_gmx_recovery,
            name=self.get_fw_label(step_label),
            spec={
                '_allow_fizzled_parents': True,
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
            parents=[*fws_root])

        fw_list.append(fw_gmx_recovery)

        return fw_list, [fw_gmx_recovery], [fw_gmx_recovery]
