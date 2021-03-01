# -*- coding: utf-8 -*-
import datetime
import glob
import os
import pymongo

from fireworks.user_objects.firetasks.filepad_tasks import GetFilesByQueryTask
from imteksimfw.fireworks.user_objects.firetasks.cmd_tasks import CmdTask

from jlhpy.utilities.wf.workflow_generator import (
    WorkflowGenerator, ProcessAnalyzeAndVisualize)
from jlhpy.utilities.wf.mixin.mixin_wf_storage import (
   DefaultPullMixin, DefaultPushMixin)

from jlhpy.utilities.wf.building_blocks.sub_wf_gromacs_analysis import GromacsVacuumTrajectoryAnalysis
from jlhpy.utilities.wf.building_blocks.sub_wf_gromacs_vis import GromacsTrajectoryVisualization
import jlhpy.utilities.wf.file_config as file_config


class GromacsEnergyMinimizationMain(WorkflowGenerator):
    """
    Energy minimization with GROMACS.

    dynamic infiles:
        only queried in pull stub, otherwise expected through data flow

    - data_file:     default.gro
        queried by { 'metadata->type': 'initial_config_gro' }
    - topology_file: default.top
        queried by { 'metadata->type': 'initial_config_top' }
    - restraint_file: default.posre.itp
        queried by { 'metadata->type': 'initial_config_posre_itp' }

    static infiles:
        always queried within main trunk

    - parameter_file: default.mdp,
        queried by {'metadata->name': file_config.GMX_EM_MDP}


    outfiles:

    - log_file:        em.log
        tagged as {'metadata->type': 'em_log'}
    - energy_file:     em.edr
        tagged as {'metadata->type': 'em_edr'}
    - trajectory_file: em.xtc
        tagged as {'metadata->type': 'em_xtc'}
    - data_file:       em.gro
        tagged as {'metadata->type': 'em_gro'}
    """
    def push_infiles(self, fp):

        # static infiles for main
        # -----------------------
        step_label = self.get_step_label('push_infiles')

        infiles = sorted(glob.glob(os.path.join(
            self.infile_prefix,
            file_config.GMX_MDP_SUBDIR,
            file_config.GMX_EM_MDP)))

        files = {os.path.basename(f): f for f in infiles}

        # metadata common to all these files
        metadata = {
            'project': self.project_id,
            'type': 'input',
            'name': file_config.GMX_EM_MDP,
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

        fts_pull = [
            GetFilesByQueryTask(
                query={
                    'metadata->project': self.project_id,
                    'metadata->name':    file_config.GMX_EM_MDP,
                },
                sort_key='metadata.datetime',
                sort_direction=pymongo.DESCENDING,
                limit=1,
                new_file_names=['default.mdp'])]

        fw_pull = self.build_fw(
            fts_pull, step_label,
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_noqueue_category'])

        fw_list.append(fw_pull)

        # GMX grompp
        # ----------
        step_label = self.get_step_label('gmx_grompp')

        files_in = {
            'input_file':      'default.mdp',
            'data_file':       'default.gro',
            'topology_file':   'default.top',
            'restraint_file':  'default.posre.itp'}
        files_out = {
            'input_file': 'default.tpr',
            'parameter_file': 'mdout.mdp',
            'topology_file': 'default.top',  # passed throught unmodified
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

        fw_gmx_grompp = self.build_fw(
            fts_gmx_grompp, step_label,
            parents=[*fws_root, fw_pull],
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_queue_category'],
            queueadapter=self.hpc_specs['quick_single_core_job_queueadapter_defaults'])

        fw_list.append(fw_gmx_grompp)


        # GMX mdrun
        # ---------
        step_label = self.get_step_label('gmx_mdrun')

        files_in = {
            'input_file':   'default.tpr',
            'energy_file':  'default.edr',
        }
        files_out = {
            'log_file':        'default.log',
            'energy_file':     'default.edr',
            'uncompressed_trajectory_file': 'default.trr',
            'data_file':       'default.gro',
            'topology_file':   'default.top',  # passed throught unmodified
            'run_file':        'default.tpr',  # passed throught unmodified
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

        fw_gmx_mdrun = self.build_fw(
            fts_gmx_mdrun, step_label,
            parents=[fw_gmx_grompp],
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_queue_category'],
            queueadapter=self.hpc_specs['single_node_job_queueadapter_defaults'])

        fw_list.append(fw_gmx_mdrun)

        # for some mysterious reason, energy minimization won't write xtc directly

        # GMX trjconv
        # ---------
        step_label = self.get_step_label('gmx_trjconv')

        files_in = {
            'run_file': 'default.tpr',
            'uncompressed_trajectory_file': 'default.trr',
        }
        files_out = {
            'trajectory_file': 'default.xtc',
        }

        fts_gmx_trjconv = [CmdTask(
            cmd='gmx',
            opt=['trjconv',
                 '-f', 'default.trr',
                 '-s', 'default.tpr',
                 '-o', 'default.xtc'],
            env='python',
            stdin_key='stdin',
            stderr_file='std.err',
            stdout_file='std.out',
            stdlog_file='std.log',
            store_stdout=True,
            store_stderr=True,
            fizzle_bad_rc=True)]

        fw_gmx_trjconv = self.build_fw(
            fts_gmx_trjconv, step_label,
            parents=[fw_gmx_mdrun],
            files_in=files_in,
            files_out=files_out,
            stdin='0\n',  # select the whole system
            category=self.hpc_specs['fw_queue_category'],
            queueadapter=self.hpc_specs['single_task_job_queueadapter_defaults'])

        fw_list.append(fw_gmx_trjconv)

        return fw_list, [fw_gmx_mdrun, fw_gmx_trjconv], [fw_gmx_grompp]

class GromacsEnergyMinimization(
        DefaultPullMixin, DefaultPushMixin,
        ProcessAnalyzeAndVisualize,
        ):
    def __init__(self, *args, **kwargs):
        ProcessAnalyzeAndVisualize.__init__(self,
            main_sub_wf=GromacsEnergyMinimizationMain,
            analysis_sub_wf=GromacsVacuumTrajectoryAnalysis,
            vis_sub_wf=GromacsTrajectoryVisualization,
            *args, **kwargs)
