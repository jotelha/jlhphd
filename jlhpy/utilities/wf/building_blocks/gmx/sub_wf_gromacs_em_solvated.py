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

from jlhpy.utilities.wf.workflow_generator import (
    WorkflowGenerator, ProcessAnalyzeAndVisualize)

from jlhpy.utilities.wf.mixin.mixin_wf_storage import (
   DefaultPullMixin, DefaultPushMixin)

from jlhpy.utilities.wf.building_blocks.sub_wf_gromacs_analysis import GromacsDefaultTrajectoryAnalysis
from jlhpy.utilities.wf.building_blocks.sub_wf_gromacs_vis import GromacsTrajectoryVisualization

import jlhpy.utilities.wf.file_config as file_config


class GromacsEnergyMinimizationAfterSolvationMain(WorkflowGenerator):
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

    vis static infiles:
    - script_file: renumber_png.sh,
        queried by {'metadata->name': file_config.BASH_RENUMBER_PNG}
    - template_file: default.pml.template,
        queried by {'metadata->name': file_config.PML_MOVIE_TEMPLATE}

    outfiles:
    - log_file:        em.log
        tagged as {'metadata->type': 'em_solvated_log'}
    - energy_file:     em.edr
        tagged as {'metadata->type': 'em_solvated_edr'}
    - trajectory_file: em.trr
        tagged as {'metadata->type': 'em_solvated_trr'}
    - data_file:       em.gro
        tagged as {'metadata->type': 'em_solvated_gro'}

    - topology_file:  default.top
        passed through unmodified

    vis outfiles:
    - mp4_file: default.mp4
        tagged as {'metadata->type': 'mp4_file'}
    """

    def push_infiles(self, fp):
        # static infiles for main
        # -----------------------
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
        }
        files_out = {
            'input_file':     'default.tpr',
            'parameter_file': 'mdout.mdp',
            'topology_file':  'default.top',  # passed throught unmodified
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
            'topology_file': 'default.top',  # passed throught unmodified
        }
        files_out = {
            'log_file':        'default.log',
            'energy_file':     'default.edr',
            'trajectory_file': 'default.trr',
            'data_file':       'default.gro',
            'topology_file':   'default.top',  # passed throught unmodified
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

        return fw_list, [fw_gmx_mdrun], [fw_gmx_grompp]


class GromacsEnergyMinimizationAfterSolvation(
        DefaultPullMixin, DefaultPushMixin,
        ProcessAnalyzeAndVisualize,
        ):
    def __init__(self, *args, **kwargs):
        super().__init__(
            main_sub_wf=GromacsEnergyMinimizationAfterSolvationMain,
            analysis_sub_wf=GromacsDefaultTrajectoryAnalysis,
            vis_sub_wf=GromacsTrajectoryVisualization,
            *args, **kwargs)
