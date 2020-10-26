# -*- coding: utf-8 -*-
"""Indenter bounding sphere sub workflow."""

import glob
import logging
import os
import pymongo

from fireworks import Firework
from fireworks.user_objects.firetasks.filepad_tasks import GetFilesByQueryTask
from fireworks.user_objects.firetasks.filepad_tasks import AddFilesTask
from fireworks.user_objects.firetasks.templatewriter_task import TemplateWriterTask
from imteksimfw.fireworks.user_objects.firetasks.cmd_tasks import CmdTask
from imteksimfw.fireworks.user_objects.firetasks.recover_tasks import RecoverTask

from jlhpy.utilities.wf.workflow_generator import (
    WorkflowGenerator, ProcessAnalyzeAndVisualize)
from jlhpy.utilities.wf.mixin.mixin_wf_storage import (
   DefaultPullMixin, DefaultPushMixin)

from jlhpy.utilities.wf.building_blocks.sub_wf_gromacs_analysis import GromacsDefaultTrajectoryAnalysis
from jlhpy.utilities.wf.building_blocks.sub_wf_gromacs_vis import GromacsTrajectoryVisualization

import jlhpy.utilities.wf.file_config as file_config


class GromacsNPTEquilibrationMain(WorkflowGenerator):
    """
    NPT equilibration with GROMACS.

    dynamic infiles:
        only queried in pull stub, otherwise expected through data flow

    - data_file:       default.gro
        tagged as {'metadata->type': 'nvt_gro'}
    - index_file:      default.ndx
        tagged as {'metadata->type': 'nvt_ndx'}
    - topology_file: default.top
        queried by { 'metadata->type': 'solvate_top' }

    static infiles:
        always queried within main trunk

    - parameter_file: default.mdp,
        queried by {'metadata->name': file_config.GMX_NPT_Z_ONLY_MDP}

    outfiles:
    - log_file:        default.log
        tagged as {'metadata->type': 'npt_log'}
    - energy_file:     default.edr
        tagged as {'metadata->type': 'npt_edr'}
    - trajectory_file: default.xtc
        tagged as {'metadata->type': 'npt_xtc'}
    - data_file:       default.gro
        tagged as {'metadata->type': 'npt_gro'}

    - index_file:      default.ndx
        pass through untouched
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
            file_config.GMX_NPT_Z_ONLY_MDP)))

        files = {os.path.basename(f): f for f in infiles}

        # metadata common to all these files
        metadata = {
            'project': self.project_id,
            'type': 'input',
            'name': file_config.GMX_NPT_Z_ONLY_MDP,
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
                    'metadata->name':    file_config.GMX_NPT_Z_ONLY_MDP,
                },
                sort_key='metadata.datetime',
                sort_direction=pymongo.DESCENDING,
                limit=1,
                new_file_names=['default.mdp'])]

        fw_pull_mdp = self.build_fw(
            fts_pull_mdp, step_label,
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_noqueue_category'])

        fw_list.append(fw_pull_mdp)

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
                 '-maxwarn', 2,
                ],
            env='python',
            stderr_file='std.err',
            stdout_file='std.out',
            stdlog_file='std.log',
            store_stdout=True,
            store_stderr=True,
            fizzle_bad_rc=True)]
        # -maxwarn 2 allows for the following two warnings:
        #
        # WARNING 1 [file default.mdp]:
        #   Some atoms are not part of any center of mass motion removal group.
        #   This may lead to artifacts.
        #   In most cases one should use one group for the whole system.
        #
        # WARNING 2 [file default.mdp]:
        #   You are using pressure coupling with absolute position restraints, this
        #   will give artifacts. Use the refcoord_scaling option.

        # with anisotropic pressure coupling
        # WARNING 1 [file default.mdp, line 141]:
        #  All off-diagonal reference pressures are non-zero. Are you sure you want
        #  to apply a threefold shear stress?

        fw_gmx_grompp = self.build_fw(
            fts_gmx_grompp, step_label,
            parents=[*fws_root, fw_pull_mdp],
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_queue_category'],
            queueadapter=self.hpc_specs['quick_single_core_job_queueadapter_defaults'])

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
            queueadapter=self.hpc_specs['four_nodes_job_queueadapter_defaults'])

        fw_list.append(fw_gmx_mdrun)

        return fw_list, [fw_gmx_mdrun], [fw_gmx_grompp]


class GromacsNPTEquilibration(
        DefaultPullMixin, DefaultPushMixin,
        ProcessAnalyzeAndVisualize,
        ):
    def __init__(self, *args, **kwargs):
        super().__init__(
            main_sub_wf=GromacsNPTEquilibrationMain,
            analysis_sub_wf=GromacsDefaultTrajectoryAnalysis,
            vis_sub_wf=GromacsTrajectoryVisualization,
            *args, **kwargs)
