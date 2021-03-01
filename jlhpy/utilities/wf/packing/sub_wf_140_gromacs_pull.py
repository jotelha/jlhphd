# -*- coding: utf-8 -*-
"""Indenter bounding sphere sub workflow."""

from imteksimfw.fireworks.user_objects.firetasks.cmd_tasks import CmdTask

from jlhpy.utilities.wf.workflow_generator import (
    WorkflowGenerator, ProcessAnalyzeAndVisualize)
from jlhpy.utilities.wf.mixin.mixin_wf_storage import (
   DefaultPullMixin, DefaultPushMixin)

from jlhpy.utilities.wf.building_blocks.sub_wf_gromacs_analysis import GromacsVacuumTrajectoryAnalysis
from jlhpy.utilities.wf.building_blocks.sub_wf_gromacs_vis import GromacsTrajectoryVisualization

import jlhpy.utilities.wf.file_config as file_config


class GromacsPullMain(WorkflowGenerator):
    """
    Pseudo-pulling via GROMACS.

    dynamic infiles:
        only queried in pull stub, otherwise expected through data flow

    - data_file:     default.gro
        queried by { 'metadata->type': 'initial_config_gro' }
    - topology_file: default.top
        queried by { 'metadata->type': 'pull_top' }
    - input_file:    default.mdp
        queried by { 'metadata->type': 'pull_mdp' }
    - index_file:    default.ndx
        queried by { 'metadata->type': 'pull_ndx' }

    outfiles:
        use regex replacement /'([^']*)':(\\s*)'([^']*)',/- $1:$2$3/
        to format from files_out dict

    - log_file:        default.log
        tagged as {'metadata->type': 'pull_log'}
    - energy_file:     default.edr
        tagged as {'metadata->type': 'pull_edr'}
    - trajectory_file: default.xtc
        tagged as {'metadata->type': 'pull_xtc'}
    - compressed_trajectory_file: default.xtc
        tagged as {'metadata->type': 'pull_xtc'}
    - data_file:       default.gro
        tagged as {'metadata->type': 'pull_gro'}
    - pullf_file:      default_pullf.xvg
        tagged as {'metadata->type': 'pullf_xvg'}
    - pullx_file:      default_pullx.xvg
        tagged as {'metadata->type': 'pullx_xvg'}

    - topology_file:  default.top
        passed through unmodified
    """

    def main(self, fws_root=[]):
        fw_list = []

        # GMX grompp
        # ----------
        step_label = self.get_step_label('gmx_grompp')

        files_in = {
            'input_file':      'default.mdp',
            'index_file':      'default.ndx',
            'data_file':       'default.gro',
            'topology_file':   'default.top',
        }
        files_out = {
            'input_file':     'default.tpr',
            'parameter_file': 'mdout.mdp',
            'topology_file':  'default.top',  # pass through unmodified
            'index_file':     'default.ndx',  # pass through untouched
        }

        fts_gmx_grompp = [CmdTask(
            cmd='gmx',
            opt=['grompp',
                 '-f', 'default.mdp',  # parameter file
                 '-n', 'default.ndx',  # index file
                 '-c', 'default.gro',  # coordinates file
                 '-r', 'default.gro',  # restraint positions
                 '-p', 'default.top',  # topology file
                 '-o', 'default.tpr',  # compiled output
                ],
            env='python',
            stderr_file='std.err',
            stdout_file='std.out',
            stdlog_file='std.log',
            store_stdout=True,
            store_stderr=True,
            store_stdlog=False,
            fizzle_bad_rc=True)]

        fw_gmx_grompp = self.build_fw(
            fts_gmx_grompp, step_label,
            parents=fws_root,
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_queue_category'],
            queueadapter=self.hpc_specs['quick_single_core_job_queueadapter_defaults'])

        fw_list.append(fw_gmx_grompp)


        # GMX mdrun
        # ---------
        step_label = self.get_step_label('gmx_mdrun')

        files_in = {
            'input_file': 'default.tpr',
            'topology_file':  'default.top',  # pass through unmodified
            'index_file': 'default.ndx',  # pass through untouched
        }
        files_out = {
            'log_file':        'default.log',
            'energy_file':     'default.edr',
            'trajectory_file': 'default.xtc',
            'compressed_trajectory_file': 'default.xtc',
            'data_file':       'default.gro',
            'pullf_file':      'default_pullf.xvg',
            'pullx_file':      'default_pullx.xvg',
            'topology_file':   'default.top',  # pass through unmodified
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
            store_stdout=True,
            store_stderr=True,
            fizzle_bad_rc=True)]

        # NOTE: JUWELS GROMACS
        # module("load","Stages/2019a","Intel/2019.3.199-GCC-8.3.0","IntelMPI/2019.3.199")
        # module("load","GROMACS/2019.3","GROMACS-Top/2019.3")
        # fails with segmentation fault when using SMT (96 logical cores)
        # NOTE: later encountered problems with any parallelization
        # run serial, only 1000 steps, to many issues with segmentation faults
        fw_gmx_mdrun = self.build_fw(
            fts_gmx_mdrun, step_label,
            parents=[fw_gmx_grompp],
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_queue_category'],
            queueadapter=self.hpc_specs['quick_single_core_job_queueadapter_defaults'])

        fw_list.append(fw_gmx_mdrun)

        return fw_list, [fw_gmx_mdrun], [fw_gmx_grompp]


class GromacsPull(
        DefaultPullMixin, DefaultPushMixin,
        ProcessAnalyzeAndVisualize,
        ):
    def __init__(self, *args, **kwargs):
        ProcessAnalyzeAndVisualize.__init__(self,
            main_sub_wf=GromacsPullMain,
            analysis_sub_wf=GromacsVacuumTrajectoryAnalysis,
            vis_sub_wf=GromacsTrajectoryVisualization,
            *args, **kwargs)
