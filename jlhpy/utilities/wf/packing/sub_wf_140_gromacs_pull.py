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
    SubWorkflowGenerator, ProcessAnalyzeAndVisualizeSubWorkflowGenerator)
from jlhpy.utilities.wf.mixin.mixin_wf_storage import (
   DefaultPullMixin, DefaultPushMixin)

from jlhpy.utilities.wf.building_blocks.sub_wf_gromacs_analysis import GromacsVacuumTrajectoryAnalysisSubWorkflowGenerator
from jlhpy.utilities.wf.building_blocks.sub_wf_gromacs_vis import GromacsTrajectoryVisualizationSubWorkflowGenerator

import jlhpy.utilities.wf.file_config as file_config


class GromacsPullMain(SubWorkflowGenerator):
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

    static infiles:
        always queried within main trunk

    - template_file: sys.top.template,
        queried by {'metadata->name': file_config.GMX_PULL_TOP_TEMPLATE}
    - parameter_file: pull.mdp.template,
        queried by {'metadata->name': file_config.GMX_PULL_MDP_TEMPLATE}

    fw_spec inputs:
    - metadata->system->surfactant->nmolecules
    - metadata->system->surfactant->name
    - metadata->system->counterion->nmolecules
    - metadata->system->counterion->name
    - metadata->system->substrate->natoms
    - metadata->system->substrate->name

    - metadata->step_specific->pulling->pull_atom_name
    - metadata->step_specific->pulling->spring_constant
    - metadata->step_specific->pulling->rate

    outfiles:
        use regex replacement /'([^']*)':(\\s*)'([^']*)',/- $1:$2$3/
        to format from files_out dict

    - log_file:        default.log
        tagged as {'metadata->type': 'pull_log'}
    - energy_file:     default.edr
        tagged as {'metadata->type': 'pull_edr'}
    - trajectory_file: default.trr
        tagged as {'metadata->type': 'pull_trr'}
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

    def __init__(self, *args, **kwargs):
        sub_wf_name = 'GromacsPullMain'
        if 'wf_name_prefix' not in kwargs:
            kwargs['wf_name_prefix'] = sub_wf_name
        else:
            kwargs['wf_name_prefix'] = ':'.join((kwargs['wf_name_prefix'], sub_wf_name))
        super().__init__(*args, **kwargs)

    def push_infiles(self, fp):
        step_label = self.get_step_label('push_static_infiles')

        fp_files = []

        # static pymol infile for vis
        # ---------------------------
        infiles = sorted(glob.glob(os.path.join(
            self.infile_prefix,
            file_config.PML_SUBDIR,
            file_config.PML_MOVIE_TEMPLATE)))

        files = {os.path.basename(f): f for f in infiles}

        # metadata common to all these files
        metadata = {
            'project': self.project_id,
            'type': 'input',
            'name': file_config.PML_MOVIE_TEMPLATE,
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

        # static bash cript infile for vis
        # --------------------------------
        infiles = sorted(glob.glob(os.path.join(
            self.infile_prefix,
            file_config.BASH_SCRIPT_SUBDIR,
            file_config.BASH_RENUMBER_PNG)))

        files = {os.path.basename(f): f for f in infiles}

        # metadata common to all these files
        metadata = {
            'project': self.project_id,
            'type': 'input',
            'name': file_config.BASH_RENUMBER_PNG,
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
            parents=[*fws_root])

        fw_list.append(fw_gmx_grompp)


        # GMX mdrun
        # ---------
        step_label = self.get_step_label('gmx_mdrun')

        files_in = {
            'input_file': 'default.tpr',
            'topology_file':  'default.top',  # pass through unmodified
        }
        files_out = {
            'log_file':        'default.log',
            'energy_file':     'default.edr',
            'trajectory_file': 'default.trr',
            'compressed_trajectory_file': 'default.xtc',
            'data_file':       'default.gro',
            'pullf_file':      'default_pullf.xvg',
            'pullx_file':      'default_pullx.xvg',
            'topology_file':  'default.top',  # pass through unmodified
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

        fw_gmx_mdrun = Firework(fts_gmx_mdrun,
            name=self.get_fw_label(step_label),
            spec={
                '_category': self.hpc_specs['fw_queue_category'],
                '_queueadapter': {
                    **self.hpc_specs['quick_single_core_job_queueadapter_defaults'],
                    # NOTE: JUWELS GROMACS
                    # module("load","Stages/2019a","Intel/2019.3.199-GCC-8.3.0","IntelMPI/2019.3.199")
                    # module("load","GROMACS/2019.3","GROMACS-Top/2019.3")
                    # fails with segmentation fault when using SMT (96 logical cores)
                    # NOTE: later encountered problems with any parallelization
                    # run serial, only 1000 steps, to many issues with segementation faults
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


class GromacsPullSubWorkflowGenerator(
        DefaultPullMixin, DefaultPushMixin,
        ProcessAnalyzeAndVisualizeSubWorkflowGenerator,
        ):
    def __init__(self, *args, **kwargs):
        sub_wf_name = 'GromacsPull'
        if 'wf_name_prefix' not in kwargs:
            kwargs['wf_name_prefix'] = sub_wf_name
        else:
            kwargs['wf_name_prefix'] = ':'.join((kwargs['wf_name_prefix'], sub_wf_name))
        ProcessAnalyzeAndVisualizeSubWorkflowGenerator.__init__(self,
            main_sub_wf=GromacsPullMain(*args, **kwargs),
            analysis_sub_wf=GromacsVacuumTrajectoryAnalysisSubWorkflowGenerator(*args, **kwargs),
            vis_sub_wf=GromacsTrajectoryVisualizationSubWorkflowGenerator(*args, **kwargs),
            *args, **kwargs)
