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
from jlhpy.utilities.wf.mixin.mixin_wf_gromacs_analysis import GromacsVacuumTrajectoryAnalysisMixin

import jlhpy.utilities.wf.file_config as file_config


class GromacsPullSubWorkflowGenerator(
        GromacsVacuumTrajectoryAnalysisMixin, SubWorkflowGenerator):
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

    vis static infiles:
    - script_file: renumber_png.sh,
        queried by {'metadata->name': file_config.BASH_RENUMBER_PNG}
    - template_file: default.pml.template,
        queried by {'metadata->name': file_config.PML_MOVIE_TEMPLATE}

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

    vis outfiles:
    - mp4_file: default.mp4
        tagged as {'metadata->type': 'mp4_file'}
    """

    def __init__(self, *args, **kwargs):
        if 'wf_name_prefix' not in kwargs:
            kwargs['wf_name_prefix'] = 'GROMACS pull sub-workflow'
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

    def pull(self, fws_root=[]):
        fw_list = []

        step_label = self.get_step_label('pull')

        files_in = {}
        files_out = {
            'data_file': 'default.gro',
            'topology_file':   'default.top',
            #'restraint_file':  'default.posre.itp',
            'input_file':      'default.mdp',
            'index_file':      'default.ndx',
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
                new_file_names=['default.gro']),
            GetFilesByQueryTask(
                query={
                    'metadata->project':    self.source_project_id,
                    'metadata->type':       'pull_gro',
                    **self.parameter_dict
                },
                sort_key='metadata.datetime',
                sort_direction=pymongo.DESCENDING,
                limit=1,
                new_file_names=['default.top']),
            GetFilesByQueryTask(
                query={
                    'metadata->project':    self.source_project_id,
                    'metadata->type':       'pull_top',
                    **self.parameter_dict
                },
                sort_key='metadata.datetime',
                sort_direction=pymongo.DESCENDING,
                limit=1,
                new_file_names=['default.ndx']),
            GetFilesByQueryTask(
                query={
                    'metadata->project':    self.source_project_id,
                    'metadata->type':       'pull_ndx',
                    **self.parameter_dict
                },
                sort_key='metadata.datetime',
                sort_direction=pymongo.DESCENDING,
                limit=1,
                new_file_names=['default.mdp']),
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

        # GMX grompp
        # ----------
        step_label = self.get_step_label('gmx grompp')

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
        step_label = self.get_step_label('gmx mdrun')

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
                    **self.hpc_specs['no_smt_job_queueadapter_defaults'],
                    # NOTE: JUWELS GROMACS
                    # module("load","Stages/2019a","Intel/2019.3.199-GCC-8.3.0","IntelMPI/2019.3.199")
                    # module("load","GROMACS/2019.3","GROMACS-Top/2019.3")
                    # fails with segmentation fault when using SMT (96 logical cores)
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

        files_in = {
            'log_file':        'default.log',
            'energy_file':     'default.edr',
            'trajectory_file': 'default.trr',
            'compressed_trajectory_file': 'default.xtc',
            'data_file':       'default.gro',
            'pullf_file':      'default_pullf.xvg',
            'pullx_file':      'default_pullx.xvg',
        }
        files_out = {}

        fts_push = [
            AddFilesTask({
                'compress': True,
                'paths': "default.log",
                'metadata': {
                    'project': self.project_id,
                    'datetime': str(datetime.datetime.now()),
                    'type':    'pull_log',
                }
            }),
            AddFilesTask({
                'compress': True,
                'paths': "default.edr",
                'metadata': {
                    'project': self.project_id,
                    'datetime': str(datetime.datetime.now()),
                    'type':    'pull_edr',
                }
            }),
            AddFilesTask({
                'compress': True,
                'paths': "default.trr",
                'metadata': {
                    'project': self.project_id,
                    'datetime': str(datetime.datetime.now()),
                    'type':    'pull_trr',
                }
            }),
            AddFilesTask({
                'compress': True,
                'paths': "default.xtc",
                'metadata': {
                    'project': self.project_id,
                    'datetime': str(datetime.datetime.now()),
                    'type':    'pull_xtc',
                }
            }),
            AddFilesTask({
                'compress': True,
                'paths': "default.gro",
                'metadata': {
                    'project': self.project_id,
                    'datetime': str(datetime.datetime.now()),
                    'type':    'pull_gro',
                }
            }),
            AddFilesTask({
                'compress': True,
                'paths': "default_pullf.xvg",
                'metadata': {
                    'project': self.project_id,
                    'datetime': str(datetime.datetime.now()),
                    'type':    'pullf_xvg',
                }
            }),
            AddFilesTask({
                'compress': True,
                'paths': "default_pullx.xvg",
                'metadata': {
                    'project': self.project_id,
                    'datetime': str(datetime.datetime.now()),
                    'type':    'pullx_xvg',
                }
            })
        ]

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

    def vis_main(self, fws_root=[]):
        fw_list = []

        # pull pymol template
        # -------------------

        step_label = self.get_step_label('vis pull pymol template')

        files_in = {}
        files_out = {
            'template_file': 'default.pml.template',
        }

        fts_pull_pymol_template = [
            GetFilesByQueryTask(
                query={
                    'metadata->project':    self.project_id,
                    'metadata->name':       file_config.PML_MOVIE_TEMPLATE,
                },
                sort_key='metadata.datetime',
                sort_direction=pymongo.DESCENDING,
                limit=1,
                new_file_names=['default.pml.template'])]

        fw_pull_pymol_template = Firework(fts_pull_pymol_template,
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
            parents=[])

        fw_list.append(fw_pull_pymol_template)

        # PYMOL input script template
        # -----------------------------
        step_label = self.get_step_label('vis pymol template')

        files_in =  {'template_file': 'default.template'}
        files_out = {'input_file': 'default.pml'}

        # Jinja2 context:
        static_context = {
            'header': ', '.join((
                self.project_id,
                self.get_fw_label(step_label),
                str(datetime.datetime.now()))),
        }

        dynamic_context = {
            'counterion': 'metadata->system->counterion->resname',
            'solvent': 'metadata->system->solvent->resname',
            'substrate':  'metadata->system->substrate->resname',
            'surfactant': 'metadata->system->surfactant->resname',
        }

        ft_template = TemplateWriterTask({
            'context': static_context,
            'context_inputs': dynamic_context,
            'template_file': 'default.template',
            'template_dir': '.',
            'output_file': 'default.pml'})

        fw_template = Firework([ft_template],
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
            parents=[fw_pull_pymol_template])

        fw_list.append(fw_template)

        # pull renumber bash script
        # -------------------------

        step_label = self.get_step_label('vis pull renumber bash script')

        files_in = {}
        files_out = {
            'script_file': 'renumber_png.sh',
        }

        fts_pull_renumber_bash_script = [
            GetFilesByQueryTask(
                query={
                    'metadata->project':    self.project_id,
                    'metadata->name':       file_config.BASH_RENUMBER_PNG,
                },
                sort_key='metadata.datetime',
                sort_direction=pymongo.DESCENDING,
                limit=1,
                new_file_names=['renumber_png.sh'])]

        fw_pull_renumber_bash_script = Firework(fts_pull_renumber_bash_script,
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
            parents=[])

        fw_list.append(fw_pull_renumber_bash_script)

        # Render trajectory
        # ----------------
        step_label = self.get_step_label('vis pymol')

        files_in = {
            'data_file': 'default.gro',
            'trajectory_file': 'default.trr',
            'input_file': 'default.pml',
            'script_file': 'renumber_png.sh',
        }
        files_out = {
            'mp4_file': 'default.mp4',
        }

        fts_vis = [
            CmdTask(
                cmd='pymol',
                opt=['-c', 'default.pml', '--',
                     'default.gro',  # positional arguments to pml script
                     'default.trr',
                     'frame',  # prefix to png out files
                     1,  # starting frame
                    ],
                env='python',
                stderr_file='pymol_std.err',
                stdout_file='pymol_std.out',
                stdlog_file='pymol_std.log',
                store_stdout=True,
                store_stderr=True,
                fizzle_bad_rc=True
            ),
            CmdTask(
                cmd='bash',
                opt=['renumber_png.sh',
                     'frame',
                    ],
                stderr_file='renumber_std.err',
                stdout_file='renumber_std.out',
                stdlog_file='renumber_std.log',
                store_stdout=True,
                store_stderr=True,
                fizzle_bad_rc=True,
            ),
            # standard format from https://github.com/pastewka/GroupWiki/wiki/Make-movies
            # ffmpeg -r 60 -f image2 -i frame%04d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p test.mp4
            CmdTask(
                cmd='ffmpeg',
                opt=[
                    '-r', 30,  # frame rate
                    '-f', 'image2',
                    '-i', 'frame%06d.png',
                    '-vcodec', 'libx264',
                    '-crf', 25,
                    '-pix_fmt', 'yuv420p',
                    'default.mp4'
                    ],
                env='python',
                stderr_file='ffmpeg_std.err',
                stdout_file='ffmpeg_std.out',
                stdlog_file='ffmpeg_std.log',
                store_stdout=True,
                store_stderr=True,
                fizzle_bad_rc=True)
            ]

        fw_vis = Firework(fts_vis,
            name=self.get_fw_label(step_label),
            spec={
                '_category': self.hpc_specs['fw_noqueue_category'],
                '_files_in':  files_in,
                '_files_out': files_out,
                'metadata': {
                    'project':  self.project_id,
                    'datetime': str(datetime.datetime.now()),
                    'step':     step_label,
                     **self.kwargs
                }
            },
            parents=[*fws_root, fw_template, fw_pull_renumber_bash_script]
        )

        fw_list.append(fw_vis)

        return fw_list, [fw_vis], [fw_template, fw_vis]

    def vis_push(self, fws_root=[]):
        fw_list = []

        step_label = self.get_step_label('vis_push')

        files_in = {'mp4_file': 'default.mp4'}
        files_out = {}

        fts_push = [AddFilesTask({
            'compress': True,
            'paths': "default.mp4",
            'metadata': {
                'project': self.project_id,
                'datetime': str(datetime.datetime.now()),
                'type':    'mp4_file',
            }
        })]

        fw_push = Firework(fts_push,
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

        fw_list.append(fw_push)

        return fw_list, [fw_push], [fw_push]
