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
from jlhpy.utilities.wf.mixin.mixin_wf_storage import PushToDtoolRepositoryMixin

import jlhpy.utilities.wf.file_config as file_config


class GromacsNVTEquilibrationSubWorkflowGenerator(
        GromacsVacuumTrajectoryAnalysisMixin,
        PushToDtoolRepositoryMixin,
        SubWorkflowGenerator):
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

    vis static infiles:
    - script_file: renumber_png.sh,
        queried by {'metadata->name': file_config.BASH_RENUMBER_PNG}
    - template_file: default.pml.template,
        queried by {'metadata->name': file_config.PML_MOVIE_TEMPLATE}

    outfiles:
    - log_file:        default.log
        tagged as {'metadata->type': 'nvt_log'}
    - energy_file:     default.edr
        tagged as {'metadata->type': 'nvt_edr'}
    - trajectory_file: default.trr
        tagged as {'metadata->type': 'nvt_trr'}
    - data_file:       default.gro
        tagged as {'metadata->type': 'nvt_gro'}
    - index_file:      default.ndx
        tagged as {'metadata->type': 'nvt_ndx'}

    - topology_file:   default.top
        pass through untouched

    vis outfiles:
    - mp4_file: default.mp4
        tagged as {'metadata->type': 'mp4_file'}
    """

    def __init__(self, *args, **kwargs):
        if 'wf_name_prefix' not in kwargs:
            kwargs['wf_name_prefix'] = 'GROMACS NVT equilibration sub-workflow'
        super().__init__(*args, **kwargs)

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
            'data_file':      'default.gro',
            'topology_file':  'default.top',
        }

        fts_pull = [
            GetFilesByQueryTask(
                query={
                    'metadata->project':    self.source_project_id,
                    'metadata->type':       'em_solvated_gro',
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
        step_label = self.get_step_label('gmx make_ndx')

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
        step_label = self.get_step_label('gmx non-Substrate group')

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
        step_label = self.get_step_label('gmx grompp')

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
        step_label = self.get_step_label('gmx mdrun')

        files_in = {
            'input_file':    'default.tpr',
            'topology_file': 'default.top',  # pass through untouched
            'index_file':    'default.ndx',  # pass through untouched
        }
        files_out = {
            'log_file':        'default.log',
            'energy_file':     'default.edr',
            'trajectory_file': 'default.trr',
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


    @property
    def files_out_list(self):
        return [
            {
                'file_label': 'log_file',
                'file_name':  'default.log',
                'type_label': 'nvt_log',
            },
            {
                'file_label': 'energy_file',
                'file_name':  'default.edr',
                'type_label': 'nvt_edr',
            },
            {
                'file_label': 'trajectory_file',
                'file_name':  'default.trr',
                'type_label': 'nvt_trr',
            },
            {
                'file_label': 'data_file',
                'file_name':  'default.gro',
                'type_label': 'nvt_gro',
            },
            {
                'file_label': 'index_file',
                'file_name':  'default.ndx',
                'type_label': 'nvt_ndx',
            },
        ]

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
