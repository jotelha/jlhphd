# -*- coding: utf-8 -*-
"""Generic GROMACS trajectory visualization sub workflow."""

import datetime
import glob
import os
import pymongo

from abc import ABC, abstractmethod

from fireworks import Firework
from fireworks.user_objects.firetasks.filepad_tasks import GetFilesByQueryTask
from fireworks.user_objects.firetasks.filepad_tasks import AddFilesTask
from fireworks.user_objects.firetasks.templatewriter_task import TemplateWriterTask
from imteksimfw.fireworks.user_objects.firetasks.cmd_tasks import CmdTask

from jlhpy.utilities.wf.workflow_generator import WorkflowGenerator

import jlhpy.utilities.wf.file_config as file_config


class GromacsTrajectoryVisualizationWorkflowGenerator(WorkflowGenerator):
    """
    Visualize GROMACS trajectory with PyMol.

    vis static infiles:
    - script_file: renumber_png.sh,
        queried by {'metadata->name': file_config.BASH_RENUMBER_PNG}
    - template_file: default.pml.template,
        queried by {'metadata->name': file_config.PML_MOVIE_TEMPLATE}

    vis fw_spec inputs:
    - metadata->system->counterion->resname
    - metadata->system->solvent->resname
    - metadata->system->substrate->resname
    - metadata->system->surfactant->resname

    vis outfiles:
    - mp4_file: default.mp4
        tagged as {'metadata->type': 'mp4_file'}
    """
    def push_infiles(self, fp):
        step_label = self.get_step_label('push_infiles')

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

        fp_files = []

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

    def main(self, fws_root=[]):
        fw_list = []

        # pull pymol template
        # -------------------

        step_label = self.get_step_label('vis_pull_pymol_template')

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
        step_label = self.get_step_label('vis_pymol_template')

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

        step_label = self.get_step_label('vis_pull_renumber_bash_script')

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
        step_label = self.get_step_label('vis_pymol')

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

    # def push(self, fws_root=[]):
    #     fw_list = []
    #
    #     step_label = self.get_step_label('vis_push')
    #
    #     files_in = {'mp4_file': 'default.mp4'}
    #     files_out = {}
    #
    #     fts_push = [AddFilesTask({
    #         'compress': True,
    #         'paths': "default.mp4",
    #         'metadata': {
    #             'project': self.project_id,
    #             'datetime': str(datetime.datetime.now()),
    #             'type':    'mp4_file',
    #         }
    #     })]
    #
    #     fw_push = Firework(fts_push,
    #         name=self.get_fw_label(step_label),
    #         spec={
    #             '_category': self.hpc_specs['fw_noqueue_category'],
    #             '_files_in': files_in,
    #             '_files_out': files_out,
    #             'metadata': {
    #                 'project': self.project_id,
    #                 'datetime': str(datetime.datetime.now()),
    #                 'step':    step_label,
    #                  **self.kwargs
    #             }
    #         },
    #         parents=fws_root)
    #
    #     fw_list.append(fw_push)
    #
    #     return fw_list, [fw_push], [fw_push]
