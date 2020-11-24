# -*- coding: utf-8 -*-
"""Generic GROMACS trajectory visualization sub workflow."""

import datetime
import glob
import os
import pymongo

from fireworks import Firework
from fireworks.user_objects.firetasks.filepad_tasks import GetFilesByQueryTask
from fireworks.user_objects.firetasks.templatewriter_task import TemplateWriterTask
from imteksimfw.fireworks.user_objects.firetasks.cmd_tasks import CmdTask, PickledPyEnvTask
from imteksimfw.utils.serialize import serialize_module_obj

from jlhpy.utilities.wf.workflow_generator import WorkflowGenerator

import jlhpy.utilities.wf.file_config as file_config


def count_files_by_glob_pattern(glob_pattern='*'):
    import glob
    return len(glob.glob(glob_pattern))


def ffmpeg_frame_file_template_from_nframes(nframes):
    import numpy as np
    width = np.ceil(np.log10(nframes)).astype(int)
    return 'default_%0{}d.png'.format(width)

class GromacsTrajectoryVisualization(WorkflowGenerator):
    """
    Visualize GROMACS trajectory with PyMol.

    vis static infiles:
    - script_file: renumber_png.sh,
        queried by {'metadata->name': file_config.BASH_RENUMBER_PNG}
    - template_file: default.pml.template,
        queried by {'metadata->name': file_config.PML_VIEW_TEMPLATE}

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
            file_config.PML_VIEW_TEMPLATE)))

        files = {os.path.basename(f): f for f in infiles}

        # metadata common to all these files
        metadata = {
            'project': self.project_id,
            'type': 'input',
            'name': file_config.PML_VIEW_TEMPLATE,
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
                    'metadata->name':       file_config.PML_VIEW_TEMPLATE,
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

        files_in = {'template_file': 'default.template'}
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

        # Render trajectory
        # ----------------
        step_label = self.get_step_label('vis_pymol')

        files_in = {
            'data_file':       'default.gro',
            'trajectory_file': 'default.xtc',
            'run_file':        'default.tpr',
            'input_file':      'default.pml',
        }
        files_out = {
            'mp4_file': 'default.mp4',
        }

        fts_vis = [
            # split trajectory in chunks for parallel processing
            CmdTask(
                cmd='python',
                opt=[
                    '-m', 'mpi4py.futures',  # mpi4py.futures wrapper necessary for static process allocation
                    '-m', 'imteksimcs.mpi4py.mpi_pool_executor',
                    'imteksimcs.GROMACS.gmx_split_traj.split_traj_by_mpi_ranks',
                    # {'key': 'metadata->step_specific->gromacs_vis->serial_chunks'},  # use default 4
                    ],
                env='mdanalysis',  # needs mdanalysis, gromacswrapper, mpi4py
                stderr_file='split_std.err',
                stdout_file='split_std.out',
                stdlog_file='split_std.log',
                store_stdout=True,
                store_stderr=True,
                fizzle_bad_rc=True
            ),

            # use default arguments of
            #   render_chunks(struc_file='default.gro', chunk_file_glob_pattern='default_*.xtc',
            #                 out_prefix="default_", pymol_func=run_pymol, **kwargs):
            # and
            #   run_pymol(pml, struc_file="default.gro", traj_file="default.xtc",
            #             out_prefix="default_", pml_script="default.pml"):
            CmdTask(
                cmd='python',
                opt=[
                    '-m', 'mpi4py.futures',  # mpi4py.futures wrapper necessary for static process allocation
                    '-m', 'imteksimcs.mpi4py.mpi_pool_executor',
                    'imteksimcs.PyMOL.pymol_mpi.render_chunks',
                    ],
                env='pymol',  # needs mpi4py and pymol
                stderr_file='pymol_std.err',
                stdout_file='pymol_std.out',
                stdlog_file='pymol_std.log',
                store_stdout=True,
                store_stderr=True,
                fizzle_bad_rc=True
            ),

            # count frames
            PickledPyEnvTask(
                func=serialize_module_obj(count_files_by_glob_pattern),
                args=['*.png'],
                env='imteksimpy',
                outputs=['run->gromacs_vis->nframes'],
            ),

            # construct frame file template
            PickledPyEnvTask(
                func=serialize_module_obj(ffmpeg_frame_file_template_from_nframes),
                inputs=['run->gromacs_vis->nframes'],
                env='imteksimpy',
                outputs=['run->gromacs_vis->ffmpeg_frame_file_template'],
            ),

            # standard format from https://github.com/pastewka/GroupWiki/wiki/Make-movies
            # ffmpeg -r 60 -f image2 -i frame%04d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p test.mp4
            CmdTask(
                cmd='ffmpeg',
                opt=[
                    '-r', 30,  # frame rate
                    '-f', 'image2',
                    '-i', {'key': 'run->gromacs_vis->ffmpeg_frame_file_template'},
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

        # pymol should make use of all CPU via multithreading
        fw_vis = Firework(fts_vis,
            name=self.get_fw_label(step_label),
            spec={
                '_category': self.hpc_specs['fw_queue_category'],
                # hera no_smt as PyMOL needs too much memory and will exceed its limit on JUWELS if using all cores
                '_queueadapter': self.hpc_specs['no_smt_single_node_job_queueadapter_defaults'],
                '_files_in':  files_in,
                '_files_out': files_out,
                'metadata': {
                    'project':  self.project_id,
                    'datetime': str(datetime.datetime.now()),
                    'step':     step_label,
                     **self.kwargs
                }
            },
            parents=[*fws_root, fw_template]
        )

        fw_list.append(fw_vis)

        return fw_list, [fw_vis], [fw_template, fw_vis]
