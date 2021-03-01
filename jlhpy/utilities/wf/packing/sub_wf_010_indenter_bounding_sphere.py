# -*- coding: utf-8 -*-
import datetime
import glob
import os
import pymongo

from fireworks import Firework
from fireworks.user_objects.firetasks.filepad_tasks import GetFilesByQueryTask
from imteksimfw.fireworks.user_objects.firetasks.cmd_tasks import PickledPyEnvTask

from jlhpy.utilities.geometry.bounding_sphere import get_bounding_sphere_via_ase
from jlhpy.utilities.vis.plot_side_views_with_spheres import plot_side_views_with_spheres_via_ase

from imteksimfw.utils.serialize import serialize_module_obj
from jlhpy.utilities.wf.workflow_generator import (
    WorkflowGenerator, ProcessAnalyzeAndVisualize)
from jlhpy.utilities.wf.mixin.mixin_wf_storage import (
   DefaultPullMixin, DefaultPushMixin)

import jlhpy.utilities.wf.file_config as file_config


class IndenterBoundingSphereMain(WorkflowGenerator):
    """Indenter bounding sphere sub workflow.

    dynamic infiles:
    - indenter_file:     default.pdb
        queried by { 'metadata->type': 'em_solvated_gro' }

    outfiles:
    - indenter_file:     default.pdb (unchanged)

    outputs:
        - metadata->system->indenter->bounding_sphere->center ([float])
        - metadata->system->indenter->bounding_sphere->radius (float)
    """
    def push_infiles(self, fp):
        step_label = self.get_step_label('push_infiles')
        self.source_step = step_label  # NOTE: remove for reference to previous step

        infiles = sorted(glob.glob(os.path.join(
            self.infile_prefix,
            file_config.INDENTER_SUBDIR, file_config.INDENTER_PDB)))

        files = {os.path.basename(f): f for f in infiles}

        # metadata common to all these files
        metadata = {
            'project': self.project_id,
            'type': 'indenter_file',
            'step': step_label,
            'name': file_config.INDENTER_PDB
        }

        fp_files = []

        # insert these input files into data base
        for name, file_path in files.items():
            identifier = '/'.join((self.project_id, name))  # identifier is like a path on a file system
            fp_files.append(
                fp.add_file(
                    file_path,
                    identifier=identifier,
                    metadata=metadata))

        return fp_files

    def main(self, fws_root=[]):
        fw_list = []

        step_label = self.get_step_label('input_files_pull')

        files_in = {}
        files_out = {
            'indenter_file': 'default.pdb',
        }

        fts_pull = [
            GetFilesByQueryTask(
                query={
                    'metadata->project': self.project_id,  # earlier
                    'metadata->type': 'indenter_file',
                },
                sort_key='metadata.datetime',
                sort_direction=pymongo.DESCENDING,
                limit=1,
                new_file_names=['default.pdb'])]

        fw_pull = self.build_fw(
            fts_pull, step_label,
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_noqueue_category'])

        fw_list.append(fw_pull)

        # Bounding sphere Fireworks
        # -------------------------
        step_label = self.get_step_label('bounding_sphere')

        files_in = {
            'indenter_file':      'default.pdb',
        }
        files_out = {
            'indenter_file':      'default.pdb',
        }

        func_str = serialize_module_obj(get_bounding_sphere_via_ase)

        fts_bounding_sphere = [PickledPyEnvTask(
            func=func_str,
            args=['default.pdb'],
            outputs=[
                'metadata->system->indenter->bounding_sphere->center',
                'metadata->system->indenter->bounding_sphere->radius',
            ],
            env='imteksimpy',
            stderr_file='std.err',
            stdout_file='std.out',
            stdlog_file='std.log',
            store_stdout=True,
            store_stderr=True,
            store_stdlog=True,
            propagate=True,
        )]

        fw_bounding_sphere = self.build_fw(
            fts_bounding_sphere, step_label,
            parents=[*fws_root, fw_pull],
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_noqueue_category'])

        fw_list.append(fw_bounding_sphere)

        return fw_list, [fw_bounding_sphere], [fw_bounding_sphere]


class IndenterBoundingSphereVis(
        WorkflowGenerator):
    """Indenter bounding sphere visualization sub workflow.

    dynamic infiles:
    - indenter_file:     default.pdb

    inputs:
    - metadata->system->indenter->bounding_sphere->center ([float])
    - metadata->system->indenter->bounding_sphere->radius (float)

    outfiles:
    - png_file:     default.png
    """
    def main(self, fws_root=[]):
        fw_list = []
        # Plot sideviews
        # --------------
        step_label = self.get_step_label('vis')

        files_in = {
            'indenter_file': 'default.pdb',
        }
        files_out = {
            'png_file': 'default.png'
        }

        func_str = serialize_module_obj(plot_side_views_with_spheres_via_ase)

        fts_vis = [PickledPyEnvTask(
            func=func_str,
            args=['default.pdb', 'default.png'],
            inputs=[
                'metadata->system->indenter->bounding_sphere->center',
                'metadata->system->indenter->bounding_sphere->radius',
            ],  # inputs appended to args
            env='imteksimpy',
            stderr_file='std.err',
            stdout_file='std.out',
            store_stdout=True,
            store_stderr=True,
            propagate=True,
        )]

        fw_vis = self.build_fw(
            fts_vis, step_label,
            parents=fws_root,
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_noqueue_category'])

        fw_list.append(fw_vis)

        return fw_list, [fw_vis], [fw_vis]


class IndenterBoundingSphere(
        DefaultPullMixin, DefaultPushMixin,
        ProcessAnalyzeAndVisualize,
        ):
    def __init__(self, *args, **kwargs):
        ProcessAnalyzeAndVisualize.__init__(self,
            main_sub_wf=IndenterBoundingSphereMain,
            vis_sub_wf=IndenterBoundingSphereVis,
            *args, **kwargs)
