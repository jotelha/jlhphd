# -*- coding: utf-8 -*-
"""Surfactant molecule mesasures sub workflow."""

import datetime
import glob
import os
import pymongo
import warnings

from fireworks import Firework
from fireworks.user_objects.firetasks.filepad_tasks import GetFilesByQueryTask
from fireworks.user_objects.firetasks.filepad_tasks import AddFilesTask
from imteksimfw.fireworks.user_objects.firetasks.cmd_tasks \
    import PickledPyEnvTask, EvalPyEnvTask

from jlhpy.utilities.geometry.bounding_sphere import \
    get_bounding_sphere_via_parmed, \
    get_atom_position_via_parmed, get_distance
from jlhpy.utilities.vis.plot_side_views_with_spheres import \
    plot_side_views_with_spheres_via_parmed

from imteksimfw.utils.serialize import serialize_module_obj
from jlhpy.utilities.wf.workflow_generator import (
    WorkflowGenerator, ProcessAnalyzeAndVisualize)
from jlhpy.utilities.wf.mixin.mixin_wf_storage import (
   DefaultPullMixin, DefaultPushMixin)

import jlhpy.utilities.wf.file_config as file_config
import jlhpy.utilities.wf.phys_config as phys_config


class SurfactantMoleculeMeasuresMain(WorkflowGenerator):
    """Surfactant molecule mesasures sub workflow.


    static infiles:
    - surfactant_file: default.pdb,

    inputs:
    - metadata->system->surfactant->connector_atom->index (int)

    outputs:
    - metadata->system->surfactant->bounding_sphere->center ([float])
    - metadata->system->surfactant->bounding_sphere->radius (float)
    - metadata->system->surfactant->bounding_sphere->radius_connector_atom (float)
    - metadata->system->surfactant->connector_atom->position ([float])
    - metadata->system->surfactant->head_group->diameter (float)
    """
    def push_infiles(self, fp):
        step_label = self.get_step_label('push_infiles')

        # try to get surfactant pdb file from kwargs
        try:
            surfactant = self.kwargs["system"]["surfactant"]["name"]
        except:
            surfactant = phys_config.DEFAULT_SURFACTANT
            warnings.warn("No surfactant specified, falling back to {:s}.".format(surfactant))

        surfactant_pdb = file_config.SURFACTANT_PDB_PATTERN.format(name=surfactant)

        infiles = sorted(glob.glob(os.path.join(
            self.infile_prefix,
            file_config.PDB_SUBDIR,
            surfactant_pdb)))

        files = {os.path.basename(f): f for f in infiles}

        # metadata common to all these files
        metadata = {
            'project': self.project_id,
            'type': 'surfactant_file',
            'step': step_label,
        }

        fp_files = []

        # insert these input files into data base
        for name, file_path in files.items():
            identifier = '/'.join((self.project_id, name))  # identifier is like a path on a file system
            metadata["name"] = name
            fp_files.append(
                fp.add_file(
                    file_path,
                    identifier=identifier,
                    metadata=metadata))

        return fp_files

    def main(self, fws_root=[]):
        fw_list = []

        step_label = self.get_step_label('input_files_pull')

        fw_list = []

        files_in = {}
        files_out = {
            'surfactant_file': 'default.pdb',
        }

        fts_pull = [
            GetFilesByQueryTask(
                query={
                    'metadata->project':    self.project_id, # earlier
                    'metadata->type':       'surfactant_file',
                },
                sort_key='metadata.datetime',
                sort_direction=pymongo.DESCENDING,
                limit=1,
                new_file_names=['default.pdb'])]

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
            parents=None)

        fw_list.append(fw_pull)

        # Bounding sphere Fireworks
        # -------------------------
        step_label = self.get_step_label('bounding_sphere')

        files_in = {
            'indenter_file':   'indenter.pdb',
            'surfactant_file': 'default.pdb',
        }
        files_out = {
            'indenter_file':   'indenter.pdb',
            'surfactant_file': 'default.pdb',
        }

        func_str = serialize_module_obj(get_bounding_sphere_via_parmed)

        fts_bounding_sphere = [PickledPyEnvTask(
            func=func_str,
            args=['default.pdb'],
            kwargs={'atomic_number_replacements': {'0': 1}},  # ase needs > 0
            outputs=[
                'metadata->system->surfactant->bounding_sphere->center',
                'metadata->system->surfactant->bounding_sphere->radius',
            ],
            env='imteksimpy',
            stderr_file='std.err',
            stdout_file='std.out',
            store_stdout=True,
            store_stderr=True,
            propagate=True,
        )]

        fw_bounding_sphere = Firework(fts_bounding_sphere,
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
            parents=[*fws_root, fw_pull])

        fw_list.append(fw_bounding_sphere)

        # Get head atom position
        # ----------------------
        step_label = self.get_step_label('head_atom_position')

        files_in = {
            'surfactant_file':      'default.pdb',
        }
        files_out = {
        }

        func_str = serialize_module_obj(get_atom_position_via_parmed)

        fts_connector_atom_position = [PickledPyEnvTask(
            func=func_str,
            args=['default.pdb'],
            inputs=[
                'metadata->system->surfactant->connector_atom->index',
            ],
            kwargs={'atomic_number_replacements': {'0': 1}},  # ase needs > 0
            outputs=[
                'metadata->system->surfactant->connector_atom->position',
            ],
            env='imteksimpy',
            stderr_file='std.err',
            stdout_file='std.out',
            store_stdout=True,
            store_stderr=True,
            propagate=True,
        )]

        fw_connector_atom_position = Firework(fts_connector_atom_position,
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
            parents=[*fws_root,fw_pull])

        fw_list.append(fw_connector_atom_position)

        # Get head atom - center distance
        # -------------------------------
        step_label = self.get_step_label('head_atom_to_center_distance')

        files_in = {}
        files_out = {}

        func_str = serialize_module_obj(get_distance)

        fts_radius_connector_atom = [PickledPyEnvTask(
            func=func_str,
            inputs=[
                'metadata->system->surfactant->connector_atom->position',
                'metadata->system->surfactant->bounding_sphere->center',
            ],
            outputs=[
                'metadata->system->surfactant->bounding_sphere->radius_connector_atom',
            ],
            env='imteksimpy',
            stderr_file='std.err',
            stdout_file='std.out',
            store_stdout=True,
            store_stderr=True,
            propagate=True,
        )]

        fw_radius_connector_atom = Firework(fts_radius_connector_atom,
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
            parents=[fw_bounding_sphere, fw_connector_atom_position])

        fw_list.append(fw_radius_connector_atom)

        # head group diameter
        # -------------------
        step_label = self.get_step_label('head_group_diameter')

        files_in = {
            'indenter_file': 'indenter.pdb',
        }
        files_out = {
            'indenter_file': 'indenter.pdb',
        }

        # func_str = serialize_module_obj(get_distance)

        fts_diameter_head_group = [EvalPyEnvTask(
            func='lambda x, y: x - y',
            inputs=[
                'metadata->system->surfactant->bounding_sphere->radius',
                'metadata->system->surfactant->bounding_sphere->radius_connector_atom',
            ],
            outputs=[
                'metadata->system->surfactant->head_group->diameter',  # rough estimate
            ],
            # env='imteksimpy',
            stderr_file='std.err',
            stdout_file='std.out',
            store_stdout=True,
            store_stderr=True,
            propagate=True,
        )]

        fw_diameter_head_group = Firework(fts_diameter_head_group,
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
            parents=[fw_radius_connector_atom])  # for data dependency fw_bounding_sphere

        fw_list.append(fw_diameter_head_group)

        return (
            fw_list,
            [fw_diameter_head_group],
            [fw_bounding_sphere, fw_connector_atom_position])


class SurfactantMoleculeMeasuresVis(WorkflowGenerator):
    """Surfactant molecule measures visualization sub workflow.

    dynamic infiles:
    - surfactant_file: default.pdb

    inputs:
    - metadata->system->surfactant->bounding_sphere->center
    - metadata->system->surfactant->bounding_sphere->radius

    outfiles:
    - png_file:     default.png
    """
    # visualization branch

    def main(self, fws_root=[]):

        fw_list = []

        # Plot sideviews
        # --------------
        step_label = self.get_step_label('vis')

        files_in = {
            'surfactant_file': 'default.pdb',
        }
        files_out = {
            'png_file': 'default.png',
        }

        func_str = serialize_module_obj(plot_side_views_with_spheres_via_parmed)

        fts_vis = [PickledPyEnvTask(
            func=func_str,
            args=['default.pdb', 'default.png'],
            inputs=[
                'metadata->system->surfactant->bounding_sphere->center',
                'metadata->system->surfactant->bounding_sphere->radius',
            ],  # inputs appended to args
            kwargs={'atomic_number_replacements': {'0': 1}},  # ase needs > 0
            env='imteksimpy',
            stderr_file='std.err',
            stdout_file='std.out',
            store_stdout=True,
            store_stderr=True,
            propagate=True,
        )]

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
            parents=fws_root)

        fw_list.append(fw_vis)

        return (
            fw_list,
            [fw_vis],
            [fw_vis])


class SurfactantMoleculeMeasuresWorkflowGenerator(
        DefaultPullMixin, DefaultPushMixin,
        ProcessAnalyzeAndVisualize,
        ):
    def __init__(self, *args, **kwargs):
        ProcessAnalyzeAndVisualize.__init__(self,
            main_sub_wf=SurfactantMoleculeMeasuresMain(*args, **kwargs),
            vis_sub_wf=SurfactantMoleculeMeasuresVis(*args, **kwargs),
            *args, **kwargs)
