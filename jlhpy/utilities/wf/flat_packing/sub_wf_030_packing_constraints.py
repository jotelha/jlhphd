# -*- coding: utf-8 -*-
"""Packing constraints sub workflows."""

import datetime
import pymongo

from fireworks import Firework
from fireworks.user_objects.firetasks.filepad_tasks import GetFilesByQueryTask
from fireworks.user_objects.firetasks.filepad_tasks import AddFilesTask
from fireworks.user_objects.firetasks.dataflow_tasks import JoinListTask
from imteksimfw.fireworks.user_objects.firetasks.cmd_tasks \
    import EvalPyEnvTask, PickledPyEnvTask, PyEnvTask

from jlhpy.utilities.vis.plot_side_views_with_spheres import \
    plot_side_views_with_spheres_via_parmed

from imteksimfw.fireworks.utilities.serialize import serialize_module_obj, serialize_obj
from jlhpy.utilities.wf.workflow_generator import (
    SubWorkflowGenerator, ProcessAnalyzeAndVisualizeSubWorkflowGenerator)
from jlhpy.utilities.wf.mixin.mixin_wf_storage import (
   DefaultPullMixin, DefaultPushMixin)

class MonolayerPackingConstraintsMain(SubWorkflowGenerator):
    """Monolayer packing constraint planes sub workflow.

    Inputs:
    - metadata->system->substrate->bounding_box ([[float]])
    - metadata->system->surfactant->bounding_sphere->radius (float)
    - metadata->system->surfactant->head_group->diameter (float)
    - metadata->step_specific->packing->surfactant_substrate->tolerance (float)

    Outputs:
    - metadata->step_specific->packing->surfactant_substrate->constraints->monolayer->bounding_box ([[float]])
    - metadata->step_specific->packing->surfactant_substrate->constraints->monolayer->z_lower_constraint (float)
    - metadata->step_specific->packing->surfactant_substrate->constraints->monolayer->z_upper_constraint (float)
    """
    def __init__(self, *args, **kwargs):
        sub_wf_name = 'PackingConstraintPlanesMain'
        if 'wf_name_prefix' not in kwargs:
            kwargs['wf_name_prefix'] = sub_wf_name
        else:
            kwargs['wf_name_prefix'] = ':'.join((kwargs['wf_name_prefix'], sub_wf_name))
        super().__init__(*args, **kwargs)

    def main(self, fws_root=[]):
        fw_list = []

        # bounding box
        # ------------
        step_label = self.get_step_label('bounding_box')

        files_in = {
            'data_file': 'default.pdb',  # pass through
        }
        files_out = {
            'data_file': 'default.pdb',  # pass through
        }

        fts_bb = [
            EvalPyEnvTask(
                func='lambda bb, r, tol: [[*bb[0][0:2], bb[1][2] + tol], [*bb[1][0:2], bb[1][2] + 2*r + 2*tol]',
                inputs=[
                    'metadata->system->substrate->bounding_box',
                    'metadata->system->surfactant->bounding_sphere->radius',
                    'metadata->step_specific->packing->surfactant_substrate->tolerance',
                ],
                outputs=[
                    'metadata->step_specific->packing->surfactant_substrate->constraints->monolayer->bounding_box',
                ],
                stderr_file='std.err',
                stdout_file='std.out',
                store_stdout=True,
                store_stderr=True,
                propagate=True,
            )
        ]

        fw_bb = Firework(fts_bb,
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

        fw_list.append(fw_bb)

        # z constraints
        # -------------
        step_label = self.get_step_label('z_constraints')

        files_in = {
            'data_file': 'default.pdb',
        }
        files_out = {
            'data_file': 'default.pdb',  # pass through
        }

        fts_z_constraints = [
            EvalPyEnvTask(
                func='lambda bb, d_head_group, tol: bb[0][2] + d_head_group/2. + tol, bb[1][2] - d_head_group/2. - tol',
                inputs=[
                    'metadata->step_specific->packing->surfactant_substrate->constraints->monolayer->bounding_box',
                    'metadata->system->surfactant->head_group->diameter',
                    'metadata->step_specific->packing->surfactant_substrate->tolerance',
                ],
                outputs=[
                    'metadata->step_specific->packing->surfactant_substrate->constraints->z_lower_constraint',
                    'metadata->step_specific->packing->surfactant_substrate->constraints->z_upper_constraint',
                ],
                stderr_file='std.err',
                stdout_file='std.out',
                store_stdout=True,
                store_stderr=True,
                propagate=True,
            )
        ]

        fw_z_constraints = Firework(fts_z_constraints,
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

        fw_list.append(fw_z_constraints)

        return fw_list, [fw_bb, fw_z_constraints], [fw_bb, fw_z_constraints]


class CylindricalPackingConstraintsMain(SubWorkflowGenerator):
    """Cylinder packing constraints sub workflow.

    Inputs:
    - metadata->system->substrate->bounding_box ([[float]])
    - metadata->system->surfactant->bounding_sphere->radius (float)
    - metadata->system->surfactant->head_group->diameter (float)
    - metadata->step_specific->packing->surfactant_substrate->tolerance (float)

    Outputs:
    - metadata->step_specific->packing->surfactant_substrate->N_aggregates (int)
    - metadata->step_specific->packing->surfactant_substrate->constraints->cylinders->base_center ([[float]])

    - metadata->step_specific->packing->surfactant_substrate->constraints->cylinders->length (float)
    - metadata->step_specific->packing->surfactant_substrate->constraints->cylinders->R_inner (float)
    - metadata->step_specific->packing->surfactant_substrate->constraints->cylinders->R_inner_constraint (float)
    - metadata->step_specific->packing->surfactant_substrate->constraints->cylinders->R_outer_constraint (float)
    - metadata->step_specific->packing->surfactant_substrate->constraints->cylinders->R_outer (float)
    """
    def __init__(self, *args, **kwargs):
        sub_wf_name = 'PackingConstraintPlanesMain'
        if 'wf_name_prefix' not in kwargs:
            kwargs['wf_name_prefix'] = sub_wf_name
        else:
            kwargs['wf_name_prefix'] = ':'.join((kwargs['wf_name_prefix'], sub_wf_name))
        super().__init__(*args, **kwargs)

    def main(self, fws_root=[]):
        fw_list = []

        # length
        # ------
        step_label = self.get_step_label('length')

        files_in = {}
        files_out = {}

        # width of substrate / diameter of cylinder
        fts_length = [
            EvalPyEnvTask(
                func='lambda bb: bb[1][1] - bb[0][1]',
                inputs=[
                    'metadata->system->substrate->bounding_box',
                ],
                outputs=[
                    'metadata->step_specific->packing->surfactant_substrate->constraints->cylinders->length',
                ],
                stderr_file='std.err',
                stdout_file='std.out',
                store_stdout=True,
                store_stderr=True,
                propagate=True,
            )
        ]

        fw_length = Firework(fts_length,
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

        fw_list.append(fw_length)

        # R_inner
        # -------
        step_label = self.get_step_label('R_inner')

        files_in = {
            'data_file': 'default.pdb',  # pass through
        }
        files_out = {
            'data_file': 'default.pdb',  # pass through
        }

        fts_R_inner = [
            EvalPyEnvTask(
                func='lambda tol: tol',
                inputs=[
                    'metadata->step_specific->packing->surfactant_substrate->tolerance',
                ],
                outputs=[
                    'metadata->step_specific->packing->surfactant_substrate->constraints->cylinders->R_inner',
                ],
                # env='imteksimpy',
                stderr_file='std.err',
                stdout_file='std.out',
                store_stdout=True,
                store_stderr=True,
                propagate=True,
            )
        ]

        fw_R_inner = Firework(fts_R_inner,
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

        fw_list.append(fw_R_inner)

        # R_inner_constraint
        # ------------------
        step_label = self.get_step_label('R_inner_constraint')

        files_in = {}
        files_out = {}

        fts_R_inner_constraint = [
            EvalPyEnvTask(
                func='lambda R, d_head_group, tol: R+d_head_group+tol',
                inputs=[
                    'metadata->step_specific->packing->surfactant_substrate->tolerance',
                    'metadata->system->surfactant->head_group->diameter',
                    'metadata->step_specific->packing->surfactant_substrate->tolerance',
                ],
                outputs=[
                    'metadata->step_specific->packing->surfactant_substrate->constraints->cylinders->R_inner_constraint',
                ],
                # env='imteksimpy',
                stderr_file='std.err',
                stdout_file='std.out',
                store_stdout=True,
                store_stderr=True,
                propagate=True,
            )
        ]

        fw_R_inner_constraint = Firework(fts_R_inner_constraint,
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

        fw_list.append(fw_R_inner_constraint)

        # R_outer_constraint
        # ------------------
        step_label = self.get_step_label('R_outer_constraint')

        files_in = {}
        files_out = {}

        # def get_R_outer_constraint(R, R_surfactant, tol):
        #     return R+2.0*R_surfactant+tol

        # func_str = serialize_obj(get_R_outer_constraint)

        fts_R_outer_constraint = [
            EvalPyEnvTask(
                func='lambda R, R_surfactant, tol: R+2.0*R_surfactant+tol',
                inputs=[
                    'metadata->step_specific->packing->surfactant_substrate->tolerance',
                    'metadata->system->surfactant->bounding_sphere->radius',
                    'metadata->step_specific->packing->surfactant_substrate->tolerance',
                ],
                outputs=[
                    'metadata->step_specific->packing->surfactant_substrate->constraints->cylinders->R_outer_constraint',
                ],
                # env='imteksimpy',
                stderr_file='std.err',
                stdout_file='std.out',
                store_stdout=True,
                store_stderr=True,
                propagate=True,
            )
        ]

        fw_R_outer_constraint = Firework(fts_R_outer_constraint,
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

        fw_list.append(fw_R_outer_constraint)

        # R_outer
        # ------------------
        step_label = self.get_step_label('R_outer')

        files_in = {}
        files_out = {}

        #def get_R_outer(R, R_surfactant, tol):
        #    return R+2.0*R_surfactant+2*tol

        #func_str = serialize_obj(get_R_outer)

        fts_R_outer = [
            EvalPyEnvTask(
                func='lambda R, R_surfactant, tol: R+2.0*R_surfactant+2*tol',
                inputs=[
                    'metadata->step_specific->packing->surfactant_substrate->tolerance',
                    'metadata->system->surfactant->bounding_sphere->radius',
                    'metadata->step_specific->packing->surfactant_substrate->tolerance',
                ],
                outputs=[
                    'metadata->step_specific->packing->surfactant_substrate->constraints->cylinders->R_outer',
                ],
                stderr_file='std.err',
                stdout_file='std.out',
                store_stdout=True,
                store_stderr=True,
                propagate=True,
            )
        ]

        fw_R_outer = Firework(fts_R_outer,
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

        fw_list.append(fw_R_outer)


        # N aggregates
        # -------
        step_label = self.get_step_label('N_aggregates')

        files_in = {}
        files_out = {}

        # width of substrate / diameter of cylinder
        fts_N_aggregates = [
            EvalPyEnvTask(
                func='lambda bb, R: int(floor((bb[1][0]-bb[0][0])/(2.*R)))',
                inputs=[
                    'metadata->system->substrate->bounding_box',
                    'metadata->step_specific->packing->surfactant_substrate->constraints->cylinders->R_outer',
                ],
                outputs=[
                    'metadata->step_specific->packing->surfactant_substrate->N_aggregates',
                ],
                stderr_file='std.err',
                stdout_file='std.out',
                store_stdout=True,
                store_stderr=True,
                propagate=True,
            )
        ]

        fw_N_aggregates = Firework(fts_N_aggregates,
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
            parents=fw_R_outer)

        fw_list.append(fw_N_aggregates)

        # base_center
        # -------------
        step_label = self.get_step_label('base_center')

        files_in = {}
        files_out = {}

        fts_base_center = [
            EvalPyEnvTask(
                func='lambda bb, R, N, tol: [[bb[0][0]+(i+0.5)/N*(bb[1][0]-bb[0][0]), bb[0][1], bb[1][2] + R + tol] for i in range(N)]',
                inputs=[
                    'metadata->system->substrate->bounding_box',
                    'metadata->step_specific->packing->surfactant_substrate->constraints->cylinders->R_outer',
                    'metadata->step_specific->packing->surfactant_substrate->N_aggregates',
                    'metadata->step_specific->packing->surfactant_substrate->tolerance',
                ],
                outputs=[
                    'metadata->step_specific->packing->surfactant_substrate->constraints->cylinders->base_center',
                ],
                stderr_file='std.err',
                stdout_file='std.out',
                store_stdout=True,
                store_stderr=True,
                propagate=True,
            )
        ]

        fw_base_center = Firework(fts_base_center,
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
            parents=fw_N_aggregates)

        fw_list.append(fw_base_center)

        return (
            fw_list,
            [fw_R_inner, fw_R_inner_constraint, fw_R_outer_constraint, fw_R_outer, fw_N_aggregates, fw_base_center, fw_length],
            [fw_R_inner, fw_R_inner_constraint, fw_R_outer_constraint, fw_R_outer])


class PackingConstraintSpheresVis(
        SubWorkflowGenerator):
    """Packing constraint spheres visualization sub workflow.

    dynamic infiles:
    - data_file:     default.pdb

    inputs:
    - metadata->system->indenter->bounding_sphere->center ([float])
    - metadata->system->indenter->bounding_sphere->radius (float)
    - metadata->step_specific->packing->surfactant_indenter->constraints->R_inner (float)
    - metadata->step_specific->packing->surfactant_indenter->constraints->R_inner_constraint (float)
    - metadata->step_specific->packing->surfactant_indenter->constraints->R_outer_constraint (float)
    - metadata->step_specific->packing->surfactant_indenter->constraints->R_outer (float)

    outfiles:
    - png_file:     default.png
    """
    def __init__(self, *args, **kwargs):
        sub_wf_name = 'PackingConstraintSpheresVis'
        if 'wf_name_prefix' not in kwargs:
            kwargs['wf_name_prefix'] = sub_wf_name
        else:
            kwargs['wf_name_prefix'] = ':'.join((kwargs['wf_name_prefix'], sub_wf_name))
        super().__init__(*args, **kwargs)

    def main(self, fws_root=[]):
        fw_list = []

        # Join radii and centers
        # ----------------------
        step_label = self.get_step_label('join_radii_in_list')

        files_in = {}
        files_out = {}

        fts_join = [
            JoinListTask(
                inputs=[
                    'metadata->system->indenter->bounding_sphere->radius',
                    'metadata->step_specific->packing->surfactant_indenter->constraints->R_inner',
                    'metadata->step_specific->packing->surfactant_indenter->constraints->R_inner_constraint',
                    'metadata->step_specific->packing->surfactant_indenter->constraints->R_outer_constraint',
                    'metadata->step_specific->packing->surfactant_indenter->constraints->R_outer',
                ],
                output='metadata->step_specific->packing->surfactant_indenter->constraints->R_list',
            ),
            JoinListTask(
                inputs=[
                    'metadata->system->indenter->bounding_sphere->center',
                    'metadata->system->indenter->bounding_sphere->center',
                    'metadata->system->indenter->bounding_sphere->center',
                    'metadata->system->indenter->bounding_sphere->center',
                    'metadata->system->indenter->bounding_sphere->center',
                ],
                output='metadata->step_specific->packing->surfactant_indenter->constraints->C_list',
            )
        ]

        fw_join = Firework(fts_join,
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

        fw_list.append(fw_join)

        # Plot sideviews
        # --------------
        step_label = self.get_step_label('vis')

        files_in = {
            'data_file': 'default.pdb',
        }
        files_out = {
            'png_file': 'default.png',
        }

        func_str = serialize_module_obj(plot_side_views_with_spheres_via_parmed)

        fts_vis = [PickledPyEnvTask(
            func=func_str,
            args=['default.pdb', 'default.png'],
            inputs=[
                'metadata->step_specific->packing->surfactant_indenter->constraints->C_list',
                'metadata->step_specific->packing->surfactant_indenter->constraints->R_list',
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
            parents=[*fws_root, fw_join])

        fw_list.append(fw_vis)

        return fw_list, [fw_vis], [fw_join, fw_vis]


class PackingConstraintSpheresSubWorkflowGenerator(
        DefaultPullMixin, DefaultPushMixin,
        ProcessAnalyzeAndVisualizeSubWorkflowGenerator,
        ):
    def __init__(self, *args, **kwargs):
        sub_wf_name = 'PackingConstraintSpheres'
        if 'wf_name_prefix' not in kwargs:
            kwargs['wf_name_prefix'] = sub_wf_name
        else:
            kwargs['wf_name_prefix'] = ':'.join((kwargs['wf_name_prefix'], sub_wf_name))
        ProcessAnalyzeAndVisualizeSubWorkflowGenerator.__init__(self,
            main_sub_wf=PackingConstraintSpheresMain(*args, **kwargs),
            vis_sub_wf=PackingConstraintSpheresVis(*args, **kwargs),
            *args, **kwargs)
