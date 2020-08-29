# -*- coding: utf-8 -*-
"""Packing constraint spheres sub workflow."""

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
    WorkflowGenerator, ProcessAnalyzeAndVisualizeWorkflowGenerator)
from jlhpy.utilities.wf.mixin.mixin_wf_storage import (
   DefaultPullMixin, DefaultPushMixin)

class PackingConstraintSpheresMain(WorkflowGenerator):
    """Packing constraint spheres sub workflow.

    Inputs:
    - metadata->system->indenter->bounding_sphere->radius (float)
    - metadata->system->surfactant->bounding_sphere->radius (float)
    - metadata->system->surfactant->head_group->diameter (float)
    - metadata->step_specific->packing->surfactant_indenter->tolerance (float)

    Outputs:
    - metadata->step_specific->packing->surfactant_indenter->constraints->R_inner (float)
    - metadata->step_specific->packing->surfactant_indenter->constraints->R_inner_constraint (float)
    - metadata->step_specific->packing->surfactant_indenter->constraints->R_outer_constraint (float)
    - metadata->step_specific->packing->surfactant_indenter->constraints->R_outer (float)
    """
    def main(self, fws_root=[]):
        fw_list = []

        # R_inner
        # -------
        step_label = self.get_step_label('R_inner')

        files_in = {
            'indenter_file': 'default.pdb',  # pass through
        }
        files_out = {
            'indenter_file': 'default.pdb',  # pass through
        }

        fts_R_inner = [
            EvalPyEnvTask(
                func='lambda x, y: x + y',
                inputs=[
                    'metadata->system->indenter->bounding_sphere->radius',
                    'metadata->step_specific->packing->surfactant_indenter->tolerance',
                ],
                outputs=[
                    'metadata->step_specific->packing->surfactant_indenter->constraints->R_inner',
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
                    'metadata->system->indenter->bounding_sphere->radius',
                    'metadata->system->surfactant->head_group->diameter',
                    'metadata->step_specific->packing->surfactant_indenter->tolerance',
                ],
                outputs=[
                    'metadata->step_specific->packing->surfactant_indenter->constraints->R_inner_constraint',
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
                    'metadata->system->indenter->bounding_sphere->radius',
                    'metadata->system->surfactant->bounding_sphere->radius',
                    'metadata->step_specific->packing->surfactant_indenter->tolerance',
                ],
                outputs=[
                    'metadata->step_specific->packing->surfactant_indenter->constraints->R_outer_constraint',
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
                    'metadata->system->indenter->bounding_sphere->radius',
                    'metadata->system->surfactant->bounding_sphere->radius',
                    'metadata->step_specific->packing->surfactant_indenter->tolerance',
                ],
                outputs=[
                    'metadata->step_specific->packing->surfactant_indenter->constraints->R_outer',
                ],
                # env='imteksimpy',
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

        return (
            fw_list,
            [fw_R_inner, fw_R_inner_constraint, fw_R_outer_constraint, fw_R_outer],
            [fw_R_inner, fw_R_inner_constraint, fw_R_outer_constraint, fw_R_outer])


class PackingConstraintSpheresVis(
        WorkflowGenerator):
    """Packing constraint spheres visualization sub workflow.

    dynamic infiles:
    - indenter_file:     default.pdb

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
            'indenter_file': 'default.pdb',
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


class PackingConstraintSpheresWorkflowGenerator(
        DefaultPullMixin, DefaultPushMixin,
        ProcessAnalyzeAndVisualizeWorkflowGenerator,
        ):
    def __init__(self, *args, **kwargs):
        sub_wf_name = 'PackingConstraintSpheres'
        if 'wf_name_prefix' not in kwargs:
            kwargs['wf_name_prefix'] = sub_wf_name
        else:
            kwargs['wf_name_prefix'] = ':'.join((kwargs['wf_name_prefix'], sub_wf_name))
        ProcessAnalyzeAndVisualizeWorkflowGenerator.__init__(self,
            main_sub_wf=PackingConstraintSpheresMain(*args, **kwargs),
            vis_sub_wf=PackingConstraintSpheresVis(*args, **kwargs),
            *args, **kwargs)
