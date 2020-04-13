# -*- coding: utf-8 -*-
"""Packing constraint spheres sub workflow."""
import datetime

from fireworks import Firework
from fireworks.user_objects.firetasks.filepad_tasks import GetFilesByQueryTask
from fireworks.user_objects.firetasks.filepad_tasks import AddFilesTask
from fireworks.user_objects.firetasks.templatewriter_task import TemplateWriterTask

from imteksimfw.fireworks.user_objects.firetasks.cmd_tasks import CmdTask, PickledPyEnvTask

# from jlhpy.utilities.vis.plot_side_views_with_spheres import plot_side_views_with_spheres_via_ase

from jlhpy.utilities.wf.serialize import serialize_module_obj
from jlhpy.utilities.wf.workflow_generator import SubWorkflowGenerator

# import jlhpy.utilities.wf.file_config as file_config
from jlhpy.utilities.templates.spherical_packing import generate_pack_sphere_packmol_template_context

class SphericalSurfactantPackingSubWorkflowGenerator(SubWorkflowGenerator):
    """Packing constraint spheres sub workflow.

    Inputs:
        - metadata->system->counterion->name (str)
        - metadata->system->indenter->bounding_sphere->center ([float])
        - metadata->system->indenter->bounding_sphere->radius (float)
        - metadata->system->surfactant->name (str)
        - metadata->system->surfactant->nmolecules (int)
        - metadata->system->packing->surfactant_indenter->inner_atom_index (int)
        - metadata->system->packing->surfactant_indenter->outer_atom_index (int)
        - metadata->system->packing->surfactant_indenter->constraints->R_inner (float)
        - metadata->system->packing->surfactant_indenter->constraints->R_inner_constraint (float)
        - metadata->system->packing->surfactant_indenter->constraints->R_outer_constraint (float)
        - metadata->system->packing->surfactant_indenter->constraints->R_outer (float)
        - metadata->system->packing->surfactant_indenter->tolerance (float)
    """

    def push(self, fws_root=[]):
        fw_list = []

        # pull
        # ----
        step_label = self.get_step_label('pull')

        files_in = {}
        files_out = {
            'input_file':      'input.template',
            'indenter_file':   'indenter.pdb',
            'surfatcant_file': '1_SDS.pdb',
            'counterion_file': '1_NA.pdb'
        }

        fts_pull = [
            GetFilesByQueryTask(
                query={
                    'metadata->project': self.project_id,
                    'metadata->name':    'surfactants_on_sphere.inp'
                },
                limit=1,
                new_file_names=['input.template']),
            GetFilesByQueryTask(
                query={
                    'metadata->project': self.project_id,
                    'metadata->name':    'indenter_reres.pdb'
                },
                limit=1,
                new_file_names=['indenter.pdb']),
            GetFilesByQueryTask(
                query={
                    'metadata->project': self.project_id,
                    'metadata->name':    '1_SDS.pdb'
                },
                limit=1,
                new_file_names=['1_SDS.pdb']),
            GetFilesByQueryTask(
                query={
                    'metadata->project': self.project_id,
                    'metadata->name':    '1_NA.pdb'
                },
                limit=1,
                new_file_names=['1_NA.pdb'])
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
                      'step':    step_label
                }
            },
            parents=fws_root)

        fw_list.append(fw_pull)

        return fw_list, [fw_pull], [fw_pull]

    def main(self, fws_root=[]):
        fw_list = []

        # Context generator
        # -----------------
        step_label = self.get_step_label('packmol template context')

        files_in = {}
        files_out = {}

        func_str = serialize_module_obj(generate_pack_sphere_packmol_template_context)

        fts_context_generator = [PickledPyEnvTask(
            func=func_str,
            kwargs_inputs={
                'C': 'metadata->system->indenter->bounding_sphere->center',
                'R': 'metadata->system->indenter->bounding_sphere->radius',
                'R_inner': 'metadata->system->packing->surfactant_indenter->constraints->R_inner',
                'R_inner_constraint': 'metadata->system->packing->surfactant_indenter->constraints->R_inner_constraint', # shell inner radius
                'R_outer_constraint': 'metadata->system->packing->surfactant_indenter->constraints->R_outer_constraint', # shell outer radius
                'R_outer': 'metadata->system->packing->surfactant_indenter->constraints->R_outer',
                'sfN': 'metadata->system->surfactant->nmolecules',  # number of surfactant molecules
                'inner_atom_number': 'metadata->system->packing->surfactant_indenter->inner_atom_index',  # inner atom
                'outer_atom_number': 'metadata->system->packing->surfactant_indenter->outer_atom_index',  # outer atom
                'surfactant': 'metadata->system->surfactant->name',
                'counterion': 'metadata->system->counterion->name',
                'tolerance': 'metadata->system->packing->surfactant_indenter->tolerance',
            },
            outputs=['run->template->context'],
            env='imteksimpy',
            stderr_file='std.err',
            stdout_file='std.out',
            store_stdout=True,
            store_stderr=True,
            propagate=False,
        )]

        fw_context_generator = Firework(fts_context_generator,
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

        fw_list.append(fw_context_generator)

        # PACKMOL input script template
        # -----------------------------
        step_label = self.get_step_label('packmol template')

        files_in =  {'input_file': 'input.template'}
        files_out = {'input_file': 'input.inp'}

        # Jinja2 context:
        static_packmol_script_context = {
            'write_restart': True,

            'static_components': [
                {
                    'name': 'indenter'
                }
            ]
        }

        context_inputs = {
            'tolerance': 'metadata->system->packing->surfactant_indenter->tolerance',
            'spheres': 'run->template->context->spheres',
            'ionspheres': 'run->template->context->ionspheres',
            'movebadrandom': 'run->template->context->movebadrandom',
        }

        # use pack_sphere function at the notebook's head to generate template context
        # packmol_script_context.update(
        #     pack_sphere(
        #         C,
        #         R_inner_constraint,
        #         R_outer_constraint, d["nmolecules"],
        #         tail_atom_number+1, head_atom_number+1, surfactant, counterion, tolerance))
        #

        ft_template = TemplateWriterTask( {
            'context': static_packmol_script_context,
            'context_inputs': context_inputs,
            'template_file': 'input.template',
            'template_dir': '.',
            'output_file': 'input.inp'} )


        fw_template = Firework([ft_template],
            name=self.get_fw_label(step_label),
            spec = {
                '_category': self.hpc_specs['fw_noqueue_category'],
                '_files_in':  files_in,
                '_files_out': files_out,
                'metadata': {
                    'project':  self.project_id,
                    'datetime': str(datetime.datetime.now()),
                    'step':    'fill_template',
                     **self.kwargs
                }
            },
            parents=[*fws_root, fw_context_generator])

        fw_list.append(fw_template)

        # PACKMOL
        # -------
        step_label = self.get_step_label('packmol')

        files_in = {
            'input_file': 'input.inp',
            'indenter_file': 'indenter.pdb',
            'surfatcant_file': '1_SDS.pdb',
            'counterion_file': '1_NA.pdb' }
        files_out = {
            'data_file': '*_packmol.pdb'}

        ft_pack = CmdTask(
            cmd='packmol',
            # opt=['< input.inp'],
            stdin_file   = 'input.inp',
            stderr_file  = 'std.err',
            stdout_file  = 'std.out',
            store_stdout = True,
            store_stderr = True,
            use_shell    = False,
            fizzle_bad_rc= True)

        fw_pack = Firework([ft_pack],
            name=self.get_fw_label(step_label),
            spec={
                '_category': self.hpc_specs['fw_queue_category'],
                '_queueadapter': {
                    'queue':           self.hpc_specs['queue'],
                    'walltime':        self.hpc_specs['walltime'],
                    'ntasks':          1,  # packmol only serial
                },
                '_files_in':  files_in,
                '_files_out': files_out,
                'metadata': {
                    'project':  self.project_id,
                    'datetime': str(datetime.datetime.now()),
                    'step':    'packmol',
                    **self.kwargs
                }
            },
            parents=[*fws_root, fw_template])

        fw_list.append(fw_pack)

        return fw_list, [fw_pack], [fw_context_generator, fw_template, fw_pack]

    def push(self, fws_root=[]):
        fw_list = []

        step_label = self.get_step_label('push')

        files_in = {'data_file': 'packed.pdb'}
        files_out = {}

        fts_push = [AddFilesTask({
            'compress': True,
            'paths': "packed.pdb",
            'metadata': {
                'project': self.project_id,
                'datetime': str(datetime.datetime.now()),
                'type':    'initial_config',
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
