# -*- coding: utf-8 -*-
"""Packing constraints sub workflows."""

import datetime
import pymongo

from fireworks import Firework
from fireworks.user_objects.firetasks.script_task import PyTask
from fireworks.user_objects.firetasks.filepad_tasks import GetFilesByQueryTask
from fireworks.user_objects.firetasks.filepad_tasks import AddFilesTask
# from fireworks.user_objects.firetasks.dataflow_tasks import JoinListTask
from imteksimfw.fireworks.user_objects.firetasks.cmd_tasks \
    import CmdTask, EvalPyEnvTask, PickledPyEnvTask, PyEnvTask

from imteksimfw.fireworks.utilities.geometry.morphology import (
    monolayer_above_substrate, bilayer_above_substrate, cylinders_above_substrate, hemicylinders_above_substrate)


from imteksimfw.fireworks.utilities.templates.flat_packing import (
    generate_pack_alternating_multilayer_packmol_template_context)

from imteksimfw.fireworks.utilities.templates.cylindrical_packing import (
    generate_cylinders_packmol_template_context)

from imteksimfw.fireworks.utilities.serialize import serialize_module_obj, serialize_obj
from jlhpy.utilities.wf.workflow_generator import (
    WorkflowGenerator, ChainWorkflowGenerator, ProcessAnalyzeAndVisualizeWorkflowGenerator)
from jlhpy.utilities.wf.mixin.mixin_wf_storage import (
   DefaultPullMixin, DefaultPushMixin)


class PackingConstraintsMain(WorkflowGenerator):
    """Packing constraints sub workflow ABC.

    Inputs:
    - metadata->system->substrate->bounding_box ([[float]])
    - metadata->system->surfactant->bounding_sphere->radius (float)
    - metadata->system->surfactant->head_group->diameter (float)
    - metadata->step_specific->packing->surfactant_substrate->tolerance (float)

    Outputs:
    - metadata->step_specific->packing->surfactant_substrate->constraints (dict)
    """
    def main(self, fws_root=[]):
        fw_list = []

        # constraints
        # -----------
        step_label = self.get_step_label('constraints')

        files_in = {
            'data_file': 'default.pdb',  # pass through
        }
        files_out = {
            'data_file': 'default.pdb',  # pass through
        }

        fts_constraints = [
            PickledPyEnvTask(
                func=self.func_str,
                inputs=[
                    'metadata->system->substrate->bounding_box',
                    'metadata->system->surfactant->bounding_sphere->radius',
                    'metadata->system->surfactant->head_group->diameter',
                    'metadata->step_specific->packing->surfactant_substrate->tolerance',
                ],
                outputs=[
                    'metadata->step_specific->packing->surfactant_substrate->constraints',
                ],
                stderr_file='std.err',
                stdout_file='std.out',
                stdlog_file='std.log',
                propagate=True,
            )
        ]

        fw_constraints = Firework(fts_constraints,
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

        fw_list.append(fw_constraints)

        return fw_list, [fw_constraints], [fw_constraints]


class MonolayerPackingConstraintsMain(PackingConstraintsMain):
    """Monolayer packing constraint planes sub workflow. """
    func_str = serialize_module_obj(monolayer_above_substrate)
class BilayerPackingConstraintsMain(PackingConstraintsMain):
    """Bilayer packing constraint planes sub workflow. """
    func_str = serialize_module_obj(bilayer_above_substrate)
class CylindricalPackingConstraintsMain(PackingConstraintsMain):
    """Cylinder packing constraints sub workflow."""
    func_str = serialize_module_obj(cylinders_above_substrate)
class HemicylindricalPackingConstraintsMain(PackingConstraintsMain):
    """Cylinder packing constraints sub workflow."""
    func_str = serialize_module_obj(hemicylinders_above_substrate)
class LayeredPackingContextMain(WorkflowGenerator):
    """Layered packing template context sub workflow.

    Inputs:
    - metadata->step_specific->packing->surfactant_substrate->constraints->layers (list)
    - metadata->step_specific->packing->surfactant_substrate->tolerance (float)

    - metadata->system->counterion->name (str)
    - metadata->system->surfactant->name (str)
    - metadata->system->surfactant->nmolecules (int)
    - metadata->system->surfactant->head_atom->index (int)
    - metadata->system->surfactant->tail_atom->index (int)

    Outputs:
    """
    func_str = serialize_module_obj(generate_pack_alternating_multilayer_packmol_template_context)
    def main(self, fws_root=[]):
        fw_list = []

        # context
        # -----------
        step_label = self.get_step_label('context')

        files_in = {
            'data_file': 'default.pdb',  # pass through
        }
        files_out = {
            'data_file': 'default.pdb',  # pass through
        }

        fts_context = [
            PickledPyEnvTask(
                func=self.func_str,
                inputs=[
                    'metadata->step_specific->packing->surfactant_substrate->constraints->layers',
                    'metadata->system->surfactant->nmolecules',
                    'metadata->system->surfactant->head_atom->index',
                    'metadata->system->surfactant->tail_atom->index',
                    'metadata->system->surfactant->name',
                    'metadata->system->counterion->name',
                    'metadata->step_specific->packing->surfactant_substrate->tolerance',
                ],
                outputs=[
                    'run->template->context',
                ],
                stderr_file='std.err',
                stdout_file='std.out',
                stdlog_file='std.log',
                propagate=False,
            )
        ]

        fw_context = Firework(fts_context,
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

        fw_list.append(fw_context)

        return fw_list, [fw_context], [fw_context]


class CylindricalPackingContextMain(WorkflowGenerator):
    """Cylindrical packing template context sub workflow.

    Inputs:
    - metadata->step_specific->packing->surfactant_substrate->constraints->cylinders (list)
    - metadata->step_specific->packing->surfactant_substrate->tolerance (float)

    - metadata->system->counterion->name (str)
    - metadata->system->surfactant->name (str)
    - metadata->system->surfactant->nmolecules (int)
    - metadata->system->surfactant->head_atom->index (int)
    - metadata->system->surfactant->tail_atom->index (int)

    Outputs:
    -
    """
    func_str = serialize_module_obj(generate_cylinders_packmol_template_context)
    def main(self, fws_root=[]):
        fw_list = []

        # context
        # -----------
        step_label = self.get_step_label('context')

        files_in = {
            'data_file': 'default.pdb',  # pass through
        }
        files_out = {
            'data_file': 'default.pdb',  # pass through
        }

        fts_context = [
            PickledPyEnvTask(
                func=self.func_str,
                inputs=[
                    'metadata->step_specific->packing->surfactant_substrate->constraints->cylinders',
                    'metadata->system->surfactant->nmolecules',
                    'metadata->system->surfactant->head_atom->index',
                    'metadata->system->surfactant->tail_atom->index',
                    'metadata->system->surfactant->name',
                    'metadata->system->counterion->name',
                    'metadata->step_specific->packing->surfactant_substrate->tolerance',
                ],
                outputs=[
                    'run->template->context',
                ],
                stderr_file='std.err',
                stdout_file='std.out',
                stdlog_file='std.log',
                propagate=False,
            )
        ]

        fw_context = Firework(fts_context,
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

        fw_list.append(fw_context)

        return fw_list, [fw_context], [fw_context]


class HemicylindricalPackingContextMain(CylindricalPackingConstraintsMain):
    """Hemicylindrical packing template context sub workflow."""

    func_str = serialize_module_obj(generate_upper_hemicylinders_packmol_template_context)
class PackingMain(WorkflowGenerator):
    """Packmol packing."""

    context_inputs = {
        'tolerance': 'metadata->step_specific->packing->surfactant_substrate->tolerance',
        'layers': 'run->template->context->layers',
        'ionlayers': 'run->template->context->ionlayers',
        'movebadrandom': 'run->template->context->movebadrandom',
    }
    def main(self, fws_root=[]):
        fw_list = []

        # coordinates pull
        # ----------------
        step_label = self.get_step_label('coordinates_pull')

        files_in = {}
        files_out = {
            'surfatcant_file': 'surfactant.pdb',
            'counterion_file': 'counterion.pdb',
        }

        fts_coordinates_pull = [
            GetFilesByQueryTask(
                query={
                    'metadata->project': self.project_id,
                    'metadata->type':    'surfactant_file',
                },
                limit=1,
                new_file_names=['surfactant.pdb']),
            GetFilesByQueryTask(
                query={
                    'metadata->project': self.project_id,
                    'metadata->type':    'counterion_file',
                },
                limit=1,
                new_file_names=['counterion.pdb'])
        ]

        fw_coordinates_pull = Firework(fts_coordinates_pull,
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
            parents=None)

        fw_list.append(fw_coordinates_pull)

        # input files pull
        # ----------------
        step_label = self.get_step_label('inputs_pull')

        files_in = {}
        files_out = {
            'input_file':      'input.template',
        }

        fts_inputs_pull = [GetFilesByQueryTask(
            query={
                'metadata->project': self.project_id,
                'metadata->name':    file_config.PACKMOL_FLAT_TEMPLATE,
            },
            limit=1,
            new_file_names=['input.template'])]

        fw_inputs_pull = Firework(fts_inputs_pull,
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
            parents=None)

        fw_list.append(fw_inputs_pull)

        # PACKMOL input script template
        # -----------------------------
        step_label = self.get_step_label('packmol_template')

        files_in = {'input_file': 'input.template'}
        files_out = {'input_file': 'input.inp'}

        # Jinja2 context:
        static_packmol_script_context = {
            'system_name': 'default',
            'header': ', '.join((
                self.project_id,
                self.get_fw_label(step_label),
                str(datetime.datetime.now()))),

            'write_restart': True,

            'static_components': [
                {
                    'name': 'default'
                }
            ]
        }

        ft_template = TemplateWriterTask({
            'context': static_packmol_script_context,
            'context_inputs': self.context_inputs,
            'template_file': 'input.template',
            'template_dir': '.',
            'output_file': 'input.inp'})

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
            parents=[fws_root, fw_inputs_pull])

        fw_list.append(fw_template)

        # PACKMOL
        # -------
        step_label = self.get_step_label('packmol')

        files_in = {
            'input_file': 'input.inp',
            'data_file':       'default.pdb',
            'surfatcant_file': 'surfactant.pdb',
            'counterion_file': 'counterion.pdb',
        }
        files_out = {
            'data_file': 'default_packmol.pdb'}

        # ATTENTION: packmol return code == 0 even for failure
        fts_pack = [
            CmdTask(
                cmd='packmol',
                env='python',
                stdin_file   = 'input.inp',
                stderr_file  = 'std.err',
                stdout_file  = 'std.out',
                store_stdout = True,
                store_stderr = True,
                use_shell    = False,
                fizzle_bad_rc= True),
            PyTask(  # check for output file and fizzle if not existant
                func='open',
                args=['default_packmol.pdb'])]

        fw_pack = Firework(fts_pack,
            name=self.get_fw_label(step_label),
            spec={
                '_category': self.hpc_specs['fw_queue_category'],
                '_queueadapter': {
                    **self.hpc_specs['single_core_job_queueadapter_defaults'],  # packmol only serial
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
            parents=[fw_coordinates_pull, fw_template])

        fw_list.append(fw_pack)

        return fw_list, [fw_pack], [fw_template, fw_pack]

class LayeredPackingMain(PackingMain):
    """Layered packmol packing."""
    context_inputs = {
        'tolerance': 'metadata->step_specific->packing->surfactant_substrate->tolerance',
        'layers': 'run->template->context->layers',
        'ionlayers': 'run->template->context->ionlayers',
    }


class CylindricalPackingMain(PackingMain):
    """Cylindrical packmol packing."""
    context_inputs = {
        'tolerance': 'metadata->step_specific->packing->surfactant_substrate->tolerance',
        'cylinders': 'run->template->context->cylinders',
        'ioncylinders': 'run->template->context->ioncylinders',
        'movebadrandom': 'run->template->context->movebadrandom',
    }


class HemicylindricalPackingMain(CylindricalPackingMain):
    pass


class MonolayerPackingWorkflowGenerator(ChainWorkflowGenerator):
    """Pack a monolayer on flat substrate with PACKMOL sub workflow.

    Concatenates
    - MonolayerPackingConstraintsMain
    - MonolayerPackingContextMain
    - MonolayerPackingMain
    """

    def __init__(self, *args, **kwargs):
        sub_wf_components = [
            MonolayerPackingConstraintsMain(*args, **kwargs),
            LayeredPackingContextMain(*args, **kwargs),
            LayeredPackingMain(*args, **kwargs),
        ]
        sub_wf_name = 'MonolayerPacking'
        if 'wf_name_prefix' not in kwargs:
            kwargs['wf_name_prefix'] = sub_wf_name
        else:
            kwargs['wf_name_prefix'] = ':'.join((kwargs['wf_name_prefix'], sub_wf_name))
        super().__init__(sub_wf_components, *args, **kwargs)


class BilayerPackingWorkflowGenerator(ChainWorkflowGenerator):
    """Pack a monolayer on flat substrate with PACKMOL sub workflow.

    Concatenates
    - BilayerPackingConstraintsMain
    - BilayerPackingContextMain
    - BilayerPackingMain
    """

    def __init__(self, *args, **kwargs):
        sub_wf_components = [
            BilayerPackingConstraintsMain(*args, **kwargs),
            LayeredPackingContextMain(*args, **kwargs),
            LayeredPackingMain(*args, **kwargs),
        ]
        sub_wf_name = 'BilayerPacking'
        if 'wf_name_prefix' not in kwargs:
            kwargs['wf_name_prefix'] = sub_wf_name
        else:
            kwargs['wf_name_prefix'] = ':'.join((kwargs['wf_name_prefix'], sub_wf_name))
        super().__init__(sub_wf_components, *args, **kwargs)


class CylindricalPackingWorkflowGenerator(ChainWorkflowGenerator):
    """Pack cylinders on flat substrate with PACKMOL sub workflow.

    Concatenates
    - CylindricalPackingConstraintsMain
    - CylindricalPackingContextMain
    - CylindricalPackingMain
    """

    def __init__(self, *args, **kwargs):
        sub_wf_components = [
            CylindricalPackingConstraintsMain(*args, **kwargs),
            CylindricalPackingContextMain(*args, **kwargs),
            CylindricalPackingMain(*args, **kwargs),
        ]
        sub_wf_name = 'CylindricalPacking'
        if 'wf_name_prefix' not in kwargs:
            kwargs['wf_name_prefix'] = sub_wf_name
        else:
            kwargs['wf_name_prefix'] = ':'.join((kwargs['wf_name_prefix'], sub_wf_name))
        super().__init__(sub_wf_components, *args, **kwargs)


class HemicylindricalPackingWorkflowGenerator(ChainWorkflowGenerator):
    """Pack cylinders on flat substrate with PACKMOL sub workflow.

    Concatenates
    - HemicylindricalPackingConstraintsMain
    - HemicylindricalPackingContextMain
    - HemicylindricalPackingMain
    """

    def __init__(self, *args, **kwargs):
        sub_wf_components = [
            HemicylindricalPackingConstraintsMain(*args, **kwargs),
            HemicylindricalPackingContextMain(*args, **kwargs),
            HemicylindricalPackingMain(*args, **kwargs),
        ]
        sub_wf_name = 'HemicylindricalPacking'
        if 'wf_name_prefix' not in kwargs:
            kwargs['wf_name_prefix'] = sub_wf_name
        else:
            kwargs['wf_name_prefix'] = ':'.join((kwargs['wf_name_prefix'], sub_wf_name))
        super().__init__(sub_wf_components, *args, **kwargs)


# class PackingConstraintSpheresWorkflowGenerator(
#         DefaultPullMixin, DefaultPushMixin,
#         CylindricalPackingWorkflowGenerator,
#         ):
#     def __init__(self, *args, **kwargs):
#         sub_wf_name = 'PackingConstraintSpheres'
#         if 'wf_name_prefix' not in kwargs:
#             kwargs['wf_name_prefix'] = sub_wf_name
#         else:
#             kwargs['wf_name_prefix'] = ':'.join((kwargs['wf_name_prefix'], sub_wf_name))
#         super().__init__(*args, **kwargs)
