# -*- coding: utf-8 -*-
"""Packing constraints sub workflows."""

from abc import abstractmethod

import datetime
import glob
import os.path
import warnings

from fireworks.user_objects.firetasks.script_task import PyTask
from fireworks.user_objects.firetasks.filepad_tasks import GetFilesByQueryTask
from fireworks.user_objects.firetasks.filepad_tasks import AddFilesTask
from fireworks.user_objects.firetasks.templatewriter_task import TemplateWriterTask

# from fireworks.user_objects.firetasks.dataflow_tasks import JoinListTask

from imteksimfw.fireworks.user_objects.firetasks.cmd_tasks \
    import CmdTask, EvalPyEnvTask, PickledPyEnvTask, PyEnvTask
from imteksimfw.fireworks.utilities.serialize import serialize_module_obj

from jlhpy.utilities.geometry.morphology import (
    monolayer_above_substrate, bilayer_above_substrate, cylinders_above_substrate, hemicylinders_above_substrate)
from jlhpy.utilities.templates.flat_packing import (
    generate_alternating_multilayer_packmol_template_context,
    generate_inverse_alternating_multilayer_packmol_template_context)
from jlhpy.utilities.templates.cylindrical_packing import (
    generate_cylinders_packmol_template_context, generate_upper_hemicylinders_packmol_template_context)

from jlhpy.utilities.wf.workflow_generator import (
    WorkflowGenerator, ChainWorkflowGenerator, ProcessAnalyzeAndVisualize)
from jlhpy.utilities.wf.mixin.mixin_wf_storage import (
   DefaultPullMixin, DefaultPushMixin)

import jlhpy.utilities.wf.file_config as file_config
import jlhpy.utilities.wf.file_config as phys_config


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

    @property
    @abstractmethod
    def func_str(self):
        pass

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
            )]

        fw_constraints = self.build_fw(
            fts_constraints, step_label,
            parents=fws_root,
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_noqueue_category'])
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


class PackingContextMain(WorkflowGenerator):
    """Packing template context sub workflow.

    Inputs:
    - metadata->system->counterion->name (str)
    - metadata->system->surfactant->name (str)
    - metadata->system->surfactant->nmolecules (int)
    - metadata->system->surfactant->head_atom->index (int)
    - metadata->system->surfactant->tail_atom->index (int)

    Outputs:
    - run->template->context
    """

    pass


# TODO: bilayer
class LayeredPackingContextMain(PackingContextMain):
    """Layered packing template context sub workflow.

    Head groups will face upwards in first layer.

    Inputs:
    - metadata->step_specific->packing->surfactant_substrate->constraints->layers (list)
    - metadata->step_specific->packing->surfactant_substrate->tolerance (float)
    """

    func_str = serialize_module_obj(generate_alternating_multilayer_packmol_template_context)

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
                kwargs={
                    'surfactant': 'surfactant',
                    'counterion': 'counterion',
                },
                kwargs_inputs={
                    'layers': 'metadata->step_specific->packing->surfactant_substrate->constraints->layers',
                    'sfN': 'metadata->system->surfactant->nmolecules',
                    'tail_atom_number': 'metadata->system->surfactant->tail_atom->index',
                    'head_atom_number': 'metadata->system->surfactant->head_atom->index',
                    'tolerance': 'metadata->step_specific->packing->surfactant_substrate->tolerance',
                },
                outputs=[
                    'run->template->context',
                ],
                stderr_file='std.err',
                stdout_file='std.out',
                stdlog_file='std.log',
                propagate=False,
            )
        ]

        fw_context = self.build_fw(
            fts_context, step_label,
            parents=fws_root,
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_noqueue_category'])
        fw_list.append(fw_context)

        return fw_list, [fw_context], [fw_context]

class InverseLayeredPackingContextMain(LayeredPackingContextMain):
    """Layered packing template context sub workflow.

    Head groups wil face downwards in first layer."""

    func_str = serialize_module_obj(generate_inverse_alternating_multilayer_packmol_template_context)


class CylindricalPackingContextMain(PackingContextMain):
    """Cylindrical packing template context sub workflow.

    Inputs:
    - metadata->step_specific->packing->surfactant_substrate->constraints->cylinders (list)
    - metadata->step_specific->packing->surfactant_substrate->tolerance (float)
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
                kwargs={
                    'surfactant': 'surfactant',
                    'counterion': 'counterion',
                },
                kwargs_inputs={
                    'cylinders': 'metadata->step_specific->packing->surfactant_substrate->constraints->cylinders',
                    'sfN': 'metadata->system->surfactant->nmolecules',
                    'inner_atom_number': 'metadata->system->surfactant->tail_atom->index',
                    'outer_atom_number': 'metadata->system->surfactant->head_atom->index',
                    'tolerance': 'metadata->step_specific->packing->surfactant_substrate->tolerance',
                },
                outputs=[
                    'run->template->context',
                ],
                stderr_file='std.err',
                stdout_file='std.out',
                stdlog_file='std.log',
                propagate=False,
            )
        ]

        fw_context = self.build_fw(
            fts_context, step_label,
            parents=fws_root,
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_noqueue_category'])

        fw_list.append(fw_context)

        return fw_list, [fw_context], [fw_context]


class HemicylindricalPackingContextMain(CylindricalPackingContextMain):
    """Hemicylindrical packing template context sub workflow."""

    func_str = serialize_module_obj(generate_upper_hemicylinders_packmol_template_context)


class PackingMain(DefaultPushMixin, WorkflowGenerator):
    """Packmol packing."""

    context_inputs = {
        'tolerance': 'metadata->step_specific->packing->surfactant_substrate->tolerance',
        'layers': 'run->template->context->layers',
        'ionlayers': 'run->template->context->ionlayers',
        'movebadrandom': 'run->template->context->movebadrandom',
    }

    def push_infiles(self, fp):
        fp_files = []

        step_label = self.get_step_label('push_infiles')

        # input files
        infiles = sorted(glob.glob(os.path.join(
            self.infile_prefix,
            file_config.PACKMOL_SUBDIR,
            file_config.PACKMOL_FLAT_TEMPLATE)))

        # metadata common to all these files
        metadata = {
            'project': self.project_id,
            'type': 'template',
            'step': step_label,
            'name': file_config.PACKMOL_FLAT_TEMPLATE
        }

        files = {os.path.basename(f): f for f in infiles}

        # insert these input files into data base
        for name, file_path in files.items():
            identifier = '/'.join((self.project_id, name))  # identifier is like a path on a file system
            fp_files.append(
                fp.add_file(
                    file_path,
                    identifier=identifier,
                    metadata=metadata))

        # datafiles:

        # try to get surfactant pdb file from kwargs
        try:
            surfactant = self.kwargs["system"]["surfactant"]["name"]
        except:
            surfactant = phys_config.DEFAULT_SURFACTANT
            warnings.warn("No surfactant specified, falling back to {:s}.".format(surfactant))

        surfactant_pdb = file_config.SURFACTANT_PDB_PATTERN.format(name=surfactant)

        datafiles = sorted(glob.glob(os.path.join(
            self.infile_prefix,
            file_config.PDB_SUBDIR,
            surfactant_pdb)))

        files = {os.path.basename(f): f for f in datafiles}

        # metadata common to all these files
        metadata = {
            'project': self.project_id,
            'type': 'surfactant_file',
            'step': step_label,
            **self.kwargs
        }

        # insert these input files into data base
        for name, file_path in files.items():
            identifier = '/'.join((self.project_id, name))
            metadata["name"] = name
            fp_files.append(
                fp.add_file(
                    file_path,
                    identifier=identifier,
                    metadata=metadata))

        # try to get counterion pdb file from kwargs
        try:
            counterion = self.kwargs["system"]["counterion"]["name"]
        except:
            counterion = phys_config.DEFAULT_COUNTERION
            warnings.warn("No counterion specified, falling back to {:s}.".format(surfactant))

        counterion_pdb = file_config.COUNTERION_PDB_PATTERN.format(name=counterion)

        datafiles = sorted(glob.glob(os.path.join(
            self.infile_prefix,
            file_config.PDB_SUBDIR,
            counterion_pdb)))

        files = {os.path.basename(f): f for f in datafiles}

        # metadata common to all these files
        metadata = {
            'project': self.project_id,
            'type': 'counterion_file',
            'step': step_label,
            **self.kwargs
        }

        # insert these input files into data base
        for name, file_path in files.items():
            identifier = '/'.join((self.project_id, name))
            metadata["name"] = name
            fp_files.append(
                fp.add_file(
                    file_path,
                    identifier=identifier,
                    metadata=metadata))

        return fp_files

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

        fw_coordinates_pull = self.build_fw(
            fts_coordinates_pull, step_label,
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_noqueue_category'])
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

        fw_inputs_pull = self.build_fw(
            fts_inputs_pull, step_label,
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_noqueue_category'])

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

        fts_template = [TemplateWriterTask({
            'context': static_packmol_script_context,
            'context_inputs': self.context_inputs,
            'template_file': 'input.template',
            'template_dir': '.',
            'output_file': 'input.inp'})]

        fw_template = self.build_fw(
            fts_template, step_label,
            parents=[*fws_root, fw_inputs_pull],
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_noqueue_category'])

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
                stdin_file='input.inp',
                stderr_file='std.err',
                stdout_file='std.out',
                store_stdout=True,
                store_stderr=True,
                use_shell=False,
                fizzle_bad_rc=True),
            PyTask(  # check for output file and fizzle if not existant
                func='open',
                args=['default_packmol.pdb'])]

        fw_pack = self.build_fw(
            fts_pack, step_label,
            parents=[fw_coordinates_pull, fw_template],
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_queue_category'],
            queueadapter=self.hpc_specs['single_core_job_queueadapter_defaults'])
        fw_list.append(fw_pack)

        return fw_list, [fw_pack], [fw_template, fw_pack]


class LayeredPackingMain(PackingMain):
    """Layered packmol packing."""
    context_inputs = {
        'tolerance': 'metadata->step_specific->packing->surfactant_substrate->tolerance',
        'layers': 'run->template->context->layers',
        'ionlayers': 'run->template->context->ionlayers',
        'nloop': 'run->template->context->nloop',
        'maxit': 'run->template->context->maxit',
    }


class CylindricalPackingMain(PackingMain):
    """Cylindrical packmol packing."""
    context_inputs = {
        'tolerance': 'metadata->step_specific->packing->surfactant_substrate->tolerance',
        'cylinders': 'run->template->context->cylinders',
        'ioncylinders': 'run->template->context->ioncylinders',
        'movebadrandom': 'run->template->context->movebadrandom',
        'nloop': 'run->template->context->nloop',
        'maxit': 'run->template->context->maxit',
    }


class HemicylindricalPackingMain(CylindricalPackingMain):
    pass


class MonolayerPacking(ChainWorkflowGenerator):
    """Pack a monolayer on flat substrate with PACKMOL sub workflow.

    Concatenates
    - MonolayerPackingConstraintsMain
    - MonolayerPackingContextMain
    - MonolayerPackingMain
    """

    def __init__(self, *args, **kwargs):
        sub_wf_components = [
            MonolayerPackingConstraintsMain,
            LayeredPackingContextMain,
            LayeredPackingMain,
        ]
        super().__init__(*args, sub_wf_components=sub_wf_components, **kwargs)


class BilayerPacking(ChainWorkflowGenerator):
    """Pack a monolayer on flat substrate with PACKMOL sub workflow.

    Concatenates
    - BilayerPackingConstraintsMain
    - BilayerPackingContextMain
    - BilayerPackingMain
    """

    def __init__(self, *args, **kwargs):
        sub_wf_components = [
            BilayerPackingConstraintsMain,
            InverseLayeredPackingContextMain,
            LayeredPackingMain,
        ]
        super().__init__(*args, sub_wf_components=sub_wf_components, **kwargs)


class CylindricalPacking(ChainWorkflowGenerator):
    """Pack cylinders on flat substrate with PACKMOL sub workflow.

    Concatenates
    - CylindricalPackingConstraintsMain
    - CylindricalPackingContextMain
    - CylindricalPackingMain
    """

    def __init__(self, *args, **kwargs):
        sub_wf_components = [
            CylindricalPackingConstraintsMain,
            CylindricalPackingContextMain,
            CylindricalPackingMain,
        ]
        super().__init__(*args, sub_wf_components=sub_wf_components, **kwargs)


class HemicylindricalPacking(ChainWorkflowGenerator):
    """Pack cylinders on flat substrate with PACKMOL sub workflow.

    Concatenates
    - HemicylindricalPackingConstraintsMain
    - HemicylindricalPackingContextMain
    - HemicylindricalPackingMain
    """

    def __init__(self, *args, **kwargs):
        sub_wf_components = [
            HemicylindricalPackingConstraintsMain,
            HemicylindricalPackingContextMain,
            HemicylindricalPackingMain,
        ]
        super().__init__(*args, sub_wf_components=sub_wf_components, **kwargs)
