# -*- coding: utf-8 -*-
"""Packing constraint spheres sub workflow."""
import datetime
import glob
import os

from fireworks import Firework
from fireworks.user_objects.firetasks.dataflow_tasks import JoinListTask
from fireworks.user_objects.firetasks.filepad_tasks import GetFilesByQueryTask
from fireworks.user_objects.firetasks.filepad_tasks import AddFilesTask
from fireworks.user_objects.firetasks.script_task import PyTask
from fireworks.user_objects.firetasks.templatewriter_task import TemplateWriterTask

from imteksimfw.fireworks.user_objects.firetasks.cmd_tasks import CmdTask, PickledPyEnvTask

# from jlhpy.utilities.vis.plot_side_views_with_spheres import plot_side_views_with_spheres_via_ase

from imteksimfw.fireworks.utilities.serialize import serialize_module_obj
from jlhpy.utilities.wf.workflow_generator import SubWorkflowGenerator

from jlhpy.utilities.templates.spherical_packing import generate_pack_sphere_packmol_template_context
from jlhpy.utilities.vis.plot_side_views_with_spheres import \
    plot_side_views_with_spheres_via_parmed

import jlhpy.utilities.wf.file_config as file_config


class SphericalSurfactantPackingSubWorkflowGenerator(SubWorkflowGenerator):
    """Packing constraint spheres sub workflow.

    Inputs:
        - # metadata->system->counterion->name (str)
        - metadata->system->indenter->bounding_sphere->center ([float])
        - metadata->system->indenter->bounding_sphere->radius (float)
        - # metadata->system->surfactant->name (str)
        - metadata->system->surfactant->nmolecules (int)
        - metadata->system->packing->surfactant_indenter->inner_atom_index (int)
        - metadata->system->packing->surfactant_indenter->outer_atom_index (int)
        - metadata->system->packing->surfactant_indenter->constraints->R_inner (float)
        - metadata->system->packing->surfactant_indenter->constraints->R_inner_constraint (float)
        - metadata->system->packing->surfactant_indenter->constraints->R_outer_constraint (float)
        - metadata->system->packing->surfactant_indenter->constraints->R_outer (float)
        - metadata->system->packing->surfactant_indenter->tolerance (float)
    """

    def push_infiles(self, fp):
        fp_files = []

        step_label = self.get_step_label('push_infiles')

        # input files
        infiles = sorted(glob.glob(os.path.join(
            self.infile_prefix,
            file_config.PACKMOL_SUBDIR,
            file_config.PACKMOL_SPHERES_TEMPLATE)))

        # metadata common to all these files
        metadata = {
            'project': self.project_id,
            'type': 'template',
            'step': step_label,
            'name': file_config.PACKMOL_SPHERES_TEMPLATE
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

        # data files
        datafiles = [
            *sorted(glob.glob(os.path.join(
                self.infile_prefix,
                file_config.PDB_SUBDIR,
                file_config.SURFACTANT_PDB))),
            *sorted(glob.glob(os.path.join(
                self.infile_prefix,
                file_config.PDB_SUBDIR,
                file_config.COUNTERION_PDB))),
            *sorted(glob.glob(os.path.join(
                self.infile_prefix,
                file_config.INDENTER_SUBDIR,
                file_config.INDENTER_PDB)))]

        files = {os.path.basename(f): f for f in datafiles}

        # metadata common to all these files
        metadata = {
            'project': self.project_id,
            'type': 'data',
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
        step_label = self.get_step_label('coordinates pull')

        files_in = {}
        files_out = {
            'indenter_file':   'indenter.pdb',
            'surfatcant_file': 'surfactant.pdb',
            'counterion_file': 'counterion.pdb',
        }

        fts_coordinates_pull = [
            GetFilesByQueryTask(
                query={
                    'metadata->project': self.project_id,
                    'metadata->name':    file_config.INDENTER_PDB,
                },
                limit=1,
                new_file_names=['indenter.pdb']),
            GetFilesByQueryTask(
                query={
                    'metadata->project': self.project_id,
                    'metadata->name':    file_config.SURFACTANT_PDB,
                },
                limit=1,
                new_file_names=['surfactant.pdb']),
            GetFilesByQueryTask(
                query={
                    'metadata->project': self.project_id,
                    'metadata->name':    file_config.COUNTERION_PDB,
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
        step_label = self.get_step_label('inputs pull')

        files_in = {}
        files_out = {
            'input_file':      'input.template',
        }

        fts_inputs_pull = [GetFilesByQueryTask(
            query={
                'metadata->project': self.project_id,
                'metadata->name':    file_config.PACKMOL_SPHERES_TEMPLATE,
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

        # Context generator
        # -----------------
        step_label = self.get_step_label('packmol template context')

        files_in = {}
        files_out = {}

        func_str = serialize_module_obj(generate_pack_sphere_packmol_template_context)

        fts_context_generator = [PickledPyEnvTask(
            func=func_str,
            kwargs={
                'surfactant': 'surfactant',
                'counterion': 'counterion',
            },
            kwargs_inputs={
                'C': 'metadata->system->indenter->bounding_sphere->center',
                'R': 'metadata->system->indenter->bounding_sphere->radius',
                'R_inner': 'metadata->system->packing->surfactant_indenter->constraints->R_inner',
                'R_inner_constraint': 'metadata->system->packing->surfactant_indenter->constraints->R_inner_constraint',  # shell inner radius
                'R_outer_constraint': 'metadata->system->packing->surfactant_indenter->constraints->R_outer_constraint',  # shell outer radius
                'R_outer': 'metadata->system->packing->surfactant_indenter->constraints->R_outer',
                'sfN': 'metadata->system->surfactant->nmolecules',  # number of surfactant molecules
                'inner_atom_number': 'metadata->system->packing->surfactant_indenter->inner_atom_index',  # inner atom
                'outer_atom_number': 'metadata->system->packing->surfactant_indenter->outer_atom_index',  # outer atom
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
            'system_name': 'default',
            'header': ', '.join((
                self.project_id,
                self.get_fw_label(step_label),
                str(datetime.datetime.now()))),

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

        ft_template = TemplateWriterTask({
            'context': static_packmol_script_context,
            'context_inputs': context_inputs,
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
            parents=[fw_context_generator, fw_inputs_pull])

        fw_list.append(fw_template)

        # PACKMOL
        # -------
        step_label = self.get_step_label('packmol')

        files_in = {
            'input_file': 'input.inp',
            'indenter_file': 'indenter.pdb',
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
                # opt=['< input.inp'],
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
            parents=[fw_coordinates_pull, fw_template])

        fw_list.append(fw_pack)

        return fw_list, [fw_pack], [fw_context_generator]

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

    def vis_main(self, fws_root=[]):
        fw_list = []

        # Join radii and centers
        # ----------------------
        step_label = self.get_step_label('join radii in list')

        files_in = {}
        files_out = {}

        fts_join = [
            JoinListTask(
                inputs=[
                    'metadata->system->indenter->bounding_sphere->radius',
                    'metadata->system->packing->surfactant_indenter->constraints->R_inner',
                    'metadata->system->packing->surfactant_indenter->constraints->R_inner_constraint',
                    'metadata->system->packing->surfactant_indenter->constraints->R_outer_constraint',
                    'metadata->system->packing->surfactant_indenter->constraints->R_outer',
                ],
                output='metadata->system->packing->surfactant_indenter->constraints->R_list',
            ),
            JoinListTask(
                inputs=[
                    'metadata->system->indenter->bounding_sphere->center',
                    'metadata->system->indenter->bounding_sphere->center',
                    'metadata->system->indenter->bounding_sphere->center',
                    'metadata->system->indenter->bounding_sphere->center',
                    'metadata->system->indenter->bounding_sphere->center',
                ],
                output='metadata->system->packing->surfactant_indenter->constraints->C_list',
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
                'metadata->system->packing->surfactant_indenter->constraints->C_list',
                'metadata->system->packing->surfactant_indenter->constraints->R_list',
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

        return fw_list, [fw_vis], [fw_join]

    def vis_push(self, fws_root=[]):
        fw_list = []

        step_label = self.get_step_label('vis_push')

        files_in = {'png_file': 'default.png'}
        files_out = {}

        fts_push = [AddFilesTask({
            'compress': True,
            'paths': "default.png",
            'metadata': {
                'project': self.project_id,
                'datetime': str(datetime.datetime.now()),
                'type':    'png_file',
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
