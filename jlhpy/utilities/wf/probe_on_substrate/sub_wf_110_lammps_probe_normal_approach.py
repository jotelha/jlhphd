# -*- coding: utf-8 -*-
"""Probe on substrate normal approach."""

import datetime
import glob
import logging
import os
import pymongo
import warnings

from fireworks import Firework, Workflow
from fireworks.user_objects.firetasks.fileio_tasks import FileTransferTask
from fireworks.user_objects.firetasks.filepad_tasks import GetFilesByQueryTask
from fireworks.user_objects.firetasks.templatewriter_task import TemplateWriterTask
from imteksimfw.fireworks.user_objects.firetasks.cmd_tasks import CmdTask
from imteksimfw.fireworks.user_objects.firetasks.recover_tasks import RecoverTask

from jlhpy.utilities.wf.workflow_generator import (
    WorkflowGenerator, ProcessAnalyzeAndVisualize)
from jlhpy.utilities.wf.mixin.mixin_wf_storage import (
   DefaultPullMixin, DefaultPushMixin)

from jlhpy.utilities.wf.building_blocks.sub_wf_lammps_analysis import LAMMPSTrajectoryAnalysis

import jlhpy.utilities.wf.file_config as file_config
import jlhpy.utilities.wf.phys_config as phys_config

# TODO: separate exclusions for recovery, restart and detour


class LAMMPSProbeNormalApproachMain(WorkflowGenerator):
    """
    Probe approach production.

    inputs:
    - metadata->step_specific->probe_normal_approach->freeze_substrate_layer
    - metadata->step_specific->probe_normal_approach->rigid_indenter_core_radius
    - metadata->step_specific->probe_normal_approach->constant_indenter_velocity
    - metadata->step_specific->probe_normal_approach->temperature

    - metadata->step_specific->probe_normal_approach->steps
    - metadata->step_specific->probe_normal_approach->netcdf_frequency
    - metadata->step_specific->probe_normal_approach->thermo_frequency
    - metadata->step_specific->probe_normal_approach->thermo_average_frequency

    - metadata->step_specific->probe_normal_approach->ewald_accuracy
    - metadata->step_specific->probe_normal_approach->coulomb_cutoff
    - metadata->step_specific->probe_normal_approach->neigh_delay
    - metadata->step_specific->probe_normal_approach->neigh_every
    - metadata->step_specific->probe_normal_approach->neigh_check
    - metadata->step_specific->probe_normal_approach->skin_distance

    - metadata->system->substrate->element
    - metadata->system->substrate->lmp->type

    dynamic infiles:
        only queried in pull stub, otherwise expected through data flow

    - data_file:       default.lammps

    static infiles:
        always queried within main trunk

    - input_template: lmp.input.template
    - mass_file: mass.input
    - coeff_file: coeff.input
    - eam_file:   default.eam.alloy

    outfiles:
    - coeff_file:      coeff.input  # untouched
    - data_file:       default.lammps
    - eam_file:        default.eam.alloy  # untouched
    - index_file:      groups.ndx
    - input_file:      default.input  # untouched
    - log_file:        log.lammps
    - mass_file:       mass.input  # untouched
    - thermo_ave_file: thermo_ave.out
    - trajectory_file: default.nc
    """
    def push_infiles(self, fp):

        # static infiles for main
        # -----------------------
        step_label = self.get_step_label('push_infiles')

        # try to get surfactant pdb file from kwargs
        try:
            surfactant = self.kwargs["system"]["surfactant"]["name"]
        except:
            surfactant = phys_config.DEFAULT_SURFACTANT
            warnings.warn("No surfactant specified, falling back to {:s}.".format(surfactant))

        lmp_coeff_input = file_config.LMP_COEFF_HYBRID_INPUT_PATTERN.format(name=surfactant)

        glob_patterns = [
            os.path.join(
                self.infile_prefix,
                file_config.LMP_INPUT_TEMPLATE_SUBDIR,
                file_config.LMP_INPUT_TEMPLATE),
            os.path.join(
                self.infile_prefix,
                file_config.LMP_FF_SUBDIR,
                lmp_coeff_input),
            os.path.join(
                self.infile_prefix,
                file_config.LMP_FF_SUBDIR,
                file_config.LMP_MASS_INPUT),
            os.path.join(
                self.infile_prefix,
                file_config.LMP_FF_SUBDIR,
                file_config.LMP_EAM_ALLOY)
        ]

        infiles = sorted([
            file for pattern in glob_patterns for file in glob.glob(pattern)])

        files = {os.path.basename(f): f for f in infiles}

        # metadata common to all these files
        metadata = {
            'project': self.project_id,
            'type': 'input',
            'step': step_label,
        }

        fp_files = []

        # insert these input files into data base
        for name, file_path in files.items():
            identifier = '/'.join((self.project_id, name))
            metadata['name'] = name
            fp_files.append(
                fp.add_file(
                    file_path,
                    identifier=identifier,
                    metadata=metadata))

        return fp_files

    def main(self, fws_root=[]):
        # try to get surfactant pdb file from kwargs
        try:
            surfactant = self.kwargs["system"]["surfactant"]["name"]
        except:
            surfactant = phys_config.DEFAULT_SURFACTANT
            warnings.warn("No surfactant specified, falling back to {:s}.".format(surfactant))

        lmp_coeff_input = file_config.LMP_COEFF_HYBRID_INPUT_PATTERN.format(name=surfactant)

        fw_list = []

        # query input files
        # -----------------
        step_label = self.get_step_label('input_files_pull')

        files_in = {}
        files_out = {
            'input_template': 'lmp.input.template',
            'mass_file':      'mass.input',
            'coeff_file':     'coeff.input',
            'eam_file':       'default.eam.alloy',
        }

        fts_pull = [
            GetFilesByQueryTask(
                query={
                    'metadata->project': self.project_id,
                    'metadata->name':    file_config.LMP_INPUT_TEMPLATE,
                },
                sort_key='metadata.datetime',
                sort_direction=pymongo.DESCENDING,
                limit=1,
                new_file_names=['lmp.input.template']),
            GetFilesByQueryTask(
                query={
                    'metadata->project': self.project_id,
                    'metadata->name':    lmp_coeff_input,
                },
                sort_key='metadata.datetime',
                sort_direction=pymongo.DESCENDING,
                limit=1,
                new_file_names=['coeff.input']),
            GetFilesByQueryTask(
                query={
                    'metadata->project': self.project_id,
                    'metadata->name':    file_config.LMP_MASS_INPUT,
                },
                sort_key='metadata.datetime',
                sort_direction=pymongo.DESCENDING,
                limit=1,
                new_file_names=['mass.input']),
            GetFilesByQueryTask(
                query={
                    'metadata->project': self.project_id,
                    'metadata->name':    file_config.LMP_EAM_ALLOY,
                },
                sort_key='metadata.datetime',
                sort_direction=pymongo.DESCENDING,
                limit=1,
                new_file_names=['default.eam.alloy']),
        ]

        fw_pull = self.build_fw(
            fts_pull, step_label,
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_noqueue_category'])

        fw_list.append(fw_pull)

        # fill input file template
        # -------------------
        step_label = self.get_step_label('fill_template')

        files_in = {
            'input_template': 'default.input.template',

        }
        files_out = {
            'input_file': 'default.input',
        }

        # Jinja2 context:
        static_template_context = {
            'mode':                    'production',
            'coeff_infile':            'coeff.input',
            'compute_group_properties': True,
            'data_file':                'datafile.lammps',
            # 'dilate_solution_only':     True,
            'has_indenter':             True,
            'mpiio':                    True,
            # 'pressurize_z_only':        True,
            'restrained_indenter':      False,
            'rigid_h_bonds':            True,
            'store_forces':             True,
            'temper_solid_only':        True,
            'use_barostat':             False,
            'use_dpd_tstat':            True,
            'use_eam':                  True,
            'use_ewald':                True,
            'write_coeff_to_datafile':  False,
        }

        dynamic_template_context = {
            'freeze_substrate_layer':     'metadata->step_specific->probe_normal_approach->freeze_substrate_layer',
            'rigid_indenter_core_radius': 'metadata->step_specific->probe_normal_approach->rigid_indenter_core_radius',
            'constant_indenter_velocity': 'metadata->step_specific->probe_normal_approach->constant_indenter_velocity',
            'temperatureT':               'metadata->step_specific->probe_normal_approach->temperature',

            'production_steps': 'metadata->step_specific->probe_normal_approach->steps',
            'netcdf_frequency': 'metadata->step_specific->probe_normal_approach->netcdf_frequency',
            'thermo_frequency': 'metadata->step_specific->probe_normal_approach->thermo_frequency',
            'thermo_average_frequency': 'metadata->step_specific->probe_normal_approach->thermo_average_frequency',
            'restart_frequency': 'metadata->step_specific->probe_normal_approach->restart_frequency',

            'ewald_accuracy':   'metadata->step_specific->probe_normal_approach->ewald_accuracy',
            'coulomb_cutoff':   'metadata->step_specific->probe_normal_approach->coulomb_cutoff',
            'neigh_delay':      'metadata->step_specific->probe_normal_approach->neigh_delay',
            'neigh_every':      'metadata->step_specific->probe_normal_approach->neigh_every',
            'neigh_check':      'metadata->step_specific->probe_normal_approach->neigh_check',
            'skin_distance':    'metadata->step_specific->probe_normal_approach->skin_distance',

            'substrate_element': 'metadata->system->substrate->element',
            'substrate_type': 'metadata->system->substrate->lmp->type',
        }

        fts_template = [TemplateWriterTask({
            'context_inputs': dynamic_template_context,
            'context': static_template_context,
            'template_file': 'default.input.template',
            'template_dir': '.',
            'output_file': 'default.input'})]

        fw_template = self.build_fw(
            fts_template, step_label,
            parents=[fw_pull, *fws_root],
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_noqueue_category'])

        fw_list.append(fw_template)

        # LAMMPS run
        # ----------
        step_label = self.get_step_label('lmp_run')

        files_in = {
            'data_file':  'datafile.lammps',
            'input_file': 'default.input',
            'mass_file':  'mass.input',
            'coeff_file': 'coeff.input',
            'eam_file':   'default.eam.alloy',
        }
        files_out = {
            'coeff_file':      'coeff.input',  # untouched
            'data_file':       'default.lammps',
            'eam_file':        'default.eam.alloy',  # untouched
            'index_file':      'groups.ndx',
            'conserved_input_file': 'default.input',  # untouched
            'log_file':        'log.lammps',
            'mass_file':       'mass.input',  # untouched
            'thermo_ave_file': 'thermo_ave.out',
            'trajectory_file': 'default.nc',
        }
        fts_lmp_run = [CmdTask(
            cmd='lmp',
            opt=['-in', 'default.input'],
            env='python',
            stderr_file='std.err',
            stdout_file='std.out',
            stdlog_file='std.log',
            store_stdout=True,
            store_stderr=True,
            fizzle_bad_rc=True)]

        fw_lmp_run = self.build_fw(
            fts_lmp_run, step_label,
            parents=[fw_template, fw_pull, *fws_root],
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_queue_category'],
            queueadapter=self.hpc_specs['four_nodes_job_queueadapter_defaults']
        )

        fw_list.append(fw_lmp_run)

        return fw_list, [fw_lmp_run], [fw_lmp_run, fw_template]


class LAMMPSRecoverableProbeNormalApproachMain(WorkflowGenerator):
    """
    Probe aaproach production.

    inputs:
    - metadata->step_specific->probe_normal_approach->freeze_substrate_layer
    - metadata->step_specific->probe_normal_approach->rigid_indenter_core_radius
    - metadata->step_specific->probe_normal_approach->constant_indenter_velocity
    - metadata->step_specific->probe_normal_approach->temperature

    - metadata->step_specific->probe_normal_approach->steps
    - metadata->step_specific->probe_normal_approach->netcdf_frequency
    - metadata->step_specific->probe_normal_approach->thermo_frequency
    - metadata->step_specific->probe_normal_approach->thermo_average_frequency

    - metadata->step_specific->probe_normal_approach->ewald_accuracy
    - metadata->step_specific->probe_normal_approach->coulomb_cutoff
    - metadata->step_specific->probe_normal_approach->neigh_delay
    - metadata->step_specific->probe_normal_approach->neigh_every
    - metadata->step_specific->probe_normal_approach->neigh_check
    - metadata->step_specific->probe_normal_approach->skin_distance

    - metadata->system->substrate->element
    - metadata->system->substrate->lmp->type

    dynamic infiles:
        only queried in pull stub, otherwise expected through data flow

    - data_file:       default.lammps

    static infiles:
        always queried within main trunk

    - input_template: lmp.input.template
    - mass_file: mass.input
    - coeff_file: coeff.input
    - eam_file:   default.eam.alloy

    outfiles:
    - coeff_file:      coeff.input  # untouched
    - data_file:       default.lammps
    - eam_file:        default.eam.alloy  # untouched
    - index_file:      groups.ndx
    - input_file:      default.input  # untouched
    - log_file:        log.lammps
    - mass_file:       mass.input  # untouched
    - thermo_ave_file: thermo_ave.out
    - trajectory_file: default.nc
    """
    def push_infiles(self, fp):

        # static infiles for main
        # -----------------------
        step_label = self.get_step_label('push_infiles')

        # try to get surfactant pdb file from kwargs
        try:
            surfactant = self.kwargs["system"]["surfactant"]["name"]
        except:
            surfactant = phys_config.DEFAULT_SURFACTANT
            warnings.warn("No surfactant specified, falling back to {:s}.".format(surfactant))

        lmp_coeff_input = file_config.LMP_COEFF_HYBRID_INPUT_PATTERN.format(name=surfactant)

        glob_patterns = [
            os.path.join(
                self.infile_prefix,
                file_config.LMP_INPUT_TEMPLATE_SUBDIR,
                file_config.LMP_INPUT_TEMPLATE),
            os.path.join(
                self.infile_prefix,
                file_config.LMP_FF_SUBDIR,
                lmp_coeff_input),
            os.path.join(
                self.infile_prefix,
                file_config.LMP_FF_SUBDIR,
                file_config.LMP_MASS_INPUT),
            os.path.join(
                self.infile_prefix,
                file_config.LMP_FF_SUBDIR,
                file_config.LMP_EAM_ALLOY)
        ]

        infiles = sorted([
            file for pattern in glob_patterns for file in glob.glob(pattern)])

        files = {os.path.basename(f): f for f in infiles}

        # metadata common to all these files
        metadata = {
            'project': self.project_id,
            'type': 'input',
            'step': step_label,
        }

        fp_files = []

        # insert these input files into data base
        for name, file_path in files.items():
            identifier = '/'.join((self.project_id, name))
            metadata['name'] = name
            fp_files.append(
                fp.add_file(
                    file_path,
                    identifier=identifier,
                    metadata=metadata))

        return fp_files

    def main(self, fws_root=[]):
        # try to get surfactant pdb file from kwargs
        try:
            surfactant = self.kwargs["system"]["surfactant"]["name"]
        except:
            surfactant = phys_config.DEFAULT_SURFACTANT
            warnings.warn("No surfactant specified, falling back to {:s}.".format(surfactant))

        lmp_coeff_input = file_config.LMP_COEFF_HYBRID_INPUT_PATTERN.format(name=surfactant)

        fw_list = []

        # query input files
        # -----------------
        step_label = self.get_step_label('input_files_pull')

        files_in = {}
        files_out = {
            'input_template': 'lmp.input.template',
            'mass_file':      'mass.input',
            'coeff_file':     'coeff.input',
            'eam_file':       'default.eam.alloy',
        }

        fts_pull = [
            GetFilesByQueryTask(
                query={
                    'metadata->project': self.project_id,
                    'metadata->name':    file_config.LMP_INPUT_TEMPLATE,
                },
                sort_key='metadata.datetime',
                sort_direction=pymongo.DESCENDING,
                limit=1,
                new_file_names=['lmp.input.template']),
            GetFilesByQueryTask(
                query={
                    'metadata->project': self.project_id,
                    'metadata->name':    lmp_coeff_input,
                },
                sort_key='metadata.datetime',
                sort_direction=pymongo.DESCENDING,
                limit=1,
                new_file_names=['coeff.input']),
            GetFilesByQueryTask(
                query={
                    'metadata->project': self.project_id,
                    'metadata->name':    file_config.LMP_MASS_INPUT,
                },
                sort_key='metadata.datetime',
                sort_direction=pymongo.DESCENDING,
                limit=1,
                new_file_names=['mass.input']),
            GetFilesByQueryTask(
                query={
                    'metadata->project': self.project_id,
                    'metadata->name':    file_config.LMP_EAM_ALLOY,
                },
                sort_key='metadata.datetime',
                sort_direction=pymongo.DESCENDING,
                limit=1,
                new_file_names=['default.eam.alloy']),
        ]

        fw_pull = self.build_fw(
            fts_pull, step_label,
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_noqueue_category'])

        fw_list.append(fw_pull)

        # fill input file template
        # -------------------
        step_label = self.get_step_label('fill_template')

        files_in = {
            'input_template': 'default.input.template',

        }
        files_out = {
            'input_file': 'default.input',
        }

        # Jinja2 context:
        static_template_context = {
            'mode':                    'production',
            'coeff_infile':            'coeff.input',
            'compute_group_properties': True,
            'data_file':                'datafile.lammps',
            # 'dilate_solution_only':     True,
            'has_indenter':             True,
            'mpiio':                    True,
            # 'pressurize_z_only':        True,
            'restrained_indenter':      False,
            'rigid_h_bonds':            True,
            'store_forces':             True,
            'temper_solid_only':        True,
            'use_barostat':             False,
            'use_dpd_tstat':            True,
            'use_eam':                  True,
            'use_ewald':                True,
            'write_coeff_to_datafile':  False,
        }

        dynamic_template_context = {
            'freeze_substrate_layer':     'metadata->step_specific->probe_normal_approach->freeze_substrate_layer',
            'rigid_indenter_core_radius': 'metadata->step_specific->probe_normal_approach->rigid_indenter_core_radius',
            'constant_indenter_velocity': 'metadata->step_specific->probe_normal_approach->constant_indenter_velocity',
            'temperatureT':               'metadata->step_specific->probe_normal_approach->temperature',

            'production_steps': 'metadata->step_specific->probe_normal_approach->steps',
            'netcdf_frequency': 'metadata->step_specific->probe_normal_approach->netcdf_frequency',
            'thermo_frequency': 'metadata->step_specific->probe_normal_approach->thermo_frequency',
            'thermo_average_frequency': 'metadata->step_specific->probe_normal_approach->thermo_average_frequency',
            'restart_frequency': 'metadata->step_specific->probe_normal_approach->restart_frequency',

            'ewald_accuracy':   'metadata->step_specific->probe_normal_approach->ewald_accuracy',
            'coulomb_cutoff':   'metadata->step_specific->probe_normal_approach->coulomb_cutoff',
            'neigh_delay':      'metadata->step_specific->probe_normal_approach->neigh_delay',
            'neigh_every':      'metadata->step_specific->probe_normal_approach->neigh_every',
            'neigh_check':      'metadata->step_specific->probe_normal_approach->neigh_check',
            'skin_distance':    'metadata->step_specific->probe_normal_approach->skin_distance',

            'substrate_element': 'metadata->system->substrate->element',
            'substrate_type': 'metadata->system->substrate->lmp->type',
        }

        fts_template = [TemplateWriterTask({
            'context_inputs': dynamic_template_context,
            'context': static_template_context,
            'template_file': 'default.input.template',
            'template_dir': '.',
            'output_file': 'default.input'})]

        fw_template = self.build_fw(
            fts_template, step_label,
            parents=[fw_pull, *fws_root],
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_noqueue_category'])

        fw_list.append(fw_template)

        # LAMMPS run
        # ----------
        step_label = self.get_step_label('lmp_run')

        files_in = {
            'data_file':  'datafile.lammps',
            'input_file': 'default.input',
            'mass_file':  'mass.input',
            'coeff_file': 'coeff.input',
            'eam_file':   'default.eam.alloy',
        }
        files_out = {
            'coeff_file':      'coeff.input',  # untouched
            'data_file':       'default.lammps',
            'eam_file':        'default.eam.alloy',  # untouched
            'index_file':      'groups.ndx',
            'conserved_input_file': 'default.input',  # untouched
            'log_file':        'log.lammps',
            'mass_file':       'mass.input',  # untouched
            'thermo_ave_file': 'thermo_ave.out',
            'trajectory_file': 'default.nc',
        }
        fts_lmp_run = [CmdTask(
            cmd='lmp',
            opt=['-in', 'default.input'],
            env='python',
            stderr_file='std.err',
            stdout_file='std.out',
            stdlog_file='std.log',
            store_stdout=True,
            store_stderr=True,
            fizzle_bad_rc=True)]

        fw_lmp_run = self.build_fw(
            fts_lmp_run, step_label,
            parents=[fw_template, fw_pull, *fws_root],
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_queue_category'],
            queueadapter=self.hpc_specs['four_nodes_job_queueadapter_defaults']
        )

        fw_list.append(fw_lmp_run)

        # lmp restart wf
        # --------------
        # --------------

        # restart query input files
        # -------------------------
        step_label = self.get_step_label('input_files_pull')

        files_in = {}
        files_out = {
            'input_template': 'lmp.input.template',
            'mass_file': 'mass.input',
            'coeff_file': 'coeff.input',
            'eam_file': 'default.eam.alloy',
        }

        fts_restart_pull = [
            GetFilesByQueryTask(
                query={
                    'metadata->project': self.project_id,
                    'metadata->name': file_config.LMP_INPUT_TEMPLATE,
                },
                sort_key='metadata.datetime',
                sort_direction=pymongo.DESCENDING,
                limit=1,
                new_file_names=['lmp.input.template']),
            GetFilesByQueryTask(
                query={
                    'metadata->project': self.project_id,
                    'metadata->name': lmp_coeff_input,
                },
                sort_key='metadata.datetime',
                sort_direction=pymongo.DESCENDING,
                limit=1,
                new_file_names=['coeff.input']),
            GetFilesByQueryTask(
                query={
                    'metadata->project': self.project_id,
                    'metadata->name': file_config.LMP_MASS_INPUT,
                },
                sort_key='metadata.datetime',
                sort_direction=pymongo.DESCENDING,
                limit=1,
                new_file_names=['mass.input']),
            GetFilesByQueryTask(
                query={
                    'metadata->project': self.project_id,
                    'metadata->name': file_config.LMP_EAM_ALLOY,
                },
                sort_key='metadata.datetime',
                sort_direction=pymongo.DESCENDING,
                limit=1,
                new_file_names=['default.eam.alloy']),
        ]

        fw_restart_pull = self.build_fw(
            fts_restart_pull, step_label,
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_noqueue_category'])

        # restart fill input file template
        # --------------------------------
        step_label = self.get_step_label('fill_template')
        fw_restart_template_name= self.get_fw_label(step_label)

        files_in = {
            'input_template': 'default.input.template',

        }
        files_out = {
            'input_file': 'default.input',
        }

        # Jinja2 context, as before with tiny modification:
        restart_static_template_context = static_template_context.copy()
        restart_static_template_context['is_restart'] = True

        # dynamic_template_context won't change
        restart_dynamic_template_context = dynamic_template_context.copy()

        fts_restart_template = [TemplateWriterTask({
            'context': restart_static_template_context,
            'context_inputs': restart_dynamic_template_context,
            'template_file': 'default.input.template',
            'template_dir': '.',
            'output_file': 'default.input'})]

        fw_restart_template = self.build_fw(
            fts_restart_template, step_label,
            parents=[fw_restart_pull],
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_noqueue_category'])

        # restart lmp run
        # ---------------

        step_label = self.get_step_label('lmp_run')
        fw_restart_lmp_run_name= self.get_fw_label(step_label)

        files_in = {
            'input_file':   'default.input',
            'mass_file':    'mass.input',  # obsolete
            'coeff_file':   'coeff.input',
            'eam_file':     'default.eam.alloy',  # obsolete
            'restart_file': 'default.mpiio.restart',
        }
        files_out = {
            'coeff_file':      'coeff.input',  # untouched
            'data_file':       'default.lammps',
            'eam_file':        'default.eam.alloy',  # untouched
            'index_file':      'groups.ndx',
            'conserved_input_file': 'default.input',  # untouched
            'log_file':        'log.lammps',
            'mass_file':       'mass.input',  # untouched
            'thermo_ave_file': 'thermo_ave.out',
            'trajectory_file': 'default.nc',
        }

        fts_lmp_run_restart = [CmdTask(
            cmd='lmp',
            opt=['-in', 'default.input'],
            env='python',
            stderr_file='std.err',
            stdout_file='std.out',
            stdlog_file='std.log',
            store_stdout=True,
            store_stderr=True,
            fizzle_bad_rc=True)]

        # as many spec as possible derived from fizzled parent
        fw_lmp_run_restart = self.build_fw(
            fts_lmp_run_restart, step_label,
            parents=[fw_restart_pull, fw_restart_template],
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_queue_category'],
            queueadapter=self.hpc_specs['four_nodes_job_queueadapter_defaults']
        )

        wf_restart = Workflow([fw_restart_pull, fw_restart_template, fw_lmp_run_restart])

        wf_restart_root_fw_ids = [
            fw.fw_id for fw in wf_restart.fws if fw.name in [fw_restart_lmp_run_name, fw_restart_template_name]]
        wf_restart_leaf_fw_ids = [
            fw.fw_id for fw in wf_restart.fws if fw.name in [fw_restart_lmp_run_name]]

        #----------------
        #----------------

        # analysis detour
        # ---------------

        step_label = self.get_step_label('lmp_analysis')

        files_in = {
            # from successfull postprocessing of failed lammps run:
            'trajectory_file': 'default.nc',
            'log_file':        'log.lammps',
            'thermo_ave_file': 'thermo_ave.out',

            # from successfull post processung run, forwarded via recovery fw:
            'joint_traj_file':   'previous.default.nc',
            'joint_thermo_file': 'previous.thermo.out',
            'joint_ave_file':    'previous.thermo_ave.out',
        }

        files_out = {
            'joint_traj_file':   'default.nc',
            'joint_thermo_file': 'thermo.out',
            'joint_ave_file':    'thermo_ave.out',
        }

        fts_detour = [
            CmdTask(  # extract thermo
                cmd="cat log.lammps | sed -n '/^Step/,/^Loop time/p' | sed '/^colvars:/d' | head -n-1 > thermo.out",
                fizzle_bad_rc=False,
                use_shell=True),
            CmdTask(  # concatenate previous and current thermo output
                cmd='join_thermo',
                opt=['-v', 'previous.thermo.out', 'thermo.out', 'joint.thermo.out'],
                env='python',
                stderr_file='join_thermo.err',
                stdout_file='join_thermo.out',
                stdlog_file='join_thermo.log',
                store_stdout=True,
                store_stderr=True,
                fizzle_bad_rc=False),
            CmdTask(  # concatenate previous and current thermo_ave output
                cmd='join_thermo',
                opt=['-v', '--hashed-header',
                     'previous.thermo_ave.out', 'thermo_ave.out', 'joint.thermo_ave.out'],
                env='python',
                stderr_file='join_thermo_ave.err',
                stdout_file='join_thermo_ave.out',
                stdlog_file='join_thermo_ave.log',
                store_stdout=True,
                store_stderr=True,
                fizzle_bad_rc=False),
            # fixed outfile name of ncjoin is traj.nc
            CmdTask(  # concatenate previous and current trajectory
                cmd='ncjoin',
                opt=['-v', 'time', '-f', 'traj.nc', '-x', '--',
                     'previous.default.nc', 'default.nc'],
                env='python',
                stderr_file='ncjoin.err',
                stdout_file='ncjoin.out',
                stdlog_file='ncjoin.log',
                store_stdout=True,
                store_stderr=True,
                fizzle_bad_rc=False),
            # if no previous files provided via _files_in, just forward current files
            # by letting the following file copy task fail deliberately
            FileTransferTask(
                mode='copy',
                ignore_errors=True,
                files=[
                    {'src':  'joint.thermo.out',
                     'dest': 'thermo.out'},
                    {'src':  'joint.thermo_ave.out',
                     'dest': 'thermo_ave.out'},
                    {'src':  'traj.nc',
                     'dest': 'default.nc'}
                ]
            )
        ]

        fw_detour = self.build_fw(
            fts_detour, step_label,
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_queue_category'],
            queueadapter=self.hpc_specs['single_core_job_queueadapter_defaults']
        )

        wf_detour = Workflow([fw_detour])

        #---------
        #---------

        # recovery
        # --------

        step_label = self.get_step_label('lmp_recovery')

        files_in = {
            # from successfull LAMMPS run
            'coeff_file': 'coeff.input',  # untouched
            'data_file': 'default.lammps',
            'eam_file': 'default.eam.alloy',  # untouched
            'index_file': 'groups.ndx',
            'conserved_input_file': 'default.input',  # untouched
            'log_file': 'log.lammps',
            'mass_file': 'mass.input',  # untouched
            'thermo_ave_file': 'thermo_ave.out',
            'trajectory_file': 'default.nc',

            # from successfull postprocessing of failed lammps run:
            'joint_traj_file': 'joint.default.nc',
            'joint_thermo_file': 'joint.thermo.out',
            'joint_ave_file': 'joint.thermo_ave.out',
        }
        files_out = {
            # either forwarded from successfull or recovered from failed lammps run:
            'coeff_file': 'coeff.input',  # untouched
            'data_file': 'default.lammps',
            'eam_file': 'default.eam.alloy',  # untouched
            'index_file': 'groups.ndx',
            'conserved_input_file': 'default.input',  # untouched
            'log_file': 'log.lammps',
            'mass_file': 'mass.input',  # untouched
            'thermo_ave_file': 'thermo_ave.out',
            'trajectory_file': 'default.nc',

            # from successfull postprocessing of failed lammps run:
            'joint_traj_file': 'joint.default.nc',
            'joint_thermo_file': 'joint.thermo.out',
            'joint_ave_file': 'joint.thermo_ave.out',

            # recovered from failed lammps run
            'restart_file': 'default.mpiio.restart',
        }

        fts_lmp_recovery = [RecoverTask(
            restart_wf=wf_restart.as_dict(),
            restart_fws_root=wf_restart_root_fw_ids,
            restart_fws_leaf=wf_restart_leaf_fw_ids,
            detour_wf=wf_detour.as_dict(),
            superpose_restart_on_parent_fw_spec=True,
            superpose_detour_on_parent_fw_spec=True,
            repeated_recover_fw_name=step_label,
            max_restarts=100,  # TODO: from global kwargs or fwspec
            fizzle_on_no_restart_file=False,
            restart_file_glob_patterns="*.restart[0-9]",
            restart_file_dests = 'default.mpiio.restart',
            other_glob_patterns=[
                "default.input",
                "default.nc",
                "log.lammps",
                "thermo_ave.out",
            ],
            restart_counter='metadata->step_specific->lmp_run->restart_count',
            store_stdlog=True,
            stdlog_file='std.log',
            loglevel=logging.DEBUG)]

        fw_lmp_recovery = self.build_fw(
            fts_lmp_recovery, step_label,
            parents=[fw_lmp_run],
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_noqueue_category'],
            fw_spec={'_allow_fizzled_parents': True})

        fw_list.append(fw_lmp_recovery)

        return fw_list, [fw_lmp_recovery], [fw_lmp_run, fw_template]


# class LAMMPSProbeNormalApproach(
#         DefaultPullMixin, DefaultPushMixin,
#         ProcessAnalyzeAndVisualize,
#         ):
#     def __init__(self, *args, **kwargs):
#         super().__init__(
#             main_sub_wf=LAMMPSProbeNormalApproachMain,
#             analysis_sub_wf=LAMMPSTrajectoryAnalysis,
#             *args, **kwargs)

class LAMMPSProbeNormalApproach(
        DefaultPullMixin, DefaultPushMixin,
        LAMMPSRecoverableProbeNormalApproachMain,
        ):
    pass