# -*- coding: utf-8 -*-
"""Substrate NPT equilibration sub workflow."""

import datetime
import glob
import os
import pymongo
import warnings

from fireworks import Firework
from fireworks.user_objects.firetasks.filepad_tasks import GetFilesByQueryTask
from fireworks.user_objects.firetasks.filepad_tasks import AddFilesTask
from fireworks.user_objects.firetasks.templatewriter_task import TemplateWriterTask
from imteksimfw.fireworks.user_objects.firetasks.cmd_tasks import CmdTask

from jlhpy.utilities.wf.workflow_generator import (
    WorkflowGenerator, ProcessAnalyzeAndVisualizeWorkflowGenerator)
from jlhpy.utilities.wf.mixin.mixin_wf_storage import (
   DefaultPullMixin, DefaultPushMixin)

from jlhpy.utilities.wf.building_blocks.sub_wf_lammps_analysis import LAMMPSSubstrateTrajectoryAnalysisWorkflowGenerator

import jlhpy.utilities.wf.file_config as file_config
import jlhpy.utilities.wf.phys_config as phys_config


class LAMMPSEquilibrationNPTMain(WorkflowGenerator):
    """
    LAMMPS run building block.

    The following placeholders are available within lmp_input/template/lmp.input.template.
    For their defaults and types, see template file.

        add_vacuum
        barostat_damping
        base_name
        bonded_coeff_infile
        box_shift_z
        coeff_infile
        coeff_outfile
        colvars_file
        compute_group_properties
        compute_interactions
        constant_indenter_velocity
        coulomb_cutoff
        counterion_type
        data_file
        dilate_solution_only
        direction_of_linear_movement
        dpd_cutoff
        dpd_damping_parameter
        eam_alloy_file
        ewald_accuracy
        exclude_frozen_interactions
        freeze_substrate
        freeze_substrate_layer
        has_indenter
        has_vacuum
        indenter_height
        indenter_nve_noforce
        indenter_substrate_dist
        inner_cutoff
        is_restart
        langevin_damping
        manual_indenter_region
        minimization_ftol
        minimization_maxeval
        minimization_maxiter
        mode
        mpiio
        ndx_file
        neigh_check
        neigh_delay
        neigh_every
        neigh_one
        neigh_page
        netcdf_frequency
        outer_cutoff
        pair_coeff_infile
        pbc2d
        png_frequency
        pressureP
        pressurize_solution_only
        pressurize_z_only
        production_steps
        random_seed
        read_datafile
        read_groups_from_file
        region_tolerance
        reinitialize_velocities
        relax_box
        remove_drift
        restart_frequency
        restrained_indenter
        restrain_substrate_layer
        rigid_h_bonds
        rigid_indenter
        rigid_indenter_core_radius
        robust_minimization
        shrink_wrap_once
        skin_distance
        solvent_angle
        solvent_types
        store_forces
        substrate_element
        substrate_recenter
        substrate_thickness
        substrate_type
        temperatureT
        temper_solid_only
        temper_substrate_only
        thermo_average_frequency
        thermo_frequency
        thermo_to_netcdf
        timestep
        use_barostat
        use_berendsen_bstat
        use_colvars
        use_dpd_tstat
        use_eam
        use_ewald
        verbose
        write_coeff
        write_coeff_to_datafile
        write_groups_to_file


    dynamic infiles:
    - data_file:       default.lammps

    static infiles:
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

        lmp_coeff_input = file_config.LMP_COEFF_HYBRID_NONEWALD_NONBONDED_INPUT_PATTERN.format(name=surfactant)

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

        lmp_coeff_input = file_config.LMP_COEFF_HYBRID_NONEWALD_NONBONDED_INPUT_PATTERN.format(name=surfactant)

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
            'dilate_solution_only':     False,
            'mpiio':                    True,
            'reinitialize_velocities':  False,
            'rigid_h_bonds':            False,
            'store_forces':             False,
            'temper_solid_only':        False,
            'use_barostat':             True,
            'use_eam':                  True,
            'use_ewald':                False,
            'write_coeff_to_datafile':  False,
        }

        dynamic_template_context = {
            'pressureP':        'metadata->step_specific->equilibration->npt->pressure',
            'temperatureT':     'metadata->step_specific->equilibration->npt->temperature',
            'langevin_damping': 'metadata->step_specific->equilibration->npt->langevin_damping',
            'production_steps': 'metadata->step_specific->equilibration->npt->steps',
            'netcdf_frequency': 'metadata->step_specific->equilibration->npt->netcdf_frequency',
            'thermo_frequency': 'metadata->step_specific->equilibration->npt->thermo_frequency',
            'thermo_average_frequency': 'metadata->step_specific->equilibration->npt->thermo_average_frequency',
            'neigh_delay':      'metadata->step_specific->equilibration->npt->neigh_delay',
            'neigh_every':      'metadata->step_specific->equilibration->npt->neigh_every',
            'neigh_check':      'metadata->step_specific->equilibration->npt->neigh_check',
            'skin_distance':    'metadata->step_specific->equilibration->npt->skin_distance',

            'substrate_element': 'metadata->system->substrate->element',
            'substrate_type': 'metadata->system->substrate->lmp->type',
        }

        fts_template = [TemplateWriterTask({
            'context_inputs': dynamic_template_context,
            'context': static_template_context,
            'template_file': 'default.input.template',
            'template_dir': '.',
            'output_file': 'default.input'})]

        fw_template = Firework(fts_template,
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
            parents=[fw_pull, *fws_root])

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
            'input_file':      'default.input',  # untouched
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

        fw_lmp_run = Firework(fts_lmp_run,
            name=self.get_fw_label(step_label),
            spec={
                '_category': self.hpc_specs['fw_queue_category'],
                '_queueadapter': {
                    **self.hpc_specs['single_node_job_queueadapter_defaults'],  # get 1 node
                },
                '_files_in':  files_in,
                '_files_out': files_out,
                'metadata': {
                    'project': self.project_id,
                    'datetime': str(datetime.datetime.now()),
                    'step':    step_label,
                    **self.kwargs
                }
            },
            parents=[fw_template, fw_pull, *fws_root])

        fw_list.append(fw_lmp_run)

        return fw_list, [fw_lmp_run], [fw_lmp_run, fw_template]


class LAMMPSEquilibrationNPTWorkflowGenerator(
        DefaultPullMixin, DefaultPushMixin,
        ProcessAnalyzeAndVisualizeWorkflowGenerator,
        ):
    def __init__(self, *args, **kwargs):
        ProcessAnalyzeAndVisualizeWorkflowGenerator.__init__(self,
            main_sub_wf=LAMMPSEquilibrationNPTMain(*args, **kwargs),
            analysis_sub_wf=LAMMPSSubstrateTrajectoryAnalysisWorkflowGenerator(*args, **kwargs),
            *args, **kwargs)
