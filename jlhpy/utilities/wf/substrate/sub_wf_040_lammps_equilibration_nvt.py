# -*- coding: utf-8 -*-
"""Substrate NVT equilibration sub workflow."""

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


class LAMMPSEquilibrationNVTMain(WorkflowGenerator):
    """
    NVT equilibration with LAMMPS.

    inputs:
    - metadata->step_specific->equilibration->nvt->temperature
    - metadata->step_specific->equilibration->nvt->langevin_damping

    - metadata->step_specific->equilibration->nvt->steps
    - metadata->step_specific->equilibration->nvt->netcdf_frequency
    - metadata->step_specific->equilibration->nvt->thermo_frequency
    - metadata->step_specific->equilibration->nvt->thermo_average_frequency

    - metadata->step_specific->equilibration->nvt->neigh_delay
    - metadata->step_specific->equilibration->nvt->neigh_every
    - metadata->step_specific->equilibration->nvt->neigh_check
    - metadata->step_specific->equilibration->nvt->skin_distance

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
            'initialT':                 0.0,
            'mpiio':                    True,
            'reinitialize_velocities':  True,
            'rigid_h_bonds':            False,
            'store_forces':             False,
            'temper_solid_only':        False,
            'use_barostat':             False,
            'use_eam':                  True,
            'use_ewald':                False,
            'write_coeff_to_datafile':  False,
        }

        dynamic_template_context = {
            'temperatureT':     'metadata->step_specific->equilibration->nvt->temperature',
            'langevin_damping': 'metadata->step_specific->equilibration->nvt->langevin_damping',
            'production_steps': 'metadata->step_specific->equilibration->nvt->steps',
            'netcdf_frequency': 'metadata->step_specific->equilibration->nvt->netcdf_frequency',
            'thermo_frequency': 'metadata->step_specific->equilibration->nvt->thermo_frequency',
            'thermo_average_frequency': 'metadata->step_specific->equilibration->nvt->thermo_average_frequency',
            'neigh_delay':      'metadata->step_specific->equilibration->nvt->neigh_delay',
            'neigh_every':      'metadata->step_specific->equilibration->nvt->neigh_every',
            'neigh_check':      'metadata->step_specific->equilibration->nvt->neigh_check',
            'skin_distance':    'metadata->step_specific->equilibration->nvt->skin_distance',

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


class LAMMPSEquilibrationNVTWorkflowGenerator(
        DefaultPullMixin, DefaultPushMixin,
        ProcessAnalyzeAndVisualizeWorkflowGenerator,
        ):
    def __init__(self, *args, **kwargs):
        ProcessAnalyzeAndVisualizeWorkflowGenerator.__init__(self,
            main_sub_wf=LAMMPSEquilibrationNVTMain(*args, **kwargs),
            analysis_sub_wf=LAMMPSSubstrateTrajectoryAnalysisWorkflowGenerator(*args, **kwargs),
            *args, **kwargs)
