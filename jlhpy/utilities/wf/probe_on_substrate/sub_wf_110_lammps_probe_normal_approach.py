# -*- coding: utf-8 -*-
"""Substrate NVT equilibration with DPD thermostat sub workflow."""

import datetime
import glob
import os
import pymongo
import warnings

from fireworks import Firework
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


class LAMMPSProbeNormalApproachMain(WorkflowGenerator):
    """
    Probe aaproach production.

    inputs:
    - metadata->step_specific->probe_normal_approach->freeze_substrate_layer',
    - metadata->step_specific->probe_normal_approach->rigid_indenter_core_radius',
    - metadata->step_specific->probe_normal_approach->constant_indenter_velocity',
    - metadata->step_specific->probe_normal_approach->temperature',

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
 NPT relaxation with GROMACS without restraints on ions.

    dynamic infiles:
        only queried in pull stub, otherwise expected through data flow

    - data_file:       default.gro
        tagged as {'metadata->type': 'npt_gro'}
    - index_file:      default.ndx
        tagged as {'metadata->type': 'npt_ndx'}
    - topology_file: default.top
        queried by { 'metadata->type': 'solvate_top' }

    static infiles:
        always queried within main trunk

    - parameter_file: default.mdp,
        queried by {'metadata->name': file_config.GMX_RELAX_Z_ONLY_MDP}

    outfiles:
    - log_file:        default.log
        tagged as {'metadata->type': 'relax_log'}
    - energy_file:     default.edr
        tagged as {'metadata->type': 'relax_edr'}
    - trajectory_file: default.xtc
        tagged as {'metadata->type': 'relax_xtc'}
    - data_file:       default.gro
        tagged as {'metadata->type': 'relax_gro'}

    - index_file:      default.ndx
        pass through untouched
    - topology_file:   default.top
        pass through untouched
    """
    def push_infiles(self, fp):

        # static infiles for main
        # -----------------------
        step_label = self.get_step_label('push_infiles')

        infiles = sorted(glob.glob(os.path.join(
            self.infile_prefix,
            file_config.GMX_MDP_SUBDIR,
            file_config.GMX_RELAX_Z_ONLY_MDP)))

        files = {os.path.basename(f): f for f in infiles}

        # metadata common to all these files
        metadata = {
            'project': self.project_id,
            'type': 'input',
            'name': file_config.GMX_RELAX_Z_ONLY_MDP,
            'step': step_label,
        }

        fp_files = []

        # insert these input files into data base
        for name, file_path in files.items():
            identifier = '/'.join((self.project_id, name))
            fp_files.append(
                fp.add_file(
                    file_path,
                    identifier=identifier,
                    metadata=metadata))

        return fp_files

    def main(self, fws_root=[]):
        fw_list = []

        # query input files
        # -----------------
        step_label = self.get_step_label('input_files_pull')

        files_in = {}
        files_out = {'input_file': 'default.mdp'}

        fts_pull_mdp = [
            GetFilesByQueryTask(
                query={
                    'metadata->project': self.project_id,
                    'metadata->name':    file_config.GMX_RELAX_Z_ONLY_MDP,
                },
                sort_key='metadata.datetime',
                sort_direction=pymongo.DESCENDING,
                limit=1,
                new_file_names=['default.mdp'])]

        fw_pull_mdp = self.build_fw(
            fts_pull_mdp, step_label,
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_noqueue_category'])

        fw_list.append(fw_pull_mdp)

        # GMX grompp
        # ----------
        step_label = self.get_step_label('gmx_grompp')

        files_in = {
            'index_file':      'default.ndx',
            'input_file':      'default.mdp',
            'data_file':       'default.gro',
            'topology_file':   'default.top',
        }
        files_out = {
            'input_file':     'default.tpr',
            'parameter_file': 'mdout.mdp',
            'topology_file':  'default.top',  # pass through untouched
            'index_file':     'default.ndx',  # pass through untouched
        }

        # gmx grompp -f nvt.mdp -n nvt.ndx -c em_solvated.gro -r em_solvated.gro -o nvt.tpr -p sys.top
        fts_gmx_grompp = [CmdTask(
            cmd='gmx',
            opt=['grompp',
                 '-f', 'default.mdp',
                 '-n', 'default.ndx',
                 '-c', 'default.gro',
                 '-r', 'default.gro',
                 '-o', 'default.tpr',
                 '-p', 'default.top',
                 '-maxwarn', 2,
                ],
            env='python',
            stderr_file='std.err',
            stdout_file='std.out',
            stdlog_file='std.log',
            store_stdout=True,
            store_stderr=True,
            fizzle_bad_rc=True)]
        # -maxwarn 2 allows for the following two warnings:
        #
        # WARNING 1 [file default.mdp]:
        #   Some atoms are not part of any center of mass motion removal group.
        #   This may lead to artifacts.
        #   In most cases one should use one group for the whole system.
        #
        # WARNING 2 [file default.mdp]:
        #   You are using pressure coupling with absolute position restraints, this
        #   will give artifacts. Use the refcoord_scaling option.

        fw_gmx_grompp = self.build_fw(
            fts_gmx_grompp, step_label,
            parents=[*fws_root, fw_pull_mdp],
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_queue_category'],
            queueadapter=self.hpc_specs['quick_single_core_job_queueadapter_defaults'])

        fw_list.append(fw_gmx_grompp)

        # GMX mdrun
        # ---------
        step_label = self.get_step_label('gmx_mdrun')

        files_in = {
            'input_file':    'default.tpr',
            'topology_file': 'default.top',  # pass through untouched
            'index_file':    'default.ndx',  # pass through untouched
        }
        files_out = {
            'log_file':        'default.log',
            'energy_file':     'default.edr',
            'trajectory_file': 'default.xtc',
            'data_file':       'default.gro',
            'run_file':        'default.tpr',  # passed throught unmodified
            'topology_file':   'default.top',  # pass through untouched
            'index_file':      'default.ndx',  # pass through untouched
        }

        fts_gmx_mdrun = [CmdTask(
            cmd='gmx',
            opt=['mdrun',
                 '-deffnm', 'default', '-v'],
            env='python',
            stderr_file='std.err',
            stdout_file='std.out',
            stdlog_file='std.log',
            store_stdout=True,
            store_stderr=True,
            fizzle_bad_rc=True)]

        fw_gmx_mdrun = self.build_fw(
            fts_gmx_mdrun, step_label,
            parents=[fw_gmx_grompp],
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_queue_category'],
            queueadapter=self.hpc_specs['four_nodes_job_queueadapter_defaults'])

        fw_list.append(fw_gmx_mdrun)

        # GMX mdrun restart
        # -----------------
        files_in = {
            'checkpoint_file': 'default.cpt',
            'log_file':        'default.log',
            'energy_file':     'default.edr',
            'uncompressed_trajectory_file': 'default.trr',
            'trajectory_file': 'default.xtc',
            'data_file':       'default.gro',
            'run_file':        'default.tpr',
            'topology_file':   'default.top',
            'index_file':      'default.ndx',
        }
        files_out = {
            'log_file':        'default.log',
            'energy_file':     'default.edr',
            'trajectory_file': 'default.xtc',
            'data_file':       'default.gro',
            'run_file':        'default.tpr',  # passed throught unmodified
            'topology_file':   'default.top',  # pass through untouched
            'index_file':      'default.ndx',  # pass through untouched
        }

        step_label = self.get_step_label('gmx_mdrun')

        fts_gmx_mdrun_restart = [CmdTask(
            cmd='gmx',
            opt=['mdrun', '-v', '-deffnm', 'default', '-cpi', 'default'],
            env='python',
            stderr_file='std.err',
            stdout_file='std.out',
            stdlog_file='std.log',
            store_stdout=True,
            store_stderr=True,
            fizzle_bad_rc=True)]

        # as many spec as possible derived from fizzled parent
        fw_gmx_mdrun_restart = Firework(fts_gmx_mdrun_restart,
                                name=self.get_fw_label(step_label),
                                spec={
                                    '_files_in':  files_in,
                                    '_files_out': files_out,
                                })

        restart_wf = Workflow([fw_gmx_mdrun_restart])

        # recovery
        # --------

        files_in = {
            'log_file':        'default.log',
            'energy_file':     'default.edr',
            'trajectory_file': 'default.xtc',
            # 'uncompressed_trajectory_file': 'default.trr',
            # by not specifying the trr as an input, it is effectively filtered out
            'data_file':       'default.gro',
            'run_file':        'default.tpr',  # passed throught unmodified
            'topology_file':   'default.top',
            'index_file':      'default.ndx',
        }

        files_out = {
            'checkpoint_file': 'default.cpt',
            'log_file':        'default.log',
            'energy_file':     'default.edr',
            'uncompressed_trajectory_file': 'default.trr',   # restart needs trr, otherwise fails
            'trajectory_file': 'default.xtc',
            'data_file':       'default.gro',
            'topology_file':   'default.top',
            'run_file':        'default.tpr',
            'index_file':      'default.ndx',
        }

        step_label = self.get_step_label('gmx_recovery')

        fts_gmx_recovery = [RecoverTask(
            restart_wf=restart_wf.as_dict(),
            superpose_restart_on_parent_fw_spec=True,
            repeated_recover_fw_name=step_label,
            max_restarts=20,
            fizzle_on_no_restart_file=False,
            restart_file_glob_patterns='default.cpt',
            other_glob_patterns=[
                "default.gro",
                "default.edr",
                "default.ndx",
                "default.tpr",
                "default.trr",
                "default.xtc",
                "default.log",
                "default.top",
            ],
            restart_counter='metadata->step_specific->gmx_relaxation->restart_count',
            store_stdlog=True,
            stdlog_file='std.log',
            loglevel=logging.DEBUG)]

        fw_gmx_recovery = self.build_fw(
            fts_gmx_recovery, step_label,
            parents=[fw_gmx_mdrun],
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_noqueue_category'],
            fw_spec={'_allow_fizzled_parents': True})
        fw_list.append(fw_gmx_recovery)

        return fw_list, [fw_gmx_recovery], [fw_gmx_grompp]


class LAMMPSProbeNormalApproch(
        DefaultPullMixin, DefaultPushMixin,
        ProcessAnalyzeAndVisualize,
        ):
    def __init__(self, *args, **kwargs):
        super().__init__(
            main_sub_wf=LAMMPSProbeNormalApproachMain,
            analysis_sub_wf=LAMMPSTrajectoryAnalysis,
            *args, **kwargs)





