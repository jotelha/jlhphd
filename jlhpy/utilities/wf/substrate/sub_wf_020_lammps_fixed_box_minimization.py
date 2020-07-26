# -*- coding: utf-8 -*-
"""Indenter bounding sphere sub workflow."""

import datetime
import glob
import os
import pymongo

from fireworks import Firework
from fireworks.user_objects.firetasks.filepad_tasks import GetFilesByQueryTask
from fireworks.user_objects.firetasks.filepad_tasks import AddFilesTask
from fireworks.user_objects.firetasks.templatewriter_task import TemplateWriterTask
from imteksimfw.fireworks.user_objects.firetasks.cmd_tasks import CmdTask

from jlhpy.utilities.wf.workflow_generator import (
    SubWorkflowGenerator, ProcessAnalyzeAndVisualizeSubWorkflowGenerator)
from jlhpy.utilities.wf.mixin.mixin_wf_storage import (
   DefaultPullMixin, DefaultPushMixin)

# from jlhpy.utilities.wf.building_blocks.sub_wf_gromacs_analysis import GromacsVacuumTrajectoryAnalysisSubWorkflowGenerator
# from jlhpy.utilities.wf.building_blocks.sub_wf_gromacs_vis import GromacsTrajectoryVisualizationSubWorkflowGenerator

import jlhpy.utilities.wf.file_config as file_config


class LAMMPSFixedBoxMinimizationMain(SubWorkflowGenerator):
    """
    Fixed box minimization with LAMMPS.

    dynamic infiles:
        only queried in pull stub, otherwise expected through data flow

    - data_file:       default.lammps
        tagged as {'metadata->type': 'initial_config'}

    static infiles:
        always queried within main trunk

    - input_header_file:      header.input
        tagged as {'metadata->type': 'em_input_header'}
    - input_body_file:        body.input
        tagged as {'metadata->type': 'em_input_body'}
    - mass_file: mass.input
        queried by { 'metadata->type': 'mass_input' }
    - coeff_file: coeff.input
        queried by { 'metadata->type': 'coeff_input' }

    vis static infiles:
    - script_file: renumber_png.sh,
        queried by {'metadata->name': file_config.BASH_RENUMBER_PNG}
    - template_file: default.pml.template,
        queried by {'metadata->name': file_config.PML_MOVIE_TEMPLATE}

    outfiles:
    - log_file:        log.lammps
        tagged as {'metadata->type': 'em_log'}
    - trajectory_file: default.nc
        tagged as {'metadata->type': 'em_nc'}
    - data_file:       default.gro
        tagged as {'metadata->type': 'em_lammps'}

    - index_file:      default.ndx
        pass through untouched
    - topology_file:   default.top
        pass through untouched

    vis outfiles:
    - mp4_file: default.mp4
        tagged as {'metadata->type': 'mp4_file'}
    """

    def __init__(self, *args, **kwargs):
        sub_wf_name = 'LAMMPSFixedBoxMinimizationMain'
        if 'wf_name_prefix' not in kwargs:
            kwargs['wf_name_prefix'] = sub_wf_name
        else:
            kwargs['wf_name_prefix'] = ':'.join((kwargs['wf_name_prefix'], sub_wf_name))
        super().__init__(*args, **kwargs)

    def push_infiles(self, fp):

        # static infiles for main
        # -----------------------
        step_label = self.get_step_label('push_infiles')

        glob_patterns = [
            os.path.join(
                self.infile_prefix,
                file_config.LMP_INPUT_STEMPLATE_UBDIR,
                file_config.LMP_HEADER_INPUT_TEMPLATE),
            os.path.join(
                self.infile_prefix,
                file_config.LMP_INPUT_SUBDIR,
                file_config.LMP_MINIMIZATION),
            os.path.join(
                self.infile_prefix,
                file_config.LMP_INPUT_SUBDIR,
                file_config.LMP_COEFF_INPUT),
            os.path.join(
                self.infile_prefix,
                file_config.LMP_INPUT_SUBDIR,
                file_config.LMP_MASS_INPUT)
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
        fw_list = []

        # query input files
        # -----------------
        step_label = self.get_step_label('input_files_pull')

        files_in = {}
        files_out = {
            'input_header_template': 'lmp_header.input.template',
            'input_body_template':   'lmp_minimization.input.template',
            'mass_file':             'mass.input',
            'coeff_file':            'coeff.input',
        }

        fts_pull = [
            GetFilesByQueryTask(
                query={
                    'metadata->project': self.project_id,
                    'metadata->name':    file_config.LMP_HEADER_INPUT_TEMPLATE,
                },
                sort_key='metadata.datetime',
                sort_direction=pymongo.DESCENDING,
                limit=1,
                new_file_names=['header.input']),
            GetFilesByQueryTask(
                query={
                    'metadata->project': self.project_id,
                    'metadata->name':    file_config.LMP_MINIMIZATION_INPUT_TEMPLATE,
                },
                sort_key='metadata.datetime',
                sort_direction=pymongo.DESCENDING,
                limit=1,
                new_file_names=['body.input']),
            GetFilesByQueryTask(
                query={
                    'metadata->project': self.project_id,
                    'metadata->name':    file_config.LMP_COEFF_INPUT,
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
                new_file_names=['mass.input'])]

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

        # input file template
        # -------------------
        step_label = self.get_step_label('fill_template')

        files_in = {
            'input_header_template': 'lmp_header.input.template',
            'input_body_template':   'default.input.template',

        }
        files_out = {
            'input_file': 'default.input',
        }

        # Jinja2 context:
        static_template_context = {
            'coeff_infile':            'coeff.input',
            'compute_interactions':     False,
            'data_file':                'datafile.lammps',
            'freeze_substrate':         False,
            'has_indenter':             False,
            'has_vacuum':               False,
            'mpiio':                    False,
            'pbc2d':                    False,
            'relax_box':                False,
            'rigid_h_bonds':            False,
            'robust_minimization':      False,
            'shrink_wrap_once':         False,
            'store_forces':             False,
            'use_eam':                  True,
            'use_ewald':                False,
            'write_coeff_to_datafile':  False,
        }

        dynamic_template_context = {
            'minimization_ftol': 'metadata->step_specific->minimizeation->ftol',
            'minimization_maxiter': 'metadata->step_specific->minimizeation->maxiter',
            'minimization_maxeval': 'metadata->step_specific->minimizeation->maxeval',
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
            parents=[fw_pull])

        fw_list.append(fw_template)

        # LAMMPS run
        # ----------
        step_label = self.get_step_label('lmp_run')

        files_in = {
            'data_file':  'datafile.lammps',
            'input_file': 'default.input',
            'mass_file':  'mass.input',
            'coeff_file': 'coeff.input',
        }
        files_out = {
            'data_file':  'default.lammps',
            'input_file': 'default.input',  # untouched
            'mass_file':  'mass.input',  # untouched
            'coeff_file': 'coeff.input',  # untouched
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

        return fw_list, [fw_lmp_run], [fw_lmp_run]


class LAMMPSFixedBoxMinimizationSubWorkflowGenerator(
        DefaultPullMixin, DefaultPushMixin,
        ):
    def __init__(self, *args, **kwargs):
        sub_wf_name = 'LAMMPSFixedBoxMinimization'
        if 'wf_name_prefix' not in kwargs:
            kwargs['wf_name_prefix'] = sub_wf_name
        else:
            kwargs['wf_name_prefix'] = ':'.join((kwargs['wf_name_prefix'], sub_wf_name))
        super().__init__(*args, **kwargs)
