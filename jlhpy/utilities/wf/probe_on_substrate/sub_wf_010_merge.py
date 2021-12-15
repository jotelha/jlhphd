# -*- coding: utf-8 -*-
import glob
import os
import pymongo

from fireworks.user_objects.firetasks.filepad_tasks import GetFilesByQueryTask
from fireworks.user_objects.firetasks.templatewriter_task import TemplateWriterTask
from imteksimfw.fireworks.user_objects.firetasks.cmd_tasks import CmdTask

from jlhpy.utilities.wf.workflow_generator import (
    WorkflowGenerator, ProcessAnalyzeAndVisualize)
from jlhpy.utilities.wf.mixin.mixin_wf_storage import (
   DefaultPullMixin, DefaultPushMixin)

import jlhpy.utilities.wf.file_config as file_config


class MergeSubstrateAndProbeSystemsMain(WorkflowGenerator):
    """
    Merge substrate system and probe system.

    dynamic infiles:
        only queried in pull stub, otherwise expected through data flow

    - probe_data_file:     probe.gro
    - substrate_data_file: substrate.gro

    static infiles:
        always queried within main trunk

    - vmd_input_file_template: default.tcl.template,
        queried by {'metadata->name': file_config.VMD_MERGE_TEMPLATE}

    outfiles:
    - data_file:       default.pdb
    """
    def push_infiles(self, fp):

        # static infiles for main
        # -----------------------
        step_label = self.get_step_label('push_infiles')

        infiles = sorted(glob.glob(os.path.join(
            self.infile_prefix,
            file_config.VMD_SUBDIR,
            file_config.VMD_MERGE_TEMPLATE)))

        files = {os.path.basename(f): f for f in infiles}

        # metadata common to all these files
        metadata = {
            'project': self.project_id,
            'type': 'input',
            'name': file_config.VMD_MERGE_TEMPLATE,
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
        files_out = {'template_file': 'default.tcl.template'}

        fts_pull_template = [
            GetFilesByQueryTask(
                query={
                    'metadata->project': self.project_id,
                    'metadata->name':    file_config.VMD_MERGE_TEMPLATE,
                },
                sort_key='metadata.datetime',
                sort_direction=pymongo.DESCENDING,
                limit=1,
                new_file_names=['default.tcl.template'])]

        fw_pull_template = self.build_fw(
            fts_pull_template, step_label,
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_noqueue_category'])

        fw_list.append(fw_pull_template)

        # VMD input script template
        # -----------------------------
        step_label = self.get_step_label('vmd_merge_template')

        files_in = {'template_file': 'default.tcl.template'}
        files_out = {'input_file': 'default.tcl'}

        # Jinja2 context:
        static_context = {
            'tol': 2.0,
            'substrate_gro': 'substrate.gro',
            'particle_gro': 'probe.gro',
            'out_pdb': 'default.pdb'
        }

        dynamic_context = {
            'counterion': 'metadata->system->counterion->resname',
            'solvent': 'metadata->system->solvent->resname',
            'substrate': 'metadata->system->substrate->resname',
            'surfactant': 'metadata->system->surfactant->resname',

            'tol': 'metadata->step_specific->merge->tol',
            'z_dist': 'metadata->step_specific->merge->z_dist',
            'x_shift': 'metadata->step_specific->merge->x_shift',
            'y_shift': 'metadata->step_specific->merge->y_shift',
        }

        fts_template = [
            TemplateWriterTask({
                'context': static_context,
                'context_inputs': dynamic_context,
                'template_file': 'default.tcl.template',
                'template_dir': '.',
                'output_file': 'default.tcl'})
        ]

        fw_template = self.build_fw(
            fts_template, step_label,
            parents=[fw_pull_template],
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_noqueue_category'])

        fw_list.append(fw_template)

        # VMD run
        # -------
        step_label = self.get_step_label('vmd_run')

        files_in = {
            'input_file':          'default.tcl',
            'substrate_data_file': 'substrate.gro',
            'probe_data_file':     'probe.gro',
        }
        files_out = {
            'input_file':      'default.tcl',
            'data_file':       'default.pdb',
        }

        fts_vmd_run = [CmdTask(
            cmd='vmd',
            opt=['-eofexit', '-e', 'default.tcl'],
            env='python',
            stderr_file='std.err',
            stdout_file='std.out',
            stdlog_file='std.log',
            store_stdout=True,
            store_stderr=True,
            fizzle_bad_rc=True)]

        fw_vmd_run = self.build_fw(
            fts_vmd_run, step_label,
            parents=[fw_template,*fws_root],
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_queue_category'],
            queueadapter=self.hpc_specs['single_core_job_queueadapter_defaults'])

        fw_list.append(fw_vmd_run)

        # # PDB chain
        # # ---------
        # step_label = self.get_step_label('pdb_chain')
        #
        # files_in = {'data_file': 'in.pdb'}
        # files_out = {'data_file': 'out.pdb'}
        #
        # fts_pdb_chain = [CmdTask(
        #     cmd='pdb_chain',
        #     env='python',
        #     stdin_file='in.pdb',
        #     stdout_file='out.pdb',
        #     store_stdout=False,
        #     store_stderr=False,
        #     fizzle_bad_rc=True)]
        #
        # fw_pdb_chain = self.build_fw(
        #     fts_pdb_chain, step_label,
        #     parents=[fw_vmd_run],
        #     files_in=files_in,
        #     files_out=files_out,
        #     category=self.hpc_specs['fw_noqueue_category'])
        #
        # fw_list.append(fw_pdb_chain)
        #
        # # PDB tidy
        # # --------
        # step_label = self.get_step_label('pdb_tidy')
        #
        # files_in = {'data_file': 'in.pdb'}
        # files_out = {'data_file': 'default.pdb'}
        #
        # fts_pdb_tidy = [CmdTask(
        #     cmd='pdb_tidy',
        #     env='python',
        #     stdin_file='in.pdb',
        #     stdout_file='default.pdb',
        #     store_stdout=False,
        #     store_stderr=False,
        #     fizzle_bad_rc=True)]
        #
        # fw_pdb_tidy = fw_pdb_chain = self.build_fw(
        #     fts_pdb_tidy, step_label,
        #     parents=[fw_pdb_chain],
        #     files_in=files_in,
        #     files_out=files_out,
        #     category=self.hpc_specs['fw_noqueue_category'])
        #
        # fw_list.append(fw_pdb_tidy)

        return fw_list, [fw_vmd_run], [fw_vmd_run]


class MergeSubstrateAndProbeSystems(
        DefaultPullMixin, DefaultPushMixin,
        ProcessAnalyzeAndVisualize,
        ):
    pass

    def __init__(self, *args, **kwargs):
       super().__init__(
           main_sub_wf=MergeSubstrateAndProbeSystemsMain,
           *args, **kwargs)
