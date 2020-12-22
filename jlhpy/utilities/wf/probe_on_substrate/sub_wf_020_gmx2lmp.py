# -*- coding: utf-8 -*-
import datetime
import glob
import os
import pymongo

from fireworks.user_objects.firetasks.filepad_tasks import GetFilesByQueryTask
from fireworks.user_objects.firetasks.script_task import ScriptTask
from fireworks.user_objects.firetasks.templatewriter_task import TemplateWriterTask
from imteksimfw.fireworks.user_objects.firetasks.cmd_tasks import PickledPyEnvTask, EvalPyEnvTask, CmdTask
# from imteksimfw.fireworks.user_objects.firetasks.storage_tasks import GetObjectFromFilepadTask
from imteksimfw.utils.serialize import serialize_module_obj

from jlhpy.utilities.prep.segids import make_seg_id_seg_pdb_dict
from jlhpy.utilities.geometry.simulation_box import get_gro_box

from jlhpy.utilities.wf.mixin.mixin_wf_storage import DefaultPullMixin, DefaultPushMixin
from jlhpy.utilities.wf.workflow_generator import WorkflowGenerator

import jlhpy.utilities.wf.file_config as file_config


class CHARMM36GMX2LMPMain(WorkflowGenerator):
    """Convert CHARMM36 system from .gro file to lammps data file file format.

    inputs:
    - metadata->system->substrate->resname
    - metadata->system->surfactant->resname
    - metadata->system->counerion->resname
    - metadata->system->solvent->resname

    static infiles:
    - template_file: default.sh.template
        identified by 'metadata->name': file_config.BASH_GMX2PDB_TEMPLATE
    - template_file: gmx2pdb.tcl.template
        identified by 'metadata->name': file_config.VMD_PSFGEN_TEMPLATE

    dynamic infiles:
    - data_file: default.gro
    - run_file: default.tpr

    outfiles:
    - data_file: default.data
    - input_file: default.in
    - pdb_file: default.pdb
    - psf_file: default.psf
    - prm_file: par_default.prm
    - rtf_file: top_default.rtf
    - ctrl_pdb: default_ctrl.pdb
    - ctrl_psf: default_ctrl.psf
    """

    def push_infiles(self, fp):
        fp_files = []

        step_label = self.get_step_label('push_infiles')

        # gmx2pdb template file
        infiles = sorted(glob.glob(os.path.join(
            self.infile_prefix,
            file_config.BASH_SCRIPT_SUBDIR,
            file_config.BASH_GMX2PDB_TEMPLATE)))

        metadata = {
            'project': self.project_id,
            'type': 'template',
            'step': step_label,
            'name': file_config.BASH_GMX2PDB_TEMPLATE
        }

        files = {os.path.basename(f): f for f in infiles}

        for name, file_path in files.items():
            identifier = '/'.join((self.project_id, name))   # identifier is like a path on a file system
            fp_files.append(
                fp.add_file(
                    file_path,
                    identifier=identifier,
                    metadata=metadata))


        # vmd psfgen tcl template
        infiles = sorted(glob.glob(os.path.join(
            self.infile_prefix,
            file_config.VMD_SUBDIR,
            file_config.VMD_PSFGEN_TEMPLATE)))

        metadata = {
            'project': self.project_id,
            'type': 'template',
            'step': step_label,
            'name': file_config.VMD_PSFGEN_TEMPLATE
        }

        files = {os.path.basename(f): f for f in infiles}

        for name, file_path in files.items():
            identifier = '/'.join((self.project_id, name))  # identifier is like a path on a file system
            fp_files.append(
                fp.add_file(
                    file_path,
                    identifier=identifier,
                    metadata=metadata))

        # charmm36 prm parameter file
        infiles = sorted(glob.glob(os.path.join(
            self.infile_prefix,
            file_config.CHARMM_FF_SUBDIR,
            file_config.CHARMM36_PRM)))

        metadata = {
            'project': self.project_id,
            'type': 'input',
            'step': step_label,
            'name': file_config.CHARMM36_PRM
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

        # charmm36 rtf topology file
        infiles = sorted(glob.glob(os.path.join(
            self.infile_prefix,
            file_config.CHARMM_FF_SUBDIR,
            file_config.CHARMM36_RTF)))

        metadata = {
            'project': self.project_id,
            'type': 'input',
            'step': step_label,
            'name': file_config.CHARMM36_RTF
        }

        files = {os.path.basename(f): f for f in infiles}

        for name, file_path in files.items():
            identifier = '/'.join((self.project_id, name))  # identifier is like a path on a file system
            fp_files.append(
                fp.add_file(
                    file_path,
                    identifier=identifier,
                    metadata=metadata))

        return fp_files

    def main(self, fws_root=[]):
        fw_list = []

        # TODO: order migh be important, not sure
        components = [
            'substrate',
            'surfactant',
            'counterion',
            'solvent'
        ]

        # conserve gro simulation box measures to begin with

        step_label = self.get_step_label('box_measures')

        files_in = {
            'data_file': 'default.gro'
        }
        files_out = {}

        func_str = serialize_module_obj(get_gro_box)

        fts_gro_box = [
            PickledPyEnvTask(
                func=func_str,
                args=['default.gro'],
                outputs=[
                    'metadata->system->box->length',
                    'metadata->system->box->width',
                    'metadata->system->box->height',
                ],
                env='mdanalysis',
                propagate=True,
            )
        ]

        fw_gro_box = self.build_fw(
            fts_gro_box, step_label,
            parents=fws_root,
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_noqueue_category'])

        fw_list.append(fw_gro_box)

        # make residues whole
        # -------------------

        step_label = self.get_step_label('gmx_trjconv')

        files_in = {
            'data_file': 'in.gro',
            'run_file':  'in.tpr',
        }
        files_out = {
            'data_file': 'out.gro',
        }

        fts_gmx_trjconv = [
            CmdTask(
                cmd='gmx',
                opt=['trjconv',
                     '-f', 'in.gro',
                     '-s', 'in.tpr',
                     '-o', 'out.gro',
                     '-pbc', 'res'],
                env='python',
                stdin_key='stdin',
                store_stdout=True,
                store_stderr=True,
            ),
        ]

        fw_gmx_trjconv = self.build_fw(
            fts_gmx_trjconv, step_label,
            parents=fws_root,
            files_in=files_in,
            files_out=files_out,
            stdin='0\n',  # select the whole system
            category=self.hpc_specs['fw_noqueue_category'])

        fw_list.append(fw_gmx_trjconv)

        ### GMX2PDB
        
        # pull gmx2pdb bash script template
        # ---------------------------------

        step_label = self.get_step_label('pull_gmx2pdb_template')

        files_in = {}
        files_out = {
            'template_file': 'default.sh.template',
        }

        fts_pull_template = [
            GetFilesByQueryTask(
                query={
                    'metadata->project': self.project_id,
                    'metadata->name': file_config.BASH_GMX2PDB_TEMPLATE,
                },
                sort_key='metadata.datetime',
                sort_direction=pymongo.DESCENDING,
                limit=1,
                new_file_names=['default.sh.template'])]

        fw_pull_template = self.build_fw(
            fts_pull_template, step_label,
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_noqueue_category'])

        fw_list.append(fw_pull_template)

        # fill gmx2pdb bash script template
        # ---------------------------------

        step_label = self.get_step_label('fill_gmx2pdb_template')

        files_in = {'template_file': 'default.template'}
        files_out = {'script_file': 'default.sh'}

        # Jinja2 context:
        static_context = {
            'header': ', '.join((
                self.project_id,
                self.get_fw_label(step_label),
                str(datetime.datetime.now()))),
        }

        dynamic_context = {
            'components': 'run->template_writer->context->components'
        }

        resname_inputs = ['metadata->system->{}->resname'.format(c) for c in components]

        fts_gmx2pdb_template = [
            EvalPyEnvTask(
                func='lambda **kwargs: [{"name": k, "resname": v} for k,v in kwargs.items()]',
                kwargs_inputs={component: resname_key for component, resname_key in zip(components, resname_inputs)},
                outputs=['run->template_writer->context->components'],
                propagate=False,
            ),
            TemplateWriterTask({
                'context': static_context,
                'context_inputs': dynamic_context,  # TODO: dynamic number of files (currently limitted to 999 in template)
                'template_file': 'default.template',
                'template_dir': '.',
                'output_file': 'default.sh'}
            )]

        fw_gmx2pdb_template = self.build_fw(
            fts_gmx2pdb_template, step_label,
            parents=[fw_pull_template, *fws_root],
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_noqueue_category'])

        fw_list.append(fw_gmx2pdb_template)

        # run gmx2pdb bash script
        # -----------------------

        step_label = self.get_step_label('run_gmx2pdb')

        files_in = {
            'script_file': 'default.sh',
            'data_file': 'default.gro'
        }
        files_out = {
            'script_file': 'default.sh',  # untouche for archiving
            'tar_file': 'segments.tar.gz'
        }

        pdb_segment_chunk_glob_pattern = '*_[0-9][0-9][0-9].pdb'

        fts_gmx2pdb = [
            CmdTask(
                cmd='bash',
                opt=['default.sh'],
                env='gmx_and_vmd',
                stderr_file='gxm2pdb.err',
                stdout_file='gxm2pdb.out',
                stdlog_file='gxm2pdb.log',
                store_stdout=True,
                store_stderr=True,
            ),
            ScriptTask.from_str(
                'tar -czf segments.tar.gz {:s}'.format(pdb_segment_chunk_glob_pattern),
                {
                    'use_shell': True,
                    'fizzle_bad_rc': True
                }
            ),
        ]

        fw_gmx2pdb = self.build_fw(
            fts_gmx2pdb, step_label,
            parents=[fw_gmx2pdb_template, fw_gmx_trjconv],
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_noqueue_category'])

        fw_list.append(fw_gmx2pdb)

        ### PSFGEN

        # pull psfgen template
        # --------------------

        step_label = self.get_step_label('pull_psfgen_template')

        files_in = {}
        files_out = {
            'template_file': 'default.tcl.template',
        }

        fts_pull_psfgen_template = [
            GetFilesByQueryTask(
                query={
                    'metadata->project': self.project_id,
                    'metadata->name': file_config.VMD_PSFGEN_TEMPLATE,
                },
                sort_key='metadata.datetime',
                sort_direction=pymongo.DESCENDING,
                limit=1,
                new_file_names=['default.tcl.template'])]

        fw_pull_psfgen_template = self.build_fw(
            fts_pull_psfgen_template, step_label,
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_noqueue_category'])

        fw_list.append(fw_pull_psfgen_template)

        # pull prm and rtf
        # ----------------

        step_label = self.get_step_label('pull_prm_rtf')

        files_in = {}
        files_out = {
            'prm_file': 'par_default.prm',
            'rtf_file': 'top_default.rtf',
        }

        fts_pull_prm_rtf = [
            GetFilesByQueryTask(
                query={
                    'metadata->project': self.project_id,
                    'metadata->name': file_config.CHARMM36_PRM,
                },
                sort_key='metadata.datetime',
                sort_direction=pymongo.DESCENDING,
                limit=1,
                new_file_names=['par_default.prm']),
            GetFilesByQueryTask(
                query={
                    'metadata->project': self.project_id,
                    'metadata->name': file_config.CHARMM36_RTF,
                },
                sort_key='metadata.datetime',
                sort_direction=pymongo.DESCENDING,
                limit=1,
                new_file_names=['top_default.rtf'])
        ]

        fw_pull_prm_rtf = self.build_fw(
            fts_pull_prm_rtf, step_label,
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_noqueue_category'])

        fw_list.append(fw_pull_prm_rtf)

        # run psfgen
        # ----------

        step_label = self.get_step_label('run_psfgen')

        files_in = {
            'tar_file': 'default.tar.gz',
            'template_file': 'default.template',
            'prm_file': 'par_default.prm',
            'rtf_file': 'top_default.rtf',
        }
        files_out = {
            'tcl_file': 'default.tcl',  # for archiving
            'pdb_file': 'default.pdb',
            'psf_file': 'default.psf',
        }

        # PickledPyEnvTask
        func_str = serialize_module_obj(make_seg_id_seg_pdb_dict)

        # TemplateWriterTask
        static_context = {
            'header': ', '.join((
                self.project_id,
                self.get_fw_label(step_label),
                str(datetime.datetime.now()))),
            'rtf_in': 'top_default.rtf',
            'pdb_out': 'default.pdb',
            'psf_out': 'default.psf',
        }

        dynamic_context = {
            'residues': 'metadata->step_specific->psfgen->residues',
            'segments': 'run->psfgen->segments'
        }

        fts_psfgen = [
            ScriptTask.from_str('tar -xf default.tar.gz',
                                {
                                    'use_shell': True,
                                    'fizzle_bad_rc': True
                                }),
            PickledPyEnvTask(
                func=func_str,
                args=[pdb_segment_chunk_glob_pattern],
                outputs=['run->psfgen->segments'],
                env='imteksimpy',
                propagate=False,
            ),
            TemplateWriterTask({  # TODO: more context
                'context': static_context,
                'context_inputs': dynamic_context,
                'template_file': 'default.template',
                'template_dir': '.',
                'output_file': 'default.tcl'}
            ),
            CmdTask(  # run psgen
                cmd='vmd',
                opt=['-eofexit', '-e', 'default.tcl'],
                env='python',
                fizzle_bad_rc=True)
        ]

        fw_psfgen = self.build_fw(
            fts_psfgen, step_label,
            parents=[fw_pull_psfgen_template, fw_pull_prm_rtf, fw_gmx2pdb],
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_queue_category'],
            queueadapter=self.hpc_specs['single_core_job_queueadapter_defaults'])

        fw_list.append(fw_psfgen)

        ### charmm2lammps

        # pull prm and rtf again
        # ----------------------

        step_label = self.get_step_label('pull_prm_rtf')

        files_in = {}
        files_out = {
            'prm_file': 'par_default.prm',
            'rtf_file': 'top_default.rtf',
        }

        fts_pull_prm_rtf_again = [
            GetFilesByQueryTask(
                query={
                    'metadata->project': self.project_id,
                    'metadata->name': file_config.CHARMM36_PRM,
                },
                sort_key='metadata.datetime',
                sort_direction=pymongo.DESCENDING,
                limit=1,
                new_file_names=['par_default.prm']),
            GetFilesByQueryTask(
                query={
                    'metadata->project': self.project_id,
                    'metadata->name': file_config.CHARMM36_RTF,
                },
                sort_key='metadata.datetime',
                sort_direction=pymongo.DESCENDING,
                limit=1,
                new_file_names=['top_default.rtf'])
        ]

        fw_pull_prm_rtf_again = self.build_fw(
            fts_pull_prm_rtf_again, step_label,
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_noqueue_category'])

        fw_list.append(fw_pull_prm_rtf_again)

        # run ch2lmp
        # ----------

        step_label = self.get_step_label('run_ch2lmp')

        files_in = {
            'pdb_file': 'default.pdb',
            'psf_file': 'default.psf',
            'prm_file': 'par_default.prm',
            'rtf_file': 'top_default.rtf',
        }
        files_out = {
            'input_file': 'default.in',
            'data_file': 'default.data',
            'prm_file': 'par_default.prm',  # untouched for archiving
            'rtf_file': 'top_default.rtf',  # untouched for archiving
            'ctrl_pdb': 'default_ctrl.pdb',  # for archiving
            'ctrl_psf': 'default_ctrl.psf',  # for archiving
        }

        fts_ch2lmp = [
            EvalPyEnvTask(
                func='lambda border, lx, ly, lz: ["-border={:f}".format(border), "-lx={:f}".format(lx), "-ly={:f}".format(ly), "-lz={:f}".format(lz)]',
                args = [0.0],  # border
                inputs = [
                    'metadata->system->box->length',
                    'metadata->system->box->width',
                    'metadata->system->box->height',
                ],
                outputs=[
                    'run->charmm2lammps->opts->border',
                    'run->charmm2lammps->opts->lx',
                    'run->charmm2lammps->opts->ly',
                    'run->charmm2lammps->opts->lz',
                ],
                propagate=False,
            ),
            CmdTask(  # run charm2lammps
                cmd='charmm2lammps.pl',
                # TODO: adapt box measures
                # first is for 'par_default.prm' and 'top_default.rtf', second is for 'default.pdb'
                opt=['default', 'default',
                    {'key': 'run->charmm2lammps->opts->border'},
                    {'key': 'run->charmm2lammps->opts->lx'},
                    {'key': 'run->charmm2lammps->opts->ly'},
                    {'key': 'run->charmm2lammps->opts->lz'},
                ],
                env='python',
                fizzle_bad_rc=True)
        ]

        # outfile of charmm2lammps should be .data and .in
        # additional outfiles should be _ctrl.pdb and _ctrl.psf

        fw_ch2lmp = self.build_fw(
            fts_ch2lmp, step_label,
            parents=[fw_pull_prm_rtf_again, fw_gro_box, fw_psfgen],
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_noqueue_category'])

        fw_list.append(fw_ch2lmp)

        return fw_list, [fw_psfgen, fw_ch2lmp], [fw_gmx2pdb_template, fw_gro_box, fw_gmx_trjconv]


class CHARMM36GMX2LMP(
        DefaultPullMixin, DefaultPushMixin,
        CHARMM36GMX2LMPMain):
    pass
