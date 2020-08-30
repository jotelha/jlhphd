# -*- coding: utf-8 -*-
"""Indenter bounding sphere sub workflow."""

import datetime
import pymongo

from fireworks import Firework
from fireworks.user_objects.firetasks.filepad_tasks import GetFilesByQueryTask
from fireworks.user_objects.firetasks.filepad_tasks import AddFilesTask
from imteksimfw.fireworks.user_objects.firetasks.cmd_tasks import CmdTask, EvalPyEnvTask

from jlhpy.utilities.wf.workflow_generator import (
    WorkflowGenerator, ProcessAnalyzeAndVisualize)
from jlhpy.utilities.wf.mixin.mixin_wf_storage import (
   DefaultPullMixin, DefaultPushMixin)

class GromacsPrepMain(WorkflowGenerator):
    """Prepare system for processing with GROMACS.

    dynamic infiles:
    - data_file:     in.pdb
    - topology_file: default.top

    outfiles:
    - data_file:       default.gro
    - topology_file:   default.top
    - restraint_file:  default.posre.itp
    """

    def main(self, fws_root=[]):
        fw_list = []

        # box dimensions
        # --------------
        step_label = self.get_step_label('box_dim')

        fts_box_dim = [
            EvalPyEnvTask(
                func='lambda v: v[0], v[1], v[2]',
                inputs=[
                    'metadata->system->substrate->measures',
                ],
                outputs=[
                    'metadata->system->substrate->length',
                    'metadata->system->substrate->width',
                    'metadata->system->substrate->height',
                ],
                propagate=True,
            ),
            EvalPyEnvTask(
                func='lambda v, h: v[0], v[1], v[2] + h',
                inputs=[
                    'metadata->system->substrate->measures',
                    'metadata->system->solvent->height',
                ],
                outputs=[
                    'metadata->system->box->length',
                    'metadata->system->box->width',
                    'metadata->system->box->height',
                ],
                propagate=True,
            ),
        ]

        fw_box_dim = self.build_fw(
            fts_box_dim, step_label,
            parents=fws_root,
            category=self.hpc_specs['fw_noqueue_category'])

        # PDB chain
        # ---------
        step_label = self.get_step_label('pdb_chain')

        files_in =  {'data_file': 'in.pdb'}
        files_out = {'data_file': 'out.pdb'}

        fts_pdb_chain = [CmdTask(
            cmd='pdb_chain',
            env='python',
            stdin_file='in.pdb',
            stdout_file='out.pdb',
            store_stdout=False,
            store_stderr=False,
            fizzle_bad_rc=True)]

        fw_pdb_chain = self.build_fw(
            fts_pdb_chain, step_label,
            parents=fws_root,
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_noqueue_category'])

        fw_list.append(fw_pdb_chain)

        # PDB tidy
        # --------
        step_label = self.get_step_label('pdb_tidy')

        files_in =  {'data_file': 'in.pdb'}
        files_out = {'data_file': 'out.pdb'}

        fts_pdb_tidy = [CmdTask(
            cmd='pdb_tidy',
            env='python',
            stdin_file='in.pdb',
            stdout_file='out.pdb',
            store_stdout=False,
            store_stderr=False,
            fizzle_bad_rc=True)]

        fw_pdb_tidy = self.build_fw(
            fts_pdb_tidy, step_label,
            parents=[fw_pdb_chain],
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_noqueue_category'])

        fw_list.append(fw_pdb_tidy)

        # GMX pdb2gro
        # -----------
        step_label = self.get_step_label('gmx_gmx2gro')

        files_in =  {'data_file': 'in.pdb'}
        files_out = {
            'data_file': 'default.gro',
            'topology_file':   'default.top',
            'restraint_file':  'default.posre.itp'}

        fts_gmx_pdb2gro = [CmdTask(
            cmd='gmx',
            opt=['pdb2gmx',
                 '-f', 'in.pdb',
                 '-o', 'default.gro',
                 '-p', 'default.top',
                 '-i', 'default.posre.itp',
                 '-ff', 'charmm36',
                 '-water', 'tip3p'],
            env='python',
            stderr_file='std.err',
            stdout_file='std.out',
            store_stdout=True,
            store_stderr=True,
            fizzle_bad_rc=True)]

        fw_gmx_pdb2gro = self.build_fw(
            fts_gmx_pdb2gro, step_label,
            parents=[fw_pdb_tidy],
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_noqueue_category'])

        fw_list.append(fw_gmx_pdb2gro)

        # GMX editconf
        # ------------
        step_label = self.get_step_label('gmx_editconf')

        files_in = {
            'data_file': 'in.gro',
            'topology_file':   'default.top',
            'restraint_file':  'default.posre.itp'}
        files_out = {
            'data_file': 'default.gro',
            'topology_file':   'default.top',
            'restraint_file':  'default.posre.itp'}

        fts_gmx_editconf = [CmdTask(
            cmd='gmx',
            opt=['editconf',
                 '-f', 'in.gro',
                 '-o', 'default.gro',
                 '-b',
                 {'key': 'metadata->system->box->length'},
                 {'key': 'metadata->system->box->width'},
                 {'key': 'metadata->system->box->height'},
                 '-bt', 'triclinic',  # box type
                 ],
            env='python',
            stderr_file='std.err',
            stdout_file='std.out',
            store_stdout=True,
            store_stderr=True,
            fizzle_bad_rc=True)]

        fw_gmx_editconf = self.build_fw(
            fts_gmx_editconf, step_label,
            parents=[fw_gmx_pdb2gro, fw_box_dim],
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_noqueue_category'])

        fw_list.append(fw_gmx_editconf)

        return fw_list, [fw_gmx_editconf], [fw_pdb_chain, fw_box_dim]


class GromacsPrep(DefaultPullMixin, DefaultPushMixin, ProcessAnalyzeAndVisualize):
    def __init__(self, *args, **kwargs):
        super().__init__(main_sub_wf=GromacsPrepMain, *args, **kwargs)
