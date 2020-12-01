# -*- coding: utf-8 -*-
from imteksimfw.fireworks.user_objects.firetasks.cmd_tasks import CmdTask
from jlhpy.utilities.wf.workflow_generator import WorkflowGenerator
from jlhpy.utilities.wf.mixin.mixin_wf_storage import (
   DefaultPullMixin, DefaultPushMixin)


class PDBCleanupMain(WorkflowGenerator):
    """Clean up PDB file.

    dynamic infiles:
        only queried in pull stub, otherwise expected through data flow

    - data_file: default.pdb

    outfiles:
    - data_file: default.pdb
    """
    def main(self, fws_root=[]):
        fw_list = []

        # PDB cleanup
        # -----------
        step_label = self.get_step_label('pdb_cleanup')

        files_in = {'data_file': 'in.pdb'}
        files_out = {'data_file': 'default.pdb'}

        # implementation of bash snippet:
        # GROMACS parses pdbs with more than 9999 rsidues and 99999 atoms.
        # resids and atom ids can be reused for distinct entities as long
        # as the entries are not neighbours within the .pdb file.
        # PACKMOL outputs non-standard PDB (atom ids can be hex),
        # we convert that to some other non-standard PDB format.

        # # split substrate and non-substrate components
        # pdb_selresname -AUM in.pdb > subst.pdb
        # pdb_delresname -AUM in.pdb > non_subst.pdb
        #
        # # make sure each substrate atom has its own resid with 0 <= resid <=9999
        # pdb_reres_by_atom_9999 -0 subst.pdb > subst_reres.pdb
        # # make sure each non-substrate resid 0 <= resid <= 9999
        # pdb_reres_9999 -0 non_subst.pdb > non_subst_reres.pdb
        #
        # # concatenate components. Remove last line 'END' from head
        # # and first six lines header from tail.
        # head -n -1 subst_reres.pdb > trimmed_head.pdb
        # tail -n +6 not_subst_reres.pdb > trimmed_tail.pdb
        #
        # cat trimmed_head.pdb trimmed_tail.pdb > cat.pdb
        # pdb_reatom_99999 cat.pdb > reatom.pdb
        # pdb_reres_9999 -0 reatom.pdb > reres.pdb

        fts_pdb_cleanup = [
            CmdTask(
                cmd='pdb_selresname',
                opt=['-AUM'],  # TODO: make dynmic
                env='python',
                stdin_file='in.pdb',
                stdout_file='subst.pdb',
                store_stdout=False,
                store_stderr=False,
                fizzle_bad_rc=True),
            CmdTask(
                cmd='pdb_delresname',
                opt=['-AUM'],  # TODO: make dynmic
                env='python',
                stdin_file='in.pdb',
                stdout_file='non_subst.pdb',
                store_stdout=False,
                store_stderr=False,
                fizzle_bad_rc=True),
            CmdTask(
                cmd='pdb_reres_by_atom_9999',
                opt=['-0'],  # start numbering at resid 0
                env='python',
                stdin_file='subst.pdb',
                stdout_file='subst_reres.pdb',
                store_stdout=False,
                store_stderr=False,
                fizzle_bad_rc=True),
            CmdTask(
                cmd='pdb_reres_9999',
                opt=['-0'],  # start numbering at resid 0
                env='python',
                stdin_file='non_subst.pdb',
                stdout_file='non_subst_reres.pdb',
                store_stdout=False,
                store_stderr=False,
                fizzle_bad_rc=True),
            CmdTask(
                cmd='head',
                opt=['-n', '-1'],
                env='python',
                stdin_file='subst_reres.pdb',
                stdout_file='trimmed_head.pdb',
                store_stdout=False,
                store_stderr=False,
                fizzle_bad_rc=True),
            CmdTask(
                cmd='tail',
                opt=['-n', '+6'],
                env='python',
                stdin_file='non_subst_reres.pdb',
                stdout_file='trimmed_tail.pdb',
                store_stdout=False,
                store_stderr=False,
                fizzle_bad_rc=True),
            CmdTask(
                cmd='cat',
                opt=['trimmed_head.pdb', 'trimmed_tail.pdb'],
                env='python',
                stdout_file='cat.pdb',
                store_stdout=False,
                store_stderr=False,
                fizzle_bad_rc=True),
            CmdTask(
                cmd='pdb_reatom_99999',
                opt=['-0'],  # start numbering at resid 0
                env='python',
                stdin_file='cat.pdb',
                stdout_file='reatom.pdb',
                store_stdout=False,
                store_stderr=False),
            CmdTask(
                cmd='pdb_reres_9999',
                opt=['-0'],  # start numbering at resid 0
                env='python',
                stdin_file='reatom.pdb',
                stdout_file='reres.pdb',
                store_stdout=False,
                store_stderr=False,
                fizzle_bad_rc=True),
            CmdTask(
                cmd='pdb_chain',
                env='python',
                stdin_file='reres.pdb',
                stdout_file='default.pdb',
                store_stdout=False,
                store_stderr=False,
                fizzle_bad_rc=True),
            ]

        fw_pdb_cleanup = self.build_fw(
            fts_pdb_cleanup, step_label,
            parents=fws_root,
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_noqueue_category'])

        fw_list.append(fw_pdb_cleanup)

        return fw_list, [fw_pdb_cleanup], [fw_pdb_cleanup]


class PDBCleanup(DefaultPullMixin, DefaultPushMixin, PDBCleanupMain):
    pass
