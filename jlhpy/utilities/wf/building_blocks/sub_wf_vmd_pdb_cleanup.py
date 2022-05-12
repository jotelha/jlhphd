# -*- coding: utf-8 -*-
from imteksimfw.fireworks.user_objects.firetasks.cmd_tasks import CmdTask, EvalPyEnvTask
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
        # VMD outputs non-standard PDB (atom ids can be hex),
        # we convert that to some other non-standard PDB format that works with GROMACS.

        # make sure all residues of the same name are listed together

        # pdb_reatom_99999 in.pdb > reatom.pdb
        # pdb_reres_9999 -0 reatom.pdb > reres.pdb

        components = [
            'substrate',
            'surfactant',
            'counterion',
            'solvent'
        ]

        resname_inputs = ['metadata->system->{}->resname'.format(c) for c in components]
        opt_keys = ['run->pdb_selresname->{}'.format(c) for c in components]
        pdb_names = ['{}.pdb'.format(c) for c in components]

        fts_prep_opt = [
            EvalPyEnvTask(
                func='lambda s: "-{}".format(s)',
                inputs=[input],
                outputs=[output],
                propagate=False,
                fork=True, # otherwise module environment polluted, subsequent CmdTask fails
            ) for input, output in zip(resname_inputs, opt_keys)]

        fts_pdb_selresname = [
            CmdTask(
                cmd='pdb_selresname',
                opt=[{'key': input_key}],
                env='python',
                stdin_file='in.pdb',
                stdout_file=pdb_output,
                store_stdout=False,
                store_stderr=False,
                fork=True,
            ) for input_key, pdb_output in zip(opt_keys, pdb_names)
        ]

        # Remove header and footer where necessary:
        # pdbs are expected to start like ...
        #   CRYST1  149.836  149.725  319.939  90.00  90.00  90.00 P 1           1
        #   ATOM      0  AU  AUM     0       1.470   0.560 149.290  0.00  0.00
        #   ATOM      1  AU  AUM     1       4.360   0.560 149.310  0.00  0.00
        # ... and end like ...
        #   ATOM  83274  HW1 SOL  9376     111.823  25.402 231.790  0.00  0.00
        #   ATOM  83275  HW2 SOL  9376     111.493  26.372 232.910  0.00  0.00
        #   END
        # ... thus when concatenating (pdb_merge) pdb files, we wand to remove
        # such headers and footers except the header of the first and the footer of
        # the last file.

        fts_prep_for_merge = [
            CmdTask(
                cmd='sed',
                opt=['-nE', '/^END/!p', '-i', pdb_names[0]],
                env='python',
                fork=True,
            ),
            *[CmdTask(
                cmd='sed',
                opt=['-nE', '/^(CRYST|END)/!p', '-i', pdb_name],
                env='python',
                fork=True,
            ) for pdb_name in pdb_names[1:-1]],
            CmdTask(
                cmd='sed',
                opt=['-nE', '/^CRYST/!p', '-i', pdb_names[-1]],
                env='python',
                fork=True,
            )
        ]

        fts_pdb_cleanup = [
            *fts_prep_opt,
            *fts_pdb_selresname,
            *fts_prep_for_merge,
            CmdTask(
                cmd='pdb_merge',
                opt=[pdb_input for pdb_input in pdb_names],
                env='python',
                stdout_file='merged.pdb',
                store_stdout=False,
                store_stderr=False,
                fork=True),
            CmdTask(
                cmd='pdb_reatom_99999',
                opt=['-0'],  # start numbering at atomid 0
                env='python',
                stdin_file='merged.pdb',
                stdout_file='reatom.pdb',
                store_stdout=False,
                store_stderr=False,
                fork=True),
            CmdTask(
                cmd='pdb_reres_9999',
                opt=['-0'],  # start numbering at resid 0
                env='python',
                stdin_file='reatom.pdb',
                stdout_file='reres.pdb',
                store_stdout=False,
                store_stderr=False,
                fizzle_bad_rc=True,
                fork=True),
            CmdTask(
                cmd='pdb_chain',
                env='python',
                stdin_file='reres.pdb',
                stdout_file='default.pdb',
                store_stdout=False,
                store_stderr=False,
                fizzle_bad_rc=True,
                fork=True),
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
