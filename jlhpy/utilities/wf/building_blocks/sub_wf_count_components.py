# -*- coding: utf-8 -*-
from imteksimfw.fireworks.user_objects.firetasks.cmd_tasks import PickledPyEnvTask

from jlhpy.utilities.analysis.count_components import count_pdb_components_by_resname

from imteksimfw.utils.serialize import serialize_module_obj
from jlhpy.utilities.wf.workflow_generator import WorkflowGenerator
from jlhpy.utilities.wf.mixin.mixin_wf_storage import DefaultPullMixin, DefaultPushMixin


class CountComponentsMain(WorkflowGenerator):
    """Count components in system.

    Counts atoms and molecules by resiude names.
    Assumes one resiude to match one molecule.

    inputs:
    - metadata->system->surfactant->resname
    - metadata->system->counterion->resname
    - metadata->system->substrate->resname
    - metadata->system->solvent->resname

    dynamic infiles:
    - data_file: default.pdb

    outfiles:
    - data_file: default.pdb  # untouched

    outputs:
    - metadata->system->surfactant->natoms
    - metadata->system->surfactant->nmolecules
    - metadata->system->counterion->natoms
    - metadata->system->counterion->nmolecules
    - metadata->system->substrate->natoms
    - metadata->system->substrate->nmolecules
    - metadata->system->solvent->natoms
    - metadata->system->solvent->nmolecules
    """
    def main(self, fws_root=[]):
        fw_list = []

        # convert
        # -------------------------
        step_label = self.get_step_label('count')

        files_in = {
            'data_file': 'default.pdb',
        }
        files_out = {
            'data_file': 'default.pdb',  # untouched
        }

        func_str = serialize_module_obj(count_pdb_components_by_resname)

        labels = [
            'surfactant',
            'counterion',
            'substrate',
            'solvent'
        ]

        resname_inputs = [
            'metadata->system->surfactant->resname',
            'metadata->system->counterion->resname',
            'metadata->system->substrate->resname',
            'metadata->system->solvent->resname'
        ]

        natoms_outputs =  [
            'metadata->system->surfactant->natoms',
            'metadata->system->counterion->natoms',
            'metadata->system->substrate->natoms',
            'metadata->system->solvent->natoms',
        ]

        nmolecules_outputs = [
            'metadata->system->surfactant->nmolecules',
            'metadata->system->counterion->nmolecules',
            'metadata->system->substrate->nmolecules',
            'metadata->system->solvent->nmolecules',
        ]

        fts_count = [PickledPyEnvTask(
            func=func_str,
            args=['default.pdb'],
            kwargs_inputs={
                'resname': resname_input_key,
            },
            outputs=[natoms_output_key, nmolecules_output_key],
            env='mdanalysis',
            stderr_file='{}.err'.format(label),
            stdout_file='{}.out'.format(label),
            stdlog_file='{}.log'.format(label),
            store_stdout=True,
            store_stderr=True,
            store_stdlog=True,
            propagate=True,
        ) for label, resname_input_key, natoms_output_key, nmolecules_output_key
            in zip(labels, resname_inputs, natoms_outputs, nmolecules_outputs)]

        fw_count = self.build_fw(
            fts_count, step_label,
            parents=fws_root,
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_noqueue_category'])

        fw_list.append(fw_count)

        return fw_list, [fw_count], [fw_count]


class CountComponents(DefaultPullMixin, CountComponentsMain):
    pass