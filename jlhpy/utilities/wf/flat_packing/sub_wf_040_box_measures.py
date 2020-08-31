# -*- coding: utf-8 -*-
from imteksimfw.fireworks.user_objects.firetasks.cmd_tasks import EvalPyEnvTask
from jlhpy.utilities.wf.workflow_generator import WorkflowGenerator

class SimulationBoxMeasures(WorkflowGenerator):
    """Simulation box measures from substrate measures and desired height of solvent layer.

    inputs:
    - metadata->system->substrate->measures ([float])
    - metadata->system->solvent->height (float)

    outputs:
    - metadata->system->substrate->length (float)
    - metadata->system->substrate->width (float)
    - metadata->system->substrate->height (float)
    - metadata->system->box->length (float)
    - metadata->system->box->width (float)
    - metadata->system->box->height (float)


    """
    def main(self, fws_root=[]):
        fw_list = []

        # box dimensions
        # --------------
        step_label = self.get_step_label('box_dim')

        fts_box_dim = [
            EvalPyEnvTask(
                func='lambda v: (v[0], v[1], v[2])',
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
                func='lambda v, h: (v[0], v[1], v[2] + h)',
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

        fw_list.append(fw_box_dim)

        return fw_list, [fw_box_dim], [fw_box_dim]
