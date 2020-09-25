# -*- coding: utf-8 -*-

from jlhpy.utilities.wf.workflow_generator import WorkflowGenerator, ChainWorkflowGenerator #, ParametricBranchingWorkflowGenerator
from jlhpy.utilities.wf.substrate.sub_wf_010_create_fcc_111_substrate import CreateSubstrateWorkflowGenerator
from jlhpy.utilities.wf.substrate.sub_wf_020_lammps_fixed_box_minimization import LAMMPSFixedBoxMinimizationWorkflowGenerator
from jlhpy.utilities.wf.substrate.sub_wf_030_lammps_relaxed_box_minimization import LAMMPSRelaxedBoxMinimizationWorkflowGenerator
from jlhpy.utilities.wf.substrate.sub_wf_040_lammps_equilibration_nvt import LAMMPSEquilibrationNVTWorkflowGenerator
from jlhpy.utilities.wf.substrate.sub_wf_050_lammps_equilibration_npt import LAMMPSEquilibrationNPTWorkflowGenerator

class FCCSubstrateCreationChainWorkflowGenerator(ChainWorkflowGenerator):
    """FCC substrate creation workflow.

    Concatenates
    - CreateSubstrateWorkflowGenerator
    - LAMMPSFixedBoxMinimizationWorkflowGenerator
    - LAMMPSRelaxedBoxMinimizationWorkflowGenerator
    - LAMMPSEquilibrationNVTWorkflowGenerator
    - LAMMPSEquilibrationNPTWorkflowGenerator
    """

    def __init__(self, *args, **kwargs):
        sub_wf_components = [
            CreateSubstrateWorkflowGenerator,
            LAMMPSFixedBoxMinimizationWorkflowGenerator,
            LAMMPSRelaxedBoxMinimizationWorkflowGenerator,
            LAMMPSEquilibrationNVTWorkflowGenerator,
            LAMMPSEquilibrationNPTWorkflowGenerator,
        ]
        super().__init__(*args, sub_wf_components=sub_wf_components, **kwargs)
