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
            CreateSubstrateWorkflowGenerator(*args, **kwargs),
            LAMMPSFixedBoxMinimizationWorkflowGenerator(*args, **kwargs),
            LAMMPSRelaxedBoxMinimizationWorkflowGenerator(*args, **kwargs),
            LAMMPSEquilibrationNVTWorkflowGenerator(*args, **kwargs),
            LAMMPSEquilibrationNPTWorkflowGenerator(*args, **kwargs),
        ]
        sub_wf_name = 'FCCSubstrateCreation'
        if 'wf_name_prefix' not in kwargs:
            kwargs['wf_name_prefix'] = sub_wf_name
        else:
            kwargs['wf_name_prefix'] = ':'.join((kwargs['wf_name_prefix'], sub_wf_name))
        super().__init__(sub_wf_components, *args, **kwargs)
