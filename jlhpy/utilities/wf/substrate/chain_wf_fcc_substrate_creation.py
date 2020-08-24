# -*- coding: utf-8 -*-

from jlhpy.utilities.wf.workflow_generator import SubWorkflowGenerator, ChainWorkflowGenerator #, ParametricBranchingWorkflowGenerator
from jlhpy.utilities.wf.substrate.sub_wf_010_create_fcc_111_substrate import CreateSubstrateSubWorkflowGenerator
from jlhpy.utilities.wf.substrate.sub_wf_020_lammps_fixed_box_minimization import LAMMPSFixedBoxMinimizationSubWorkflowGenerator
from jlhpy.utilities.wf.substrate.sub_wf_030_lammps_relaxed_box_minimization import LAMMPSRelaxedBoxMinimizationSubWorkflowGenerator
from jlhpy.utilities.wf.substrate.sub_wf_040_lammps_equilibration_nvt import LAMMPSEquilibrationNVTSubWorkflowGenerator
from jlhpy.utilities.wf.substrate.sub_wf_050_lammps_equilibration_npt import LAMMPSEquilibrationNPTSubWorkflowGenerator

class FCCSubstrateCreationChainWorkflowGenerator(ChainWorkflowGenerator):
    """FCC substrate creation workflow.

    Concatenates
    - CreateSubstrateSubWorkflowGenerator
    - LAMMPSFixedBoxMinimizationSubWorkflowGenerator
    - LAMMPSRelaxedBoxMinimizationSubWorkflowGenerator
    - LAMMPSEquilibrationNVTSubWorkflowGenerator
    - LAMMPSEquilibrationNPTSubWorkflowGenerator
    """

    def __init__(self, *args, **kwargs):
        sub_wf_components = [
            CreateSubstrateSubWorkflowGenerator(*args, **kwargs),
            LAMMPSFixedBoxMinimizationSubWorkflowGenerator(*args, **kwargs),
            LAMMPSRelaxedBoxMinimizationSubWorkflowGenerator(*args, **kwargs),
            LAMMPSEquilibrationNVTSubWorkflowGenerator(*args, **kwargs),
            LAMMPSEquilibrationNPTSubWorkflowGenerator(*args, **kwargs),
        ]
        sub_wf_name = 'FCCSubstrateCreation'
        if 'wf_name_prefix' not in kwargs:
            kwargs['wf_name_prefix'] = sub_wf_name
        else:
            kwargs['wf_name_prefix'] = ':'.join((kwargs['wf_name_prefix'], sub_wf_name))
        super().__init__(sub_wf_components, *args, **kwargs)
