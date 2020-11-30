# -*- coding: utf-8 -*-

from jlhpy.utilities.wf.building_blocks.gmx.chain_wf_gromacs import GromacsMinimizationEquilibrationRelaxation
from jlhpy.utilities.wf.building_blocks.sub_wf_surfactant_molecule_measures import SurfactantMoleculeMeasures
from jlhpy.utilities.wf.probe_on_substrate.sub_wf_010_merge import MergeSubstrateAndProbeSystems
from jlhpy.utilities.wf.utils import get_nested_dict_value
from jlhpy.utilities.wf.workflow_generator import (EncapsulatingWorkflowGenerator,
                                                   ChainWorkflowGenerator, BranchingWorkflowGenerator,
                                                   ParametricBranchingWorkflowGenerator)


class ProbeOnSubstrate(ChainWorkflowGenerator):
    """Merge, minimize and equilibrate substrate and probe.

    Concatenates
    - MergeSubstrateAndProbeSystems
    """

    def __init__(self, *args, **kwargs):
        sub_wf_components = [
            MergeSubstrateAndProbeSystems,
        ]
        super().__init__(*args, sub_wf_components=sub_wf_components, **kwargs)
