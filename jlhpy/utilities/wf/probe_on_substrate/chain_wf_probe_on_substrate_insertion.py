# -*- coding: utf-8 -*-


from jlhpy.utilities.wf.workflow_generator import (EncapsulatingWorkflowGenerator,
                                                   ChainWorkflowGenerator, BranchingWorkflowGenerator,
                                                   ParametricBranchingWorkflowGenerator)
from jlhpy.utilities.wf.probe_on_substrate.sub_wf_010_merge import MergeSubstrateAndProbeSystems
from jlhpy.utilities.wf.building_blocks.sub_wf_vmd_pdb_cleanup import PDBCleanup
from jlhpy.utilities.wf.building_blocks.sub_wf_count_components import CountComponents
from jlhpy.utilities.wf.building_blocks.gmx.chain_wf_gromacs import \
    GromacsMinimizationEquilibrationRelaxationNoSolvation as GromacsMinimizationEquilibrationRelaxation


class ProbeOnSubstrateTest(ChainWorkflowGenerator):
    """Merge, minimize and equilibrate substrate and probe.

    Concatenates
    - MergeSubstrateAndProbeSystems
    """

    def __init__(self, *args, **kwargs):
        sub_wf_components = [
            PDBCleanup,  # clean up dirty VMD pdb
            CountComponents,  # count atoms and molecules (i.e. residues) in system
            GromacsMinimizationEquilibrationRelaxation,
        ]
        super().__init__(*args, sub_wf_components=sub_wf_components, **kwargs)


class ProbeOnSubstrate(ChainWorkflowGenerator):
    """Merge, minimize and equilibrate substrate and probe.

    Concatenates
    - MergeSubstrateAndProbeSystems
    """

    def __init__(self, *args, **kwargs):
        sub_wf_components = [
            MergeSubstrateAndProbeSystems,
            PDBCleanup,  # clean up dirty VMD pdb
            CountComponents,  # count atoms and molecules (i.e. residues) in system
            GromacsMinimizationEquilibrationRelaxation,
        ]
        super().__init__(*args, sub_wf_components=sub_wf_components, **kwargs)
