# -*- coding: utf-8 -*-


from jlhpy.utilities.wf.workflow_generator import (ChainWorkflowGenerator)
from jlhpy.utilities.wf.probe_on_substrate.sub_wf_010_merge import MergeSubstrateAndProbeSystems
from jlhpy.utilities.wf.building_blocks.sub_wf_vmd_pdb_cleanup import PDBCleanup
from jlhpy.utilities.wf.building_blocks.sub_wf_count_components import CountComponents
from jlhpy.utilities.wf.building_blocks.gmx.chain_wf_gromacs import \
    GromacsMinimizationEquilibrationRelaxationNoSolvation as GromacsMinimizationEquilibrationRelaxation
from jlhpy.utilities.wf.probe_on_substrate.sub_wf_020_gmx2lmp import CHARMM36GMX2LMP

from jlhpy.utilities.wf.probe_on_substrate.sub_wf_030_split_datafile import SplitDatafile
from jlhpy.utilities.wf.probe_on_substrate.sub_wf_040_lammps_minimization import LAMMPSMinimization
from jlhpy.utilities.wf.probe_on_substrate.sub_wf_050_lammps_equilibration_nvt import LAMMPSEquilibrationNVT
from jlhpy.utilities.wf.probe_on_substrate.sub_wf_060_lammps_equilibration_npt import LAMMPSEquilibrationNPT
from jlhpy.utilities.wf.probe_on_substrate.sub_wf_070_lammps_equilibration_dpd import LAMMPSEquilibrationDPD


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


class ProbeOnSubstrateMergeAndEqulibration(ChainWorkflowGenerator):
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


class ProbeOnSubstrateConversion(ChainWorkflowGenerator):
    """Merge, minimize and equilibrate substrate and probe.

    Concatenates
    - MergeSubstrateAndProbeSystems
    """

    def __init__(self, *args, **kwargs):
        sub_wf_components = [
            CHARMM36GMX2LMP,
            SplitDatafile,
        ]
        super().__init__(*args, sub_wf_components=sub_wf_components, **kwargs)


class ProbeOnSubstrateMinizationAndEquilibration(ChainWorkflowGenerator):
    """Merge, minimize and equilibrate substrate and probe.

    Concatenates
    - MergeSubstrateAndProbeSystems
    """

    def __init__(self, *args, **kwargs):
        sub_wf_components = [
            LAMMPSMinimization,
            LAMMPSEquilibrationNVT,
            LAMMPSEquilibrationNPT,
            LAMMPSEquilibrationDPD,
        ]
        super().__init__(*args, sub_wf_components=sub_wf_components, **kwargs)
