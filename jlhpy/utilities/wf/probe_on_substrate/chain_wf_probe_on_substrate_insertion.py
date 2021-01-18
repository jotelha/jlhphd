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
from jlhpy.utilities.wf.probe_on_substrate.sub_wf_110_lammps_probe_normal_approach import LAMMPSProbeNormalApproach
from jlhpy.utilities.wf.probe_on_substrate.sub_wf_120_probe_analysis import ProbeAnalysis


# TODO: reduce, sort and eliminate obsolete

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


class ProbeOnSubstrateMergeAndGROMACSEqulibration(ChainWorkflowGenerator):
    """Merge, minimize and equilibrate substrate and probe.

    Concatenates
    - MergeSubstrateAndProbeSystems
    - PDBCleanup
    - CountComponents
    - GromacsMinimizationEquilibrationRelaxation
    """

    def __init__(self, *args, **kwargs):
        sub_wf_components = [
            MergeSubstrateAndProbeSystems,
            PDBCleanup,  # clean up dirty VMD pdb
            CountComponents,  # count atoms and molecules (i.e. residues) in system
            GromacsMinimizationEquilibrationRelaxation,

        ]
        super().__init__(*args, sub_wf_components=sub_wf_components, **kwargs)


class ProbeOnSubstrateGMX2LMPConversion(ChainWorkflowGenerator):
    """Convert GROMACS system to LAMMPS system using CHARMM36 force field and
    splits output into coordinates and topology datafile and parameters input file.

    Concatenates
    - CHARMM36GMX2LMP
    - SplitDatafile
    """

    def __init__(self, *args, **kwargs):
        sub_wf_components = [
            CHARMM36GMX2LMP,
            SplitDatafile,
        ]
        super().__init__(*args, sub_wf_components=sub_wf_components, **kwargs)


class ProbeOnSubstrateMinimizationAndEquilibration(ChainWorkflowGenerator):
    """Minimize and equilibrate substrate and probe with LAMMPS.

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


class ProbeOnSubstrateConversionMinimizationAndEquilibration(ChainWorkflowGenerator):
    """Convert GROMACS system to LAMMPS system using CHARMM36 force field,
    then minimize and equilibrate substrate and probe with LAMMPS.

    Concatenates
    - LAMMPSMinimization
    - ProbeOnSubstrateConversion
    - ProbeOnSubstrateConversionMinizationAndEquilibration
    """

    def __init__(self, *args, **kwargs):
        sub_wf_components = [
            ProbeOnSubstrateGMX2LMPConversion,
            ProbeOnSubstrateMinimizationAndEquilibration,
        ]
        super().__init__(*args, sub_wf_components=sub_wf_components, **kwargs)


class ProbeOnSubstrateMergeConversionMinimizationAndEquilibration(ChainWorkflowGenerator):
    """Merge probe and substrate component, minimize and equilibrate with GROMACS,
    convert GROMACS system to LAMMPS system using CHARMM36 force field,
    then minimize and equilibrate substrate and probe with LAMMPS.

    Concatenates
    - LAMMPSMinimization
    - ProbeOnSubstrateConversion
    - ProbeOnSubstrateConversionMinizationAndEquilibration
    """

    def __init__(self, *args, **kwargs):
        sub_wf_components = [
            ProbeOnSubstrateMergeAndGROMACSEqulibration,
            ProbeOnSubstrateGMX2LMPConversion,
            ProbeOnSubstrateMinimizationAndEquilibration,
        ]
        super().__init__(*args, sub_wf_components=sub_wf_components, **kwargs)


class ProbeOnSubstrateNormalApproach(ChainWorkflowGenerator):
    """Run and analyze probe on substrate approach with LAMMPS.

    Concatenates
    - LAMMPSProbeNormalApproach
    - ProbeAnalysis
    """

    def __init__(self, *args, **kwargs):
        sub_wf_components = [
            LAMMPSProbeNormalApproach,
            ProbeAnalysis,
        ]
        super().__init__(*args, sub_wf_components=sub_wf_components, **kwargs)


class ProbeOnSubstrateMinimizationEquilibrationAndNormalApproach(ChainWorkflowGenerator):
    """Minimize and equilibrate substrate and probe with LAMMPS.

    Concatenates
    - MergeSubstrateAndProbeSystems
    """

    def __init__(self, *args, **kwargs):
        sub_wf_components = [
            LAMMPSMinimization,
            LAMMPSEquilibrationNVT,
            LAMMPSEquilibrationNPT,
            LAMMPSEquilibrationDPD,
            LAMMPSProbeNormalApproach,
            ProbeAnalysis,
        ]
        super().__init__(*args, sub_wf_components=sub_wf_components, **kwargs)

class ProbeOnSubstrateMergeConversionMinimizationEquilibrationAndApproach(ChainWorkflowGenerator):
    """Merge probe and substrate component, minimize, equilibrate with GROMACS,
    convert GROMACS system to LAMMPS system using CHARMM36 force field,
    then minimize, equilibrate, run approach production substrate and probe with LAMMPS.

    Concatenates
    - LAMMPSMinimization
    - ProbeOnSubstrateConversion
    - ProbeOnSubstrateConversionMinizationAndEquilibration
    - ProbeAnalysis
    """

    def __init__(self, *args, **kwargs):
        sub_wf_components = [
            ProbeOnSubstrateMergeAndGROMACSEqulibration,
            ProbeOnSubstrateGMX2LMPConversion,
            LAMMPSMinimization,
            LAMMPSEquilibrationNVT,
            LAMMPSEquilibrationNPT,
            LAMMPSEquilibrationDPD,
            LAMMPSProbeNormalApproach,
            ProbeAnalysis,
        ]
        super().__init__(*args, sub_wf_components=sub_wf_components, **kwargs)
