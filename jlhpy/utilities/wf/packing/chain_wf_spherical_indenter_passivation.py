# -*- coding: utf-8 -*-

from jlhpy.utilities.wf.workflow_generator import WorkflowGenerator, ChainWorkflowGenerator, ParametricBranchingWorkflowGenerator
from jlhpy.utilities.wf.packing.sub_wf_010_indenter_bounding_sphere import IndenterBoundingSphere
from jlhpy.utilities.wf.packing.sub_wf_020_surfactant_molecule_measures import SurfactantMoleculeMeasures
from jlhpy.utilities.wf.packing.sub_wf_030_packing_constraint_spheres import PackingConstraintSpheres
from jlhpy.utilities.wf.packing.sub_wf_040_spherical_surfactant_packing import SphericalSurfactantPacking

from jlhpy.utilities.wf.packing.sub_wf_110_gromacs_prep import GromacsPrep
from jlhpy.utilities.wf.packing.sub_wf_120_gromacs_em import GromacsEnergyMinimization

from jlhpy.utilities.wf.packing.sub_wf_130_gromacs_pull_prep import GromacsPullPrep
from jlhpy.utilities.wf.packing.sub_wf_140_gromacs_pull import GromacsPull

from jlhpy.utilities.wf.packing.sub_wf_150_gromacs_solvate import GromacsSolvate

from jlhpy.utilities.wf.building_blocks.gmx.sub_wf_gromacs_em_solvated import GromacsEnergyMinimizationAfterSolvation
from jlhpy.utilities.wf.building_blocks.gmx.sub_wf_gromacs_nvt import GromacsNVTEquilibration
from jlhpy.utilities.wf.building_blocks.gmx.sub_wf_gromacs_npt import GromacsNPTEquilibration
from jlhpy.utilities.wf.building_blocks.gmx.sub_wf_gromacs_relax import GromacsRelaxation

class SphericalSurfactantPackingPrep(ChainWorkflowGenerator):
    """Spherical surfactant packing with PACKMOL sub workflow.

    Concatenates
    - IndenterBoundingSphere
    - SurfactantMoleculeMeasures
    - PackingConstraintSpheres
    """

    def __init__(self, *args, **kwargs):
        sub_wf_components = [
            IndenterBoundingSphere,
            SurfactantMoleculeMeasures,
            PackingConstraintSpheres,
        ]
        super().__init__(*args, sub_wf_components=sub_wf_components, **kwargs)


class GromacsPackingMinimizationEquilibration(ChainWorkflowGenerator):
    """Minimization of spherical surfactant packing with GROMACS chain workflow.

    Concatenates
    - SphericalSurfactantPacking

    - GromacsPrep
    - GromacsEnergyMinimization

    - GromacsPullPrep
    - GromacsPull

    - GromacsSolvate
    - GromacsEnergyMinimizationAfterSolvation

    - GromacsNVTEquilibration
    - GromacsNPTEquilibration
    - GromacsRelaxation
    """

    def __init__(self, *args, **kwargs):
        sub_wf_components = [
            SphericalSurfactantPacking,
            GromacsPrep,
            GromacsEnergyMinimization,
            GromacsPullPrep,
            GromacsPull,
            GromacsSolvate,
            GromacsEnergyMinimizationAfterSolvation,
            GromacsNVTEquilibration,
            GromacsNPTEquilibration,
            GromacsRelaxation,
        ]
        super().__init__(*args, sub_wf_components=sub_wf_components, **kwargs)


class IndenterPassivation(ChainWorkflowGenerator):
    """Spherical surfactant packing with PACKMOL sub workflow.

    Concatenates
    - SphericalSurfactantPackingPrep
    - GromacsPackingMinimizationChainWorkflowGenerator
    """

    def __init__(self, *args, **kwargs):
        sub_wf_components = [
            SphericalSurfactantPackingPrep,
            GromacsPackingMinimizationEquilibration,
        ]
        super().__init__(*args, sub_wf_components=sub_wf_components, **kwargs)


class ParametricGromacsPackingMinimizationEquilibration(ParametricBranchingWorkflowGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, sub_wf=GromacsPackingMinimizationEquilibration, **kwargs)


class ParametricIndenterPassivation(ChainWorkflowGenerator):
    """Spherical surfactant packing with PACKMOL sub workflow.

    Concatenates
    - SphericalSurfactantPackingPrep
    - GromacsPackingMinimizationChainWorkflowGenerator
    """

    def __init__(self, *args, **kwargs):
        sub_wf_components = [
            SphericalSurfactantPackingPrep,
            ParametricGromacsPackingMinimizationEquilibration,
        ]
        super().__init__(*args, sub_wf_components=sub_wf_components, **kwargs)
