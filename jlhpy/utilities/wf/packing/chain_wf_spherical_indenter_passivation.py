# -*- coding: utf-8 -*-

from jlhpy.utilities.wf.workflow_generator import WorkflowGenerator, ChainWorkflowGenerator, ParametricBranchingWorkflowGenerator
from jlhpy.utilities.wf.packing.sub_wf_010_indenter_bounding_sphere import IndenterBoundingSphereWorkflowGenerator
from jlhpy.utilities.wf.packing.sub_wf_020_surfactant_molecule_measures import SurfactantMoleculeMeasuresWorkflowGenerator
from jlhpy.utilities.wf.packing.sub_wf_030_packing_constraint_spheres import PackingConstraintSpheresWorkflowGenerator
from jlhpy.utilities.wf.packing.sub_wf_040_spherical_surfactant_packing import SphericalSurfactantPackingWorkflowGenerator

from jlhpy.utilities.wf.packing.sub_wf_110_gromacs_prep import GromacsPrepWorkflowGenerator
from jlhpy.utilities.wf.packing.sub_wf_120_gromacs_em import GromacsEnergyMinimizationWorkflowGenerator

from jlhpy.utilities.wf.packing.sub_wf_130_gromacs_pull_prep import GromacsPullPrepWorkflowGenerator
from jlhpy.utilities.wf.packing.sub_wf_140_gromacs_pull import GromacsPullWorkflowGenerator

from jlhpy.utilities.wf.packing.sub_wf_150_gromacs_solvate import GromacsSolvateWorkflowGenerator
from jlhpy.utilities.wf.packing.sub_wf_160_gromacs_em_solvated import GromacsEnergyMinimizationAfterSolvationWorkflowGenerator

from jlhpy.utilities.wf.packing.sub_wf_170_gromacs_nvt import GromacsNVTEquilibrationWorkflowGenerator
from jlhpy.utilities.wf.packing.sub_wf_180_gromacs_npt import GromacsNPTEquilibrationWorkflowGenerator
from jlhpy.utilities.wf.packing.sub_wf_190_gromacs_relax import GromacsRelaxationWorkflowGenerator

class SphericalSurfactantPackingChainWorkflowGenerator(ChainWorkflowGenerator):
    """Spherical surfactant packing with PACKMOL sub workflow.

    Concatenates
    - IndenterBoundingSphereWorkflowGenerator
    - SurfactantMoleculeMeasuresWorkflowGenerator
    - PackingConstraintSpheresWorkflowGenerator
    """

    def __init__(self, *args, **kwargs):
        sub_wf_components = [
            IndenterBoundingSphereWorkflowGenerator(*args, **kwargs),
            SurfactantMoleculeMeasuresWorkflowGenerator(*args, **kwargs),
            PackingConstraintSpheresWorkflowGenerator(*args, **kwargs),
        ]
        super().__init__(*args, sub_wf_components=sub_wf_components, **kwargs)


class GromacsPackingMinimizationEquilibrationChainWorkflowGenerator(ChainWorkflowGenerator):
    """Minimization of spherical surfactant packing with GROMACS chain workflow.

    Concatenates
    - SphericalSurfactantPackingWorkflowGenerator

    - GromacsPrepWorkflowGenerator
    - GromacsEnergyMinimizationWorkflowGenerator

    - GromacsPullPrepWorkflowGenerator
    - GromacsPullWorkflowGenerator

    - GromacsSolvateWorkflowGenerator
    - GromacsEnergyMinimizationAfterSolvationWorkflowGenerator

    - GromacsNVTEquilibrationWorkflowGenerator
    - GromacsNPTEquilibrationWorkflowGenerator
    - GromacsRelaxationWorkflowGenerator
    """

    def __init__(self, *args, **kwargs):
        sub_wf_components = [
            SphericalSurfactantPackingWorkflowGenerator(*args, **kwargs),
            GromacsPrepWorkflowGenerator(*args, **kwargs),
            GromacsEnergyMinimizationWorkflowGenerator(*args, **kwargs),
            GromacsPullPrepWorkflowGenerator(*args, **kwargs),
            GromacsPullWorkflowGenerator(*args, **kwargs),
            GromacsSolvateWorkflowGenerator(*args, **kwargs),
            GromacsEnergyMinimizationAfterSolvationWorkflowGenerator(*args, **kwargs),
            GromacsNVTEquilibrationWorkflowGenerator(*args, **kwargs),
            GromacsNPTEquilibrationWorkflowGenerator(*args, **kwargs),
            GromacsRelaxationWorkflowGenerator(*args, **kwargs),
        ]
        super().__init__(*args, sub_wf_components=sub_wf_components, **kwargs)


class IndenterPassivationChainWorkflowGenerator(ChainWorkflowGenerator):
    """Spherical surfactant packing with PACKMOL sub workflow.

    Concatenates
    - SphericalSurfactantPackingChainWorkflowGenerator
    - GromacsPackingMinimizationChainWorkflowGenerator
    """

    def __init__(self, *args, **kwargs):
        sub_wf_components = [
            SphericalSurfactantPackingChainWorkflowGenerator(*args, **kwargs),
            GromacsPackingMinimizationEquilibrationChainWorkflowGenerator(*args, **kwargs),
        ]
        super().__init__(*args, sub_wf_components=sub_wf_components, **kwargs)


class IndenterPassivationParametricWorkflowGenerator(ChainWorkflowGenerator):
    """Spherical surfactant packing with PACKMOL sub workflow.

    Concatenates
    - SphericalSurfactantPackingChainWorkflowGenerator
    - GromacsPackingMinimizationChainWorkflowGenerator
    """

    def __init__(self, *args, **kwargs):
        sub_wf_components = [
            SphericalSurfactantPackingChainWorkflowGenerator(*args, **kwargs),
            ParametricBranchingWorkflowGenerator(
                sub_wf=GromacsPackingMinimizationEquilibrationChainWorkflowGenerator,
                *args, **kwargs)
        ]
        super().__init__(*args, sub_wf_components=sub_wf_components, **kwargs)
