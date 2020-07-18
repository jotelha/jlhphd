# -*- coding: utf-8 -*-

from jlhpy.utilities.wf.workflow_generator import SubWorkflowGenerator, ChainWorkflowGenerator, ParametricBranchingWorkflowGenerator
from jlhpy.utilities.wf.packing.sub_wf_010_indenter_bounding_sphere import IndenterBoundingSphereSubWorkflowGenerator
from jlhpy.utilities.wf.packing.sub_wf_020_surfactant_molecule_measures import SurfactantMoleculeMeasuresSubWorkflowGenerator
from jlhpy.utilities.wf.packing.sub_wf_030_packing_constraint_spheres import PackingConstraintSpheresSubWorkflowGenerator
from jlhpy.utilities.wf.packing.sub_wf_040_spherical_surfactant_packing import SphericalSurfactantPackingSubWorkflowGenerator

from jlhpy.utilities.wf.packing.sub_wf_110_gromacs_prep import GromacsPrepSubWorkflowGenerator
from jlhpy.utilities.wf.packing.sub_wf_120_gromacs_em import GromacsEnergyMinimizationSubWorkflowGenerator

from jlhpy.utilities.wf.packing.sub_wf_130_gromacs_pull_prep import GromacsPullPrepSubWorkflowGenerator
from jlhpy.utilities.wf.packing.sub_wf_140_gromacs_pull import GromacsPullSubWorkflowGenerator

from jlhpy.utilities.wf.packing.sub_wf_150_gromacs_solvate import GromacsSolvateSubWorkflowGenerator
from jlhpy.utilities.wf.packing.sub_wf_160_gromacs_em_solvated import GromacsEnergyMinimizationAfterSolvationSubWorkflowGenerator

from jlhpy.utilities.wf.packing.sub_wf_170_gromacs_nvt import GromacsNVTEquilibrationSubWorkflowGenerator
from jlhpy.utilities.wf.packing.sub_wf_180_gromacs_npt import GromacsNPTEquilibrationSubWorkflowGenerator
from jlhpy.utilities.wf.packing.sub_wf_190_gromacs_relax import GromacsRelaxationSubWorkflowGenerator

class SphericalSurfactantPackingChainWorkflowGenerator(ChainWorkflowGenerator):
    """Spherical surfactant packing with PACKMOL sub workflow.

    Concatenates
    - IndenterBoundingSphereSubWorkflowGenerator
    - SurfactantMoleculeMeasuresSubWorkflowGenerator
    - PackingConstraintSpheresSubWorkflowGenerator
    """

    def __init__(self, *args, **kwargs):
        sub_wf_components = [
            IndenterBoundingSphereSubWorkflowGenerator(*args, **kwargs),
            SurfactantMoleculeMeasuresSubWorkflowGenerator(*args, **kwargs),
            PackingConstraintSpheresSubWorkflowGenerator(*args, **kwargs),
        ]
        sub_wf_name = 'SphericalSurfactantPacking'
        if 'wf_name_prefix' not in kwargs:
            kwargs['wf_name_prefix'] = sub_wf_name
        else:
            kwargs['wf_name_prefix'] = ':'.join((kwargs['wf_name_prefix'], sub_wf_name))
        super().__init__(sub_wf_components, *args, **kwargs)


class GromacsPackingMinimizationEquilibrationChainWorkflowGenerator(ChainWorkflowGenerator):
    """Minimization of spherical surfactant packing with GROMACS chain workflow.

    Concatenates
    - SphericalSurfactantPackingSubWorkflowGenerator

    - GromacsPrepSubWorkflowGenerator
    - GromacsEnergyMinimizationSubWorkflowGenerator

    - GromacsPullPrepSubWorkflowGenerator
    - GromacsPullSubWorkflowGenerator

    - GromacsSolvateSubWorkflowGenerator
    - GromacsEnergyMinimizationAfterSolvationSubWorkflowGenerator

    - GromacsNVTEquilibrationSubWorkflowGenerator
    - GromacsNPTEquilibrationSubWorkflowGenerator
    - GromacsRelaxationSubWorkflowGenerator
    """

    def __init__(self, *args, **kwargs):
        sub_wf_components = [
            SphericalSurfactantPackingSubWorkflowGenerator(*args, **kwargs),
            GromacsPrepSubWorkflowGenerator(*args, **kwargs),
            GromacsEnergyMinimizationSubWorkflowGenerator(*args, **kwargs),
            GromacsPullPrepSubWorkflowGenerator(*args, **kwargs),
            GromacsPullSubWorkflowGenerator(*args, **kwargs),
            GromacsSolvateSubWorkflowGenerator(*args, **kwargs),
            GromacsEnergyMinimizationAfterSolvationSubWorkflowGenerator(*args, **kwargs),
            GromacsNVTEquilibrationSubWorkflowGenerator(*args, **kwargs),
            GromacsNPTEquilibrationSubWorkflowGenerator(*args, **kwargs),
            GromacsRelaxationSubWorkflowGenerator(*args, **kwargs),
        ]
        sub_wf_name = 'GromacsPackingMinimizationEquilibration'
        if 'wf_name_prefix' not in kwargs:
            kwargs['wf_name_prefix'] = sub_wf_name
        else:
            kwargs['wf_name_prefix'] = ':'.join((kwargs['wf_name_prefix'], sub_wf_name))
        super().__init__(sub_wf_components, *args, **kwargs)


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
        sub_wf_name = 'IndenterPassivation'
        if 'wf_name_prefix' not in kwargs:
            kwargs['wf_name_prefix'] = sub_wf_name
        else:
            kwargs['wf_name_prefix'] = ':'.join((kwargs['wf_name_prefix'], sub_wf_name))
        super().__init__(sub_wf_components, *args, **kwargs)

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
        sub_wf_name = 'IndenterPassivation'
        if 'wf_name_prefix' not in kwargs:
            kwargs['wf_name_prefix'] = sub_wf_name
        else:
            kwargs['wf_name_prefix'] = ':'.join((kwargs['wf_name_prefix'], sub_wf_name))
        super().__init__(sub_wf_components, *args, **kwargs)
