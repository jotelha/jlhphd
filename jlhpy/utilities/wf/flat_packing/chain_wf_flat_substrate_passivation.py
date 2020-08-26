# -*- coding: utf-8 -*-

from jlhpy.utilities.wf.workflow_generator import SubWorkflowGenerator, ChainWorkflowGenerator
from jlhpy.utilities.wf.flat_packing.sub_wf_010_indenter_bounding_sphere import IndenterBoundingSphereSubWorkflowGenerator
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

class FlatSubstratePackingChainWorkflowGenerator(ChainWorkflowGenerator):
    """Flat substrate packing with PACKMOL sub workflow.

    Concatenates
    - SurfactantMoleculeMeasuresSubWorkflowGenerator
    - PackingConstraintPlanesSubWorkflowGenerator
    - PlanarSurfactantPackingSubWorkflowGenerator
    """

    def __init__(self, *args, **kwargs):
        sub_wf_components = [
            SurfactantMoleculeMeasuresSubWorkflowGenerator(*args, **kwargs),
            PackingConstraintPlanesSubWorkflowGenerator(*args, **kwargs),
            PlanarSurfactantPackingSubWorkflowGenerator(*args, **kwargs),
        ]
        sub_wf_name = 'FlatSubstratePacking'
        if 'wf_name_prefix' not in kwargs:
            kwargs['wf_name_prefix'] = sub_wf_name
        else:
            kwargs['wf_name_prefix'] = ':'.join((kwargs['wf_name_prefix'], sub_wf_name))
        super().__init__(sub_wf_components, *args, **kwargs)


class GromacsPackingMinimizationEquilibrationChainWorkflowGenerator(ChainWorkflowGenerator):
    """Minimization of spherical surfactant packing with GROMACS chain workflow.

    Concatenates
    - GromacsPrepSubWorkflowGenerator

    - GromacsSolvateSubWorkflowGenerator
    - GromacsEnergyMinimizationAfterSolvationSubWorkflowGenerator

    - GromacsNVTEquilibrationSubWorkflowGenerator
    - GromacsNPTEquilibrationSubWorkflowGenerator
    """

    def __init__(self, *args, **kwargs):
        sub_wf_components = [
            GromacsPrepSubWorkflowGenerator(*args, **kwargs),
            GromacsEnergyMinimizationSubWorkflowGenerator(*args, **kwargs),
            GromacsPullPrepSubWorkflowGenerator(*args, **kwargs),
            GromacsPullSubWorkflowGenerator(*args, **kwargs),
            GromacsSolvateSubWorkflowGenerator(*args, **kwargs),
            GromacsEnergyMinimizationAfterSolvationSubWorkflowGenerator(*args, **kwargs),
            GromacsNVTEquilibrationSubWorkflowGenerator(*args, **kwargs),
            GromacsNPTEquilibrationSubWorkflowGenerator(*args, **kwargs),
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
