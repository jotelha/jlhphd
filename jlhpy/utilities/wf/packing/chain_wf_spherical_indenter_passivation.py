# -*- coding: utf-8 -*-

from jlhpy.utilities.wf.workflow_generator import SubWorkflowGenerator, ChainWorkflowGenerator
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


class SphericalSurfactantPackingChainWorkflowGenerator(ChainWorkflowGenerator):
    """Spherical surfactant packing with PACKMOL sub workflow.

    Concatenates
    - IndenterBoundingSphereSubWorkflowGenerator
    - SurfactantMoleculeMeasuresSubWorkflowGenerator
    - PackingConstraintSpheresSubWorkflowGenerator
    - SphericalSurfactantPackingSubWorkflowGenerator
    """

    def __init__(self, *args, **kwargs):
        sub_wf_components = [
            IndenterBoundingSphereSubWorkflowGenerator(*args, **kwargs),
            SurfactantMoleculeMeasuresSubWorkflowGenerator(*args, **kwargs),
            PackingConstraintSpheresSubWorkflowGenerator(*args, **kwargs),
            SphericalSurfactantPackingSubWorkflowGenerator(*args, **kwargs),
        ]
        if 'wf_name_prefix' not in kwargs:
            kwargs['wf_name_prefix'] = 'spherical surfactant packing chain workflow'
        super().__init__(sub_wf_components, *args, **kwargs)


class GromacsPackingMinimizationChainWorkflowGenerator(ChainWorkflowGenerator):
    """Minimization of spherical surfactant packing with GROMACS chain workflow.

    Concatenates
    - GromacsPrepSubWorkflowGenerator
    - GromacsEnergyMinimizationSubWorkflowGenerator

    - GromacsPullPrepSubWorkflowGenerator
    - GromacsPullSubWorkflowGenerator

    - GromacsSolvateSubWorkflowGenerator
    - GromacsEnergyMinimizationAfterSolvationSubWorkflowGenerator
    """

    def __init__(self, *args, **kwargs):
        sub_wf_components = [
            GromacsPrepSubWorkflowGenerator(*args, **kwargs),
            GromacsEnergyMinimizationSubWorkflowGenerator(*args, **kwargs),
            GromacsPullPrepSubWorkflowGenerator(*args, **kwargs),
            GromacsPullSubWorkflowGenerator(*args, **kwargs),
            GromacsSolvateSubWorkflowGenerator(*args, **kwargs),
            GromacsEnergyMinimizationAfterSolvationSubWorkflowGenerator(*args, **kwargs),
        ]
        if 'wf_name_prefix' not in kwargs:
            kwargs['wf_name_prefix'] = 'intermmediate testing workflow'
        super().__init__(sub_wf_components, *args, **kwargs)

class IntermediateTestingWorkflow(ChainWorkflowGenerator):
    """Spherical surfactant packing with PACKMOL sub workflow.

    Concatenates
    - SphericalSurfactantPackingChainWorkflowGenerator
    - GromacsPackingMinimizationChainWorkflowGenerator
    """

    def __init__(self, *args, **kwargs):
        sub_wf_components = [
            SphericalSurfactantPackingChainWorkflowGenerator(*args, **kwargs),
            GromacsPackingMinimizationChainWorkflowGenerator(*args, **kwargs),
        ]
        if 'wf_name_prefix' not in kwargs:
            kwargs['wf_name_prefix'] = 'intermmediate testing workflow'
        super().__init__(sub_wf_components, *args, **kwargs)
