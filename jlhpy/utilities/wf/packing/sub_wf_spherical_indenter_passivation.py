# -*- coding: utf-8 -*-

from jlhpy.utilities.wf.workflow_generator import SubWorkflowGenerator,ChainWorkflowGenerator
from jlhpy.utilities.wf.packing.sub_wf_indenter_bounding_sphere import IndenterBoundingSphereSubWorkflowGenerator
from jlhpy.utilities.wf.packing.sub_wf_surfactant_molecule_measures import SurfactantMoleculeMeasuresSubWorkflowGenerator
from jlhpy.utilities.wf.packing.sub_wf_packing_constraint_spheres import PackingConstraintSpheresSubWorkflowGenerator
from jlhpy.utilities.wf.packing.sub_wf_spherical_surfactant_packing import SphericalSurfactantPackingSubWorkflowGenerator

from jlhpy.utilities.wf.packing.sub_wf_gromacs_prep import GromacsPrepSubWorkflowGenerator
from jlhpy.utilities.wf.packing.sub_wf_gromacs_em import GromacsEnergyMinimizationSubWorkflowGenerator


class SphericalIndenterPassivationSubWorkflowGenerator(ChainWorkflowGenerator):
    """Spherical surfactant packing with PACKMOL sub workflow.

    Concatenates
    - IndenterBoundingSphereSubWorkflowGenerator
    - SurfactantMoleculeMeasuresSubWorkflowGenerator
    - PackingConstraintSpheresSubWorkflowGenerator
    - SphericalSurfactantPackingWorkflowGenerator
    """

    def __init__(self, *args, **kwargs):
        sub_wf_components = [
            IndenterBoundingSphereSubWorkflowGenerator(*args, **kwargs),
            SurfactantMoleculeMeasuresSubWorkflowGenerator(*args, **kwargs),
            PackingConstraintSpheresSubWorkflowGenerator(*args, **kwargs),
            SphericalSurfactantPackingSubWorkflowGenerator(*args, **kwargs),
        ]
        if 'wf_name_prefix' not in kwargs:
            kwargs['wf_name_prefix'] = 'indenter passivation chain workflow'
        super().__init__(sub_wf_components, *args, **kwargs)


class IntermediateTestingWorkflow(ChainWorkflowGenerator):
    """Spherical surfactant packing with PACKMOL sub workflow.

    Concatenates
    - SphericalIndenterPassivationSubWorkflowGenerator
    - GromacsPrepSubWorkflowGenerator
    - GromacsEnergyMinimizationSubWorkflowGenerator
    """

    def __init__(self, *args, **kwargs):
        sub_wf_components = [
            SphericalIndenterPassivationSubWorkflowGenerator(*args, **kwargs),
            GromacsPrepSubWorkflowGenerator(*args, **kwargs),
            GromacsEnergyMinimizationSubWorkflowGenerator(*args, **kwargs),
        ]
        if 'wf_name_prefix' not in kwargs:
            kwargs['wf_name_prefix'] = 'intermmediate testing workflow'
        super().__init__(sub_wf_components, *args, **kwargs)
