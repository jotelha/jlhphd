# -*- coding: utf-8 -*-

from jlhpy.utilities.wf.workflow_generator import (
    ChainWorkflowGenerator, BranchingWorkflowGenerator, ParametricBranchingWorkflowGenerator)

from jlhpy.utilities.wf.building_blocks.sub_wf_surfactant_molecule_measures import SurfactantMoleculeMeasures

from jlhpy.utilities.wf.flat_packing.sub_wf_005_format_conversion import FormatConversion
from jlhpy.utilities.wf.flat_packing.sub_wf_010_flat_substrate_measures import FlatSubstrateMeasures
from jlhpy.utilities.wf.flat_packing.sub_wf_030_packing import (
    MonolayerPacking,
    BilayerPacking,
    CylindricalPacking,
    HemicylindricalPacking,)


class SubstratePreparation(ChainWorkflowGenerator):
    """Flat substrate format conversion and measures sub workflow.

    Concatenates
    - FormatConversion
    - FlatSubstrateMeasures
    """

    def __init__(self, *args, **kwargs):
        sub_wf_components = [
            FormatConversion,
            FlatSubstrateMeasures,
        ]
        super().__init__(*args, sub_wf_components=sub_wf_components, **kwargs)

class ComponentMeasures(BranchingWorkflowGenerator):
    """Determine measures of surfactant and substrate.

    Branches into
    - SubstratePreparation
    - SurfactantMoleculeMeasures
    """

    def __init__(self, *args, **kwargs):
        sub_wf_components = [
            SubstratePreparation,
            SurfactantMoleculeMeasures,
        ]
        super().__init__(*args, sub_wf_components=sub_wf_components, **kwargs)


class SurfactantPacking(BranchingWorkflowGenerator):
    """Pack different film morphologies on flat substrate.

    Branches into
    - MonolayerPacking
    - BilayerPacking
    - CylindricalPacking
    - HemicylindricalPacking
    """

    def __init__(self, *args, **kwargs):
        sub_wf_components = [
            MonolayerPacking,
            BilayerPacking,
            CylindricalPacking,
            HemicylindricalPacking,

        ]
        super().__init__(*args, sub_wf_components=sub_wf_components, **kwargs)


# class PackingOnFlatSubstrate(ChainWorkflowGenerator):
#     """Flat substrate packing with PACKMOL sub workflow.
#
#     Concatenates
#     - ComponentMeasures
#     - SurfactantPacking
#     """
#
#     def __init__(self, *args, **kwargs):
#         sub_wf_components = [
#             ComponentMeasures,
#             SurfactantPacking,
#         ]
#         super().__init__(*args, sub_wf_components=sub_wf_components, **kwargs)


class SurfactantPackingParametricBranching(ParametricBranchingWorkflowGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, sub_wf=SurfactantPacking, **kwargs)


class SubstratePassivationParametricWorkflowGenerator(ChainWorkflowGenerator):
    """Film packing on flat substrate with PACKMOL parametric workflow.

    Concatenates
    - SphericalSurfactantPacking
    - SurfactantPackingParametricBranching
    """

    def __init__(self, *args, **kwargs):
        sub_wf_components = [
            ComponentMeasures,
            SurfactantPackingParametricBranching
        ]
        super().__init__(*args, sub_wf_components=sub_wf_components, **kwargs)
