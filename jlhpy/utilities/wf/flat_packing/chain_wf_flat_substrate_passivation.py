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

from jlhpy.utilities.wf.building_blocks.gmx.chain_wf_gromacs import GromacsMinimizationEquilibrationRelaxation

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


class MonolayerPackingAndEquilibartion(ChainWorkflowGenerator):
    def __init__(self, *args, **kwargs):
        sub_wf_components = [
            MonolayerPacking,
            GromacsMinimizationEquilibrationRelaxation,
        ]
        super().__init__(*args, sub_wf_components=sub_wf_components, **kwargs)


class BilayerPackingAndEquilibartion(ChainWorkflowGenerator):
    def __init__(self, *args, **kwargs):
        sub_wf_components = [
            BilayerPacking,
            GromacsMinimizationEquilibrationRelaxation,
        ]
        super().__init__(*args, sub_wf_components=sub_wf_components, **kwargs)


class CylindricalPackingAndEquilibartion(ChainWorkflowGenerator):
    def __init__(self, *args, **kwargs):
        sub_wf_components = [
            CylindricalPacking,
            GromacsMinimizationEquilibrationRelaxation,
        ]
        super().__init__(*args, sub_wf_components=sub_wf_components, **kwargs)


class HemicylindricalPackingAndEquilibartion(ChainWorkflowGenerator):
    def __init__(self, *args, **kwargs):
        sub_wf_components = [
            HemicylindricalPacking,
            GromacsMinimizationEquilibrationRelaxation,
        ]
        super().__init__(*args, sub_wf_components=sub_wf_components, **kwargs)


class SurfactantPackingAndEquilibration(BranchingWorkflowGenerator):
    """Pack different film morphologies on flat substrate.

    Branches into
    - MonolayerPackingAndEquilibartion
    - BilayerPackingAndEquilibartion
    - CylindricalPackingAndEquilibartion
    - HemicylindricalPackingAndEquilibartion
    """

    def __init__(self, *args, **kwargs):
        sub_wf_components = [
            MonolayerPackingAndEquilibartion,
            BilayerPackingAndEquilibartion,
            CylindricalPackingAndEquilibartion,
            HemicylindricalPackingAndEquilibartion,

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


class SurfactantPackingAndEquilibrationParametricBranching(ParametricBranchingWorkflowGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, sub_wf=SurfactantPackingAndEquilibration, **kwargs)


class SubstratePassivation(ChainWorkflowGenerator):
    """Film packing on flat substrate with PACKMOL parametric workflow.

    Concatenates
    - SphericalSurfactantPacking
    - SurfactantPackingParametricBranching
    """

    def __init__(self, *args, **kwargs):
        sub_wf_components = [
            ComponentMeasures,
            SurfactantPackingAndEquilibrationParametricBranching
        ]
        super().__init__(*args, sub_wf_components=sub_wf_components, **kwargs)
