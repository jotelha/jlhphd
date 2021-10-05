# -*- coding: utf-8 -*-

from jlhpy.utilities.wf.utils import get_nested_dict_value

from jlhpy.utilities.wf.workflow_generator import (EncapsulatingWorkflowGenerator,
    ChainWorkflowGenerator, BranchingWorkflowGenerator, ParametricBranchingWorkflowGenerator)

from jlhpy.utilities.wf.building_blocks.sub_wf_surfactant_molecule_measures import SurfactantMoleculeMeasures

from jlhpy.utilities.wf.flat_packing.sub_wf_005_format_conversion import FormatConversion
from jlhpy.utilities.wf.flat_packing.sub_wf_010_flat_substrate_measures import FlatSubstrateMeasures
from jlhpy.utilities.wf.flat_packing.sub_wf_030_packing import (
    MonolayerPacking,
    BilayerPacking,
    CylindricalPacking,
    HemicylindricalPacking,)

from jlhpy.utilities.wf.flat_packing.sub_wf_035_pdb_cleanup import PDBCleanup
from jlhpy.utilities.wf.flat_packing.sub_wf_040_box_measures import SimulationBoxMeasures

# TODO: relaxation postprocessing analysis_rdf often takes longer than default 6 h wall time configured for JUWELS

from jlhpy.utilities.wf.building_blocks.gmx.chain_wf_gromacs import GromacsMinimizationEquilibrationRelaxation


class SubstrateAndBoxMeasures(BranchingWorkflowGenerator):
    """Determine measures of substrate and simulation box.

    Branches into
    - FlatSubstrateMeasures
    - SimulationBoxMeasures
    """

    def __init__(self, *args, **kwargs):
        sub_wf_components = [
            FlatSubstrateMeasures,
            SimulationBoxMeasures,
        ]
        super().__init__(*args, sub_wf_components=sub_wf_components, **kwargs)


class SubstratePreparation(ChainWorkflowGenerator):
    """Flat substrate format conversion and measures sub workflow.

    Concatenates
    - FormatConversion
    - SubstrateAndBoxMeasures
    """

    def __init__(self, *args, **kwargs):
        sub_wf_components = [
            FormatConversion,
            SubstrateAndBoxMeasures,
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


# TODO: remove four classes, pull up dynamic decision for packing workflow
class MonolayerPackingAndEquilibration(ChainWorkflowGenerator):
    def __init__(self, *args, **kwargs):
        sub_wf_components = [
            MonolayerPacking,
            PDBCleanup,
            GromacsMinimizationEquilibrationRelaxation,
        ]
        super().__init__(*args, sub_wf_components=sub_wf_components, **kwargs)


class BilayerPackingAndEquilibration(ChainWorkflowGenerator):
    def __init__(self, *args, **kwargs):
        sub_wf_components = [
            BilayerPacking,
            PDBCleanup,
            GromacsMinimizationEquilibrationRelaxation,
        ]
        super().__init__(*args, sub_wf_components=sub_wf_components, **kwargs)


class CylindricalPackingAndEquilibration(ChainWorkflowGenerator):
    def __init__(self, *args, **kwargs):
        sub_wf_components = [
            CylindricalPacking,
            PDBCleanup,
            GromacsMinimizationEquilibrationRelaxation,
        ]
        super().__init__(*args, sub_wf_components=sub_wf_components, **kwargs)


class HemicylindricalPackingAndEquilibration(ChainWorkflowGenerator):
    def __init__(self, *args, **kwargs):
        sub_wf_components = [
            HemicylindricalPacking,
            PDBCleanup,
            GromacsMinimizationEquilibrationRelaxation,
        ]
        super().__init__(*args, sub_wf_components=sub_wf_components, **kwargs)


AGGREGATE_SHAPE_SUB_WF_DICT = {
    'monolayer': MonolayerPackingAndEquilibration,
    'bilayer': BilayerPackingAndEquilibration,
    'cylinders': CylindricalPackingAndEquilibration,
    'hemicylinders': HemicylindricalPackingAndEquilibration,
}


class SurfactantPackingAndEquilibration(EncapsulatingWorkflowGenerator):
    def __init__(self, *args, **kwargs):
        shape = get_nested_dict_value(kwargs, 'system->surfactant->aggregates->shape')
        assert shape in AGGREGATE_SHAPE_SUB_WF_DICT, "'{}' not in '{}'".format(
            shape, AGGREGATE_SHAPE_SUB_WF_DICT.keys())
        super().__init__(*args, sub_wf=AGGREGATE_SHAPE_SUB_WF_DICT[shape], **kwargs)


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
