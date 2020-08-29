# -*- coding: utf-8 -*-

from jlhpy.utilities.wf.workflow_generator import WorkflowGenerator, ChainWorkflowGenerator
from jlhpy.utilities.wf.flat_packing.sub_wf_005_format_conversion import FormatConversionWorkflowGenerator
from jlhpy.utilities.wf.flat_packing.sub_wf_010_flat_substrate_measures import FlatSubstrateMeasuresWorkflowGenerator


class FlatSubstratePackingChainWorkflowGenerator(ChainWorkflowGenerator):
    """Flat substrate packing with PACKMOL sub workflow.

    Concatenates
    - SurfactantMoleculeMeasuresWorkflowGenerator
    - FlatSubstrateMeasures
    - PlanarSurfactantPackingWorkflowGenerator
    """

    def __init__(self, *args, **kwargs):
        sub_wf_components = [
            FormatConversionWorkflowGenerator(*args, **kwargs),
            FlatSubstrateMeasuresWorkflowGenerator(*args, **kwargs),
        ]
        sub_wf_name = 'FlatSubstratePacking'
        if 'wf_name_prefix' not in kwargs:
            kwargs['wf_name_prefix'] = sub_wf_name
        else:
            kwargs['wf_name_prefix'] = ':'.join((kwargs['wf_name_prefix'], sub_wf_name))
        super().__init__(sub_wf_components, *args, **kwargs)
