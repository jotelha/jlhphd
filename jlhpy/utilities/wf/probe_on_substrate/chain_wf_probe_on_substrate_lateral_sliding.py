# -*- coding: utf-8 -*-
from jlhpy.utilities.wf.workflow_generator import ChainWorkflowGenerator

from jlhpy.utilities.wf.probe_on_substrate.sub_wf_170_wrap_join_datafile import WrapJoinDataFile
from jlhpy.utilities.wf.probe_on_substrate.sub_wf_190_lammps_equilibration_dpd import LAMMPSEquilibrationDPD
from jlhpy.utilities.wf.probe_on_substrate.sub_wf_210_lammps_probe_lateral_sliding import LAMMPSProbeLateralSliding
from jlhpy.utilities.wf.probe_on_substrate.sub_wf_120_probe_analysis import ProbeAnalysis3D


class WrapJoinAndDPDEquilibration(ChainWorkflowGenerator):
    """Concatenates
        - WrapJoinDataFile
        - LAMMPSEquilibrationDPD
        """

    def __init__(self, *args, **kwargs):
        sub_wf_components = [
            WrapJoinDataFile,
            LAMMPSEquilibrationDPD,
        ]
        super().__init__(*args, sub_wf_components=sub_wf_components, **kwargs)


class ProbeOnSubstrateWrapJoinEquilibrationAndLateralSliding(ChainWorkflowGenerator):
    """Run lateral sliding production with LAMMPS.

    Concatenates
    - LAMMPSProbeLateralSliding
    - LAMMPSEquilibrationDPD
    - ProbeAnalysis3D
    """

    def __init__(self, *args, **kwargs):
        sub_wf_components = [
            WrapJoinDataFile,
            LAMMPSEquilibrationDPD,
            LAMMPSProbeLateralSliding,
            ProbeAnalysis3D,
        ]
        super().__init__(*args, sub_wf_components=sub_wf_components, **kwargs)