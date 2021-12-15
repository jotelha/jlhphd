# -*- coding: utf-8 -*-


from jlhpy.utilities.wf.workflow_generator import ChainWorkflowGenerator
from jlhpy.utilities.wf.probe_on_substrate.sub_wf_040_lammps_minimization import LAMMPSMinimization
from jlhpy.utilities.wf.probe_on_substrate.sub_wf_050_lammps_equilibration_nvt import LAMMPSEquilibrationNVT
from jlhpy.utilities.wf.probe_on_substrate.sub_wf_060_lammps_equilibration_npt import LAMMPSEquilibrationNPT
from jlhpy.utilities.wf.probe_on_substrate.sub_wf_070_lammps_equilibration_dpd import LAMMPSEquilibrationDPD
from jlhpy.utilities.wf.probe_on_substrate.sub_wf_210_lammps_probe_lateral_sliding import LAMMPSProbeLateralSliding
from jlhpy.utilities.wf.probe_on_substrate.sub_wf_120_probe_analysis import ProbeAnalysis3D

class ProbeOnSubstrateLateralSliding(ChainWorkflowGenerator):
    """Run lateral sliding production with LAMMPS.

    Concatenates
    - LAMMPSProbeLateralSliding
    - ProbeAnalysis3D
    """

    def __init__(self, *args, **kwargs):
        sub_wf_components = [
            LAMMPSProbeLateralSliding,
            ProbeAnalysis3D,
        ]
        super().__init__(*args, sub_wf_components=sub_wf_components, **kwargs)



class ProbeOnSubstrateMinimizationEquilibrationLateralSliding(ChainWorkflowGenerator):
    """Minimize, equilibrate, run lateral sliding production with LAMMPS.

    Concatenates
    - LAMMPSMinimization
    - LAMMPSEquilibrationNVT
    - LAMMPSEquilibrationNPT
    - LAMMPSEquilibrationDPD
    - LAMMPSProbeLateralSliding
    - ProbeAnalysis3D
    """

    def __init__(self, *args, **kwargs):
        sub_wf_components = [
            LAMMPSMinimization,
            LAMMPSEquilibrationNVT,
            LAMMPSEquilibrationNPT,
            LAMMPSEquilibrationDPD,
            LAMMPSProbeLateralSliding,
            ProbeAnalysis3D,
        ]
        super().__init__(*args, sub_wf_components=sub_wf_components, **kwargs)