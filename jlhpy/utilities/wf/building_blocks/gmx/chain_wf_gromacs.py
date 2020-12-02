
from jlhpy.utilities.wf.workflow_generator import ChainWorkflowGenerator

from jlhpy.utilities.wf.building_blocks.gmx.sub_wf_gromacs_prep import GromacsPrep
from jlhpy.utilities.wf.building_blocks.gmx.sub_wf_gromacs_prep_wo_box_dimensions import GromacsPrep as GromacsPrepWoBoxDimensions

from jlhpy.utilities.wf.building_blocks.gmx.sub_wf_gromacs_solvate import GromacsSolvate
from jlhpy.utilities.wf.building_blocks.gmx.sub_wf_gromacs_em_solvated import GromacsEnergyMinimizationAfterSolvation

from jlhpy.utilities.wf.building_blocks.gmx.sub_wf_gromacs_nvt import GromacsNVTEquilibration
from jlhpy.utilities.wf.building_blocks.gmx.sub_wf_gromacs_npt import GromacsNPTEquilibration
from jlhpy.utilities.wf.building_blocks.gmx.sub_wf_gromacs_relax import GromacsRelaxation

class GromacsMinimizationEquilibrationRelaxation(ChainWorkflowGenerator):
    """Minimization, equilibration and relaxation with GROMACS chain workflow.

    Concatenates
    - GromacsPrep
    - GromacsSolvate
    - GromacsEnergyMinimizationAfterSolvation
    - GromacsNVTEquilibration
    - GromacsNPTEquilibration
    - GromacsRelaxation
    """

    def __init__(self, *args, **kwargs):
        sub_wf_components = [
            GromacsPrep,
            GromacsSolvate,
            GromacsEnergyMinimizationAfterSolvation,
            GromacsNVTEquilibration,
            GromacsNPTEquilibration,
            GromacsRelaxation,
        ]
        super().__init__(*args, sub_wf_components=sub_wf_components, **kwargs)


class GromacsMinimizationEquilibrationRelaxationNoSolvation(ChainWorkflowGenerator):
    """Minimization, equilibration and relaxation with GROMACS chain workflow.

    Without adjusing box size and without solvation.

    Concatenates
    - GromacsPrepWoBoxDimensions
    - GromacsEnergyMinimizationAfterSolvation
    - GromacsNVTEquilibration
    - GromacsNPTEquilibration
    - GromacsRelaxation
    """

    def __init__(self, *args, **kwargs):
        sub_wf_components = [
            GromacsPrepWoBoxDimensions,
            GromacsEnergyMinimizationAfterSolvation,
            GromacsNVTEquilibration,
            GromacsNPTEquilibration,
            GromacsRelaxation,
        ]
        super().__init__(*args, sub_wf_components=sub_wf_components, **kwargs)