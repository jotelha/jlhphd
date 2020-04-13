# -*- coding: utf-8 -*-

from jlhpy.utilities.wf.workflow_generator import SubWorkflowGenerator
from jlhpy.utilities.wf.packing.sub_wf_indenter_bounding_sphere import IndenterBoundingSphereSubWorkflowGenerator
from jlhpy.utilities.wf.packing.sub_wf_surfactant_molecule_measures import SurfactantMoleculeMeasuresSubWorkflowGenerator
from jlhpy.utilities.wf.packing.sub_wf_packing_constraint_spheres import PackingConstraintSpheresSubWorkflowGenerator
from jlhpy.utilities.wf.packing.sub_wf_spherical_surfactant_packing import SphericalSurfactantPackingSubWorkflowGenerator

class SphericalIndenterPassivationSubWorkflowGenerator(SubWorkflowGenerator):
    """Spherical surfactant packing with PACKMOL sub workflow.

    Concatenates
    - IndenterBoundingSphereSubWorkflowGenerator
    - SurfactantMoleculeMeasuresSubWorkflowGenerator
    - PackingConstraintSpheresSubWorkflowGenerator
    - SphericalSurfactantPackingWorkflowGenerator
    """

    def __init__(self, *args, **kwargs):
        self.bounding_sphere_wfg = IndenterBoundingSphereSubWorkflowGenerator(
            *args, **kwargs)
        self.surfactant_measures_wfg = \
            SurfactantMoleculeMeasuresSubWorkflowGenerator(*args, **kwargs)
        self.constraint_spheres_wfg = PackingConstraintSpheresSubWorkflowGenerator(
            *args, **kwargs)
        self.spherical_packing_wfg = SphericalSurfactantPackingSubWorkflowGenerator(
            *args, **kwargs)

        if 'wf_name_prefix' not in kwargs:
            kwargs['wf_name_prefix'] = 'packing parameters sub-workflow'
        super().__init__(*args, **kwargs)

    def push_infiles(self, fp):
        """fp: FilePad"""
        self.bounding_sphere_wfg.push_infiles(fp)
        self.surfactant_measures_wfg.push_infiles(fp)
        # TODO: self.constraint_spheres_wfg.push_infiles(fp)

    def main(self, fws_root=[]):
        # fws_pull, fws_pull_leaf, _ = self.pull()
        fws_bs, fws_bs_leaf, fws_bs_root = \
            self.bounding_sphere_wfg.get_as_root(fws_root)
        fws_sm, fws_sm_leaf, fws_sm_root = \
            self.surfactant_measures_wfg.get_as_root(fws_bs_leaf)
        fws_cs, fws_cs_leaf, fws_cs_root = \
            self.constraint_spheres_wfg.get_as_root(fws_sm_leaf)
        fws_sp, fws_sp_leaf, fws_sp_root = \
            self.spherical_packing_wfg.get_as_root(fws_cs_leaf)

        return [*fws_bs, *fws_sm, *fws_cs, *fws_sp], fws_cs_leaf, fws_sp_root
