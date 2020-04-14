import os.path

# TODO: looks through modules and replace hard-coded surfactant-specific names
PDB_SUBDIR     = 'pdb'
SURFACTANT_PDB = '1_SDS.pdb'
COUNTERION_PDB = '1_NA.pdb'

DAT_SUBDIR      = 'dat'
INDENTER_SUBDIR = os.path.join(DAT_SUBDIR, 'indenter')
INDENTER_PDB    = 'AU_111_r_25.pdb'

PACKMOL_SUBDIR  = 'packmol'
PACKMOL_SPHERES_TEMPLATE = 'sphere.inp.template'
