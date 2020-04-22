import os.path

# TODO: looks through modules and replace hard-coded surfactant-specific names

GMX_MDP_SUBDIR = os.path.join('gmx_input', 'mdp')
GMX_EM_MDP = 'em.mdp'
GMX_PULL_MDP_TEMPLATE = 'pull.mdp.template'
GMX_EM_SOLVATED_MDP = 'em_solvated.mdp'

GMX_TOP_SUBDIR = os.path.join('gmx_input', 'top')
GMX_PULL_TOP_TEMPLATE = 'sys.top.template'

PDB_SUBDIR     = 'pdb'
SURFACTANT_PDB = '1_SDS.pdb'
COUNTERION_PDB = '1_NA.pdb'

DAT_SUBDIR      = 'dat'
INDENTER_SUBDIR = os.path.join(DAT_SUBDIR, 'indenter')
INDENTER_PDB    = 'AU_111_r_25.pdb'

PACKMOL_SUBDIR  = 'packmol'
PACKMOL_SPHERES_TEMPLATE = 'sphere.inp.template'

# visualization-related

PML_SUBDIR = 'pymol'
PML_MOVIE_TEMPLATE = 'movie.pml.template'

BASH_SCRIPT_SUBDIR = 'bash'
BASH_RENUMBER_PNG = 'renumber_png.sh'
