import os.path
# python package-internal paths and file names are to be defined here

# TODO: looks through modules and replace hard-coded surfactant-specific names

# GROMACS-related
GMX_MDP_SUBDIR = os.path.join('gmx_input', 'mdp')
GMX_EM_MDP = 'em.mdp'
GMX_PULL_MDP_TEMPLATE = 'pull.mdp.template'
GMX_EM_SOLVATED_MDP = 'em_solvated.mdp'
GMX_NVT_MDP = 'nvt.mdp'
GMX_NPT_MDP = 'npt.mdp'
GMX_RELAX_MDP = 'relax.mdp'

GMX_TOP_SUBDIR = os.path.join('gmx_input', 'top')
GMX_PULL_TOP_TEMPLATE = 'sys.top.template'

# LAMMPS-related
LMP_INPUT_SUBDIR = 'lmp_input'
LMP_INPUT_TEMPLATE_SUBDIR = os.path.join(LMP_INPUT_SUBDIR, 'template')
LMP_CONVERT_XYZ_INPUT_TEMPLATE = 'lmp_convert_xyz.input.template'
LMP_HEADER_INPUT_TEMPLATE = 'lmp_header.input.template'
LMP_MINIMIZATION_INPUT_TEMPLATE = 'lmp_minimization.input.template'


LMP_FF_SUBDIR = 'ff'
LMP_MASS_INPUT = 'SDS_in_H2O_on_AU_masses.input'
LMP_COEFF_INPUT = 'SDS_in_H2O_on_AU_masses.input'
LMP_EAM_ALLOY = 'Au-Grochola-JCP05-units-real.eam.alloy'

PDB_SUBDIR     = 'pdb'
SURFACTANT_PDB_PATTERN = '1_{name:s}.pdb'
COUNTERION_PDB_PATTERN = '1_{name:s}.pdb'

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
