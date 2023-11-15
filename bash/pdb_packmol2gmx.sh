#!/bin/bash -x
# prepares packmol output for gromacs

if [ -n "$1" ]; then
  BASENAME=${1%_packmol.pdb} # removes pdb ending if passed
  BASENAME=${BASENAME%.pdb} # removes pdb ending if passed
else
  echo "No input file given!"
  exit 1
fi

module load mdtools

# Remove chain id
pdb_chain.py  "${BASENAME}_packmol.pdb" > "${BASENAME}_nochainid.pdb"

# extracts surface and everything else into separate parts
# surface residue must be 1st in file
pdb_rslice.py :1 "${BASENAME}_nochainid.pdb" > "${BASENAME}_substrate_only.pdb"
pdb_rslice.py 2: "${BASENAME}_nochainid.pdb" > "${BASENAME}_surfactant_only.pdb"


# assign unique residue ids
# Gromacs requires number of molecules in residue to match rtp entry.
# In our modified gromacs charmm36.ff, the SURF residue consists of 1 AU atom
pdb_reres_by_atom_9999.py "${BASENAME}_substrate_only.pdb" \
  -resid 1 > "${BASENAME}_substrate_only_reres.pdb"

# merges two pdb just by concatenating
head -n -1 "${BASENAME}_substrate_only_reres.pdb" \
  > "${BASENAME}_concatenated.pdb"  # last line contains END statement
tail -n +6 "${BASENAME}_surfactant_only.pdb" \
  >> "${BASENAME}_concatenated.pdb" # first 5 lines are packmol-generated header

# ATTENTION: pdb_reres writes residue numbers > 9999 without complains,
# however thereby produces non-standard PDB format
pdb_reres_9999.py "${BASENAME}_concatenated.pdb" -resid 1 > "${BASENAME}.pdb"
