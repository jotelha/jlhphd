#!/bin/bash
#
# Utilizes c2p and vmd topotools to convert an atomey .cfg file to
# a LAMMPS data file of atom style "full"
set -e

if [ -n "$1" ] ; then
  INFILE="$1"
  ext=${INFILE##*.}
  BASENAME=$(basename $INFILE .$ext)
  OUTFILE="${BASENAME}.lammps"
  PDBFILE="${BASENAME}.pdb"

  if [ -n "$2" ] ; then
    OUTFILE="$2"
    if [ -n "$3" ] ; then
      PDBFILE="$3"
    fi
  fi
else
  echo "No input file provided!"
  exit 1
fi

module load vmd/1.9.3-text
module load mdtools # contains c2p

echo "Converting from '${INFILE}' via '${PDBFILE}' to '${OUTFILE}'..."

c2p "${INFILE}" "${PDBFILE}"
echo "pdb2lmp ${PDBFILE} ${OUTFILE}" | vmd -eofexit -e pdb2lmp.tcl

