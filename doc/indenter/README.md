# Prepare lammps input from atomeye .cfg

  module load vmd/1.9.3-text
  module load mdtools

  c2p indenter.cfg 
  rlwrap vmd

  vmd> mol new indenter.cfg.pdb
  vmd> package require topotools
  vmd> topo writelammpsdata indenter.lammps

or, just in two commands

  c2p 15Ang_amorph.cfg 15Ang_amorph.pdb
  echo "pdb2lmp 15Ang_amorph.pdb 15Ang_amorph.lammps" | vmd -eofexit -e pdb2lmp.tcl 2>&1 | tee 15Ang_amorph.vmd.log

with the tcl script `pdb2lmp.tcl`

    proc pdb2lmp { infile outfile } {
      mol new $infile
      package require topotools
      topo writelammpsdata $outfile
    }

All summarized in one command, cfg2lmp

```
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
```

In order to execute for a batch of .cfg files, use

    ls -1 *.cfg | xargs -n 1 cfg2lmp.sh 
