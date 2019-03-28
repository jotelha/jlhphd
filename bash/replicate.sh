#!/bin/bash -x
#
# replicates some AU unit cell .gro and converts to .pdb
#
# sample call:
#
#  ./replicate.sh 21 12 1 111
#
# here, all output files will be prefixed AU_111_21x12x1
# set -e

X=1
Y=1
Z=1
PLANE="111"
if [ -n "$1" ]; then
  X=$1
  Y=$X
  Z=$X
  if [ -n "$2" ]; then
    Y=$2
    Z=$Y
    if [ -n "$3" ]; then
      Z=$3
      if [ -n "$4" ]; then
        PLANE="$4"
      fi
    fi
  fi
fi

BASENAME="AU_${PLANE}_${X}x${Y}x${Z}"


module load mdtools
module load gromacs/2018.1-gnu-5.2

echo "Building surface ${BASENAME}..."
# Multiply .gro unit cell into .pdb
gmx genconf -f au_cell_P1_${PLANE}.gro -o ${BASENAME}.pdb -nbox $X $Y $Z -norenumber

# "Standardize" dirty .pdb format
pdb_tidy.py ${BASENAME}.pdb > ${BASENAME}_tidy.pdb

# Assign indidividual residue numbers for each atom
pdb_reres_by_atom.py ${BASENAME}_tidy.pdb -resid 1 > ${BASENAME}_reres.pdb
# When called from within python with subprocess.run, this script throws
  #   # First argument is always file name
  #   # If it is an option (or no args), assume reading from input stream
  #   if not arg_list or arg_list[0][0] == '-':
  #       if not sys.stdin.isatty():
  #           pdbfh = sys.stdin
  #       else:
  #           sys.stderr.write(USAGE)
  #           sys.exit(1)
  #   else:
  #       if not sys.stdin.isatty():
  #           sys.stderr.write('Error: multiple sources of input' + '\n')
  #           sys.exit(1)
  #       pdbfh = open(arg_list[0])
  #       arg_list = arg_list[1:]
# thus it has been commented out

exit 0