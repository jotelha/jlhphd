#!/bin/bash -x
#
# Johannes Hoermann, johannes.hoermann@imtek.uni-freiburg.de, 2019
#
# lmp.sh
# ------
# meant to load correct modules from
# expects file containing LAMMPS command as single argument
# and $MPIRUN_OPTIONS set by msub_options.sh
#
# Argument $1 (such as the sample lmp.heat.sh in this folder) must not contain
# any additional lines or comments
#
echo "##############"
echo "### lmp.sh ###"
echo "##############"
module purge
module use /work/ws/nemo/fr_lp1029-IMTEK_SIMULATION-0/modulefiles
module load lammps/08May19-git-master-gnu-7.3-openmpi-3.1-colvars-08May19
echo "##############"
printenv
echo "##############"
cmd=$(<$1)
echo "mpirun ${MPIRUN_OPTIONS} ${cmd}"
eval "mpirun ${MPIRUN_OPTIONS} ${cmd}"
echo "##############
