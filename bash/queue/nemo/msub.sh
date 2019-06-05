#!/bin/bash
#MSUB -v OMP_NUM_THREADS=1
#MSUB -N lammps
#MSUB -l nodes=1:ppn=20
#MSUB -l walltime=06:00:00
#MSUB -l pmem=5000mb

# Johannes Hoermann, johannes.hoermann@imtek.uni-freiburg.de, 2019
#
# Sample job file for NEMO
# ------------------------
# queue with some command like
#
#   CMD_FILE=lmp.heat.sh msub -q express -l walltime=00:15:00 -v CMD_FILE msub.sh
#
# requires
#
#   msub_options.sh
#   lmp.sh
#   lmp.heat.sh
#
# in submit directory

# if possible, map tasks by socket
cd ${MOAB_SUBMITDIR}

source msub_options.sh
bash lmp.sh ${CMD_FILE}
# the mpirun command must use ${MPIRUN_OPTIONS}
