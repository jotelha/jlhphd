 #!/bin/bash -x
#MSUB -E
#MSUB -v OMP_NUM_THREADS=1
#MSUB -l nodes=1:ppn=20
#MSUB -l walltime=96:00:00
#MSUB -l pmem=5000mb
#MSUB -l partition=torque
#MSUB -m ae
#MSUB -M johannes.hoermann@imtek.uni-freiburg.de
#MSUB -N lmp_16Mar18
set -e
set -o pipefail

SUFFIX_EXE=""
if [ -z "${OMP_NUM_THREADS}" ]; then
   export OMP_NUM_THREADS=1
fi
# if possible, map tasks by socket, NEMO specific
MAP_BY=socket
if [[ $((10 % $OMP_NUM_THREADS)) != 0 ]]; then
  echo "Allow tasks to distribute threads across different sockets, map by node."
  MAP_BY=node
fi


if [ -n "${MOAB_JOBNAME}" ]; then
  cd "${MOAB_SUBMITDIR}"

  TASK_COUNT=$((${MOAB_PROCCOUNT}/${OMP_NUM_THREADS}))
  MPI_PPN_COUNT=$((${PBS_NUM_PPN}/${OMP_NUM_THREADS}))

  if [[ -n "${OMP_NUM_THREADS}" ]] && [[ "${OMP_NUM_THREADS}" -gt "1" ]] ; then
    SUFFIX_EXE="-sf omp"
  fi
  ## check if $SCRIPT_FLAGS is "set"
  if [ -n "${SCRIPT_FLAGS}" ] ; then
     ## but if positional parameters are already present
     ## we are going to ignore $SCRIPT_FLAGS
     if [ -z "${*}"  ] ; then
        set -- ${SCRIPT_FLAGS}
     fi
  fi

  #echo "${EXECUTABLE} running on ${MOAB_NODECOUNT} and ${MOAB_PROCCOUNT} cores " \
  #  "with ${TASK_COUNT} mpi tasks and ${OMP_NUM_THREADS} omp threads " \
  #  "in ${PBS_O_WORKDIR}"

  printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -

  echo -e "MOAB_JOBID     \t ${MOAB_JOBID}"
  echo -e "MOAB_JOBNAME   \t ${MOAB_JOBNAME}"
  echo -e "MOAB_NODECOUNT \t ${MOAB_NODECOUNT}" # usually empty
  echo -e "PBD_NUM_NODES  \t ${PBS_NUM_NODES}" # reliable
  echo -e "MOAB_PROCCOUNT \t ${MOAB_PROCCOUNT}" # total number of cores
  echo -e "#tasks         \t ${TASK_COUNT}"
  echo -e "PBS_TASKNUMt   \t ${PBS_TASKNUM}" # just 1
  echo -e "OMP_NUM_THREADS\t ${OMP_NUM_THREADS}"
  echo -e "#ppn \t\t ${PBS_NUM_PPN}"
  echo -e "nodes \t\t ${MOAB_NODELIST}"
  echo -e "PBD_0_WORKDIR  \t ${PBS_O_WORKDIR}"
  echo -e "MOAB_SUBMITDIR \t ${MOAB_SUBMITDIR}"
else
  # Standard values
  export OMP_NUM_THREADS=1
  TASK_COUNT=1
fi

# LAMMPS input via command line argument
if [ -n "$1" ]; then
  INFILE="$1"
fi

MPIRUN_OPTIONS="--bind-to core --map-by $MAP_BY:PE=${OMP_NUM_THREADS}"
MPIRUN_OPTIONS="${MPIRUN_OPTIONS} -n ${TASK_COUNT} --report-bindings"

#############################################################
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "ENVIRONMENT"
printenv
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
#############################################################

module purge
module use /work/ws/nemo/fr_lp1029-IMTEK_SIMULATION-0/modulefiles
module load lammps/16Mar18-gnu-5.2-openmpi-2.1

# mpirun ${MPIRUN_OPTIONS} lmp -in lmp_equilibration_nvt.input
# mpirun ${MPIRUN_OPTIONS} lmp -in lmp_equilibration_npt.input
echo "LAMMPS CALLED"
mpirun ${MPIRUN_OPTIONS} lmp -in "${INFILE}" ${SUFFIX_EXE}
echo "LAMMPS FINISHED"
