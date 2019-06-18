#!/bin/bash
#
# Johannes Hoermann, johannes.hoermann@imtek.uni-freiburg.de, 2019
#
# msub_options.sh
# ---------------
# prints environment information and compiles a favorable options string for
# mpirun based upon MOAB job resources into $MPIRUN_OPTIONS

# if possible, map tasks by socket
if [ -z "${OMP_NUM_THREADS}" ]; then
  export OMP_NUM_THREADS=1
fi

MAP_BY=socket
if [[ $((10 % ${OMP_NUM_THREADS})) != 0 ]]; then
  echo "Allow tasks to distribute threads across different sockets, map by node."
  MAP_BY=node
fi

TASK_COUNT=$((${MOAB_PROCCOUNT}/${OMP_NUM_THREADS}))
MPI_PPN_COUNT=$((${PBS_NUM_PPN}/${OMP_NUM_THREADS}))
export MPIRUN_OPTIONS="--bind-to core --map-by $MAP_BY:PE=${OMP_NUM_THREADS}"
export MPIRUN_OPTIONS="${MPIRUN_OPTIONS} -n ${TASK_COUNT} --report-bindings"

export MOAB_NODECOUNT=${PBS_NUM_NODES}

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
printenv

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -

echo -e "job id \t\t ${MOAB_JOBID}"
echo -e "job name \t ${MOAB_JOBNAME}"
echo -e "#nodes \t\t ${MOAB_NODECOUNT}" # usually empty
echo -e "#nodes (pbs)\t ${PBS_NUM_NODES}" # reliable
echo -e "#cores \t\t ${MOAB_PROCCOUNT}" # total number of cores
echo -e "#tasks \t\t ${TASK_COUNT}"
echo -e "#tasks (pbs)\t ${PBS_TASKNUM}" # just 1
echo -e "#threads \t ${OMP_NUM_THREADS}"
echo -e "#ppn \t\t ${PBS_NUM_PPN}"
echo -e "nodes \t\t ${MOAB_NODELIST}"
echo -e "pbs dir \t ${PBS_O_WORKDIR}"

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
