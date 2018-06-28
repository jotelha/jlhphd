#!/bin/bash -x
#MSUB -E
#MSUB -v OMP_NUM_THREADS=1
#MSUB -l nodes=1:ppn=20
#MSUB -l walltime=01:00:00
#MSUB -l pmem=5000mb
#MSUB -l partition=torque
#MSUB -m ae
#MSUB -M johannes.hoermann@imtek.uni-freiburg.de
#MSUB -N lmp_16Mar18

#
# call with
#   msub -v INFILE='INFILENAME' lmp_16Mar18.sh

#   msub -N '${jname}' -l nodes=$nn:ppn=$GLOBAL_PPN \
#        -l walltime=${wt_string} -v OMP_NUM_THREADS=$nt \
#        -v INFILE='${in_file}' ${run_script}"

# OMP_NUM_THREADS is number of OpenMP threads per MPI task
# Attention: MPI tasks, but OpenMP threads!

# example:
#    msub ... -v OMP_NUM_THREADS=2 \
#      -l nodes=4:ppn=20,walltime=10:00:00 gmx_2016.3_node_intel_fftw3.sh
# results in
#    PBS_NUM_NODES    = 4
#    PBS_NUM_PPN      = 20
#    MOAB_PROCCOUNT   = 4x20 = 80
#    OMP_NUM_THREADS  = 2
#    TASK_COUNT       = 80 / 2 = 40
#    PPN_COUNT        = 20 / 2 = 10
# and runs 40 MPI tasks on 4 nodes, each task with 2 OpenMP threads bound to 2 cores
# in this case, mpirun must be called with
#    mpirun ... -n 40 -ppn 10 --bind-to core --map-by socket:PE=2 --report-bindings ...
# for in total 40 MPI tasks, 10 by node, bound to 2 cores each

# ATTENTION: NEMO single node architecure (lscpu)
# Architecture:          x86_64
# CPU op-mode(s):        32-bit, 64-bit
# Byte Order:            Little Endian
# CPU(s):                40
# On-line CPU(s) list:   0-39
# Thread(s) per core:    2
# Core(s) per socket:    10
# Socket(s):             2
# NUMA node(s):          2
# Vendor ID:             GenuineIntel
# CPU family:            6
# Model:                 79
# Model name:            Intel(R) Xeon(R) CPU E5-2630 v4 @ 2.20GHz
# Stepping:              1
# CPU MHz:               1200.203
# BogoMIPS:              4395.36
# Virtualization:        VT-x
# L1d cache:             32K
# L1i cache:             32K
# L2 cache:              256K
# L3 cache:              25600K
# NUMA node0 CPU(s):     0-9,20-29
# NUMA node1 CPU(s):     10-19,30-39

# essence: 2 sockets, 10 cores per socket
# conclusion: cannot map by socket if 10 mod $OMP_NUM_THREADS != 0
# workaround:

# if possible, map tasks by socket
MAP_BY=socket
if [[ $((10 % $OMP_NUM_THREADS)) != 0 ]]; then
  echo "Allow tasks to distribute threads across different sockets, map by node."
  MAP_BY=node
fi

# BUT: (http://www.gromacs.org/Documentation/Acceleration_and_parallelization)
# Note that for good performance on multi-socket servers, groups of OpenMP
# threads belonging to an MPI process/thread-MPI thread should run on the
# same CPU/socket. This requires that the number of processes is a multiple
# of the number of CPUs/sockets in the respective machine and the number of
# cores per CPU is divisible by the number of threads per process.
# E.g. on a dual 6-core machine N=6, M=2 or N=3, M=4 should run more
# efficiently than N=4 M=3.

# question: why does “lscpu” list 40 CPUS?
# in connection with that maybe GROMACS note:
# Note: 40 CPUs configured, but only 20 were detected to be online.
#      X86 Hyperthreading is likely disabled; enable it for better performance.

# Here a discussion on Gromacs and Intel's Hyperthreading:
# http://gromacs.org_gmx-users.maillist.sys.kth.narkive.com/gtR56mgo/hyper-threading-gromacs-5-0-1
# conclusion: HT not advisible

## check if $SCRIPT_FLAGS is "set"
if [ -n "${SCRIPT_FLAGS}" ] ; then
   ## but if positional parameters are already present
   ## we are going to ignore $SCRIPT_FLAGS
   if [ -z "${*}"  ] ; then
      set -- ${SCRIPT_FLAGS}
   fi
fi

# parse arguments
args=$(getopt -n "$0" -l "dry-run,no-mpi,forward-mpi-options" -o "d1f" -- "$@")
eval set -- "$args"
DRY_RUN=false
NO_MPI=false

echo "Got $# arguments: " $@
while true; do
  case "$1" in
    -d | --dry-run ) DRY_RUN=true; shift ;;
    -1 | --no-mpi ) NO_MPI=true; shift ;;
    -f | --forward-mpi-options ) FORWARD_MPI_OPTIONS=true; shift ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done

# only defined for local use, not used by other executables:
TASK_COUNT=$((${MOAB_PROCCOUNT}/${OMP_NUM_THREADS}))
MPI_PPN_COUNT=$((${PBS_NUM_PPN}/${OMP_NUM_THREADS}))
MPIRUN_OPTIONS="--bind-to core --map-by $MAP_BY:PE=${OMP_NUM_THREADS}"
MPIRUN_OPTIONS="${MPIRUN_OPTIONS} -n ${TASK_COUNT} --report-bindings"
# source: https://software.intel.com/en-us/get-started-with-mpi-for-linux
# mpirun -n <# of processes> -ppn <# of processes per node> ./myprog

# Source: https://www.bwhpc-c5.de/wiki/index.php/Batch_Jobs#Multithreaded_.2B_MPI_parallel_Programs
#  1.3.2.4 Multithreaded + MPI parallel Programs
# Multithreaded + MPI parallel programs operate faster than serial programs on
# multi CPUs with multiple cores. All threads of one process share resources
# such as memory. On the contrary MPI tasks do not share memory but can be
# spawned over different nodes.
# Multiple MPI tasks using OpenMPI must be launched by the MPI parallel program
# mpirun. For multithreaded programs based on Open Multi-Processing (OpenMP)
# number of threads are defined by the environment variable OMP_NUM_THREADS.
# By default this variable is set to 1 (OMP_NUM_THREADS=1).

# export KMP_AFFINITY=scatter
export MOAB_NODECOUNT=$PBS_NUM_NODES

# pmem is required memory per task

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

#echo -e "out dir \t ${DIRECTORY}"

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
################################################################################

module use /work/ws/nemo/fr_lp1029-IMTEK_SIMULATION-0/modulefiles
module purge
module load lammps/16Mar18-gnu-5.2-openmpi-2.1

cd ${MOAB_SUBMITDIR}
export EXECUTABLE="lmp -in ${INFILE} -sf omp"
echo "${EXECUTABLE} running on ${MOAB_NODECOUNT} and ${MOAB_PROCCOUNT} cores " \
  "with ${TASK_COUNT} tasks and ${OMP_NUM_THREADS} threads in ${PBS_O_WORKDIR}"

# http://lammps.sandia.gov/doc/accelerate_omp.html
# Run with the USER-OMP package from the command line:
# The mpirun or mpiexec command sets the total number of MPI tasks used by
# LAMMPS (one or multiple per compute node) and the number of MPI tasks used per
# node. E.g. the mpirun command in MPICH does this via its -np and -ppn switches.
# Ditto for OpenMPI via -np and -npernode.
# You need to choose how many OpenMP threads per MPI task will be used by the
# USER-OMP package. Note that the product of MPI tasks * threads/task should not
# exceed the physical number of cores (on a node), otherwise performance will
# suffer.
# As in the lines above, use the “-sf omp” command-line switch, which will
# automatically append “omp” to styles that support it. The “-sf omp” switch
# also issues a default package omp 0 command, which will set the number of
# threads per MPI task via the OMP_NUM_THREADS environment variable.

# You can also use the “-pk omp Nt” command-line switch, to explicitly set
# Nt = # of OpenMP threads per MPI task to use, as well as additional options.
# Its syntax is the same as the package omp command whose doc page gives
# details, including the default values used if it is not specified. It also
# gives more details on how to set the number of threads via the OMP_NUM_THREADS
#  environment variable.
if [ "$NO_MPI" == true ]; then
  startexe="${EXECUTABLE}"
elif [ "$FORWARD_MPI_OPTIONS" == true ]; then
  export MPIRUN_OPTIONS
  startexe="${EXECUTABLE} --mpirun-options '$MPIRUN_OPTIONS'"
else
  startexe="mpirun ${MPIRUN_OPTIONS} ${EXECUTABLE}"
fi
startexe=$(echo $startexe | tr -s " ")

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "Run..."

echo $startexe
# eval takes its arguments, concatenates them separated by spaces, and executes
# the resulting string as Bash code in the current execution environment.
if [ ! "$DRY_RUN" == true ]; then
  eval $startexe
fi

echo "Done!"

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
