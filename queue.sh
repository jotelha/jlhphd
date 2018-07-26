#!/bin/bash -x
set -e
LMP_JOB_SCRIPT=lmp_16Mar18.sh
job_prefix="1_CTAB_on_111_AU_21x12x2"

# EQUILIBRATION
jidfile="msub_lmp_equilibration.jid"
# RUN_AFTER="gmx grompp -f ../template/nvt.mdp -c em.gro -r em.gro"
# RUN_AFTER="${RUN_AFTER} -p ${surfactant}.top -o nvt.tpr"
# RUN_AFTER="${RUN_AFTER} 2>&1 | tee gmx_grompp_nvt.log"
echo "### Queueing lmp equilibration ###"
# echo "RUN_AFTER = '${RUN_AFTER}'"
jid=$(msub -N "${job_prefix}_equilibration" \
      -v INFILE='lmp_equilibration.input' \
      -v OMP_NUM_THREADS=1 -l nodes=1:ppn=20,walltime=01:00:00 \
      ${LMP_JOB_SCRIPT} 2>&1 | sed '/^$/d')
if [[ "${jid}" =~ "ERROR" ]] ; then
  echo "    -> submission failed: ${jid}" ; exit 1
else
  echo "    -> submitted job number = ${jid}"
fi
echo "${jid}"  > "${jidfile}"
equilibration_jid=${jid}

# NPT PRODUCTION
jidfile="msub_lmp_npt_production.jid"
echo "### Queueing lmp npt production ###"
# echo "RUN_AFTER = '${RUN_AFTER}'"
jid=$(msub -N "${job_prefix}_npt_production" \
      -v INFILE='lmp_1ns_npt_with_restarts.input' \
      -v OMP_NUM_THREADS=1 -l nodes=1:ppn=20,walltime=02:00:00 \
      -l depend=afterok:${equilibration_jid} \
      ${LMP_JOB_SCRIPT} 2>&1 | sed '/^$/d')
if [[ "${jid}" =~ "ERROR" ]] ; then
  echo "    -> submission failed: ${jid}" ; exit 1
else
  echo "    -> submitted job number = ${jid}"
fi
echo "${jid}" > "${jidfile}"

# NVE PRODUCTION
jidfile="msub_lmp_nve_production.jid"
echo "### Queueing lmp nve production ###"
# echo "RUN_AFTER = '${RUN_AFTER}'"
jid=$(msub -N "${job_prefix}_nve_production" \
      -v INFILE='lmp_1ns_nve_with_restarts.input' \
      -v OMP_NUM_THREADS=2 -l nodes=1:ppn=20,walltime=02:00:00 \
      -l depend=afterok:${equilibration_jid} \
      ${LMP_JOB_SCRIPT} 2>&1 | sed '/^$/d')
if [[ "${jid}" =~ "ERROR" ]] ; then
  echo "    -> submission failed: ${jid}" ; exit 1
else
  echo "    -> submitted job number = ${jid}"
fi
echo "${jid}" > "${jidfile}"
