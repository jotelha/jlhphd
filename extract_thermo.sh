#! /bin/sh
# Extracts thermo output from lammps log.
# Requires a SINGLE thermo section within the log.
# Does NOT extract multiple thermo sections, i.e.
# subsequent minimization and equilibration.
# 1st and 2nd positional arguments are optional and
# allow to specify input log file and output text file.
LOGFILE="lammps.log"
OUTFILE="thermo.out"
if [ -n "${1}" ] ; then
  LOGFILE="${1}"
  if [ -n "${2}" ] ; then
    OUTFILE="${2}"
  fi
fi
cat "${LOGFILE}" | sed -n '/^Step/,/^Loop time/p' | head -n-1 > "${OUTFILE}"
