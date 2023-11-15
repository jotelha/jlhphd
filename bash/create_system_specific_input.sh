#!/bin/bash
# infer own system name
SYSTEM_SPECIFIC_INPUT_FILE="system_specific.input"
SYSTEM_NAME=$( basename $(pwd) )
# create one-line lammps include file defining the system's name
echo "variable baseName string ${SYSTEM_NAME}" > ${SYSTEM_SPECIFIC_INPUT_FILE}
