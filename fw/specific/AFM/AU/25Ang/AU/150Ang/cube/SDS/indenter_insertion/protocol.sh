#!/bin/bash
module purge
module load MDTools/jlh-25Jan19
module load FireWorks/1.8.7
filepad.py --action pull --verbose --file indenter.lammps --metadata-file indenter.yaml indenter/AU/111/50Ang/100ns.lammps
filepad.py --action pull --verbose --file interface.lammps --metadata-file interface.yaml interface/SDS/646/AU/111/51x30x21/hemicylinders/equilibrated.lammps

module purge
module load MDTools/jlh-25Jan19-python-2.7

strip_comments.py interface.lammps interface_stripped.lammps
extract_bb.py     interface_stripped.lammps

module purge
module load VMD

vmd -eofexit <<-EOF
  package require jlhvmd
  jlh set distance 30.0 indenterInfile indenter.lammps interfaceInfile interface.lammps outputPrefix system
  jlh use sds
  jlh read bb bb.yaml
  jlh insert
EOF

# vmd -eofexit "package require jlhvmd; jlh set distance 30.0 indenterInfile indenter.lammps interfaceInfile interface.lammps outputPrefix system; jlh use sds; jlh read bb bb.yaml; jlh insert"
