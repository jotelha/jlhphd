module purge
module load mdtools fireworks

filepad.py --action pull --file interface.lammps --metadata-file interface.yaml -- interface/AU/111/184Ang/cube/CTAB/653/cylinders/equilibration_npt/201908101456/default.lammps
filepad.py --action pull --file indenter.lammps --metadata-file indenter.yaml -- indenter/AU/111/25Ang/100ns.lammps

strip_comments.py interface.lammps interface_stripped.lammps

module purge
module load mdtools/12Mar19-python-2.7
extract_bb.py interface_stripped.lammps

# segmentation fault on NEMO login node, run on compute node:
module purge
module load vmd

vmd -eofexit <<-EOF
  package require jlhvmd
  jlh set distance 50.0 indenterInfile indenter.lammps interfaceInfile interface.lammps outputPrefix system
  jlh use ctab
  jlh read bb bb.yaml
  jlh insert
EOF
