## Creation protocol:

Initial files

* CTAB_in_H2O_on_AU_coeff.input
* SDS_in_H2O_on_AU_coeff.input

created with `../lmp_input/lmp_split_datafile.input` and according pure CHARMM
system. Atom type files

* CTAB_in_H2O_on_AU_masses.input
* SDS_in_H2O_on_AU_masses.input

created manually by copying `Masses` section from LAMMPS data file and
prefixing each line with `mass`. Hybrid sytle files created with commands

```bash
to_hybrid.py CTAB_in_H2O_on_AU_coeff.input \
  CTAB_in_H2O_on_AU_coeff_hybrid_lj_charmmfsw_coul_long.input
to_hybrid.py --pair-style 'lj/charmmfsw/coul/charmmfsh' \
  CTAB_in_H2O_on_AU_coeff.input \
  CTAB_in_H2O_on_AU_coeff_hybrid_lj_charmmfsw_coul_charmmfsh.input
```
and analogous for SDS. Partial sets

* *_nonbonded.input
* *_bonded.input

created by stripping the original files off the unwanted sections.

`to_hybrid.py` is part of https://github.com/jotelha/mdtools-jlh
