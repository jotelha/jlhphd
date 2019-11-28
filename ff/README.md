# Force fields

## Content:


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

## Atom type legend

```
# SDS - specific
#       1      1.008  # HAL2
#       2      1.008  # HAL3
#       3     12.011  # CTL2
#       4     12.011  # CTL3
#       5    15.9994  # OSL
#       6    15.9994  # O2L
#       7      32.06  # SL
#       8      1.008  # HT
#       9    15.9994  # OT
#      10   22.98977  # SOD
#      11   196.9665  # AU

# CTAB - specific
#
#        1      1.008  # HL
#        2      1.008  # HAL2
#        3      1.008  # HAL3
#        4     12.011  # CTL2
#        5     12.011  # CTL3
#        6     12.011  # CTL5
#        7     14.007  # NTL
#        8      1.008  # HT
#        9    15.9994  # OT
#       10     79.904  # BR
#       11   196.9665  # AU
```
