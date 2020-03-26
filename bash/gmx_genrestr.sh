#!/bin/bash

## sample: generate GROMACS bonds and bontypes restraints

gmx genrestr -f 1_SDS.gro -o 1_SDS.disre.itp -disre
cat 1_SDS.disre.itp | sed -E 's/^(\s*[0-9]+\s+[0-9]+\s*)1.*$/\110/' > 1_SDS.disre_bonds.itp

# convert
# ;   i     j ? label      funct         lo        up1        up2     weight
#     1     2 1     0          1  0.0635971   0.263597     1.2636          1
# to
# ;   i     j funct         lo        up1        up2     k0
#
cat 1_SDS.disre.itp | sed -E 's/^(\s*[0-9]+\s+[0-9]+\s+)([0-9]+\s+[0-9]+\s+[0-9]+)(\s+[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?\s+[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?+\s+[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?\s+)([0-9]+)\s*$/\110\31000/' > 1_SDS.disre_bondtypes.itp