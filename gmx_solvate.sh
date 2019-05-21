#!/bin/bash -x
#{{header}}
set -e

#module load gromacs/2018.1-gnu-5.2

system={{system_name}}
surfactant={{surfactant}}
water_model="tip3p"
force_field="charmm36"

cation="NA"
anion="BR"
ncation={{ncation|int}}
nanion={{nanion|int}}

bwidth={{"%.4f"|format(box[0])}}
bheight={{"%.4f"|format(box[1])}}
bdepth={{"%.4f"|format(box[2])}}

# TODO: shift gold COM onto boundary
bcx=$(bc <<< "scale=4;$bwidth/2.0")
bcy=$(bc <<< "scale=4;$bheight/2.0")
bcz=$(bc <<< "scale=4;$bdepth/2.0")

gmx pdb2gmx -f "1_${surfactant}.pdb" -o "1_${surfactant}.gro" \
    -p "1_${surfactant}.top" -i "1_${surfactant}_posre.itp" \
    -ff "${force_field}" -water "${water_model}" -v

gmx pdb2gmx -f "${system}.pdb" -o "${system}.gro" \
    -p "${system}.top" -i "${system}.posre.itp" \
    -ff "${force_field}" -water "${water_model}" -v

# Packmol centrered the system at (x,y) = (0,0) but did align
# the substrate at z = 0. GROMACS-internal, the box's origin is alway at (0,0,0)
# Thus we shift the whole system in (x,y) direction by (width/2,depth/2):
gmx editconf -f "${system}.gro" -o "${system}_boxed.gro"  \
    -box $bwidth $bheight $bdepth -noc -translate $bcx $bcy 0

# For exact number of solvent molecules:
# gmx solvate -cp "${system}_boxed.gro" -cs spc216.gro \
#     -o "${system}_solvated.gro" -p "${system}.top" \
#    -scale 0.5 -maxsol $nSOL

# For certain solvent density
# scale 0.57 should correspond to standard condition ~ 1 kg / l (water)
gmx solvate -cp "${system}_boxed.gro" -cs spc216.gro \
    -o "${system}_solvated.gro" -p "${system}.top" \
    -scale 0.57

{% if ionize %}
gmx grompp -f ionize.mdp -c "${system}_solvated.gro" \
    -p "${system}.top" -o "${system}_ionized.tpr"

gmx select -s "${system}_ionized.tpr" \
    -on "${system}_SOL_selection.ndx" \
    -select SOL

gmx genion -s "${system}_ionized.tpr" \
    -n "${system}_SOL_selection.ndx" \
    -o "${system}_ionized.gro" -p "${system}.top" \
    -pname "${cation}" -np $ncation -nname "${anion}" -nn $nanion
{% endif %}
