#!/bin/bash

# assume a surfactant / substrate system

system=200_SDS_on_50_Ang_AFM_tip_model
force_field=charmm36
water_model=tip3p

# remove chain id
pdb_chain.py ${system}_packmol.pdb > ${system}.pdb

gmx pdb2gmx -f "1_SDS.pdb" -o "1_SDS.gro" -p "1_SDS.top" -i "1_SDS.posre.itp" \
  -ff charmm36 -water tip3p > gmx_surfactant_prep.out 2> gmx_surfactant_prep.err

# gmx pdb2gmx -f "${system}.pdb" -o "${system}.gro" \
#    -p "dummy.top" -i "dummy.posre.itp" \
#    -ff "${force_field}" -water "${water_model}" -v

gmx pdb2gmx -f "${system}.pdb" -o "${system}.gro" \
    -p "${system}.top" -i "${system}.posre.itp" \
    -ff "${force_field}" -water "${water_model}" -v \
    > gmx_system_prep.out 2> gmx_system_prep.err


gmx editconf -f "${system}.gro" -o "${system}_boxed.gro" -d 2.0 -bt cubic

gmx solvate -cp "${system}_boxed.gro" -cs spc216.gro -o "${system}_solvated.gro" -p "${system}.top" # -scale 0.5 -maxsol $nSOL
# n=200
# sed -E "$(echo 's/^(Surfactant[[:blank:]]+)1[[:blank:]]*$/\1' $n '/')" 1_SDS.top > "${system}.top"

gmx grompp -f em.mdp -c "${system}_solvated.gro" -p "${system}.top" \
    -o em.tpr 2> gmx_grompp_em.err > gmx_grompp_em.log

mpirun -n 2 gmx_mpi mdrun -v -deffnm em