#!/bin/bash -x
# {{header}}
components=( "substrate" "surfactant" "solvent" "ions" )

module load gromacs/2018.1-gnu-5.2
module load vmd/1.9.3-text

system="{{system_name}}"

gmx select -s "${system}_ionized.gro" -on "${system}_substrate.ndx" \
  -select 'resname SURF'
gmx select -s "${system}_ionized.gro" -on "${system}_surfactant.ndx" \
  -select 'resname SDS CTAB'
gmx select -s "${system}_ionized.gro" -on "${system}_solvent.ndx" \
  -select 'resname SOL'
gmx select -s "${system}_ionized.gro" -on "${system}_ions.ndx" \
  -select 'resname NA BR'


# convert .gro to .pdb chunks with max 9999 residues each  
for component in ${components[@]}; do 
    echo "Processing component ${component}..."
    
    # Create separate .gro and begin residue numbers at 1 within each:
    gmx editconf -f "${system}_ionized.gro" -n "${system}_${component}.ndx" \
      -o "${system}_${component}_000.gro" -resnr 1
      
    # maximum number of chunks, not very important as long as large enough
    for (( num=0; num<=999; num++ )); do 
      numstr=$(printf "%03d" $num); 
      nextnumstr=$(printf "%03d" $((num+1))); 

      # ATTENTION: gmx select offers two different keywords, 'resid' / 'residue'
      # While 'resid' can occur multiple times, 'residue' is a continuous id for 
      # all residues in system.
      
      # create selection with first 9999 residues
      gmx select -s "${system}_${component}_${numstr}.gro" \
        -on "${system}_${component}_${numstr}.ndx" -select 'residue < 10000'

      # write nth 9999 residue .pdb package
      gmx editconf -f "${system}_${component}_${numstr}.gro" \
        -n "${system}_${component}_${numstr}.ndx" \
        -o "${system}_${component}_${numstr}_gmx.pdb" -resnr 1
        
      # use vmd to get numbering right
      vmdcmd="mol load pdb \"${system}_${component}_${numstr}_gmx.pdb\"; "
      vmdcmd="${vmdcmd} set sel [atomselect top \"all\"]; \$sel writepdb " 
      vmdcmd="${vmdcmd} \"${system}_${component}_${numstr}.pdb\"; exit"

      echo "${vmdcmd}" | vmd -eofexit
      
      # create selection with remaining residues
      gmx select -s "${system}_${component}_${numstr}.gro" \
        -on "${system}_${component}_remainder.ndx" -select 'not (residue < 10000)'

      if [ $? -ne 0 ] ; then
        echo "No more ${component} residues left. Wrote $((num+1)) .pdb"
        break
      fi

      # renumber remaining residues in new .gro file
      gmx editconf -f "${system}_${component}_${numstr}.gro" \
        -n "${system}_${component}_remainder.ndx" \
        -o "${system}_${component}_${nextnumstr}.gro" -resnr 1
    done
done
