# Surfactant Adsorption Workflows

[![DOI](https://zenodo.org/badge/145995562.svg)](https://doi.org/10.5281/zenodo.14007768)

This repository contains input files and workflows underlying 

> J. L. Hörmann, C. (刘宸旭) Liu, Y. (孟永钢) Meng, and L. Pastewka, “Molecular simulations of sliding on SDS surfactant films,” The Journal of Chemical Physics, vol. 158, no. 24, p. 244703, Jun. 2023, doi: [10.1063/5.0153397](https://doi.org/10.1063/5.0153397).

as well as an installable Python package `jlhpy`

MD parameters are based upon [CHARMM36 Jul17 package](http://mackerell.umaryland.edu/download.php?filename=CHARMM_ff_params_files/toppar_c36_jul17.tgz).

## Content overview

* `bash`:       tiny bash tools
* `dat`:        indenter and substrate coordinate files
* `ff`:         force fields
* `gmx_input`:  GROMACS input files
* `ipynb`:      Jupyter notebooks
* `jlhpy`:      Python utilities
* `lmp_input`:  LAMMPS input files and templates
* `nco`:        NetCDF operators scripts
* `packmol`:    PACKMOL input script templates
* `pdb`:        pdb (protein database) format data files
* `pymol`:      Pymol script templates 
* `ref`:        reference data extracted from other publications
* `regex`:      useful regular expressions
* `vmd`:        VMD-executable tcl scripts and templates
* `wf/scripts`: scripts that generated FireWorks workflows
 
## Overview on initial configuration preparation

 1. Create GROMACS .hdb, .rtp for surfactants, dummy .rtp for ions
 2. Load system description from some stored pandas.Dataframe, e.g. from pickle.
    Should contain information on substrate measures, box size, etc.
 3. Identify all unique substrate slabs and create according .pdb files from multiples of unit cell with
    `gmx genconf`.
 4. Create a subfolder for every system and copy (or links) some necessary files
 5. Use packmol to create bilayer on gold surface
 6. Use `pdb_packmol2gmx` to run sum simple renumbering on residues.
    This is mainly necessary because gmx needs each substrate atom to have its own residue number.
 7. Create GROMACS .gro and .top from .pdb, solvate (and ionize system).
    A bash script suffixed `_gmx_solvate.sh` is created for this purpose.
 8. Split .gro system into .pdb chunks of at most 9999 residues
    in order not to violate .pdb format restritcions. psfgen cannot work on .pdb > 9999 residues.
    The bash script suffixed `_gmx2pdb.sh` does this.
 9. Generate .psf with VMD's `psfgen`.
 10.Generate LAMMPS data from psfgen-generated .psf and .pdb with `charmm2lammps.pl`

## Detailed description of initial configuration preparation

### Surface sample preparation

Initially, a bash script "replicate.sh" is used to construct an N x M x L multiple of surface chunks originally stemming from INTERFACE FF, i.e.

```bash
replicate.sh 21 12 1 111
```

will read the file `au_cell_P1_111.gro` with content

```
Periodic slab: SURF, t= 0.0
     6
    1 SURF   Au    1   0.000   0.000   0.000
    1 SURF   Au    2   0.144   0.250   0.000
    1 SURF   Au    3   0.000   0.166   0.235
    1 SURF   Au    4   0.144   0.416   0.235
    1 SURF   Au    5   0.144   0.083   0.471
    1 SURF   Au    6   0.000   0.333   0.471
   0.28837   0.49948   0.70637
```
will write intermediate files `AU_111_21x12x1.pdb`
```pdb

TITLE     Periodic slab: SURF, t= 0.0
REMARK    THIS IS A SIMULATION BOX
CRYST1   60.558   59.938    7.064  90.00  90.00  90.00 P 1           1
MODEL        1
ATOM      1  Au  SURF    1       0.000   0.000   0.000  1.00  0.00            
ATOM      2  Au  SURF    1       1.440   2.500   0.000  1.00  0.00  
...
ATOM   1511  Au  SURF    1      59.114  55.773   4.710  1.00  0.00            
ATOM   1512  Au  SURF    1      57.674  58.273   4.710  1.00  0.00            
TER
ENDMDL
```
as well as standardized `AU_111_21x12x1_tidy.pdb`
```pdb
ATOM      1  Au  SURF    1       0.000   0.000   0.000  1.00  0.00            
ATOM      2  Au  SURF    1       1.440   2.500   0.000  1.00  0.00
...
ATOM   1511  Au  SURF    1      59.114  55.773   4.710  1.00  0.00            
ATOM   1512  Au  SURF    1      57.674  58.273   4.710  1.00  0.00            
TER    1513      SUR     1                                                      
END  
```
stripped of the header and the final `AU_111_21x12x1_reres.pdb`
```pdb
ATOM      1  Au  SURF    1       0.000   0.000   0.000  1.00  0.00            
ATOM      2  Au  SURF    2       1.440   2.500   0.000  1.00  0.00
...
ATOM   1511  Au  SURF 1511      59.114  55.773   4.710  1.00  0.00            
ATOM   1512  Au  SURF 1512      57.674  58.273   4.710  1.00  0.00            
TER    1513      SUR  1513                                                      
END  

```
with all atoms assigned individual, consecutive residue numbers.

This step is performed as FireWorks `sb_replicate` for different sets of substrates via the member `prepare_substrates(system_names)` of `JobAdmin`.


### Aggregate preassembly

Second, aggregates of surfactant molecules are preassembled upon the substrate slab via
[packmol](http://m3g.iqm.unicamp.br/packmol/home.shtml)
For this purpose, a packmol input script template [packmol.inp](https://github.com/jotelha/jlhphd/blob/master/packmol.inp) is filled by `packmol_fill_scipt_template`.

`recover_packmol` helps to get the latest packing configuration, even if `packmol` did not finish successfully.

`store_packmol_files` pushes final configuraation to FilePad. TODO: elaborate
`forward_packmol_files` assures the next step receives the right configuration. TODO: elaborate

### Formatting: Preparation of PDB files to be read by GROMACS
Gromacs requires the number of molecules in a residue to exactly match the rtp entry.
In our modified gromacs charmm36.ff, the SURF residue consists of 1 AU atom.
`packmol2gmx` makes the bash script `pdb_packmol2gmx.sh` reformat PDB as necessary, making use of `pdb-tools`.
Both are within `MDTools/jlh-25Jan19`.

```bash
#!/bin/bash -x
# prepares packmol output for gromacs

if [ -n "$1" ]; then
  BASENAME=${1%_packmol.pdb} # removes pdb ending if passed
  BASENAME=${BASENAME%.pdb} # removes pdb ending if passed
else
  echo "No input file given!"
  exit 1
fi

# Remove chain id
pdb_chain.py  "${BASENAME}_packmol.pdb" > "${BASENAME}_nochainid.pdb"

# extracts surface and everything else into separate parts
# surface residue must be 1st in file
pdb_rslice.py :1 "${BASENAME}_nochainid.pdb" > "${BASENAME}_substrate_only.pdb"
pdb_rslice.py 2: "${BASENAME}_nochainid.pdb" > "${BASENAME}_surfactant_only.pdb"

# assign unique residue ids
# Gromacs requires number of molecules in residue to match rtp entry.
# In our modified gromacs charmm36.ff, the SURF residue consists of 1 AU atom
pdb_reres_by_atom_9999.py "${BASENAME}_substrate_only.pdb" \
  -resid 1 > "${BASENAME}_substrate_only_reres.pdb"

# merges two pdb just by concatenating
head -n -1 "${BASENAME}_substrate_only_reres.pdb" \
  > "${BASENAME}_concatenated.pdb"  # last line contains END statement
tail -n +6 "${BASENAME}_surfactant_only.pdb" \
  >> "${BASENAME}_concatenated.pdb" # first 5 lines are packmol-generated header

# ATTENTION: pdb_reres writes residue numbers > 9999 without complains,
# however thereby produces non-standard PDB format
pdb_reres_9999.py "${BASENAME}_concatenated.pdb" -resid 1 > "${BASENAME}.pdb"
```

### Solvation
`gmx_solvate` uses GROMACS' preprocessing functionality to solvate the system in water. A bash script like
```bash
#!/bin/bash -x
#Generated on Sun Jan 27 19:14:50 2019 by testuser@jlh-cloud-10jan19
set -e

system=646_SDS_on_AU_111_51x30x2_hemicylinders_with_counterion
surfactant=SDS
water_model="tip3p"
force_field="charmm36"

cation="NA"
anion="BR"
ncation=646
nanion=0

bwidth=14.7000
bheight=15.0000
bdepth=18.0000

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
```
is automatically generated.

### Formatting: Convert GROMACS output back to PDB
`gmx2pdb` similarly generates a bash script such as
```bash
#!/bin/bash -x
# Generated on Sun Jan 27 19:14:50 2019 by testuser@jlh-cloud-10jan19
components=( "substrate" "surfactant" "solvent" "ions" )

system="646_SDS_on_AU_111_51x30x2_hemicylinders_with_counterion"

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
```
converting the system into a set of PDB files with at most 9999 resiudues each. This is necessary in order to make `psfgen` work on the system.

### Convert PDB to CHARMM  PSF
`psfgen` derives a PSF from the system's PDB, CHARMM's native genreic topology information RTF and force field parameters PRM. PSF ([*protein structure file*](https://www.ks.uiuc.edu/Training/Tutorials/namd/namd-tutorial-unix-html/node23.html) is CHARMM's native topology format. This is the ssential step to actually map CHARMM Force Field parameters on the preassembld system.

### Create LAMMPS input
`ch2lmp` utilizes the Perl script `charmm2lammps.pl` in order to take PDB, PSF, RTF and PRM (four files) and converts it into LAMMPS .in and .data files (two files). Lateron in our workflow, .data is used. .in is obsolete.