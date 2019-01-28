# Surfactant Adsorption Workflow

  * [TODO](#todo)
  * [Software requirements](#software-requirements)
  * [Overview](#overview)
    + [Selection of simulation parameters](#selection-of-simulation-parameters)
    + [Initial configuration preparation](#initial-configuration-preparation)
    + [Surfactant film MD](#surfactant-film-md)
    + [Indenter on film MD](#indenter-on-film-md)
    + [Realization as a FireWorks workflow](#realization-as-a-fireworks-workflow)
  * [Detailed description of initial configuration preparation](#detailed-description-of-initial-configuration-preparation)
    + [Surface sample preparation](#surface-sample-preparation)
    + [Aggregate preassembly](#aggregate-preassembly)
    + [Formatting: Preparation of PDB files to be read by GROMACS](#formatting-preparation-of-pdb-files-to-be-read-by-gromacs)
    + [Solvation](#solvation)
    + [Formatting: Convert GROMACS output back to PDB](#formatting-convert-gromacs-output-back-to-pdb)
    + [Convert PDB to CHARMM  PSF](#convert-pdb-to-charmm--psf)
    + [Create LAMMPS input](#create-lammps-input)

Tool scripts and template input files are to be found within the repository

* [N_surfactant_on_substrate_template](https://github.com/jotelha/N_surfactant_on_substrate_template)

MD parameters are based upon

* [CHARMM36 Jul17 package, modified](https://github.com/jotelha/jlh_toppar_c36_jul17)

## TODO 
(from high to low priority):

- [ ] Attach all simulation meta data and parameters available to workflows.
- [ ] Remove any absolute path dependency from `JobAdmin.py` and `job_admin.ipynb`. 
      Push and pull any relevant files to the data base. 
- [ ] Make workflow environment-independent (i.e. nothing machine-dependent in `JobAdmin.py` and `job_admin.ipynb`,
      especially no `module load` commands). Using worker-specific parameters as descibed within the 
      [FireWorks ducumentation](https://materialsproject.github.io/fireworks/worker_tutorial.html) should help.
- [ ] Make choice of *worker* and *queue* (i.e. bwCloud, NEMO or JUWELS) really possible by single Fireworks option
      `category`, without further adaptions. (For this, ssh has to work again between NEMO and bwCloud!)
- [ ] Right now, the workflow technically decays into four subsequent, but independent workflows: 
      Substrate slab preparation, intital film configuration preparation, quasi-equilibration MD, indenter MD.
      I would like to have generic `push_datafile_to_db` and `pull_datafile_from_db` to be applicable, 
      but never necessary, at any point of the workflow.
- [ ] Transfer workflow from Python to a library of .yaml text files.

## Software requirements

All required software is set up ready-to-use within the openstack image (TODO: make available)

List of relevant modules, loadbable with `module load ${MODULE}` within image as of 28th Jan 2019:
```console
$ module avail
FireWorks/jlh-25Jan19
GROMACS-Top/jlh-2018.1
GROMACS/2018.1-gnu-7.3-openmpi-2.1.1
LAMMPS/22Aug18-gnu-7.3-openmpi-2.1.1-netcdf-4.6.1-pnetcdf-1.8.1-colvars-16Nov18
MDTools/jlh-25Jan19-python-2.7
MDTools/jlh-25Jan19                                                             
Ovito/3.0.0-dev301-x86_64                                                      
VMD/1.9.3-text
```

The packages `MDTools/jlh-25Jan19` and `MDTools/jlh-25Jan19-python-2.7` contain
* charmm2lammps (comes with LAMMPS 16Mar18 sources)
* [packmol](http://m3g.iqm.unicamp.br/packmol/home.shtml)
* pdb-tools, modified. forked from https://github.com/haddocking/pdb-tools
* [pizza.py](https://pizza.sandia.gov) (
* custom scripts `ncjoin.py`, `netcdf2data.py`, `pdb_packmol2gmx.sh`, `replicate.sh`
Except `Pizza.py`, all Python-based tools run on Python 3. For `Pizza.py`,it is necessary to load the module's Python 2 variant.

`FireWorks/jlh-25Jan19` contains

* Fireworks modifications & extensions from [fireworks-jlh][https://github.com/jotelha/fireworks-jlh]

Fireworks itself is available systemwide outside of the module framework.

`GROMACS/2018.1-gnu-7.3-openmpi-2.1.1` contains a  modifed

* GROMACS 2018.1 top folder including charmm36.ff, available at
  [jlh_gmx_2018.1_top](https://github.com/jotelha/jlh_gmx_2018.1_top)
  
overiding the `GROMACS/2018.1-gnu-7.3-openmpi-2.1.1` module's standard force field collection, when loaded afterwards.

`VMD/1.9.3-text` contains

* vmd-1.9.3 with psfgen plugin, pre-compiled distribution 
  [vmd-1.9.3.bin.LINUXAMD64.text.tar.gz](http://www.ks.uiuc.edu/Development/Download/download.cgi?PackageName=VMD)


## Overview

### Selection of simulation parameters

Simulation meta data and meaningful parameters are selected and cast into a pandas dataframe via some
Jupyter notebook. TODO: details here.
Columns and their types are
```text
box                                         object
ci_initial_placement                        object
counterion                                  object
indenter                                    object
indenter_pdb                                object
pbc                                          int64
pressure                                     int64
sb_area                                    float64
sb_area_per_sf_molecule                    float64
sb_circular_area_per_sf_molecule_radius    float64
sb_crystal_plane                             int64
sb_measures                                 object
sb_multiples                                object
sb_name                                     object
sb_normal                                    int64
sb_square_area_per_sf_molecule_side        float64
sb_thickness                               float64
sb_unit_cell                                object
sb_volume                                  float64
sf_concentration                           float64
sf_expected_aggregates                      object
sf_nmolecules                                int64
sf_preassembly                              object
solvent                                     object
substrate                                   object
surfactant                                  object
sv_density                                   int64
sv_preassembly                              object
temperature                                  int64  
```
(extensible)

*sb* means substrated, *sf* surfactant, *sv* solvent and *ci* counter ion. Fields `box`, `sb_measures` and `sb_unit_cell` are 3-tuples (i.e. type `list`) of `float `, while `sb_multiples` is 3-tuple of `int`. Sample entry:
```text
> sim_df.loc['1010_CTAB_on_AU_111_63x36x2_bilayer']
box                                        [1.82e-08, 1.8000000000000002e-08, 1.800000000...
ci_initial_placement                                                                  random
counterion                                                                                BR
indenter                                                                                None
indenter_pdb                                                                            None
pbc                                                                                      111
pressure                                                                                   1
sb_area                                                                          3.26671e-16
sb_area_per_sf_molecule                                                            3.233e-19
sb_circular_area_per_sf_molecule_radius                                                2e-10
sb_crystal_plane                                                                         111
sb_measures                                [1.82e-08, 1.8000000000000002e-08, 1.400000000...
sb_multiples                                                                     [63, 36, 2]
sb_name                                                                       AU_111_63x36x2
sb_normal                                                                                  2
sb_square_area_per_sf_molecule_side                                                    6e-10
sb_thickness                                                                         1.4e-09
sb_unit_cell                                           [3e-10, 5e-10, 7.000000000000001e-10]
sb_volume                                                                        4.61502e-25
sf_concentration                                                                      0.0005
sf_expected_aggregates                                                         intermmediate
sf_nmolecules                                                                           1010
sf_preassembly                                                                       bilayer
solvent                                                                                  H2O
substrate                                                                                 AU
surfactant                                                                              CTAB
sv_density                                                                               997
sv_preassembly                                                                        random
temperature                                                                              298
Name: 1010_CTAB_on_AU_111_63x36x2_bilayer, dtype: object
```

The table is serialited (TODO: details) as `surfactant_on_AU_111.json` and pushed to FireWork's *FilePad* with identifier `surfactant_on_AU_111_df_json`.

### Initial configuration preparation

 1. Create GROMACS .hdb, .rtp for surfactants, dummy .rtp for ions
 2. Load system description from some stored pandas.Dataframe, e.g. from pickle.
    Should contain information on substrate measures, box size, etc.
    TODO: ADD DETAILED DESCRIPTION OF EXPECTED VALUES.
 3. Identify all unique substrate slabs and create according .pdb files from multiples of unit cell with
    `gmx genconf`.
 4. Create a subfolter for every system and copy (or links) some necessary files
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
 
### Surfactant film MD

 1. Minimization
 2. NVT equilibration
 3. NPT equilibration
 4. 10 ns NPT run
 
### Indenter on film MD

 1. Indenter insertion
 2. Minimization
 3. NPT equilibration
 4. Indenter approach

### Realization as a FireWorks workflow

A workflow contains fireworks suffixed as follows:

 1. sb_replicate
 2. packmol_fill_script_template, packmol, recover_packmol, store_packmol_files, forward_packmol_files
 4. packmol2gmx, gmx_solvate, gmx2pdb
 5. psfgen
 6. ch2lmp
 7. (pull_datafile_from_db) # TODO: make optional
 8. (prepare_system_files) # TODO: make obsolete
 9. minimization
10. equilibration_nvt
11. equilibration_npt
12. 10ns_production_mixed
13. (store_data_file) #TODO: make optional
14. initiate_indenter_workflow, indenter_insertion, pizzapy_merge
15. (pull_datafile_from_db)
16. minimization
17. equilibration_npt
 
Several custom FireTasks are to be found within 
[`fireworks/user_objects/firetasks/jlh_tasks.py`](https://github.com/jotelha/fireworks-jlh/blob/master/lib/python3.6/site-packages/fireworks/user_objects/firetasks/jlh_tasks.py) 
and a python class
[`JobAdmin.py`](https://github.com/jotelha/fireworks-jlh/blob/master/lib/python3.6/site-packages/fwtools/JobAdmin.py)
within the [fireworks-jlh](https://github.com/jotelha/fireworks-jlh)
repository helps to set up workflows, i.e. from within a Jupyter notebook such as 
[job_admin.ipynb](https://github.com/jotelha/fireworks-jlh/blob/master/examples/job_admin.ipynb). 

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
For this purpose, a packmol input script template [packmol.inp](https://github.com/jotelha/N_surfactant_on_substrate_template/blob/master/packmol.inp) is filled by `packmol_fill_scipt_template`.

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
