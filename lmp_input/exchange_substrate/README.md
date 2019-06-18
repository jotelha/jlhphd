# Exchange substrate in existing interfacial system:

## Sample bash work flow

```bash
# on bwCloud, load the modules
# module load MDTools FireWorks

# get files from database
filepad.py --action pull --file substrate.lammps --verbose \
  substrate/AU/111/51x30x21/equilibrated.lammps
filepad.py --action pull --file interface.lammps --verbose \
  646_SDS_on_AU_111_51x30x2_hemicylinders_with_counterion_10ns.lammps
filepad.py --action pull --file coeff_nonbonded.input --verbose \
  SDS/in/H2O/on/AU/coeff/nonbonded.input
filepad.py --action pull --file coeff.input --verbose \
  SDS/in/H2O/on/AU/coeff.input

# replace substrate with lammps commands only

# first delete "old" susbtrate from interface:
lmp -in lmp_delete_subst.input \
  -v dataFile       interface.lammps \
  -v outfile        solution.lammps

# create png snapshots of systems for debugging purposes
lmp -in lmp_snapshot.input \
  -v dataFile substrate.lammps \
  -v snapshotName substrate
lmp -in lmp_snapshot.input \
  -v dataFile solution.lammps \
  -v snapshotName solution

# shift substrate in order to align its upper 111 surface
# with origin in z direction:
lmp -in lmp_shift_surface_to_zero_z.input \
  -v coeffFile      coeff_nonbonded.input \
  -v dataFile       substrate.lammps \
  -v outfile        shiftedSubstrate.lammps

# shift solution in order to align its lower liquid-vacuum interface
# with origin in z direction and tighten box around content:
lmp -in lmp_shift_solution_to_zero_z.input \
  -v coeffFile      coeff.input \
  -v dataFile       solution.lammps \
  -v outfile        shiftedSolution.lammps

lmp -in lmp_snapshot.input \
  -v dataFile       shiftedSubstrate.lammps \
  -v snapshotName   shiftedSubstrate
lmp -in lmp_snapshot.input \
  -v dataFile       shiftedSolution.lammps \
  -v snapshotName   shiftedSolution

# scale solution a little bit in x- and y-direction in order to
# exactly match substrate area. The distortion is to be alleviated
# during subsequent minimization, equilibration, ...
lmp -in lmp_scale_solution_to_substrate.input \
  -v coeffFile      coeff.input \
  -v substrateFile  shiftedSubstrate.lammps \
  -v solutionFile   shiftedSolution.lammps \
  -v outfile        scaledSolution.lammps
# concatenate aligned substrate and solution to form new interface:
lmp -in lmp_merge.input \
  -v coeffFile      coeff.input \
  -v dataFile       scaledSolution.lammps \
  -v dataFileToAppend shiftedSubstrate.lammps \
  -v outfile        merged.lammps

lmp -in lmp_snapshot.input -v dataFile merged.lammps -v snapshotName merged

filepad.py --action delete push -- verbose \
  --metadata-file metadata.yaml --file merged.lammps \
  interface/SDS/646/AU/111/52x30x21/hemicylinders/initial_config.lammps
```

For the example here, the following `metadata.yaml` file has been used:system_name: 646_SDS_on_AU_111_51x30x21_hemicylinders_with_counterion

```yaml
state: 10 ns NPH/NPT, then substrate exchanged
ci_preassembly:                               at polar heads
counterion:                                   NA
pbc:                                          111 # periodic in all directions
pressure:                                     1 # atm
pressure_unit:                                atm
sb_name:                                      AU_111_51x30x21
sb_area:                                      2.2351e-16
sb_area_unit:                                 m^2
sb_crystal_plane:                             111
sb_measures:
- 1.48e-8
- 1.51e-8
- 1.51e-8
sb_measures_unit:                             m
sb_multiples:
- 51
- 30
- 21
sb_normal:                                    2 # z
sb_thickness:                                 1.51e-8
sb_thickness_unit:                            m
sb_unit_cell:
- 3e-10
- 5e-10
- 7e-10
sb_unit_cell_unit:                            m
sb_volume:                                    3.37e-24
sb_volume_unit:                               m^3
sf_concentration:                             0.00680627
sf_concentration_unit:                        M # M = mol L^-3
sf_nmolecules:                                646
sf_preassembly:                               hemicylinders
solvent:                                      H2O
substrate:                                    AU
surfactant:                                   SDS
sv_density:                                   997 # kg m^-3
sv_density_unit:                              kg m^-3
sv_preassembly:                               random
temperature:                                  298 # K
temperature_unit:                             K
```
