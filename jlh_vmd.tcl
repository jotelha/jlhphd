#!/usr/bin/tclsh
# JlhVmd, a VMD package to manipulate interfacial systems or other
# topology related properties in VMD with the help of TopoTools and PbcTools
#
# Copyright (c) 2018
#               by Johannes Hörmann <johannes.hoermann@imtek.uni-freiburg.de>
#
# Sample usage:
#
#   vmd> source jlh_vmd.tcl
#   vmd> namespace import ::JlhVmd::*
#   vmd> use_SDS
#   vmd> batch_process_pdb_visual interface.lammps indenter.pdb interface_with_immersed_indenter
#
# will use parameters for SDS, merge an interfacial system's 'interface.lammps'
# data file with an indenter's 'indenter.pdb', remove any overlap and write
# interface_with_immersed_indenter.lammps, interface_with_immersed_indenter.psf
# interface_with_immersed_indenter.pdb data files as well as a tga snapshot
# interface_with_immersed_indenter.tga of the resulting system

namespace eval ::JlhVmd:: {
  variable version 0.1

  namespace export system_data_file indenter_data_file indenter_pdb_file

  namespace export system_id indenter_id combined_id
  namespace export system substrate surfactant solvent counterion indenter
  namespace export nonsolvent

  namespace export use_CTAB use_SDS
  namespace export init_system position_system merge
  namespace export read_indenter_pdb read_indenter_lmp
  namespace export init_indenter scale_indenter position_indenter clip_indenter
  namespace export merge identify_overlap remove_overlap
  namespace export set_visual show_nonsolvent show_solvent_only show_overlap
  namespace export batch_merge_lmp batch_merge_pdb
  namespace export batch_process_lmp batch_process_pdb
  namespace export batch_process_lmp_visual batch_process_pdb_visual
  namespace export write_out_indenter_immersed render_scene

  package require topotools
  package require pbctools

  set system_data_file   377_SDS_on_AU_111_51x30x2_monolayer_with_counterion_production_mixed.lammps
  # scaled, positioned and sliced indenter sample file
  set indenter_data_file 100Ang_stepped_scaled_positioned_sliced.lammps
  # unscaled, unpositioned indenter sample file
  set indenter_pdb_file  100Ang_stepped_psfgen.pdb
  # sample indenter files are 111 surface

  # lattice constant for original indenter pdb
  # confirmed for 100Ang_stepped.pdb and 100Ang_stepped.pdb :
  # 2.651 ~ 2.652

  # therefrom derived:
  # scale indenter as to fit substrate lattice constant (AU 111 at standard cond.)
  set scale_factor 1.0943

  # shift tip as to z-center apex 12 nm above substrate
  set desired_distance 120.0

  # distance within which molecules are regarded as overlapping
  set overlap_distance 2.0

  # two small supportive functions to enhance vmd's matrix manipulation toolset
  proc scalar_times_matrix {s m} {
    set res {}
    foreach row $m {
        lappend res [vecscale $s $row ]
    }
    return $res
  }
  
  # returns a scaling transformation matrix
  proc transscale {s} {
    set res  [ scalar_times_matrix $s [transidentity] ]
    lset res {3 3} 1.0
    return $res
  }


  proc use_CTAB {} {
    variable surfactant_resname CTAB
    variable counterion_name    BR
    variable counterion_resname BR
    variable solvent_resname    TIP3
    variable substrate_name     AU
    variable substrate_resname  AUM

    variable H2O_H_type 8
    variable H2O_O_type 9

    # suggestion from https://lammps.sandia.gov/threads/msg21297.html
    variable type_name_list { \
        1 HL  \
        2 HAL2  \
        3 HAL3  \
        4 CTL2  \
        5 CTL3  \
        6 CTL5  \
        7 NTL  \
        8 HT  \
        9 OT  \
       10 BR  \
       11 AU  \
    }
  }

  proc use_SDS {} {
    variable surfactant_resname SDS
    variable counterion_name    SOD
    variable counterion_resname SOD
    variable solvent_resname    TIP3
    variable substrate_name     AU
    variable substrate_resname  AUM

    variable H2O_H_type 8
    variable H2O_O_type 9

    # from SDS-related data file
    variable type_name_list { \
        1 HAL2  \
        2 HAL3  \
        3 CTL2  \
        4 CTL3  \
        5 OSL  \
        6 O2L  \
        7 SL  \
        8 HT  \
        9 OT  \
       10 SOD  \
       11 AU  \
    }
  }

  # ##################
  # au_cell_P1_111.pdb
  # ##################
  # extremes (~ measure minmax)
  #   0.28837   0.49948   0.70637  nm
  #   2.884     4.995     7.064    Angstrom
  # ##############################################################################
  # TITLE     Periodic slab: SURF, t= 0.0
  # REMARK    THIS IS A SIMULATION BOX
  # CRYST1    2.884    4.995    7.064  90.00  90.00  90.00 P 1           1
  # MODEL        1
  # ATOM	  1  Au  SURF    1	 0.000   0.000   0.000  1.00  0.00
  # ATOM	  2  Au  SURF    1	 1.440   2.500   0.000  1.00  0.00
  # ATOM	  3  Au  SURF    1	 0.000   1.660   2.350  1.00  0.00
  # ATOM	  4  Au  SURF    1	 1.440   4.160   2.350  1.00  0.00
  # ATOM	  5  Au  SURF    1	 1.440   0.830   4.710  1.00  0.00
  # ATOM	  6  Au  SURF    1	 0.000   3.330   4.710  1.00  0.00
  # TER
  # ENDMDL
  # ##############################################################################

  # number of distinct atomic layers within AU substrate building block cell
  set sb_cell_inner_lattice_points { 2.0 6.0 3.0}
  # multiples of cell building block in substrate
  set sb_cell_multiples {51.0 30.0 2.0}

  set system_rep 0
  set indenter_rep 0

  proc init_system { infile } {
    variable system_id
    variable system
    variable type_name_list
    set system_id [topo readlammpsdata $infile full]
    set system [atomselect $system_id all]
    $system global

    # https://sites.google.com/site/akohlmey/software/topotools/topotools-tutorial---various-tips-tricks
    topo guessatom element mass
    # topo guessatom name element
    topo guessatom radius element

    # suggestionf from https://lammps.sandia.gov/threads/msg21297.html
    foreach {type name} $type_name_list {
      set sel [atomselect $system_id "type $type"]
      $sel set name $name
      $sel delete
    }
    mol rename $system_id interface
  }

  proc position_system {} {
    variable surfactant_resname
    variable counterion_name
    variable counterion_resname
    variable solvent_resname
    variable substrate_name
    variable substrate_resname
    variable H2O_H_type
    variable H2O_O_type

    variable system
    variable system_id
    variable surfactant
    variable substrate
    variable counterion
    variable solvent
    variable nonsolvent

    variable sb_cell_inner_lattice_points
    variable sb_cell_multiples

    vmdcon -info "Selecting substrate ..."
    set substrate [atomselect $system_id "name $substrate_name"]
    $substrate global
    $substrate set resname $substrate_resname
    vmdcon -info [format "%-30.30s %12d" "#atoms in $substrate_resname:" [$substrate num]]

    set counterion [atomselect $system_id "name $counterion_name"]
    $counterion global
    $counterion set resname $counterion_resname
    vmdcon -info [format "%-30.30s %12d" "#atoms in $counterion_resname:" [$counterion num]]

    set solvent [atomselect $system_id "type $H2O_H_type $H2O_O_type"]
    $solvent global
    $solvent set resname $solvent_resname
    vmdcon -info [format "%-30.30s %12d" "#atoms in $solvent_resname:" [$solvent num]]

    set surfactant [atomselect $system_id "not resname $substrate_resname \
      $counterion_resname $solvent_resname"]
    $surfactant global
    $surfactant set resname $surfactant_resname
    vmdcon -info [format "%-30.30s %12d" "#atoms in $surfactant_resname:" [$surfactant num]]

    set nonsolvent [atomselect $system_id "not resname $substrate_resname"]
    $nonsolvent global

    # get substrate COM and measures
    set sb_center [measure center $substrate]
    vmdcon -info "substrate center:   [format "%8.4f %8.4f %8.4f" {*}$sb_center]"

    # measure inertia returns COM as a 3-vector in first list entry
    set sb_com    [lindex [measure inertia $substrate] 0]
    vmdcon -info "substrate COM:      [format "%8.4f %8.4f %8.4f" {*}$sb_com]"


    # calculate desired reference COM of substrate
    # (in case of substrate corner in origin 0 0 0)
    set sb_measures [ vecscale -1.0 [ vecsub {*}[measure minmax $substrate] ] ]
    vmdcon -info "substrate measures: [format "%8.4f %8.4f %8.4f" {*}$sb_measures]"

    set sb_com_reference [ vecscale 0.5 $sb_measures ]
    vmdcon -info "reference COM:      [format "%8.4f %8.4f %8.4f" {*}$sb_com_reference]"
    # pbc get returns list of box descriptions as 6-vectors:
    # 3 measures, 3 angles. Use first entry in list, first 3 indices
    # corresponding to 3-vector of length, width, height
    set cell [ lrange [ lindex [pbc get -molid $system_id] 0 ] 0 2 ]
    vmdcon -info "box:                [format "%8.4f %8.4f %8.4f" {*}$cell]"
    set cell_center [vecscale 0.5 $cell]
    vmdcon -info "box center:         [format "%8.4f %8.4f %8.4f" {*}$cell_center]"

    # discrepancy between box measures and substrate measures
    # should meet about one grid constant in x-y direction due to periodicity
    set sb_spacing [ vecsub $cell $sb_measures ]


    # estimate lattice constants in 2D-periodic directions
    # based on subsrate extremes and box measures
    set sb_lattice_constant {}
    set sb_lattice_constant_reference {}
    foreach i {0 1} {
        lappend sb_lattice_constant [ expr [ lindex $sb_measures $i] / ( [lindex $sb_cell_multiples $i] * [lindex $sb_cell_inner_lattice_points $i] - 1.0 ) ]
        lappend sb_lattice_constant_reference [expr [ lindex $cell $i] / ( [lindex $sb_cell_multiples $i] * [lindex $sb_cell_inner_lattice_points $i] )  ]
    }

    # shift whole system as to place substrate  in x-y center of box,
    # but all substrate above 0 (unwrapped) in z direction
    set sb_com_offset {}
    foreach i {0 1} {
        lappend sb_com_offset [expr [lindex $cell_center $i] - [lindex $sb_com $i] ]
    }
    lappend sb_com_offset [expr [lindex $sb_com_reference 2] - [lindex $sb_com 2] ]

    vmdcon -info "################################################################################"
    vmdcon -info "effective lattice constant (substrate extremes as reference): [format "%8.4f %8.4f" {*}$sb_lattice_constant]"
    vmdcon -info "reference lattice constant (periodic box as reference):       [format "%8.4f %8.4f" {*}$sb_lattice_constant_reference]"
    vmdcon -info "difference between substrate measures and box measures:       [format "%8.4f %8.4f" {*}$sb_spacing]"


    vmdcon -info "substrate center: [format "%8.4f %8.4f %8.4f" {*}$sb_center]"
    vmdcon -info "substrate COM:    [format "%8.4f %8.4f %8.4f" {*}$sb_com]"
    vmdcon -info "reference COM:    [format "%8.4f %8.4f %8.4f" {*}$sb_com_reference]"
    vmdcon -info "box center:       [format "%8.4f %8.4f %8.4f" {*}$cell_center]"
    vmdcon -info "offset system by: [format "%8.4f %8.4f %8.4f" {*}$sb_com_offset]"

    $system moveby $sb_com_offset

    set sb_com_new    [lindex [measure inertia $substrate] 0]
    vmdcon -info "new substrate COM:[format "%8.4f %8.4f %8.4f" {*}$sb_com_new]"
    vmdcon -info "################################################################################"

    #pbc box -on
    pbc wrap -compound residue -all
  }

  proc read_indenter_pdb { infile } {
    variable system_id
    variable system
    variable scale_factor
    variable indenter_id
    variable indenter

    set indenter_id [mol new $infile]
    set indenter [atomselect $indenter_id all]
    $indenter global
    vmdcon -info [format "%-30.30s %12d" "#atoms in indenter:" [$indenter num]]

    mol rename $indenter_id indenter
  }

  proc read_indenter_lmp { infile } {
    variable system_id
    variable system
    variable scale_factor
    variable indenter_id
    variable indenter

    set indenter_id [topo readlammpsdata $infile full]
    set indenter [atomselect $indenter_id all]
    $indenter global
    vmdcon -info [format "%-30.30s %12d" "#atoms in indenter:" [$indenter num]]
  }

  proc init_indenter {} {
    variable indenter
    variable substrate
    # use same atom type as substrate
    $indenter set name [lindex [$substrate get name] 0]
    $indenter set resname [lindex [$substrate get resname] 0]
    $indenter set type [lindex [$substrate get type] 0]
    $indenter set element [lindex [$substrate get element] 0]
    $indenter set mass [lindex [$substrate get mass] 0]
  }

  proc scale_indenter {} {
    variable indenter
    variable scale_factor
    $indenter move [transscale $scale_factor]
  }

  proc position_indenter {} {
    variable system_id
    variable substrate
    variable indenter
    variable desired_distance

    set in_com    [lindex [measure inertia $indenter] 0]

    # indenter height (lowest )
    set in_measures [ vecscale -1.0 [ vecsub {*}[measure minmax $indenter] ] ]
    set in_height [lindex $in_measures 2]

    set cell [ lrange [ lindex [pbc get -molid $system_id] 0 ] 0 2 ]
    set cell_center [vecscale 0.5 $cell]

    set sb_height [lindex [measure minmax $substrate] { 1 2 } ]
    set sb_surface_to_cell_center [ expr [ lindex $cell_center 2] - $sb_height ]

    set cell_center_to_desired_distance \
      [ expr $desired_distance - $sb_surface_to_cell_center]
    set z_shift [ list 0. 0. \
      [expr 0.5 * $in_height + $cell_center_to_desired_distance ] ]

    set in_offset [ vecadd [vecsub $cell_center $in_com] $z_shift ]
    $indenter moveby $in_offset

    # check
    set sb_in_distance [expr [lindex [measure minmax $indenter] {0 2}] - [lindex [measure minmax $substrate] {1 2}] ]
    vmdcon -info "substrate - indenter apex distance after positioning: $sb_in_distance"
  }

  proc clip_indenter {} {
    variable system_id
    variable indenter_id
    variable indenter

    set a [ molinfo $system_id get a ]
    set b [ molinfo $system_id get b ]
    set c [ molinfo $system_id get c ]

    vmdcon -info [format "%-30.30s %12d" \
      "# atoms in original indenter:" [$indenter num]]

    set indenter [ atomselect $indenter_id \
      "(x > 0.0) and (y > 0.0) and (z > 0.0) and (x < $a) and (y < $b) and (z <$c)"]
    $indenter global

    vmdcon -info [format "%-30.30s %12d" \
        "# atoms in clipped indenter:" [$indenter num]]
    # check
  }


  proc merge {} {
    variable system_id
    variable indenter_id
    variable combined_id

    variable surfactant_resname
    variable counterion_resname
    variable solvent_resname
    variable substrate_resname

    variable system
    variable substrate

    variable indenter
    variable surfactant
    variable counterion
    variable solvent
    variable nonsolvent

    variable overlap_distance

    #set combined_id [::TopoTools::mergemols "$system_id $indenter_id"]
    set combined_id [::TopoTools::selections2mol "$system $indenter"]
    # transfer box measures
    molinfo $combined_id set {a b c alpha beta gamma} \
      [molinfo $system_id get {a b c alpha beta gamma}]

    # indenter
    #atomselect macro solute "index >= [$system num]"

    # assumes substrate annd identer of same material
    set indenter [atomselect $combined_id "index >= [$system num]"]
    $indenter global
    vmdcon -info [format "%30s" "#atoms in indenter $substrate_resname:"] \
      [format "%12d" [$indenter num]]

    set substrate [atomselect $combined_id \
      "resname $substrate_resname and index < [$system num]"]
    $substrate global
    vmdcon -info [format "%30s" "#atoms in substrate $substrate_resname:"] \
      [format "%12d" [$substrate num]]

    set counterion [atomselect $combined_id "resname $counterion_resname"]
    $counterion global
    vmdcon -info [format "%30s" "#atoms in $counterion_resname:"] \
      [format "%12d" [$counterion num]]

    set solvent [atomselect $combined_id "resname $solvent_resname"]
    $solvent global
    vmdcon -info [format "%30s" "#atoms in $solvent_resname:"] \
      [format "%12d" [$solvent num]]

    set surfactant [atomselect $combined_id "resname $surfactant_resname"]
    $surfactant global
    vmdcon -info [format "%30s" "#atoms in $surfactant_resname:"] \
      [format "%12d" [$surfactant num]]

    set nonsolvent [atomselect $combined_id "not resname $solvent_resname"]
    $nonsolvent global
    vmdcon -info [format "%30s" "#atoms not in $solvent_resname:"] \
     [format "%12d" [$nonsolvent num]]

    mol off $system_id
    mol off $indenter_id

    mol rename $combined_id merged
  }

  proc identify_overlap {} {
    variable combined_id

    variable system
    variable overlap_distance
    variable overlapping
    variable nonoverlapping

    variable surfactant_resname
    variable counterion_resname
    variable solvent_resname
    variable substrate_resname

    atomselect macro immersed "index >= [$system num]"

    set overlapping [atomselect $combined_id \
      "same fragment as (exwithin $overlap_distance of immersed)"]
    $overlapping global
    set nonoverlapping [atomselect $combined_id \
      "not (same fragment as (exwithin $overlap_distance of immersed))"]
    $nonoverlapping global

    # report on overlapping molecules
    variable overlapping_counterion [atomselect \
      $combined_id "resname $counterion_resname and ([$overlapping text])"]
    $overlapping_counterion global
    vmdcon -info [format "%-30.30s" "#atoms in overlapping $counterion_resname:"] \
      [format "%12d" [$overlapping_counterion num]]

    variable overlapping_solvent [atomselect $combined_id \
      "resname $solvent_resname and ([$overlapping text])"]
    $overlapping_solvent global
    vmdcon -info [format "%-30.30s" "#atoms in overlapping $solvent_resname:"] \
      [format "%12d" [$overlapping_solvent num]]

    variable overlapping_surfactant [atomselect $combined_id \
      "resname $surfactant_resname and ([$overlapping text])"]
    $overlapping_surfactant global
    vmdcon -info [format "%-30.30s" "#atoms in overlapping $surfactant_resname:"] \
      [format "%12d" [$overlapping_surfactant num]]

    return $overlapping
  }

  # routine for possibly overlapping surfactant and counterion molecules needed!

  proc remove_overlap {} {
    variable system_id
    variable combined_id
    variable nonoverlapping
    variable overlapping

    variable surfactant_resname
    variable counterion_resname
    variable solvent_resname
    variable substrate_resname

    variable substrate
    variable indenter
    variable surfactant
    variable counterion
    variable solvent
    variable nonsolvent

    variable desired_distance

    variable indenter_immersed_id [::TopoTools::selections2mol $nonoverlapping]
    variable overlap_id [::TopoTools::selections2mol $overlapping]

    # copy periodic box
    molinfo $indenter_immersed_id set {a b c alpha beta gamma} \
      [molinfo $system_id get {a b c alpha beta gamma}]

    # assumes substrate annd identer of same material, distinguishes by z loc

    # rough selection based on distance, not pretty, not robust
    # does not account for substrate thickness
    set indenter [atomselect $indenter_immersed_id  \
      "resname $substrate_resname and z >= $desired_distance"]
    $indenter global
    vmdcon -info [format "%30s" "#atoms in indenter $substrate_resname:"] \
      [format "%12d" [$indenter num]]

    set substrate [atomselect $indenter_immersed_id \
      "resname $substrate_resname and z < $desired_distance"]
    $substrate global
    vmdcon -info [format "%30s" "#atoms in substrate $substrate_resname:"] \
      [format "%12d" [$substrate num]]

    set counterion [atomselect $indenter_immersed_id \
      "resname $counterion_resname"]
    $counterion global
    vmdcon -info [format "%30s" "#atoms in $counterion_resname:"] \
      [format "%12d" [$counterion num]]

    set solvent [atomselect $indenter_immersed_id "resname $solvent_resname"]
    $solvent global
    vmdcon -info [format "%30s" "#atoms in $solvent_resname:"] \
      [format "%12d" [$solvent num]]

    set surfactant [atomselect $indenter_immersed_id \
      "resname $surfactant_resname"]
    $surfactant global
    vmdcon -info [format "%30s" "#atoms in $surfactant_resname:"] \
      [format "%12d" [$surfactant num]]

    set nonsolvent [atomselect $indenter_immersed_id \
      "not resname $solvent_resname"]
    $nonsolvent global
    vmdcon -info [format "%30s" "#atoms not in $solvent_resname:"] \
      [format "%12d" [$nonsolvent num]]

    mol off $combined_id
    mol off $overlap_id

    mol rename $indenter_immersed_id indenter_immersed
    mol rename $overlap_id overlap

    return $indenter_immersed_id
  }

  # writes lammps data file, pdb and psf
  proc write_out_indenter_immersed { outname } {
    variable indenter_immersed_id
    set sel [atomselect $indenter_immersed_id all]
    topo -molid $indenter_immersed_id writelammpsdata $outname.lammps full
    $sel writepsf $outname.psf
    $sel writepdb $outname.pdb
  }


  proc show_nonsolvent { {mol_id 0} {rep_id 0} } {
    # atomselect keywords
    # name type backbonetype residuetype index serial atomicnumber element residue
    # resname altloc resid insertion chain segname segid all none fragment pfrag
    # nfrag numbonds backbone sidechain protein nucleic water waters
    # vmd_fast_hydrogen helix alpha_helix helix_3_10 pi_helix sheet betasheet
    # beta_sheet extended_beta bridge_beta turn coil structure pucker user user2
    # user3 user4 x y z vx vy vz ufx ufy ufz phi psi radius mass charge beta
    # occupancy sequence rasmol sqr sqrt abs floor ceil sin cos tan atan asin acos
    # sinh cosh tanh exp log log10 volindex0 volindex1 volindex2 volindex3 volindex4
    # volindex5 volindex6 volindex7 vol0 vol1 vol2 vol3 vol4 vol5 vol6 vol7
    # interpvol0 interpvol1 interpvol2 interpvol3 interpvol4 interpvol5 interpvol6
    # interpvol7 at acidic cyclic acyclic aliphatic alpha amino aromatic basic
    # bonded buried cg charged hetero hydrophobic small medium large neutral polar
    # purine pyrimidine surface lipid lipids ion ions sugar solvent glycan carbon
    # hydrogen nitrogen oxygen sulfur noh heme conformationall conformationA
    # conformationB conformationC conformationD conformationE conformationF drude
    # unparametrized addedmolefacture qwikmd_protein qwikmd_nucleic qwikmd_glycan
    # qwikmd_lipid qwikmd_hetero

    variable solvent_resname
    variable substrate
    variable indenter
    variable counterion

    # make solid atoms appear as thick beads
    $substrate  set radius 5.0
    $indenter   set radius 5.0
    $counterion set radius 3.0

    mol selection not resname $solvent_resname
    mol representation CPK
    # or VDW
    mol color element
    mol material Opaque
    # color by element name

    mol modrep $rep_id $mol_id

    pbc box -on -molid $mol_id
  }

  proc show_solvent_only { {mol_id 0} {rep_id 0} } {
    variable solvent_resname
    mol selection resname $solvent_resname
    mol representation lines
    mol color element
    mol material Glass3

    mol modrep $rep_id $mol_id
  }

  proc show_overlap { {mol_id 0} {rep_id 0} } {
    variable system
    variable overlap_distance

    mol representation Licorice
    mol color element
    mol material Transparent

    mol selection \
      "same fragment as (exwithin $overlap_distance of (index >= [$system num]))"

    mol modrep $rep_id $mol_id
  }

  # hides solvent
  proc set_visual {} {
    variable substrate
    variable indenter
    variable counterion

    # make solid atoms appear as thick beads
    $substrate  set radius 5.0
    $indenter   set radius 5.0
    $counterion set radius 3.0

    display resetview

    color Display Background    gray
    color Display BackgroundTop white
    color Display BackgroundBot gray
    color Element Na            green
    display backgroundgradient on

    # after resetview usually centered top view
    # these should result in a centered side view
    rotate x by -90
    # values set empirically
    translate by 0 0.5 0
    scale by 0.4
  }

  proc render_scene { outname } {
    render TachyonInternal $outname.tga
  }

  # read both system and indenter from lammps data file, merges them and
  # removes overlap
  proc batch_merge_lmp { system_infile indenter_infile } {
    vmdcon -info "Read system from LAMMPS data file $system_infile..."
    init_system $system_infile
    vmdcon -info "Position system..."
    position_system
    vmdcon -info "Read indenter from LAMMPS data file $indenter_infile..."
    read_indenter_lmp $indenter_infile
    vmdcon -info "Init indenter..."
    init_indenter
    vmdcon -info "Merge systems..."
    merge
    vmdcon -info "Identify ovelap..."
    identify_overlap
    vmdcon -warn "ATTENTION: No routine for (counter-)ion replacement implemented yet"
    vmdcon -info "Remove overlap..."
    return [ remove_overlap ]
  }

  # read system from lammps data file, indenter from raw pdb file
  proc batch_merge_pdb { system_infile indenter_infile } {
    vmdcon -info "Read system from LAMMPS data file $system_infile..."
    init_system $system_infile
    vmdcon -info "Position system..."
    position_system
    vmdcon -info "Read indenter from PDB file $indenter_infile..."
    read_indenter_pdb $indenter_infile
    vmdcon -info "Init indenter..."
    init_indenter
    vmdcon -info "Scale indenter..."
    scale_indenter
    vmdcon -info "Position indenter..."
    position_indenter
    vmdcon -info "Clip indenter..."
    clip_indenter
    vmdcon -info "Merge systems..."
    merge
    vmdcon -info "Identify ovelap..."
    identify_overlap
    vmdcon -info "ATTENTION: No routine for (counter-)ion replacement implemented yet"
    vmdcon -info "Remove overlap..."
    return [ remove_overlap ]
  }

  proc batch_process_lmp { system_infile indenter_infile outname } {
    set out_id [ batch_merge_lmp $system_infile $indenter_infile ]
    write_out_indenter_immersed $outname
    return $out_id
  }

  proc batch_process_pdb { system_infile indenter_infile outname } {
    set out_id [ batch_merge_pdb $system_infile $indenter_infile ]
    write_out_indenter_immersed $outname
    return $out_id
  }

  proc batch_process_lmp_visual { system_infile indenter_infile outname } {
    set out_id [ batch_process_lmp $system_infile $indenter_infile $outname ]
    set_visual
    show_nonsolvent $out_id
    render_scene $outname
  }

  proc batch_process_pdb_visual { system_infile indenter_infile outname } {
    set out_id [ batch_process_pdb $system_infile $indenter_infile $outname ]
    set_visual
    show_nonsolvent $out_id
    render_scene $outname
  }
}

# interp alias {} jlh {} ::JlhVmd
package provide jlhvmd $::JlhVmd::version
