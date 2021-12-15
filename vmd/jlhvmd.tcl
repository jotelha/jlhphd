#!/usr/bin/tclsh
# JlhVmd, a VMD script to wrap-join molecualr LAMMPS data files under
# preservation of the bounding box.
#
# Copyright (c) 2018,2019, 2020, 2021
#               by Johannes Hoermann <johannes.hoermann@imtek.uni-freiburg.de>
#
# $Id: jlhvmd.tcl,v 0.3 2021/12/15 $
#
#
# Sample usage for wrapping periodic images and joining split residues:
#
#   vmd> jlh set data_file initial_config.lammps outPrefix default
#   vmd> jlh read bb bb.yaml
#   vmd> jlh init
#   vmd> jlh show surfactant
#   vmd> jlh wrap atom
#   vmd> jlh join residue
#   vmd> jlh write
#
# CHANGELOG
#
# ## [0.3] 2021-12-15
# ### Changed
# - Threw away obsolete functionality
#
# ## [0.2] 2019-05-14
# ### Changed
# - Suppose to have substrate terminated at z = 0
# - Package structure copied from Axel Kohlmeyer's topotools
#
# ## [0.2.1] 2019-05-19
# ### Changed
# - jlh user interface
#
# ## [0.2.2] 2019-05-20
# ### Changed
# - random position changed from small volume between indenter and substrate to
#   whole bounding box except lower substrate part
#
# ## [0.2.3] 2019-08-10
# ### Changed
# - number of digits to fill in with type names now determined by counting
#   characters of largest type name after integer sorting instead of
#   using the actual number of types.
#
# ## [0.2.4] 2019-08-11
# ### Changed
# - commands case-insenitive, allow i.e. "jlh use SDS" as well as "jlh use sds"
# - removed obsolete pdb processing functionality and other code snippets
#
# ## [0.2.5] 2019-08-22
# ### Changed
# - added bb_center as global variable
# - split bb positioning and wrapping
# - added several wrapping and joining commands

namespace eval ::JlhVmd:: {
    variable version 0.3

    package require topotools
    package require pbctools
    package require yaml

    # default values:
    set bounding_box { {0. 0. 0.} {150. 150. 150.} }
    set bb_center    { 75. 75. 75. }

    # io
    set data_file "default.lammps"
    set out_prefix "system"
}

# help/usage/error message and online documentation.
proc ::JlhVmd::usage {} {
    vmdcon -info ""
    vmdcon -info "JlhVmd, a VMD package to manipulate interfacial systems or other"
    vmdcon -info "topology related properties in VMD with the help of TopoTools and PbcTools."
    vmdcon -info ""
    vmdcon -info ""
    vmdcon -info "usage: jlh <command> \[args...\] <flags>"
    vmdcon -info ""
    vmdcon -info ""
    vmdcon -info "common flags (not implemented):"
    vmdcon -info ""
    vmdcon -info "  -molid     <num>|top    molecule id (default: 'top')"
    vmdcon -info "  -sel       <selection>  atom selection function or text (default: 'all')"
    vmdcon -info ""
    vmdcon -info ""
    vmdcon -info "commands:"
    vmdcon -info ""
    vmdcon -info "  help                                          prints this message"
    vmdcon -info ""
    vmdcon -info "  read <key> <file> \[ <key> <file> \[ ... \] \]    reads parameter <key> from <file>."
    vmdcon -info "  set <key> <value> \[ <key> <value> \[ ... \] \]   sets parameter <key> to <value>."
    vmdcon -info "  lmp2pdb <file> <file>                         converts lammps data file to pdb file."
    vmdcon -info ""
    vmdcon -info "  info                                          display system information."
    vmdcon -info "  init                                          initializes system without manipulation."
    vmdcon -info "  join <key>                                    (re-)joins residues split across boundries."
    vmdcon -info "  render <key>                                  render image of scene to .tga file."
    vmdcon -info "  wrap <key>                                    wrap system into one periodic image."
    vmdcon -info "  write                                         write .lammps, .psf and .pdb output files."
    vmdcon -info ""
    vmdcon -info "key - value pairs"
    vmdcon -info ""
    vmdcon -info "  join:   residue           "
    vmdcon -info "  read:   bb                bounding box, expects yaml file with keys 'xlo', 'xhi', 'ylo', 'yhi', 'zlo', 'zhi'."
    vmdcon -info "  render: nonsolvent        "
    vmdcon -info "          solvent           "
    vmdcon -info "          surfactant        "
    vmdcon -info "  set:    bb                bounding box ({ { float float float } { float float float } })"
    vmdcon -info "          distance          desired distance between surface and indenter (float)"
    vmdcon -info "          interfaceInfile   input LAMMPS data file of interface (str)"
    vmdcon -info "          indenterInfile    input LAMMPS data file of indenter (str)"
    vmdcon -info "          outputPrefix      output prefix prepended to all resulting files (str)"
    vmdcon -info "  show:   nonsolvent        "
    vmdcon -info "          solvent           "
    vmdcon -info "          surfactant        "
    vmdcon -info "  wrap:   atom              "
    vmdcon -info "          residue           "
    vmdcon -info ""
    vmdcon -info "Copyright (c) 2018,2019,2020,2021"
    vmdcon -info "              by Johannes Hoermann <johannes.hoermann@imtek.uni-freiburg.de>"
    vmdcon -info ""
    return
}


proc ::JlhVmd::jlh { args } {
    variable version

    variable bounding_box

    variable data_file
    variable out_prefix

    set cmd {}
    # set sel {} ; # need to initialize it here for scoping

    # process generic arguments and remove them
    # from argument list.
    set newargs {}
    for {set i 0} {$i < [llength $args]} {incr i} {
        set arg [lindex $args $i]

        if {[string match -?* $arg]} {

            set val [lindex $args [expr $i+1]]

            switch -- $arg {
                -molid {
                    if {[catch {molinfo $val get name} res]} {
                        vmdcon -err "Invalid -molid argument '$val': $res"
                        return
                    }
                    set molid $val
                    if {[string equal $molid "top"]} {
                        set molid [molinfo top]
                    }
                    incr i
                }

                -sel {
                    # check if the argument to -sel is a valid atomselect command
                    if {([info commands $val] != "") && ([string equal -length 10 $val atomselect])} {
                        set localsel 0
                        set selmol [$val molid]
                        set sel $val
                    } else {
                        set localsel 1
                        set seltxt $val
                    }
                    incr i
                }

                -- break

                default {
                    vmdcon -info "default: $arg"
                }
            }
        } else {
            lappend newargs $arg
        }
    }

    set retval ""
    if {[llength $newargs] > 0} {
        set cmd [lindex $newargs 0]
        set newargs [lrange $newargs 1 end]
    } else {
        set newargs {}
        set cmd help
    }

    # check whether we have a valid command.
    set validcmd {
      "lmp2pdb"
      "set" "read" "use" "init"
      "show" "render"
      "wrap" "join"
      "write"
      "help" "info" }
    if {[lsearch -exact $validcmd $cmd] < 0} {
        vmdcon -err "Unknown sub-command '$cmd'"
        usage
        return
    }

    # branch out to the various subcommands
    switch -nocase -- $cmd {
        "lmp2pdb" {
            set filein  [lindex $newargs 0]
            set fileout [lindex $newargs 1]
            topo readlammpsdata $filein
            set sel [atomselect top all]
            $sel writepdb $fileout
            set retval 0
        }
        "set" {
            while {[llength $newargs] > 1} {
                set key [lindex $newargs 0]
                set newargs [lrange $newargs 1 end]
                switch -nocase -- $key {
                    bb {
                        set bounding_box [lindex $newargs 0]
                        compute_bb_center
                        set newargs [lrange $newargs 1 end]
                    }

                    data_file {
                        set data_file [lindex $newargs 0]
                        set newargs [lrange $newargs 1 end]
                    }

                    out_prefix {
                        set out_prefix [lindex $newargs 0]
                        set newargs [lrange $newargs 1 end]
                    }

                    default {
                        vmdcon -warn "Unknown parameter: $key"
                    }
                }
            }
            set retval 0
        }

        "read" {
            set key [lindex $newargs 0]
            set newargs [lrange $newargs 1 end]
            switch -nocase -- $key {
                bb {
                    set bb_file [lindex $newargs 0]
                    set newargs [lrange $newargs 1 end]
                    set retval [read_bb_from_yaml $bb_file]
                }

                default {
                    vmdcon -err "Unknown parameter: $key"
                    set retval 1
                }
            }
        }

        "render" {
            set key [lindex $newargs 0]
            set newargs [lrange $newargs 1 end]
            switch -nocase -- $key {
                solvent {
                    render_solvent
                    set retval 0
                }

                surfactant {
                    render_surfactant
                    set retval 0
                }

                nonsolvent {
                    render_nonsolvent
                    set retval 0
                }

                default {
                    vmdcon -err "Unknown parameter: $key"
                    set retval 1
                }
            }
        }

        use {
            set key [lindex $newargs 0]
            set newargs [lrange $newargs 1 end]
            switch -nocase -- $key {
                ctab {
                    use_CTAB
                    set retval 0
                }

                sds {
                    use_SDS
                    set retval 0
                }

                default {
                    vmdcon -err "Unknown parameter: $key"
                    set retval 1
                }
            }
        }

        "show" {
            set key [lindex $newargs 0]
            set newargs [lrange $newargs 1 end]
            switch -nocase -- $key {
                solvent {
                    show_solvent_only
                    set retval 0
                }

                surfactant {
                    show_surfactant_only
                    set retval 0
                }

                nonsolvent {
                    show_nonsolvent
                    set retval 0
                }

                default {
                    vmdcon -err "Unknown parameter: $key"
                    set retval 1
                }
            }
        }

        "wrap" {
            set key [lindex $newargs 0]
            set newargs [lrange $newargs 1 end]
            switch -nocase -- $key {
                residue {
                    position_bb
                    wrap_residue_into_bb
                    set retval 0
                }

                atom {
                    position_bb
                    wrap_atom_into_bb
                    set retval 0
                }

                default {
                    vmdcon -err "Unknown parameter: $key"
                    set retval 1
                }
            }
        }

        "join" {
            set key [lindex $newargs 0]
            set newargs [lrange $newargs 1 end]
            switch -nocase -- $key {
                residue {
                    position_bb
                    join_residue
                    set retval 0
                }

                default {
                    vmdcon -err "Unknown parameter: $key"
                    set retval 1
                }
            }
        }

        "init" {
            initialize data_file
            set retval 0
        }

        "write" {
            write_top_all $out_prefix
            set retval 0
        }

        "info" {
            display_system_information
            set retval 0
        }

        "help" {
            usage
            set retval 0
        }

        default {
            vmdcon -err "Unknown sub-command: $cmd"
            set retval 1
        }
    }
    return $retval
}

proc ::JlhVmd::read_bb_from_yaml { bb_file } {
    variable bounding_box
    # read bounding box from .yaml file
    set bb [::yaml::yaml2dict -file $bb_file]
    set bounding_box [ list \
      [ list [ dict get $bb xlo ] [ dict get $bb ylo ] [ dict get $bb zlo ] ] \
      [ list [ dict get $bb xhi ] [ dict get $bb yhi ] [ dict get $bb zhi ] ] ]
    vmdcon -info [ format "read bounding box from %s: %26s; %26s" \
        $bb_file \
        [ format "%8.4f %8.4f %8.4f" {*}[ lindex $bounding_box 0 ] ] \
        [ format "%8.4f %8.4f %8.4f" {*}[ lindex $bounding_box 1 ] ] ]
    compute_bb_center
    return $bounding_box
}

proc ::JlhVmd::compute_bb_center { } {
    variable bounding_box
    variable bb_center
    variable bb_measure

    vmdcon -info [ format "         bounding box: %26s; %26s" \
        [ format "%8.4f %8.4f %8.4f" {*}[ lindex $bounding_box 0 ] ] \
        [ format "%8.4f %8.4f %8.4f" {*}[ lindex $bounding_box 1 ] ] ]

    set bb_measures [ vecscale -1.0 [ vecsub {*}$bounding_box ] ]
    set bb_center   [ vecscale 0.5  [ vecadd {*}$bounding_box ] ]
    vmdcon -info [ format "bounding box measures: %26s" \
        [ format "%8.4f %8.4f %8.4f" {*}$bb_measures ] ]
    vmdcon -info [ format "  bounding box center: %26s" \
        [ format "%8.4f %8.4f %8.4f" {*}$bb_center ] ]
}

# adjust position of bounding box
proc ::JlhVmd::position_bb {} {
  variable bb_center
  pbc box -center origin -shiftcenter $bb_center -on
}

# wraps everything into one periodic image
proc ::JlhVmd::wrap_atom_into_bb {} {
    variable bb_center
    pbc wrap -center origin -shiftcenter $bb_center -nocompound -all -verbose
}

# wraps into one periodic image, but keeps residues connected
proc ::JlhVmd::wrap_residue_into_bb {} {
    variable bb_center
    pbc wrap -center origin -shiftcenter $bb_center -compound residue -all -verbose
}

# tries to join residues split across bb boundaries
proc ::JlhVmd::join_residue {} {
    variable bb_center
    pbc join residue -bondlist -all -verbose
}

proc ::JlhVmd::init_system { infile { psffile "" } } {
  variable system_id
  variable system
  variable type_name_list

  if { $psffile ne "" } {
    set system_id [mol new $psffile type psf waitfor all]
    topo readlammpsdata $infile full -molid $system_id
  } else {
    # no psf topology, use topotools to derive types
    set system_id [topo readlammpsdata $infile full]

    # https://sites.google.com/site/akohlmey/software/topotools/topotools-tutorial---various-tips-tricks
    topo guessatom element mass
    # topo guessatom name element
    +topo guessatom radius element

    # suggestion from https://lammps.sandia.gov/threads/msg21297.html
    foreach {type name} $type_name_list {
      set sel [atomselect $system_id "type '$type'"]
      $sel set name $name
      $sel delete
    }
  }

  set system [atomselect $system_id all]
  $system global

  mol rename $system_id interface
}

proc ::JlhVmd::display_system_information { {mol_id 0} } {
  vmdcon -info "Number of objects:"
  vmdcon -info "Number of atoms:           [format "% 12d" [topo numatoms -molid ${mol_id} ]]"
  vmdcon -info "Number of bonds:           [format "% 12d" [topo numbonds -molid ${mol_id} ]]"
  vmdcon -info "Number of angles:          [format "% 12d" [topo numangles -molid ${mol_id} ]]"
  vmdcon -info "Number of dihedrals:       [format "% 12d" [topo numdihedrals -molid ${mol_id} ]]"
  vmdcon -info "Number of impropers:       [format "% 12d" [topo numimpropers -molid ${mol_id} ]]"

  vmdcon -info "Number of object types:"
  vmdcon -info "Number of atom types:      [format "% 12d" [topo numatomtypes -molid ${mol_id} ]]"
  vmdcon -info "Number of bond types:      [format "% 12d" [topo numbondtypes -molid ${mol_id} ]]"
  vmdcon -info "Number of angle types:     [format "% 12d" [topo numangletypes -molid ${mol_id} ]]"
  vmdcon -info "Number of dihedral types:  [format "% 12d" [topo numdihedraltypes -molid ${mol_id} ]]"
  vmdcon -info "Number of improper types:  [format "% 12d" [topo numimpropertypes -molid ${mol_id} ]]"

  vmdcon -info "Object type names:"
  vmdcon -info "Atom type names:      [topo atomtypenames -molid ${mol_id} ]"
  vmdcon -info "Bond type names:      [topo bondtypenames -molid ${mol_id} ]"
  vmdcon -info "Angle type names:     [topo angletypenames -molid ${mol_id} ]"
  vmdcon -info "Dihedral type names:  [topo dihedraltypenames -molid ${mol_id} ]"
  vmdcon -info "Improper type names:  [topo impropertypenames -molid ${mol_id} ]"
}

proc ::JlhVmd::show_nonsolvent { {mol_id 0} {rep_id 0} } {
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
  variable bounding_box
  variable bb_center
  variable bb_measure

  # make solid atoms appear as thick beads
  $substrate  set radius 5.0
  if { [ info exists indenter ] > 0 } {
    $indenter   set radius 5.0
  }
  $counterion set radius 3.0

  mol selection not resname $solvent_resname
  mol representation CPK
  # or VDW
  mol color element
  mol material Opaque
  # color by element name

  mol modrep $rep_id $mol_id

  pbc box -on -center origin -shiftcenter $bb_center -molid $mol_id
}

proc ::JlhVmd::show_solvent_only { {mol_id 0} {rep_id 0} } {
  variable solvent_resname
  mol selection resname $solvent_resname
  mol representation lines
  mol color element
  mol material Glass3

  mol modrep $rep_id $mol_id
}

proc ::JlhVmd::show_surfactant_only { {mol_id 0} {rep_id 0} } {
  variable surfactant_resname
  mol selection resname $surfactant_resname

  mol representation CPK
  mol color element
  mol material Opaque

  mol modrep $rep_id $mol_id
}

proc ::JlhVmd::show_overlap { {mol_id 0} {rep_id 0} } {
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
proc ::JlhVmd::set_visual {} {
  variable substrate
  variable indenter
  variable counterion

  # make solid atoms appear as thick beads
  $substrate  set radius 5.0
  if { [ info exists indenter ] > 0 } {
    $indenter   set radius 5.0
  }
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

proc ::JlhVmd::render_scene { outname } {
  render TachyonInternal $outname.tga
}

# initialization without manipulation
proc ::JlhVmd::initialize { system_infile } {
  vmdcon -info "-------------------------------------------------------------"
  vmdcon -info "Read system from LAMMPS data file $system_infile..."
  init_system $system_infile
  vmdcon -info "-------------------------------------------------------------"
  vmdcon -info "Objects in system read from $system_infile:"
  display_system_information
  vmdcon -info "-------------------------------------------------------------"
  vmdcon -info "Make types ascii-sortable to preserve original order..."
  make_types_ascii_sortable
  vmdcon -info "-------------------------------------------------------------"
  vmdcon -info "Objects in system after type renaming:"
  display_system_information
  vmdcon -info "-------------------------------------------------------------"
  vmdcon -info "Populating global selections and variables..."
  populate_selections
  vmdcon -info "-------------------------------------------------------------"
  vmdcon -info "Position bounding box..."
  position_bb
}

proc ::JlhVmd::render_nonsolvent {} {
  variable out_prefix
  vmdcon -info "-------------------------------------------------------------"
  vmdcon -info "Set visualization properties..."
  set_visual
  vmdcon -info "-------------------------------------------------------------"
  vmdcon -info "Show everything except solvent..."
  show_nonsolvent
  vmdcon -info "-------------------------------------------------------------"
  vmdcon -info "Render snapshot..."
  render_scene $out_prefix
}

proc ::JlhVmd::render_solvent {} {
  variable out_prefix
  vmdcon -info "-------------------------------------------------------------"
  vmdcon -info "Set visualization properties..."
  set_visual
  vmdcon -info "-------------------------------------------------------------"
  vmdcon -info "Show only solvent..."
  show_solvent_only
  vmdcon -info "-------------------------------------------------------------"
  vmdcon -info "Render snapshot..."
  render_scene $out_prefix
}

proc ::JlhVmd::render_surfactant {} {
  variable out_prefix
  vmdcon -info "-------------------------------------------------------------"
  vmdcon -info "Set visualization properties..."
  set_visual
  vmdcon -info "-------------------------------------------------------------"
  vmdcon -info "Show only surfactant..."
  show_surfactant_only
  vmdcon -info "-------------------------------------------------------------"
  vmdcon -info "Render snapshot..."
  render_scene $out_prefix
}

# namespace eval ::TopoTools::
# adapted from ::TopoTools::retypebonds
proc ::TopoTools::make_bond_types_ascii_sortable {sel} {
  set bondlist  [bondinfo getbondlist $sel type]

  set newbonds {}

  # sort type names as inetgers, select last element (largest) and
  # determine its field width:
  set num_digits [
    string length [ lindex [ lsort -integer [ topo bondtypenames ] ] end ] ]

  vmdcon -info "Prepending zeros to bond types filling ${num_digits} digits."

  foreach bond $bondlist {
      set type [format "%0${num_digits}d" [ lindex $bond 2 ]]
      lappend newbonds [list [lindex $bond 0] [lindex $bond 1] $type]
  }
  setbondlist $sel type $newbonds
}

# adapted from proc ::TopoTools::retypeangles
proc ::TopoTools::make_angle_types_ascii_sortable {sel} {
    set anglelist [angleinfo getanglelist $sel]
    set newanglelist {}

    # sort type names as inetgers, select last element (largest) and
    # determine its field width:
    set num_digits [
      string length [ lindex [ lsort -integer [ topo angletypenames ] ] end ] ]

    vmdcon -info "Prepending zeros to angle types filling ${num_digits} digits."
    foreach angle $anglelist {
        lassign $angle type i1 i2 i3
        set type [format "%0${num_digits}d" $type]
        lappend newanglelist [list $type $i1 $i2 $i3]
    }
    setanglelist $sel $newanglelist
}

# adapted from ::TopoTools::retypedihedrals
proc ::TopoTools::make_dihedral_types_ascii_sortable {sel} {
  set dihedrallist [dihedralinfo getdihedrallist $sel]
  set newdihedrallist {}

  # sort type names as inetgers, select last element (largest) and
  # determine its field width:
  set num_digits [
    string length [ lindex [ lsort -integer [ topo dihedraltypenames ] ] end ] ]

  vmdcon -info "Prepending zeros to angle types filling ${num_digits} digits."
  foreach dihedral $dihedrallist {
      lassign $dihedral type i1 i2 i3 i4
      set type [format "%0${num_digits}d" $type]
      lappend newdihedrallist [list $type $i1 $i2 $i3 $i4]
  }
  setdihedrallist $sel $newdihedrallist
}

# adapted from ::TopoTools::retypeimpropers
proc ::TopoTools::make_improper_types_ascii_sortable {sel} {
  set improperlist [improperinfo getimproperlist $sel]
  set newimproperlist {}
  set num_digits [
    string length [ lindex [ lsort -integer [ topo impropertypenames ] ] end ] ]

  vmdcon -info "Prepending zeros to improper types filling ${num_digits} digits."
  foreach improper $improperlist {
      lassign $improper type i1 i2 i3 i4
      set type [format "%0${num_digits}d" $type]
      lappend newimproperlist [list $type $i1 $i2 $i3 $i4]
  }
  setimproperlist $sel $newimproperlist
}

proc ::JlhVmd::populate_selections {} {
  variable counterion_name
  variable counterion_resname
  variable substrate_name
  variable substrate_resname
  variable solvent_resname
  variable surfactant_resname
  variable H2O_H_type
  variable H2O_O_type

  variable system
  variable system_id
  variable counterion
  variable nonsolvent
  variable solvent
  variable substrate
  variable surfactant

  vmdcon -info "Selecting substrate ..."
  set substrate [atomselect $system_id "name $substrate_name"]
  $substrate global
  $substrate set resname $substrate_resname
  vmdcon -info [format "%-30.30s %12d" "#atoms in $substrate_resname:" [$substrate num]]

  set counterion [atomselect $system_id "name $counterion_name"]
  $counterion global
  $counterion set resname $counterion_resname
  vmdcon -info [format "%-30.30s %12d" "#atoms in $counterion_resname:" [$counterion num]]

  # for types with leading zeroes: single quotation marks necessary, otherwise selection fails
  vmdcon -info "Solvent selection by 'type '$H2O_H_type' '$H2O_O_type''"
  set solvent [atomselect $system_id "type '$H2O_H_type' '$H2O_O_type'"]
  $solvent global
  $solvent set resname $solvent_resname
  vmdcon -info [format "%-30.30s %12d" "#atoms in $solvent_resname:" [$solvent num]]

  set surfactant [atomselect $system_id "not resname $substrate_resname \
    $counterion_resname $solvent_resname"]
  $surfactant global
  $surfactant set resname $surfactant_resname
  vmdcon -info [format "%-30.30s %12d" "#atoms in $surfactant_resname:" [$surfactant num]]

  set nonsolvent [atomselect $system_id "not resname $solvent_resname"]
  $nonsolvent global
  vmdcon -info [format "%-30.30s %12d" "#atoms in nonsolvent:" [$nonsolvent num]]
}

interp alias {} jlh {} ::JlhVmd::jlh
package provide jlhvmd $::JlhVmd::version