proc lmp2pdb { infile outfile } {
  package require topotools
  topo readlammpsdata $infile
  set sel [atomselect top all]
  $sel writepdb $outfile
}
