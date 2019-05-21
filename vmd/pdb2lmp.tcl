proc pdb2lmp { infile outfile } {
  mol new $infile
  package require topotools
  topo writelammpsdata $outfile
}
