#!/usr/bin/env python
# Wrapper for MergeLammpsDataFiles.merge_lammps_datafiles
# Takes a full, reliable system description from reffile
# and a newly generated system from datafile.
# Checks for sections that are missing in datafile, but present in reffile.
# These sections are either copied as they are, or specific mappings are 
# applied if available:
# TopoTools v1.7 generated LAMMPS datafiles do not contain any Coeffs sections.
# However, they contain commented index <--> name mappings for all Coeffs
# sections. Each missing section is checked for such mappings and they are
# applied if available.
# Next to the outfile, reffile and datafile are stripped off their comments
# and written to reffile_stripped and datafile_stripped, since pizza.py
# does not process commented files.
#
# load pizza.py via
#   deactivate # if venv loaded
#   module --force purge
#   module load GCC/5.5.0 Python/2.7.14
#   source ${HOME}/venv/jlh-juwels-python-2.7/bin/acitvate
#   module load  jlh/mdtools/26Jun18-jlh-python-2.7 # pizza.py contained here
#
# execute with
#   pizza.py -f merge.py datafile reffile outfile
import sys
import argparse
from MergeLammpsDataFiles import merge_lammps_datafiles

# recommendation from https://pizza.sandia.gov/doc/Section_basics.html#3_2

if not globals().has_key("argv"): argv = sys.argv
print("Called with '{}'".format(argv))

# reffile  = "377_SDS_on_AU_111_51x30x2_monolayer_with_counterion_psfgen.data"
# datafile = "377_SDS_on_AU_111_51x30x2_monolayer_with_counterion_100Ang_stepped.lammps"
# outfile  = "377_SDS_on_AU_111_51x30x2_monolayer_with_counterion_100Ang_stepped_parametrized.lammps"

parser = argparse.ArgumentParser(
    description='Merges LAMMPS data file with missing entries from reference.')
parser.add_argument('datafile', nargs='?', metavar='datafile.lammps',
    default='377_SDS_on_AU_111_51x30x2_monolayer_with_counterion_100Ang_stepped.lammps',
    help="LAMMPS data file to process.")
parser.add_argument('reffile', nargs='?', metavar='reffile.lammps',
    default='377_SDS_on_AU_111_51x30x2_monolayer_with_counterion_psfgen.data',
    help="Reference data file containing complete system information.")
parser.add_argument('outfile', nargs='?', metavar='outfile.lammps',
    default='377_SDS_on_AU_111_51x30x2_monolayer_with_counterion_100Ang_stepped_parametrized.lammps',
    help="Merged output data file.")
args = parser.parse_args(argv[1:])

print("Using datafile: {}, reffile: {}; outfile: {}".format( args.datafile, args.reffile, args.outfile ) )

try:
    merge_lammps_datafiles( args.datafile, args.reffile, args.outfile )
except:
    print("Failed.")
    exit(1)

exit()
