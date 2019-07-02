#!/bin/bash
# sample script on how to modify a NetCDF trajectory with NetCDF operators
set -e

# module load NCO

# delete unwanted quantities
echo "01 -- DELETE"
perf stat -B ncks -t 8 -D5 -O -x -v c_peratom_stress,f_peratom_stress_ave,mass,mol,velocities $1 _thinned_$2

# Note that renaming a dimension to the name of a dependent variable can be used to invert the relationship between an independent coordinate variable and a dependent variable.  In  this  case,
#       the named dependent variable must be one-dimensional and should have no missing values.  Such a variable will become a coordinate variable.

# permute dimensions
echo "02 -- PERMUTE"
perf stat -B ncpdq -t 8 -D5 -O -a atom,frame,spatial _thinned_$2 _thinned_permuted_sorted_$2

# rename dimension
# ncrename -d atom,id _thinned_permuted_$2 _thinned_permuted_renamed_$2

# sort NetCDF by atom ID
# ncap2 -O -v -S sortByAtomID.nco _thinned_permuted_$2 _thinned_permuted_sorted_$2

#
echo "03 -- SELECT"
perf stat -B ncap2 -t 8 -D5 -O -v -S selectByAtomType.nco _thinned_permuted_sorted_$2 _thinned_permuted_sorted_selected_$2

# drop old atom coordinate
echo "04 -- DROP"
perf stat -B ncks -t 8 -D5 -O -x -v atom _thinned_permuted_sorted_selected_$2 _thinned_permuted_sorted_selected_deleted_$2

# rename new atom coordinate
echo "05 -- RENAME"
perf stat -B ncrename -t 8 -D5 -O -d atom_out,atom _thinned_permuted_sorted_selected_deleted_$2 _thinned_permuted_sorted_selected_deleted_renamed_$2

# permute back
echo "06 -- PERMUTE"
perf stat -B ncpdq -D5 -t 8 -O -a frame,atom,spatial _thinned_permuted_sorted_selected_deleted_renamed_$2 $2
