# Data processing

## NetCDF

### concatenate

    ncrcat: INFO/WARNING Multi-file concatenator encountered packing attribute scale_factor for variable time. NCO copies the packing attributes from the first file to the output file. The packing attributes from the remaining files must match exactly those in the first file or data from subsequent files will not unpack correctly. Be sure all input files share the same packing attributes. If in doubt, unpack (with ncpdq -U) the input files, then concatenate them, then pack the result (with ncpdq). This message is printed only once per invocation.
    ncks -d frame,0,10 -v time 653_CTAB_on_AU_111_63x36x2_bilayer_with_counterion_50Ang_stepped_production_mixed.nc

# Visualization

## Ovito

### Particle radii

```
C   0.5
N   0.7
BR  0.9
AU  0.7
