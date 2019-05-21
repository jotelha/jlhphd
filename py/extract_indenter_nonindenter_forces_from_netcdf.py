#!/usr/bin/env python
""" Extracts indenter - nonindenter forces from NetCDF """

import logging
import globals
import os.path
pyfile = os.path.basename(__file__)

def extractIndenterNonIndenterForcesFromNetCDF(
    t2n_array = globals.sds_t2n_array, # from global.py
    force_keys = [
        'forces',
        'f_storeAnteSHAKEForces',
        'f_storeAnteStatForces',
        'f_storeUnconstrainedForces',
        'f_storeAnteSHAKEForcesAve',
        'f_storeAnteStatForcesAve',
        'f_storeUnconstrainedForcesAve' ],
    indenter_z_forces_file_name    = {
        'json': 'indenter_z_forces.json',
        'txt':  'indenter_z_forces.txt' },
    nonindenter_z_forces_file_name = {
        'json': 'nonindenter_z_forces.json',
        'txt':  'nonindenter_z_forces.txt' },
    separating_xy_plane_z_pos      = 20,
    # 20: some arbitrary xy-plane seperating substrate & indenter
    solid_element                  = 'Au',
    netcdf                         = 'default.nc',
    dimension_of_interest          = 2, # forces in z dir
    netcdf_output_interval         = globals.std_netcdf_output_interval,
    output_formats                 = ['json','txt'] ):

    logger = logging.getLogger(
        '{:s}.extractIndenterNonIndenterForcesFromNetCDF'.format(pyfile))

    # import ase.io
    # from ase.io import read
    import numpy as np
    import pandas as pd
    from ase.io import NetCDFTrajectory

    t2n_array = t2n_array_dict[system_name]

    logger.info("Opening NetCDF '{:s}'.".format(netcdf))
    tmp_traj = NetCDFTrajectory(netcdf, 'r',
        types_to_numbers = list( t2n_array ),
        keep_open=True )

    # use first frame to crudely identify indenter
    solid_selection = (
        tmp_traj[0].get_atomic_numbers() == ase.data.atomic_numbers[
            solid_element])
    indenter_selection = ( solid_selection & (
        tmp_traj[0].get_positions()[:,2] > separating_xy_plane_z_pos ) )

    logger.info("solid: {: 9d} atoms, therof indenter: {: 9d} atoms.".format(
        np.count_nonzero(solid_selection),
        np.count_nonzero(indenter_selection) ) )

    logger.info("Reading NetCDF frame by frame.")
    indenter_force_sum_dict = { key: [] for key in force_keys }
    nonindenter_force_sum_dict = { key: [] for key in force_keys }
    for key in force_keys:
        if key in tmp_traj[0].arrays:
            indenter_force_sum_dict[key] = np.array(
                [ f[indenter_selection].arrays[key].sum(axis=0)
                     for f in tmp_traj ] )
            nonindenter_force_sum_dict[key] = np.array(
                [ f[~indenter_selection].arrays[key].sum(axis=0)
                     for f in tmp_traj ] )
        else:
            logger.warn("Warning: key '{:s}' not in NetCDF".format(key))

    # only keep z forces and create data frames
    indenter_force_z_sum_dict = {
        key: value[:,dimension_of_interest] for key, value
            in indenter_force_sum_dict.items() }

    nonindenter_force_z_sum_dict = {
        key: value[:,dimension_of_interest] for key, value
            in nonindenter_force_sum_dict.items() }

    indenter_force_z_sum_df = pd.DataFrame.from_dict(
        indenter_force_z_sum_dict, dtype=float)

    nonindenter_force_z_sum_df = pd.DataFrame.from_dict(
        nonindenter_force_z_sum_dict, dtype=float)

    # TODO: read time from netcdf
    # netcdf_output_interval = colvars_traj_df.index[-1]/(len(tmp_traj)-1)
    logger.info("netcdf stores every {:d}th frame.".format(
        int(netcdf_output_interval) ) )

    indenter_force_z_sum_df.set_index(
        (indenter_force_z_sum_df.index*netcdf_output_interval).astype(int),
        inplace=True )
    nonindenter_force_z_sum_df.set_index(
        (nonindenter_force_z_sum_df.index*netcdf_output_interval).astype(int),
        inplace=True )

    # store z forces in json files
    if type(output_formats) is str: output_formats = [ output_formats]
    if type(indenter_z_forces_file_name) is str:
        indenter_z_forces_file_name = {
            'json': indenter_z_forces_file_name + '.json',
            'txt':  indenter_z_forces_file_name + '.txt' }
        nonindenter_z_forces_file_name = {
            'json': nonindenter_z_forces_file_name + '.json',
            'txt': nonindenter_z_forces_file_name + '.txt' }

    if 'json' in output_formats and 'json' in indenter_z_forces_file_name:
        indenter_force_z_sum_df.to_json(
            indenter_z_forces_file_name["json"],  orient='index')
    if 'json' in output_formats and 'json' in nonindenter_z_forces_file_name:
        nonindenter_force_z_sum_df.to_json(
            nonindenter_z_forces_file_name,  orient='index')
    if 'txt' in output_formats and 'txt' in indenter_z_forces_file_name:
        indenter_force_z_sum_df.to_csv(
            indenter_z_forces_file_name, sep=' ', header=True,
            index=True, float_format='%g')
    if 'txt' in output_formats and 'txt' in nonindenter_z_forces_file_name:
        nonindenter_force_z_sum_df.to_csv(
            nonindenter_z_forces_file_name, sep=' ', header=True,
            index=True, float_format='%g')

    return indenter_force_z_sum_df, nonindenter_force_z_sum_df

def main():
  import argparse
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--element', '-e', metavar='EL',
    help="Element of solid. Default: 'Au'",
    default='Au')
  parser.add_argument('--dim', '-d', type=int, metavar='Z',
    help="Spatial dimension of interest ( 0 = x, 1 = y, 2 = z). Default: 2",
    default=2)
  parser.add_argument('--plane', '-p', type=float, metavar='Z_POS',
    help="Z oosition of indenter-substrtae separating plane. Default: 20",
    default=20)
  parser.add_argument('--output-formats', '-f', metavar='FORMAT',
    help="Output formats to write. Default: json txt", nargs='+',
    default = ['json', 'txt'] )
  parser.add_argument('--force-keys',
    help="Keys in NetCDF to process.", nargs='+', metavar='KEY',
    default = [
        'forces',
        'f_storeAnteSHAKEForces',
        'f_storeAnteStatForces',
        'f_storeUnconstrainedForces',
        'f_storeAnteSHAKEForcesAve',
        'f_storeAnteStatForcesAve',
        'f_storeUnconstrainedForcesAve' ] )
  parser.add_argument('--netcdf-output-interval', type=int, metavar='N',
    default = globals.std_netcdf_output_interval,
    help    = "Default: {:d}".format(globals.std_netcdf_output_interval))
  parser.add_argument('--verbose', '-v', action='store_true',
    help='Make this tool more verbose')
  parser.add_argument('--debug', action='store_true',
    help='Make this tool print debug info')
  parser.add_argument('netcdf', help='NetCDF trajectory', metavar='NETCDF')
  parser.add_argument('indenter_force_outfile', metavar='INDENTER_FORCES_OUT',
    help="""Basename of output file for forces on indenter (without extension)
    Default: 'indenter_z_forces'""",
    default='indenter_z_forces', nargs='?')
  parser.add_argument('nonindenter_force_outfile',
    metavar='NONINDENTER_FORCES_OUT',
    help="""Basename of output file for forces on non-indenter atoms
    (without extension). Default: 'indenter_z_forces'""",
    default='indenter_z_forces', nargs='?')

  args = parser.parse_args()

  if args.debug:
    loglevel = logging.DEBUG
  elif args.verbose:
    loglevel = logging.INFO
  else:
    loglevel = logging.WARNING

  logging.basicConfig(level = loglevel)
  logger = logging.getLogger('{}.main'.format(__file__))

  extractIndenterNonIndenterForcesFromNetCDF(
      force_keys = args.force_keys,
      indenter_z_forces_file_name    = args.indenter_force_outfile,
      nonindenter_z_forces_file_name = args.nonindenter_force_outfile,
      separating_xy_plane_z_pos      = args.plane,
      solid_element                  = args.element,
      netcdf                         = args.netcdf,
      dimension_of_interest          = args.dim,
      netcdf_output_interval         = args.netcdf_output_interval,
      output_formats                 = args.output_formats )

  return

if __name__ == '__main__':
    main()
