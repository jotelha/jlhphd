# -*- coding: utf-8 -*-
""" Extracts group - nongroup forces from NetCDF """

def extract_summed_forces_from_netcdf(
        force_keys=[
            'forces',
            'f_storeAnteShakeForces',
            'f_storeAnteStatForces',
            'f_storeAnteFreezeForces',
            'f_storeUnconstrainedForces',
            'f_storeForcesAve',
            'f_storeAnteShakeForcesAve',
            'f_storeAnteStatForcesAve',
            'f_storeAnteFreezeForcesAve',
            'f_storeUnconstrainedForcesAve'],
        forces_file_name={
            'json': 'group_z_forces.json',
            'txt': 'group_z_forces.txt'},
        netcdf='default.nc',
        dimension_of_interest=2,  # forces in z dir
        output_formats=['json', 'txt']):

    """Extracts sum of force compoents along particular spatial dimension."""
    import logging

    logger = logging.getLogger(__name__)

    import numpy as np
    import pandas as pd
    from ase.io import NetCDFTrajectory

    logger.info("Opening NetCDF '{:s}'.".format(netcdf))
    tmp_traj = NetCDFTrajectory(netcdf, 'r',
                                keep_open=True)
    logger.info("Opened NetCDF '{:s}' with {:d} frames.".format(
        netcdf, len(tmp_traj)))

    logger.info("Reading NetCDF frame by frame.")
    force_sum_dict = {key: [] for key in force_keys}
    for key in force_keys:
        if key in tmp_traj[0].arrays:
            force_sum_dict[key] = np.array(
                [f.arrays[key].sum(axis=0)
                 for f in tmp_traj])
        else:
            logger.warning("Warning: key '{:s}' not in NetCDF".format(key))

    # only keep z forces and create data frames
    force_z_sum_dict = {
        key: value[:, dimension_of_interest] for key, value
        in force_sum_dict.items()}

    force_z_sum_df = pd.DataFrame.from_dict(
        force_z_sum_dict, dtype=float)

    # store z forces in json files
    if type(output_formats) is str: output_formats = [output_formats]
    if type(forces_file_name) is str:
        forces_file_name = {
            'json': forces_file_name + '.json',
            'txt': forces_file_name + '.txt'}

    if 'json' in output_formats and 'json' in forces_file_name:
        force_z_sum_df.to_json(
            forces_file_name["json"], orient='index')
    if 'txt' in output_formats and 'txt' in forces_file_name:
        force_z_sum_df.to_csv(
            forces_file_name["txt"], sep=' ', header=True,
            index=True, float_format='%g')

    return force_z_sum_df
