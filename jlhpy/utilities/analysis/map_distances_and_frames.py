# -*- coding: utf-8 -*-
"""Compute trajectory frame numbers from distances."""

import numpy as np

# NOTE: never return numpy types, always convert to standard types

# https://stackoverflow.com/questions/9452775/converting-numpy-dtypes-to-native-python-types
def as_std_type(value):
    """Convert numpy type to standard type."""
    return getattr(value, "tolist", lambda: value)()


def compute_frame_number_from_distance(desired_z_dist, initial_z_dist,
                                       approach_velocity, output_frequency,
                                       time_step=2):
    """Compute the frame number in a trajectory for a specific distance."""
    return (
        np.round(
            (np.array(desired_z_dist).astype(np.float)-np.array(initial_z_dist).astype(np.float))
            / (np.array(approach_velocity).astype(np.float)
               * np.array(time_step).astype(np.float)
               * np.array(output_frequency).astype(np.float))
        )
    ).astype(np.int)


def get_frame_range_from_distance_range(start_distance, stop_distance, step_distance,
                                        initial_z_dist, approach_velocity,
                                        output_frequency, time_step=2):
    """Return start_frame, stop_frame, step_frame from distances."""
    n_intervals = (np.float(stop_distance) - np.float(start_distance)) / np.float(step_distance)

    start_frame, stop_frame = compute_frame_number_from_distance(
                                    np.array([np.float(start_distance),np.float(stop_distance)], np.float),
                                    initial_z_dist, approach_velocity, output_frequency,
                                    time_step
                                )

    step_frame = np.int(np.round(np.float(stop_frame - start_frame) / n_intervals))
    return as_std_type(start_frame), as_std_type(stop_frame), as_std_type(step_frame)


def get_frame_range_from_full_distance_range(step_distance, initial_z_dist, approach_velocity,
                                             output_frequency, time_step=2):
    return get_frame_range_from_distance_range(initial_z_dist, 0.0, step_distance,
                                               initial_z_dist, approach_velocity,
                                               output_frequency, time_step)


def compute_distance_from_frame_number(frame_number, initial_z_dist,
                                       approach_velocity, output_frequency,
                                       time_step=2):
    """Compute the distance in a trajectory from a specific frame number."""
    return as_std_type((
        np.array(initial_z_dist).astype(np.float)
            + (np.array(approach_velocity).astype(np.float)
               * np.array(time_step).astype(np.float)
               * np.array(output_frequency).astype(np.float)
               * np.array(frame_number).astype(np.float))
        )
    )

