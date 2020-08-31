# -*- coding: utf-8 -*-
"""jinja2 template-related helpers for cylindrical packing."""
import logging


def generate_cylinders_packmol_template_context(
        cylinders,  # geometrical description of cylinders
        sfN,  # number of surfactant molecules
        inner_atom_number,  # inner atom
        outer_atom_number,  # outer atom
        surfactant='surfactant',
        counterion='ion',
        tolerance=2,
        ioncylinder_outside=True,
        ioncylinder_within=True,
        hemi=None):
    """Creates context for filling Jinja2 PACKMOL input template in order to
    generate preassembled surfactant cylinders or hemicylinders with
    couinterions at polar heads"""
    logger = logging.getLogger(__name__)

    N_aggregates = len(cylinders)
    sfN_per_aggregate = sfN // N_aggregates
    remaining_sfN = sfN % N_aggregates
    correction_every_nth = N_aggregates // remaining_sfN if remaining_sfN > 0 else 0

    ioncylinders = []
    for i, cylinder in enumerate(cylinders):
        cylinder["surfactant"] = surfactant
        cylinder["inner_atom_number"] = inner_atom_number
        cylinder["outer_atom_number"] = outer_atom_number

        cylinder["N"] = sfN_per_aggregate
        cylinder["N"] += 1 if correction_every_nth > 0 and i % correction_every_nth == 0 and i / correction_every_nth < remaining_sfN else 0

        if hemi == 'upper':
            cylinder["upper_hemi"] = True
        elif hemi == 'lower':
            cylinder["lower_hemi"] = True

        logger.info(
            "cylinder with {:d} molecules at {}, length {}, inner radius {}, outer radius {}.".format(
                cylinder["N"], cylinder["base_center"], cylinder["length"],
                cylinder["r_inner"], cylinder["r_outer"]))

        # ions at outer surface
        ioncylinder = {}
        ioncylinder["ion"] = counterion
        ioncylinder["N"] = cylinder["N"]
        ioncylinder["base_center"] = cylinder["base_center"]
        ioncylinder["length"] = cylinder["length"]

        if ioncylinder_outside and ioncylinder_within:
            ioncylinder["r_inner"] = cylinder["r_outer"] - tolerance
            ioncylinder["r_outer"] = cylinder["r_outer"]
        elif ioncylinder_outside:
            ioncylinder["r_inner"] = cylinder["r_outer"]
            ioncylinder["r_outer"] = cylinder["r_outer"] + tolerance
        elif ioncylinder_within:
            ioncylinder["r_inner"] = cylinder["r_inner"]
            ioncylinder["r_outer"] = cylinder["r_inner"] + tolerance
        else:
            ioncylinder["r_inner"] = cylinder["r_inner"] - tolerance
            ioncylinder["r_outer"] = cylinder["r_inner"]

        if hemi == 'upper':
            ioncylinder["upper_hemi"] = True
        elif hemi == 'lower':
            ioncylinder["lower_hemi"] = True

        logger.info(
            "ion cylinder with {:d} molecules at {}, length {}, inner radius {}, outer radius {}.".format(
                ioncylinder["N"], ioncylinder["base_center"],
                ioncylinder["length"], ioncylinder["r_inner"],
                ioncylinder["r_outer"]))

        ioncylinders.append(ioncylinder)

    # experience shows: movebadrandom advantegous for (hemi-) cylinders
    context = {
        'cylinders':     cylinders,
        'ioncylinders':  ioncylinders,
        'movebadrandom': True,
        'nloop':         200,
    }

    return context

def generate_upper_hemicylinders_packmol_template_context(*args, **kwargs):
    return generate_cylinders_packmol_template_context(*args, hemi='upper', **kwargs)
