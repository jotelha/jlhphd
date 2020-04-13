# -*- coding: utf-8 -*-
"""jinja2 template-related helpers for spherical packing."""

def generate_pack_sphere_packmol_template_context(
        C, R,
        R_inner,
        R_outer,
        R_inner_constraint, # shell inner radius
        R_outer_constraint, # shell outer radius
        sfN, # number  of surfactant molecules
        inner_atom_number, # inner atom
        outer_atom_number, # outer atom
        surfactant='SDS',
        counterion='NA',
        tolerance=2,
    ):
    """Creates context for filling Jinja2 PACKMOL input template in order to
    generate preassembled surfactant spheres with couinterions at polar heads"""
    import logging

    logger = logging.getLogger(__name__)
    logger.info(
        "sphere with {:d} surfactant molecules in total.".format(sfN))

    sphere = {}
    ionsphere = {}

    sphere["surfactant"] = surfactant

    sphere["inner_atom_number"] = inner_atom_number
    sphere["outer_atom_number"] = outer_atom_number

    sphere["N"] = sfN

    sphere["c"] = C

    sphere["r_inner"] = R_inner
    sphere["r_inner_constraint"] = R_inner_constraint
    sphere["r_outer_constraint"] = R_outer_constraint
    sphere["r_outer"] = R_outer

    logging.info(
        "sphere with {:d} molecules at {}, radius {}".format(
        sphere["N"], sphere["c"], sphere["r_outer"]))

    # ions at outer surface
    ionsphere["ion"] = counterion


    ionsphere["N"] = sphere["N"]
    ionsphere["c"] = sphere["c"]
    ionsphere["r_inner"] = sphere["r_outer"]
    ionsphere["r_outer"] = sphere["r_outer"] + tolerance


    # experience shows: movebadrandom advantegous for (hemi-) spheres
    context = {
        'spheres':     [sphere],
        'ionspheres':  [ionsphere],
        'movebadrandom': True,
    }
    return context
