# -*- coding: utf-8 -*-
"""jinja2 template-related helpers for cylindrical packing."""


def generate_pack_cylinder_packmol_template_context(
        C, l,
        R_inner,
        R_outer,
        R_inner_constraint,  # shell inner radius
        R_outer_constraint,  # shell outer radius
        sfN,  # number  of surfactant molecules
        inner_atom_number,  # inner atom
        outer_atom_number,  # outer atom
        surfactant='SDS',
        counterion='NA',
        tolerance=2,
        ioncylinder_outside=True,
        ioncylinder_within=True,
        hemi=None):
    """Creates context for filling Jinja2 PACKMOL input template in order to
    generate preassembled surfactant cylinders or hemicylinders with
    couinterions at polar heads"""
    import logging

    logger = logging.getLogger(__name__)
    logger.info(
        "cylinder with {:d} surfactant molecules in total.".format(sfN))

    cylinder = {}
    ioncylinder = {}

    cylinder["surfactant"] = surfactant

    cylinder["inner_atom_number"] = inner_atom_number
    cylinder["outer_atom_number"] = outer_atom_number

    cylinder["N"] = sfN

    cylinder["base_center"] = C
    cylinder["length"] = l

    cylinder["r_inner"] = R_inner
    cylinder["r_inner_constraint"] = R_inner_constraint
    cylinder["r_outer_constraint"] = R_outer_constraint
    cylinder["r_outer"] = R_outer

    logging.info(
        "cylinder with {:d} molecules at {}, length {}, radius {}".format(
            cylinder["N"], cylinder["base_center"], cylinder["length"], cylinder["r_outer"]))

    # ions at outer surface
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

    # experience shows: movebadrandom advantegous for (hemi-) cylinders
    context = {
        'cylinders':     [cylinder],
        'ioncylinders':  [ioncylinder],
        'movebadrandom': True,
    }
    if hemi == 'upper':
        context["upper_hemi"] = True
    elif hemi == 'lower':
        context["lower_hemi"] = True

    return context


def generate_pack_cylinders_packmol_template_context(largs=None, lkwargs=None, *args, **kwargs):
    if largs is not None:
        N = len(largs)
    elif lkwargs is not None:
        N = len(lkwargs)
    else:
        raise ValueError("Either largs or lwargs or both must be list.")

    if largs is None:
        largs = [[]*N]
    if lkwargs is None:
        lkwargs = [{}*N]

    assert len(largs) == len(lkwargs)

    for i, (cur_args, cur_kwargs) in enumerate(zip(largs, lkwargs)):
        missing_kwargs = set(kwargs) - set(cur_kwargs)
        cur_kwargs.update({k: kwargs[k] for k in missing_kwargs})

        current_context = generate_pack_cylinder_packmol_template_context(*cur_args, **cur_kwargs)
        if i == 0:
            context = current_context
        else:
            context["cylinders"].append(current_context["cylinders"][0])
            context["ioncylinders"].append(current_context["ioncylinders"][0])

    return context


# def pack_cylinders(self, sfN, sb_measures, surfactant, counterion,
#                    l_surfactant     = l_SDS,
#                    head_atom_number = head_atom_number_SDS,
#                    tail_atom_number = tail_atom_number_SDS,
#                    ncylinders       = 3,
#                    hemicylinders    = False ):
#     """Creates preassembled (hemi-) cylinders on substrate with couinterions at polar heads"""
#
#     hemistr = 'hemi-' if hemicylinders else ''
#     logging.info( "{:d} {}cylinders with {:d} surfactant molecules in total.".format(ncylinders, hemistr, sfN ) )
#
#     sbX, sbY, sbZ = sb_measures
#
#     # place box at coordinate zero in z-direction
#     sb_pos = - sb_measures / 2 * np.array( [1,1,0] )
#
#     sf_molecules_per_cylinder = sfN // ncylinders
#     excess_sf_molecules = sfN % ncylinders
#
#     cylinder_spacing = sbY / ncylinders
#
#     # cylinders parallelt to x-axis
#     cylinders = [{} for _ in range(ncylinders)]
#     ioncylinders = [{} for _ in range(ncylinders)]
#
#     # surfactant cylinders
#     #   inner constraint radius: 1*tolerance
#     #   outer constraint radius: 1*tolerance + l_surfactant
#     # ions between cylindric planes at
#     #   inner radius:            1*tolerance + l_surfactant
#     #   outer radius:            2*tolerance + l_surfactant
#     for n, cylinder in enumerate(cylinders):
#         cylinder["surfactant"] = surfactant
#
#         if hemicylinders:
#             cylinder["upper_hemi"] = True
#
#         cylinder["inner_atom_number"] = tail_atom_number
#         cylinder["outer_atom_number"] = head_atom_number
#
#         cylinder["N"] = sf_molecules_per_cylinder
#         if n < excess_sf_molecules:
#             cylinder["N"] += 1
#
#         # if packing hemicylinders, center just at substrate
#         cylinder["base_center"] = [
#             sb_pos[0],
#             sb_pos[1] + (0.5 + float(n))*cylinder_spacing,
#             sb_measures[2] ]
#         # if packing full cylinders, shift center by one radius in z dir
#         if not hemicylinders:
#             cylinder["base_center"][2] += l_surfactant + 2*tolerance
#
#         cylinder["length"] = sb_measures[0] - tolerance # to be on top of gold surfface
#         cylinder["radius"] = 0.5*cylinder_spacing
#
#         cylinder["inner_constraint_radius"] = tolerance
#
#         maximum_constraint_radius = (0.5*cylinder_spacing - tolerance)
#         cylinder["outer_constraint_radius"] = tolerance + l_surfactant \
#             if tolerance + l_surfactant < maximum_constraint_radius \
#             else maximum_constraint_radius
#
#         logging.info(
#             "Cylinder {:d} with {:d} molecules at {}, length {}, radius {}".format(
#             n, cylinder["N"], cylinder["base_center"],
#             cylinder["length"], cylinder["radius"]))
#
#         # ions at outer surface
#         ioncylinders[n]["ion"] = counterion
#
#         if hemicylinders:
#             ioncylinders[n]["upper_hemi"] = True
#
#         ioncylinders[n]["N"] = cylinder["N"]
#         ioncylinders[n]["base_center"] = cylinder["base_center"]
#         ioncylinders[n]["length"] = cylinder["length"]
#         ioncylinders[n]["inner_radius"] = cylinder["outer_constraint_radius"]
#         ioncylinders[n]["outer_radius"] = cylinder["outer_constraint_radius"] + tolerance
#
#
#     # experience shows: movebadrandom advantegous for (hemi-) cylinders
#     context = {
#         'sb_pos':        sb_pos,
#         'cylinders':     cylinders,
#         'ioncylinders':  ioncylinders,
#         'movebadrandom': True,
#     }
#     return context
