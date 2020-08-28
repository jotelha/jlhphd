# -*- coding: utf-8 -*-
"""jinja2 template-related helpers for layered packing."""


def generate_pack_layer_packmol_template_context(
        bounding_box,
        z_lower_constraint,
        z_upper_constraint,
        sfN,  # number  of surfactant molecules
        lower_atom_number,  # lower atom
        upper_atom_number,  # upper atom
        surfactant='SDS',
        counterion='NA',
        tolerance=2,
        ionlayer_above=True,
        ionlayer_within=True):
    """Creates context for filling Jinja2 PACKMOL input template in order to
    generate preassembled surfactant layer with couinterions at polar heads"""
    import logging

    logger = logging.getLogger(__name__)
    logger.info(
        "layer with {:d} surfactant molecules in total.".format(sfN))

    layer = {}
    ionlayer = {}

    layer["surfactant"] = surfactant
    layer["N"] = sfN

    layer["lower_atom_number"] = lower_atom_number
    layer["upper_atom_number"] = upper_atom_number

    layer["bb_lower"] = bounding_box[0]
    layer["bb_upper"] = bounding_box[1]
    layer["z_lower_constraint"] = z_lower_constraint
    layer["z_upper_constraint"] = z_upper_constraint

    logging.info(
        "layer with {:d} molecules between lower corner {} and upper corner {}".format(
            layer["N"], layer["bb_lower"], layer["bb_upper"]))

    # ions at outer surface
    ionlayer["ion"] = counterion
    ionlayer["N"] = layer["N"]

    if ionlayer_above and ionlayer_within:
        ionlayer["bb_lower"] = [*layer["bb_lower"][0:2], layer["bb_upper"][2] - tolerance]
        ionlayer["bb_upper"] = [*layer["bb_upper"][0:2], layer["bb_upper"][2]]
    elif ionlayer_above:
        ionlayer["bb_lower"] = [*layer["bb_lower"][0:2], layer["bb_upper"][2]]
        ionlayer["bb_upper"] = [*layer["bb_upper"][0:2], layer["bb_upper"][2] + tolerance]
    elif ionlayer_within:
        ionlayer["bb_lower"] = [*layer["bb_lower"][0:2], layer["bb_lower"][2]]
        ionlayer["bb_upper"] = [*layer["bb_upper"][0:2], layer["bb_lower"][2] + tolerance]
    else:
        ionlayer["bb_lower"] = [*layer["bb_lower"][0:2], layer["bb_lower"][2] - tolerance]
        ionlayer["bb_upper"] = [*layer["bb_upper"][0:2], layer["bb_lower"][2]]

    logging.info(
        "ion layer with {:d} molecules between lower corner {} and upper corner {}".format(
            ionlayer["N"], ionlayer["bb_lower"], ionlayer["bb_upper"]))

    context = {
        'layers':     [layer],
        'ionlayers':  [ionlayer],
    }
    return context


def generate_pack_layers_packmol_template_context(largs=None, lkwargs=None, *args, **kwargs):
    """Loop over zipped lists of args (largs) and kwargs (lkwargs) and fill
    missing keyword arguments  with defaults from **kwargs"""

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

        current_context = generate_pack_layer_packmol_template_context(*cur_args, **cur_kwargs)
        if i == 0:
            context = current_context
        else:
            context["layers"].append(current_context["layers"][0])
            context["ionlayers"].append(current_context["ionlayers"][0])

    return context


def generate_pack_monolayer_packmol_template_context(
        bounding_box,
        z_lower_constraint_offset,  # offset added to bounding_box[0,2]
        z_upper_constraint_offset,  # offset subtracted from bounding_box[1,2]
        sfN,  # number  of surfactant molecules
        lower_atom_number,  # lower atom
        upper_atom_number,  # upper atom
        surfactant='SDS',
        counterion='NA',
        tolerance=2,
        ionlayer_above=True,
        ionlayer_within=True,):
    return generate_pack_layer_packmol_template_context(
        bounding_box,
        bounding_box[0][2] + z_lower_constraint_offset,
        bounding_box[1][2] - z_upper_constraint_offset,
        sfN,  # number  of surfactant molecules
        lower_atom_number,  # lower atom
        upper_atom_number,  # upper atom
        surfactant,
        counterion,
        tolerance,
        ionlayer_above)


def generate_pack_alternating_multilayer_packmol_template_context(
        bounding_box,  # for one monolayer
        z_lower_constraint_offset,  # offset added to bounding_box[0,2]
        z_upper_constraint_offset,  # offset subtracted from bounding_box[1,2]
        sfN,  # number  of surfactant molecules
        tail_atom_number,  # starts with this atom as lower atom
        head_atom_number,
        surfactant='SDS',
        counterion='NA',
        tolerance=2,
        ionlayer_above=True,
        ionlayer_within=True,
        accumulative_ionlayer=False,  # put all ions in one concluding layer
        number_of_layers=2):
    current_bb = bounding_box
    bb_height = current_bb[1][2] - current_bb[0][2]

    sfN_per_layer = sfN // number_of_layers
    remaining_sfN = sfN % number_of_layers
    correction_every_nth = number_of_layers % remaining_sfN
    for l in range(number_of_layers):
        lower_atom_number = tail_atom_number if l % 2 == 0 else head_atom_number
        upper_atom_number = head_atom_number if l % 2 == 0 else tail_atom_number

        current_sfN = sfN_per_layer
        current_sfN += 1 if l % correction_every_nth == 0 else 0

        current_context = generate_pack_monolayer_packmol_template_context(
            current_bb,
            z_lower_constraint_offset,
            z_upper_constraint_offset,
            current_sfN,  # number  of surfactant molecules
            lower_atom_number,  # lower atom
            upper_atom_number,  # upper atom
            surfactant,
            counterion,
            tolerance,
            ionlayer_above,
            ionlayer_within)

        if l == 0:
            context = current_context
        else:
            context["layers"].append(current_context["layers"][0])
            context["ionlayers"].append(current_context["ionlayers"][0])

        current_bb[0][2] += bb_height
        current_bb[1][2] += bb_height

    if accumulative_ionlayer and ionlayer_above:
        # get rid of all except last ionlayer and put all ions in there
        context["ionlayers"] = [context["ionlayers"][-1]]
        context["ionlayers"][0]["N"] = sfN
    elif accumulative_ionlayer:
        # get rid of all except first ionlayer and put all ions in there
        context["ionlayers"] = [context["ionlayers"][0]]
        context["ionlayers"][0]["N"] = sfN

    return context


def generate_pack_hydrophilic_bilayer_packmol_template_context(
        bounding_box,  # for one monolayer
        z_lower_constraint_offset,  # offset added to bounding_box[0,2]
        z_upper_constraint_offset,  # offset subtracted from bounding_box[1,2]
        sfN,  # number  of surfactant molecules
        tail_atom_number,  # starts with this atom as lower atom
        head_atom_number,
        surfactant='SDS',
        counterion='NA',
        tolerance=2,
        ionlayer_above=True,
        ionlayer_within=True,
        accumulative_ionlayer=False):
    return generate_pack_alternating_multilayer_packmol_template_context(
            bounding_box,  # for one monolayer
            z_lower_constraint_offset,
            z_upper_constraint_offset,
            sfN,
            tail_atom_number,
            head_atom_number,
            surfactant,
            counterion',
            tolerance,
            ionlayer_above,
            ionlayer_within,
            accumulative_ionlayer,
            number_of_layers=2):


# def pack_monolayer(self, sfN, sb_measures, surfactant, counterion,
#                    l_surfactant     = l_SDS,
#                    head_atom_number = head_atom_number_SDS,
#                    tail_atom_number = tail_atom_number_SDS ):
#     """Creates preassembled monolayer"""
#
#     logging.info( "Monolayer with {:d} surfactant molecules.".format(sfN ) )
#
#     sbX, sbY, sbZ = sb_measures
#
#     # place box at coordinate zero in z-direction
#     sb_pos = - sb_measures / 2 * np.array( [1,1,0] )
#
#     # 1st monolayer above substrate, polar head towards surface
#     # NOT applying http://www.ime.unicamp.br/~martinez/packmol/userguide.shtml
#     # recommendation on periodic bc
#
#     na_layer_1_bb  = np.array([ [ - sbX / 2.,
#                                     sbX / 2. ],
#                                 [ - sbY / 2.,
#                                     sbY / 2. ],
#                                 [ sbZ,
#                                   sbZ + tolerance ] ])
#
#     monolayer_bb_1 = np.array([ [ - sbX / 2.,
#                                     sbX / 2. ],
#                                 [ - sbY / 2.,
#                                     sbY / 2. ],
#                                 [ sbZ,
#                                   sbZ + 2*tolerance + l_surfactant ] ] )
#
#     lower_constraint_plane_1 = sbZ + 1*tolerance
#     upper_constraint_plane_1 = sbZ + 1*tolerance + l_surfactant
#
#     monolayers = [
#         {
#             'surfactant':             surfactant,
#             'N':                      sfN,
#             'lower_atom_number':      tail_atom_number,
#             'upper_atom_number':      head_atom_number,
#             'bb_lower':               monolayer_bb_1[:,0],
#             'bb_upper':               monolayer_bb_1[:,1],
#             'lower_constraint_plane': lower_constraint_plane_1,
#             'upper_constraint_plane': upper_constraint_plane_1
#         } ]
#     ionlayers = [
#         {
#             'ion':                    counterion,
#             'N':                      sfN,
#             'bb_lower':               na_layer_1_bb[:,0],
#             'bb_upper':               na_layer_1_bb[:,1]
#         } ]
#
#     context = {
#         'sb_pos':     sb_pos,
#         'monolayers': monolayers,
#         'ionlayers':  ionlayers
#     }
#     return context
#
# def pack_bilayer(  self, sfN, sb_measures, surfactant, counterion,
#                    l_surfactant     = l_SDS,
#                    head_atom_number = head_atom_number_SDS,
#                    tail_atom_number = tail_atom_number_SDS ):
#     """Creates a single bilayer on substrate with couinterions at polar heads"""
#     sbX, sbY, sbZ = sb_measures
#
#     # place box at coordinate zero in z-direction
#     sb_pos = - sb_measures / 2 * np.array( [1,1,0] )
#
#     N_inner_monolayer = (sfN // 2) + (sfN % 2)
#     N_outer_monolayer = sfN//2
#
#     na_layer_1_bb  = np.array([ [ - sbX / 2. ,
#                                     sbX / 2. ],
#                                 [ - sbY / 2. ,
#                                     sbY / 2. ],
#                                 [ sbZ,
#                                   sbZ + tolerance ] ])
#
#     monolayer_bb_1 = np.array([ [ - sbX / 2. ,
#                                     sbX / 2. ],
#                                 [ - sbY / 2. ,
#                                     sbY / 2.  ],
#                                 [ sbZ,
#                                   sbZ + 2*tolerance + l_surfactant ] ])
#
#     lower_constraint_plane_1 = sbZ + 1*tolerance
#     upper_constraint_plane_1 = sbZ + 1*tolerance + l_surfactant
#     z_shift_monolayer_2 = 1*tolerance + l_surfactant # overlap
#     z_shift_na_layer_2 =  2*z_shift_monolayer_2 - 1*tolerance
#
#     monolayer_bb_2 = monolayer_bb_1 + np.array([[0,0],[0,0],
#                                                 [z_shift_monolayer_2,
#                                                  z_shift_monolayer_2]])
#     lower_constraint_plane_2 = lower_constraint_plane_1 + z_shift_monolayer_2
#     upper_constraint_plane_2 = upper_constraint_plane_1 + z_shift_monolayer_2
#
#     na_layer_2_bb = na_layer_1_bb + np.array([[0,0],[0,0],
#                                               [z_shift_na_layer_2,
#                                                z_shift_na_layer_2]])
#
#     monolayers = [
#         {
#             'surfactant':             surfactant,
#             'N':                      N_inner_monolayer,
#             'lower_atom_number':      head_atom_number,
#             'upper_atom_number':      tail_atom_number,
#             'bb_lower':               monolayer_bb_1[:,0],
#             'bb_upper':               monolayer_bb_1[:,1],
#             'lower_constraint_plane': lower_constraint_plane_1,
#             'upper_constraint_plane': upper_constraint_plane_1
#         },
#         {
#             'surfactant':             surfactant,
#             'N':                      N_outer_monolayer,
#             'lower_atom_number':      tail_atom_number,
#             'upper_atom_number':      head_atom_number,
#             'bb_lower':               monolayer_bb_2[:,0],
#             'bb_upper':               monolayer_bb_2[:,1],
#             'lower_constraint_plane': lower_constraint_plane_2,
#             'upper_constraint_plane': upper_constraint_plane_2
#         } ]
#     ionlayers = [
#         {
#             'ion':                    counterion,
#             'N':                      N_inner_monolayer,
#             'bb_lower':               na_layer_1_bb[:,0],
#             'bb_upper':               na_layer_1_bb[:,1]
#         },
#         {
#             'ion':                    counterion,
#             'N':                      N_outer_monolayer,
#             'bb_lower':               na_layer_2_bb[:,0],
#             'bb_upper':               na_layer_2_bb[:,1]
#         } ]
#
#     context = {
#         'sb_pos':     sb_pos,
#         'monolayers': monolayers,
#         'ionlayers':  ionlayers
#     }
#     return context
#
