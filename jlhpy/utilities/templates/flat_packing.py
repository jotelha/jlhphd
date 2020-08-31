# -*- coding: utf-8 -*-
"""jinja2 template-related helpers for layered packing."""
import logging


def generate_alternating_multilayer_packmol_template_context(
        layers,
        sfN,  # number  of surfactant molecules
        tail_atom_number,  # starts with this atom as lower atom
        head_atom_number,
        surfactant='surfactant',
        counterion='counterion',
        tolerance=2,
        ionlayer_above=True,
        ionlayer_within=True,
        accumulative_ionlayer=False,  # put all ions in one concluding layer
        ):
    """Creates context for filling Jinja2 PACKMOL input template in order to
    generate preassembled surfactant monolayers, bilayers and multilayers with
    couinterions at polar heads"""
    logger = logging.getLogger(__name__)

    N_aggregates = len(layers)
    sfN_per_aggregate = sfN // N_aggregates
    remaining_sfN = sfN % N_aggregates
    correction_every_nth = N_aggregates // remaining_sfN if remaining_sfN > 0 else 0

    ionlayers = []
    for i, layer in enumerate(layers):
        layer["surfactant"] = surfactant
        layer["lower_atom_number"] = tail_atom_number if i % 2 == 0 else head_atom_number
        layer["upper_atom_number"] = head_atom_number if i % 2 == 0 else tail_atom_number

        layer["N"] = sfN_per_aggregate
        layer["N"] += 1 if correction_every_nth > 0 and i % correction_every_nth == 0 and i / correction_every_nth < remaining_sfN else 0

        layer["bb_lower"] = layer["bounding_box"][0]
        layer["bb_upper"] = layer["bounding_box"][1]

        logger.info(
            "layer with {:d} molecules between lower corner {} and upper corner {}".format(
                layer["N"], layer["bb_lower"], layer["bb_upper"]))

        # ions at outer surface
        ionlayer = {}
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

        logger.info(
            "ion layer with {:d} molecules between lower corner {} and upper corner {}".format(
                ionlayer["N"], ionlayer["bb_lower"], ionlayer["bb_upper"]))

        ionlayers.append(ionlayer)

    if accumulative_ionlayer and ionlayer_above:
        # get rid of all except last ionlayer and put all ions in there
        ionlayers = ionlayers[-1]
        ionlayers[0]["N"] = sfN
    elif accumulative_ionlayer:
        # get rid of all except first ionlayer and put all ions in there
        ionlayers = ionlayers[0]
        ionlayers[0]["N"] = sfN

    context = {
        'layers':     layers,
        'ionlayers':  ionlayers,
    }
    return context


def generate_inverse_alternating_multilayer_packmol_template_context(
        layers, sfN, tail_atom_number, head_atom_number, **kwargs):
    return generate_alternating_multilayer_packmol_template_context(
        layers, sfN, tail_atom_number, head_atom_number, **kwargs)
