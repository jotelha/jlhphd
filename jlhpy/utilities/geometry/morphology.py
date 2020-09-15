#!/usr/bin/env python
#
# morphology.py
#
# Copyright (C) 2020 IMTEK Simulation
# Author: Johannes Hoermann, johannes.hoermann@imtek.uni-freiburg.de
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Generate morphology descriptions."""

import logging


def layer(
        layer_bounding_box,
        surfactant_head_group_diameter,
        tolerance=2.0):
    logger = logging.getLogger(__name__)
    tol = tolerance
    d = surfactant_head_group_diameter
    bb = layer_bounding_box

    layer = {
        'bounding_box': bb,
        'lower_constraint_plane': bb[0][2] + d/2. + tol,
        'upper_constraint_plane': bb[1][2] - d/2. - tol,
    }
    logger.info("layer geometrical description: {}".format(layer))
    return layer


def cylinder(
        surfactant_bounding_sphere_radius,
        surfactant_head_group_diameter,
        tolerance=2.0):
    logger = logging.getLogger(__name__)

    r = surfactant_bounding_sphere_radius
    tol = tolerance
    d = surfactant_head_group_diameter

    R_inner = tol
    R_inner_constraint = R_inner + d + 2*tol
    R_outer = R_inner + 2.*r + 4.*tol
    R_outer_constraint = R_outer - d - 2*tol

    cylinder = {
        'r_inner': R_inner,
        'r_inner_constraint': R_inner_constraint,
        'r_outer_constraint': R_outer_constraint,
        'r_outer': R_outer
    }
    logger.info("cylinder geometrical description: {}".format(cylinder))
    return cylinder


def multilayer_above_substrate(
        substrate_bounding_box,
        surfactant_bounding_sphere_radius,
        surfactant_head_group_diameter,
        tolerance=2.0,
        N=1):
    logger = logging.getLogger(__name__)
    r = surfactant_bounding_sphere_radius
    tol = tolerance
    d = surfactant_head_group_diameter

    z0 = substrate_bounding_box[1][2] + tol
    height = 2*r + tol

    # stack bounding boxes
    bbs = [[
        [*substrate_bounding_box[0][0:2], z0 + height*i],
        [*substrate_bounding_box[1][0:2], z0 + height*(i+1)]
    ] for i in range(N)]

    layers = [layer(bb, d, tol) for bb in bbs]

    geometry = {
        'layers': layers
    }
    logger.info("layers geometrical description: {}".format(geometry))
    return geometry


def monolayer_above_substrate(*args, **kwargs):
    return multilayer_above_substrate(*args, N=1, **kwargs)


def bilayer_above_substrate(*args, **kwargs):
    return multilayer_above_substrate(*args, N=2, **kwargs)


def cylinders_above_substrate(
        substrate_bounding_box,
        surfactant_bounding_sphere_radius,
        surfactant_head_group_diameter,
        tolerance=2.0,
        vertical_offset=0):
    """Maximum multiple of x-axis-aligned cylinders along y dimension."""

    logger = logging.getLogger(__name__)

    r = surfactant_bounding_sphere_radius
    tol = tolerance
    d = surfactant_head_group_diameter
    bb = substrate_bounding_box

    length = bb[1][0] - bb[0][0]
    width = bb[1][1] - bb[0][1]

    cylinder_defaults = cylinder(r, d, tol)
    cylinder_defaults["length"] = length
    R = cylinder_defaults["r_outer"]

    # number of cylinders to fit
    N = int(width/(2.*R))
    # list of base_center points, shape (N, dim)
    base_center = [[bb[0][0], bb[0][1]+(i+0.5)/N*width, bb[1][2] + R + vertical_offset + tol] for i in range(N)]

    geometry = {
        'cylinders': [{'base_center': bc, **cylinder_defaults} for bc in base_center],
        'N': N,
    }

    logger.info("cylinders geometrical description: {}".format(geometry))
    return geometry


def hemicylinders_above_substrate(
        substrate_bounding_box,
        surfactant_bounding_sphere_radius,
        surfactant_head_group_diameter,
        tolerance=2.0):
    """Maximum multiple of x-axis-aligned hemicylinders along y dimension."""

    logger = logging.getLogger(__name__)

    r = surfactant_bounding_sphere_radius
    tol = tolerance
    d = surfactant_head_group_diameter
    bb = substrate_bounding_box

    length = bb[1][0] - bb[0][0]
    width = bb[1][1] - bb[0][1]

    hemicylinder_defaults = cylinder(r, d, tol)
    hemicylinder_defaults["length"] = length
    R = hemicylinder_defaults["r_outer"]

    # number of cylinders to fit
    N = int(width/(2.*R))
    # list of base_center points, shape (N, dim)
    base_center = [[bb[0][0], bb[0][1]+(i+0.5)/N*width, bb[1][2] + tol] for i in range(N)]

    geometry = {
        'cylinders': [{'base_center': bc, **hemicylinder_defaults} for bc in base_center],
        'N': N,
    }

    logger.info("hemicylinders geometrical description: {}".format(geometry))
    return geometry
