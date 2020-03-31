#!/usr/bin/env python
#
# bounding_spher.py
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
"""Bounding sphere of coordinates set read from file."""

# NOTE: for serializing and deserializing snippets with
# the utilities within wf.serialize and dill, formulate
# imports as below (otherwise, the builtin __import__
# will be missing when deserializing the functions)


def get_bounding_sphere_from_coordinates(coordinates):
    miniball = __import__('miniball')  # import miniball
    np = __import__('numpy')  # import numpy as np
    C, R_sq = miniball.get_bounding_ball(coordinates)
    R = np.sqrt(R_sq)
    return C, R


def get_bounding_sphere_from_ase_atoms(atoms):
    coordinates = atoms.get_positions()
    return get_bounding_sphere_from_coordinates(coordinates)


def get_bounding_sphere_from_file(
        infile, format='proteindatabank'):
    ase = __import__('ase.io')  # import ase.io
    atoms = ase.io.read(infile, format=format)
    return get_bounding_sphere_from_ase_atoms(atoms)
