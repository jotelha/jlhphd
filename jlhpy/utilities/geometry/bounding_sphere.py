#!/usr/bin/env python
#
# cmd_tasks.py
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


class BoundingSphere():
    """Bounding sphere of coordinates set read from file."""

    def get_bounding_sphere_from_coordinates(self, coordinates=None):
        if coordinates:
            self.coordinates = coordinates
        assert self.coordinates, "No coordinates!"

        import miniball
        import numpy as np
        C, R_sq = miniball.get_bounding_ball(coordinates)
        R = np.sqrt(R_sq)
        self.C = C
        self.R = R
        return C, R

    def get_bounding_sphere_from_ase_atoms(self, atoms=None):
        if atoms:
            self.atoms = atoms

        self.coordinates = self.atoms.get_positions()
        return self.get_bounding_sphere_from_coordinates(self.coordinates)

    def get_bounding_sphere_from_pdb_file(
            self, infile=None, format='proteindatabank'):
        if infile:
            self.infile = infile

        assert self.infile, "No data file!"

        import ase.io
        self.atoms = ase.io.read(self.infile, format=format)
        return self.get_bounding_sphere_from_ase_atoms(self.atoms)

    def __init__(self, infile=None):
        self.C = None
        self.R = None
        self.infile = infile

        if infile:
            self.get_bounding_sphere_from_file(infile)

    def __call__(self, infile=None):
        if not infile:
            if self.C and self.R:
                return self.C, self.R
        else:
            self.infile = infile

        return self.get_bounding_sphere_from_file(self.infile)
