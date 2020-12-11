#!/usr/bin/env python
#
# unwrap.py
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
"""Unwrap datafiles or trajectories."""


def unwrap_lammps_data(infile='in.lammps', outfile='out.lammps', atom_style='full'):
    """Unwrap LAMMPS data file (i.e. remove image flags and modify all coordinates accordingly).

    Tested with Ovito PyPI package v3.3.4"""
    from ovito.io import import_file, export_file
    from ovito.modifiers import UnwrapTrajectoriesModifier
    node = import_file(infile, atom_style='full')
    node.modifiers.append(UnwrapTrajectoriesModifier())
    export_file(node, outfile, 'lammps_data', atom_style=atom_style)
