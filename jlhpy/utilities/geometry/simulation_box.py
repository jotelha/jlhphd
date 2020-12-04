#!/usr/bin/env python
#
# bounding_box.py
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
"""get simulation box as defined in data files."""

# NOTE: never return numpy types, always convert to standard types

# https://stackoverflow.com/questions/9452775/converting-numpy-dtypes-to-native-python-types
def as_std_type(value):
    """Convert numpy type to standard type."""
    return getattr(value, "tolist", lambda: value)()


def get_gro_box(gro):
    """Return length, width, and height of simulaion box defined in .gro.

    Parameters
    ----------
        gro: str
            gro file name

    Returns
    -------
        [float, float, float]: length, width, and height in Angstrom
    """
    import MDAnalysis as mda

    u = mda.Universe(gro)
    dimensions = u.dimensions[:3]  # u.dimensions.shape is (6,), holds angles
    return as_std_type(dimensions)