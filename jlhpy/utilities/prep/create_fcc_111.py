#!/usr/bin/env python
#
# bounding_sphere.py
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
"""Create FCC 111 block."""


def as_std_type(value):
    """Convert numpy type to standard type."""
    return getattr(value, "tolist", lambda: value)()


def create_fcc_111(
        approximate_measures=[150., 150., 150.],
        lattice_constant=4.075, element='Au'):
    """Create FCC crystal with 111 surface facing upwards."""
    import numpy as np
    from ase.build import fcc111

    print((
        "Create {element:} FCC 111 block with lattice constant "
        "a = {lattice_constant:} and approximate measures "
        "{approximate_measures:}").format(
            element=element,
            lattice_constant=lattice_constant,
            approximate_measures=approximate_measures))
    # create a reference cell with minimal measures
    unit_cell = fcc111(
        element, size=(1,2,3), a=lattice_constant, periodic=True, orthogonal=True)
    print(
        "Reference FCC 111 (1,2,3) block has measures {measures:}".format(
            measures=np.diagonal(unit_cell.cell)))
    multiples_123 = np.array(approximate_measures)/np.diagonal(unit_cell.cell)
    print("Desired measures correspond to FCC 111 (1,2,3) block {} multiples.".format(multiples_123))
    multiples_111 = np.round(multiples_123).astype(int)*np.array([1,2,3])
    substrate = fcc111(
        element, size=multiples_111, a=lattice_constant, periodic=True, orthogonal=True)
    exact_measures = np.diagonal(substrate.cell)
    print((
        "Created {element:} FCC 111 (1,1,1) {multiples:} block with exact measures "
        "{measures:}").format(
            element=element,
            multiples=multiples_111,
            measures=exact_measures))

    return substrate


def create_fcc_111_data_file(outfile='default.pdb', *args, **kwargs):
    """Create FCC crystal with 111 surface facing upwards and write fo file.

    Returns:
        list of float: exact measures
    """
    import ase.io
    import numpy as np
    substrate = create_fcc_111(*args, **kwargs)
    ase.io.write(outfile, substrate)
    return as_std_type(np.diagonal(substrate.cell))
