#!/usr/bin/env python
#
# convert.py
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
"""Convert formats."""


def as_std_type(value):
    """Convert numpy type to standard type."""
    return getattr(value, "tolist", lambda: value)()


def convert_lammps_data_to_pdb(
        infile='default.lammps', outfile='default.pdb',
        lammps_style='full', lammps_units='real',
        lmp_ase_type_mapping={11: 'Au'},
        ase_pmd_type_mapping={'Au': 'AU'},
        ase_pmd_residue_mapping={'Au': 'AUM'}):
    """Convert LAMMPS data format to PDB.

        ASE types can be specified as str (element name) or int (atomic number).
    """
    import ase.io.lammpsdata
    import ase.data
    import parmed as pmd

    print("Use LAMMPS to ASE type mapping {}, LAMMPS units {} and LAMMPS atom style {} for reading {}.".format(
        lmp_ase_type_mapping, lammps_units, lammps_style, infile))

    infile_type_mapping = {
        k: ase.data.atomic_numbers[v] if isinstance(v, str) else v
        for k, v in lmp_ase_type_mapping.items()}

    print("LAMMPS to ASE type mapping converted to atomic numbers: {}.".format(infile_type_mapping))

    ase_data = ase.io.lammpsdata.read_lammps_data(
        infile, Z_of_type=infile_type_mapping,
        style=lammps_style, sort_by_id=True, units=lammps_units)

    print("Read {} from {}.".format(ase_data, infile))

    print("Use ASE to ParmEd type mapping {}, residue mapping {}.".format(
          ase_pmd_type_mapping, ase_pmd_residue_mapping))

    outfile_type_mapping = {
        ase.data.atomic_numbers[k] if isinstance(k, str) else k: v
        for k, v in ase_pmd_type_mapping.items()}

    outfile_residue_mapping = {
        ase.data.atomic_numbers[k] if isinstance(k, str) else k: v
        for k, v in ase_pmd_residue_mapping.items()}

    print("ASE to ParmEd type and residue mappings converted to atomic numbers: {}, {}.".format(
          outfile_type_mapping, outfile_residue_mapping))

    pmd_data = pmd.Structure()

    for atom_in in ase_data:
        atom_out = pmd.Atom(atomic_number=atom_in.number,
                            name=outfile_type_mapping[atom_in.number])
        # use atom indices as residue numbers:
        pmd_data.add_atom(atom_out,
                          outfile_residue_mapping[atom_in.number],
                          atom_in.index)

    pmd_data.coordinates = ase_data.get_positions()

    pmd_data.write_pdb(outfile)
    print("Wrote {} to {}.".format(pmd_data, outfile))
