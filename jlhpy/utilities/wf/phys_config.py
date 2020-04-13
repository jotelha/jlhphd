import pint


UNITS = pint.UnitRegistry()

# hard-coded system-sepcific

SURFACTANTS = {
    'SDS': {
        # sds length, from head sulfur to tail carbon
        'length': 14.0138 * UNITS.angstrom,
        # atom  1:   S, in head group
        # atom 39: C12, in tail
        'head_atom_index': 1,   # 1-indexed, S in pdb
        'tail_atom_index': 39,  # 1-indexed, C12 in pdb
    },
    'CTAB': {
        # ctab length, from head nitrogen to tail carbon
        'length': 19.934 * UNITS.angstrom,
        # atom 17: N1, in head group
        # atom  1: C1, in tail
       'head_atom_index': 17,
       'tail_atom_index': 1,
    }
}

TOLERANCE = 2 # Angstrom
