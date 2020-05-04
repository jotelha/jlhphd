import pint


UNITS = pint.UnitRegistry()

# hard-coded system-sepcific

COUNTERIONS = {
    'NA': {
        'name': 'NA',
        'resname': 'NA',
        'reference_atom': { 'name: NA'},
    }
}

SOLVENTS = {
    'H2O': {
        'name': 'H2O',
        'resname': 'SOL',
        'reference_atom': { 'name': 'OW' },
    }
}

SUBSTRATES = {
    'AU': {
        'name': 'AU',
        'resname': 'AUM',
        'reference_atom': { 'name': 'AU' },
    }
}

SURFACTANTS = {
    'SDS': {
        # sds length, from head sulfur to tail carbon
        'length': 14.0138 * UNITS.angstrom,
        # atom  1:   S, in head group
        # atom 39: C12, in tail
        'head_atom_index': 1,   # 1-indexed, S in pdb
        'head_atom_name': 'S',
        'connector_atom_index': 2,   # 1-indexed, OS1 in pdb, connects head and tail
        'connector_atom_name': 'OS1',
        'tail_atom_index': 39,  # 1-indexed, C12 in pdb
        'tail_atom_name': 'C12',
    },
    'CTAB': {
        # ctab length, from head nitrogen to tail carbon
        # atom 17: N1, in head group
        'length': 19.934 * UNITS.angstrom,
        # atom  1: C1, in tail
        'head_atom_index': 17,
        'tail_atom_index': 1,
    }
}

TOLERANCE = 2  # Angstrom
