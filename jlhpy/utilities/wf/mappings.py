# -*- coding: utf-8 -*-
""" Mappings """
import numpy as np
import ase

# type dict, manually for SDS
sds_t2n = {1: 1, 2: 1, 3: 6, 4: 6, 5: 8, 6: 8, 7: 16, 8: 1, 9: 8, 10: 11, 12: 79}
sds_t2n_array = np.array([0,*list(sds_t2n.values())],dtype='uint64')
sds_t2e_array = np.array(ase.data.chemical_symbols)[sds_t2n_array] # double-check against LAMMPS data file

# type dict, manually for CTAB
ctab_t2n = {1: 1, 2: 1, 3: 1, 4: 6, 5: 6, 6: 6, 7: 7, 8: 1, 9: 8, 10: 35, 11: 79}
ctab_t2n_array = np.array([0,*list(ctab_t2n.values())],dtype='uint64')
ctab_t2e_array = np.array(ase.data.chemical_symbols)[ctab_t2n_array] # double-check against LAMMPS data file

lmp_type_ase_element_mapping = {
    '11': 'Au',
}

ase_type_pmd_type_mapping = {
    'Au': 'AU',
}

ase_type_pmd_residue_mapping = {
    'Au': 'AUM',
}

pdb_residue_charmm_residue_mapping = {
    'SOL': 'TIP3',
    'NA': 'SOD',
    'AUM': 'AUM',
    'SDS': 'SDS',
}


pdb_type_charmm_type_mapping = {
    'TIP3': {
        'OW': 'OH2',
        'HW1': 'H1',
        'HW2': 'H2',
    },
    'SOD': {
        'NA': 'SOD',
    },
    'AUM': {
        'AU': 'AU'
    },
    'SDS': {}
    # SDS names don't change
}

psfgen_mappings_template_context = {
    'residues': [
        {
            'in': res_in,
            'out': res_out,
            'atoms': [
                {
                    'in': atm_in,
                    'out': atm_out,
                } for atm_in, atm_out in pdb_type_charmm_type_mapping[res_out].items()
            ]
        } for res_in, res_out in pdb_residue_charmm_residue_mapping.items()
    ]
}

sds_lammps_type_atom_name_mapping = {
    1: 'HAL2',
    2: 'HAL3',
    3: 'CTL2',
    4: 'CTL3',
    5: 'OSL',
    6: 'O2L',
    7: 'SL',
    8: 'HT',
    9: 'OT',
    10: 'SOD',
    11: 'AU',
}
